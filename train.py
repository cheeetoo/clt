import os
import argparse
from pathlib import Path

import torch
from torch import Tensor
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp.grad_scaler import GradScaler

from clt import FeatureParallelCLT
from dataset import CLTActivationDataset


def init_distributed() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return dist.get_rank()


def all_gather_batch(x: Tensor, world: int) -> Tensor:
    T, *rest = x.shape
    out = x.new_empty(world * T, *rest)
    dist.all_gather_into_tensor(out, x)
    return out


def preload_next(iter_loader, stream, world):
    pre_cpu, post_cpu = next(iter_loader)
    pre_gpu = pre_cpu.cuda(non_blocking=True)
    post_gpu = post_cpu.cuda(non_blocking=True)

    with torch.cuda.stream(stream):
        pre_super = all_gather_batch(pre_gpu, world)
        post_super = all_gather_batch(post_gpu, world)

    return pre_super, post_super


def save_single_device(model: FeatureParallelCLT, world, rank, out_path):
    full_sd = {}
    for k, v in model.state_dict().items():
        if v.ndim == 0:
            if rank == 0:
                full_sd[k] = v.cpu()
        else:
            gathered = [torch.empty_like(v) for _ in range(world)]
            dist.all_gather(gathered, v)
            if rank == 0:
                full_sd[k] = torch.cat(gathered, dim=-1).cpu()
    if rank == 0:
        torch.save(
            {
                "model_state": full_sd,
                "n_layers": model.n_layers,
                "d_model": model.d_model,
                "n_features": model.n_features,
                "bandwidth": model.bandwidth,
            },
            out_path,
        )


def main(args):
    rank = init_distributed()
    world = dist.get_world_size()
    device = torch.device("cuda", rank)

    dataset = CLTActivationDataset(args.bin_path, args.meta_path)
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, sampler=sampler, batch_size=args.bs, pin_memory=True)

    model = FeatureParallelCLT(
        dataset.shape[1],
        dataset.shape[2],
        args.features,
        args.bandwidth,
        args.threshold,
        args.lambda_p,
        args.c,
    ).to(device)

    scaler = GradScaler()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    stream_gather = torch.cuda.Stream()

    total_steps = args.epochs * len(loader)
    cpkt_future = None

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        iter_loader = iter(loader)

        pre_next, post_next = preload_next(iter_loader, stream_gather, world)
        for step in range(len(loader)):
            torch.cuda.current_stream().wait_stream(stream_gather)
            global_step = step + epoch * len(loader)
            lambda_s = args.lambda_s * global_step / total_steps

            pre, post = pre_next, post_next
            if step < len(loader) - 1:
                pre_next, post_next = preload_next(iter_loader, stream_gather, world)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h, acts, x_hat = model.forward(pre)
                loss = model.get_loss(post, h, x_hat, acts, lambda_s)

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

            if step % 10_000 == 0 and step != 0:
                if cpkt_future:
                    cpkt_future.result()
                cpkt_dir = Path("checkpoints") / f"ep{epoch:04d}s{step:05d}"
                cpkt_dir.mkdir(parents=True, exist_ok=True)
                state_dict = {
                    "epoch": torch.tensor(epoch).cuda(),
                    "model": model,
                    "optim": optim,
                    "scaler": scaler,
                }
                cpkt_future = dcp.async_save(
                    state_dict=state_dict, checkpoint_id=str(cpkt_dir)
                )

        if rank == 0:
            print(f"Epoch {epoch} done; loss {loss.item():.4f}")  # type: ignore

    if cpkt_future:
        cpkt_future.result()
    save_single_device(model, world, rank, args.out_path)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_path", type=str, default="acts.bin")
    parser.add_argument("--meta_path", type=str, default="acts_meta.pt")
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--features", type=int, default=300_000_000)
    parser.add_argument("--bandwidth", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--lambda_p", type=float, default=3e-6)
    parser.add_argument("--lambda_s", type=float, default=10.0)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--out_path", type=str, default="model.pt")
    args = parser.parse_args()
    main(args)

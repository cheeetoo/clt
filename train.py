import os
import argparse
from pathlib import Path

import wandb
import torch
from torch import Tensor
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.nn.functional import all_reduce
from tqdm import tqdm

from clt import FeatureParallelCLT
from data import StreamingActivationDataset


def init_distributed() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    dist.init_process_group("cpu:gloo,cuda:nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return dist.get_rank()


def _all_gather_batch(x: Tensor, world: int) -> Tensor:
    x = x.contiguous()
    T, *rest = x.shape
    out = x.new_empty(world * T, *rest)
    dist.all_gather_into_tensor(out, x)
    return out


def gather_next(iter_loader, stream, world) -> tuple[Tensor, Tensor]:
    pre, post = next(iter_loader)
    with torch.cuda.stream(stream):
        pre_super = _all_gather_batch(pre, world)
        post_super = _all_gather_batch(post, world)

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


def train_step(model, pre, post, lambda_s, optim):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        h, acts, x_hat = model.forward(pre)
        all_reduce(x_hat, op=dist.ReduceOp.SUM)

        loss = model.get_loss(post, h, x_hat, acts, lambda_s)
        loss.backward()
        optim.step()
        optim.zero_grad()

    return loss


def main(args):
    rank = init_distributed()
    world = dist.get_world_size()
    device = torch.device("cuda", rank)

    if rank == 0:
        wandb.init(
            project="clt-training",
            config={
                "model_name": args.model_name,
                "dataset_name": args.dataset_name,
                "dataset_conf": args.dataset_conf,
                "n_toks": args.n_toks,
                "batch_size": args.bs,
                "n_features": args.n_features,
                "bandwidth": args.bandwidth,
                "threshold": args.threshold,
                "lambda_p": args.lambda_p,
                "lambda_s": args.lambda_s,
                "c": args.c,
                "lr": args.lr,
                "epochs": args.epochs,
                "world_size": world,
            },
        )

    dataset = StreamingActivationDataset(
        args.model_name,
        args.dataset_name,
        args.dataset_conf,
        args.bs,
        args.n_toks,
        args.seq_len,
        device,
        torch.bfloat16,
    )

    model = FeatureParallelCLT(
        dataset.n_layers,
        dataset.d_model,
        args.n_features,
        args.bandwidth,
        args.threshold,
        args.lambda_p,
        args.c,
    ).to(device)
    model = torch.compile(model, mode="max-autotune", fullgraph=True)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, fused=True)

    stream_gather = torch.cuda.Stream()

    total_steps = args.epochs * len(dataset)
    cpkt_future = None

    if rank == 0:
        pbar = tqdm(total=total_steps, desc="Training", unit="step")

    for epoch in range(args.epochs):
        iter_loader = iter(dataset)

        pre_next, post_next = gather_next(iter_loader, stream_gather, world)
        for step in range(len(dataset)):
            torch.cuda.current_stream().wait_stream(stream_gather)
            global_step = step + epoch * len(dataset)
            lambda_s = args.lambda_s * global_step / total_steps

            pre, post = pre_next, post_next
            if step < len(dataset) - 1:
                pre_next, post_next = gather_next(iter_loader, stream_gather, world)

            loss = train_step(model, pre, post, lambda_s, optim)

            # Update tqdm and WandB logging
            if rank == 0:
                pbar.update(1)
                pbar.set_postfix({"epoch": epoch + 1, "step": step + 1})
                if step % 10 == 0:
                    wandb.log(
                        {
                            "loss": loss.detach().item(),
                            "lambda_s": lambda_s,
                            "epoch": epoch,
                            "step": step,
                            "global_step": global_step,
                            "lr": optim.param_groups[0]["lr"],
                        }
                    )

            if step % 1_000 == 0 and step != 0:
                if cpkt_future:
                    cpkt_future.result()
                cpkt_dir = Path("checkpoints") / f"ep{epoch:04d}s{step:05d}"
                cpkt_dir.mkdir(parents=True, exist_ok=True)
                state_dict = {
                    "model": model,
                    "optim": optim,
                }
                cpkt_future = dcp.async_save(
                    state_dict=state_dict, checkpoint_id=str(cpkt_dir)
                )

    if cpkt_future:
        cpkt_future.result()
    save_single_device(model, world, rank, args.out_path)

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        pbar.close()
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_conf", type=str, default="")
    parser.add_argument("--n_toks", type=int)
    parser.add_argument("--bs", type=int)
    parser.add_argument("--n_features", type=int)
    parser.add_argument("--bandwidth", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--lambda_p", type=float, default=3e-6)
    parser.add_argument("--lambda_s", type=float, default=10.0)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--out_path", type=str, default="model.pt")
    args = parser.parse_args()
    main(args)

import os
import argparse

import torch
import torch.distributed as dist
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-path", type=str, default="acts.bin")
    parser.add_argument("--meta-path", type=str, defualt="acts_meta.pt")
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--features", type=int, default=300_000_000)
    parser.add_argument("--bandwidth", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--lambda-p", type=float, default=3e-6)
    parser.add_argument("--lambda-s", type=float, default=10.0)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    rank = init_distributed()
    device = torch.device("cuda")

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

    total_steps = args.epochs * len(loader)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)

        for step, (pre, post) in enumerate(loader):
            pre = pre.cuda(non_blocking=True)
            post = post.cuda(non_blocking=True)
            global_step = step * epoch * len(loader)
            lambda_s = args.lambda_s * global_step / total_steps

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h, acts, x_hat = model.forward(pre)
                loss = model.get_loss(post, h, x_hat, acts, lambda_s)

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

        if rank == 0:
            print(f"Epoch {epoch} done; loss {loss.item():.4f}")  # type: ignore

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

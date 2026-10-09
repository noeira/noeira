#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | PyTorch DDP twin of examples/nn/distributed/gpt_ddp.mojo (timing mode)
# +--------------------------------------------------------------------------+ #
"""PyTorch DistributedDataParallel on the char GPT, for the M2 scaling table.

Same model and recipe as `gpt_ddp.mojo`'s timing mode: the `GPT` of
`torch_nn_reference.py` (6x384, seq 256, dropout 0.2, tied head, N(0, 0.02),
scaled residual projections), AdamW 1e-3 (0.9, 0.99) wd 0.1 on the matrices,
constant LR, global grad-norm clip 1.0, `--b-local` rows per rank (weak
scaling: the global batch grows with the rank count). Windows are drawn on the
device; each rank draws its own.

    # one node, N GPUs (run WITHOUT the project's LD_PRELOAD — see
    # `dump-lewm-ref` in pixi.toml: the interceptor stalls torch's CUDA build)
    env -u LD_PRELOAD .pixi/envs/act-ref/bin/torchrun --standalone \\
        --nproc_per_node 2 tools/nn/torch_ddp_gpt.py --b-local 64

    # Mac smoke test (gloo on the CPU, the dev config)
    pixi run -e act-ref torchrun --standalone --nproc_per_node 2 \\
        tools/nn/torch_ddp_gpt.py --small --b-local 4 --steps 3

`--mode compile` wraps the DDP module in `torch.compile` (DDPOptimizer splits
the graph at bucket boundaries so the allreduce still overlaps). `--no-sync`
runs every step under `model.no_sync()`, i.e. without the gradient allreduce:
the step time with minus without is the exposed communication, the same
measurement as `gpt_ddp.mojo -D NO_COMM`. Rank 0 prints a `RESULT {json}`.

TF32 is on by default, as in `torch_nn_reference.py` (our CUDA GEMMs pick
TF32 by shape).
"""

import argparse
import contextlib
import json
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch_nn_reference import CACHE, GPT  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--b-local", type=int, default=64, help="rows per rank")
    ap.add_argument("--steps", type=int, default=50, help="timed steps")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--mode", choices=["eager", "compile"], default="eager")
    ap.add_argument("--no-sync", action="store_true",
                    help="skip the gradient allreduce (timing only)")
    ap.add_argument("--small", action="store_true",
                    help="the 2x64, seq 64 dev config of gpt_ddp.mojo")
    ap.add_argument("--tf32", choices=["on", "off"], default="on")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    if cuda:
        torch.cuda.set_device(local)
        dev = torch.device("cuda", local)
    else:
        dev = torch.device("cpu")
    torch.backends.cuda.matmul.allow_tf32 = args.tf32 == "on"
    torch.backends.cudnn.allow_tf32 = args.tf32 == "on"

    seq, d, h, layers = (64, 64, 4, 2) if args.small else (256, 384, 6, 6)
    text = open(os.path.join(CACHE, "tinyshakespeare", "input.txt")).read()
    chars = sorted(set(text))
    ids = torch.tensor([chars.index(c) for c in text], dtype=torch.long)
    train = ids[: int(len(ids) * 0.9)].to(dev)

    torch.manual_seed(args.seed)  # same init on every rank (DDP also broadcasts)
    net = GPT(vocab=len(chars), seq=seq, d=d, h=h, layers=layers).to(dev)
    model = DDP(net, device_ids=[local] if cuda else None)
    if args.mode == "compile":
        model = torch.compile(model)
    decay = [p for n, p in net.named_parameters() if p.dim() >= 2 and n != "pos"]
    no_decay = [p for n, p in net.named_parameters() if p.dim() < 2 or n == "pos"]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=1e-3, betas=(0.9, 0.99), fused=cuda)
    gen = torch.Generator(device=dev).manual_seed(args.seed + rank)
    ar = torch.arange(seq, device=dev)
    b = args.b_local

    def step():
        s = torch.randint(0, train.numel() - seq - 1, (b,), device=dev, generator=gen)
        w = s[:, None] + ar[None, :]
        x, y = train[w], train[w + 1]
        sync_ctx = model.no_sync() if args.no_sync else contextlib.nullcontext()
        with sync_ctx:
            loss = F.cross_entropy(model(x).view(-1, len(chars)), y.view(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        return loss.detach()

    def sync():
        if cuda:
            torch.cuda.synchronize()
        dist.barrier()

    for _ in range(args.warmup):
        step()
    sync()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        loss = step()
    sync()
    ms = 1000.0 * (time.perf_counter() - t0) / max(args.steps, 1)
    lt = loss.clone()
    dist.all_reduce(lt)
    if rank == 0:
        tokens = world * b * seq
        out = dict(world=world, b_local=b, seq=seq, mode=args.mode,
                   no_sync=args.no_sync, tf32=args.tf32, torch=torch.__version__,
                   ms_per_step=ms, tokens_per_s=tokens / (ms / 1e3),
                   loss=lt.item() / world)
        print(f"[torch ddp] N={world} B_LOCAL={b} {args.mode}"
              f"{' no_sync' if args.no_sync else ''}: {ms:.3f} ms/step,"
              f" {out['tokens_per_s']:.0f} tokens/s, loss {out['loss']:.4f}")
        print("RESULT " + json.dumps(out))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

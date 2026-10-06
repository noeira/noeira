"""Times, alone, the ops that dominate the GPT train step's profile on the
5090, and the alternatives a rule could emit instead:

- ``embedding``: the embedding gradient, ``scatter_nd_add`` of the
  [batch, seq, dim] cotangent into the [vocab, dim] table, against a spread
  scatter and a one-hot GEMM, on uniform and on text-distributed indices;
- ``bmm``: attention's batched products (forward and backward shapes), at
  the rank the GPT emits (4) and flattened (3);
- ``head``: the tied head's weight gradient, ``transpose(matmul(transpose(a),
  g))`` added to the embedding gradient, against ``matmul(transpose(g), a)``.

    noeira_max/autodiff/run.sh noeira_max/autodiff/bench/kernel_probe.py
    noeira_max/autodiff/run.sh noeira_max/autodiff/bench/kernel_probe.py --only bmm
    noeira_max/autodiff/run.sh noeira_max/autodiff/bench/kernel_probe.py --device cpu --vocab 65

Prints one ``PROBE {json}`` line per graph: compile seconds and milliseconds
per call (pipelined over ``--calls``, then the median of synchronised calls).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

from noeira_max.autodiff.models import data


def onehot_t(idx, vocab: int, dtype, dev):
    """[vocab, n] one-hot of the flat indices ``idx`` [n]."""
    rows = ops.range(0, vocab, 1, out_dim=vocab, device=dev, dtype=DType.int64)
    hit = ops.equal(ops.unsqueeze(rows, 1), ops.unsqueeze(idx, 0))
    return ops.cast(hit, dtype)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--vocab", type=int, nargs="+", default=[65, 4096, 50304])
    ap.add_argument("--calls", type=int, default=50)
    ap.add_argument("--only", nargs="+", choices=["embedding", "bmm", "head"],
                    default=["embedding", "bmm", "head"])
    args = ap.parse_args()

    device = Accelerator() if args.device == "gpu" and accelerator_count() else CPU()
    dev = DeviceRef.from_device(device)
    session = InferenceSession(devices=[device])
    rng = np.random.default_rng(0)
    n, d, f32 = args.batch * args.seq, args.dim, DType.float32

    def run(name: str, types: list[TensorType], build, arrays: list[np.ndarray], **info):
        with Graph(name, input_types=types) as g:
            g.output(build(*g.inputs))
        start = time.perf_counter()
        model = session.load(g)
        compile_s = time.perf_counter() - start
        inputs = [Buffer.from_numpy(a).to(device) for a in arrays]
        for _ in range(5):
            model.execute(*inputs)
        device.synchronize()
        start = time.perf_counter()
        for _ in range(args.calls):
            model.execute(*inputs)
        device.synchronize()
        pipelined = 1000.0 * (time.perf_counter() - start) / args.calls
        synced = []
        for _ in range(min(20, args.calls)):
            t = time.perf_counter()
            model.execute(*inputs)
            device.synchronize()
            synced.append(1000.0 * (time.perf_counter() - t))
        out = dict(name=name, device=str(device), compile_s=round(compile_s, 2),
                   ms=round(pipelined, 4), median_synced_ms=round(statistics.median(synced), 4),
                   **info)
        print("PROBE " + json.dumps(out), flush=True)
        return model.execute(*inputs)[0].to_numpy()

    if "embedding" in args.only:
        # Embedding gradient: the rule's scatter, a scatter spread over one copy of
        # the table per batch row (then summed), and a one-hot GEMM. Indices are
        # uniform, or (``text``) the characters of real Shakespeare windows: as
        # skewed as the train step's (a space is ~15% of them).
        g3 = rng.standard_normal((args.batch, args.seq, d)).astype(np.float32)
        zeros = lambda shape: ops.broadcast_to(ops.constant(0, f32, dev), shape)  # noqa: E731
        _, text = data.shakespeare_vocab()
        starts = rng.integers(0, len(text) - args.seq, args.batch)
        windows = np.stack([text[s : s + args.seq] for s in starts]).astype(np.int64)
        cases = [(v, "uniform", rng.integers(0, v, (args.batch, args.seq), dtype=np.int64))
                 for v in args.vocab]
        cases.insert(1, (int(text.max()) + 1, "text", windows))
        for vocab, dist, idx in cases:
            types = [TensorType(f32, [args.batch, args.seq, d], dev),
                     TensorType(DType.int64, [args.batch, args.seq], dev)]
            expected = np.zeros((vocab, d), np.float64)
            np.add.at(expected, idx.reshape(-1), g3.reshape(-1, d).astype(np.float64))

            def spread(g, i):
                rows = ops.range(0, args.batch, 1, out_dim=args.batch, device=dev, dtype=DType.int64)
                rows = ops.broadcast_to(ops.unsqueeze(rows, 1), [args.batch, args.seq])
                pairs = ops.stack([rows, i], axis=-1)
                return ops.sum(ops.scatter_nd_add(zeros([args.batch, vocab, d]), g, pairs), axis=0)

            builds = [("scatter", lambda g, i: ops.scatter_nd_add(
                zeros([vocab, d]), g, ops.unsqueeze(i, -1)))]
            if args.batch * vocab * d <= 1 << 28:  # one table per row must fit (1 GiB)
                builds.append(("spread_scatter", lambda g, i: ops.squeeze(spread(g, i), 0)))
            if vocab * n <= 1 << 28:  # the one-hot matrix must fit
                builds.append(("onehot_gemm", lambda g, i: ops.matmul(
                    onehot_t(ops.reshape(i, [n]), vocab, f32, dev), ops.reshape(g, [n, d]))))
            for kind, build in builds:
                name = f"{kind}_v{vocab}_{dist}"
                got = run(name, types, build, [g3, idx], vocab=vocab, indices=dist)
                err = float(np.abs(got - expected).max())
                print(f"  {name} max |err| vs float64 np.add.at: {err:.3g}", flush=True)

    if "bmm" in args.only:
        # Attention's batched products, forward (q kᵀ, w v) and backward
        # (g vᵀ, wᵀ g), as the GPT emits them (rank 4) and flattened to rank 3:
        # does the rank change the kernel MAX picks?
        heads, hd = 6, d // 6
        b, t = args.batch, args.seq
        q = rng.standard_normal((b, heads, t, hd)).astype(np.float32)
        w = rng.standard_normal((b, heads, t, t)).astype(np.float32)
        flops = 2.0 * b * heads * t * t * hd
        for rank in (4, 3):
            lead = [b, heads] if rank == 4 else [b * heads]
            qt = TensorType(f32, [*lead, t, hd], dev)
            wt = TensorType(f32, [*lead, t, t], dev)
            qa, wa = q.reshape(*lead, t, hd), w.reshape(*lead, t, t)
            tr = lambda x: ops.transpose(x, -1, -2)  # noqa: E731
            for name, types, build, arrays in (
                ("bmm_qkT", [qt, qt], lambda x, y: ops.matmul(x, tr(y)), [qa, qa]),
                ("bmm_wv", [wt, qt], lambda x, y: ops.matmul(x, y), [wa, qa]),
                ("bmm_wT_g", [wt, qt], lambda x, y: ops.matmul(tr(x), y), [wa, qa]),
            ):
                run(f"{name}_rank{rank}", types, build, arrays, gflop=flops / 1e9)

    if "head" in args.only:
        # The tied head's weight gradient, accumulated into the embedding's.
        vocab = args.vocab[0]
        a2 = rng.standard_normal((n, d)).astype(np.float32)
        g2 = rng.standard_normal((n, vocab)).astype(np.float32)
        acc = rng.standard_normal((vocab, d)).astype(np.float32)
        types = [TensorType(f32, [n, d], dev), TensorType(f32, [n, vocab], dev),
                 TensorType(f32, [vocab, d], dev)]
        t = lambda x: ops.transpose(x, 0, 1)  # noqa: E731
        dw = g2.T.astype(np.float64) @ a2.astype(np.float64)
        for name, build, expected in (
            ("head_dw", lambda a, g, c: ops.matmul(t(a), g), dw.T),
            ("head_dw_t", lambda a, g, c: t(ops.matmul(t(a), g)), dw),
            ("head_dw_t_add", lambda a, g, c: t(ops.matmul(t(a), g)) + c, dw + acc),
            ("head_dw_direct_add", lambda a, g, c: ops.matmul(t(g), a) + c, dw + acc),
        ):
            got = run(name, types, build, [a2, g2, acc], vocab=vocab)
            err = float(np.abs(got - expected).max() / np.abs(expected).max())
            print(f"  {name} max relative err vs float64: {err:.3g}", flush=True)

if __name__ == "__main__":
    main()

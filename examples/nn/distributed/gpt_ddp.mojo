"""Char-GPT (TinyShakespeare) trained data-parallel over N ranks.

The step, per rank and enqueued rank by rank from one host thread:
device batch (this rank's rows of the global batch) -> zero_grad -> forward ->
SeqCE (device accumulator, no host read) -> SeqCE vjp -> net.vjp; then the
arena allreduce on every rank, the device grad clip, and AdamW.

Modes:
- default: weak-scaling timing. `B_LOCAL` rows per rank, `ITERS` steps after
  `WARMUP`; prints ms/step, tokens/s and the final train loss.
- `-D DDP_GATE`: the M1 equivalence gate on the GPT. Dropout off, constant LR;
  N ranks with B/N rows each vs 1 rank with B rows, same global batch at every
  step (`shard_batch`). Prints the max loss difference, the max parameter
  difference and the replica agreement (must be bit for bit).

Defines (`mojo build -D ...`, then run the binary):
  NGPUS=N         ranks (default 2)
  DDP_DEVICES     rank r on GPU r with MAX comm (default: every rank on one
                  context, the simulator; runs on a Mac or one GPU)
  GPT_FULL        nanoGPT 6x384, seq 256 (default: a 2x64, seq 64 dev config)
  B_LOCAL=b       rows per rank in timing mode (default 64 full / 8 dev)
  NN_ITERS=k      timed steps (default 50)

Mac:     pixi run -e apple mojo run -I . examples/nn/distributed/gpt_ddp.mojo
2 GPUs:  pixi run -e nvidia mojo build -I . -D DDP_DEVICES -D GPT_FULL \
             examples/nn/distributed/gpt_ddp.mojo -o /tmp/gpt_ddp && /tmp/gpt_ddp
"""

from std.random import seed, random_ui64
from std.sys import get_defined_int, is_defined
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from layout import Layout

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import child_refs
from noeira.nn.core.initializer import Normal
from noeira.nn.core.ptr import mptr
from noeira.nn.datasets import CharTokenizer, load_text, train_val_split
from noeira.nn.loss.sequence_cross_entropy import SequenceCrossEntropyLoss
from noeira.nn.models.gpt import GPTDropTied, gpt_scale_residual_proj, gpt_wire_tie
from noeira.nn.training.window_batch_kernels import advance_step_kernel
from noeira.nn.distributed.process_group import ProcessGroup, backend_name
from noeira.nn.distributed.data_parallel import DataParallel
from noeira.nn.distributed.shard_batch import window_onehot_shard_kernel


comptime FULL = is_defined["GPT_FULL"]()
comptime GATE = is_defined["DDP_GATE"]()
comptime USE_DEVICES = is_defined["DDP_DEVICES"]()
comptime NGPUS = get_defined_int["NGPUS", 2]()

comptime VOCAB = 65
comptime SEQ = 256 if FULL else 64
comptime EMBED = 384 if FULL else 64
comptime HEADS = 6 if FULL else 4
comptime LAYERS = 6 if FULL else 2
comptime FF_MULT = 4
comptime B_LOCAL = get_defined_int["B_LOCAL", 64 if FULL else 8]()
comptime ITERS = get_defined_int["NN_ITERS", 50]()
comptime WARMUP = 5
comptime GATE_ITERS = 30
comptime B_GATE = 32 if FULL else 16  # global batch in the gate

comptime LR: Scalar[DT] = 1e-3
comptime BETA2: Scalar[DT] = 0.99
comptime WD: Scalar[DT] = 0.1
comptime CLIP: Scalar[DT] = 1.0
# Dropout off in the gate (N ranks cannot draw one rank's masks); nanoGPT's 0.2
# for timing. NOTE: the dropout SEED is a type parameter, so every rank draws
# the same masks on different rows — see the results doc.
comptime DROPOUT_P: Float64 = 0.0 if GATE else 0.2
comptime SEED_BASE = UInt64(0xC0FFEE)
comptime USE_MAX_ATTN = True
comptime NET = GPTDropTied[
    VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P, SEED_BASE,
    USE_MAX_ATTN,
]
comptime ROW = SEQ * VOCAB
comptime LOSS = SequenceCrossEntropyLoss[SEQ, VOCAB]


struct _Rank(Movable):
    """One rank's step buffers, corpus copy and sampler state."""

    var in_t: Tensor
    var tgt_t: Tensor
    var logits: Tensor
    var grad: Tensor
    var gi: Tensor
    var loss: LOSS
    var corpus: TensorImpl[DType.int32]
    var rng: TensorImpl[DType.uint64]

    def __init__(
        out self, c: DeviceContext, bl: Int, ids: List[Int], seed_word: UInt64
    ) raises:
        var total = bl * ROW
        self.in_t = Tensor.alloc_gpu(c, total)
        self.tgt_t = Tensor.alloc_gpu(c, total)
        self.logits = Tensor.alloc_gpu(c, total)
        self.grad = Tensor.alloc_gpu(c, total)
        self.gi = Tensor.alloc_gpu(c, total)
        self.loss = LOSS.make_gpu(c)
        self.corpus = TensorImpl[DType.int32]()
        self.corpus.ensure(len(ids))
        for i in range(len(ids)):
            self.corpus.data[i] = Int32(ids[i])
        self.corpus.n = len(ids)
        self.corpus.upload(c)
        self.rng = TensorImpl[DType.uint64]()
        self.rng.ensure(2)
        self.rng.data[0] = seed_word  # the SAME seed on every rank
        self.rng.data[1] = 0
        self.rng.n = 2
        self.rng.upload(c)


struct _Result(Movable):
    var losses: List[Float64]
    var params: List[List[Scalar[DT]]]
    var ms_per_step: Float64
    var arena: Int

    def __init__(out self):
        self.losses = List[Float64]()
        self.params = List[List[Scalar[DT]]]()
        self.ms_per_step = 0.0
        self.arena = 0


def _make_pg[N: Int](ctx: DeviceContext, max_elems: Int) raises -> ProcessGroup[N]:
    comptime if USE_DEVICES and N >= 2:
        return ProcessGroup[N].devices(max_elems)
    else:
        return ProcessGroup[N].shared(ctx)


def _rank_step[N: Int, BL: Int](
    mut dp: DataParallel[NET, N], mut rk: _Rank, r: Int
) raises:
    """Batch -> forward -> SeqCE (accumulated on device) -> vjp, rank r."""
    comptime TOTAL = BL * ROW
    var c = dp.ctx(r)
    var co = Optional(c)
    with c.push_context():
        c.enqueue_function[window_onehot_shard_kernel[DT, BL, SEQ, VOCAB]](
            mptr(rk.corpus.dev.value().unsafe_ptr()),
            Int64(rk.corpus.n - SEQ),
            Int64(r * BL),
            rk.rng.lt["gpu", Layout.row_major(2)](),
            rk.in_t.lt["gpu", Layout.row_major(TOTAL)](),
            rk.tgt_t.lt["gpu", Layout.row_major(TOTAL)](),
            grid_dim=(TOTAL + TPB - 1) // TPB,
            block_dim=TPB,
        )
        c.enqueue_function[advance_step_kernel](
            rk.rng.lt["gpu", Layout.row_major(2)](), grid_dim=1, block_dim=1
        )
        dp.nets[r].forward["gpu", BL](
            child_refs[NET.ARITY, DT](rk.in_t), rk.logits, co
        )
        rk.loss.forward_accumulate["gpu", BL](rk.logits, rk.tgt_t, co)
        rk.loss.vjp["gpu", BL](rk.logits, rk.tgt_t, rk.grad, co)
        dp.nets[r].vjp["gpu", BL](
            child_refs[NET.ARITY, DT](rk.in_t),
            rk.grad,
            child_refs[NET.ARITY, DT](rk.gi),
            co,
        )


def _run[N: Int, BL: Int](
    ctx: DeviceContext, ids: List[Int], steps: Int, warmup: Int, keep_params: Bool
) raises -> _Result:
    seed(42)
    var seed_word = random_ui64(0, UInt64.MAX)
    var dp = DataParallel[NET, N].make[Normal[0.0, 0.02]](
        _make_pg[N](ctx, 16 << 20), lr=LR, beta2=BETA2, wd=WD
    )
    for r in range(N):
        var co = Optional(dp.ctx(r))
        gpt_scale_residual_proj[
            "gpu", VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P,
            SEED_BASE, USE_MAX_ATTN,
        ](dp.nets[r], co)
        gpt_wire_tie[
            "gpu", VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P,
            SEED_BASE, USE_MAX_ATTN,
        ](dp.nets[r])
    dp.synchronize()
    dp.sync_params()
    var ranks = List[_Rank](capacity=N)
    for r in range(N):
        ranks.append(_Rank(dp.ctx(r), BL, ids, seed_word))
    dp.synchronize()

    var res = _Result()
    res.arena = dp.total
    var t0 = perf_counter_ns()
    for it in range(warmup + steps):
        if it == warmup:
            dp.synchronize()
            t0 = perf_counter_ns()
        dp.zero_grad()
        for r in range(N):
            _rank_step[N, BL](dp, ranks[r], r)
        dp.allreduce_grads()
        dp.clip_grads_device(CLIP)
        dp.step()
        comptime if GATE:
            # Per-step loss: the global batch's mean CE = mean of the ranks'.
            var l = 0.0
            for r in range(N):
                l += Float64(ranks[r].loss.read_accum["gpu"](Optional(dp.ctx(r))))
                ranks[r].loss.reset_accum["gpu"]()
            res.losses.append(l / Float64(N))
    dp.synchronize()
    res.ms_per_step = Float64(perf_counter_ns() - t0) / 1e6 / Float64(max(steps, 1))
    comptime if not GATE:
        # Mean train CE over the whole run (device accumulators, read once).
        var l = 0.0
        for r in range(N):
            l += Float64(ranks[r].loss.read_accum["gpu"](Optional(dp.ctx(r))))
        res.losses.append(l / Float64(N))
    if keep_params:
        for r in range(N):
            res.params.append(dp.download_params(r))
    return res^


def _max_abs_diff(a: List[Scalar[DT]], b: List[Scalar[DT]]) -> Float64:
    var m = 0.0
    for i in range(len(a)):
        m = max(m, Float64(abs(a[i] - b[i])))
    return m


def main() raises:
    var text = load_text()
    var tok = CharTokenizer(text)
    if tok.vocab_size != VOCAB:
        raise Error("vocab mismatch: " + String(tok.vocab_size))
    var ids = tok.encode(text)
    var split = train_val_split(ids, 0.1)
    var ctx = DeviceContext()
    print(
        "GPT DDP: layers=" + String(LAYERS) + " embed=" + String(EMBED)
        + " seq=" + String(SEQ) + " N=" + String(NGPUS)
        + (" devices" if USE_DEVICES else " shared-context simulator")
    )

    comptime if GATE:
        comptime assert B_GATE % NGPUS == 0, "B_GATE must divide by NGPUS"
        print("[gate] N=1 x B=" + String(B_GATE) + " vs N=" + String(NGPUS)
              + " x B=" + String(B_GATE // NGPUS) + ", " + String(GATE_ITERS) + " steps")
        var one = _run[1, B_GATE](ctx, split.train, GATE_ITERS, 0, True)
        var many = _run[NGPUS, B_GATE // NGPUS](ctx, split.train, GATE_ITERS, 0, True)
        var dl = 0.0
        for i in range(GATE_ITERS):
            dl = max(dl, abs(one.losses[i] - many.losses[i]))
        var dp_ = _max_abs_diff(one.params[0], many.params[0])
        var drep = 0.0
        for r in range(1, NGPUS):
            drep = max(drep, _max_abs_diff(many.params[0], many.params[r]))
        print("  arena =", one.arena, " loss", one.losses[0], "->", one.losses[GATE_ITERS - 1])
        print("  max|loss N1 - N" + String(NGPUS) + "| =", dl)
        print("  max|param N1 - N" + String(NGPUS) + "| =", dp_)
        print("  replica agreement max|rank0 - rank r| =", drep)
        if drep != 0.0:
            raise Error("GATE FAIL: replicas drifted apart")
        if one.losses[GATE_ITERS - 1] > one.losses[0] - 0.3:
            raise Error("GATE FAIL: the reference run did not train")
        print("GPT DDP GATE: replicas bit-identical; N-vs-1 differences above")
    else:
        var res = _run[NGPUS, B_LOCAL](ctx, split.train, ITERS, WARMUP, False)
        var tokens = Float64(NGPUS * B_LOCAL * SEQ)
        print("  arena =", res.arena, " B_LOCAL =", B_LOCAL,
              " global batch =", NGPUS * B_LOCAL)
        print("  ms/step =", res.ms_per_step,
              " tokens/s =", tokens / (res.ms_per_step / 1e3),
              " mean train CE =", res.losses[0])

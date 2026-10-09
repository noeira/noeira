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
  ZERO1           timing mode with ZeRO-1 instead of DDP
  ZERO_GATE       DDP vs ZeRO-1 at N ranks (clip on, dropout off)
  B_LOCAL=b       rows per rank in timing mode (default 64 full / 8 dev)
  NN_ITERS=k      timed steps (default 50)
  GRAPH=g         0 eager (default); 1 one CUDA graph per GPU, collective
                  inside; 2 compute captured, collective eager between two
                  graphs (forced when the backend cannot capture it: no P2P).
                  Real capture is NVIDIA-only; elsewhere every mode is eager.
  NO_COMM         skip the gradient collective (timing only, wrong numerics):
                  step time with minus without = the exposed collective

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
from noeira.nn.distributed.gpt_marked import (
    GPTMarked,
    gpt_marked_scale_residual_proj,
    gpt_marked_wire_tie,
)
from noeira.nn.training.window_batch_kernels import advance_step_kernel
from noeira.nn.distributed.process_group import ProcessGroup, backend_name
from noeira.nn.distributed.data_parallel import DataParallel
from noeira.nn.distributed.zero import Zero1
from noeira.nn.distributed.shard_batch import window_onehot_shard_kernel
from noeira.nn.distributed.rank_graphs import RankGraphs


comptime FULL = is_defined["GPT_FULL"]()
comptime GATE = is_defined["DDP_GATE"]()
comptime ZERO_GATE = is_defined["ZERO_GATE"]()
comptime ZERO1 = is_defined["ZERO1"]()
comptime PER_STEP_LOSS = GATE or ZERO_GATE
comptime USE_DEVICES = is_defined["DDP_DEVICES"]()
comptime GRAPH_EAGER = 0
comptime GRAPH_WHOLE = 1
comptime GRAPH_SPLIT = 2
comptime GRAPH = get_defined_int["GRAPH", GRAPH_EAGER]()
comptime NO_COMM = is_defined["NO_COMM"]()
comptime NGPUS = get_defined_int["NGPUS", 2]()
comptime OVERLAP = is_defined["OVERLAP"]()
"""DDP reduces gradient buckets on a second stream per GPU while the backward
runs (`DataParallel.enable_overlap`), one mark per transformer block."""
comptime BUCKET_MB = get_defined_int["BUCKET_MB", 8]()

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
comptime CLIP: Scalar[DT] = 0.0 if is_defined["NO_CLIP"]() else 1.0  # 0 = no clip
# Dropout off in the gate (N ranks cannot draw one rank's masks); nanoGPT's 0.2
# for timing. NOTE: the dropout SEED is a type parameter, so every rank draws
# the same masks on different rows — see the results doc.
comptime DROPOUT_P: Float64 = 0.0 if (GATE or ZERO_GATE) else 0.2
comptime SEED_BASE = UInt64(0xC0FFEE)
comptime USE_MAX_ATTN = True
# The GPT with a gradient mark after each block, compiled in only with
# OVERLAP (`GPTMarked[..., ACTIVE=False]` trains bit-identically to
# `GPTDropTied`: tests/nn/distributed/test_gpt_marked.mojo).
comptime NET = GPTMarked[
    VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P, SEED_BASE,
    USE_MAX_ATTN, OVERLAP,
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
    var state_bytes: Int

    def __init__(out self):
        self.losses = List[Float64]()
        self.params = List[List[Scalar[DT]]]()
        self.ms_per_step = 0.0
        self.arena = 0
        self.state_bytes = 0


def _make_pg[N: Int, SIM: Bool = False](
    ctx: DeviceContext, max_elems: Int
) raises -> ProcessGroup[N]:
    """`SIM`: the shared-context simulator even under `DDP_DEVICES` (the
    gate's exact reference for the MAX comm path)."""
    comptime if USE_DEVICES and N >= 2 and not SIM:
        return ProcessGroup[N].devices(max_elems)
    else:
        return ProcessGroup[N].shared(ctx)


def _rank_step[BL: Int](
    mut net: NET, c: DeviceContext, mut rk: _Rank, r: Int
) raises:
    """Batch -> forward -> SeqCE (accumulated on device) -> vjp, rank r."""
    comptime TOTAL = BL * ROW
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
        net.forward["gpu", BL](
            child_refs[NET.ARITY, DT](rk.in_t), rk.logits, co
        )
        rk.loss.forward_accumulate["gpu", BL](rk.logits, rk.tgt_t, co)
        rk.loss.vjp["gpu", BL](rk.logits, rk.tgt_t, rk.grad, co)
        net.vjp["gpu", BL](
            child_refs[NET.ARITY, DT](rk.in_t),
            rk.grad,
            child_refs[NET.ARITY, DT](rk.gi),
            co,
        )


def _surgery(mut net: NET, c: DeviceContext) raises:
    """nanoGPT's scaled residual init + the tied head, on one replica."""
    gpt_marked_scale_residual_proj[
        "gpu", VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P,
        SEED_BASE, USE_MAX_ATTN, OVERLAP,
    ](net, Optional(c))
    gpt_marked_wire_tie[
        VOCAB, SEQ, EMBED, HEADS, LAYERS, FF_MULT, True, DROPOUT_P,
        SEED_BASE, USE_MAX_ATTN, OVERLAP,
    ](net)


trait _Job(Movable & Deinitable):
    """One data-parallel run: the wrapper (DDP or ZeRO-1) plus every rank's
    buffers, owned in ONE struct so a capturing closure mentions only it.
    The step is `compute` (every rank: batch, forward, loss, vjp), `comm` (the
    gradient collective) and `update` (clip, optimizer, ZeRO's all-gather)."""

    def compute(mut self) raises:
        ...

    def comm(mut self) raises:
        ...

    def update(mut self) raises:
        ...

    def synchronize(self) raises:
        ...

    def graph_ctxs(self) -> List[DeviceContext]:
        ...

    def describe(self) -> String:
        """Extra line for the report (the bucket plan), or empty."""
        ...

    def capturable_collectives(self) -> Bool:
        ...

    def collectives_in_update(self) -> Bool:
        """Whether `update` runs a collective (ZeRO-1's all-gather and norm
        allreduce), so even a split capture needs capturable collectives."""
        ...

    def losses(mut self, reset: Bool) raises -> Float64:
        ...

    def download(mut self, r: Int) raises -> List[Scalar[DT]]:
        ...

    def arena(self) -> Int:
        ...

    def state_bytes(self) -> Int:
        ...


def _losses[N: Int](
    mut ranks: List[_Rank], ctxs: List[DeviceContext], reset: Bool
) raises -> Float64:
    """Mean over ranks of each rank's device CE accumulator (a host read)."""
    var l = 0.0
    for r in range(N):
        l += Float64(ranks[r].loss.read_accum["gpu"](Optional(ctxs[r])))
        if reset:
            ranks[r].loss.reset_accum["gpu"]()
    return l / Float64(N)


def _make_ranks[N: Int, BL: Int](
    ctxs: List[DeviceContext], ids: List[Int], seed_word: UInt64
) raises -> List[_Rank]:
    var ranks = List[_Rank](capacity=N)
    for r in range(N):
        ranks.append(_Rank(ctxs[r], BL, ids, seed_word))
    return ranks^


struct _DdpJob[N: Int, BL: Int, SIM: Bool = False](_Job):
    var dp: DataParallel[NET, Self.N]
    var ranks: List[_Rank]

    def __init__(out self, ctx: DeviceContext, ids: List[Int]) raises:
        seed(42)
        var seed_word = random_ui64(0, UInt64.MAX)
        self.dp = DataParallel[NET, Self.N].make[Normal[0.0, 0.02]](
            _make_pg[Self.N, Self.SIM](ctx, 16 << 20), lr=LR, beta2=BETA2, wd=WD
        )
        for r in range(Self.N):
            _surgery(self.dp.nets[r], self.dp.ctx(r))
            comptime if GRAPH != GRAPH_EAGER:
                # Weight-derived caches (padded / cast copies) rebuild on every
                # step, so the rebuild kernel is in the graph, not skipped by a
                # host-side version check frozen at capture time.
                self.dp.nets[r].set_attr["capture_recast"](Scalar[DT](1.0))
        comptime if OVERLAP:
            # MB -> fp32 elements; 0 = one bucket per mark (no merging)
            self.dp.enable_overlap(max(1, BUCKET_MB << 18))
        self.dp.synchronize()
        self.dp.sync_params()
        self.ranks = _make_ranks[Self.N, Self.BL](self.dp.pg.ctxs, ids, seed_word)
        self.dp.synchronize()

    def compute(mut self) raises:
        self.dp.zero_grad()
        self.dp.begin_backward()
        for r in range(Self.N):
            _rank_step[Self.BL](self.dp.nets[r], self.dp.ctx(r), self.ranks[r], r)

    def comm(mut self) raises:
        # With overlap: every bucket on the comm streams, each waiting for
        # its mark (recorded during `compute`), then the join and the 1/N.
        self.dp.reduce_grads()

    def describe(self) -> String:
        comptime if OVERLAP:
            if self.dp.n_marks < 0:  # N = 1: nothing to reduce, no plan
                return String("")
            return self.dp.bucket_summary()
        else:
            return String("")

    def update(mut self) raises:
        self.dp.clip_grads_device(CLIP)
        self.dp.step()

    def synchronize(self) raises:
        self.dp.synchronize()

    def graph_ctxs(self) -> List[DeviceContext]:
        return self.dp.pg.graph_ctxs()

    def capturable_collectives(self) -> Bool:
        return self.dp.pg.capturable_collectives()

    def collectives_in_update(self) -> Bool:
        return False

    def losses(mut self, reset: Bool) raises -> Float64:
        return _losses[Self.N](self.ranks, self.dp.pg.ctxs, reset)

    def download(mut self, r: Int) raises -> List[Scalar[DT]]:
        return self.dp.download_params(r)

    def arena(self) -> Int:
        return self.dp.total

    def state_bytes(self) -> Int:
        return self.dp.state_bytes_per_rank()


struct _ZeroJob[N: Int, BL: Int](_Job):
    var z: Zero1[NET, Self.N]
    var ranks: List[_Rank]

    def __init__(out self, ctx: DeviceContext, ids: List[Int]) raises:
        seed(42)
        var seed_word = random_ui64(0, UInt64.MAX)
        self.z = Zero1[NET, Self.N].make[Normal[0.0, 0.02]](
            _make_pg[Self.N](ctx, 16 << 20), lr=LR, beta2=BETA2, wd=WD
        )
        for r in range(Self.N):
            _surgery(self.z.nets[r], self.z.ctx(r))
            comptime if GRAPH != GRAPH_EAGER:
                # Weight-derived caches (padded / cast copies) rebuild on every
                # step, so the rebuild kernel is in the graph, not skipped by a
                # host-side version check frozen at capture time.
                self.z.nets[r].set_attr["capture_recast"](Scalar[DT](1.0))
        self.z.synchronize()
        self.z.sync_params()
        self.ranks = _make_ranks[Self.N, Self.BL](self.z.pg.ctxs, ids, seed_word)
        self.z.synchronize()

    def compute(mut self) raises:
        self.z.zero_grad()
        for r in range(Self.N):
            _rank_step[Self.BL](self.z.nets[r], self.z.ctx(r), self.ranks[r], r)

    def comm(mut self) raises:
        self.z.reduce_scatter_grads()

    def update(mut self) raises:
        self.z.clip_grads_device(CLIP)
        self.z.step()

    def synchronize(self) raises:
        self.z.synchronize()

    def graph_ctxs(self) -> List[DeviceContext]:
        return self.z.pg.graph_ctxs()

    def describe(self) -> String:
        return String("")

    def capturable_collectives(self) -> Bool:
        return self.z.pg.capturable_collectives()

    def collectives_in_update(self) -> Bool:
        return Self.N > 1

    def losses(mut self, reset: Bool) raises -> Float64:
        return _losses[Self.N](self.ranks, self.z.pg.ctxs, reset)

    def download(mut self, r: Int) raises -> List[Scalar[DT]]:
        return self.z.download_params(r)

    def arena(self) -> Int:
        return self.z.total

    def state_bytes(self) -> Int:
        return self.z.state_bytes_per_rank(0)


def _graph_mode[J: _Job](job: J) raises -> Int:
    """The capture mode this backend allows for the requested one."""
    comptime if OVERLAP and GRAPH != GRAPH_EAGER:
        # The overlapped collective lives INSIDE the backward's timeline, so
        # it is captured with the step or the step runs eagerly.
        if job.capturable_collectives():
            return GRAPH_WHOLE
        print("  [graph] OVERLAP: this backend's collectives cannot be"
              " captured, so the step runs eager")
        return GRAPH_EAGER
    comptime if GRAPH == GRAPH_WHOLE:
        if job.capturable_collectives():
            return GRAPH_WHOLE
        if job.collectives_in_update():
            raise Error("GRAPH: this backend's collectives cannot be captured"
                        " and ZeRO-1 runs one inside its update; use GRAPH=0")
        print("  [graph] collectives not capturable on this backend:"
              " compute-only capture, collective eager (GRAPH=2)")
        return GRAPH_SPLIT
    elif GRAPH == GRAPH_SPLIT:
        if job.collectives_in_update() and not job.capturable_collectives():
            raise Error("GRAPH=2: ZeRO-1's update runs a collective this"
                        " backend cannot capture; use GRAPH=0")
        return GRAPH_SPLIT
    else:
        return GRAPH_EAGER


def _run[J: _Job, N: Int](
    var job: J, steps: Int, warmup: Int, keep_params: Bool
) raises -> _Result:
    """`steps` timed steps after `warmup`, eager or captured (`GRAPH`)."""
    var res = _Result()
    res.arena = job.arena()
    res.state_bytes = job.state_bytes()
    var mode = _graph_mode(job)
    var whole = RankGraphs(job.graph_ctxs())
    var pre = RankGraphs(job.graph_ctxs())
    var post = RankGraphs(job.graph_ctxs())

    def _step() capturing raises -> None:
        job.compute()
        comptime if not NO_COMM:
            job.comm()
        job.update()

    def _compute() capturing raises -> None:
        job.compute()

    def _update() capturing raises -> None:
        job.update()

    var t0 = perf_counter_ns()
    for it in range(warmup + steps):
        if it == warmup:
            job.synchronize()
            t0 = perf_counter_ns()
        if mode == GRAPH_WHOLE:
            whole.run[_step]()
        elif mode == GRAPH_SPLIT:
            pre.run[_compute]()
            comptime if not NO_COMM:
                job.comm()
            post.run[_update]()
        else:
            _step()
        comptime if PER_STEP_LOSS:
            res.losses.append(job.losses(True))
    job.synchronize()
    res.ms_per_step = Float64(perf_counter_ns() - t0) / 1e6 / Float64(max(steps, 1))
    var d = job.describe()
    if d.byte_length() > 0:
        print("  [overlap]", d)
    comptime if not PER_STEP_LOSS:
        res.losses.append(job.losses(False))
    if keep_params:
        for r in range(N):
            res.params.append(job.download(r))
    return res^


def _run_ddp[N: Int, BL: Int, SIM: Bool = False](
    ctx: DeviceContext, ids: List[Int], steps: Int, warmup: Int, keep: Bool
) raises -> _Result:
    return _run[_DdpJob[N, BL, SIM], N](
        _DdpJob[N, BL, SIM](ctx, ids), steps, warmup, keep
    )


def _run_zero[N: Int, BL: Int](
    ctx: DeviceContext, ids: List[Int], steps: Int, warmup: Int, keep: Bool
) raises -> _Result:
    return _run[_ZeroJob[N, BL], N](_ZeroJob[N, BL](ctx, ids), steps, warmup, keep)


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
        + " GRAPH=" + String(GRAPH) + (" NO_COMM" if NO_COMM else "")
        + (" OVERLAP buckets<=" + String(BUCKET_MB) + "MB" if OVERLAP else "")
    )
    comptime assert not (OVERLAP and (ZERO1 or ZERO_GATE)), (
        "OVERLAP buckets DDP's allreduce only, not ZeRO-1's reduce-scatter"
    )

    comptime if GATE:
        comptime assert B_GATE % NGPUS == 0, "B_GATE must divide by NGPUS"
        print("[gate] N=1 x B=" + String(B_GATE) + " vs N=" + String(NGPUS)
              + " x B=" + String(B_GATE // NGPUS) + ", " + String(GATE_ITERS) + " steps")
        var one = _run_ddp[1, B_GATE](ctx, split.train, GATE_ITERS, 0, True)
        var many = _run_ddp[NGPUS, B_GATE // NGPUS](ctx, split.train, GATE_ITERS, 0, True)
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
        print("  per-step loss N1 / N" + String(NGPUS) + ":")
        for i in range(GATE_ITERS):
            if i < 3 or i % 5 == 4:
                print("    step", i, one.losses[i], many.losses[i],
                      " diff", one.losses[i] - many.losses[i])
        if drep != 0.0:
            raise Error("GATE FAIL: replicas drifted apart")
        comptime if USE_DEVICES:
            # Same N, same per-rank shapes and kernels, on ONE GPU: a sum of
            # two values is exact, so MAX comm must reproduce it bit for bit.
            # This isolates the comm path from the N-vs-1 shape differences.
            var sim = _run_ddp[NGPUS, B_GATE // NGPUS, True](
                ctx, split.train, GATE_ITERS, 0, True
            )
            var dsl = 0.0
            for i in range(GATE_ITERS):
                dsl = max(dsl, abs(sim.losses[i] - many.losses[i]))
            var dsp = _max_abs_diff(sim.params[0], many.params[0])
            print("  devices vs simulator (same N): max|loss| =", dsl,
                  " max|param| =", dsp)
            if NGPUS == 2 and (dsl != 0.0 or dsp != 0.0):
                raise Error("GATE FAIL: MAX comm differs from the exact 2-rank sum")
        if one.losses[GATE_ITERS - 1] > one.losses[0] - 0.3:
            raise Error("GATE FAIL: the reference run did not train")
        print("GPT DDP GATE: replicas bit-identical; N-vs-1 differences above")
    elif ZERO_GATE:
        comptime BL = B_GATE // NGPUS
        print("[zero gate] DDP vs ZeRO-1, N=" + String(NGPUS) + " x B=" + String(BL)
              + ", clip " + String(CLIP) + ", " + String(GATE_ITERS) + " steps")
        var d = _run_ddp[NGPUS, BL](ctx, split.train, GATE_ITERS, 0, True)
        var z = _run_zero[NGPUS, BL](ctx, split.train, GATE_ITERS, 0, True)
        var dl = 0.0
        for i in range(GATE_ITERS):
            dl = max(dl, abs(d.losses[i] - z.losses[i]))
        var dzp = 0.0
        var drep = 0.0
        for r in range(NGPUS):
            dzp = max(dzp, _max_abs_diff(d.params[r], z.params[r]))
            drep = max(drep, _max_abs_diff(z.params[0], z.params[r]))
        print("  arena =", d.arena, " loss", d.losses[0], "->", d.losses[GATE_ITERS - 1])
        print("  max|loss DDP - ZeRO1| =", dl, " max|param DDP - ZeRO1| =", dzp)
        print("  ZeRO-1 replica agreement =", drep)
        print("  state bytes/rank: DDP", d.state_bytes, " ZeRO-1", z.state_bytes)
        if drep != 0.0:
            raise Error("ZERO GATE FAIL: replicas drifted apart")
        print("GPT ZERO-1 GATE done")
    else:
        var res: _Result
        comptime if ZERO1:
            res = _run_zero[NGPUS, B_LOCAL](ctx, split.train, ITERS, WARMUP, False)
        else:
            res = _run_ddp[NGPUS, B_LOCAL](ctx, split.train, ITERS, WARMUP, False)
        var tokens = Float64(NGPUS * B_LOCAL * SEQ)
        print("  " + ("ZeRO-1" if ZERO1 else "DDP") + "  arena =", res.arena,
              " B_LOCAL =", B_LOCAL, " global batch =", NGPUS * B_LOCAL,
              " state MB/rank =", Float64(res.state_bytes) / 1e6)
        print("  ms/step =", res.ms_per_step,
              " tokens/s =", tokens / (res.ms_per_step / 1e3),
              " mean train CE =", res.losses[0])

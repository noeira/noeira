"""DistributedAutoregressiveTrainer — next-token LM training on N ranks through
any `ShardingStrategy`: DDP, ZeRO-1, ZeRO-2 or ZeRO-3.

`AutoregressiveTrainer`'s regime on N replicas: streaming random windows, a
cosine LR with warmup, the global grad-norm clip, per-token validation CE and
generation. The strategy owns the replicas, the optimizer state and the
collectives; this driver owns the data and each rank's work, and never names
a stage:

    var dist = S.make[INIT](pg^, lr=..., beta2=..., wd=...)
    # per-replica surgery (weight tying, scaled init) on dist.net(r), then:
    var tr = DistributedAutoregressiveTrainer[S, VOCAB, SEQ, B_RANK].make_from(
        dist^, tok^, split^, base_lr, warmup_iters, total_iters, grad_clip=1.0
    )
    _ = tr.fit(eval_every=250, n_val_windows=256)
    print(tr.generate("ROMEO:", 200, temperature=0.8))

One training step is four calls on the strategy:

    dist.set_lr(base_lr * cosine_warmup_lr_scale(it, ...))
    dist.forward_backward(ranks)      # each rank: batch, forward, SeqCE, vjp
    dist.clip_grads_device(grad_clip)
    dist.step()

`ranks[r]` (`_RankLM`) is rank r's work and buffers. It builds its rows of
the batch on its own device, runs its replica's forward and vjp, and keeps
its train and validation CE in device accumulators, so a step never waits
for the host.

Data. Rank r builds rows `r*B_RANK .. (r+1)*B_RANK - 1` of a global batch of
`N*B_RANK` rows drawn from one counter hash (`window_onehot_shard_kernel`).
The ranks' rows are disjoint, and together they are the batch one GPU with
`N*B_RANK` rows draws. The validation windows come from the same hash with
their own seed and a frozen step: every eval scores the same windows, split
over the ranks.

Dropout. Every replica's dropouts share the seed of their type, so rank r is
moved to its own Philox stream (`set_attr["rng_stream"](r)`). Rank 0 keeps
the default stream, so at N = 1 this driver trains exactly as
`AutoregressiveTrainer` (`tests/nn/distributed/test_dist_trainer_single.mojo`).

Eval and generation run their forward through `dist.forward_only`. ZeRO-3
holds no full copy of the weights anywhere, so its forward gathers each block
on every rank. The other strategies hold full replicas, so generation runs on
rank 0 alone.

GPU only, fp32, eager: no CUDA-graph capture (ZeRO-2/3 enqueue from fibers;
`examples/nn/distributed/gpt_ddp.mojo` measures the captured DDP step).
"""

from std.random import random_ui64
from std.memory import Pointer
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import child_refs
from noeira.nn.core.ptr import mptr
from noeira.nn.loss.sequence_cross_entropy import SequenceCrossEntropyLoss
from noeira.nn.datasets import CharTokenizer, DatasetSplit
from noeira.nn.training.window_batch_kernels import advance_step_kernel
from noeira.nn.training.lm_helpers import (
    cosine_warmup_lr_scale,
    upload_onehot,
    sample_next_token,
)

from .fibers import RankStep
from .strategy import ShardingStrategy
from .shard_batch import window_onehot_shard_kernel


comptime _TRAIN = 0
comptime _EVAL = 1
comptime _GENERATE = 2


def _resident_ids(
    c: DeviceContext, ids: List[Int]
) raises -> TensorImpl[DType.int32]:
    """A token corpus on the device, as int32."""
    var t = TensorImpl[DType.int32]()
    t.ensure(len(ids))
    for i in range(len(ids)):
        t.data[i] = Int32(ids[i])
    t.n = len(ids)
    t.upload(c)
    return t^


def _sampler(
    c: DeviceContext, seed_word: UInt64
) raises -> TensorImpl[DType.uint64]:
    """A window sampler's state on the device: [seed, step = 0]."""
    var t = TensorImpl[DType.uint64]()
    t.ensure(2)
    t.data[0] = seed_word
    t.data[1] = 0
    t.n = 2
    t.upload(c)
    return t^


def _window_batch[
    BL: Int, SEQ: Int, VOCAB: Int
](
    c: DeviceContext,
    mut corpus: TensorImpl[DType.int32],
    mut sampler: TensorImpl[DType.uint64],
    row_off: Int,
    mut inp: Tensor,
    mut tgt: Tensor,
) raises:
    """Enqueue the one-hots of global rows `row_off .. row_off + BL - 1` of
    the sampler's current batch."""
    comptime TOTAL = BL * SEQ * VOCAB
    c.enqueue_function[window_onehot_shard_kernel[DT, BL, SEQ, VOCAB]](
        mptr(corpus.dev.value().unsafe_ptr()),
        Int64(corpus.n - SEQ),
        Int64(row_off),
        sampler.tt["gpu", 2](),
        inp.tt["gpu", TOTAL](),
        tgt.tt["gpu", TOTAL](),
        grid_dim=(TOTAL + TPB - 1) // TPB,
        block_dim=TPB,
    )


struct _RankLM[NET: Module, VOCAB: Int, SEQ: Int, BL: Int](Movable, RankStep):
    """Rank r's work in every call through the strategy, and the buffers it
    keeps between steps. One per rank (fibers.mojo's contract)."""

    comptime ROW = Self.SEQ * Self.VOCAB
    comptime LOSS = SequenceCrossEntropyLoss[Self.SEQ, Self.VOCAB]

    var net: Int
    """This rank's replica, by address (replicas never move: strategy.mojo)."""
    var ctx: DeviceContext
    var nranks: Int
    var mode: Int
    """What `run_rank` does next: `_TRAIN`, `_EVAL` or `_GENERATE`."""
    var eval_batches: Int
    var in_t: Tensor
    var tgt_t: Tensor
    var logits: Tensor
    var grad: Tensor
    var gi: Tensor
    var train_ce: Self.LOSS
    var val_ce: Self.LOSS
    var corpus: TensorImpl[DType.int32]
    var val_corpus: TensorImpl[DType.int32]
    var sampler: TensorImpl[DType.uint64]
    """[seed, step] of the training windows; the step advances every batch."""
    var val_sampler: TensorImpl[DType.uint64]
    """[seed, 0] of the validation windows, never advanced."""
    var gen_in: Tensor
    var gen_out: Tensor

    def __init__(
        out self,
        net: Int,
        ctx: DeviceContext,
        nranks: Int,
        train_ids: List[Int],
        val_ids: List[Int],
        seed_word: UInt64,
        val_seed: UInt64,
    ) raises:
        comptime TOTAL = Self.BL * Self.ROW
        self.net = net
        self.ctx = ctx
        self.nranks = nranks
        self.mode = _TRAIN
        self.eval_batches = 0
        self.in_t = Tensor.alloc_gpu(ctx, TOTAL)
        self.tgt_t = Tensor.alloc_gpu(ctx, TOTAL)
        self.logits = Tensor.alloc_gpu(ctx, TOTAL)
        self.grad = Tensor.alloc_gpu(ctx, TOTAL)
        self.gi = Tensor.alloc_gpu(ctx, TOTAL)
        self.train_ce = Self.LOSS.make_gpu(ctx)
        self.val_ce = Self.LOSS.make_gpu(ctx)
        self.corpus = _resident_ids(ctx, train_ids)
        self.val_corpus = _resident_ids(ctx, val_ids)
        self.sampler = _sampler(ctx, seed_word)
        self.val_sampler = _sampler(ctx, val_seed)
        self.gen_in = Tensor.alloc_gpu(ctx, Self.ROW)
        self.gen_out = Tensor.alloc_gpu(ctx, Self.ROW)

    def run_rank(mut self, r: Int) raises:
        comptime A = Self.NET.ACT_DT
        comptime assert A == DT, (
            "DistributedAutoregressiveTrainer is fp32 (an fp32-flow model)"
        )
        var net = Pointer[Self.NET, MutUntrackedOrigin](
            unsafe_from_address=self.net
        )
        var co = Optional(self.ctx)
        # noeira's vendor-BLAS wrappers use the thread's current CUDA context.
        with self.ctx.push_context():
            if self.mode == _TRAIN:
                _window_batch[Self.BL, Self.SEQ, Self.VOCAB](
                    self.ctx, self.corpus, self.sampler, r * Self.BL,
                    self.in_t, self.tgt_t,
                )
                self.ctx.enqueue_function[advance_step_kernel](
                    self.sampler.tt["gpu", 2](),
                    grid_dim=1,
                    block_dim=1,
                )
                net[].forward["gpu", Self.BL](
                    child_refs[Self.NET.ARITY, A](self.in_t),
                    rebind[TensorImpl[A]](self.logits),
                    co,
                )
                self.train_ce.forward_accumulate["gpu", Self.BL](
                    self.logits, self.tgt_t, co
                )
                self.train_ce.vjp["gpu", Self.BL](
                    self.logits, self.tgt_t, self.grad, co
                )
                net[].vjp["gpu", Self.BL](
                    child_refs[Self.NET.ARITY, A](self.in_t),
                    rebind[TensorImpl[A]](self.grad),
                    child_refs[Self.NET.ARITY, A](self.gi),
                    co,
                )
            elif self.mode == _EVAL:
                for k in range(self.eval_batches):
                    _window_batch[Self.BL, Self.SEQ, Self.VOCAB](
                        self.ctx, self.val_corpus, self.val_sampler,
                        (k * self.nranks + r) * Self.BL,
                        self.in_t, self.tgt_t,
                    )
                    net[].forward["gpu", Self.BL](
                        child_refs[Self.NET.ARITY, A](self.in_t),
                        rebind[TensorImpl[A]](self.logits),
                        co,
                    )
                    self.val_ce.forward_accumulate["gpu", Self.BL](
                        self.logits, self.tgt_t, co
                    )
            else:
                net[].forward["gpu", 1](
                    child_refs[Self.NET.ARITY, A](self.gen_in),
                    rebind[TensorImpl[A]](self.gen_out),
                    co,
                )


struct DistributedAutoregressiveTrainer[
    S: ShardingStrategy, VOCAB: Int, SEQ: Int, B_RANK: Int
](Movable):
    comptime N = Self.S.RANKS
    comptime BATCH = Self.N * Self.B_RANK
    """The global batch: `B_RANK` rows on each of the N ranks."""
    comptime RankT = _RankLM[Self.S.Net, Self.VOCAB, Self.SEQ, Self.B_RANK]

    var dist: Self.S
    var ranks: List[Self.RankT]
    var tok: CharTokenizer
    var base_lr: Scalar[DT]
    var warmup_iters: Int
    var total_iters: Int
    var min_lr_scale: Float64
    var grad_clip: Scalar[DT]

    def __init__(
        out self,
        var dist: Self.S,
        var ranks: List[Self.RankT],
        var tok: CharTokenizer,
        base_lr: Scalar[DT],
        warmup_iters: Int,
        total_iters: Int,
        min_lr_scale: Float64,
        grad_clip: Scalar[DT],
    ):
        self.dist = dist^
        self.ranks = ranks^
        self.tok = tok^
        self.base_lr = base_lr
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.min_lr_scale = min_lr_scale
        self.grad_clip = grad_clip

    @staticmethod
    def make_from(
        var dist: Self.S,
        var tok: CharTokenizer,
        var split: DatasetSplit,
        base_lr: Scalar[DT],
        warmup_iters: Int,
        total_iters: Int,
        min_lr_scale: Float64 = 0.1,
        grad_clip: Scalar[DT] = 0.0,
    ) raises -> Self:
        """Synchronize the replicas (rank 0's weights to every rank: do any
        surgery on `dist.net(r)` before this call), put rank r on dropout
        stream r, and give each rank its copy of the corpus and its window
        samplers. The training windows' seed is drawn from the process RNG
        here, where `AutoregressiveTrainer.make_from` draws its own; the
        validation windows' seed right after it."""
        dist.sync_params()
        var seed_word = random_ui64(0, UInt64.MAX)
        var val_seed = random_ui64(0, UInt64.MAX)
        var ranks = List[Self.RankT](capacity=Self.N)
        for r in range(Self.N):
            var c = dist.ctx(r)
            ref net = dist.net(r)
            net.set_attr["rng_stream"](Scalar[DT](r))
            var addr = Int(Pointer(to=net))
            ranks.append(
                Self.RankT(
                    addr, c, Self.N, split.train, split.val, seed_word,
                    val_seed,
                )
            )
        dist.synchronize()
        return Self(
            dist^, ranks^, tok^, base_lr, warmup_iters, total_iters,
            min_lr_scale, grad_clip,
        )

    def lr_scale(self, it: Int) -> Scalar[DT]:
        """The schedule's factor on the base LR at iteration `it`."""
        return cosine_warmup_lr_scale(
            it, self.warmup_iters, self.total_iters, self.min_lr_scale
        )

    def _set_training(mut self, on: Bool):
        var v = Scalar[DT](1.0) if on else Scalar[DT](0.0)
        for r in range(Self.N):
            self.dist.net(r).set_attr["training"](v)

    # ----- training -------------------------------------------------------

    def train_step(mut self, it: Int) raises:
        """One optimizer step at iteration `it`: a fresh global batch, the
        mean gradient over ranks, the clip, the update."""
        self.dist.set_lr(self.base_lr * self.lr_scale(it))
        for r in range(Self.N):
            self.ranks[r].mode = _TRAIN
        self.dist.forward_backward(self.ranks)
        if self.grad_clip > Scalar[DT](0.0):
            self.dist.clip_grads_device(self.grad_clip)
        self.dist.step()

    def train_loss(mut self, reset: Bool = True) raises -> Float64:
        """Mean train CE over the steps since the last reset, averaged over
        the ranks (one host read per rank)."""
        var l = 0.0
        for r in range(Self.N):
            var c = self.ranks[r].ctx
            l += Float64(self.ranks[r].train_ce.read_accum["gpu"](Optional(c)))
            if reset:
                self.ranks[r].train_ce.reset_accum["gpu"]()
        return l / Float64(Self.N)

    def fit(
        mut self,
        eval_every: Int = 0,
        n_val_windows: Int = 0,
        print_progress: Bool = True,
    ) raises -> Float64:
        """`total_iters` steps from iteration 0. With `eval_every > 0` and
        `n_val_windows > 0`, reports the mean train CE since the last report
        and the validation CE every `eval_every` steps, and returns the last
        validation CE; otherwise returns the mean train CE of the run."""
        var do_eval = eval_every > 0 and n_val_windows > 0
        if do_eval and n_val_windows % Self.BATCH != 0:
            raise Error(
                "fit: n_val_windows must be a multiple of the global batch ("
                + String(Self.BATCH) + ")"
            )
        for r in range(Self.N):
            self.ranks[r].train_ce.reset_accum["gpu"]()
        var last = 0.0
        for it in range(self.total_iters):
            self.train_step(it)
            if do_eval and (
                (it + 1) % eval_every == 0 or (it + 1) == self.total_iters
            ):
                var tl = self.train_loss(reset=True)
                var v = self.val_loss(n_val_windows)
                last = v
                if print_progress:
                    print(
                        "  iter " + String(it + 1) + "/"
                        + String(self.total_iters)
                        + "  train=" + String(Float32(tl))
                        + "  val=" + String(Float32(v))
                        + "  lr_scale=" + String(self.lr_scale(it))
                    )
        if not do_eval:
            last = self.train_loss(reset=True)
        return last

    # ----- eval and generation --------------------------------------------

    def val_loss(mut self, n_windows: Int) raises -> Float64:
        """Mean per-token validation CE (nats) over `n_windows` windows, the
        same at every call, split over the ranks; dropout off. `n_windows` is
        a multiple of the global batch."""
        if n_windows <= 0 or n_windows % Self.BATCH != 0:
            raise Error(
                "val_loss: n_windows must be a positive multiple of the global"
                " batch (" + String(Self.BATCH) + ")"
            )
        self._set_training(False)
        for r in range(Self.N):
            self.ranks[r].val_ce.reset_accum["gpu"]()
            self.ranks[r].mode = _EVAL
            self.ranks[r].eval_batches = n_windows // Self.BATCH
        self.dist.forward_only(self.ranks)
        self._set_training(True)
        var l = 0.0
        for r in range(Self.N):
            var c = self.ranks[r].ctx
            l += Float64(self.ranks[r].val_ce.read_accum["gpu"](Optional(c)))
        return l / Float64(Self.N)

    def generate(
        mut self,
        prompt: String,
        n_tokens: Int,
        temperature: Float64 = 0.0,
        top_k: Int = 0,
        pad_id: Int = 0,
    ) raises -> String:
        """Continue `prompt` by `n_tokens`, `AutoregressiveTrainer.generate`'s
        rule: the last min(n, SEQ) ids at positions 0.., the next token read
        at the last of them. Rank 0's logits pick each token; under ZeRO-3
        every rank runs the forward (each holds a shard of every block)."""
        self._set_training(False)
        var all_ids = self.tok.encode(prompt)
        var prompt_len = len(all_ids)
        if prompt_len == 0:
            raise Error("generate: empty prompt")
        comptime n_run = 1 if Self.S.REPLICATED else Self.N
        for _ in range(n_tokens):
            var n_have = len(all_ids)
            var n_eff = n_have if n_have <= Self.SEQ else Self.SEQ
            var first = 0 if n_have <= Self.SEQ else n_have - Self.SEQ
            var win = List[Int](capacity=Self.SEQ)
            for t in range(Self.SEQ):
                win.append(all_ids[first + t] if t < n_eff else pad_id)
            for r in range(n_run):
                var c = self.ranks[r].ctx
                upload_onehot[DT, Self.SEQ, Self.VOCAB](
                    c, self.ranks[r].gen_in, win, 1
                )
                self.ranks[r].mode = _GENERATE
            comptime if Self.S.REPLICATED:
                self.ranks[0].run_rank(0)
            else:
                self.dist.forward_only(self.ranks)
            var c0 = self.ranks[0].ctx
            self.ranks[0].gen_out.download(c0)
            all_ids.append(
                sample_next_token[DT, Self.VOCAB](
                    self.ranks[0].gen_out,
                    (n_eff - 1) * Self.VOCAB,
                    temperature,
                    top_k,
                )
            )
        var gen = List[Int](capacity=n_tokens)
        for i in range(prompt_len, len(all_ids)):
            gen.append(all_ids[i])
        self._set_training(True)
        return self.tok.decode(gen)

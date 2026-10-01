"""LeWMDecoderTrainer — trains the reconstruction probe on frozen `emb` (storage).

Owns `LeWMDecoderLossGraph` + one Adam + owned IO scratch `Tensor`s.
`train_step(emb, tgt)` runs zero_grad → set_input → forward → mean-reduce →
seed 1/B → vjp → Adam.step on the decoder params (the encoder is external and
frozen — `emb` is fed as data). `recon_into` reads the `recon` node (patch
space) for visualization.

Storage surface: same shape as `LeWMTrainer` (owned `Tensor` scratch, stored
`ctx`, per-param Adam over `graph.for_each_param`, `node_output`); the public
facade keeps raw `TileTensor`/host-pointer args, bridged into the graph.
"""

from max.gpu import thread_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.memory import AddressSpace
from layout import Layout, TileTensor, row_major

from noeira.nn.constants import DT, TPB_REDUCE
from noeira.nn import Tensor, ParamVisitor, Kaiming, Adam
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.checkpoint import (
    BinaryCheckpointWriter,
    BinaryCheckpointReader,
    CheckpointReader,
    _write_file_bytes,
    _read_file_bytes,
    _is_v3_header,
    _split_lines,
)
from .decoder import LeWMDecoderLossGraph

from noeira.deep_agents.loss.seed_grad_inv_batch import seed_grad_inv_batch


def _dec_reduce_mean_acc_kernel[
    BATCH: Int
](
    src: Pointer[Scalar[DT], MutAnyOrigin],
    acc: Pointer[Scalar[DT], MutAnyOrigin],
):
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < BATCH:
        my_sum += src[unsafe_offset=k]
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my_sum)
    if t == 0:
        acc[unsafe_offset=0] = acc[unsafe_offset=0] + total[0] / Scalar[DT](BATCH)
        acc[unsafe_offset=1] = acc[unsafe_offset=1] + Scalar[DT](1.0)


struct _SaveVisitor(ParamVisitor):
    var vals: List[Scalar[DT]]

    def __init__(out self):
        self.vals = List[Scalar[DT]]()

    def visit[
        target: StaticString, N: Int
    ](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "gpu":
            param.download(ctx.value())
        for i in range(N):
            self.vals.append(param.data[i])


struct _LoadVisitor(ParamVisitor):
    var vals: List[Scalar[DT]]
    var idx: Int

    def __init__(out self, var vals: List[Scalar[DT]]):
        self.vals = vals^
        self.idx = 0

    def visit[
        target: StaticString, N: Int
    ](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        param.ensure(N)
        for i in range(N):
            param.data[i] = self.vals[self.idx]
            self.idx += 1
        param.n = N
        # Advance `version` so `Linear`'s version-gated `w_pad` / `w_bf` pick
        # up the new weight (see `_LoadVisitor` in trainer.mojo).
        param.version += 1
        comptime if target == "gpu":
            param.upload_resident(ctx.value())


struct LeWMDecoderTrainer[
    EMB: Int,
    HID: Int,
    N_Q: Int,
    PATCH_PX: Int,
    FF: Int,
    N_LAYERS: Int,
    BATCH: Int,
    train_target: StaticString = "cpu",
](Movable & Deinitable):
    comptime DG = LeWMDecoderLossGraph[
        Self.EMB, Self.HID, Self.N_Q, Self.PATCH_PX, Self.FF, Self.N_LAYERS
    ]
    comptime RECON = Self.N_Q * Self.PATCH_PX

    var graph: Self.DG
    var opt: Adam
    var ctx: Optional[DeviceContext]
    var loss_out: Tensor  # per-sample loss [BATCH]
    var grad_seed: Tensor  # constant 1/BATCH backward seed [BATCH]
    # Dummy `tgt` so a cold `recon_into` (no prior train_step) can run the
    # full loss-graph forward — `recon` (computed before `loss`) is
    # tgt-independent, so the loss it computes is ignored.
    var tgt_dummy: Tensor
    var _loss_acc_dev: Optional[DeviceBuffer[DT]]

    def __init__(out self):
        self.graph = Self.DG()
        self.opt = Adam()
        self.ctx = None
        self.loss_out = Tensor()
        self.grad_seed = Tensor()
        self.tgt_dummy = Tensor()
        self._loss_acc_dev = None

    @staticmethod
    def make(
        lr: Scalar[DT] = 1e-3,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        var t = Self()
        t.graph = Self.DG.make[Self.train_target, Kaiming](ctx=ctx)
        t.opt = Adam(lr=lr)
        t.ctx = ctx
        comptime if Self.train_target == "gpu":
            var c = ctx.value()
            t.loss_out = Tensor.alloc_gpu(c, Self.BATCH)
            t.grad_seed = Tensor.alloc_gpu(c, Self.BATCH)
            t.tgt_dummy = Tensor.alloc_gpu(c, Self.BATCH * Self.RECON)
            var acc = c.enqueue_create_buffer[DT](2)
            acc.enqueue_fill(0.0)
            t._loss_acc_dev = acc^
        else:
            t.loss_out = Tensor.alloc(Self.BATCH)
            t.grad_seed = Tensor.alloc(Self.BATCH)
            t.tgt_dummy = Tensor.alloc(Self.BATCH * Self.RECON)
        seed_grad_inv_batch[Self.train_target, Self.BATCH](
            t.grad_seed.lt[
                Self.train_target, Layout.row_major(Self.BATCH, 1)
            ](),
            ctx=ctx,
        )
        return t^

    def _seed_input[
        slot_name: StaticString, N: Int
    ](
        mut self,
        src: TileTensor[
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            
            origin=MutAnyOrigin,
            ...,
        ],
    ) raises:
        """Bridge a raw input tile into the named graph input slot."""
        var tt = Tensor()
        comptime if Self.train_target == "cpu":
            tt.data = List[Scalar[DT]](length=N, fill=Scalar[DT](0))
            for i in range(N):
                tt.data[i] = rebind[Scalar[DT]](src.ptr[unsafe_offset=i])
            tt.n = N
        else:
            var c = self.ctx.value()
            var sp = rebind[Pointer[Scalar[DT], MutAnyOrigin]](src.ptr)
            tt.dev = DeviceBuffer[DT](c, sp, N, owning=False)
            tt.n = N
        self.graph.set_input[slot_name, Self.BATCH](tt, self.ctx)

    def train_step[
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        emb: TileTensor[
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            
            origin=MutAnyOrigin,
            ...,
        ],
        tgt: TileTensor[
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            
            origin=MutAnyOrigin,
            ...,
        ],
    ) raises -> Scalar[DT]:
        self.graph.zero_grad[Self.train_target](self.ctx)
        self._seed_input["emb", Self.BATCH * Self.EMB](emb)
        self._seed_input["tgt", Self.BATCH * Self.RECON](tgt)
        self.graph.forward[Self.BATCH, Self.train_target, POLICY](
            self.loss_out, self.ctx
        )

        var m: Scalar[DT] = 0.0
        comptime if Self.train_target == "cpu":
            for b in range(Self.BATCH):
                m += self.loss_out.data[b]
            m /= Scalar[DT](Self.BATCH)
        else:
            var c = self.ctx.value()
            comptime red = _dec_reduce_mean_acc_kernel[Self.BATCH]
            c.enqueue_function[red](
                self.loss_out.dev.value().unsafe_ptr(),
                self._loss_acc_dev.value().unsafe_ptr(),
                grid_dim=1,
                block_dim=TPB_REDUCE,
            )

        self.graph.vjp[Self.BATCH, Self.train_target, POLICY](
            self.grad_seed, self.ctx
        )
        self.opt.begin_step()
        self.graph.for_each_param[Self.train_target](self.opt, self.ctx)
        return m

    def recon_into[
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        emb: TileTensor[
            dtype=DT,
            address_space=AddressSpace.GENERIC,
            
            origin=MutAnyOrigin,
            ...,
        ],
        recon_host: Pointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Forward-only readout of the `recon` node (B, N_Q·PATCH_PX, patch
        space) into a host buffer. Caller un-patchifies for display."""
        # The loss graph's forward computes loss = MSE(recon, tgt), so BOTH
        # inputs must point at valid buffers even though we only read `recon`
        # (computed before `loss`, tgt-independent). Bind the dummy tgt so a
        # cold recon_into (loaded weights, no prior train_step) doesn't crash.
        self._seed_input["emb", Self.BATCH * Self.EMB](emb)
        self.graph.set_input["tgt", Self.BATCH](self.tgt_dummy, self.ctx)
        self.graph.forward[Self.BATCH, Self.train_target, POLICY](
            self.loss_out, self.ctx
        )
        comptime N = Self.BATCH * Self.RECON
        ref src = self.graph.node_output["recon"]()
        comptime if Self.train_target == "cpu":
            for i in range(N):
                recon_host[unsafe_offset=i] = src.data[i]
        else:
            var c = self.ctx.value()
            src.download(c)
            for i in range(N):
                recon_host[unsafe_offset=i] = src.data[i]

    def read_loss_accum(mut self) raises -> Scalar[DT]:
        comptime if Self.train_target == "gpu":
            var c = self.ctx.value()
            var h = c.enqueue_create_host_buffer[DT](2)
            c.enqueue_copy(h, self._loss_acc_dev.value())
            c.synchronize()
            var s = h.unsafe_ptr()[unsafe_offset=0]
            var n = h.unsafe_ptr()[unsafe_offset=1]
            if n == Scalar[DT](0.0):
                return Scalar[DT](0.0)
            return s / n
        else:
            return Scalar[DT](0.0)

    def reset_loss_accum(mut self) raises:
        comptime if Self.train_target == "gpu":
            self._loss_acc_dev.value().enqueue_fill(0.0)

    def save_params(mut self, path: String, save_moments: Bool = True) raises:
        """Write a v3 binary named checkpoint (params + state; same format
        as LeWMTrainer.save_params). Replaces the legacy positional flat
        text; `load_params` still reads it."""
        var w = BinaryCheckpointWriter(save_moments)
        w.mode = 0
        self.graph.for_each_param[Self.train_target, BinaryCheckpointWriter](
            w, self.ctx
        )
        w.mode = 1
        self.graph.for_each_state[Self.train_target, BinaryCheckpointWriter](
            w, self.ctx
        )
        _write_file_bytes(path, w.content)

    def load_params(mut self, path: String) raises:
        """Load a v3 binary / v2 named text / legacy flat-text checkpoint
        (header-dispatched; see LeWMTrainer.load_params)."""
        var bytes = _read_file_bytes(path)
        if _is_v3_header(bytes):
            var r = BinaryCheckpointReader(bytes^)
            r.mode = 0
            self.graph.for_each_param[
                Self.train_target, BinaryCheckpointReader
            ](r, self.ctx)
            r.mode = 1
            self.graph.for_each_state[
                Self.train_target, BinaryCheckpointReader
            ](r, self.ctx)
            return
        var content: String
        with open(path, "r") as f:
            content = String(f.read())
        var lines = _split_lines(content)
        if len(lines) > 0 and lines[0].startswith("storage-ckpt"):
            var body = List[String]()
            for li in range(1, len(lines)):
                body.append(lines[li])
            var r2 = CheckpointReader(body^)
            r2.mode = 0
            self.graph.for_each_param[Self.train_target, CheckpointReader](
                r2, self.ctx
            )
            r2.mode = 1
            self.graph.for_each_state[Self.train_target, CheckpointReader](
                r2, self.ctx
            )
            return
        # Legacy positional flat text: "count\nval\n..." (params only).
        var n = Int(lines[0])
        var vals = List[Scalar[DT]]()
        for i in range(n):
            vals.append(Scalar[DT](Float64(lines[i + 1])))
        var v = _LoadVisitor(vals^)
        self.graph.for_each_param[Self.train_target, _LoadVisitor](v, self.ctx)

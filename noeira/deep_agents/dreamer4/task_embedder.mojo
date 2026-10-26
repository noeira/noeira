"""TaskEmbedder — task-conditioned agent-token input (model.py:TaskEmbedder).

Produces the agent-token INPUT for `Dreamer4Dynamics` (paper §3.3): per
sequence `b`, a task embedding is looked up and added to a learned `agent_base`,
then broadcast across all `T` frames and all `NAGENT` agent tokens:

    e_b = task_table[id_b] + agent_base                       (D-vector)
    agent_in[b, t, a, :] = e_b   for all t∈[0,T), a∈[0,NAGENT)

In nn-BATCH = B·T layout the output is `[B·T, NAGENT·D]`, exactly the shape
`Dreamer4Dynamics.set_agent_in` consumes. The reference uses `use_ids=True`
(an embedding table over `n_tasks`); the `task_proj` (continuous task vector)
variant is deferred.

This is a bespoke component, NOT a `Module`: its "input" is a per-sequence task
id (a host control input, like the dynamics' signal/step indices) and its output
broadcasts over (T, NAGENT) — neither fits the fixed `forward(inputs, output)`
shape. It owns the broadcast (`embed_into`) and the matching gradient reduction
(`accumulate_grad`), and exposes `for_each_param`/`zero_grad` so a facade can
fold it into a composite optimizer. CPU + GPU.

Backward: the agent input is broadcast, so the grad of `e_b` is the sum of the
incoming grad over (t, a). That `ge_b` accumulates into BOTH `agent_base` (summed
over all b) and `task_table[id_b]` (index-masked batch reduction — one thread
per (task_row, channel) looping the batch, so no atomics even when sequences
share a task id; same trick as the dynamics signal/step tables).
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from max.gpu.memory import AddressSpace
from layout import TileTensor, row_major
from noeira.core.tiles import Tile, tile

from noeira.nn.constants import DT, TPB
from noeira.nn.core.param import Param, ParamVisitor, ParamWalkable
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.walkers import for_each_param_auto, zero_grad_auto


# ── GPU kernels ─────────────────────────────────────────────────────────
# Broadcast e_b = table[id_b] + base into every (t, a) of the output grid.
def _te_embed_kernel[
    B: Int, T: Int, NAGENT: Int, D: Int, NTASK: Int
](
    ids: Tile[DT, B],
    table: Tile[DT, NTASK * D],
    base: Tile[DT, D],
    out_buf: Tile[DT, B * T * NAGENT * D],
):
    comptime AG = NAGENT * D
    var e = Int(global_idx.x)
    if e >= B * T * AG:
        return
    var bt = e // AG
    var k = e % AG
    var d = k % D
    var b = bt // T
    var idb = Int(rebind[Scalar[DT]](ids.ptr[unsafe_offset=b]) + Scalar[DT](0.5))
    out_buf.ptr[unsafe_offset=e] = (
        rebind[Scalar[DT]](table.ptr[unsafe_offset=idb * D + d])
        + rebind[Scalar[DT]](base.ptr[unsafe_offset=d])
    )


# agent_base grad: gbase[d] += Σ_{b,t,a} grad_in[b,t,a,d].
def _te_grad_base_kernel[
    B: Int, T: Int, NAGENT: Int, D: Int
](
    grad_in: Tile[DT, B * T * NAGENT * D],
    gbase: Tile[DT, D],
):
    comptime AG = NAGENT * D
    var d = Int(global_idx.x)
    if d >= D:
        return
    var acc = Scalar[DT](0.0)
    for b in range(B):
        for t in range(T):
            var bt = b * T + t
            for a in range(NAGENT):
                acc += rebind[Scalar[DT]](grad_in.ptr[unsafe_offset=bt * AG + a * D + d])
    gbase.ptr[unsafe_offset=d] = rebind[Scalar[DT]](gbase.ptr[unsafe_offset=d]) + acc


# task_table grad (index-masked batch reduction): gtab[v,d] += Σ_{b: id_b==v}
# Σ_{t,a} grad_in[b,t,a,d]. One thread per (v, d) loops the batch → no atomics.
def _te_grad_table_kernel[
    B: Int, T: Int, NAGENT: Int, D: Int, NTASK: Int
](
    grad_in: Tile[DT, B * T * NAGENT * D],
    ids: Tile[DT, B],
    gtab: Tile[DT, NTASK * D],
):
    comptime AG = NAGENT * D
    var ed = Int(global_idx.x)
    if ed >= NTASK * D:
        return
    var v = ed // D
    var d = ed % D
    var acc = Scalar[DT](0.0)
    for b in range(B):
        var idb = Int(rebind[Scalar[DT]](ids.ptr[unsafe_offset=b]) + Scalar[DT](0.5))
        if idb == v:
            for t in range(T):
                var bt = b * T + t
                for a in range(NAGENT):
                    acc += rebind[Scalar[DT]](grad_in.ptr[unsafe_offset=bt * AG + a * D + d])
    gtab.ptr[unsafe_offset=ed] = rebind[Scalar[DT]](gtab.ptr[unsafe_offset=ed]) + acc


struct TaskEmbedder[D: Int, NTASK: Int, NAGENT: Int](ParamWalkable):
    comptime AG_DIM: Int = Self.NAGENT * Self.D

    var task_table: Param["task_table", True, Self.NTASK * Self.D]
    var agent_base: Param["agent_base", False, Self.D]

    var cache_ids: List[Int]                    # [B] task id per sequence
    var ids_dev: Optional[DeviceBuffer[DT]]     # [B] uploaded ids (gpu)
    var ids_hbuf: Optional[HostBuffer[DT]]      # host staging for ids
    var scratch_b: Int

    def __init__(out self):
        self.task_table = Param["task_table", True, Self.NTASK * Self.D]()
        self.agent_base = Param["agent_base", False, Self.D]()
        self.cache_ids = List[Int]()
        self.ids_dev = None
        self.ids_hbuf = None
        self.scratch_b = 0

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "TaskEmbedder: target must be 'cpu' or 'gpu'"
        )
        comptime NT = Self.NTASK * Self.D
        var m = Self()
        m.task_table = Param["task_table", True, NT].make[target](ctx)
        m.agent_base = Param["agent_base", False, Self.D].make[target](ctx)
        # storage `init_weight` does the host-fill AND (on GPU) the upload.
        INIT.init_weight[target](
            m.task_table.val, NT, Self.NTASK, Self.D, ctx
        )
        INIT.init_weight[target](
            m.agent_base.val, Self.D, 1, Self.D, ctx
        )
        return m^

    @staticmethod
    def display_label() -> String:
        return String("TaskEmbedder")

    def _cache_ids(mut self, task_ids: Pointer[Scalar[DT], MutAnyOrigin], B: Int):
        if len(self.cache_ids) < B:
            self.cache_ids.resize(B, 0)
        for b in range(B):
            self.cache_ids[b] = Int(Float64(task_ids[unsafe_offset=b]) + 0.5)

    def _ensure_ids_gpu(mut self, B: Int, ctx: Optional[DeviceContext]) raises:
        if self.scratch_b < B:
            var c = ctx.value()
            self.ids_dev = c.enqueue_create_buffer[DT](B)
            self.ids_hbuf = c.enqueue_create_host_buffer[DT](B)
            c.synchronize()
            self.scratch_b = B

    def embed_into[
        target: StaticString, B: Int, T: Int
    ](
        mut self,
        task_ids: Pointer[Scalar[DT], MutAnyOrigin],
        dst: Pointer[Scalar[DT], MutAnyOrigin],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Fill `dst` [B·T, NAGENT·D] (CPU host ptr / GPU device ptr) with the
        broadcast task embeddings. `task_ids` is a host [B] fp buffer of exact
        task ids."""
        self._cache_ids(task_ids, B)
        comptime AG = Self.AG_DIM
        comptime if target == "cpu":
            var tab = TileTensor(
                self.task_table.val.data, row_major[Self.NTASK * Self.D]()
            )
            var base = TileTensor(self.agent_base.val.data, row_major[Self.D]())
            for b in range(B):
                var idb = self.cache_ids[b]
                for t in range(T):
                    var bt = b * T + t
                    for a in range(Self.NAGENT):
                        for d in range(Self.D):
                            dst[unsafe_offset=bt * AG + a * Self.D + d] = (
                                tab[idb * Self.D + d] + base[d]
                            )
        else:
            var c = ctx.value()
            self._ensure_ids_gpu(B, ctx)
            var ih = self.ids_hbuf.value()
            for b in range(B):
                ih.unsafe_ptr()[unsafe_offset=b] = Scalar[DT](Float64(self.cache_ids[b]))
            c.enqueue_copy(self.ids_dev.value(), ih)
            var ids_lt = tile[B](self.ids_dev.value())
            var tab_lt = self.task_table.val.tt["gpu", Self.NTASK * Self.D]()
            var base_lt = self.agent_base.val.tt["gpu", Self.D]()
            var out_lt = tile[B * T * AG](dst)
            comptime ek = _te_embed_kernel[B, T, Self.NAGENT, Self.D, Self.NTASK]
            c.enqueue_function[ek](
                ids_lt, tab_lt, base_lt, out_lt,
                grid_dim=(B * T * AG + TPB - 1) // TPB, block_dim=TPB,
            )

    def accumulate_grad[
        target: StaticString, B: Int, T: Int
    ](
        mut self,
        grad_in: Pointer[Scalar[DT], MutAnyOrigin],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Accumulate the grad of the broadcast agent input (`grad_in`
        [B·T, NAGENT·D], = `dyn.grad_agent_in_*`) into the params. Uses the
        task ids cached by the preceding `embed_into`."""
        comptime AG = Self.AG_DIM
        comptime if target == "cpu":
            var gtab = TileTensor(
                self.task_table.grd.data, row_major[Self.NTASK * Self.D]()
            )
            var gbase = TileTensor(self.agent_base.grd.data, row_major[Self.D]())
            for b in range(B):
                var idb = self.cache_ids[b]
                for d in range(Self.D):
                    var ge = Scalar[DT](0.0)
                    for t in range(T):
                        var bt = b * T + t
                        for a in range(Self.NAGENT):
                            ge += grad_in[unsafe_offset=bt * AG + a * Self.D + d]
                    gbase[d] += ge
                    gtab[idb * Self.D + d] += ge
        else:
            var c = ctx.value()
            var gin_lt = tile[B * T * AG](grad_in)
            var ids_lt = tile[B](self.ids_dev.value())
            var gbase_lt = self.agent_base.grd.tt["gpu", Self.D]()
            var gtab_lt = self.task_table.grd.tt["gpu", Self.NTASK * Self.D]()
            comptime bk = _te_grad_base_kernel[B, T, Self.NAGENT, Self.D]
            c.enqueue_function[bk](
                gin_lt, gbase_lt,
                grid_dim=(Self.D + TPB - 1) // TPB, block_dim=TPB,
            )
            comptime tk = _te_grad_table_kernel[
                B, T, Self.NAGENT, Self.D, Self.NTASK
            ]
            c.enqueue_function[tk](
                gin_lt, ids_lt, gtab_lt,
                grid_dim=(Self.NTASK * Self.D + TPB - 1) // TPB, block_dim=TPB,
            )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        for_each_param_auto[Self, V, target](self, visitor, ctx, prefix)

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        zero_grad_auto[Self, target](self, ctx)

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        """No State: the table and the base are both trainable Params."""
        pass

"""ParamArena — contiguous param packing shared by the grouped optimizers + clip.

`adopt[target](model)` (GPU; NO-OP on CPU) packs every Param into two contiguous
device buffers — `val` and `grd` — and REBINDS each Param's val/grd device buffer
to a `create_sub_buffer` slice. Forward reads / backward writes the slices
transparently, so all grads land contiguously in `grd`. A per-element `decay_mask`
(1 where the param wants weight decay) carries AdamW's selective decay with no
per-param offset scan.

This is the rule-of-three extraction: `Adam` (adds m/v arenas), `SGD` (stateless),
and arena grad-clip (`grad_clip.clip_arena_grads`) all build on it. The optimizer
owns its ParamArena; the model's param slices reference it (DeviceBuffer is
refcounted → destruction order is safe). GPU-only: on CPU the per-param path has
no launch overhead to collapse, so `adopt` does nothing and `adopted` stays False.

ParamArena IS the placement `ParamVisitor` (its `visit` reads its own `val`/`grd`
+ offset → no ownership transfer); the placement walk is comptime-gated to GPU so
the device ops never compile into the CPU path.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.fill import fill_dev
from ..core.param import ParamVisitor, ParamVisitorRT, walk_params
from ..core.param import ParamWalkable
from ..core.named_params import named_params


# ── slice alignment ──────────────────────────────────────────────────────────
#
# ⚠ EVERY param slice must start on a 16-byte boundary, and this is the ONLY
# place that rule is written. MAX's `multistage_gemm` loads its A/B operands
# with 16-byte (float4) vectors, so a weight whose arena offset is not a
# multiple of 4 floats makes EVERY tile load in the GEMM misaligned:
#
#   Invalid __global__ read of size 16 bytes ... Access at 0x...078 is misaligned
#   CUDA call failed: CUDA_ERROR_LAUNCH_FAILED (unspecified launch failure)
#
# Packing at element granularity is what breaks it: ONE param of odd size
# (a LayerNorm over an odd observation width, a bias of odd width) shifts every
# later param off the boundary, and two of them land it at 8 mod 16 — the worst
# case, since even a float2 load then splits. The alignment is invisible to
# everything downstream: the gaps are zero in `val`/`grd`/`m`/`v` (all
# zero-filled at alloc) and zero in `decay_mask`, so the flat Adam/SGD/polyak/
# grad-clip kernels over `[0, total)` read and write zeros there.
#
# 128 B (not the minimum 16) because it is also the cache-line/`cp.async`
# granularity, and the cost is at most 31 floats per param.
comptime PARAM_ALIGN = 32  # elements; 128 B at float32


def align_param_off(off: Int) -> Int:
    """Round an arena element offset up to the next `PARAM_ALIGN` boundary.

    Called at EVERY offset walk over the arena — the sizing pass, the
    decay-mask pass and the placement walk in this file, and `_MomentPlacer`'s
    walk in `adam.mojo`. They must agree exactly: `m`/`v` alias `val`/`grd` by
    offset, so a walk that skips this rounds a param's moments onto a DIFFERENT
    param's values, silently."""
    return ((off + PARAM_ALIGN - 1) // PARAM_ALIGN) * PARAM_ALIGN


struct ParamArena(Movable & ParamVisitor & ParamVisitorRT):
    var val: Tensor  # contiguous param-value arena
    var grd: Tensor  # contiguous gradient arena
    var decay_mask: Tensor  # per-element 0/1 weight-decay gate
    var total: Int
    var capacity: Int  # allocated length of `val`/`grd` (>= total; see `adopt`'s `pad_to`)
    var adopted: Bool
    var _off: Int  # running offset during the placement walk

    def __init__(out self):
        self.val = Tensor()
        self.grd = Tensor()
        self.decay_mask = Tensor()
        self.total = 0
        self.capacity = 0
        self.adopted = False
        self._off = 0

    def visit_rt[target: StaticString](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        n: Int,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        """Placement (adopt walk, GPU only): copy the param's value into `val` at
        the running offset and rebind its val/grd buffers to arena slices.
        `param`/`grad` ARE the val/grd Tensors; the mut refs chain back to the
        model's Param, so the rebinds persist."""
        comptime if target == "gpu":
            var c = ctx.value()
            self._off = align_param_off(self._off)
            var vsub = self.val.dev.value().create_sub_buffer[DT](self._off, n)
            c.enqueue_copy(vsub, param.dev.value())  # preserve init values
            param.dev = Optional(vsub)
            param.n = n
            var gsub = self.grd.dev.value().create_sub_buffer[DT](self._off, n)
            grad.dev = Optional(gsub)
            grad.n = n
            self._off += n

    def visit[target: StaticString, N: Int](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.visit_rt[target](name, param, grad, m, v, N, apply_decay, ctx)
    def adopt[
        target: StaticString, M: ParamWalkable
    ](
        mut self,
        mut model: M,
        ctx: Optional[DeviceContext] = None,
        pad_to: Int = 1,
    ) raises:
        """Pack `model` into the arena (GPU); NO-OP on CPU. Call ONCE after the
        model is made + initialized, before the first step.

        `pad_to` rounds the ALLOCATION of `val`/`grd` up to a multiple of it
        (`capacity`); `total` and everything that walks `[0, total)` are
        unchanged, and the tail stays zero. A sharded optimizer uses it to view
        the arena as `[capacity / PARAM_ALIGN, PARAM_ALIGN]` rows
        (`nn/distributed/zero.mojo`). The default 1 allocates exactly `total`."""
        comptime if target == "gpu":
            var c = ctx.value()
            var nps = named_params["gpu"](model)
            var total = 0
            for i in range(len(nps)):
                total = align_param_off(total) + nps[i].size
            self.total = total

            var dm = Tensor.alloc(total)  # host decay mask
            var off = 0
            for i in range(len(nps)):
                var d = Scalar[DT](1.0) if nps[i].decay else Scalar[DT](0.0)
                off = align_param_off(off)
                for k in range(nps[i].size):
                    dm.data[off + k] = d
                off += nps[i].size
            dm.upload(c)
            self.decay_mask = dm^

            self.capacity = ((total + pad_to - 1) // pad_to) * pad_to
            self.val = Tensor.alloc_gpu(c, self.capacity)  # zeroed
            self.grd = Tensor.alloc_gpu(c, self.capacity)
            self._off = 0
            walk_params["gpu"](model, self, Optional(c))
            self.adopted = True

    def adopt_multi[
        target: StaticString, *Ms: ParamWalkable
    ](mut self, ctx: Optional[DeviceContext], mut *models: *Ms) raises:
        """Pack N models into ONE arena, in pack order (GPU); NO-OP on CPU.

        ⚠ `adopt` says "call ONCE" and means it — it resets `total` and `_off`
        and reallocates, so calling it per model leaves only the LAST one in
        the arena while `adopted` reads True. A trainable set spread over
        several objects (SmolVLA's is an expert plus four projections) needs
        this instead, and needs it to be one arena rather than several: a
        GLOBAL grad-norm clip is not the same operation as N independent ones,
        and `clip_arena_grads` clips whatever the arena holds.

        Same shape as `save_params_multi` — one pass to size, one to place,
        models walked in the order given, and the caller must use that same
        order everywhere after.
        """
        comptime if target == "gpu":
            var c = ctx.value()
            var total = 0
            comptime for i in range(models.__len__()):
                var nps = named_params["gpu"](models[i])
                for j in range(len(nps)):
                    total = align_param_off(total) + nps[j].size
            self.total = total

            var dm = Tensor.alloc(total)
            var off = 0
            comptime for i in range(models.__len__()):
                var nps = named_params["gpu"](models[i])
                for j in range(len(nps)):
                    var d = (
                        Scalar[DT](1.0) if nps[j].decay else Scalar[DT](0.0)
                    )
                    off = align_param_off(off)
                    for k in range(nps[j].size):
                        dm.data[off + k] = d
                    off += nps[j].size
            dm.upload(c)
            self.decay_mask = dm^

            self.capacity = total
            self.val = Tensor.alloc_gpu(c, total)
            self.grd = Tensor.alloc_gpu(c, total)
            # ⚠ `_off` is reset ONCE and then advances ACROSS the models —
            # that is the whole difference from calling `adopt` N times.
            self._off = 0
            comptime for i in range(models.__len__()):
                walk_params["gpu"](models[i], self, Optional(c))
            self.adopted = True

    def zero_grad(mut self, c: DeviceContext) raises:
        """Zero the whole grad arena in ONE launch (vs N per-param fills).

        ⚠ NOT `enqueue_fill`: on Metal that is a synchronize per call
        (`nn/core/fill.mojo`), and this runs once per optimizer step."""
        if self.adopted and self.total > 0:
            fill_dev(self.grd.dev.value(), self.total, Scalar[DT](0), c)


def _polyak_kernel(
    target: Pointer[Scalar[DT], MutAnyOrigin],
    online: Pointer[Scalar[DT], MutAnyOrigin],
    total_arg: Int64,
    tau: Scalar[DT],
):
    """target[i] = (1-τ)·target[i] + τ·online[i] over the whole value arena."""
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var total = Int(total_arg)
    var i = Int(global_idx.x)
    if i < total:
        target[unsafe_offset=i] = (Scalar[DT](1.0) - tau) * target[unsafe_offset=i] + tau * online[unsafe_offset=i]


def polyak_arenas(
    mut target: ParamArena,
    mut online: ParamArena,
    tau: Scalar[DT],
    ctx: DeviceContext,
) raises:
    """Grouped target-net soft-update: `target = (1-τ)·target + τ·online` in ONE
    kernel over the contiguous value arenas (vs N per-param launches via
    `Module.polyak_from`). Both models must be arena-backed (same param layout →
    same `total`). `online` is `mut` for the GPU-ABI (read-only in the kernel)."""
    if target.total == 0 or target.total != online.total:
        raise Error(
            "polyak_arenas: arena size mismatch (target "
            + String(target.total) + " vs online " + String(online.total) + ")"
        )
    var nblk = (target.total + TPB - 1) // TPB
    ctx.enqueue_function[_polyak_kernel](
        target.val.dev.value(),
        online.val.dev.value(),
        Int64(target.total),
        tau,
        grid_dim=nblk,
        block_dim=TPB,
    )

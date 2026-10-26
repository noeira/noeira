"""Owned scratch tensors for the RNE velocity derivative / implicit integrator
(migration P2 / Stage-I).

`ImplicitScratch` is the stateful replacement for the `ws_implicit_*` region
of the legacy workspace slab (`implicit_extra_workspace_size` in
gpu/constants.mojo): one owned `TensorImpl` per intermediate array of
`compute_rne_vel_derivative`, allocated once by `ImplicitIntegrator`
and reused every step.

These are stored as tensors (not per-thread InlineArrays) on PURPOSE: the
big cross-body intermediates are `NBODY*6*NV` / `NV*6*NV` elements — for the
humanoid (NV≈27, NBODY≈14) that is thousands of floats per env, which as
GPU-thread-local InlineArrays would blow local memory (the RK4-ELLIPTIC OOM
lesson). Holding them in device global memory lets one per-env function
serve both the CPU and GPU targets, bit-identically.

Only the persistent cross-body arrays live here; the tiny 6×6 / 6-vector
loop temporaries stay as InlineArrays inside the kernel (GPU-safe).
"""

from max.gpu.host import DeviceContext

from noeira.nn.core.tensor import TensorImpl

from .dims import DimsLike


@always_inline
def _pos(n: Int) -> Int:
    return n if n > 0 else 1


struct ImplicitScratch[
    DTYPE: DType,
    D: DimsLike,
    BATCH: Int = 1,
](Movable):
    """RNE-velocity-derivative scratch: one owned tensor per `ws_implicit_*`
    region (9 tensors).

    ⚠ Second container on `D` (1c.3). Unlike `Rk4Scratch` it DOES escape into
    a signature — `qderiv.compute_qderiv`, one site — so that signature is
    part of this change. It is still spellable with a local `Dims[...]`
    adapter rather than dragging `compute_qderiv`'s callers along, because
    `Dims[nv=9, nbody=8]` names one type however it is written.
    """

    # Body unchanged — see `Rk4Scratch` for why the aliases are here.
    comptime NV = Self.D.NV
    comptime NBODY = Self.D.NBODY

    var cinert: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*10]
    var cdof_sc: TensorImpl[Self.DTYPE]  # [BATCH, NV*6]
    var cvel_sc: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*6]
    var cdof_dot: TensorImpl[Self.DTYPE]  # [BATCH, NV*6]
    var dcvel: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*6*NV]
    var dcdofdot: TensorImpl[Self.DTYPE]  # [BATCH, NV*6*NV]
    var dcacc: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*6*NV]
    var dcfrcbody: TensorImpl[Self.DTYPE]  # [BATCH, NBODY*6*NV]
    var qderiv: TensorImpl[Self.DTYPE]  # [BATCH, NV*NV]

    # The provider as a VALUE (3a). See the same field on `Data`; six
    # dispatchers (`ldl_*`, `lu_*`, `compute_m_inv*`) take this container and
    # nothing else, so this is where their runtime layouts get their extents.
    var dims: Self.D

    def __init__(out self) raises:
        """Dimensions from the comptime provider; raises on a dynamic one.
        See `DimsLike.comptime_value`."""
        self = Self(Self.D.comptime_value())

    def __init__(out self, dims: Self.D) raises:
        """Dimensions passed in, and ALLOCATED FROM (3b).

        ⚠ Every size below reads `dims`, never a comptime member. Those
        members still exist and still size the GPU layouts, but they are
        `DIM_POISON` on a dynamic provider, so an `alloc` that read one
        would ask for a NEGATIVE length. See the twin on `Data`."""
        self.dims = dims
        comptime B = Self.BATCH
        self.cinert = TensorImpl[Self.DTYPE].alloc(_pos(B * dims.get_nbody() * 10))
        self.cdof_sc = TensorImpl[Self.DTYPE].alloc(_pos(B * dims.get_nv() * 6))
        self.cvel_sc = TensorImpl[Self.DTYPE].alloc(_pos(B * dims.get_nbody() * 6))
        self.cdof_dot = TensorImpl[Self.DTYPE].alloc(_pos(B * dims.get_nv() * 6))
        self.dcvel = TensorImpl[Self.DTYPE].alloc(
            _pos(B * dims.get_nbody() * 6 * dims.get_nv())
        )
        self.dcdofdot = TensorImpl[Self.DTYPE].alloc(
            _pos(B * dims.get_nv() * 6 * dims.get_nv())
        )
        self.dcacc = TensorImpl[Self.DTYPE].alloc(
            _pos(B * dims.get_nbody() * 6 * dims.get_nv())
        )
        self.dcfrcbody = TensorImpl[Self.DTYPE].alloc(
            _pos(B * dims.get_nbody() * 6 * dims.get_nv())
        )
        self.qderiv = TensorImpl[Self.DTYPE].alloc(_pos(B * dims.get_nv() * dims.get_nv()))

    def upload_all(mut self, ctx: DeviceContext) raises:
        """Create device buffers for every scratch tensor (once, at setup —
        contents are produced on-device thereafter)."""
        self.cinert.upload(ctx)
        self.cdof_sc.upload(ctx)
        self.cvel_sc.upload(ctx)
        self.cdof_dot.upload(ctx)
        self.dcvel.upload(ctx)
        self.dcdofdot.upload(ctx)
        self.dcacc.upload(ctx)
        self.dcfrcbody.upload(ctx)
        self.qderiv.upload(ctx)

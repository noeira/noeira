"""Owned RK4 stage tensors for the per-field tensor pipeline (P2).

`Rk4Scratch` is the stateful replacement for the RK4-extra region of the
flat workspace slab (`rk4_extra_workspace_size` = NQ + 7*NV, addressed via
`ws_rk4_*_offset` in gpu/constants.mojo): one owned `TensorImpl` per
logical region, allocated once by `RK4Integrator` and reused every
step. Region inventory mirrors the legacy layout exactly:

    q0 (NQ)  — qpos saved at stage 0 (all stages integrate FROM q0)
    v0 (NV)  — qvel saved at stage 0 (C[0] = v0)
    A  (4*NV in legacy; A0/A1/A2 here) — per-stage constrained qacc.
              The legacy A[3] slot is allocated but never written: the
              combine kernel reads A[3] straight from qacc_constrained,
              and the combine REUSES the A[0] slot to hold v_combined —
              so A0 doubles as the combine's velocity buffer here too.
    C1 (NV)  — stage-2 velocity intermediate: v0 + dt/2 * A[0]
    C2 (NV)  — stage-3 velocity intermediate: v0 + dt/2 * A[1]
"""

from max.gpu.host import DeviceContext

from noeira.nn.core.tensor import TensorImpl

from .dims import DimsLike
from noeira.core.tiles import RowMajor


struct Rk4Scratch[
    DTYPE: DType,
    D: DimsLike,
    BATCH: Int = 1,
](Movable):
    """RK4 stage scratch: one owned tensor per legacy `ws_rk4_*` region
    (the unused legacy A[3] slot is not materialized).

    ⚠ FIRST CONTAINER ON `D` (phase 1c.2), and it was chosen because it is the
    ONLY one that can go alone: `Rk4Scratch` appears in **zero** function
    signatures tree-wide — it is a field of `RK4Integrator` and nothing else.
    Every other container escapes into signatures (`Data` into 311 across 98
    files), and a provider type must match along the whole call chain —
    `Dims[nq=NV]` and `ModelDims[MD]` are DIFFERENT TYPES even when every
    value agrees — so converting those forces the signature sweep with them.
    §10.4 lists 1c and 2a as separate phases; the type system disagrees.
    """

    # ⚠ THE BODY BELOW IS UNTOUCHED, DELIBERATELY. Re-pointing `Self.NQ` at
    # `Self.D.NQ` here means the ~40 uses of `Self.NQ`/`Self.NV` in the
    # allocation and layout code keep their exact spelling, so the diff is the
    # parameter list plus these two lines. A container conversion that also
    # retyped its body would put a transcription error and a dimension error
    # in the same commit with one gate to catch both.
    comptime NQ = Self.D.NQ
    comptime NV = Self.D.NV

    comptime L_Q0 = RowMajor[Self.BATCH, Self.NQ]
    comptime L_NV = RowMajor[Self.BATCH, Self.NV]

    var q0: TensorImpl[Self.DTYPE]  # [BATCH, NQ]
    var v0: TensorImpl[Self.DTYPE]  # [BATCH, NV]
    var A0: TensorImpl[Self.DTYPE]  # [BATCH, NV] (also v_combined in combine)
    var A1: TensorImpl[Self.DTYPE]  # [BATCH, NV]
    var A2: TensorImpl[Self.DTYPE]  # [BATCH, NV]
    var C1: TensorImpl[Self.DTYPE]  # [BATCH, NV]
    var C2: TensorImpl[Self.DTYPE]  # [BATCH, NV]

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
        self.q0 = TensorImpl[Self.DTYPE].alloc(B * dims.get_nq())
        self.v0 = TensorImpl[Self.DTYPE].alloc(B * dims.get_nv())
        self.A0 = TensorImpl[Self.DTYPE].alloc(B * dims.get_nv())
        self.A1 = TensorImpl[Self.DTYPE].alloc(B * dims.get_nv())
        self.A2 = TensorImpl[Self.DTYPE].alloc(B * dims.get_nv())
        self.C1 = TensorImpl[Self.DTYPE].alloc(B * dims.get_nv())
        self.C2 = TensorImpl[Self.DTYPE].alloc(B * dims.get_nv())

    def upload_all(mut self, ctx: DeviceContext) raises:
        """Create device buffers for every scratch tensor (once, at setup —
        contents are produced on-device thereafter)."""
        self.q0.upload(ctx)
        self.v0.upload(ctx)
        self.A0.upload(ctx)
        self.A1.upload(ctx)
        self.A2.upload(ctx)
        self.C1.upload(ctx)
        self.C2.upload(ctx)

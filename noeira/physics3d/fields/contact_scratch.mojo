"""Owned PGS contact-solver workspace tensor (migration P4).

`ContactScratch` is the stateful replacement for the "solver workspace"
section of the flat workspace slab (`ws_solver_offset` in
gpu/constants.mojo): one owned `TensorImpl` holding the per-env PGS contact
workspace, allocated once and reused every step. All offsets in the
consumers (constraints/contact_solve.mojo) are relative to the row
start — the legacy `solver_idx` base is gone.
"""

from ..constraints.solver_ws import ws_budget, _max_one_rt

from max.gpu.host import DeviceContext

from noeira.nn.core.tensor import TensorImpl

from .dims import DimsLike
from noeira.core.tiles import RowMajor


struct ContactScratch[
    DTYPE: DType,
    D: DimsLike,
    BATCH: Int = 1,
    # Per-env scalars for the blocked-Newton constraint Jacobian spill. 0 (the
    # default) means "Je fits threadgroup memory" and NO buffer is allocated —
    # see `solver/je_budget.mojo`, which is the ONLY place this number should
    # be computed. Callers pass `je_ws_size[...]()`; nothing else.
    JE_WS: Int = 0,
](Movable):
    """PGS contact-solver workspace: one owned tensor, `[BATCH, SOLVER_WS]`
    with `SOLVER_WS = 81*MC + 12*MC*NV` (the legacy
    `PGSSolver.solver_workspace_size`).

    Row layout (offsets relative to the row start; `MC = max(MAX_CONTACTS, 1)`):

    Common normal block (15*MC + 2*MC*NV — constraint_builder_gpu.mojo header):
      [0*MC..1*MC)                  lambda_n   Normal impulses
      [1*MC..2*MC)                  K_n        Effective mass
      [2*MC..3*MC)                  c_dist     Contact distance (- includemargin)
      [3*MC..4*MC)                  c_body     Body A index
      [4*MC..5*MC)                  c_body_b   Body B index
      [5*MC..8*MC)                  c_px/py/pz Contact position
      [8*MC..11*MC)                 c_nx/ny/nz Contact normal
      [11*MC..12*MC)                pos_bias   Impedance position correction
      [12*MC..13*MC)                inv_K_imp  imp/K ratio
      [13*MC..14*MC)                imp_n      Normal impedance (direct R_n)
      [14*MC..15*MC)                diag_n     Body invweight0 diagonal
      [15*MC..15*MC+MC*NV)          J_n        Normal Jacobian
      [15*MC+MC*NV..15*MC+2*MC*NV)  MinvJn     M_inv @ J_n^T

    Friction block at FWS = 15*MC + 2*MC*NV (66*MC + 10*MC*NV — the comptime
    ws_* offsets of the legacy PGSSolver.solve_gpu):
      [FWS+0*MC..FWS+5*MC)          lambda_f[5*MC]
      [FWS+5*MC..FWS+10*MC)         K_f[5*MC]
      [FWS+10*MC..FWS+25*MC)        dir_f[15*MC]
      [FWS+25*MC..FWS+30*MC)        fric_coef[5*MC]
      [FWS+30*MC..FWS+31*MC)        condim[MC]
      [FWS+31*MC..FWS+36*MC)        R_f[5*MC]
      [FWS+36*MC..FWS+41*MC)        bias_f[5*MC]
      [FWS+41*MC..+5*MC*NV)         J_f[5*MC*NV]
      [..+5*MC*NV)                  MinvJ_f[5*MC*NV]
      Pyramidal extras (5*MC each): lambda_edge_neg, C_nt, K_edge_pos,
      K_edge_neg, R_edge

    Total = 15*MC + 2*MC*NV + 66*MC + 10*MC*NV = 81*MC + 12*MC*NV.
    """

    # Body unchanged — see `Rk4Scratch` for why the aliases live here.
    comptime NV = Self.D.NV
    comptime MAX_CONTACTS = Self.D.MAX_CONTACTS

    comptime MC = Self.MAX_CONTACTS if Self.MAX_CONTACTS > 0 else 1
    comptime SOLVER_WS = 81 * Self.MC + 12 * Self.MC * Self.NV
    comptime L_SOLVER = RowMajor[Self.BATCH, Self.SOLVER_WS]

    # ⚠ `Je` GETS ITS OWN TENSOR RATHER THAN WIDENING `solver`. `SOLVER_WS` is
    # the ROW STRIDE of a `[BATCH, SOLVER_WS]` view, and that literal is
    # recomputed independently in five solver files. Growing the tensor without
    # growing every one of those views would leave each row after 0 reading the
    # wrong memory — silent corruption, not a crash. A separate tensor leaves
    # every stride untouched and confines the change to the blocked path.
    comptime JE_ELEMS = Self.JE_WS if Self.JE_WS > 0 else 1
    comptime L_JE = RowMajor[Self.BATCH, Self.JE_ELEMS]

    var solver: TensorImpl[Self.DTYPE]  # [BATCH, SOLVER_WS]
    var je: TensorImpl[Self.DTYPE]  # [BATCH, JE_ELEMS] — 1 scalar when unused

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
        # ⚠⚠ THE RUNTIME BUDGET, NOT `Self.SOLVER_WS`. This line read the
        # comptime member until 2026-08-18 — in direct contradiction of the
        # docstring three lines above it, which is the reason it survived 3b's
        # sweep unnoticed. On a dynamic provider `D.NV` is DIM_POISON and
        # `MC` floors to 1, so `SOLVER_WS` is `81 - 12 = 69` scalars for EVERY
        # model, while `contact_solve` indexes the tensor at offsets derived
        # from the RUNTIME nv/mc (walker2d at mc=64 needs 12096). The result is
        # a heap overflow on the first solve of any runtime-loaded model —
        # which crashed at free, not at the write, and so read as an allocator
        # bug. `ws_budget` is the same formula the five solver views use.
        self.solver = TensorImpl[Self.DTYPE].alloc(
            Self.BATCH
            * ws_budget(_max_one_rt(dims.get_max_contacts()), dims.get_nv())
        )
        # ⚠ FLOORED AT 1: a zero-extent tensor operand SEGFAULTS rather than
        # behaving as an empty allocation, so the non-spilling case pays one
        # scalar per env instead of nothing.
        self.je = TensorImpl[Self.DTYPE].alloc(Self.BATCH * Self.JE_ELEMS)

    def upload_all(mut self, ctx: DeviceContext) raises:
        """Create the device buffers (once, at setup — contents are produced
        on-device thereafter)."""
        self.solver.upload(ctx)
        self.je.upload(ctx)

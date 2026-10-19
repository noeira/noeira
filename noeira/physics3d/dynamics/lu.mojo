"""Dense LU factorization + solve over per-field tensors (migration P2 /
Stage-I, single-source). Per-field ports of `lu_factor_gpu`,
`lu_solve_workspace_gpu`, `compute_M_inv_from_lu_gpu`
(dynamics/lu_factorization.mojo) — arithmetic verbatim.

Used by the fields `ImplicitIntegrator`, where
`M_hat = M + armature - dt*qDeriv` is NON-symmetric (the dense RNE velocity
derivative) so LDL cannot be used. Reuses the same `DynamicsScratch` tensors
as the LDL path — `factor` reads `scratch.M`, writes the LU factors into
`scratch.L` and the pivot indices (as floats) into `scratch.D`; `solve`
reads `scratch.L`/`scratch.D`/`scratch.fnet`, writes `scratch.qacc_ws`.
Partial pivoting; near-zero pivots regularized to 1e-30 (verbatim legacy).

Serial per-env kernels only — LU is inherently sequential in the elimination
index k (like LDL's columns). A cooperative `_mt` schedule is a later NVIDIA
perf lever, not a correctness requirement.
"""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout
from noeira.core.tiles import RowMajor, Tile

from ..fields import (
    DynamicsScratch,
    Dims,
    DimsLike,
    AsStatic,
    Scratch,
    cap,
    DYN2,
    rl2,
)

comptime LU_TPB: Int = 64


@always_inline
def _lu_factor_env[
    DTYPE: DType,
    DIMS: DimsLike,
    L_M: TensorLayout,
    L_D: TensorLayout,
](
    env: Int,
    dims: DIMS,
    M: TileTensor[DTYPE, L_M, MutAnyOrigin],
    L: TileTensor[DTYPE, L_M, MutAnyOrigin],
    D: TileTensor[DTYPE, L_D, MutAnyOrigin],
):
    """In-place LU factorization with partial pivoting for one env (verbatim
    from `lu_factor_gpu`). Copies M into the L slot and factorizes there;
    pivot indices (as floats) land in the D slot."""
    var nv = dims.get_nv()
    # Copy M -> L (LU overwrites in-place in the L slot)
    for i in range(nv * nv):
        L[env, i] = M[env, i]

    # Initialize pivots to identity
    for i in range(nv):
        D[env, i] = Scalar[DTYPE](i)

    for k in range(nv):
        # Find pivot: row with largest |value| in column k, rows k..nv-1
        var max_val = abs(rebind[Scalar[DTYPE]](L[env, k * nv + k]))
        var max_row = k
        for i in range(k + 1, nv):
            var val = abs(rebind[Scalar[DTYPE]](L[env, i * nv + k]))
            if val > max_val:
                max_val = val
                max_row = i

        # Swap rows k and max_row
        if max_row != k:
            D[env, k] = Scalar[DTYPE](max_row)
            for j in range(nv):
                var tmp = L[env, k * nv + j]
                L[env, k * nv + j] = L[env, max_row * nv + j]
                L[env, max_row * nv + j] = tmp

        # Regularize near-zero pivot
        var pivot = rebind[Scalar[DTYPE]](L[env, k * nv + k])
        if abs(pivot) < Scalar[DTYPE](1e-30):
            L[env, k * nv + k] = Scalar[DTYPE](1e-30)
            pivot = Scalar[DTYPE](1e-30)

        # Compute multipliers + update trailing submatrix
        var inv_pivot = Scalar[DTYPE](1) / pivot
        for i in range(k + 1, nv):
            var lik = (
                rebind[Scalar[DTYPE]](L[env, i * nv + k]) * inv_pivot
            )
            L[env, i * nv + k] = lik
            for j in range(k + 1, nv):
                var cur = rebind[Scalar[DTYPE]](L[env, i * nv + j])
                var ukj = rebind[Scalar[DTYPE]](L[env, k * nv + j])
                L[env, i * nv + j] = cur - lik * ukj


@always_inline
def _lu_solve_env[
    DTYPE: DType,
    DIMS: DimsLike,
    L_L: TensorLayout,
    L_D: TensorLayout,
](
    env: Int,
    dims: DIMS,
    L: TileTensor[DTYPE, L_L, MutAnyOrigin],
    D: TileTensor[DTYPE, L_D, MutAnyOrigin],
    b: TileTensor[DTYPE, L_D, MutAnyOrigin],
    x: TileTensor[DTYPE, L_D, MutAnyOrigin],
):
    """Solve A*x = b using the LU factors + pivots for one env (verbatim from
    `lu_solve_workspace_gpu`). L holds the in-place LU, D holds the pivots."""
    var nv = dims.get_nv()
    comptime V_CAP = cap[DIMS.NV]()

    # Copy b, then apply the row permutation from the pivots
    var y = Scratch[L.ElementType, V_CAP](nv, uninitialized=0)
    for i in range(nv):
        y[i] = b[env, i]
    for i in range(nv):
        var piv_i = Int(rebind[Scalar[DTYPE]](D[env, i]))
        if piv_i != i:
            var tmp = y[i]
            y[i] = y[piv_i]
            y[piv_i] = tmp

    # Forward substitution: L * y = Pb (L has unit diagonal)
    for i in range(nv):
        for j in range(i):
            y[i] = y[i] - L[env, i * nv + j] * y[j]

    # Backward substitution: U * x = y
    for i in range(nv - 1, -1, -1):
        var s = y[i]
        for j in range(i + 1, nv):
            s = s - L[env, i * nv + j] * x[env, j]
        var diag = rebind[Scalar[DTYPE]](L[env, i * nv + i])
        if abs(diag) > Scalar[DTYPE](1e-30):
            x[env, i] = s / diag
        else:
            x[env, i] = Scalar[DTYPE](0)


@always_inline
def _lu_m_inv_col_env[
    DTYPE: DType,
    DIMS: DimsLike,
    L_L: TensorLayout,
    L_D: TensorLayout,
](
    env: Int,
    j_col: Int,
    dims: DIMS,
    L: TileTensor[DTYPE, L_L, MutAnyOrigin],
    D: TileTensor[DTYPE, L_D, MutAnyOrigin],
    m_inv: TileTensor[DTYPE, L_L, MutAnyOrigin],
):
    """One column of M^-1 from LU factors (solve A*col = e_j). Verbatim from
    `compute_M_inv_from_lu_gpu`'s column body."""
    var nv = dims.get_nv()
    comptime V_CAP = cap[DIMS.NV]()
    var e = Scratch[L.ElementType, V_CAP](nv, uninitialized=0)
    var col = Scratch[L.ElementType, V_CAP](nv, uninitialized=0)

    for i in range(nv):
        e[i] = 0
    e[j_col] = 1

    # Apply permutation
    for i in range(nv):
        var piv_i = Int(rebind[Scalar[DTYPE]](D[env, i]))
        if piv_i != i:
            var tmp = e[i]
            e[i] = e[piv_i]
            e[piv_i] = tmp

    # Forward substitution: L * y = Pe (L has unit diagonal)
    var y = Scratch[L.ElementType, V_CAP](nv, uninitialized=0)
    for i in range(nv):
        var s = e[i]
        for k in range(i):
            s = s - L[env, i * nv + k] * y[k]
        y[i] = s

    # Backward substitution: U * col = y
    for i in range(nv - 1, -1, -1):
        var s = y[i]
        for k in range(i + 1, nv):
            s = s - L[env, i * nv + k] * col[k]
        var diag = rebind[Scalar[DTYPE]](L[env, i * nv + i])
        if abs(diag) > Scalar[DTYPE](1e-30):
            col[i] = s / diag
        else:
            col[i] = 0

    for i in range(nv):
        m_inv[env, i * nv + j_col] = col[i]


@always_inline
def _lu_m_inv_env[
    DTYPE: DType,
    DIMS: DimsLike,
    L_L: TensorLayout,
    L_D: TensorLayout,
](
    env: Int,
    dims: DIMS,
    L: TileTensor[DTYPE, L_L, MutAnyOrigin],
    D: TileTensor[DTYPE, L_D, MutAnyOrigin],
    m_inv: TileTensor[DTYPE, L_L, MutAnyOrigin],
):
    """Full dense M^-1 via per-column LU solves."""
    var nv = dims.get_nv()
    for j_col in range(nv):
        _lu_m_inv_col_env[DTYPE](env, j_col, dims, L, D, m_inv)


# ── launchable kernels (serial: one thread per env) ───────────────────────
def _lu_factor_fields_kernel[
    DTYPE: DType,
    NV: Int,
    BATCH: Int,
](
    M: Tile[DTYPE, BATCH, NV * NV],
    L: Tile[DTYPE, BATCH, NV * NV],
    D: Tile[DTYPE, BATCH, NV],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _lu_factor_env[DTYPE](env, Dims[nv=NV](), M, L, D)


def _lu_solve_fields_kernel[
    DTYPE: DType,
    NV: Int,
    BATCH: Int,
](
    L: Tile[DTYPE, BATCH, NV * NV],
    D: Tile[DTYPE, BATCH, NV],
    b: Tile[DTYPE, BATCH, NV],
    x: Tile[DTYPE, BATCH, NV],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _lu_solve_env[DTYPE](env, Dims[nv=NV](), L, D, b, x)


def _lu_m_inv_fields_kernel[
    DTYPE: DType,
    NV: Int,
    BATCH: Int,
](
    L: Tile[DTYPE, BATCH, NV * NV],
    D: Tile[DTYPE, BATCH, NV],
    m_inv: Tile[DTYPE, BATCH, NV * NV],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _lu_m_inv_env[DTYPE](env, Dims[nv=NV](), L, D, m_inv)


# ── single-body dispatch wrappers (mirror ldl) ─────────────────────
def lu_factor[target: StaticString, DTYPE: DType, D: DimsLike, BATCH: Int = 1](
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """M -> LU factors in L + pivots in D (owned scratch), both targets."""
    comptime L_M = RowMajor[BATCH, D.NV * D.NV]
    comptime L_NV = RowMajor[BATCH, D.NV]

    comptime if target == "cpu":
        var dm = scratch.dims
        var rl_M = rl2(BATCH, dm.get_nv() * dm.get_nv())
        var rl_NV = rl2(BATCH, dm.get_nv())
        var M_v = scratch.M.tt_dyn["cpu", DYN2](rl_M)
        var L_v = scratch.L.tt_dyn["cpu", DYN2](rl_M)
        var D_v = scratch.D.tt_dyn["cpu", DYN2](rl_NV)
        for e in range(BATCH):
            _lu_factor_env[DTYPE](e, dm, M_v, L_v, D_v)
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + LU_TPB - 1) // LU_TPB
        c.enqueue_function[_lu_factor_fields_kernel[DTYPE, D.NV, BATCH]](
            scratch.M.tt["gpu", BATCH, D.NV * D.NV](),
            scratch.L.tt["gpu", BATCH, D.NV * D.NV](),
            scratch.D.tt["gpu", BATCH, D.NV](),
            grid_dim=(BLOCKS,),
            block_dim=(LU_TPB,),
        )


def lu_solve[target: StaticString, DTYPE: DType, D: DimsLike, BATCH: Int = 1](
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`qacc_ws = A^-1 fnet` via LU factors in L + pivots in D, both targets."""
    comptime L_M = RowMajor[BATCH, D.NV * D.NV]
    comptime L_NV = RowMajor[BATCH, D.NV]

    comptime if target == "cpu":
        var dm = scratch.dims
        var rl_M = rl2(BATCH, dm.get_nv() * dm.get_nv())
        var rl_NV = rl2(BATCH, dm.get_nv())
        var L_v = scratch.L.tt_dyn["cpu", DYN2](rl_M)
        var D_v = scratch.D.tt_dyn["cpu", DYN2](rl_NV)
        var b_v = scratch.fnet.tt_dyn["cpu", DYN2](rl_NV)
        var x_v = scratch.qacc_ws.tt_dyn["cpu", DYN2](rl_NV)
        for e in range(BATCH):
            _lu_solve_env[DTYPE](e, dm, L_v, D_v, b_v, x_v)
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + LU_TPB - 1) // LU_TPB
        c.enqueue_function[_lu_solve_fields_kernel[DTYPE, D.NV, BATCH]](
            scratch.L.tt["gpu", BATCH, D.NV * D.NV](),
            scratch.D.tt["gpu", BATCH, D.NV](),
            scratch.fnet.tt["gpu", BATCH, D.NV](),
            scratch.qacc_ws.tt["gpu", BATCH, D.NV](),
            grid_dim=(BLOCKS,),
            block_dim=(LU_TPB,),
        )


def compute_m_inv_from_lu[
    target: StaticString,
    DTYPE: DType,
    D: DimsLike,
    BATCH: Int = 1,
](
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """LU factors (L) + pivots (D) -> m_inv (owned scratch), both targets."""
    comptime L_M = RowMajor[BATCH, D.NV * D.NV]
    comptime L_NV = RowMajor[BATCH, D.NV]

    comptime if target == "cpu":
        var dm = scratch.dims
        var rl_M = rl2(BATCH, dm.get_nv() * dm.get_nv())
        var rl_NV = rl2(BATCH, dm.get_nv())
        var L_v = scratch.L.tt_dyn["cpu", DYN2](rl_M)
        var D_v = scratch.D.tt_dyn["cpu", DYN2](rl_NV)
        var mi_v = scratch.m_inv.tt_dyn["cpu", DYN2](rl_M)
        for e in range(BATCH):
            _lu_m_inv_env[DTYPE](e, dm, L_v, D_v, mi_v)
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + LU_TPB - 1) // LU_TPB
        c.enqueue_function[_lu_m_inv_fields_kernel[DTYPE, D.NV, BATCH]](
            scratch.L.tt["gpu", BATCH, D.NV * D.NV](),
            scratch.D.tt["gpu", BATCH, D.NV](),
            scratch.m_inv.tt["gpu", BATCH, D.NV * D.NV](),
            grid_dim=(BLOCKS,),
            block_dim=(LU_TPB,),
        )

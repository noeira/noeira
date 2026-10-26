"""`qacc_warmstart = qacc` — the tail of `mj_forward` (engine_forward.c:1087).

MuJoCo's constraint solve does not start from `qacc_smooth`; `warmstart()`
(engine_forward.c:786) starts it at whichever of `qacc_warmstart` and
`qacc_smooth` has the LOWER primal cost, and `mj_forward` ends by saving the
answer for the next call. The READ half lives inside each primal solver, where
the constraint rows it has to price are built; this module is the WRITE half.

⚠⚠ IT IS A SEPARATE STAGE AND NOT A LINE IN THE SOLVER'S WRITEBACK because
MuJoCo's copy is UNCONDITIONAL. `mj_fwdConstraint` returns `qacc = qacc_smooth`
when `nefc == 0` and every solver here early-returns in that case without
touching its writeback loop — so a save folded into that loop would leave the
previous step's value standing on exactly the steps that prove nothing is
touching. Reading `scratch.qacc_constrained` covers both: it holds
`qacc_smooth` on the way in and the solved acceleration on the way out.

⚠ THE READ HALF IS NEWTON-ONLY. `cg_solve` still cold-starts (a note at its
init site says so) and the PGS solvers are a DIFFERENT algorithm here, not a
missing one: MuJoCo's `mjSOL_PGS` branch prices the warm start in `efc_force`
space and ZEROES the forces when that cost comes out positive, rather than
choosing between two accelerations. No model in either reference tree selects
PGS or CG. This write runs for all of them regardless, exactly as MuJoCo's
unconditional copy does, so the field is live whichever solver reads it next.

⚠ IT RUNS BEFORE THE INTEGRATOR, NOT AFTER. `mj_forward` saves the CONSTRAINT
SOLVER's answer; `mj_implicit`'s `M_hat` re-solve happens afterwards and
overwrites `d->qacc` without ever reaching `qacc_warmstart`. Our implicit path
writes its re-solve back into `scratch.qacc_constrained`, so calling this after
it would carry the WRONG vector — one produced by a different linear system
than the one the next warm start prices against.
"""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout
from noeira.core.tiles import RowMajor, Tile

from ..fields import Data, DynamicsScratch, DimsLike, DYN2, rl2

comptime WS_TPB: Int = 64


@always_inline
def _save_warmstart_env[
    DTYPE: DType,
    D: DimsLike,
    L_NV: TensorLayout,
](
    env: Int,
    dims: D,
    qacc_constrained: TileTensor[DTYPE, L_NV, MutAnyOrigin],
    qacc_warmstart: TileTensor[DTYPE, L_NV, MutAnyOrigin],
):
    var nv = dims.get_nv()
    for i in range(nv):
        qacc_warmstart[env, i] = rebind[Scalar[DTYPE]](
            qacc_constrained[env, i]
        )


def _save_warmstart_kernel[
    DTYPE: DType, NV: Int, BATCH: Int
](
    qacc_constrained: Tile[DTYPE, BATCH, NV],
    qacc_warmstart: Tile[DTYPE, BATCH, NV],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    for i in range(NV):
        qacc_warmstart[env, i] = rebind[Scalar[DTYPE]](
            qacc_constrained[env, i]
        )


def save_qacc_warmstart[
    target: StaticString,
    DTYPE: DType,
    D: DimsLike,
    BATCH: Int = 1,
](
    mut d: Data[DTYPE, D, BATCH],
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`d.qacc_warmstart <- scratch.qacc_constrained`, every lane."""
    comptime L_NV = RowMajor[BATCH, D.NV]
    comptime if target == "cpu":
        var dm = d.dims
        var rl_NV = rl2(BATCH, dm.get_nv())
        var qc_v = scratch.qacc_constrained.tt_dyn["cpu", DYN2](rl_NV)
        var qw_v = d.qacc_warmstart.tt_dyn["cpu", DYN2](rl_NV)
        for e in range(BATCH):
            _save_warmstart_env[DTYPE](e, dm, qc_v, qw_v)
    else:
        comptime BLOCKS = (BATCH + WS_TPB - 1) // WS_TPB
        ctx.value().enqueue_function[
            _save_warmstart_kernel[DTYPE, D.NV, BATCH]
        ](
            scratch.qacc_constrained.tt["gpu", BATCH, D.NV](),
            d.qacc_warmstart.tt["gpu", BATCH, D.NV](),
            grid_dim=(BLOCKS,),
            block_dim=(WS_TPB,),
        )

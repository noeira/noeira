"""Default observation extraction over per-field tensors (migration P3/P5).

Single-source port of the default physics3d observation — `qpos` (with an
optional leading skip, e.g. hiding rootx) concatenated with `qvel` — which
is what `ModelDefFromXML.extract_obs` produces on CPU and the generic env
obs kernel produces on GPU. One formula body, both targets, reading
`Data.qpos/qvel` and writing a caller-owned obs tensor
`[BATCH, OBS_DIM]` (OBS_DIM = NQ - OBS_QPOS_SKIP + NV).

Env configs with custom obs (xpos/cvel features etc.) get their own fields
extractors as they are ported; this covers the default-obs family
(InvertedPendulum, hopper/walker-style qpos-skip envs)."""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext

from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.fields import Data, Dims, DimsLike
from noeira.core.tiles import RowMajor, Tile

comptime OBS_TPB: Int = 64


@always_inline
def _obs_qpos_qvel_env[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    OBS_QPOS_SKIP: Int,
    BATCH: Int,
](
    env: Int,
    qpos: Tile[DTYPE, BATCH, NQ],
    qvel: Tile[DTYPE, BATCH, NV],
    obs: Tile[DTYPE, BATCH, NQ - OBS_QPOS_SKIP + NV],
):
    comptime NQO = NQ - OBS_QPOS_SKIP
    for i in range(NQO):
        obs[env, i] = qpos[env, OBS_QPOS_SKIP + i]
    for i in range(NV):
        obs[env, NQO + i] = qvel[env, i]


def _obs_qpos_qvel_kernel[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    OBS_QPOS_SKIP: Int,
    BATCH: Int,
](
    qpos: Tile[DTYPE, BATCH, NQ],
    qvel: Tile[DTYPE, BATCH, NV],
    obs: Tile[DTYPE, BATCH, NQ - OBS_QPOS_SKIP + NV],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _obs_qpos_qvel_env[DTYPE, NQ, NV, OBS_QPOS_SKIP, BATCH](
        env, qpos, qvel, obs
    )


def extract_obs_qpos_qvel[
    target: StaticString,
    DTYPE: DType,
    D: DimsLike,
    OBS_QPOS_SKIP: Int = 0,
    BATCH: Int = 1,
](
    mut d: Data[DTYPE, D, BATCH],
    mut obs: TensorImpl[DTYPE],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Default obs (qpos[SKIP:] ‖ qvel) into a caller-owned obs tensor,
    both targets, one body."""
    comptime L_QPOS = RowMajor[BATCH, D.NQ]
    comptime L_NV = RowMajor[BATCH, D.NV]
    comptime L_OBS = RowMajor[BATCH, D.NQ - OBS_QPOS_SKIP + D.NV]

    comptime if target == "cpu":
        var qpos_v = d.qpos.tt["cpu", BATCH, D.NQ]()
        var qvel_v = d.qvel.tt["cpu", BATCH, D.NV]()
        var obs_v = obs.tt["cpu", BATCH, D.NQ - OBS_QPOS_SKIP + D.NV]()
        for e in range(BATCH):
            _obs_qpos_qvel_env[DTYPE, D.NQ, D.NV, OBS_QPOS_SKIP, BATCH](
                e, qpos_v, qvel_v, obs_v
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + OBS_TPB - 1) // OBS_TPB
        c.enqueue_function[
            _obs_qpos_qvel_kernel[DTYPE, D.NQ, D.NV, OBS_QPOS_SKIP, BATCH]
        ](
            d.qpos.tt["gpu", BATCH, D.NQ](),
            d.qvel.tt["gpu", BATCH, D.NV](),
            obs.tt["gpu", BATCH, D.NQ - OBS_QPOS_SKIP + D.NV](),
            grid_dim=(BLOCKS,),
            block_dim=(OBS_TPB,),
        )

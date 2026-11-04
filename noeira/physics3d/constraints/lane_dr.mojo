"""Per-lane domain randomisation of quantities the batch's `Model` shares.

A batched `Data` holds STATE per lane; the `Model` is one record for every
lane, so randomising a model field randomises it for the whole batch. This is
the side channel for the few model quantities a lane may override instead
("batch state, not model", noeira-docs SIM_TO_REAL_PLATFORM_PLAN §3): one
`Data.lane_dr` tensor, `[BATCH, NBODY · LANE_DR_STRIDE]`, read only by the
stages that use it, and only when the integrator / FK is built with
`LANE_DR=True` (every other build compiles exactly as before).

Per body b of lane e, at `(e · NBODY + b) · LANE_DR_STRIDE`:

    LANE_DR_FRICTION   > 0: the sliding friction of every contact this body
                       is in (the contact's mixed friction is replaced);
                       ≤ 0: the model's mix, untouched
    LANE_DR_IPOS_X..Z  an offset added to the body's CoM (`body_ipos`) in FK

FRICTION is applied AFTER contact detection, on the per-lane contact
records, so the narrow phase is untouched. It equals giving the body's geoms
that friction when the body's geoms win MuJoCo's mix (`priority` above the
other geom's; a model that randomises a foot's friction gives its geoms
priority 1). Torsional and rolling friction keep the model's mix.
"""

from std.math import ceildiv
from max.gpu import global_idx
from max.gpu.host import DeviceContext

from ..fields.data import Data
from ..fields.dims import DimsLike
from ..gpu.constants import (
    CONTACT_IDX_BODY_A, CONTACT_IDX_BODY_B, CONTACT_IDX_FRICTION, CONTACT_SIZE, METADATA_SIZE, META_IDX_NUM_CONTACTS,
)


comptime LANE_DR_STRIDE: Int = 4
comptime LANE_DR_FRICTION: Int = 0
comptime LANE_DR_IPOS_X: Int = 1


@always_inline
def _lane_friction_env[DTYPE: DType](
    env: Int, nbody: Int, max_contacts: Int,
    contacts: Pointer[Scalar[DTYPE], MutAnyOrigin], meta: Pointer[Scalar[DTYPE], MutAnyOrigin],
    dr: Pointer[Scalar[DTYPE], MutAnyOrigin],
):
    var ncon = min(Int(meta[unsafe_offset=env * METADATA_SIZE + META_IDX_NUM_CONTACTS]), max_contacts)
    var base = env * max_contacts * CONTACT_SIZE
    for c in range(ncon):
        var o = base + c * CONTACT_SIZE
        var ba = Int(contacts[unsafe_offset=o + CONTACT_IDX_BODY_A])
        var bb = Int(contacts[unsafe_offset=o + CONTACT_IDX_BODY_B])
        var fa = dr[unsafe_offset=(env * nbody + ba) * LANE_DR_STRIDE + LANE_DR_FRICTION] if ba >= 0 and ba < nbody else Scalar[DTYPE](0)
        var fb = dr[unsafe_offset=(env * nbody + bb) * LANE_DR_STRIDE + LANE_DR_FRICTION] if bb >= 0 and bb < nbody else Scalar[DTYPE](0)
        var f = max(fa, fb)
        if f > 0:
            contacts[unsafe_offset=o + CONTACT_IDX_FRICTION] = f


def _k_lane_friction[DTYPE: DType](
    batch: Int64, nbody: Int64, max_contacts: Int64,
    contacts: Pointer[Scalar[DTYPE], MutAnyOrigin], meta: Pointer[Scalar[DTYPE], MutAnyOrigin],
    dr: Pointer[Scalar[DTYPE], MutAnyOrigin],
):
    var env = Int(global_idx.x)
    if env < Int(batch):
        _lane_friction_env[DTYPE](env, Int(nbody), Int(max_contacts), contacts, meta, dr)


def apply_lane_friction[target: StaticString, DTYPE: DType, D: DimsLike, BATCH: Int](
    mut d: Data[DTYPE, D, BATCH], ctx: Optional[DeviceContext]
) raises:
    """Replace each lane's contact friction by its bodies' `LANE_DR_FRICTION`
    overrides (the larger, when both bodies have one). Call between contact
    detection and the constraint solve."""
    var nbody = d.dims.get_nbody()
    var maxc = d.dims.get_max_contacts()
    comptime if target == "gpu":
        ctx.value().enqueue_function[_k_lane_friction[DTYPE]](
            Int64(BATCH), Int64(nbody), Int64(maxc),
            d.contacts.dev.value().unsafe_ptr().as_unsafe_any_origin(),
            d.meta.dev.value().unsafe_ptr().as_unsafe_any_origin(),
            d.lane_dr.dev.value().unsafe_ptr().as_unsafe_any_origin(),
            grid_dim=ceildiv(BATCH, 64), block_dim=64,
        )
    else:
        for env in range(BATCH):
            _lane_friction_env[DTYPE](
                env, nbody, maxc,
                d.contacts.data.unsafe_ptr().as_unsafe_any_origin(),
                d.meta.data.unsafe_ptr().as_unsafe_any_origin(),
                d.lane_dr.data.unsafe_ptr().as_unsafe_any_origin(),
            )

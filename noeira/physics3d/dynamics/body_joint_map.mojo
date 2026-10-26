"""Body → joint map, derived from the joint table once per call.

The body table has no `body_jntadr` / `body_jntnum` (MuJoCo's), so every
routine that wanted "the joints on body `b`" scanned all `njoint` rows per
body — O(nbody · njoint) reads of `JOINT_IDX_BODY_ID` with a float→int
conversion each: three such scans a step on dog (RNE, and twice in the
post-constraint RNE), 62 bodies × 50 joints. Joints are stored in body
order, so a body's joints are contiguous, and this map is one O(njoint)
pass. If they ever were not contiguous the map is rejected (`False`) and
the caller keeps its scanning form, so a model the assumption does not hold
for is slower, not wrong. First written inline in the Newton for the
contact rows (PERFORMANCE.md §13.25), shared from §13.26 on.
"""

from layout import TileTensor, TensorLayout

from ..fields.scratch import Scratch
from ..gpu.constants import JOINT_IDX_BODY_ID


@always_inline
def body_joint_map[
    DTYPE: DType,
    B_CAP: Int,
    L_JOINTS: TensorLayout,
](
    njoint: Int,
    nbody: Int,
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    mut jnt_adr: Scratch[Int, B_CAP],
    mut jnt_num: Scratch[Int, B_CAP],
) -> Bool:
    """`jnt_adr[b]` = first joint on body `b` (-1 if none), `jnt_num[b]` how
    many. Returns False — and the caller must scan — if a body's joints are
    not one contiguous run."""
    for b in range(nbody):
        jnt_adr[b] = -1
        jnt_num[b] = 0
    var ok = nbody > 0
    for j in range(njoint):
        var jb = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID]))
        if jb < 0 or jb >= nbody:
            ok = False
            break
        if jnt_adr[jb] < 0:
            jnt_adr[jb] = j
        elif jnt_adr[jb] + jnt_num[jb] != j:
            ok = False
            break
        jnt_num[jb] = jnt_num[jb] + 1
    return ok

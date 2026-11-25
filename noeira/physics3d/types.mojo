"""Physics3D shared type leaves.

The legacy struct-of-Lists `Model`/`Data` (MuJoCo-style CPU engine state) were
deleted at the G4 fields sunset — the engine state is `fields.Model` /
`fields.Data` (packed per-record tensors). What remains here are the
tiny shared leaves: `_max_one`, the equality-constraint type codes, and
`ConeType`.
"""


from .joint_types import JointDef, JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from .joint_types import get_joint_qpos_size, get_joint_qvel_size

# Actuator dynamics type constants
@always_inline
def _max_one[n: Int]() -> Int:
    """Clamp a comptime size to >= 1 (zero-entity buffers still allocate)."""
    return n if n > 0 else 1


@always_inline
def _lane_rows[n: Int, batch: Int, lanes: Bool]() -> Int:
    """Rows of a model table of `n` rows when it is held once per lane
    (`lanes`), or once for the batch."""
    return n * batch if lanes else n


# Equality constraint types (MuJoCo mjtEq subset)
# ⚠ THESE ARE MuJoCo's `mjtEq` VALUES AND MUST STAY THAT WAY. `EQ_TENDON` was
# 2 until 2026-08-12, which is `mjEQ_JOINT` in MuJoCo — harmless only because
# the constant was declared and never read anywhere (tendon equalities live on
# the TENDON record, flagged by `TENDON_IDX_IS_EQUALITY`, not in the equality
# slab). Renumbered when `mjEQ_JOINT` landed, so the packed `EQ_IDX_TYPE`
# values can be diffed against `m.eq_type` directly.
comptime EQ_CONNECT: Int = 0  # Point-to-point ball joint (3 position rows)
comptime EQ_WELD: Int = 1  # Rigid attachment (3 position + 3 orientation rows)
comptime EQ_JOINT: Int = 2  # Couple two scalar joints with a quartic (1 row)
comptime EQ_TENDON: Int = 3  # Tendon equality (1 bilateral row; on the tendon)

# Equality object semantics (MuJoCo `eq_objtype`, mjOBJ_BODY / mjOBJ_SITE).
# See `EQ_IDX_OBJTYPE` in `gpu/constants.mojo` for why the site form is stored
# reduced to the body form and why the flag still has to be carried.
comptime EQ_OBJ_BODY: Int = 0
comptime EQ_OBJ_SITE: Int = 1


struct ConeType:
    comptime PYRAMIDAL: Int = 0
    comptime ELLIPTIC: Int = 1


struct SolverType:
    """MuJoCo `mjtSolver`, in MuJoCo's own numbering.

    ⚠ THE DEFAULT IS NEWTON (2), NOT PGS. `m.opt.solver` reads 2 on a model
    whose `<option>` says nothing — confirmed on the 3.10.0 runtime — while
    `EulerIntegrator`'s `SOLVER` parameter defaults to `"pgs"`. Every tool that
    does not pass one is therefore running a solver the reference does not use.
    """

    comptime PGS: Int = 0
    comptime CG: Int = 1
    comptime NEWTON: Int = 2


struct IntegratorType:
    """MuJoCo `mjtIntegrator`, in MuJoCo's own numbering."""

    comptime EULER: Int = 0
    comptime RK4: Int = 1
    comptime IMPLICIT: Int = 2
    comptime IMPLICITFAST: Int = 3

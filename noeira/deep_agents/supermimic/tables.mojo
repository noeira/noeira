# +--------------------------------------------------------------------------+ #
# | SuperMimic: a robot's runtime tables, built once from names and records
# +--------------------------------------------------------------------------+ #
"""A robot's runtime tables, built once from its names and model records.

What they hold, for the per-lane functions:

    tree   the FK table (`kinematics.sm_fk_tree`)
    head   the heads' table (`heads.SMHeadLayout[R]`): anchor slot, tracked
           FK rows, q0, soft joint limits (mjlab's: the range shrunk about
           its midpoint by the robot's factor)
    feet   the collision geoms on the foot bodies or their descendants
    com    the body whose CoM offset is randomised; root, its FK row

Both tables are flat `dt` arrays: on the GPU they become two constant buffers
every lane reads.
"""

from noeira.deep_agents.supermimic.heads import SMHeadLayout
from noeira.deep_agents.supermimic.kinematics import FP, SMFKTree, sm_fk_tree
from noeira.deep_agents.supermimic.robot import TrackingRobot, TrackingSim, sm_body_ids
from noeira.physics3d.gpu.constants import (
    BODY_IDX_PARENT, GEOM_IDX_BODY, GEOM_IDX_CONAFFINITY, GEOM_IDX_CONTYPE, JOINT_IDX_RANGE_MAX,
    JOINT_IDX_RANGE_MIN,
)


struct SMTables[dt: DType](Movable):
    var tree: SMFKTree[Self.dt]
    var head: List[Scalar[Self.dt]]
    var feet: List[Int]
    var com_body: Int
    var root_row: Int

    def __init__(out self, var tree: SMFKTree[Self.dt], var head: List[Scalar[Self.dt]], var feet: List[Int], com_body: Int, root_row: Int):
        self.tree = tree^
        self.head = head^
        self.feet = feet^
        self.com_body = com_body
        self.root_row = root_row

    def __init__(out self, *, deinit move: Self):
        self.tree = move.tree^
        self.head = move.head^
        self.feet = move.feet^
        self.com_body = move.com_body
        self.root_row = move.root_row

    def tree_ptr(mut self) -> FP[Self.dt]:
        return self.tree.ptr()

    def head_ptr(mut self) -> FP[Self.dt]:
        return self.head.unsafe_ptr().as_unsafe_any_origin()


def sm_soft_limits[R: TrackingRobot](sim: R.SIM, j: Int) -> Tuple[Float64, Float64]:
    """Hinge `j` (0-based, after the free root): its range shrunk about the
    midpoint by `R.soft_limit_factor()` (mjlab's `soft_joint_pos_limits`)."""
    var lo = sim.joint_field(j + 1, JOINT_IDX_RANGE_MIN)
    var hi = sim.joint_field(j + 1, JOINT_IDX_RANGE_MAX)
    var mid = (lo + hi) / 2.0
    var half = 0.5 * (hi - lo) * R.soft_limit_factor()
    return (mid - half, mid + half)


def sm_tables[dt: DType, R: TrackingRobot](sim: R.SIM, q0: List[Float64]) raises -> SMTables[dt]:
    """The tables for robot R with default joint positions `q0` (n)."""
    comptime L = SMHeadLayout[R]
    if len(q0) != R.NDOF:
        raise Error("sm_tables: q0 has " + String(len(q0)) + " entries, the robot " + String(R.NDOF))
    var tracked_names = R.tracked_bodies()
    if len(tracked_names) != R.N_TRACKED:
        raise Error("sm_tables: " + String(len(tracked_names)) + " tracked bodies, the robot declares " + String(R.N_TRACKED))
    var tracked = sm_body_ids[R](tracked_names)
    var anchor = sm_body_ids[R]([R.anchor_body()])[0]
    var aslot = -1
    for k in range(R.N_TRACKED):
        if tracked[k] == anchor:
            aslot = k
    if aslot < 0:
        raise Error("sm_tables: the anchor '" + R.anchor_body() + "' is not a tracked body")
    var head = List[Scalar[dt]](length=L.SIZE, fill=Scalar[dt](0))
    head[L.ANCHOR] = Scalar[dt](aslot)
    for k in range(R.N_TRACKED):
        head[L.TRACKED + k] = Scalar[dt](tracked[k] - 1)        # FK rows are body id − 1
    for j in range(R.NDOF):
        var lim = sm_soft_limits[R](sim, j)
        head[L.Q0 + j] = Scalar[dt](q0[j])
        head[L.LO + j] = Scalar[dt](lim[0])
        head[L.HI + j] = Scalar[dt](lim[1])
    # the feet: colliding geoms on a foot body or any body below one
    var foot_ids = sm_body_ids[R](R.foot_bodies())
    var feet = List[Int]()
    for g in range(sim.num_geoms()):
        if Int(sim.geom_field(g, GEOM_IDX_CONTYPE)) == 0 and Int(sim.geom_field(g, GEOM_IDX_CONAFFINITY)) == 0:
            continue
        var b = Int(sim.geom_field(g, GEOM_IDX_BODY))
        var on_foot = False
        while b > 0 and not on_foot:
            for f in foot_ids:
                if b == f:
                    on_foot = True
            b = Int(sim.body_field(b, BODY_IDX_PARENT))
        if on_foot:
            feet.append(g)
    if len(feet) == 0:
        raise Error("sm_tables: no collision geom on the foot bodies")
    var com = sm_body_ids[R]([R.com_body()])[0]
    var root = sm_body_ids[R]([R.root_body()])[0]
    return SMTables[dt](sm_fk_tree[dt, R](sim), head^, feet^, com, root - 1)

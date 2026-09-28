"""Putting a sampled episode into `qpos` — the reset adapter.

    var adrs = free_slot_addresses(f, joint_names, joints)
    reset_slots(t, f, placed, adrs, qpos, qvel)

The sampler says WHERE each active slot starts; this says where that goes in
the state vector, and parks everything else.

## ⚠ THE ADDRESSES ARE RESOLVED ONCE, THE WRITES ARE SCALAR

`free_slot_addresses` walks joint names and is host-only. `write_free_pose` is
a handful of stores over plain scalars, so P3's reset kernel calls it per lane
unchanged — the same split as `eval.mojo`'s predicates and `sampler.mojo`'s
draws, and for the same reason: a second implementation on the device leg is
how the two stop agreeing.

## ⚠⚠ qvel IS ZEROED, AND THAT IS NOT TIDINESS

A free body carries six velocity dofs that no `qpos` write touches. Reusing a
`Data` across episodes without zeroing them starts the new episode with the old
one's momentum: a prop placed neatly on the table immediately slides off, and
the trajectory looks like a physics bug rather than a reset bug.

⚠ It is also what makes PARKING an invariant rather than an initial condition.
`docs/TASK_LAYER_IMPLEMENTATION.md` Gap D: gravity is a `Model` field shared by
every lane, so a parked body FALLS. Zeroing its velocity at reset does not stop
that — only the per-step repark does — but it does stop the fall compounding
across episodes.

## ⚠ ONLY FREE SLOTS APPEAR HERE

A `static` slot has no joint and therefore no state: it is composed where it
lives (`spec.SlotSpec`) and cannot be moved at reset. Asking for its address
raises rather than returning something that looks like an address.
"""

from std.math import cos, sin
from .spec import FamilySpec, TaskSpec, SLOT_FREE, SLOT_STATIC
from .sampler import Placement
from .family import park_pos
from .obs import FREE_JOINT_NV
from noeira.physics3d.joint_types import JNT_FREE


struct SlotAddress(Copyable, ImplicitlyCopyable, Movable):
    """Where a free slot's state lives. `-1` for a slot with no free joint."""

    var qadr: Int
    var dadr: Int

    def __init__(out self, qadr: Int, dadr: Int):
        self.qadr = qadr
        self.dadr = dadr


def free_slot_addresses(
    f: FamilySpec,
    joint_names: List[String],
    joint_types: List[Int],
    joint_nq: List[Int],
    joint_nv: List[Int],
) raises -> List[SlotAddress]:
    """Every slot's `(qpos, qvel)` base address, in family slot order.

    ⚠ TAKES THE JOINT TABLE AS FOUR PARALLEL LISTS rather than
    `List[JointData]`, so `tasks/` does not import a physics3d record type
    into its signature. §7's dependency rule is one-way, and a struct in the
    signature is the kind of coupling that makes it two-way later.

    ⚠ `joint_names` IS 1:1 WITH THE JOINT TABLE — unlike `body_names`, whose
    index 0 is the worldbody. Measured on the composed family: 9 joints, 9
    names, free joints at qpos 6 / 13 / 20. Assuming the body convention here
    would shift every address by one joint and still return a plausible
    number.
    """
    var out = List[SlotAddress]()
    for si in range(len(f.slots)):
        ref s = f.slots[si]
        if s.kind != SLOT_FREE:
            out.append(SlotAddress(-1, -1))
            continue
        var want = s.name + "_"
        var qadr = 0
        var dadr = 0
        var found = -1
        for j in range(len(joint_names)):
            if (
                found < 0
                and joint_types[j] == JNT_FREE
                and String(joint_names[j]).startswith(want)
            ):
                found = j
                break
            qadr += joint_nq[j]
            dadr += joint_nv[j]
        if found < 0:
            raise Error(
                "tasks: free slot '" + s.name + "' has no free joint in the"
                " composed scene — no joint named '" + want + "*' of type"
                " FREE. Either the asset declares no <freejoint>, or the scene"
                " is stale (`pixi run gen-family-scenes`)."
            )
        out.append(SlotAddress(qadr, dadr))
    return out^


@always_inline
def write_free_pose(
    mut qpos: List[Float64], qadr: Int,
    x: Float64, y: Float64, z: Float64, yaw: Float64 = 0.0,
):
    """A free joint's seven `qpos`: position, then a quaternion — the IDENTITY
    unless `yaw` (about +z, an init's `:yaw` draw) says otherwise; yaw 0
    writes exactly (1, 0, 0, 0).

    ⚠ SPELLED OUT, NOT A LOOP, because the seven words are not
    interchangeable — three are a position and four are a quaternion whose
    identity is `(1, 0, 0, 0)`. `obs.FREE_JOINT_NQ` is the same seven seen as a
    WIDTH, which is what a reader that only has to skip them needs.

    ⚠ THE QUATERNION IS NOT OPTIONAL AND IT IS NOT ZERO. `(0,0,0,0)` is a
    DEGENERATE rotation — forward kinematics normalises it and gets a
    division by zero or a NaN pose, depending on the path. Writing only the
    three positions and leaving the quaternion at whatever the buffer held is
    the same trap one step removed.
    """
    qpos[qadr + 0] = x
    qpos[qadr + 1] = y
    qpos[qadr + 2] = z
    qpos[qadr + 3] = cos(0.5 * yaw)
    qpos[qadr + 4] = 0.0
    qpos[qadr + 5] = 0.0
    qpos[qadr + 6] = sin(0.5 * yaw)


@always_inline
def write_free_vel_zero(mut qvel: List[Float64], dadr: Int):
    """A free joint's `FREE_JOINT_NV` `qvel`. See the module header for why
    this matters.

    ⚠ THE WIDTH COMES FROM `obs.mojo`, NOT FROM A LITERAL HERE. The
    observation zeroes the SAME two spans for an inactive slot
    (`obs.write_free_slot_obs`), so a free joint's velocity width is written in
    one place and read in two.
    """
    for k in range(FREE_JOINT_NV):
        qvel[dadr + k] = 0.0


def joint_init_addresses(
    t: TaskSpec,
    joint_names: List[String],
    joint_nq: List[Int],
) raises -> List[Int]:
    """Each `jinit=`'s qpos address, in task order. RAISES on a joint the scene
    does not have, or one that is not single-DOF.

    ⚠⚠ THE REFUSAL IS THE POINT, AND IT CANNOT LIVE IN `spec.mojo`.
    `validate_task_against_family` sees the FAMILY, which names slots and
    regions and no joints at all — joint names come from the composed SCENE. So
    a mistyped `jinit=` would otherwise be silently skipped and the episode
    would start with a shut drawer holding a bowl, which MuJoCo resolves by
    ejecting it one step in.

    ⚠ SINGLE-DOF ONLY. A `jinit=` writes ONE number; a ball or free joint takes
    four or seven, and writing one would leave the rest at whatever the last
    episode ended with — a partially-restored pose that is not any pose.
    """
    var adr = List[Int]()
    for k in range(len(t.joint_inits)):
        ref j = t.joint_inits[k]
        var qa = 0
        var found = -1
        for i in range(len(joint_names)):
            if joint_names[i] == j.joint:
                found = i
                break
            qa += joint_nq[i]
        if found < 0:
            raise Error(
                "tasks: jinit names joint '" + j.joint + "', which the composed"
                " scene does not have. The importer resolved it from the"
                " asset's own `<site>` through categories.kv, so either the"
                " family's slot is named differently or the asset changed."
            )
        if joint_nq[found] != 1:
            raise Error(
                "tasks: jinit names joint '" + j.joint + "', which has "
                + String(joint_nq[found]) + " qpos words. A jinit writes ONE"
                " number; the rest would keep the previous episode's values."
            )
        adr.append(qa)
    return adr^


def apply_joint_inits(
    t: TaskSpec,
    adr: List[Int],
    values: List[Float64],
    mut qpos: List[Float64],
    mut qvel: List[Float64],
    dadr: List[Int],
) raises:
    """Write each drawn joint value, and ZERO its velocity.

    ⚠ THE VELOCITY TOO. `set_joint_qpos` alone leaves whatever the previous
    episode's drawer was doing in `qvel`, so an episode would start with a
    drawer already sliding — reproducible only until the episode order changes.
    """
    if len(adr) != len(t.joint_inits) or len(values) != len(t.joint_inits):
        raise Error(
            "tasks: apply_joint_inits got " + String(len(adr)) + " addresses"
            " and " + String(len(values)) + " values for "
            + String(len(t.joint_inits)) + " jinit lines"
        )
    if len(dadr) != len(t.joint_inits):
        raise Error(
            "tasks: apply_joint_inits needs one qvel address per jinit"
        )
    for k in range(len(t.joint_inits)):
        qpos[adr[k]] = values[k]
        qvel[dadr[k]] = 0.0


def joint_init_dof_addresses(
    t: TaskSpec,
    joint_names: List[String],
    joint_nv: List[Int],
) raises -> List[Int]:
    """Each `jinit=`'s qvel address, in task order. The `qvel` twin of
    `joint_init_addresses`; separate because `nq` and `nv` differ per joint
    type and a single walk would have to carry both."""
    var adr = List[Int]()
    for k in range(len(t.joint_inits)):
        ref j = t.joint_inits[k]
        var da = 0
        var found = -1
        for i in range(len(joint_names)):
            if joint_names[i] == j.joint:
                found = i
                break
            da += joint_nv[i]
        if found < 0:
            raise Error(
                "tasks: jinit names joint '" + j.joint + "', not in the scene"
            )
        adr.append(da)
    return adr^


def reset_slots(
    t: TaskSpec,
    f: FamilySpec,
    placed: List[Placement],
    addrs: List[SlotAddress],
    mut qpos: List[Float64],
    mut qvel: List[Float64],
) raises:
    """Write this episode: active slots where the sampler put them, every
    other free slot parked. Velocities zeroed either way.

    ⚠ EVERY FREE SLOT IS WRITTEN, not just the active ones. The fixed scene
    budget means an inactive slot still EXISTS and still has state; leaving it
    at whatever the previous episode ended with is how a "parked" object turns
    up in the middle of the table two episodes later.
    """
    for si in range(len(f.slots)):
        if f.slots[si].kind != SLOT_FREE:
            continue
        ref a = addrs[si]
        if a.qadr < 0:
            raise Error(
                "tasks: free slot '" + f.slots[si].name + "' has no address —"
                " `free_slot_addresses` was built from a different family."
            )

        var active = False
        for p in range(len(placed)):
            if placed[p].slot == si:
                active = True
                write_free_pose(
                    qpos, a.qadr, placed[p].x, placed[p].y, placed[p].z,
                    placed[p].yaw,
                )
                break
        if not active:
            var pk = park_pos(f, si)
            write_free_pose(qpos, a.qadr, pk[0], pk[1], pk[2])
        write_free_vel_zero(qvel, a.dadr)
    _ = t

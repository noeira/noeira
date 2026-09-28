"""The host half of the device reset — refuse, write the joint words, diff.

    require_device_placement[LiberoGoalPlacement](t, f)       # per task, host
    var jw = joint_init_words[LiberoGoalPlacement](t)          # per episode
    var drift = placement_table_drift[LiberoGoalPlacement](    # at startup
        f, facts, radii)

`table.reset_task_slots` cannot raise, so everything the host raises on at the
SPEC level has to be refused before the words are written, and every number the
table restates has to be compared against what the host would use.
"""

from noeira.tasks.spec import (
    FamilySpec, TaskSpec, SLOT_FREE, INIT_TARGET_SLOT,
)
from noeira.tasks.reset import SlotAddress
from noeira.tasks.family import park_pos
from noeira.tasks.sampler import RegionFrame
from noeira.tasks.placement.table import PlacementTable
from noeira.physics3d.gpu.constants import META_JINIT_SLOTS, META_JINIT_WORDS


comptime D64 = DType.float64


struct SceneFacts(Copyable, Movable):
    """What the host knows about a composed scene that a table restates.

    ⚠ A CONTAINER, NOT A SOURCE. The generator fills one from the parsed scene
    and FK; `tests/tasks/test_device_placement.mojo` fills one INDEPENDENTLY
    (joint nudges and FK, no body-chain walk) — the point of the drift check is
    that the two were derived differently.
    """

    var addrs: List[SlotAddress]
    var frames: List[RegionFrame]
    var move_joint: List[Int]
    """Per region: -1 static, k a table joint (a slide), -2 inexpressible."""
    var move_axis: List[Float64]
    """Per region, 3 words: the world axis of `move_joint`, else zeros."""
    var joint_names: List[String]
    var joint_qadr: List[Int]
    var joint_dadr: List[Int]
    var nq: Int
    var nv: Int
    var nbody: Int
    var nsite: Int
    var gripper_site: Int
    var region_site: List[Int]
    """Per region, its site id in the composed scene."""

    def __init__(out self):
        self.addrs = List[SlotAddress]()
        self.frames = List[RegionFrame]()
        self.move_joint = List[Int]()
        self.move_axis = List[Float64]()
        self.joint_names = List[String]()
        self.joint_qadr = List[Int]()
        self.joint_dadr = List[Int]()
        self.nq = 0
        self.nv = 0
        self.nbody = 0
        self.nsite = 0
        self.gripper_site = -1
        self.region_site = List[Int]()


def joint_init_words[T: PlacementTable](t: TaskSpec) raises -> List[Float64]:
    """The task's `jinit=` lines as `META_JINIT_SLOTS * META_JINIT_WORDS` words.

    ⚠ ALWAYS THE WHOLE BLOCK, zeros for the draws the task does not make.
    `meta` survives a reset, so a driver that wrote only this task's words would
    leave the PREVIOUS task's second draw in place and open a drawer this task
    never mentions. RAISES on a joint the table does not list (a robot joint, a
    typo, a stale table) and on more draws than the block holds.
    """
    if len(t.joint_inits) > META_JINIT_SLOTS:
        raise Error(
            "task '" + t.name + "': " + String(len(t.joint_inits)) + " jinit="
            " lines and `meta` holds " + String(META_JINIT_SLOTS) + " draws."
            " Widen META_JINIT_SLOTS (appended at the end of meta)."
        )
    var out = List[Float64](length=META_JINIT_SLOTS * META_JINIT_WORDS, fill=0.0)
    for k in range(len(t.joint_inits)):
        ref j = t.joint_inits[k]
        var found = -1
        for q in range(T.N_JOINTS):
            if T.joint_name(q) == j.joint:
                found = q
        if found < 0:
            raise Error(
                "task '" + t.name + "': jinit names joint '" + j.joint + "',"
                " which the placement table does not list. It lists every"
                " hinge/slide of a STATIC slot; a robot joint is not drawn at"
                " reset, and a missing fixture joint means the table is stale"
                " (`pixi run gen-placement-tables`)."
            )
        out[k * META_JINIT_WORDS] = Float64(found + 1)
        out[k * META_JINIT_WORDS + 1] = j.lo
        out[k * META_JINIT_WORDS + 2] = j.hi
    return out^


def require_device_placement[T: PlacementTable](
    t: TaskSpec, f: FamilySpec
) raises:
    """⚠⚠ REFUSE A TASK THE DEVICE CANNOT RESET THE WAY THE HOST DOES.

    Replaces `gpu_eval.require_gpu_placement`, which refused every STACK
    because the device walked the free-slot table; the kernel now walks
    `spec.order_inits` and draws `jinit=`, so what is left to refuse is:

    * **a region the kernel cannot follow** — `region_move_joint == -2`: its
      site hangs under a hinge, two joints, or a free body. A site under ONE
      slide (a drawer interior) is followed.
    * **a `jinit=` the table does not list, or too many** — `joint_init_words`.
    * **an `On` an anchored region whose fixture has no `slot_geom=`** — the
      host raises, having no `top_site` to add.
    * **a stack where either slot has no `slot_geom=`** — the host raises.
    * **a table that is not this family's** — counts and free-slot indices.

    ⚠ IT DOES NOT CHECK THE NUMBERS. That is `placement_table_drift`, which
    needs the composed scene; this needs only the spec, so it can run per task.
    """
    var n_free = 0
    for i in range(len(f.slots)):
        if f.slots[i].kind == SLOT_FREE:
            if n_free < T.N_FREE and T.free_slot(n_free) != i:
                raise Error(
                    "device placement: table's free slot " + String(n_free)
                    + " is family slot " + String(T.free_slot(n_free))
                    + " but family '" + f.name + "' has it at " + String(i)
                    + " — the table is not this family's, or is stale."
                )
            n_free += 1
    if (
        T.N_SLOTS != len(f.slots) or T.N_FREE != n_free
        or T.N_REGIONS != len(f.regions)
    ):
        raise Error(
            "device placement: table has " + String(T.N_SLOTS) + " slots / "
            + String(T.N_FREE) + " free / " + String(T.N_REGIONS)
            + " regions, family '" + f.name + "' has " + String(len(f.slots))
            + " / " + String(n_free) + " / " + String(len(f.regions))
            + ". Regenerate: `pixi run gen-placement-tables`."
        )
    _ = joint_init_words[T](t)
    for i in range(len(t.inits)):
        ref it = t.inits[i]
        var si = f.slot_index(it.slot)
        if si < 0 or f.slots[si].kind != SLOT_FREE:
            raise Error(
                "task '" + t.name + "': init '" + it.describe() + "' places a"
                " slot that is not FREE; the device init words cover free"
                " slots only."
            )
        var j = _ordinal[T](si)
        if f.init_target_kind(it.region) == INIT_TARGET_SLOT:
            var rsi = f.slot_index(it.region)
            if rsi < 0 or f.slots[rsi].kind != SLOT_FREE:
                raise Error(
                    "task '" + t.name + "': init '" + it.describe() + "'"
                    " stacks on a slot that is not FREE. The host sampler"
                    " raises on it (a stack takes a PLACED reference's pose)."
                )
            if not (T.free_has_geom(j) and T.free_has_geom(_ordinal[T](rsi))):
                raise Error(
                    "task '" + t.name + "': init '" + it.describe() + "'"
                    " stacks without slot_geom= on both slots; the host"
                    " sampler raises on it."
                )
            continue
        var r = f.region_index(it.region)
        if T.region_move_joint(r) == -2:
            raise Error(
                "task '" + t.name + "': init '" + it.describe() + "' draws in"
                " region '" + it.region + "', whose site is carried by"
                " something other than one slide joint (a hinge, two joints,"
                " a free body). The host resolves that frame by FK; the device"
                " kernel runs before FK. Run this task on the host path."
            )
        if (
            T.free_has_geom(j) and T.region_anchored(r) and not it.inside
            and not T.region_contact_has_geom(r)
        ):
            raise Error(
                "task '" + t.name + "': init '" + it.describe() + "' is On a"
                " region whose fixture has no slot_geom=; the host sampler"
                " raises on it."
            )


def _ordinal[T: PlacementTable](si: Int) raises -> Int:
    for j in range(T.N_FREE):
        if T.free_slot(j) == si:
            return j
    raise Error(
        "device placement: family slot " + String(si) + " is not in the"
        " table's free slots"
    )


def placement_table_drift[T: PlacementTable](
    f: FamilySpec, facts: SceneFacts, radii: List[Float64]
) raises -> List[String]:
    """Every number `T` restates, against what the HOST would use.

    `radii` is the caller's fallback per family slot. Returns one line per
    disagreement; empty means the table is current.

    ⚠ EXACT EQUALITY, EXCEPT THE MOVE AXES. The generator writes each float as
    Mojo's `String(Float64)` — MEASURED round-trip exact on 4000 values, and the
    compiler's literal conversion exact on 3701 — so a table built from these
    frames matches them to the bit. A move axis is a finite DIFFERENCE of two
    FK frames, and two derivations of it agree to ~1e-15, so it is compared at
    `AXIS_TOL`; whether the SHIFTED frame is right is the parity gate's job.
    """
    comptime AXIS_TOL: Float64 = 1.0e-9
    var out = List[String]()
    if (
        T.NQ != facts.nq or T.NV != facts.nv or T.NBODY != facts.nbody
        or T.NSITE != facts.nsite
    ):
        out.append(
            "NQ/NV/NBODY/NSITE " + String(T.NQ) + "/" + String(T.NV) + "/"
            + String(T.NBODY) + "/" + String(T.NSITE) + " vs scene "
            + String(facts.nq) + "/" + String(facts.nv) + "/"
            + String(facts.nbody) + "/" + String(facts.nsite)
        )
    if T.N_BASE_QPOS != len(f.base_qpos):
        out.append(
            "N_BASE_QPOS " + String(T.N_BASE_QPOS) + " vs the family's "
            + String(len(f.base_qpos))
        )
    else:
        for i in range(T.N_BASE_QPOS):
            if T.base_qpos[D64](i) != f.base_qpos[i]:
                out.append("base_qpos[" + String(i) + "]")
            # a table that forgot the jitter falls back to the trait's 0
            var h = f.base_qpos_jitter[i] if len(f.base_qpos_jitter) > 0 else 0.0
            if T.base_qpos_jitter[D64](i) != h:
                out.append("base_qpos_jitter[" + String(i) + "]")
    if T.GRIPPER_SITE != facts.gripper_site:
        out.append(
            "GRIPPER_SITE " + String(T.GRIPPER_SITE) + " vs scene "
            + String(facts.gripper_site)
        )
    if T.N_SLOTS != len(f.slots) or T.N_REGIONS != len(f.regions):
        out.append("slot/region counts")
        return out^
    var j = 0
    for si in range(len(f.slots)):
        ref s = f.slots[si]
        if s.kind != SLOT_FREE:
            continue
        if j >= T.N_FREE:
            out.append("more free slots than the table's " + String(T.N_FREE))
            return out^
        var who = "free " + String(j) + " (" + s.name + ")"
        if T.free_slot(j) != si:
            out.append(who + ": slot index")
        if (
            T.free_qadr(j) != facts.addrs[si].qadr
            or T.free_dadr(j) != facts.addrs[si].dadr
        ):
            out.append(who + ": qpos/qvel address")
        if T.free_has_geom(j) != s.has_geom:
            out.append(who + ": has_geom")
        var rest = -s.bottom_z if s.has_geom else radii[si]
        var rad = s.h_radius if s.has_geom else radii[si]
        if T.free_rest[D64](j) != rest:
            out.append(
                who + ": rest " + String(T.free_rest[D64](j)) + " vs "
                + String(rest)
            )
        if T.free_radius[D64](j) != rad:
            out.append(
                who + ": radius " + String(T.free_radius[D64](j)) + " vs "
                + String(rad)
            )
        var pp = park_pos(f, si)
        if (
            T.free_park_x[D64](j) != pp[0] or T.free_park_y[D64](j) != pp[1]
            or T.free_park_z[D64](j) != pp[2]
        ):
            out.append(who + ": park pose")
        if s.has_geom and (
            T.free_bottom_z[D64](j) != s.bottom_z
            or T.free_top_z[D64](j) != s.top_z
        ):
            out.append(who + ": bottom_z/top_z")
        j += 1
    if j != T.N_FREE:
        out.append(
            "free slot count " + String(j) + " vs table " + String(T.N_FREE)
        )

    if T.N_JOINTS != len(facts.joint_names):
        out.append(
            "drawable joints: table " + String(T.N_JOINTS) + " vs scene "
            + String(len(facts.joint_names))
        )
    else:
        for k in range(T.N_JOINTS):
            if (
                T.joint_name(k) != facts.joint_names[k]
                or T.joint_qadr(k) != facts.joint_qadr[k]
                or T.joint_dadr(k) != facts.joint_dadr[k]
            ):
                out.append(
                    "joint " + String(k) + " (" + T.joint_name(k) + " @ "
                    + String(T.joint_qadr(k)) + ") vs scene ("
                    + facts.joint_names[k] + " @ "
                    + String(facts.joint_qadr[k]) + ")"
                )

    for r in range(len(f.regions)):
        ref reg = f.regions[r]
        var who = "region " + String(r) + " (" + reg.name + ")"
        if T.region_site(r) != facts.region_site[r]:
            out.append(
                who + ": site id " + String(T.region_site(r)) + " vs scene "
                + String(facts.region_site[r])
            )
        if (
            T.region_site_x[D64](r) != facts.frames[r].x
            or T.region_site_y[D64](r) != facts.frames[r].y
            or T.region_site_z[D64](r) != facts.frames[r].z
        ):
            out.append(
                who + ": site (" + String(T.region_site_x[D64](r)) + ", "
                + String(T.region_site_y[D64](r)) + ", "
                + String(T.region_site_z[D64](r)) + ") vs FK ("
                + String(facts.frames[r].x) + ", " + String(facts.frames[r].y)
                + ", " + String(facts.frames[r].z) + ")"
            )
        if T.region_has_rect(r) != reg.has_rect:
            out.append(who + ": has_rect")
        if reg.has_rect and (
            T.region_x0[D64](r) != reg.x_min or T.region_y0[D64](r) != reg.y_min
            or T.region_x1[D64](r) != reg.x_max
            or T.region_y1[D64](r) != reg.y_max
        ):
            out.append(who + ": rectangle")
        var anchored = reg.contact.byte_length() > 0
        if T.region_anchored(r) != anchored:
            out.append(who + ": anchored")
        if anchored:
            var ci = f.slot_index(reg.contact)
            var cg = ci >= 0 and f.slots[ci].has_geom
            if T.region_contact_has_geom(r) != cg:
                out.append(who + ": contact has_geom")
            if cg and T.region_contact_top_z[D64](r) != f.slots[ci].top_z:
                out.append(who + ": contact top_z")
        if T.region_move_joint(r) != facts.move_joint[r]:
            out.append(
                who + ": carried by " + String(T.region_move_joint(r))
                + " vs scene " + String(facts.move_joint[r])
            )
        elif facts.move_joint[r] >= 0:
            if (
                abs(T.region_move_axis_x[D64](r) - facts.move_axis[r * 3])
                > AXIS_TOL
                or abs(T.region_move_axis_y[D64](r) - facts.move_axis[r * 3 + 1])
                > AXIS_TOL
                or abs(T.region_move_axis_z[D64](r) - facts.move_axis[r * 3 + 2])
                > AXIS_TOL
            ):
                out.append(who + ": move axis")
    return out^

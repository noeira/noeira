"""THE DEVICE SAMPLER AGAINST THE HOST ONE — one distribution, two writers.

    pixi run mojo run -I . tests/tasks/test_device_placement.mojo

## ⚠⚠ WHY THERE ARE TWO AT ALL, AND WHY THAT IS THE RISK

`tasks/sampler.sample_placements` places props on the HOST — it is what the
eval, the viewer and the init table use. `placement/table.place_free_slots`
places them on the DEVICE, per lane, at every reset, from a family's comptime
`PlacementTable`. Two implementations of one distribution is exactly the drift
this file exists to prevent: if they disagree, a policy trains on scenes the
eval never shows it and every other number in the system still agrees.

## ⚠⚠ THIS GATE USED TO RUN ON ONE FAMILY, AND THAT FAMILY WAS BLIND

It ran `so101_tabletop` alone — the one family whose props declare no robosuite
sites, whose regions share one site, and whose tasks never stack. The old
device hook was correct there and wrong for every LIBERO family on four counts
(`slot_geom=`, the `order_inits` walk, `TABLE_Z_OFFSET`, the fixture's
`top_site` for `On`), and this file could not have seen any of them. It now runs
every task of every LIBERO family, and section 6 COUNTS that the corpus reaches
each rule — a parity check over placements that never take a branch says
nothing about that branch.

## `jinit=`, AND THE REGION THAT FOLLOWS IT

`reset_task_slots` draws each `jinit=` BEFORE placing, and a region carried by
one slide (a drawer interior) reads the drawn `qpos`. So the host frames here
are per LANE — FK after that lane's own draws — and the table's carrying joint
and axis are checked against a NUDGE oracle, never against the generator's own
body-chain walk.

## THE SECTIONS

1. `meta`'s appended init and jinit blocks
2. `so101_tabletop`: its hand-written table, then device vs host
3. an untouched `meta` writes nothing
4. every LIBERO family: its generated table vs the family, FK and the oracle
5. every LIBERO task: joint draws and placements, device == host on every lane,
   and nothing else written; refusals exactly where the oracle says
6. the corpus reaches every rule the kernel mirrors (checked after 8)
7. the stack exemption, on a scene built to reach it — the host used to CRASH
7b. `In` a fixture, the clamp and the region exemption, which the corpus lacks
8. the refusals, on corpus tasks changed one way each
"""

from std.math import cos, sin, sqrt
from std.os import listdir

from noeira.envs.robots.so_arm101_xml import SO_ARM101_NMESH_VERTS
from noeira.tasks.spec import (
    FamilySpec, TaskSpec, JointInitSpec, load_family, load_task, parse_family,
    parse_task, parse_init, validate_task_against_family, SLOT_FREE,
    INIT_TARGET_SLOT,
)
from noeira.tasks.family import scene_path, task_path
from noeira.tasks.family_config import (
    So101TabletopConfig, So101TabletopPlacement,
)
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.so101_tower_xml import SO101_TOWER_NMESH_VERTS
from noeira.tasks.active import init_region_words
from noeira.tasks.eval import region_sites
from noeira.tasks.reset import (
    SlotAddress, free_slot_addresses, joint_init_addresses,
    joint_init_dof_addresses,
)
from noeira.tasks.sampler import (
    sample_placements, sample_joint_inits, sample_base_qpos, RegionFrame,
    SampleReport,
)
from noeira.tasks.placement.table import (
    PlacementTable, reset_task_slots,
)
from noeira.tasks.placement.check import (
    require_device_placement, placement_table_drift, joint_init_words,
    SceneFacts,
)
from noeira.envs.libero.placement.libero_goal import LiberoGoalPlacement
from noeira.envs.libero.placement.libero_kitchen_scene1 import (
    LiberoKitchenScene1Placement,
)
from noeira.envs.libero.placement.libero_kitchen_scene2 import (
    LiberoKitchenScene2Placement,
)
from noeira.envs.libero.placement.libero_kitchen_scene3 import (
    LiberoKitchenScene3Placement,
)
from noeira.envs.libero.placement.libero_kitchen_scene4 import (
    LiberoKitchenScene4Placement,
)
from noeira.envs.libero.placement.libero_kitchen_scene5 import (
    LiberoKitchenScene5Placement,
)
from noeira.envs.libero.placement.libero_kitchen_scene6 import (
    LiberoKitchenScene6Placement,
)
from noeira.envs.libero.placement.libero_kitchen_scene7 import (
    LiberoKitchenScene7Placement,
)
from noeira.envs.libero.placement.libero_kitchen_scene8 import (
    LiberoKitchenScene8Placement,
)
from noeira.envs.libero.placement.libero_kitchen_scene9 import (
    LiberoKitchenScene9Placement,
)
from noeira.envs.libero.placement.libero_kitchen_scene10 import (
    LiberoKitchenScene10Placement,
)
from noeira.envs.libero.placement.libero_living_room_scene1 import (
    LiberoLivingRoomScene1Placement,
)
from noeira.envs.libero.placement.libero_living_room_scene2 import (
    LiberoLivingRoomScene2Placement,
)
from noeira.envs.libero.placement.libero_living_room_scene3 import (
    LiberoLivingRoomScene3Placement,
)
from noeira.envs.libero.placement.libero_living_room_scene4 import (
    LiberoLivingRoomScene4Placement,
)
from noeira.envs.libero.placement.libero_living_room_scene5 import (
    LiberoLivingRoomScene5Placement,
)
from noeira.envs.libero.placement.libero_living_room_scene6 import (
    LiberoLivingRoomScene6Placement,
)
from noeira.envs.libero.placement.libero_object import LiberoObjectPlacement
from noeira.envs.libero.placement.libero_spatial import LiberoSpatialPlacement
from noeira.envs.libero.placement.libero_study_scene1 import (
    LiberoStudyScene1Placement,
)
from noeira.envs.libero.placement.libero_study_scene2 import (
    LiberoStudyScene2Placement,
)
from noeira.envs.libero.placement.libero_study_scene3 import (
    LiberoStudyScene3Placement,
)
from noeira.envs.libero.placement.libero_study_scene4 import (
    LiberoStudyScene4Placement,
)
from noeira.physics3d.gpu.constants import (
    META_IDX_NEWTON_ITER, META_SOLVER_WORDS,
    META_IDX_REWARD_MODE, META_REWARD_WORDS,
    METADATA_SIZE, META_IDX_INIT_REGION_0, META_INIT_SLOTS, META_IDX_LS_EVAL,
    META_IDX_JINIT_0, META_JINIT_SLOTS, META_JINIT_WORDS,
    MODEL_JOINT_SIZE, MODEL_BODY_SIZE, MODEL_GEOM_SIZE,
)
from noeira.physics3d.joint_types import JNT_HINGE, JNT_SLIDE
from noeira.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from noeira.physics3d.fields import Data, Model, DynDims
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics

from layout import Layout, LayoutTensor
from noeira.nn.core.tensor import TensorImpl


comptime DT = DType.float64
comptime SO101_FAMILY = "noeira/tasks/families/so101_tabletop.family"
comptime FAMILY_DIR = "noeira/envs/libero/families"
comptime TASK_DIR = "noeira/envs/libero/tasks/"
comptime BATCH = 8
comptime SEED = 7
comptime TOL: Float64 = 1.0e-12
comptime N_LIBERO_FAMILIES = 23
comptime N_LIBERO_TASKS = 129

# A free slot the device declines to place must be LEFT ALONE, so every word it
# may not touch starts at a value no placement writes.
comptime QPOS_SENTINEL: Float64 = 777.0
comptime QVEL_SENTINEL: Float64 = 7.0
comptime META_CANARY: Float64 = 0.25


struct Tally(Copyable, ImplicitlyCopyable, Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def check(mut self, ok: Bool, what: String):
        self.checks += 1
        if ok:
            print("  ok:", what)
        else:
            self.failures += 1
            print("  FAIL:", what)


struct Stats(Copyable, ImplicitlyCopyable, Movable):
    """What the parity loop compared, and which rules it went through."""

    var tasks: Int
    var refused: Int
    var lanes: Int
    var placements: Int
    var coords: Int
    var bad: Int
    var worst: Float64
    var exact: Int
    var left_alone: Int
    var left_alone_bad: Int
    var other_written: Int
    var meta_touched: Int
    var jinit_draws: Int
    var jinit_bad: Int
    var base_words: Int
    var base_bad: Int
    var base_jittered: Int
    """Rest words the family jitters that came out off the rest value — the
    vacuity guard for `base_qpos_jitter=` (0 on every family without it)."""
    var yawed: Int
    """Placements the host drew a nonzero `:yaw` for (the device's quaternion
    is compared to it word for word above)."""
    # rule coverage, per placement the HOST made
    var geom: Int
    var table_off: Int
    var on_fixture: Int
    var in_fixture: Int
    var stacks: Int
    var followed: Int
    var reordered_tasks: Int
    var rejections: Int
    var clamped: Int
    var exempt: Int
    var sep_rejected: Int
    var sep_pairs: Int
    var sep_margin: Float64
    """`:sep=`: host draws only the separation refused, pairs with a
    separation, and the smallest (distance - separation) among them (m)."""

    def __init__(out self):
        self.tasks = 0
        self.refused = 0
        self.lanes = 0
        self.placements = 0
        self.coords = 0
        self.bad = 0
        self.worst = 0.0
        self.exact = 0
        self.left_alone = 0
        self.left_alone_bad = 0
        self.other_written = 0
        self.meta_touched = 0
        self.jinit_draws = 0
        self.jinit_bad = 0
        self.base_words = 0
        self.base_bad = 0
        self.base_jittered = 0
        self.yawed = 0
        self.geom = 0
        self.table_off = 0
        self.on_fixture = 0
        self.in_fixture = 0
        self.stacks = 0
        self.followed = 0
        self.reordered_tasks = 0
        self.rejections = 0
        self.clamped = 0
        self.exempt = 0
        self.sep_rejected = 0
        self.sep_pairs = 0
        self.sep_margin = 1.0e9


struct LaneInputs(Copyable, Movable):
    """What the HOST uses for each lane: FK frames after that lane's joint
    draws, and the draws themselves. Flat: `frames[lane * n_regions + r]`,
    `jvals[lane * n_jinit + k]`."""

    var frames: List[RegionFrame]
    var n_regions: Int
    var jadr: List[Int]
    var jdadr: List[Int]
    var jvals: List[Float64]
    var n_jinit: Int

    def __init__(out self):
        self.frames = List[RegionFrame]()
        self.n_regions = 0
        self.jadr = List[Int]()
        self.jdadr = List[Int]()
        self.jvals = List[Float64]()
        self.n_jinit = 0


def _sorted(var xs: List[String]) -> List[String]:
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if xs[j] < xs[i]:
                xs[i], xs[j] = xs[j], xs[i]
    return xs^


def _libero_families() raises -> List[String]:
    var out = List[String]()
    for e in listdir(FAMILY_DIR):
        var n = String(e)
        if n.startswith("libero") and n.endswith(".family"):
            out.append(String(n[byte = 0 : n.byte_length() - 7]))
    return _sorted(out^)


def _tasks_of(family: String) raises -> List[String]:
    var out = List[String]()
    var want = family + "__"
    for e in listdir(TASK_DIR):
        var n = String(e)
        if n.startswith(want) and n.endswith(".task"):
            out.append(String(n[byte = 0 : n.byte_length() - 5]))
    return _sorted(out^)


def _host_radii(f: FamilySpec, fallback: Float64) -> List[Float64]:
    var radii = List[Float64]()
    for _ in range(len(f.slots)):
        radii.append(fallback)
    return radii^


def _coverage(
    t: TaskSpec, f: FamilySpec, move_joint: List[Int], mut st: Stats
) raises:
    """Which of the kernel's rules THIS task's placements take — per lane, so
    the counts are comparable with `st.placements`."""
    var reordered = False
    for i in range(1, len(t.inits)):
        if f.slot_index(t.inits[i].slot) < f.slot_index(t.inits[i - 1].slot):
            reordered = True
    if reordered:
        st.reordered_tasks += 1
    for i in range(len(t.inits)):
        ref it = t.inits[i]
        var si = f.slot_index(it.slot)
        if f.slots[si].has_geom:
            st.geom += BATCH
        if f.init_target_kind(it.region) == INIT_TARGET_SLOT:
            st.stacks += BATCH
            continue
        var ri = f.region_index(it.region)
        if move_joint[ri] >= 0:
            st.followed += BATCH
        ref reg = f.regions[ri]
        if not f.slots[si].has_geom:
            continue
        if reg.contact.byte_length() == 0:
            st.table_off += BATCH
        elif it.inside:
            st.in_fixture += BATCH
        else:
            st.on_fixture += BATCH


def _parity[T: PlacementTable](
    t: TaskSpec,
    f: FamilySpec,
    li: LaneInputs,
    radii: List[Float64],
    mut st: Stats,
) raises:
    """`reset_task_slots` on BATCH lanes against the host on each lane.

    ⚠ EVERY WORD THE KERNEL MAY NOT TOUCH IS CHECKED UNTOUCHED: `qpos` starts
    at 0 (the host's own pre-FK state for a fixture joint) with each free slot
    at a sentinel, `qvel` at a sentinel. A write anywhere but a placed slot or a
    drawn joint is counted."""
    comptime NQ = T.NQ
    comptime NV = T.NV
    comptime L_Q = Layout.row_major(BATCH, NQ)
    comptime L_V = Layout.row_major(BATCH, NV)
    comptime L_M = Layout.row_major(BATCH, METADATA_SIZE)
    var words = init_region_words(t, f)
    var jwords = joint_init_words[T](t)
    var qs = TensorImpl[DT].alloc(BATCH * NQ)
    var vs = TensorImpl[DT].alloc(BATCH * NV)
    var ms = TensorImpl[DT].alloc(BATCH * METADATA_SIZE)
    for i in range(BATCH * NQ):
        qs.data[i] = Scalar[DT](0)
    for e in range(BATCH):
        for j in range(T.N_FREE):
            for w in range(7):
                qs.data[e * NQ + T.free_qadr(j) + w] = Scalar[DT](QPOS_SENTINEL)
    for i in range(BATCH * NV):
        vs.data[i] = Scalar[DT](QVEL_SENTINEL)
    for e in range(BATCH):
        for k in range(METADATA_SIZE):
            ms.data[e * METADATA_SIZE + k] = Scalar[DT](META_CANARY)
        for j in range(META_INIT_SLOTS):
            ms.data[e * METADATA_SIZE + META_IDX_INIT_REGION_0 + j] = Scalar[
                DT
            ](0)
        for j in range(len(words)):
            ms.data[e * METADATA_SIZE + META_IDX_INIT_REGION_0 + j] = Scalar[
                DT
            ](words[j])
        for j in range(len(jwords)):
            ms.data[e * METADATA_SIZE + META_IDX_JINIT_0 + j] = Scalar[DT](
                jwords[j]
            )
    var meta_before = List[Float64]()
    for i in range(BATCH * METADATA_SIZE):
        meta_before.append(Float64(ms.data[i]))
    var qt = qs.lt["cpu", L_Q]()
    var vt = vs.lt["cpu", L_V]()
    var mt = ms.lt["cpu", L_M]()

    for lane in range(BATCH):
        reset_task_slots[T, DT, BATCH, NQ, NV](qt, vt, mt, lane, SEED)
    for i in range(BATCH * METADATA_SIZE):
        if Float64(ms.data[i]) != meta_before[i]:
            st.meta_touched += 1

    for lane in range(BATCH):
        var qseen = List[Bool](length=NQ, fill=False)
        var vseen = List[Bool](length=NV, fill=False)
        # ── the base asset's rest pose, with the family's per-episode draw ──
        var rest = sample_base_qpos(f, UInt64(SEED), lane)
        for i in range(T.N_BASE_QPOS):
            st.base_words += 1
            var got_q = Float64(qs.data[lane * NQ + i])
            if abs(got_q - rest[i]) > TOL:
                st.base_bad += 1
                print("      ", t.name, "lane", lane, "rest", i, ": device",
                      got_q, "host", rest[i])
            var h = f.base_qpos_jitter[i] if len(f.base_qpos_jitter) > 0 else 0.0
            if abs(got_q - f.base_qpos[i]) > h + TOL:
                st.base_bad += 1
            if h > 0.0 and got_q != f.base_qpos[i]:
                st.base_jittered += 1
            if i < NV and Float64(vs.data[lane * NV + i]) != 0.0:
                st.base_bad += 1
            qseen[i] = True
            if i < NV:
                vseen[i] = True
        # ── the joint draws ──
        for k in range(li.n_jinit):
            var want = li.jvals[lane * li.n_jinit + k]
            var got = Float64(qs.data[lane * NQ + li.jadr[k]])
            st.jinit_draws += 1
            if abs(got - want) > TOL or Float64(
                vs.data[lane * NV + li.jdadr[k]]
            ) != 0.0:
                st.jinit_bad += 1
                print("      ", t.name, "lane", lane, "jinit", k, ": device",
                      got, "host", want)
            qseen[li.jadr[k]] = True
            vseen[li.jdadr[k]] = True
        # ── the placements, on this lane's frames ──
        var frames = List[RegionFrame]()
        for r in range(li.n_regions):
            frames.append(li.frames[lane * li.n_regions + r])
        var rep = SampleReport()
        var placed = sample_placements(
            t, f, frames, radii, UInt64(SEED), lane, rep
        )
        st.lanes += 1
        st.rejections += rep.attempts - rep.accepted
        st.clamped += rep.clamped
        st.exempt += rep.exempt
        st.sep_rejected += rep.sep_rejected
        # `:sep=`: every placed pair where either init carries one keeps it
        for a in range(len(placed)):
            for b in range(a + 1, len(placed)):
                var sa = 0.0
                var sb = 0.0
                for q in range(len(t.inits)):
                    var qs_ = f.slot_index(t.inits[q].slot)
                    if qs_ == placed[a].slot:
                        sa = t.inits[q].sep()
                    if qs_ == placed[b].slot:
                        sb = t.inits[q].sep()
                var need = max(sa, sb)
                if need > 0.0:
                    var dxy = sqrt((placed[a].x - placed[b].x) ** 2
                                   + (placed[a].y - placed[b].y) ** 2)
                    st.sep_pairs += 1
                    st.sep_margin = min(st.sep_margin, dxy - need)
        for j in range(T.N_FREE):
            var si = T.free_slot(j)
            var qa = T.free_qadr(j)
            var da = T.free_dadr(j)
            for w in range(7):
                qseen[qa + w] = True
            for w in range(6):
                vseen[da + w] = True
            var k = -1
            for p in range(len(placed)):
                if placed[p].slot == si:
                    k = p
            var base = lane * NQ + qa
            var vbase = lane * NV + da
            if k < 0:
                st.left_alone += 1
                var ok = True
                for w in range(7):
                    if Float64(qs.data[base + w]) != QPOS_SENTINEL:
                        ok = False
                for w in range(6):
                    if Float64(vs.data[vbase + w]) != QVEL_SENTINEL:
                        ok = False
                if not ok:
                    st.left_alone_bad += 1
                    print("      ", t.name, "lane", lane, "slot", si,
                          ": the host did not place it and the device wrote it")
                continue
            st.placements += 1
            var want = List[Float64]()
            want.append(placed[k].x)
            want.append(placed[k].y)
            want.append(placed[k].z)
            want.append(cos(0.5 * placed[k].yaw))
            want.append(0.0)
            want.append(0.0)
            want.append(sin(0.5 * placed[k].yaw))
            if placed[k].yaw != 0.0:
                st.yawed += 1
            var this_bad = False
            for w in range(7):
                var got = Float64(qs.data[base + w])
                var dd = abs(got - want[w])
                st.coords += 1
                if dd == 0.0:
                    st.exact += 1
                if dd > st.worst:
                    st.worst = dd
                if dd > TOL:
                    st.bad += 1
                    this_bad = True
            for w in range(6):
                if Float64(vs.data[vbase + w]) != 0.0:
                    st.bad += 1
                    this_bad = True
            if this_bad and st.bad <= 20:
                print("      ", t.name, "lane", lane, "slot", si,
                      ": device (", Float64(qs.data[base]), ",",
                      Float64(qs.data[base + 1]), ",",
                      Float64(qs.data[base + 2]), ") host (", placed[k].x,
                      ",", placed[k].y, ",", placed[k].z, ")")
        # ── nothing else ──
        for i in range(NQ):
            if not qseen[i] and Float64(qs.data[lane * NQ + i]) != 0.0:
                st.other_written += 1
        for i in range(NV):
            if not vseen[i] and Float64(vs.data[lane * NV + i]) != QVEL_SENTINEL:
                st.other_written += 1


def _run_family[T: PlacementTable](
    f: FamilySpec,
    tasks: List[String],
    verts0: Int,
    fallback_radius: Float64,
    gripper_name: String,
    mut ta: Tally,
    mut st: Stats,
    mut refused: List[String],
    mut should: List[String],
    yaw_all: Bool = False,
) raises:
    """The table against an INDEPENDENTLY derived `SceneFacts`, then every task:
    refused, or parity on BATCH lanes with the host's per-lane FK frames.
    `yaw_all` turns `:yaw` on for every region init of every task (2c)."""
    var fmd = parse_model_runtime(scene_path(f))
    var verts = verts0
    var dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
    var m = Model[DT, DynDims](dims)
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") < 0:
                raise e
            verts *= 2
            dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var d = Data[DT, DynDims, 1](dims)
    var facts = SceneFacts()
    facts.nq = dims.get_nq()
    facts.nv = dims.get_nv()
    var rsites = region_sites(f, fmd.site_names)
    var nr = len(f.regions)
    facts.nbody = dims.get_nbody()
    facts.nsite = dims.get_nsite()
    facts.region_site = rsites.copy()
    for i in range(len(fmd.site_names)):
        if fmd.site_names[i] == gripper_name:
            facts.gripper_site = i

    var jt = List[Int]()
    var jqn = List[Int]()
    var jvn = List[Int]()
    for i in range(len(fmd.joints)):
        jt.append(fmd.joints[i].jnt_type)
        jqn.append(fmd.joints[i].nq)
        jvn.append(fmd.joints[i].nv)
    facts.addrs = free_slot_addresses(f, fmd.joint_names, jt, jqn, jvn)

    # the host's reset state before FK: zeros, then base_qpos
    for i in range(facts.nq):
        d.qpos.data[i] = Scalar[DT](0)
    for i in range(len(f.base_qpos)):
        d.qpos.data[i] = Scalar[DT](f.base_qpos[i])
    forward_kinematics["cpu", DT, DynDims, 1](d, m)
    var z = List[Float64]()
    for r in range(nr):
        for c in range(3):
            z.append(Float64(d.site_xpos.data[rsites[r] * 3 + c]))
        facts.frames.append(RegionFrame(z[r * 3], z[r * 3 + 1], z[r * 3 + 2]))

    # drawable joints: 1-dof hinge/slide named `<static slot>_...`
    var jidx = List[Int]()
    var qa_run = 0
    var da_run = 0
    for i in range(len(jt)):
        var owned = False
        for si in range(len(f.slots)):
            if f.slots[si].kind != SLOT_FREE and String(
                fmd.joint_names[i]
            ).startswith(f.slots[si].name + "_"):
                owned = True
        if owned and jqn[i] == 1 and (jt[i] == JNT_HINGE or jt[i] == JNT_SLIDE):
            facts.joint_names.append(String(fmd.joint_names[i]))
            facts.joint_qadr.append(qa_run)
            facts.joint_dadr.append(da_run)
            jidx.append(i)
        qa_run += jqn[i]
        da_run += jvn[i]

    # ⚠⚠ THE CARRYING-JOINT ORACLE, BY NUDGES — NOT THE GENERATOR'S BODY WALK.
    # (1) every joint at once: which sites move at all; (2) each drawable joint
    # alone at 0.05 and at 0.10: which sites it moves, and whether the move is
    # LINEAR (a slide) and the WHOLE of the global move (nothing else carries
    # it). One such joint -> it, axis = move / 0.05; moving otherwise -> -2.
    var adr = 0
    for j in range(len(jt)):
        if jqn[j] == 7 or jqn[j] == 1:
            d.qpos.data[adr] = d.qpos.data[adr] + Scalar[DT](0.05)
        adr += jqn[j]
    forward_kinematics["cpu", DT, DynDims, 1](d, m)
    var g = List[Float64]()
    for r in range(nr):
        for c in range(3):
            g.append(Float64(d.site_xpos.data[rsites[r] * 3 + c]) - z[r * 3 + c])
    for i in range(facts.nq):
        d.qpos.data[i] = Scalar[DT](0)
    for i in range(len(f.base_qpos)):
        d.qpos.data[i] = Scalar[DT](f.base_qpos[i])
    var d1 = List[Float64](length=nr * 3 * len(jidx) + 1, fill=0.0)
    var d2 = List[Float64](length=nr * 3 * len(jidx) + 1, fill=0.0)
    for k in range(len(jidx)):
        for step in range(2):
            var qv = 0.05 if step == 0 else 0.10
            d.qpos.data[facts.joint_qadr[k]] = Scalar[DT](qv)
            forward_kinematics["cpu", DT, DynDims, 1](d, m)
            for r in range(nr):
                for c in range(3):
                    var dd = Float64(d.site_xpos.data[rsites[r] * 3 + c]) - z[
                        r * 3 + c
                    ]
                    if step == 0:
                        d1[(k * nr + r) * 3 + c] = dd
                    else:
                        d2[(k * nr + r) * 3 + c] = dd
            d.qpos.data[facts.joint_qadr[k]] = Scalar[DT](0)
    for r in range(nr):
        var moved = g[r * 3] != 0.0 or g[r * 3 + 1] != 0.0 or g[r * 3 + 2] != 0.0
        var carried = -1
        var ax = 0.0
        var ay = 0.0
        var az = 0.0
        if moved:
            carried = -2
            var hits = 0
            var hk = -1
            for k in range(len(jidx)):
                var b = (k * nr + r) * 3
                if d1[b] != 0.0 or d1[b + 1] != 0.0 or d1[b + 2] != 0.0:
                    hits += 1
                    hk = k
            if hits == 1:
                var b = (hk * nr + r) * 3
                var linear = True
                var whole = True
                for c in range(3):
                    if abs(d2[b + c] - 2.0 * d1[b + c]) > 1.0e-12:
                        linear = False
                    if abs(g[r * 3 + c] - d1[b + c]) > 1.0e-12:
                        whole = False
                if linear and whole:
                    carried = hk
                    ax = d1[b] / 0.05
                    ay = d1[b + 1] / 0.05
                    az = d1[b + 2] / 0.05
        facts.move_joint.append(carried)
        facts.move_axis.append(ax)
        facts.move_axis.append(ay)
        facts.move_axis.append(az)
    forward_kinematics["cpu", DT, DynDims, 1](d, m)

    var radii = _host_radii(f, fallback_radius)
    var drift = placement_table_drift[T](f, facts, radii)
    for i in range(len(drift)):
        if i < 6:
            print("      ", f.name, ":", drift[i])
    ta.check(len(drift) == 0,
             f.name + ": the table matches the family, the addresses, FK, the"
             " drawable joints and the nudge oracle")

    for i in range(len(tasks)):
        var t = load_task(task_path(f, tasks[i]))
        if yaw_all:
            for k in range(len(t.inits)):
                if f.init_target_kind(t.inits[k].region) != INIT_TARGET_SLOT:
                    t.inits[k].yaw = True
        validate_task_against_family(t, f)
        st.tasks += 1
        var expect_refuse = False
        for k in range(len(t.inits)):
            var r = f.region_index(t.inits[k].region)
            if r >= 0 and facts.move_joint[r] == -2:
                expect_refuse = True
        for k in range(len(t.joint_inits)):
            var hit = False
            for q in range(len(facts.joint_names)):
                if facts.joint_names[q] == t.joint_inits[k].joint:
                    hit = True
            if not hit:
                expect_refuse = True
        if expect_refuse:
            should.append(t.name)
        var raised = False
        try:
            require_device_placement[T](t, f)
        except e:
            raised = True
            refused.append(t.name)
            print("      refused:", t.name, "—", String(e)[byte=0:110])
        if raised:
            st.refused += 1
            continue

        var li = LaneInputs()
        li.n_regions = nr
        li.n_jinit = len(t.joint_inits)
        li.jadr = joint_init_addresses(t, fmd.joint_names, jqn)
        li.jdadr = joint_init_dof_addresses(t, fmd.joint_names, jvn)
        for lane in range(BATCH):
            var jv = sample_joint_inits(t, UInt64(SEED), lane)
            for i2 in range(facts.nq):
                d.qpos.data[i2] = Scalar[DT](0)
            for i2 in range(len(f.base_qpos)):
                d.qpos.data[i2] = Scalar[DT](f.base_qpos[i2])
            for k in range(len(jv)):
                d.qpos.data[li.jadr[k]] = Scalar[DT](jv[k])
                li.jvals.append(jv[k])
            forward_kinematics["cpu", DT, DynDims, 1](d, m)
            for r in range(nr):
                var s = rsites[r]
                li.frames.append(RegionFrame(
                    Float64(d.site_xpos.data[s * 3]),
                    Float64(d.site_xpos.data[s * 3 + 1]),
                    Float64(d.site_xpos.data[s * 3 + 2]),
                ))
        _coverage(t, f, facts.move_joint, st)
        _parity[T](t, f, li, radii, st)


def _synth_facts(
    n_regions: Int, frames: List[RegionFrame], addrs: List[SlotAddress]
) -> SceneFacts:
    var sf = SceneFacts()
    sf.frames = frames.copy()
    sf.addrs = addrs.copy()
    for _ in range(n_regions):
        sf.move_joint.append(-1)
        for _c in range(3):
            sf.move_axis.append(0.0)
    sf.nq = 21
    sf.nv = 18
    sf.nbody = 4
    sf.nsite = n_regions
    sf.gripper_site = 0
    for r in range(n_regions):
        sf.region_site.append(r)
    return sf^


def _synth_lanes(frames: List[RegionFrame]) -> LaneInputs:
    var li = LaneInputs()
    li.n_regions = len(frames)
    for _ in range(BATCH):
        for r in range(len(frames)):
            li.frames.append(frames[r])
    return li^


# ── section 7's scene: a stack wider than its reference ─────────────────────
#
# Binary fractions throughout, so the parsed `.family` and this table are the
# same bits and the drift check can stay exact.
comptime SYNTH_FAMILY = String(
    "schema_version=1\nfamily=synth_stack\nbase=b.xml\nhorizon=10\n"
    "slot=arena:static:a.xml:0.0,0.0,0.0\n"
    "slot=ref:free:p.xml\nslot=wide:free:p.xml\nslot=near:free:p.xml\n"
    "slot_geom=ref:-0.015625,0.015625,0.0078125\n"
    "slot_geom=wide:-0.015625,0.015625,0.078125\n"
    "slot_geom=near:-0.015625,0.015625,0.0078125\n"
    "region=left:site:s0:-0.00390625,-0.00390625,0.00390625,0.00390625\n"
    "region=right:site:s1:0.046875,-0.00390625,0.05078125,0.00390625\n"
)
comptime SYNTH_TASK = String(
    "schema_version=1\ntask=synth_stack_t\nfamily=synth_stack\n"
    "goal=On(wide, ref)\n"
    "active=arena\nactive=ref\nactive=wide\nactive=near\n"
    "init=ref@left\ninit=wide@ref\ninit=near@right\n"
)
comptime SYNTH_Z: Float64 = 0.8125


struct SynthStackPlacement(PlacementTable):
    comptime N_SLOTS: Int = 4
    comptime N_FREE: Int = 3
    comptime N_REGIONS: Int = 2
    comptime NQ: Int = 21
    comptime NV: Int = 18
    comptime N_JOINTS: Int = 0
    comptime NBODY: Int = 4
    comptime NSITE: Int = Self.N_REGIONS
    comptime GRIPPER_SITE: Int = 0
    comptime N_BASE_QPOS: Int = 0

    @staticmethod
    def base_qpos[DTYPE: DType](i: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0)

    @staticmethod
    def free_slot(j: Int) -> Int:
        return j + 1

    @staticmethod
    def free_qadr(j: Int) -> Int:
        return 7 * j

    @staticmethod
    def free_dadr(j: Int) -> Int:
        return 6 * j

    @staticmethod
    def free_has_geom(j: Int) -> Bool:
        return True

    @staticmethod
    def free_rest[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.015625)

    @staticmethod
    def free_radius[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.078125 if j == 1 else 0.0078125)

    @staticmethod
    def free_bottom_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](-0.015625)

    @staticmethod
    def free_top_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.015625)

    @staticmethod
    def region_site_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](SYNTH_Z)

    @staticmethod
    def region_has_rect(r: Int) -> Bool:
        return True

    @staticmethod
    def region_x0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.046875 if r == 1 else -0.00390625)

    @staticmethod
    def region_y0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](-0.00390625)

    @staticmethod
    def region_x1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.05078125 if r == 1 else 0.00390625)

    @staticmethod
    def region_y1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.00390625)

    @staticmethod
    def region_anchored(r: Int) -> Bool:
        return False

    @staticmethod
    def region_contact_has_geom(r: Int) -> Bool:
        return False

    @staticmethod
    def region_contact_top_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_move_joint(r: Int) -> Int:
        return -1

    @staticmethod
    def region_move_axis_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0)

    @staticmethod
    def region_move_axis_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0)

    @staticmethod
    def region_move_axis_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0)

    @staticmethod
    def joint_name(k: Int) -> String:
        return String("")

    @staticmethod
    def joint_qadr(k: Int) -> Int:
        return 0

    @staticmethod
    def joint_dadr(k: Int) -> Int:
        return 0
    @staticmethod
    def free_park_x[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        # `family.park_pos` at the default park (10, 0, 50), spacing 0.5
        return Scalar[DTYPE](10.0) + Scalar[DTYPE](j + 1) * Scalar[DTYPE](0.5)

    @staticmethod
    def free_park_y[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def free_park_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](50.0)

    @staticmethod
    def region_site(r: Int) -> Int:
        return r



# ── section 7b's scene: a fixture, its top, its interior, and a table zone ──
#
# ⚠ WHAT THE CORPUS DOES NOT REACH. `In` a fixture region and an inset that
# inverts both occur in exactly one corpus placement — `libero_spatial`'s bowl
# in the top drawer — and that task is REFUSED (its region moves). No corpus
# draw overlaps an object in a different region either. So those three rules
# would otherwise be in the kernel and never compared.
comptime SYNTH_FIXTURE_FAMILY = String(
    "schema_version=1\nfamily=synth_fixture\nbase=b.xml\nhorizon=10\n"
    "slot=arena:static:a.xml:0.0,0.0,0.0\n"
    "slot=box:static:x.xml:0.0,0.0,0.75\n"
    "slot=a:free:p.xml\nslot=b:free:p.xml\nslot=c:free:p.xml\n"
    "slot_geom=box:-0.046875,0.046875,0.125\n"
    "slot_geom=a:-0.015625,0.015625,0.0078125\n"
    "slot_geom=b:-0.015625,0.015625,0.03125\n"
    "slot_geom=c:-0.015625,0.015625,0.0078125\n"
    "region=top:box:s0:-0.0625,-0.0625,0.0625,0.0625:0.0078125:box\n"
    "region=inner:box:s1:-0.0234375,-0.0234375,0.0234375,0.0234375"
    ":0.0625:box\n"
    "region=zone:site:s2:-0.00390625,-0.00390625,0.00390625,0.00390625\n"
)
comptime SYNTH_FIXTURE_TASK = String(
    "schema_version=1\ntask=synth_fixture_t\nfamily=synth_fixture\n"
    "goal=In(b, inner)\n"
    "active=arena\nactive=box\nactive=a\nactive=b\nactive=c\n"
    "init=a@top\ninit=b@inner:in\ninit=c@zone\n"
)


struct SynthFixturePlacement(PlacementTable):
    comptime N_SLOTS: Int = 5
    comptime N_FREE: Int = 3
    comptime N_REGIONS: Int = 3
    comptime NQ: Int = 21
    comptime NV: Int = 18
    comptime N_JOINTS: Int = 0
    comptime NBODY: Int = 4
    comptime NSITE: Int = Self.N_REGIONS
    comptime GRIPPER_SITE: Int = 0
    comptime N_BASE_QPOS: Int = 0

    @staticmethod
    def base_qpos[DTYPE: DType](i: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0)

    @staticmethod
    def free_slot(j: Int) -> Int:
        return j + 2

    @staticmethod
    def free_qadr(j: Int) -> Int:
        return 7 * j

    @staticmethod
    def free_dadr(j: Int) -> Int:
        return 6 * j

    @staticmethod
    def free_has_geom(j: Int) -> Bool:
        return True

    @staticmethod
    def free_rest[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.015625)

    @staticmethod
    def free_radius[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.03125 if j == 1 else 0.0078125)

    @staticmethod
    def free_bottom_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](-0.015625)

    @staticmethod
    def free_top_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.015625)

    @staticmethod
    def region_site_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def region_site_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.75 if r == 1 else SYNTH_Z)

    @staticmethod
    def region_has_rect(r: Int) -> Bool:
        return True

    @staticmethod
    def region_x0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.0625)
        return Scalar[DTYPE](-0.0234375 if r == 1 else -0.00390625)

    @staticmethod
    def region_y0[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](-0.0625)
        return Scalar[DTYPE](-0.0234375 if r == 1 else -0.00390625)

    @staticmethod
    def region_x1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.0625)
        return Scalar[DTYPE](0.0234375 if r == 1 else 0.00390625)

    @staticmethod
    def region_y1[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        if r == 0:
            return Scalar[DTYPE](0.0625)
        return Scalar[DTYPE](0.0234375 if r == 1 else 0.00390625)

    @staticmethod
    def region_anchored(r: Int) -> Bool:
        return r != 2

    @staticmethod
    def region_contact_has_geom(r: Int) -> Bool:
        return r != 2

    @staticmethod
    def region_contact_top_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.046875 if r != 2 else 0.0)

    @staticmethod
    def region_move_joint(r: Int) -> Int:
        return -1

    @staticmethod
    def region_move_axis_x[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0)

    @staticmethod
    def region_move_axis_y[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0)

    @staticmethod
    def region_move_axis_z[DTYPE: DType](r: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0)

    @staticmethod
    def joint_name(k: Int) -> String:
        return String("")

    @staticmethod
    def joint_qadr(k: Int) -> Int:
        return 0

    @staticmethod
    def joint_dadr(k: Int) -> Int:
        return 0
    @staticmethod
    def free_park_x[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        # `family.park_pos` at the default park (10, 0, 50), spacing 0.5
        return Scalar[DTYPE](10.0) + Scalar[DTYPE](j + 2) * Scalar[DTYPE](0.5)

    @staticmethod
    def free_park_y[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](0.0)

    @staticmethod
    def free_park_z[DTYPE: DType](j: Int) -> Scalar[DTYPE]:
        return Scalar[DTYPE](50.0)

    @staticmethod
    def region_site(r: Int) -> Int:
        return r



def main() raises:
    print("=== the device reset vs the host: jinit= and placements ===")
    var ta = Tally()

    # ── 1. meta's two appended blocks ─────────────────────────────────────
    print("--- 1. meta's init and jinit blocks ---")
    ta.check(
        META_IDX_INIT_REGION_0 + META_INIT_SLOTS == META_IDX_JINIT_0
        and META_IDX_JINIT_0 + META_JINIT_SLOTS * META_JINIT_WORDS
        == META_IDX_NEWTON_ITER
        and META_IDX_NEWTON_ITER + META_SOLVER_WORDS == META_IDX_REWARD_MODE
        and META_IDX_REWARD_MODE + META_REWARD_WORDS == METADATA_SIZE,
        "the init block, the jinit block, the solver counters and the reward"
        " block are contiguous and END `meta` ("
        + String(META_IDX_INIT_REGION_0) + ".." + String(METADATA_SIZE - 1)
        + "), so widening moved no other word",
    )
    ta.check(META_IDX_LS_EVAL < META_IDX_INIT_REGION_0,
             "every older word sits below the appended blocks")

    # ── 2. so101_tabletop ─────────────────────────────────────────────────
    print()
    print("--- 2. so101_tabletop: its hand-written table, then device vs host ---")
    var f = load_family(SO101_FAMILY)
    var names = List[String]()
    names.append(String("so101_reach_brick"))
    names.append(String("so101_lift_brick"))
    names.append(String("so101_gather_bricks"))
    names.append(String("so101_settle_brick"))
    var st0 = Stats()
    var r0 = List[String]()
    var s0 = List[String]()
    _run_family[So101TabletopPlacement](
        f, names, SO_ARM101_NMESH_VERTS, So101TabletopConfig.SLOT_RADIUS,
        String("robot_gripperframe"), ta, st0, r0, s0,
    )
    print("      placements", st0.placements, " coordinates", st0.coords,
          " exact", st0.exact, " worst", st0.worst)
    if st0.placements == 0:
        raise Error(
            "device placement: so101_tabletop placed NOTHING, so the parity"
            " below compared nothing"
        )
    ta.check(
        st0.bad == 0 and st0.left_alone_bad == 0 and st0.meta_touched == 0
        and st0.other_written == 0 and len(r0) == 0,
        "so101_tabletop: every coordinate agrees, nothing else is written,"
        " nothing refused",
    )
    # ⚠ THE RADIUS IS READ FROM THE PROP'S OWN ASSET. For this family it is
    # both the clash radius and the resting height.
    var cube = parse_model_runtime("noeira/tasks/assets/props/cube.xml")
    var ok_rad = False
    for i in range(len(cube.geoms)):
        if cube.geoms[i].half_x == So101TabletopConfig.SLOT_RADIUS:
            ok_rad = True
    ta.check(ok_rad,
             "SLOT_RADIUS is the prop asset's own half-size (resting height)")

    # ── 2b. so101_tower: the generated table, and the rest-pose DRAW ──────
    # The one family with `base_qpos_jitter=` (the follower starts folded,
    # pan / roll / jaw drawn per episode): the device's draw must be the
    # host's on every lane, inside the half-width, and actually drawn.
    print()
    print("--- 2b. so101_tower: generated table, rest draw device vs host ---")
    var ft = load_family(String("noeira/tasks/families/so101_tower.family"))
    var tnames = List[String]()
    tnames.append(String("so101_tower_cube_in_bowl"))
    tnames.append(String("so101_tower_lift_brick"))
    tnames.append(String("so101_tower_reach_clear"))
    var stt = Stats()
    var rt = List[String]()
    var sht = List[String]()
    _run_family[So101TowerPlacement](
        ft, tnames, SO101_TOWER_NMESH_VERTS, 0.02,
        String("robot_grasp_center"), ta, stt, rt, sht,
    )
    print("      placements", stt.placements, " rest words", stt.base_words,
          " jittered", stt.base_jittered, " worst", stt.worst)
    ta.check(
        stt.bad == 0 and stt.base_bad == 0 and stt.left_alone_bad == 0
        and stt.meta_touched == 0 and stt.other_written == 0 and len(rt) == 0,
        "so101_tower: placements and the drawn rest agree device vs host,"
        " nothing else is written, nothing refused",
    )
    # 3 of the 6 rest words are drawn; every drawn word on every lane of
    # every task must have moved (a u of exactly 1/2 has probability 0)
    ta.check(
        stt.base_words == len(tnames) * BATCH * 6
        and stt.base_jittered == len(tnames) * BATCH * 3,
        "so101_tower: the rest is 6 words and pan / roll / jaw are drawn on"
        " every lane (" + String(stt.base_jittered) + " of "
        + String(len(tnames) * BATCH * 3) + ")",
    )
    var d0 = sample_base_qpos(ft, UInt64(SEED), 0)
    var d1 = sample_base_qpos(ft, UInt64(SEED), 1)
    var d0b = sample_base_qpos(ft, UInt64(SEED + 1), 0)
    ta.check(
        d0[0] != d1[0] and d0[0] != d0b[0] and d0[1] == d1[1]
        and d0[1] == ft.base_qpos[1],
        "so101_tower: the draw moves with the lane and the seed, and a"
        " zero half-width (lift) is the rest exactly",
    )

    # ── 2c. `:yaw` on the tower: the same tasks, every region init yawed ──
    print()
    print("--- 2c. so101_tower with :yaw on every init: device vs host ---")
    var sty = Stats()
    var ry = List[String]()
    var shy = List[String]()
    _run_family[So101TowerPlacement](
        ft, tnames, SO101_TOWER_NMESH_VERTS, 0.02,
        String("robot_grasp_center"), ta, sty, ry, shy, yaw_all=True,
    )
    print("      placements", sty.placements, " yawed", sty.yawed,
          " worst", sty.worst)
    ta.check(
        sty.bad == 0 and sty.left_alone_bad == 0 and sty.other_written == 0
        and len(ry) == 0 and sty.placements == stt.placements
        and sty.yawed == sty.placements,
        "so101_tower :yaw: every placement drew a yaw, the device's quaternion"
        " is the host's, and the placements are the no-yaw ones' count",
    )
    # 2b ran the tasks as written: only their own `:yaw` inits may draw a yaw
    # (cube_in_bowl's brick since ce11b131f), one per lane each
    var n_yaw_inits = 0
    for k in range(len(tnames)):
        var tk = load_task(task_path(ft, tnames[k]))
        for q in range(len(tk.inits)):
            if tk.inits[q].yaw:
                n_yaw_inits += 1
    ta.check(
        stt.yawed == n_yaw_inits * BATCH,
        "so101_tower as written: exactly the tasks' own :yaw inits drew a yaw ("
        + String(stt.yawed) + " = " + String(n_yaw_inits) + " x "
        + String(BATCH) + " lanes), every other quaternion the identity",
    )
    var pt = parse_task(String(
        "schema_version=1\ntask=yaw_probe\nfamily=so101_tower\n"
        "language=probe\ngoal=Near(brick, bowl, 0.045)\nactive=tower\n"
        "active=desk\nactive=bowl\nactive=brick\n"
        "init=bowl@desk_left\ninit=brick@desk_right:yaw\n"
    ))
    ta.check(
        not pt.inits[0].yaw and pt.inits[1].yaw
        and pt.inits[1].describe() == "brick@desk_right:yaw",
        "':yaw' parses on its own init only and round-trips through describe",
    )
    var bad_order = False
    try:
        _ = parse_init(String("brick@desk_right:yaw:in"))
    except:
        bad_order = True
    ta.check(bad_order, "':yaw:in' (suffixes out of order) is refused")

    # ── 2d. `:sep=` — cube_in_bowl's bowl keeps 135 mm from the brick ────
    #
    # 2b ran cube_in_bowl AS WRITTEN (`init=bowl@desk_bowl:sep=0.135`), so its
    # device-vs-host agreement above already covers the separation's word and
    # the kernel's clash test; here: the rule holds on every host draw, and it
    # is NOT VACUOUS — draws the radii alone would accept were refused.
    print()
    print("--- 2d. :sep= on the tower (cube_in_bowl as written) ---")
    print("      pairs", stt.sep_pairs, " sep-only rejections", stt.sep_rejected,
          " smallest margin", stt.sep_margin)
    ta.check(
        stt.sep_pairs >= BATCH and stt.sep_margin >= 0.0,
        "so101_tower :sep=: every separated pair keeps its distance ("
        + String(stt.sep_pairs) + " pairs, margin >= 0)",
    )
    ta.check(
        stt.sep_rejected > 0,
        "so101_tower :sep=: the separation refused draws the radii alone"
        " accept (" + String(stt.sep_rejected) + ") — the rule is reached",
    )
    var ps = parse_init(String("bowl@desk_bowl:yaw:sep=0.135"))
    ta.check(
        ps.yaw and ps.sep_mm == 135
        and ps.describe() == "bowl@desk_bowl:yaw:sep=0.135",
        "':sep=' parses after ':yaw' and round-trips through describe",
    )
    var bad_sep = 0
    for spec in [String("bowl@desk_bowl:sep=0.1355"), String("bowl@desk_bowl:sep=2.0"),
                 String("bowl@desk_bowl:sep=0")]:
        try:
            _ = parse_init(spec)
        except:
            bad_sep += 1
    ta.check(bad_sep == 3, "':sep=' refuses sub-millimetre, > 1.023 m and 0")

    # ── 3. an untouched meta writes nothing ───────────────────────────────
    #
    # ⚠⚠ THE CASE A DRIVER FALLS INTO BY FORGETTING. `Data.__init__` uploads a
    # ZERO-FILLED `meta`: zero must mean no placement AND no joint draw.
    print()
    print("--- 3. an untouched meta leaves every qpos and qvel word alone ---")
    comptime NQ0 = So101TabletopPlacement.NQ
    comptime NV0 = So101TabletopPlacement.NV
    var zs = TensorImpl[DT].alloc(BATCH * NQ0)
    var zv = TensorImpl[DT].alloc(BATCH * NV0)
    var zm = TensorImpl[DT].alloc(BATCH * METADATA_SIZE)
    for i in range(BATCH * NQ0):
        zs.data[i] = Scalar[DT](QPOS_SENTINEL)
    for i in range(BATCH * NV0):
        zv.data[i] = Scalar[DT](QVEL_SENTINEL)
    for i in range(BATCH * METADATA_SIZE):
        zm.data[i] = Scalar[DT](0)
    var zqt = zs.lt["cpu", Layout.row_major(BATCH, NQ0)]()
    var zvt = zv.lt["cpu", Layout.row_major(BATCH, NV0)]()
    var zmt = zm.lt["cpu", Layout.row_major(BATCH, METADATA_SIZE)]()
    # ⚠ THE FIVE UNUSED TENSORS ARE SIZE-1 SCRATCH, named so they outlive the
    # call — the hook ends with `_ = joints` / `_ = bodies` / ...
    var dj = TensorImpl[DT].alloc(MODEL_JOINT_SIZE)
    var dm3 = TensorImpl[DT].alloc(BATCH * 3)
    var dm4 = TensorImpl[DT].alloc(BATCH * 4)
    var db = TensorImpl[DT].alloc(MODEL_BODY_SIZE)
    var dg = TensorImpl[DT].alloc(MODEL_GEOM_SIZE)
    for lane in range(BATCH):
        So101TabletopConfig.init_qpos_gpu[DT, BATCH, NQ0, 1, NV0, 1, 1](
            zqt, zvt,
            dj.lt["cpu", Layout.row_major(1, MODEL_JOINT_SIZE)](),
            dm3.lt["cpu", Layout.row_major(BATCH, 3)](),
            dm4.lt["cpu", Layout.row_major(BATCH, 4)](),
            db.lt["cpu", Layout.row_major(1, MODEL_BODY_SIZE)](),
            dg.lt["cpu", Layout.row_major(1, MODEL_GEOM_SIZE)](),
            zmt, lane, SEED,
        )
    var parked_ok = True
    for i in range(BATCH * NQ0):
        if Float64(zs.data[i]) != QPOS_SENTINEL:
            parked_ok = False
    for i in range(BATCH * NV0):
        if Float64(zv.data[i]) != QVEL_SENTINEL:
            parked_ok = False
    ta.check(parked_ok,
             "through So101TabletopConfig.init_qpos_gpu, a zero meta writes"
             " NO qpos or qvel word")

    # ── 4 + 5. every LIBERO family ────────────────────────────────────────
    print()
    print("--- 4 + 5. every LIBERO family: its table, then every task ---")
    var fams = _libero_families()
    ta.check(len(fams) == N_LIBERO_FAMILIES,
             String(len(fams)) + " libero*.family files, and this gate imports "
             + String(N_LIBERO_FAMILIES) + " tables")
    var st = Stats()
    var refused = List[String]()
    var should = List[String]()
    for i in range(len(fams)):
        ref nm = fams[i]
        var lf = load_family(String(FAMILY_DIR) + "/" + nm + ".family")
        if nm == "libero_goal":
            _run_family[LiberoGoalPlacement](
                lf, _tasks_of(nm), 32768, 0.02, String("robot_grip_site"), ta, st,
                refused, should
            )
        elif nm == "libero_kitchen_scene1":
            _run_family[LiberoKitchenScene1Placement](
                lf, _tasks_of(nm), 32768, 0.02, String("robot_grip_site"), ta, st,
                refused, should
            )
        elif nm == "libero_kitchen_scene2":
            _run_family[LiberoKitchenScene2Placement](
                lf, _tasks_of(nm), 32768, 0.02, String("robot_grip_site"), ta, st,
                refused, should
            )
        elif nm == "libero_kitchen_scene3":
            _run_family[LiberoKitchenScene3Placement](
                lf, _tasks_of(nm), 32768, 0.02, String("robot_grip_site"), ta, st,
                refused, should
            )
        elif nm == "libero_kitchen_scene4":
            _run_family[LiberoKitchenScene4Placement](
                lf, _tasks_of(nm), 32768, 0.02, String("robot_grip_site"), ta, st,
                refused, should
            )
        elif nm == "libero_kitchen_scene5":
            _run_family[LiberoKitchenScene5Placement](
                lf, _tasks_of(nm), 32768, 0.02, String("robot_grip_site"), ta, st,
                refused, should
            )
        elif nm == "libero_kitchen_scene6":
            _run_family[LiberoKitchenScene6Placement](
                lf, _tasks_of(nm), 32768, 0.02, String("robot_grip_site"), ta, st,
                refused, should
            )
        elif nm == "libero_kitchen_scene7":
            _run_family[LiberoKitchenScene7Placement](
                lf, _tasks_of(nm), 32768, 0.02, String("robot_grip_site"), ta, st,
                refused, should
            )
        elif nm == "libero_kitchen_scene8":
            _run_family[LiberoKitchenScene8Placement](
                lf, _tasks_of(nm), 32768, 0.02, String("robot_grip_site"), ta, st,
                refused, should
            )
        elif nm == "libero_kitchen_scene9":
            _run_family[LiberoKitchenScene9Placement](
                lf, _tasks_of(nm), 32768, 0.02, String("robot_grip_site"), ta, st,
                refused, should
            )
        elif nm == "libero_kitchen_scene10":
            _run_family[LiberoKitchenScene10Placement](
                lf, _tasks_of(nm), 32768, 0.02, String("robot_grip_site"), ta, st,
                refused, should
            )
        elif nm == "libero_living_room_scene1":
            _run_family[LiberoLivingRoomScene1Placement](
                lf, _tasks_of(nm), 32768, 0.02, String("robot_grip_site"), ta, st,
                refused, should
            )
        elif nm == "libero_living_room_scene2":
            _run_family[LiberoLivingRoomScene2Placement](
                lf, _tasks_of(nm), 32768, 0.02, String("robot_grip_site"), ta, st,
                refused, should
            )
        elif nm == "libero_living_room_scene3":
            _run_family[LiberoLivingRoomScene3Placement](
                lf, _tasks_of(nm), 32768, 0.02, String("robot_grip_site"), ta, st,
                refused, should
            )
        elif nm == "libero_living_room_scene4":
            _run_family[LiberoLivingRoomScene4Placement](
                lf, _tasks_of(nm), 32768, 0.02, String("robot_grip_site"), ta, st,
                refused, should
            )
        elif nm == "libero_living_room_scene5":
            _run_family[LiberoLivingRoomScene5Placement](
                lf, _tasks_of(nm), 32768, 0.02, String("robot_grip_site"), ta, st,
                refused, should
            )
        elif nm == "libero_living_room_scene6":
            _run_family[LiberoLivingRoomScene6Placement](
                lf, _tasks_of(nm), 32768, 0.02, String("robot_grip_site"), ta, st,
                refused, should
            )
        elif nm == "libero_object":
            _run_family[LiberoObjectPlacement](
                lf, _tasks_of(nm), 32768, 0.02, String("robot_grip_site"), ta, st,
                refused, should
            )
        elif nm == "libero_spatial":
            _run_family[LiberoSpatialPlacement](
                lf, _tasks_of(nm), 32768, 0.02, String("robot_grip_site"), ta, st,
                refused, should
            )
        elif nm == "libero_study_scene1":
            _run_family[LiberoStudyScene1Placement](
                lf, _tasks_of(nm), 32768, 0.02, String("robot_grip_site"), ta, st,
                refused, should
            )
        elif nm == "libero_study_scene2":
            _run_family[LiberoStudyScene2Placement](
                lf, _tasks_of(nm), 32768, 0.02, String("robot_grip_site"), ta, st,
                refused, should
            )
        elif nm == "libero_study_scene3":
            _run_family[LiberoStudyScene3Placement](
                lf, _tasks_of(nm), 32768, 0.02, String("robot_grip_site"), ta, st,
                refused, should
            )
        elif nm == "libero_study_scene4":
            _run_family[LiberoStudyScene4Placement](
                lf, _tasks_of(nm), 32768, 0.02, String("robot_grip_site"), ta, st,
                refused, should
            )
        else:
            ta.check(False, nm + ": a LIBERO family with no table in this gate"
                     " — run `pixi run gen-placement-tables` and import it")

    print()
    print("      tasks", st.tasks, " refused", st.refused, " lanes", st.lanes,
          " placements", st.placements)
    print("      coordinates", st.coords, " exact", st.exact, " worst",
          st.worst, " differing", st.bad)
    print("      jinit draws", st.jinit_draws, " differing", st.jinit_bad)
    print("      unplaced free slots", st.left_alone, " written anyway",
          st.left_alone_bad, " other words written", st.other_written,
          " meta words written", st.meta_touched)
    ta.check(st.tasks == N_LIBERO_TASKS,
             String(st.tasks) + " LIBERO tasks visited (the importer's "
             + String(N_LIBERO_TASKS) + ")")
    if st.placements == 0 or st.jinit_draws == 0:
        raise Error(
            "device placement: no LIBERO placement or joint draw was compared,"
            " so 'no differences' below would say nothing"
        )
    ta.check(st.bad == 0,
             "the device and the host agree on every coordinate of every"
             " placement, within " + String(TOL) + " m")
    print("      base_qpos words", st.base_words, " differing", st.base_bad)
    ta.check(st.base_words > 0 and st.base_bad == 0,
             String(st.base_words) + " rest-pose words: the reset writes the"
             " family's base_qpos and zeroes those velocities (q = 0 is the"
             " Panda's singular pose)")
    ta.check(st.jinit_bad == 0,
             String(st.jinit_draws) + " joint draws: the device writes the"
             " host's value and zeroes the velocity")
    ta.check(st.left_alone_bad == 0 and st.other_written == 0,
             "the device writes a free slot only where the host places it, and"
             " no other qpos/qvel word at all")
    ta.check(st.meta_touched == 0, "the kernel never writes meta")
    var same = len(refused) == len(should)
    for i in range(len(refused)):
        var hit = False
        for k in range(len(should)):
            if should[k] == refused[i]:
                hit = True
        if not hit:
            same = False
    ta.check(same,
             "the refused corpus tasks are EXACTLY those the nudge oracle says"
             " the kernel cannot follow (" + String(len(refused)) + " refused)")

    print()
    print("--- 6. the rules the CORPUS placements reach (checked after 7) ---")
    print("      slot_geom", st.geom, " table z offset", st.table_off,
          " On a fixture", st.on_fixture, " In a fixture", st.in_fixture,
          " stacks", st.stacks, " in a followed region", st.followed)
    print("      tasks whose walk is not slot order", st.reordered_tasks,
          " rejections", st.rejections, " clamped axes", st.clamped,
          " exempt overlaps", st.exempt)

    # ── 7. the stack exemption, and the host crash ────────────────────────
    #
    # ⚠⚠ THE HOST USED TO INDEX `f.regions[-1]` HERE AND MOJO ASSERTS ON THAT.
    # A stack's `of_region` is -1; a TABLE-region draw that overlapped a stack
    # evaluated `f.regions[ri_j]` and `mojo run` crashed. No corpus seed reached
    # it: a stack stands on its reference, which rejects the draw first unless
    # the stack is the WIDER of the two — so this scene makes it wider.
    print()
    print("--- 7. a draw overlapping a stack wider than its reference ---")
    var sf = parse_family(SYNTH_FAMILY)
    var stask = parse_task(SYNTH_TASK)
    validate_task_against_family(stask, sf)
    var sframes = List[RegionFrame]()
    sframes.append(RegionFrame(0.0, 0.0, SYNTH_Z))
    sframes.append(RegionFrame(0.0, 0.0, SYNTH_Z))
    var saddrs = List[SlotAddress]()
    saddrs.append(SlotAddress(-1, -1))
    for j in range(3):
        saddrs.append(SlotAddress(7 * j, 6 * j))
    var sradii = _host_radii(sf, 0.02)
    var sdrift = placement_table_drift[SynthStackPlacement](
        sf, _synth_facts(2, sframes, saddrs), sradii
    )
    for i in range(len(sdrift)):
        print("      synth:", sdrift[i])
    ta.check(len(sdrift) == 0, "the synthetic table matches its family")
    require_device_placement[SynthStackPlacement](stask, sf)
    var s7 = Stats()
    var smj = List[Int](length=2, fill=-1)
    _coverage(stask, sf, smj, s7)
    _parity[SynthStackPlacement](stask, sf, _synth_lanes(sframes), sradii, s7)
    print("      placements", s7.placements, " exempt overlaps", s7.exempt,
          " worst", s7.worst)
    ta.check(s7.exempt >= BATCH,
             "every lane's 'near' draw overlaps the wide stack and is exempted"
             " (" + String(s7.exempt) + ")")
    ta.check(s7.placements == 3 * BATCH and s7.bad == 0,
             "the host places all three without crashing, and the device"
             " agrees on every coordinate")

    # ── 7b. a fixture: On its top, In its interior, beside a table zone ───
    print()
    print("--- 7b. On a fixture, In a fixture too narrow, and a table zone ---")
    var ff = parse_family(SYNTH_FIXTURE_FAMILY)
    var ftask = parse_task(SYNTH_FIXTURE_TASK)
    validate_task_against_family(ftask, ff)
    var fframes = List[RegionFrame]()
    fframes.append(RegionFrame(0.0, 0.0, SYNTH_Z))
    fframes.append(RegionFrame(0.0, 0.0, 0.75))
    fframes.append(RegionFrame(0.0, 0.0, SYNTH_Z))
    var faddrs = List[SlotAddress]()
    faddrs.append(SlotAddress(-1, -1))
    faddrs.append(SlotAddress(-1, -1))
    for j in range(3):
        faddrs.append(SlotAddress(7 * j, 6 * j))
    var fradii = _host_radii(ff, 0.02)
    var fdrift = placement_table_drift[SynthFixturePlacement](
        ff, _synth_facts(3, fframes, faddrs), fradii
    )
    for i in range(len(fdrift)):
        print("      synth_fixture:", fdrift[i])
    ta.check(len(fdrift) == 0, "the synthetic fixture table matches its family")
    require_device_placement[SynthFixturePlacement](ftask, ff)
    var s7b = Stats()
    var fmj = List[Int](length=3, fill=-1)
    _coverage(ftask, ff, fmj, s7b)
    _parity[SynthFixturePlacement](
        ftask, ff, _synth_lanes(fframes), fradii, s7b
    )
    print("      placements", s7b.placements, " On", s7b.on_fixture, " In",
          s7b.in_fixture, " clamped", s7b.clamped, " exempt", s7b.exempt,
          " worst", s7b.worst)
    ta.check(s7b.placements == 3 * BATCH and s7b.bad == 0
             and s7b.left_alone_bad == 0,
             "all three placed, and the device agrees on every coordinate")

    # ── 8. the refusals, on tasks built to need them ──────────────────────
    #
    # ⚠ THE CORPUS NO LONGER NEEDS ONE — the drawer task is followed now — so
    # a refusal that raised on nothing would pass section 5 for free. Three
    # real corpus tasks, each changed in ONE way, must each be refused, and the
    # unchanged task must not be.
    print()
    print("--- 8. refusals: an unfollowable region, an unknown joint, too many draws ---")
    var of = load_family(String(FAMILY_DIR) + "/libero_object.family")
    var otasks = _tasks_of(String("libero_object"))
    var base_t = load_task(TASK_DIR + otasks[0] + ".task")
    validate_task_against_family(base_t, of)
    var base_ok = True
    try:
        require_device_placement[LiberoObjectPlacement](base_t, of)
    except:
        base_ok = False
    ta.check(base_ok, "the unchanged libero_object task is accepted")
    var bad_region = load_task(TASK_DIR + otasks[0] + ".task")
    bad_region.inits[0].region = String("basket_1_contain_region")
    var r1 = False
    try:
        require_device_placement[LiberoObjectPlacement](bad_region, of)
    except:
        r1 = True
    ta.check(r1,
             "a prop drawn into the basket (a region on a FREE body) is refused")
    var gf = load_family(String(FAMILY_DIR) + "/libero_goal.family")
    var gtasks = _tasks_of(String("libero_goal"))
    var bad_joint = load_task(TASK_DIR + gtasks[0] + ".task")
    bad_joint.joint_inits.append(JointInitSpec(String("not_a_joint"), 0.0, 0.1))
    var r2 = False
    try:
        require_device_placement[LiberoGoalPlacement](bad_joint, gf)
    except:
        r2 = True
    ta.check(r2, "a jinit= on a joint the table does not list is refused")
    var many = load_task(TASK_DIR + gtasks[0] + ".task")
    for _ in range(META_JINIT_SLOTS + 1):
        many.joint_inits.append(JointInitSpec(
            String(LiberoGoalPlacement.joint_name(0)), -0.16, -0.14
        ))
    var r3 = False
    try:
        require_device_placement[LiberoGoalPlacement](many, gf)
    except:
        r3 = True
    ta.check(r3, String(META_JINIT_SLOTS + 1) + " jinit= lines are refused")

    # ── 6 (checked). every rule is reached by SOMETHING compared above ────
    #
    # ⚠ THE SOURCE IS PRINTED BESIDE EACH COUNT, because "reached" by a
    # synthetic scene is a weaker statement than reached by the corpus.
    print()
    print("--- 6. every rule the kernel mirrors, reached (corpus + synthetic) ---")
    var sy = Stats()
    sy.in_fixture = s7.in_fixture + s7b.in_fixture
    sy.clamped = s7.clamped + s7b.clamped
    sy.exempt = s7.exempt + s7b.exempt
    print("      corpus   : In", st.in_fixture, " clamped", st.clamped,
          " exempt", st.exempt)
    print("      synthetic: In", sy.in_fixture, " clamped", sy.clamped,
          " exempt", sy.exempt)
    ta.check(st.geom > 0, "rule 3: resting heights from slot_geom=")
    ta.check(st.table_off > 0, "rule 4: TABLE_Z_OFFSET on a table region")
    ta.check(st.on_fixture > 0, "rule 4: a fixture's top_z for On")
    ta.check(st.in_fixture + sy.in_fixture > 0,
             "rule 4: no top_z for In (corpus " + String(st.in_fixture)
             + ", synthetic " + String(sy.in_fixture) + ")")
    ta.check(st.stacks > 0, "rule 7: stacks")
    ta.check(st.reordered_tasks > 0,
             "rule 2: a walk that is not slot order (order_inits)")
    ta.check(st.rejections > 0, "rule 6: the clash test rejects draws")
    ta.check(st.clamped + sy.clamped > 0,
             "rule 5: an inset that inverts and is clamped (corpus "
             + String(st.clamped) + ", synthetic " + String(sy.clamped) + ")")
    ta.check(st.exempt + sy.exempt > 0,
             "rule 6: overlaps exempted across regions or against a stack"
             " (corpus " + String(st.exempt) + ", synthetic "
             + String(sy.exempt) + ")")
    ta.check(st.jinit_draws > 0, "jinit=: joint draws")
    ta.check(st.followed > 0,
             "jinit=: placements in a region that FOLLOWS a drawn slide ("
             + String(st.followed) + ")")

    print()
    print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
    if ta.failures != 0:
        raise Error(
            "device placement: " + String(ta.failures) + " of "
            + String(ta.checks) + " check(s) failed"
        )
    print("=== PASS ===")

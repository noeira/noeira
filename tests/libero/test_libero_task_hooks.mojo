"""THE PER-STEP TASK HOOKS ON EVERY LIBERO FAMILY — mask, repark, observation,
and the success word.

    pixi run mojo run -I . tests/libero/test_libero_task_hooks.mojo

`tasks/task_hooks.mojo` is the one implementation of the repark and of the
observation, over a family's `PlacementTable`; `So101TabletopConfig` and
`libero_osc_config.LiberoOscConfig` both call it. `tests/tasks/test_active_mask.mojo` gates it
on `so101_tabletop`, whose regions share ONE site and whose three tasks all run
on one table. This gates it where that family cannot look.

## WHAT IT ASSERTS

1. **Every family's comptime facts agree with its files**, on all 23: the
   observation width the env allocates (`M.OBS_DIM`) is `NQ + NV + N_FREE +
   TASK_GOAL_WORDS` (what the hook writes); the generated table and the
   generated model def agree on nq/nv/nbody/nsite; the cone is ELLIPTIC and
   `MAX_CONDIM` is the parsed scene's own (4, not the default 3); the
   contact budget is ABOVE the peak `libero-contact-budget` measured
   (`contact_budget.kv`); and the ONE config's restated constants —
   `FRAME_SKIP`, `MAX_STEPS`, `get_timestep` — are this family's
   `control_freq`, `horizon` and MuJoCo-generated timestep.
2. **Every LIBERO task, two lanes** (task `i` beside task `i + 1`), each lane
   reset by the host exactly as the eval does (joint draws, FK, placements,
   `reset_slots`):
   * the device writer and the host writer produce IDENTICAL vectors;
   * both equal an ORACLE built here from the scene and the bound goal, not
     from the table: state copied, an inactive slot's 13 words zeroed, the
     active word = `t.is_active`, the gripper at `robot_grip_site` looked up
     BY NAME, and an `In`/`On` target at `region_sites(f)[b]` — the region's
     own site from the scene;
   * ⚠⚠ the negative leg: two lanes with different active sets read
     different mask words — a writer that read one lane's mask for both
     passes every single-lane check. (A writer resolving every region to one
     site, the `so101_tabletop` shape, fails the ORACLE check; the gate also
     requires the corpus's targets to span more than one site, so that check
     has something to catch.);
   * the repark pins exactly the inactive slots at `family.park_pos`, pose
     and velocity, and leaves every other word bit-identical.
3. **Every family's config hooks** — `LiberoOscConfig[P]`'s own, not the
   helpers — on all 23: both observation hooks and the repark equal the
   helpers, and the reward hook writes `META_IDX_GOAL_HELD` equal to its reward
   and to `eval_tape_gpu`, on a tape that holds and one that does not.

⚠ ANTI-VACUITY: `qvel` is seeded non-zero everywhere, so "an inactive slot's
velocity words are zero" cannot pass on a zero state; the counts of zeroed,
copied and pinned words are printed and required non-zero.
"""

from std.os import listdir
from std.os.path import exists

from noeira.core.tiles import RowMajor
from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.gpu.constants import (
    METADATA_SIZE, META_IDX_TASK_ACTIVE, META_IDX_TASK_PARAM_0,
    META_IDX_GOAL_HELD, MODEL_BODY_SIZE, MODEL_SITE_SIZE, MODEL_GEOM_SIZE,
    MODEL_CURRICULUM_SIZE, CONTACT_SIZE,
)
from noeira.physics3d.fields import Data, Model, DynDims
from noeira.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.tasks.spec import (
    FamilySpec, TaskSpec, load_family, load_task, validate_task_against_family,
    SLOT_FREE,
)
from noeira.tasks.family import scene_path, park_pos
from noeira.tasks.predicates import (
    parse_goal, bind_goal, joint_qpos_addresses, BoundGoal,
    OP_IN, OP_ON, OP_AT_REGION, OP_UPRIGHT, OP_JOINT,
)
from noeira.tasks.eval import region_sites, region_contact_bodies
from noeira.tasks.tape import encode_goal, TAPE_WORDS
from noeira.tasks.gpu_eval import eval_tape_gpu, region_table_words
from noeira.tasks.active import active_mask
from noeira.tasks.reset import (
    free_slot_addresses, reset_slots, joint_init_addresses,
    joint_init_dof_addresses, apply_joint_inits, SlotAddress,
)
from noeira.tasks.sampler import (
    sample_placements, sample_joint_inits, RegionFrame, SampleReport,
)
from noeira.tasks.task_hooks import (
    repark_inactive_slots, write_task_obs, write_task_obs_host,
    TASK_GOAL_WORDS,
)
from noeira.tasks.placement.table import PlacementTable
from noeira.physics3d.model.model_def import ModelDefLike
from noeira.physics3d.types import ConeType
from noeira.envs.phyics3d_env_config import Phyics3dEnvConfig
from noeira.envs.libero.osc_config import LiberoOscConfig
from noeira.envs.libero.placement.libero_goal import LiberoGoalPlacement
from noeira.envs.libero.models.libero_goal_xml import LiberoGoalModel
from noeira.envs.libero.placement.libero_kitchen_scene1 import LiberoKitchenScene1Placement
from noeira.envs.libero.models.libero_kitchen_scene1_xml import LiberoKitchenScene1Model
from noeira.envs.libero.placement.libero_kitchen_scene10 import LiberoKitchenScene10Placement
from noeira.envs.libero.models.libero_kitchen_scene10_xml import LiberoKitchenScene10Model
from noeira.envs.libero.placement.libero_kitchen_scene2 import LiberoKitchenScene2Placement
from noeira.envs.libero.models.libero_kitchen_scene2_xml import LiberoKitchenScene2Model
from noeira.envs.libero.placement.libero_kitchen_scene3 import LiberoKitchenScene3Placement
from noeira.envs.libero.models.libero_kitchen_scene3_xml import LiberoKitchenScene3Model
from noeira.envs.libero.placement.libero_kitchen_scene4 import LiberoKitchenScene4Placement
from noeira.envs.libero.models.libero_kitchen_scene4_xml import LiberoKitchenScene4Model
from noeira.envs.libero.placement.libero_kitchen_scene5 import LiberoKitchenScene5Placement
from noeira.envs.libero.models.libero_kitchen_scene5_xml import LiberoKitchenScene5Model
from noeira.envs.libero.placement.libero_kitchen_scene6 import LiberoKitchenScene6Placement
from noeira.envs.libero.models.libero_kitchen_scene6_xml import LiberoKitchenScene6Model
from noeira.envs.libero.placement.libero_kitchen_scene7 import LiberoKitchenScene7Placement
from noeira.envs.libero.models.libero_kitchen_scene7_xml import LiberoKitchenScene7Model
from noeira.envs.libero.placement.libero_kitchen_scene8 import LiberoKitchenScene8Placement
from noeira.envs.libero.models.libero_kitchen_scene8_xml import LiberoKitchenScene8Model
from noeira.envs.libero.placement.libero_kitchen_scene9 import LiberoKitchenScene9Placement
from noeira.envs.libero.models.libero_kitchen_scene9_xml import LiberoKitchenScene9Model
from noeira.envs.libero.placement.libero_living_room_scene1 import LiberoLivingRoomScene1Placement
from noeira.envs.libero.models.libero_living_room_scene1_xml import LiberoLivingRoomScene1Model
from noeira.envs.libero.placement.libero_living_room_scene2 import LiberoLivingRoomScene2Placement
from noeira.envs.libero.models.libero_living_room_scene2_xml import LiberoLivingRoomScene2Model
from noeira.envs.libero.placement.libero_living_room_scene3 import LiberoLivingRoomScene3Placement
from noeira.envs.libero.models.libero_living_room_scene3_xml import LiberoLivingRoomScene3Model
from noeira.envs.libero.placement.libero_living_room_scene4 import LiberoLivingRoomScene4Placement
from noeira.envs.libero.models.libero_living_room_scene4_xml import LiberoLivingRoomScene4Model
from noeira.envs.libero.placement.libero_living_room_scene5 import LiberoLivingRoomScene5Placement
from noeira.envs.libero.models.libero_living_room_scene5_xml import LiberoLivingRoomScene5Model
from noeira.envs.libero.placement.libero_living_room_scene6 import LiberoLivingRoomScene6Placement
from noeira.envs.libero.models.libero_living_room_scene6_xml import LiberoLivingRoomScene6Model
from noeira.envs.libero.placement.libero_object import LiberoObjectPlacement
from noeira.envs.libero.models.libero_object_xml import LiberoObjectModel
from noeira.envs.libero.placement.libero_spatial import LiberoSpatialPlacement
from noeira.envs.libero.models.libero_spatial_xml import LiberoSpatialModel
from noeira.envs.libero.placement.libero_study_scene1 import LiberoStudyScene1Placement
from noeira.envs.libero.models.libero_study_scene1_xml import LiberoStudyScene1Model
from noeira.envs.libero.placement.libero_study_scene2 import LiberoStudyScene2Placement
from noeira.envs.libero.models.libero_study_scene2_xml import LiberoStudyScene2Model
from noeira.envs.libero.placement.libero_study_scene3 import LiberoStudyScene3Placement
from noeira.envs.libero.models.libero_study_scene3_xml import LiberoStudyScene3Model
from noeira.envs.libero.placement.libero_study_scene4 import LiberoStudyScene4Placement
from noeira.envs.libero.models.libero_study_scene4_xml import LiberoStudyScene4Model

comptime DT = DType.float64
comptime FAMILY_DIR = "noeira/envs/libero/families"
comptime TASK_DIR = "noeira/envs/libero/tasks/"
comptime B = 2
comptime SEED = 7
comptime MC = 64
comptime N_LIBERO_FAMILIES = 23
comptime N_LIBERO_TASKS = 129
comptime GRIP_NAME = "robot_grip_site"
comptime BUDGET_KV = "noeira/envs/libero/tables/contact_budget.kv"


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

    def quiet(mut self, ok: Bool, what: String):
        """A per-family check: counted, printed only when it fails — 23
        families times eight facts would bury the summary lines."""
        self.checks += 1
        if not ok:
            self.failures += 1
            print("  FAIL:", what)


struct Acc(Copyable, Movable):
    var tasks: Int
    var lanes: Int
    var words: Int
    var dev_host_bad: Int
    var oracle_bad: Int
    var copied: Int
    var zeroed: Int
    var mask_pairs_differ: Int
    var mask_pairs_bad: Int
    var region_goal_lanes: Int
    var target_sites: List[Int]
    var pinned: Int
    var repark_bad: Int
    var untouched: Int

    def __init__(out self):
        self.tasks = 0
        self.lanes = 0
        self.words = 0
        self.dev_host_bad = 0
        self.oracle_bad = 0
        self.copied = 0
        self.zeroed = 0
        self.mask_pairs_differ = 0
        self.mask_pairs_bad = 0
        self.region_goal_lanes = 0
        self.target_sites = List[Int]()
        self.pinned = 0
        self.repark_bad = 0
        self.untouched = 0


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


def _index(names: List[String], want: String) raises -> Int:
    for i in range(len(names)):
        if names[i] == want:
            return i
    raise Error("no '" + want + "' in the scene")


def _oracle_obs(
    f: FamilySpec,
    t: TaskSpec,
    g: BoundGoal,
    addrs: List[SlotAddress],
    rsites: List[Int],
    grip: Int,
    qpos: List[Float64],
    qvel: List[Float64],
    xpos: List[Float64],
    sxp: List[Float64],
) raises -> List[Float64]:
    """What the observation MUST be, from the scene and the spec alone."""
    var nq = len(qpos)
    var nv = len(qvel)
    var out = List[Float64]()
    for i in range(nq):
        out.append(qpos[i])
    for i in range(nv):
        out.append(qvel[i])
    for si in range(len(f.slots)):
        if f.slots[si].kind != SLOT_FREE:
            continue
        var on = t.is_active(f.slots[si].name)
        if not on:
            for k in range(7):
                out[addrs[si].qadr + k] = 0.0
            for k in range(6):
                out[nq + addrs[si].dadr + k] = 0.0
        out.append(1.0 if on else 0.0)
    var op = g.terms[0].op
    var a = g.terms[0].a
    var b = g.terms[0].b
    # (is_site, id) for subject and target — the rule, spelled out here
    var s_site = False
    var s_id = a
    var t_site = False
    var t_id = b
    if op == OP_AT_REGION:
        s_site = True
        t_site = True
        t_id = rsites[b]
    elif op == OP_IN or op == OP_ON:
        t_site = True
        t_id = rsites[b]
    elif op == OP_UPRIGHT:
        t_id = a
    elif op == OP_JOINT:
        s_id = 0
        t_id = 0
    var sv = List[Float64]()
    var tv = List[Float64]()
    for c in range(3):
        sv.append(sxp[s_id * 3 + c] if s_site else xpos[s_id * 3 + c])
        tv.append(sxp[t_id * 3 + c] if t_site else xpos[t_id * 3 + c])
    for c in range(3):
        out.append(sxp[grip * 3 + c])
    for c in range(3):
        out.append(sv[c] - sxp[grip * 3 + c])
    for c in range(3):
        out.append(tv[c] - sv[c])
    return out^


def _family[T: PlacementTable](
    name: String, mut ta: Tally, mut acc: Acc
) raises:
    var f = load_family(String(FAMILY_DIR) + "/" + name + ".family")
    var fmd = parse_model_runtime(scene_path(f))
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=MC, nmesh_verts=verts)
    var m = Model[DT, DynDims](dims)
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") < 0:
                raise e
            verts *= 2
            dims = dims_from_flat(fmd, max_contacts=MC, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var d = Data[DT, DynDims, 1](dims)
    comptime NQ = T.NQ
    comptime NV = T.NV
    comptime NB = T.NBODY
    comptime NS = T.NSITE
    comptime OD = T.NQ + T.NV + T.N_FREE + TASK_GOAL_WORDS
    if (
        dims.get_nq() != NQ or dims.get_nv() != NV or dims.get_nbody() != NB
        or dims.get_nsite() != NS
    ):
        raise Error(name + ": the table's dims are not the scene's — regenerate")
    var grip = _index(fmd.site_names, String(GRIP_NAME))
    var rsites = region_sites(f, fmd.site_names)
    var jt = List[Int]()
    var jqn = List[Int]()
    var jvn = List[Int]()
    for i in range(len(fmd.joints)):
        jt.append(fmd.joints[i].jnt_type)
        jqn.append(fmd.joints[i].nq)
        jvn.append(fmd.joints[i].nv)
    var addrs = free_slot_addresses(f, fmd.joint_names, jt, jqn, jvn)
    var jadr = joint_qpos_addresses(jqn)
    var radii = List[Float64](length=len(f.slots), fill=0.02)
    var tasks = _tasks_of(name)

    comptime L_Q = RowMajor[B, NQ]
    comptime L_V = RowMajor[B, NV]
    comptime L_XP = RowMajor[B, NB * 3]
    comptime L_SP = RowMajor[B, NS * 3]
    comptime L_M = RowMajor[B, METADATA_SIZE]
    comptime L_O = RowMajor[B, OD]

    for ti in range(len(tasks)):
        acc.tasks += 1
        var qs = TensorImpl[DT].alloc(B * NQ)
        var vs = TensorImpl[DT].alloc(B * NV)
        var xs = TensorImpl[DT].alloc(B * NB * 3)
        var ss = TensorImpl[DT].alloc(B * NS * 3)
        var ms = TensorImpl[DT].alloc(B * METADATA_SIZE)
        var os = TensorImpl[DT].alloc(B * OD)
        for i in range(B * METADATA_SIZE):
            ms.data[i] = Scalar[DT](0)
        var oracles = List[Float64]()
        var hosts = List[Float64]()
        var masks = List[Float64]()
        var lane_tasks = List[String]()
        for lane in range(B):
            var tname = tasks[(ti + lane) % len(tasks)]
            lane_tasks.append(tname)
            var t = load_task(TASK_DIR + tname + ".task")
            validate_task_against_family(t, f)
            var g = bind_goal(
                parse_goal(t.goal), f, fmd.body_names, fmd.site_names,
                fmd.joint_names, jadr,
            )
            var tape = encode_goal(g)
            var mask = active_mask(t, f)
            masks.append(mask)

            # ── the host's own reset ──
            for i in range(NQ):
                d.qpos.data[i] = Scalar[DT](0)
            for i in range(len(f.base_qpos)):
                d.qpos.data[i] = Scalar[DT](f.base_qpos[i])
            var jq = joint_init_addresses(t, fmd.joint_names, jqn)
            var jd = joint_init_dof_addresses(t, fmd.joint_names, jvn)
            var jv = sample_joint_inits(t, UInt64(SEED), lane)
            for k in range(len(jv)):
                d.qpos.data[jq[k]] = Scalar[DT](jv[k])
            forward_kinematics["cpu", DT, DynDims, 1](d, m)
            var frames = List[RegionFrame]()
            for r in range(len(f.regions)):
                frames.append(RegionFrame(
                    Float64(d.site_xpos.data[rsites[r] * 3]),
                    Float64(d.site_xpos.data[rsites[r] * 3 + 1]),
                    Float64(d.site_xpos.data[rsites[r] * 3 + 2]),
                ))
            var rep = SampleReport()
            var placed = sample_placements(
                t, f, frames, radii, UInt64(SEED), lane, rep
            )
            var qpos = List[Float64]()
            for i in range(NQ):
                qpos.append(Float64(d.qpos.data[i]))
            var qvel = List[Float64](length=NV, fill=0.0)
            reset_slots(t, f, placed, addrs, qpos, qvel)
            apply_joint_inits(t, jq, jv, qpos, qvel, jd)
            # ⚠ ANTI-VACUITY: a velocity nowhere zero, so zeroing is visible.
            for i in range(NV):
                qvel[i] = 0.001 * Float64(i + 1) + 0.01 * Float64(lane)
            for i in range(NQ):
                d.qpos.data[i] = Scalar[DT](qpos[i])
            for i in range(NV):
                d.qvel.data[i] = Scalar[DT](qvel[i])
            forward_kinematics["cpu", DT, DynDims, 1](d, m)
            var xpos = List[Float64]()
            for i in range(NB * 3):
                xpos.append(Float64(d.xpos.data[i]))
            var sxp = List[Float64]()
            for i in range(NS * 3):
                sxp.append(Float64(d.site_xpos.data[i]))

            # ── the lane's words ──
            for k in range(TAPE_WORDS):
                d.meta.data[META_IDX_TASK_PARAM_0 + k] = Scalar[DT](tape[k])
                ms.data[lane * METADATA_SIZE + META_IDX_TASK_PARAM_0 + k] = (
                    Scalar[DT](tape[k])
                )
            d.meta.data[META_IDX_TASK_ACTIVE] = Scalar[DT](mask)
            ms.data[lane * METADATA_SIZE + META_IDX_TASK_ACTIVE] = Scalar[DT](mask)
            for i in range(NQ):
                qs.data[lane * NQ + i] = Scalar[DT](qpos[i])
            for i in range(NV):
                vs.data[lane * NV + i] = Scalar[DT](qvel[i])
            for i in range(NB * 3):
                xs.data[lane * NB * 3 + i] = Scalar[DT](xpos[i])
            for i in range(NS * 3):
                ss.data[lane * NS * 3 + i] = Scalar[DT](sxp[i])

            # ── host writer, and the oracle ──
            var oh = List[Scalar[DT]]()
            write_task_obs_host[T, DT, DynDims](d, oh)
            for i in range(len(oh)):
                hosts.append(Float64(oh[i]))
            var orc = _oracle_obs(f, t, g, addrs, rsites, grip, qpos, qvel, xpos, sxp)
            for i in range(len(orc)):
                oracles.append(orc[i])
            var op = g.terms[0].op
            if op == OP_IN or op == OP_ON or op == OP_AT_REGION:
                acc.region_goal_lanes += 1
                var ts = rsites[g.terms[0].b]
                var seen = False
                for q in range(len(acc.target_sites)):
                    if acc.target_sites[q] == ts:
                        seen = True
                if not seen:
                    acc.target_sites.append(ts)
            for si in range(len(f.slots)):
                if f.slots[si].kind == SLOT_FREE:
                    if t.is_active(f.slots[si].name):
                        acc.copied += 13
                    else:
                        acc.zeroed += 13
            acc.lanes += 1

        # ── the device writer ──
        var qt = qs.tt["cpu", B, NQ]()
        var vt = vs.tt["cpu", B, NV]()
        var xt = xs.tt["cpu", B, NB * 3]()
        var st = ss.tt["cpu", B, NS * 3]()
        var mt = ms.tt["cpu", B, METADATA_SIZE]()
        var ot = os.tt["cpu", B, OD]()
        for lane in range(B):
            write_task_obs[T, DT, B, NQ, NV, NB, NS * 3, OD](
                qt, vt, xt, st, mt, ot, lane
            )
        if len(hosts) != B * OD or len(oracles) != B * OD:
            raise Error(
                name + ": host obs " + String(len(hosts)) + " / oracle "
                + String(len(oracles)) + " words, expected " + String(B * OD)
            )
        for i in range(B * OD):
            acc.words += 1
            var dv = Float64(os.data[i])
            if dv != hosts[i]:
                acc.dev_host_bad += 1
            if dv != oracles[i]:
                acc.oracle_bad += 1
                if acc.oracle_bad <= 6:
                    print("      ", name, lane_tasks[i // OD], "word",
                          i % OD, ": device", dv, "oracle", oracles[i])

        # ── negative leg: two lanes, two masks ──
        if masks[0] != masks[1]:
            acc.mask_pairs_differ += 1
            for j in range(T.N_FREE):
                var w = NQ + NV + j
                var si = T.free_slot(j)
                var bit0 = ((Int(masks[0]) >> si) & 1) == 1
                var bit1 = ((Int(masks[1]) >> si) & 1) == 1
                var o0 = Float64(os.data[w]) == 1.0
                var o1 = Float64(os.data[OD + w]) == 1.0
                if o0 != bit0 or o1 != bit1:
                    acc.mask_pairs_bad += 1

        # ── the repark ──
        for lane in range(B):
            for si in range(len(f.slots)):
                if f.slots[si].kind != SLOT_FREE:
                    continue
                for k in range(7):
                    qs.data[lane * NQ + addrs[si].qadr + k] = (
                        qs.data[lane * NQ + addrs[si].qadr + k] + Scalar[DT](0.3)
                    )
        var q_before = List[Float64]()
        for i in range(B * NQ):
            q_before.append(Float64(qs.data[i]))
        var v_before = List[Float64]()
        for i in range(B * NV):
            v_before.append(Float64(vs.data[i]))
        for lane in range(B):
            repark_inactive_slots[T, DT, B, NQ, NV](qt, vt, mt, lane)
        for lane in range(B):
            var qtouch = List[Bool](length=NQ, fill=False)
            var vtouch = List[Bool](length=NV, fill=False)
            for si in range(len(f.slots)):
                if f.slots[si].kind != SLOT_FREE:
                    continue
                if ((Int(masks[lane]) >> si) & 1) == 1:
                    continue
                var pp = park_pos(f, si)
                var qa = lane * NQ + addrs[si].qadr
                var want = List[Float64]()
                want.append(pp[0])
                want.append(pp[1])
                want.append(pp[2])
                want.append(1.0)
                want.append(0.0)
                want.append(0.0)
                want.append(0.0)
                for k in range(7):
                    qtouch[addrs[si].qadr + k] = True
                    if Float64(qs.data[qa + k]) != want[k]:
                        acc.repark_bad += 1
                for k in range(6):
                    vtouch[addrs[si].dadr + k] = True
                    if Float64(vs.data[lane * NV + addrs[si].dadr + k]) != 0.0:
                        acc.repark_bad += 1
                acc.pinned += 1
            for i in range(NQ):
                if not qtouch[i]:
                    acc.untouched += 1
                    if Float64(qs.data[lane * NQ + i]) != q_before[lane * NQ + i]:
                        acc.repark_bad += 1
            for i in range(NV):
                if not vtouch[i]:
                    if Float64(vs.data[lane * NV + i]) != v_before[lane * NV + i]:
                        acc.repark_bad += 1


def _read_peaks() raises -> Dict[String, Int]:
    """`contact_budget.kv`: family -> the measured null-action contact peak."""
    var out = Dict[String, Int]()
    var text: String
    with open(String(BUDGET_KV), "r") as fh:
        text = fh.read()
    var lines = text.split("\n")
    for i in range(len(lines)):
        var l = String(String(lines[i]).strip())
        if l.byte_length() == 0 or l.startswith("#"):
            continue
        var eq = l.find("=")
        var comma = l.find(",")
        out[String(l[byte = 0 : eq])] = Int(String(l[byte = eq + 1 : comma]))
    return out^


def _statics[T: PlacementTable, M: ModelDefLike, C: Phyics3dEnvConfig](
    name: String, peaks: Dict[String, Int], mut ta: Tally
) raises:
    """Section 1 on one family: its comptime numbers against its files."""
    var f = load_family(String(FAMILY_DIR) + "/" + name + ".family")
    comptime WANT = T.NQ + T.NV + T.N_FREE + TASK_GOAL_WORDS
    ta.quiet(M.OBS_DIM == WANT,
             name + ": OBS_DIM " + String(M.OBS_DIM) + " != NQ + NV + N_FREE"
             " + TASK_GOAL_WORDS = " + String(WANT))
    ta.quiet(
        T.NQ == M.NQ and T.NV == M.NV and T.NBODY == M.NBODY
        and T.NSITE == M.NSITE,
        name + ": the placement table and the model def disagree on"
        " nq/nv/nbody/nsite",
    )
    ta.quiet(M.CONE_TYPE == ConeType.ELLIPTIC,
             name + ": the cone is not ELLIPTIC (robosuite's base.xml)")
    # ⚠ AGAINST THE PARSED SCENE, not the generated dims the def reads — a
    # def that forgets `max_condim=` compiles at the default 3 and drops the
    # finger pads' torsional row on the batch only (the CPU leg builds at 6).
    var fmd = parse_model_runtime(scene_path(f))
    ta.quiet(M.MAX_CONDIM == fmd.max_condim,
             name + ": MAX_CONDIM " + String(M.MAX_CONDIM)
             + " != the scene's " + String(fmd.max_condim))
    ta.quiet(M.ACTION_DIM == 7, name + ": ACTION_DIM is not OSC_POSE's 7")
    if name not in peaks:
        ta.quiet(False, name + ": no measured contact peak in " + BUDGET_KV)
    else:
        ta.quiet(M.MAX_CONTACTS > peaks[name],
                 name + ": max_contacts " + String(M.MAX_CONTACTS)
                 + " is not above the measured peak " + String(peaks[name]))
    # ⚠ THE ONE CONFIG RESTATES THREE NUMBERS; each family's files decide them.
    ta.quiet(C.get_timestep() == M.TIMESTEP,
             name + ": get_timestep " + String(C.get_timestep())
             + " != the generated dims' " + String(M.TIMESTEP))
    var fs = Float64(C.FRAME_SKIP) * M.TIMESTEP * Float64(f.control_freq)
    ta.quiet(abs(fs - 1.0) < 1.0e-12,
             name + ": FRAME_SKIP " + String(C.FRAME_SKIP) + " x timestep != 1 /"
             " control_freq " + String(f.control_freq))
    ta.quiet(C.MAX_STEPS == f.horizon,
             name + ": MAX_STEPS " + String(C.MAX_STEPS) + " != horizon "
             + String(f.horizon))


def _dispatch(
    name: String, section: Int, peaks: Dict[String, Int], mut ta: Tally,
    mut acc: Acc,
) raises:
    """One family, one section — the only place the 23 type triples are named."""
    if name == "libero_goal":
        if section == 1:
            _statics[LiberoGoalPlacement, LiberoGoalModel, LiberoOscConfig[LiberoGoalPlacement]](
                name, peaks, ta
            )
        elif section == 2:
            _family[LiberoGoalPlacement](name, ta, acc)
        else:
            _config_hooks[LiberoGoalPlacement, LiberoGoalModel, LiberoOscConfig[LiberoGoalPlacement]](
                name, ta, acc
            )
    elif name == "libero_kitchen_scene1":
        if section == 1:
            _statics[LiberoKitchenScene1Placement, LiberoKitchenScene1Model, LiberoOscConfig[LiberoKitchenScene1Placement]](
                name, peaks, ta
            )
        elif section == 2:
            _family[LiberoKitchenScene1Placement](name, ta, acc)
        else:
            _config_hooks[LiberoKitchenScene1Placement, LiberoKitchenScene1Model, LiberoOscConfig[LiberoKitchenScene1Placement]](
                name, ta, acc
            )
    elif name == "libero_kitchen_scene10":
        if section == 1:
            _statics[LiberoKitchenScene10Placement, LiberoKitchenScene10Model, LiberoOscConfig[LiberoKitchenScene10Placement]](
                name, peaks, ta
            )
        elif section == 2:
            _family[LiberoKitchenScene10Placement](name, ta, acc)
        else:
            _config_hooks[LiberoKitchenScene10Placement, LiberoKitchenScene10Model, LiberoOscConfig[LiberoKitchenScene10Placement]](
                name, ta, acc
            )
    elif name == "libero_kitchen_scene2":
        if section == 1:
            _statics[LiberoKitchenScene2Placement, LiberoKitchenScene2Model, LiberoOscConfig[LiberoKitchenScene2Placement]](
                name, peaks, ta
            )
        elif section == 2:
            _family[LiberoKitchenScene2Placement](name, ta, acc)
        else:
            _config_hooks[LiberoKitchenScene2Placement, LiberoKitchenScene2Model, LiberoOscConfig[LiberoKitchenScene2Placement]](
                name, ta, acc
            )
    elif name == "libero_kitchen_scene3":
        if section == 1:
            _statics[LiberoKitchenScene3Placement, LiberoKitchenScene3Model, LiberoOscConfig[LiberoKitchenScene3Placement]](
                name, peaks, ta
            )
        elif section == 2:
            _family[LiberoKitchenScene3Placement](name, ta, acc)
        else:
            _config_hooks[LiberoKitchenScene3Placement, LiberoKitchenScene3Model, LiberoOscConfig[LiberoKitchenScene3Placement]](
                name, ta, acc
            )
    elif name == "libero_kitchen_scene4":
        if section == 1:
            _statics[LiberoKitchenScene4Placement, LiberoKitchenScene4Model, LiberoOscConfig[LiberoKitchenScene4Placement]](
                name, peaks, ta
            )
        elif section == 2:
            _family[LiberoKitchenScene4Placement](name, ta, acc)
        else:
            _config_hooks[LiberoKitchenScene4Placement, LiberoKitchenScene4Model, LiberoOscConfig[LiberoKitchenScene4Placement]](
                name, ta, acc
            )
    elif name == "libero_kitchen_scene5":
        if section == 1:
            _statics[LiberoKitchenScene5Placement, LiberoKitchenScene5Model, LiberoOscConfig[LiberoKitchenScene5Placement]](
                name, peaks, ta
            )
        elif section == 2:
            _family[LiberoKitchenScene5Placement](name, ta, acc)
        else:
            _config_hooks[LiberoKitchenScene5Placement, LiberoKitchenScene5Model, LiberoOscConfig[LiberoKitchenScene5Placement]](
                name, ta, acc
            )
    elif name == "libero_kitchen_scene6":
        if section == 1:
            _statics[LiberoKitchenScene6Placement, LiberoKitchenScene6Model, LiberoOscConfig[LiberoKitchenScene6Placement]](
                name, peaks, ta
            )
        elif section == 2:
            _family[LiberoKitchenScene6Placement](name, ta, acc)
        else:
            _config_hooks[LiberoKitchenScene6Placement, LiberoKitchenScene6Model, LiberoOscConfig[LiberoKitchenScene6Placement]](
                name, ta, acc
            )
    elif name == "libero_kitchen_scene7":
        if section == 1:
            _statics[LiberoKitchenScene7Placement, LiberoKitchenScene7Model, LiberoOscConfig[LiberoKitchenScene7Placement]](
                name, peaks, ta
            )
        elif section == 2:
            _family[LiberoKitchenScene7Placement](name, ta, acc)
        else:
            _config_hooks[LiberoKitchenScene7Placement, LiberoKitchenScene7Model, LiberoOscConfig[LiberoKitchenScene7Placement]](
                name, ta, acc
            )
    elif name == "libero_kitchen_scene8":
        if section == 1:
            _statics[LiberoKitchenScene8Placement, LiberoKitchenScene8Model, LiberoOscConfig[LiberoKitchenScene8Placement]](
                name, peaks, ta
            )
        elif section == 2:
            _family[LiberoKitchenScene8Placement](name, ta, acc)
        else:
            _config_hooks[LiberoKitchenScene8Placement, LiberoKitchenScene8Model, LiberoOscConfig[LiberoKitchenScene8Placement]](
                name, ta, acc
            )
    elif name == "libero_kitchen_scene9":
        if section == 1:
            _statics[LiberoKitchenScene9Placement, LiberoKitchenScene9Model, LiberoOscConfig[LiberoKitchenScene9Placement]](
                name, peaks, ta
            )
        elif section == 2:
            _family[LiberoKitchenScene9Placement](name, ta, acc)
        else:
            _config_hooks[LiberoKitchenScene9Placement, LiberoKitchenScene9Model, LiberoOscConfig[LiberoKitchenScene9Placement]](
                name, ta, acc
            )
    elif name == "libero_living_room_scene1":
        if section == 1:
            _statics[LiberoLivingRoomScene1Placement, LiberoLivingRoomScene1Model, LiberoOscConfig[LiberoLivingRoomScene1Placement]](
                name, peaks, ta
            )
        elif section == 2:
            _family[LiberoLivingRoomScene1Placement](name, ta, acc)
        else:
            _config_hooks[LiberoLivingRoomScene1Placement, LiberoLivingRoomScene1Model, LiberoOscConfig[LiberoLivingRoomScene1Placement]](
                name, ta, acc
            )
    elif name == "libero_living_room_scene2":
        if section == 1:
            _statics[LiberoLivingRoomScene2Placement, LiberoLivingRoomScene2Model, LiberoOscConfig[LiberoLivingRoomScene2Placement]](
                name, peaks, ta
            )
        elif section == 2:
            _family[LiberoLivingRoomScene2Placement](name, ta, acc)
        else:
            _config_hooks[LiberoLivingRoomScene2Placement, LiberoLivingRoomScene2Model, LiberoOscConfig[LiberoLivingRoomScene2Placement]](
                name, ta, acc
            )
    elif name == "libero_living_room_scene3":
        if section == 1:
            _statics[LiberoLivingRoomScene3Placement, LiberoLivingRoomScene3Model, LiberoOscConfig[LiberoLivingRoomScene3Placement]](
                name, peaks, ta
            )
        elif section == 2:
            _family[LiberoLivingRoomScene3Placement](name, ta, acc)
        else:
            _config_hooks[LiberoLivingRoomScene3Placement, LiberoLivingRoomScene3Model, LiberoOscConfig[LiberoLivingRoomScene3Placement]](
                name, ta, acc
            )
    elif name == "libero_living_room_scene4":
        if section == 1:
            _statics[LiberoLivingRoomScene4Placement, LiberoLivingRoomScene4Model, LiberoOscConfig[LiberoLivingRoomScene4Placement]](
                name, peaks, ta
            )
        elif section == 2:
            _family[LiberoLivingRoomScene4Placement](name, ta, acc)
        else:
            _config_hooks[LiberoLivingRoomScene4Placement, LiberoLivingRoomScene4Model, LiberoOscConfig[LiberoLivingRoomScene4Placement]](
                name, ta, acc
            )
    elif name == "libero_living_room_scene5":
        if section == 1:
            _statics[LiberoLivingRoomScene5Placement, LiberoLivingRoomScene5Model, LiberoOscConfig[LiberoLivingRoomScene5Placement]](
                name, peaks, ta
            )
        elif section == 2:
            _family[LiberoLivingRoomScene5Placement](name, ta, acc)
        else:
            _config_hooks[LiberoLivingRoomScene5Placement, LiberoLivingRoomScene5Model, LiberoOscConfig[LiberoLivingRoomScene5Placement]](
                name, ta, acc
            )
    elif name == "libero_living_room_scene6":
        if section == 1:
            _statics[LiberoLivingRoomScene6Placement, LiberoLivingRoomScene6Model, LiberoOscConfig[LiberoLivingRoomScene6Placement]](
                name, peaks, ta
            )
        elif section == 2:
            _family[LiberoLivingRoomScene6Placement](name, ta, acc)
        else:
            _config_hooks[LiberoLivingRoomScene6Placement, LiberoLivingRoomScene6Model, LiberoOscConfig[LiberoLivingRoomScene6Placement]](
                name, ta, acc
            )
    elif name == "libero_object":
        if section == 1:
            _statics[LiberoObjectPlacement, LiberoObjectModel, LiberoOscConfig[LiberoObjectPlacement]](
                name, peaks, ta
            )
        elif section == 2:
            _family[LiberoObjectPlacement](name, ta, acc)
        else:
            _config_hooks[LiberoObjectPlacement, LiberoObjectModel, LiberoOscConfig[LiberoObjectPlacement]](
                name, ta, acc
            )
    elif name == "libero_spatial":
        if section == 1:
            _statics[LiberoSpatialPlacement, LiberoSpatialModel, LiberoOscConfig[LiberoSpatialPlacement]](
                name, peaks, ta
            )
        elif section == 2:
            _family[LiberoSpatialPlacement](name, ta, acc)
        else:
            _config_hooks[LiberoSpatialPlacement, LiberoSpatialModel, LiberoOscConfig[LiberoSpatialPlacement]](
                name, ta, acc
            )
    elif name == "libero_study_scene1":
        if section == 1:
            _statics[LiberoStudyScene1Placement, LiberoStudyScene1Model, LiberoOscConfig[LiberoStudyScene1Placement]](
                name, peaks, ta
            )
        elif section == 2:
            _family[LiberoStudyScene1Placement](name, ta, acc)
        else:
            _config_hooks[LiberoStudyScene1Placement, LiberoStudyScene1Model, LiberoOscConfig[LiberoStudyScene1Placement]](
                name, ta, acc
            )
    elif name == "libero_study_scene2":
        if section == 1:
            _statics[LiberoStudyScene2Placement, LiberoStudyScene2Model, LiberoOscConfig[LiberoStudyScene2Placement]](
                name, peaks, ta
            )
        elif section == 2:
            _family[LiberoStudyScene2Placement](name, ta, acc)
        else:
            _config_hooks[LiberoStudyScene2Placement, LiberoStudyScene2Model, LiberoOscConfig[LiberoStudyScene2Placement]](
                name, ta, acc
            )
    elif name == "libero_study_scene3":
        if section == 1:
            _statics[LiberoStudyScene3Placement, LiberoStudyScene3Model, LiberoOscConfig[LiberoStudyScene3Placement]](
                name, peaks, ta
            )
        elif section == 2:
            _family[LiberoStudyScene3Placement](name, ta, acc)
        else:
            _config_hooks[LiberoStudyScene3Placement, LiberoStudyScene3Model, LiberoOscConfig[LiberoStudyScene3Placement]](
                name, ta, acc
            )
    elif name == "libero_study_scene4":
        if section == 1:
            _statics[LiberoStudyScene4Placement, LiberoStudyScene4Model, LiberoOscConfig[LiberoStudyScene4Placement]](
                name, peaks, ta
            )
        elif section == 2:
            _family[LiberoStudyScene4Placement](name, ta, acc)
        else:
            _config_hooks[LiberoStudyScene4Placement, LiberoStudyScene4Model, LiberoOscConfig[LiberoStudyScene4Placement]](
                name, ta, acc
            )
    else:
        ta.check(False, name + ": no table or model imported for this family")


def main() raises:
    print("=== the task hooks on every LIBERO family ===")
    var ta = Tally()

    # ── 1. every family's comptime facts ──────────────────────────────────
    print("--- 1. every family: widths, dims, cone, budget, the config's constants ---")
    var fams = _libero_families()
    ta.check(len(fams) == N_LIBERO_FAMILIES,
             String(len(fams)) + " libero*.family files, and this gate imports "
             + String(N_LIBERO_FAMILIES) + " tables and models")
    var peaks = _read_peaks()
    var checks_before = ta.checks
    var fails_before = ta.failures
    var no_acc = Acc()
    for i in range(len(fams)):
        _dispatch(fams[i], 1, peaks, ta, no_acc)
    ta.check(ta.failures == fails_before,
             String(ta.checks - checks_before) + " per-family facts over "
             + String(len(fams)) + " families")

    # ── 2. every LIBERO task ──────────────────────────────────────────────
    print()
    print("--- 2. every LIBERO task: device writer == host writer == oracle ---")
    var acc = Acc()
    for i in range(len(fams)):
        _dispatch(fams[i], 2, peaks, ta, acc)
    print("      tasks", acc.tasks, " lanes", acc.lanes, " words", acc.words)
    print("      device != host", acc.dev_host_bad, "  device != oracle",
          acc.oracle_bad)
    print("      slot words copied", acc.copied, " zeroed", acc.zeroed,
          "  lane pairs with different masks", acc.mask_pairs_differ)
    print("      In/On/AtRegion lanes", acc.region_goal_lanes,
          " distinct target sites", len(acc.target_sites))
    print("      slots pinned", acc.pinned, " other words checked untouched",
          acc.untouched, " repark faults", acc.repark_bad)
    ta.check(acc.tasks == N_LIBERO_TASKS,
             String(acc.tasks) + " LIBERO tasks (the importer's "
             + String(N_LIBERO_TASKS) + ")")
    if acc.words == 0:
        raise Error("task hooks: no observation word was compared")
    ta.check(acc.dev_host_bad == 0,
             "the device writer and the host writer agree word for word")
    ta.check(acc.oracle_bad == 0,
             "both equal the oracle built from the scene and the bound goal")
    ta.check(acc.copied > 0 and acc.zeroed > 0,
             "the corpus exercises active AND inactive slots")
    ta.check(acc.mask_pairs_differ > 0 and acc.mask_pairs_bad == 0,
             String(acc.mask_pairs_differ) + " lane pairs run different active"
             " sets and each lane reads ITS OWN mask")
    # ⚠ A COVERAGE FACT ABOUT THE CORPUS, NOT A DETECTOR. A writer resolving
    # every region to one site fails the ORACLE check above (mutated and
    # measured: that check is the one that fails); this says the corpus gives
    # it more than one site to get wrong.
    ta.check(len(acc.target_sites) >= 2,
             "the corpus's In/On/AtRegion targets span "
             + String(len(acc.target_sites)) + " sites, so a one-site writer"
             " has something to get wrong")
    ta.check(acc.pinned > 0 and acc.repark_bad == 0,
             "the repark pins every inactive slot at park_pos and nothing else")

    # ── 3. every family's config hooks ────────────────────────────────────
    print()
    print("--- 3. LiberoOscConfig[P]'s hooks, not the helpers, on every family ---")
    var hacc = Acc()
    checks_before = ta.checks
    fails_before = ta.failures
    for i in range(len(fams)):
        _dispatch(fams[i], 3, peaks, ta, hacc)
    print("      families", hacc.tasks, " reward tapes true", hacc.copied,
          " false", hacc.zeroed)
    ta.check(hacc.tasks == N_LIBERO_FAMILIES and ta.failures == fails_before,
             String(ta.checks - checks_before) + " hook checks over "
             + String(hacc.tasks) + " families: obs gpu == helper, obs cpu =="
             " obs gpu, repark == helper, reward == GOAL_HELD == eval_tape_gpu")
    ta.check(hacc.copied >= N_LIBERO_FAMILIES and hacc.zeroed >= N_LIBERO_FAMILIES,
             "every family's reward hook saw a tape that holds AND one that"
             " does not")

    print()
    print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
    if ta.failures != 0:
        raise Error(
            "task hooks: " + String(ta.failures) + " of " + String(ta.checks)
            + " check(s) failed"
        )
    print("=== PASS ===")


def _config_hooks[T: PlacementTable, M: ModelDefLike, C: Phyics3dEnvConfig](
    name: String, mut ta: Tally, mut acc: Acc
) raises:
    """The config's obs hooks, repark and reward on one state of `name`.

    `acc.tasks` counts families, `acc.copied` / `acc.zeroed` the reward tapes
    that held / did not — reused fields, printed under those names by `main`."""
    comptime NQ = M.NQ
    comptime NV = M.NV
    comptime NB = M.NBODY
    comptime NS = M.NSITE
    comptime NG = M.NGEOM
    comptime NA = M.NACT
    comptime OD = M.OBS_DIM
    comptime MCG = M.MAX_CONTACTS
    acc.tasks += 1
    var f = load_family(String(FAMILY_DIR) + "/" + name + ".family")
    var fmd = parse_model_runtime(scene_path(f))
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=MCG, nmesh_verts=verts)
    var m = Model[DT, DynDims](dims)
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") < 0:
                raise e
            verts *= 2
            dims = dims_from_flat(fmd, max_contacts=MCG, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var d = Data[DT, DynDims, 1](dims)
    var jqn = List[Int]()
    for i in range(len(fmd.joints)):
        jqn.append(fmd.joints[i].nq)
    var jadr = joint_qpos_addresses(jqn)
    var tasks = _tasks_of(name)
    var t = load_task(TASK_DIR + tasks[0] + ".task")
    validate_task_against_family(t, f)

    # a state: the scene at base_qpos with every free slot at its park pose,
    # two lanes with different masks so the repark has something to pin
    for i in range(NQ):
        d.qpos.data[i] = Scalar[DT](0)
    for i in range(len(f.base_qpos)):
        d.qpos.data[i] = Scalar[DT](f.base_qpos[i])
    for j in range(T.N_FREE):
        d.qpos.data[T.free_qadr(j)] = T.free_park_x[DT](j) + Scalar[DT](0.1)
        d.qpos.data[T.free_qadr(j) + 1] = T.free_park_y[DT](j)
        d.qpos.data[T.free_qadr(j) + 2] = Scalar[DT](0.95)
        d.qpos.data[T.free_qadr(j) + 3] = Scalar[DT](1)
    for i in range(NV):
        d.qvel.data[i] = Scalar[DT](0.002 * Float64(i + 1))
    forward_kinematics["cpu", DT, DynDims, 1](d, m)

    var g_task = bind_goal(
        parse_goal(t.goal), f, fmd.body_names, fmd.site_names, fmd.joint_names,
        jadr,
    )
    var subj = f.slots[T.free_slot(0)].name
    var g_yes = bind_goal(
        parse_goal("Upright(" + subj + ", 0.1)"), f, fmd.body_names,
        fmd.site_names, fmd.joint_names, jadr,
    )
    var g_no = bind_goal(
        parse_goal("Not(Upright(" + subj + ", 0.1))"), f, fmd.body_names,
        fmd.site_names, fmd.joint_names, jadr,
    )

    comptime L_Q = RowMajor[B, NQ]
    comptime L_V = RowMajor[B, NV]
    comptime L_B3 = RowMajor[B, NB * 3]
    comptime L_B4 = RowMajor[B, NB * 4]
    comptime L_B6 = RowMajor[B, NB * 6]
    comptime L_S3 = RowMajor[B, NS * 3]
    comptime L_M = RowMajor[B, METADATA_SIZE]
    comptime L_O = RowMajor[B, OD]
    comptime L_CON = RowMajor[B, MCG * CONTACT_SIZE]
    comptime L_BR = RowMajor[NB, MODEL_BODY_SIZE]
    comptime L_SR = RowMajor[NS, MODEL_SITE_SIZE]
    comptime L_GR = RowMajor[NG, MODEL_GEOM_SIZE]
    comptime L_CUR = RowMajor[1, MODEL_CURRICULUM_SIZE]
    comptime L_ACT = RowMajor[B, 7]
    comptime L_NA = RowMajor[B, NA]

    var qs = TensorImpl[DT].alloc(B * NQ)
    var vs = TensorImpl[DT].alloc(B * NV)
    var xp = TensorImpl[DT].alloc(B * NB * 3)
    var xip = TensorImpl[DT].alloc(B * NB * 3)
    var xq = TensorImpl[DT].alloc(B * NB * 4)
    var xv = TensorImpl[DT].alloc(B * NB * 3)
    var xav = TensorImpl[DT].alloc(B * NB * 3)
    var stc = TensorImpl[DT].alloc(B * NB * 3)
    var cv = TensorImpl[DT].alloc(B * NB * 6)
    var ca = TensorImpl[DT].alloc(B * NB * 6)
    var cfi = TensorImpl[DT].alloc(B * NB * 6)
    var cfe = TensorImpl[DT].alloc(B * NB * 6)
    var sp = TensorImpl[DT].alloc(B * NS * 3)
    var spa = TensorImpl[DT].alloc(B * NS * 3)
    var xqa = TensorImpl[DT].alloc(B * NB * 4)
    var con = TensorImpl[DT].alloc(B * MCG * CONTACT_SIZE)
    var ms = TensorImpl[DT].alloc(B * METADATA_SIZE)
    var o_cfg = TensorImpl[DT].alloc(B * OD)
    var o_hlp = TensorImpl[DT].alloc(B * OD)
    var br = TensorImpl[DT].alloc(NB * MODEL_BODY_SIZE)
    var sr = TensorImpl[DT].alloc(NS * MODEL_SITE_SIZE)
    var gr = TensorImpl[DT].alloc(NG * MODEL_GEOM_SIZE)
    var cur = TensorImpl[DT].alloc(MODEL_CURRICULUM_SIZE)
    var acts = TensorImpl[DT].alloc(B * 7)
    var actv = TensorImpl[DT].alloc(B * NA)
    for i in range(NB * MODEL_BODY_SIZE):
        br.data[i] = m.bodies.data[i]
    for i in range(NS * MODEL_SITE_SIZE):
        sr.data[i] = m.sites.data[i]
    for i in range(NG * MODEL_GEOM_SIZE):
        gr.data[i] = m.geoms.data[i]
    var rsites = region_sites(f, fmd.site_names)
    var rcon = region_contact_bodies(f, fmd.body_names)
    var cw = region_table_words(f, rsites, rcon)
    for i in range(MODEL_CURRICULUM_SIZE):
        cur.data[i] = Scalar[DT](cw[i])
    for i in range(B * MCG * CONTACT_SIZE):
        con.data[i] = Scalar[DT](0)
    for i in range(B * 7):
        acts.data[i] = Scalar[DT](0)
    for i in range(B * NA):
        actv.data[i] = Scalar[DT](0)
    for i in range(B * METADATA_SIZE):
        ms.data[i] = Scalar[DT](0)
    var full_mask = active_mask(t, f)
    # lane 1 runs with its FIRST free slot switched off
    var part_mask = full_mask - Float64(1 << T.free_slot(0))
    var tape = encode_goal(g_task)
    for lane in range(B):
        for i in range(NQ):
            qs.data[lane * NQ + i] = d.qpos.data[i]
        for i in range(NV):
            vs.data[lane * NV + i] = d.qvel.data[i]
        for i in range(NB * 3):
            xp.data[lane * NB * 3 + i] = d.xpos.data[i]
            xip.data[lane * NB * 3 + i] = d.xipos.data[i]
            xv.data[lane * NB * 3 + i] = Scalar[DT](0)
            xav.data[lane * NB * 3 + i] = Scalar[DT](0)
            stc.data[lane * NB * 3 + i] = Scalar[DT](0)
        for i in range(NB * 4):
            xq.data[lane * NB * 4 + i] = d.xquat.data[i]
            xqa.data[lane * NB * 4 + i] = d.xquat.data[i]
        for i in range(NB * 6):
            cv.data[lane * NB * 6 + i] = Scalar[DT](0)
            ca.data[lane * NB * 6 + i] = Scalar[DT](0)
            cfi.data[lane * NB * 6 + i] = Scalar[DT](0)
            cfe.data[lane * NB * 6 + i] = Scalar[DT](0)
        for i in range(NS * 3):
            sp.data[lane * NS * 3 + i] = d.site_xpos.data[i]
            spa.data[lane * NS * 3 + i] = d.site_xpos.data[i]
        for k in range(TAPE_WORDS):
            ms.data[lane * METADATA_SIZE + META_IDX_TASK_PARAM_0 + k] = (
                Scalar[DT](tape[k])
            )
        ms.data[lane * METADATA_SIZE + META_IDX_TASK_ACTIVE] = Scalar[DT](
            full_mask if lane == 0 else part_mask
        )

    # observation: the config hook vs the helper
    for lane in range(B):
        _ = C.custom_extract_obs_gpu[
            DT, B, NQ, NV, NB, OD, NS * 3, MCG, NS, NG, NA
        ](
            qs.tt["cpu", B, NQ](), vs.tt["cpu", B, NV](), xp.tt["cpu", B, NB * 3](),
            xq.tt["cpu", B, NB * 4](), xv.tt["cpu", B, NB * 3](), br.tt["cpu", NB, MODEL_BODY_SIZE](),
            sp.tt["cpu", B, NS * 3](), con.tt["cpu", B, MCG * CONTACT_SIZE](), sr.tt["cpu", NS, MODEL_SITE_SIZE](),
            gr.tt["cpu", NG, MODEL_GEOM_SIZE](), ms.tt["cpu", B, METADATA_SIZE](), o_cfg.tt["cpu", B, OD](),
            xip.tt["cpu", B, NB * 3](), xav.tt["cpu", B, NB * 3](), cv.tt["cpu", B, NB * 6](),
            ca.tt["cpu", B, NB * 6](), cfi.tt["cpu", B, NB * 6](), stc.tt["cpu", B, NB * 3](),
            spa.tt["cpu", B, NS * 3](), xqa.tt["cpu", B, NB * 4](), actv.tt["cpu", B, NA](),
            lane,
        )
        write_task_obs[T, DT, B, NQ, NV, NB, NS * 3, OD](
            qs.tt["cpu", B, NQ](), vs.tt["cpu", B, NV](), xp.tt["cpu", B, NB * 3](),
            sp.tt["cpu", B, NS * 3](), ms.tt["cpu", B, METADATA_SIZE](), o_hlp.tt["cpu", B, OD](),
            lane,
        )
    var obs_same = True
    var obs_differ_lanes = False
    for i in range(B * OD):
        if Float64(o_cfg.data[i]) != Float64(o_hlp.data[i]):
            obs_same = False
    for i in range(OD):
        if Float64(o_cfg.data[i]) != Float64(o_cfg.data[OD + i]):
            obs_differ_lanes = True
    ta.quiet(obs_same,
             name + ": custom_extract_obs_gpu != write_task_obs on some lane")
    ta.quiet(obs_differ_lanes,
             name + ": the lane with a slot switched off reads the same")

    # the CPU hook, per lane, vs the device hook's row
    var cpu_same = True
    for lane in range(B):
        for k in range(TAPE_WORDS):
            d.meta.data[META_IDX_TASK_PARAM_0 + k] = Scalar[DT](tape[k])
        d.meta.data[META_IDX_TASK_ACTIVE] = Scalar[DT](
            full_mask if lane == 0 else part_mask
        )
        var oc = List[Scalar[DT]]()
        var e = List[Scalar[DT]]()
        var ok = C.custom_extract_obs_cpu[DT, DynDims](
            d, e, e, e, e, e, oc
        )
        if not ok or len(oc) != OD:
            cpu_same = False
            print("      cpu hook wrote", len(oc), "words, expected", OD)
            continue
        for i in range(OD):
            if Float64(oc[i]) != Float64(o_cfg.data[lane * OD + i]):
                cpu_same = False
    ta.quiet(cpu_same,
             name + ": custom_extract_obs_cpu != custom_extract_obs_gpu")

    # the repark: the config hook vs the helper
    var qs2 = TensorImpl[DT].alloc(B * NQ)
    var vs2 = TensorImpl[DT].alloc(B * NV)
    for i in range(B * NQ):
        qs2.data[i] = qs.data[i]
    for i in range(B * NV):
        vs2.data[i] = vs.data[i]
    for lane in range(B):
        C.pre_step_full_gpu[DT, B, NQ, NV](
            qs.tt["cpu", B, NQ](), vs.tt["cpu", B, NV](), ms.tt["cpu", B, METADATA_SIZE](), lane
        )
        repark_inactive_slots[T, DT, B, NQ, NV](
            qs2.tt["cpu", B, NQ](), vs2.tt["cpu", B, NV](), ms.tt["cpu", B, METADATA_SIZE](), lane
        )
    var rp_same = True
    for i in range(B * NQ):
        if Float64(qs.data[i]) != Float64(qs2.data[i]):
            rp_same = False
    for i in range(B * NV):
        if Float64(vs.data[i]) != Float64(vs2.data[i]):
            rp_same = False
    var pinned = Float64(qs.data[NQ + T.free_qadr(0) + 2]) == Float64(
        T.free_park_z[DT](0)
    )
    ta.quiet(rp_same and pinned,
             name + ": pre_step_full_gpu != repark_inactive_slots, or lane 1's"
             " switched-off slot was not pinned at its park height")

    # the reward: GOAL_HELD == reward == the kernel, on three tapes
    var n_true = 0
    var n_false = 0
    var reward_ok = True
    for which in range(3):
        var tp = encode_goal(g_yes) if which == 0 else (
            encode_goal(g_no) if which == 1 else encode_goal(g_task)
        )
        for k in range(TAPE_WORDS):
            ms.data[META_IDX_TASK_PARAM_0 + k] = Scalar[DT](tp[k])
        ms.data[META_IDX_GOAL_HELD] = Scalar[DT](0.5)
        var res = C.compute_reward_and_done_gpu[
            DT, B, NQ, NV, NB, 7, NS * 3, MCG, NS, NG, NA
        ](
            qs.tt["cpu", B, NQ](), vs.tt["cpu", B, NV](), xp.tt["cpu", B, NB * 3](),
            xip.tt["cpu", B, NB * 3](), xq.tt["cpu", B, NB * 4](), xv.tt["cpu", B, NB * 3](),
            br.tt["cpu", NB, MODEL_BODY_SIZE](), sp.tt["cpu", B, NS * 3](), con.tt["cpu", B, MCG * CONTACT_SIZE](),
            sr.tt["cpu", NS, MODEL_SITE_SIZE](), gr.tt["cpu", NG, MODEL_GEOM_SIZE](), cfe.tt["cpu", B, NB * 6](),
            cv.tt["cpu", B, NB * 6](), ms.tt["cpu", B, METADATA_SIZE](), cur.tt["cpu", 1, MODEL_CURRICULUM_SIZE](),
            acts.tt["cpu", B, 7](), xav.tt["cpu", B, NB * 3](), ca.tt["cpu", B, NB * 6](),
            cfi.tt["cpu", B, NB * 6](), stc.tt["cpu", B, NB * 3](), spa.tt["cpu", B, NS * 3](),
            xqa.tt["cpu", B, NB * 4](), actv.tt["cpu", B, NA](), 0, 0, 25,
            Scalar[DT](0.002),
        )
        var kern = eval_tape_gpu[DT, B, NB, NS * 3, NQ, NS, MCG](
            ms.tt["cpu", B, METADATA_SIZE](), cur.tt["cpu", 1, MODEL_CURRICULUM_SIZE](), xp.tt["cpu", B, NB * 3](),
            xq.tt["cpu", B, NB * 4](), sp.tt["cpu", B, NS * 3](), qs.tt["cpu", B, NQ](),
            sr.tt["cpu", NS, MODEL_SITE_SIZE](), br.tt["cpu", NB, MODEL_BODY_SIZE](), con.tt["cpu", B, MCG * CONTACT_SIZE](), 0,
        )
        var r = Float64(res[0])
        var gh = Float64(ms.data[META_IDX_GOAL_HELD])
        var want = 1.0 if kern else 0.0
        if r != want or gh != want or res[1] != kern:
            reward_ok = False
            print("      ", name, "tape", which, ": reward", r, " GOAL_HELD", gh,
                  " kernel", kern, " done", res[1])
        if kern:
            n_true += 1
        else:
            n_false += 1
    ta.quiet(reward_ok,
             name + ": compute_reward_and_done_gpu: reward, GOAL_HELD and"
             " eval_tape_gpu disagree")
    ta.quiet(n_true >= 1 and n_false >= 1,
             name + ": no tape that holds, or none that does not")
    acc.copied += n_true
    acc.zeroed += n_false

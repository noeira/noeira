"""Watch a TRAINED policy run a `.task` — the task layer, with a driver.

    pixi run build-imgui                                          # ONCE
    pixi run mojo run -I . examples/tasks/task_policy_viewer.mojo
    pixi run mojo run -I . examples/tasks/task_policy_viewer.mojo so101_lift_brick
    pixi run mojo run -I . examples/tasks/task_policy_viewer.mojo <task> --check
    pixi run mojo run -I . examples/tasks/task_policy_viewer.mojo so101_lift_brick --ckpt <run_id>

`task_viewer.mojo` with an actor in it. That file's header says outright what
it cannot show — *"the arm does not move, and that is the point of the
default"*, *"the GOAL READOUT tells you whether the goal is SATISFIABLE and
correctly wired, not whether anything is solving it"*. This is the other half:
the same family, the same sampler, the same goal, driven by the SAC checkpoint
trained on that `.task`.

⚠⚠ **THE TASK PICKER SWITCHES THE CHECKPOINT TOO.** One policy per task —
the newest run of that task that
`examples/tasks/sac_task_gpu.mojo` wrote (`<project runs>/<date>_sac-<task>_<hash>/
checkpoints/last.ckpt`), or the pre-run-layout `checkpoints/sac_task_<task>.ckpt`.
`--ckpt <run_id|path>` pins the STARTING task's policy to that run's
`checkpoints/last.ckpt` (or that file) instead of the newest. Selecting a task reloads the `.task`
AND the weights trained on it, because a `gather` policy driving `lift` is a
demo of nothing. The model is NOT rebuilt: every task in a family instantiates
every slot, so `nq`/`nv`/`ngeom` are constant and the switch is a data reload.

⚠⚠ **IT OPENS ON THE FREE CAMERA.** `so_arm101.xml` ships exactly one camera —
`wrist_cam`, bolted to the wrist — and the renderer opens on `active_camera =
0`, so without `request_free_camera()` you look down the gripper at whatever
the gripper faces, and dragging cannot fix it: a body-attached camera is
re-aimed EVERY frame, so the mouse fights the model and loses.
`task_viewer.mojo` and `sac_so_arm101_reach_policy_viewer.mojo` record the same
trap on the same asset.

## ⚠⚠ WHAT TO WATCH, PER TASK — THEY FAIL DIFFERENTLY AND THAT IS THE POINT

    so101_gather_bricks   0.5625 success, 1M steps. The arm should push the
                          two blocks together. THIS one works.
    so101_settle_brick    0.969 held at the END; its goal HOLDS AT RESET, so
                          the only thing to watch is whether the arm keeps the
                          brick on the table or knocks it off.
    so101_lift_brick      ~0 success. The gripper closes to about 0.05 m and
                          the brick never rises: `Above` pays only once the
                          brick is GRASPED, and a grasp is a discrete contact
                          event with no partial credit. Watch it approach and
                          stop — that is the shape of the failure.

## ⚠ THE READOUT IS THE REWARD'S OWN GEOMETRY, NOT A RE-DERIVATION

The goal distance comes from `gpu_eval.tape_distance_gpu` — the function the
reward kernel itself calls — and the subject the reach distance measures to
comes from `goal_frame_ids`. `task_shaping_probe.mojo` computed that distance
inline once and got `Near`'s rule for every predicate: `Above`'s shortfall is
Z-ONLY and `On`'s second argument is a REGION index, so `so101_settle_brick`
read 0.33 m for a goal that holds at reset. Two readers of one rule drift.

⚠ THE GOAL BOOL COMES FROM `eval.eval_goal`, NOT from `META_IDX_GOAL_HELD`.
That word is written by `custom_reward_gpu`, and this config's CPU reward hook
returns a constant zero on purpose — it is a GPU-only reward. The host
evaluator is the one that answers here.

⚠ RUN THIS ON THE LAPTOP, not a headless box — it opens an SDL3 window and
blocks on it. CPU physics and a CPU policy on purpose: one env at 60 Hz and a
256x256 MLP need no GPU.

⚠ SO-101 IS THE SLOW ONE TO COMPILE: 33 280 hull vertices.
"""

from std.os import listdir
from std.pathlib import Path
from std.random import seed as seed_rng
from std.sys import argv

from layout import Layout

from noeira.math3d import Vec3 as Vec3G, Quat as QuatG
from noeira.nn.constants import DT as NN_DT
from noeira.nn.core.tensor import TensorImpl
from noeira.deep_agents.data.any_replay import AnyReplay
from noeira.deep_agents.sac import SAC, SACAgent, SACActorNet, SACCriticNet
from noeira.tasks.sac_family_policy import (
    SacFamilyPolicy, HIDDEN as FAMILY_HIDDEN, POLICY_BATCH, POLICY_CAP,
)
from noeira.deep_agents.training.blocks import ReplaySampleStep

from noeira.core.cont_action import ContAction
from noeira.core.project import runs_root_for
from noeira.core.run import resolve_checkpoint
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.physics3d.gpu.constants import (
    METADATA_SIZE, META_IDX_TASK_PARAM_0, META_IDX_TASK_ACTIVE,
    META_IDX_INIT_REGION_0, META_IDX_SHAPE_W_GOAL, MODEL_CURRICULUM_SIZE,
)
from noeira.physics3d.parser.runtime_load import (
    parse_model_runtime, read_model_source,
)
from noeira.physics3d.parser.full_parser import parse_xml_full
from noeira.physics3d.parser.render_fields import build_render_fields
from noeira.physics3d.parser.model_def_from_xml import RfOnlyModelDef
from noeira.physics3d.model.model_renderer import ModelRenderer
from noeira.render.imgui import (
    imgui_shim_available, ig_begin_panel, ig_end, ig_text, ig_text_colored,
    ig_separator_text, ig_selectable, ig_button, ig_spacing,
)

from noeira.tasks.spec import (
    load_family, load_task, validate_task_against_family, SLOT_FREE,
    FamilySpec, TaskSpec,
)
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TabletopConfig, So101TabletopPlacement
from noeira.tasks.so101_tabletop_xml import So101TabletopModel
from noeira.tasks.predicates import (
    parse_goal, bind_goal, require_tier_a, BoundGoal,
)
from noeira.tasks.eval import (
    eval_goal, region_sites, region_rects, region_half_heights,
)
from noeira.tasks.active import active_mask, init_region_words
from noeira.tasks.shaping import shaping_words, SHAPING_WORDS
from noeira.tasks.tape import encode_goal, TAPE_WORDS
from noeira.tasks.gpu_eval import (
    region_table_words, tape_distance_gpu, goal_frame_ids,
)
from noeira.tasks.sampler import sample_placements, RegionFrame, SampleReport
from noeira.tasks.reset import (
    free_slot_addresses, reset_slots, SlotAddress,
)
from noeira.utils.fmt import fixed


comptime DT = DType.float64
comptime Vec3 = Vec3G[DT]
comptime Quat = QuatG[DT]

comptime FAMILY = "noeira/tasks/families/so101_tabletop.family"
comptime TASK_DIR = "noeira/tasks/tasks/"
comptime CKPT_PREFIX = "sac_task_"
"""⚠ MATCHES `sac_task_gpu.mojo`'s own prefix. That file names its checkpoint
per task for a reason — it wrote `sac_task_reach.ckpt` while training
`gather` for several runs, and two tasks then shared one file."""

comptime EnvT = Phyics3dEnv[So101TabletopModel, So101TabletopConfig, DT]
comptime OBS_DIM = So101TabletopModel.OBS_DIM
"""63 = NQ(27) + NV(24) + active mask(3) + goal geometry(9). ⚠ A checkpoint
is loaded by PARAMETER LAYOUT, so a stale OBS_DIM is a load error rather than
a wrong policy — which is the good outcome."""
comptime ACT_DIM = 6
comptime HIDDEN = FAMILY_HIDDEN
"""The family SAC widths, from `noeira/tasks/sac_family_policy.mojo` — the
driver trains with the same constants, so a `--policy` checkpoint loads."""
comptime BATCH = POLICY_BATCH
comptime CAP = POLICY_CAP

comptime ACTION_SCALE = 1.0
"""⚠⚠ MUST MATCH `sac_task_gpu.mojo`. The greedy action is
`tanh(mu) * action_scale`, and this family's action space is NORMALIZED —
[-1, 1] per joint, mapped affinely onto each actuator's `ctrlrange` by the env
(`So101TabletopConfig.NORMALIZED_ACTIONS`). A mismatched scale does not weaken
the policy, it commands a different pose."""

comptime NB = So101TabletopModel.NBODY
comptime NS = So101TabletopModel.NSITE
comptime L_META = Layout.row_major(1, METADATA_SIZE)
comptime L_CUR = Layout.row_major(1, MODEL_CURRICULUM_SIZE)
comptime L_XP = Layout.row_major(1, NB * 3)
comptime L_XQ = Layout.row_major(1, NB * 4)
comptime L_SP = Layout.row_major(1, NS * 3)

comptime SIDEBAR_W: Float32 = 320.0


def task_names() -> List[String]:
    """The tasks with a checkpoint on disk, in the sidebar's order.

    ⚠ ONLY THE THREE THAT WERE TRAINED. `so101_reach_brick` and
    `so101_reach_clear` are in the family and have no policy — and
    `task_null_action.mojo` records why nobody should train them: a CONSTANT
    action meets `AtRegion` for 77 consecutive steps because the arm SWEEPS
    through the region. Offering them here would put an untrained actor in a
    list of trained ones with nothing to say which is which.
    """
    var out = List[String]()
    out.append(String("so101_gather_bricks"))
    out.append(String("so101_settle_brick"))
    out.append(String("so101_lift_brick"))
    return out^


comptime RUN_PROJECT = "so101"
"""The project `sac_task_gpu.mojo` files its runs under."""


def newest_run_ckpt(task: String) -> String:
    """The newest `<runs>/<id>/checkpoints/last.ckpt` whose run is `task`'s,
    or "" when there is none.

    ⚠ The driver writes through a `RunContext`, so its checkpoint lives in the
    run's own directory: `<date>_sac-<task>_<hash>/checkpoints/last.ckpt`.
    Run ids start with the date, so the lexicographic maximum is the newest.
    A multi-task run (`sac-a+b`) is not picked for either task alone."""
    var root = runs_root_for(String(RUN_PROJECT))
    if not Path(root).exists():
        return String("")
    var want = String("_sac-") + task + "_"
    var best = String("")
    try:
        for e in listdir(root):
            var id = String(e)
            if want not in id:
                continue
            var c = root + "/" + id + "/checkpoints/last.ckpt"
            if Path(c).exists() and id > best:
                best = id
    except:
        return String("")
    if best.byte_length() == 0:
        return String("")
    return root + "/" + best + "/checkpoints/last.ckpt"


def ckpt_path(task: String) -> String:
    """The task's newest run first; then the pre-run-layout files, in
    `checkpoints/` and the CWD (where the trainer used to write)."""
    var r = newest_run_ckpt(task)
    if r.byte_length() > 0:
        return r^
    var a = String("checkpoints/") + CKPT_PREFIX + task + ".ckpt"
    if Path(a).exists():
        return a^
    return String(CKPT_PREFIX) + task + ".ckpt"


struct TaskPolicy(Movable & Deinitable):
    """One SAC checkpoint per task, loaded on demand and kept.

    ⚠ `learning_starts=0` IS LOAD-BEARING. `select_action` takes the uniform
    random WARMUP branch below that threshold, so with the default 1000 a
    "greedy" viewer would drive random actions for its first thousand frames —
    the trap `sac_so_arm101_reach_policy_viewer.mojo` and
    `dm_walker_policy_viewer.mojo` both document.
    """

    var agent: SacFamilyPolicy[OBS_DIM, ACT_DIM]
    var loaded_task: String
    var loaded: Bool
    # the task `--ckpt` named a checkpoint for, and that checkpoint ("" = none)
    var pinned_task: String
    var pinned_path: String

    def __init__(out self, pinned_task: String, pinned_path: String) raises:
        self.agent = SAC["cpu", OBS_DIM, ACT_DIM, BATCH, CAP, HIDDEN](
            action_scale=ACTION_SCALE, learning_starts=0
        )
        self.loaded_task = String("")
        self.loaded = False
        self.pinned_task = pinned_task.copy()
        self.pinned_path = pinned_path.copy()

    def __init__(out self, *, deinit move: Self):
        self.agent = move.agent^
        self.loaded_task = move.loaded_task^
        self.loaded = move.loaded
        self.pinned_task = move.pinned_task^
        self.pinned_path = move.pinned_path^

    def use(mut self, task: String) raises -> Bool:
        """Load `task`'s checkpoint, or report that there is none.

        ⚠ ALL-OR-NOTHING. A failed load leaves `loaded_task` alone rather than
        naming a policy that is not driving — the sidebar would then say one
        thing while another actor moved the arm.
        """
        if self.loaded and self.loaded_task == task:
            return True
        var p = ckpt_path(task)
        if self.pinned_path and task == self.pinned_task:
            p = self.pinned_path.copy()
        if not Path(p).exists():
            print("  ⚠ no checkpoint for", task, "at", p)
            return False
        self.agent.load(p)
        self.loaded_task = task.copy()
        self.loaded = True
        print("  policy :", p)
        return True



# ═══════════════════════════════════════════════════════════════════════════
# ⚠ MODULE SCOPE, NOT NESTED. A nested `def` cannot capture `env` mutably —
# "Could not infer capture convention of the captured value env" — so the
# state these two touch is passed explicitly. `task_eval_frozen.mojo` hoisted
# `run_eval` for the same reason.
# ═══════════════════════════════════════════════════════════════════════════


def task_reset(
    mut env: EnvT,
    f: FamilySpec,
    t: TaskSpec,
    rsites: List[Int],
    addrs: List[SlotAddress],
    lane: Int,
    tape: List[Float64],
    mask: Float64,
    iw: List[Float64],
    sw: List[Float64],
    mut obs: List[Scalar[NN_DT]],
) raises:
    """Reset the env, then place the props and write `meta`.

    ⚠⚠ THE CPU RESET DOES NOT PLACE PROPS. `init_qpos_gpu` places them on the
    DEVICE path; `custom_reset_cpu` is not overridden by this config, so a free
    slot would sit at the composed scene's PARK pose — 50 m up — and fall for
    the whole episode with its qpos and qvel in the observation.
    `task_null_action.mojo` does the same host placement for the same reason.
    """
    _ = env.reset()
    var nq = env.d.dims.get_nq()
    var nv = env.d.dims.get_nv()
    var sp0 = List[Float64]()
    for i in range(NS * 3):
        sp0.append(Float64(env.d.site_xpos.data[i]))
    var frames = List[RegionFrame]()
    for i in range(len(f.regions)):
        var rs = rsites[i]
        frames.append(
            RegionFrame(sp0[rs * 3], sp0[rs * 3 + 1], sp0[rs * 3 + 2])
        )
    var radii = List[Float64]()
    for _ in range(len(f.slots)):
        radii.append(So101TabletopConfig.SLOT_RADIUS)
    var rep = SampleReport()
    var placed = sample_placements(t, f, frames, radii, UInt64(0), lane, rep)
    var q0 = List[Float64]()
    for i in range(nq):
        q0.append(Float64(env.d.qpos.data[i]))
    var v0 = List[Float64]()
    for _ in range(nv):
        v0.append(0.0)
    reset_slots(t, f, placed, addrs, q0, v0)

    # ⚠ THE FOUR CHANNELS, EVERY EPISODE. `meta` is not zeroed between
    # episodes, but writing them at every reset costs nothing and removes the
    # question of whether the CPU reset path preserves them.
    for w in range(TAPE_WORDS):
        env.d.meta.data[META_IDX_TASK_PARAM_0 + w] = Scalar[DT](tape[w])
    env.d.meta.data[META_IDX_TASK_ACTIVE] = Scalar[DT](mask)
    for j in range(len(iw)):
        env.d.meta.data[META_IDX_INIT_REGION_0 + j] = Scalar[DT](iw[j])
    for j in range(SHAPING_WORDS):
        env.d.meta.data[META_IDX_SHAPE_W_GOAL + j] = Scalar[DT](sw[j])

    # `obs_at` sets the state and re-extracts through the config's own hook,
    # so the policy sees exactly what the trainer's obs kernel would.
    var st = env.obs_at(q0, v0)
    for i in range(OBS_DIM):
        obs[i] = Scalar[NN_DT](st.data[i])


def measure(
    mut env: EnvT,
    f: FamilySpec,
    g: BoundGoal,
    rsites: List[Int],
    mut t_meta: TensorImpl[DT],
    mut t_cur: TensorImpl[DT],
    mut t_xp: TensorImpl[DT],
    mut t_xq: TensorImpl[DT],
    mut t_sp: TensorImpl[DT],
) raises -> Tuple[Bool, Float64, Float64]:
    """`(goal holds, goal distance, gripper-to-subject distance)`.

    ⚠ THE GOAL DISTANCE COMES FROM `tape_distance_gpu`, the function the reward
    kernel itself calls, and the subject from `goal_frame_ids`.
    `task_shaping_probe.mojo` once computed that distance inline and got
    `Near`'s rule for every predicate — `Above`'s shortfall is Z-ONLY and
    `On`'s second argument is a REGION index — so `so101_settle_brick` read
    0.33 m for a goal that holds at reset.
    """
    for i in range(NB * 3):
        t_xp.data[i] = env.d.xpos.data[i]
    for i in range(NB * 4):
        t_xq.data[i] = env.d.xquat.data[i]
    for i in range(NS * 3):
        t_sp.data[i] = env.d.site_xpos.data[i]
    for i in range(METADATA_SIZE):
        t_meta.data[i] = env.d.meta.data[i]
    var goal_d = Float64(
        tape_distance_gpu[DT, 1, NB, NS * 3](
            t_meta.lt["cpu", L_META](), t_cur.lt["cpu", L_CUR](),
            t_xp.lt["cpu", L_XP](), t_xq.lt["cpu", L_XQ](),
            t_sp.lt["cpu", L_SP](), 0,
        )
    )
    var ids = goal_frame_ids(
        g.terms[0].op, g.terms[0].a, g.terms[0].b,
        So101TabletopPlacement.REGION_SITE_ID,
    )
    comptime GS = So101TabletopConfig.GRIPPER_SITE
    var sx: Float64
    var sy: Float64
    var sz: Float64
    if ids[0] == 1:
        sx = Float64(env.d.site_xpos.data[ids[1] * 3])
        sy = Float64(env.d.site_xpos.data[ids[1] * 3 + 1])
        sz = Float64(env.d.site_xpos.data[ids[1] * 3 + 2])
    else:
        sx = Float64(env.d.xpos.data[ids[1] * 3])
        sy = Float64(env.d.xpos.data[ids[1] * 3 + 1])
        sz = Float64(env.d.xpos.data[ids[1] * 3 + 2])
    var gx = Float64(env.d.site_xpos.data[GS * 3])
    var gy = Float64(env.d.site_xpos.data[GS * 3 + 1])
    var gz = Float64(env.d.site_xpos.data[GS * 3 + 2])
    var reach_d = ((gx - sx) ** 2 + (gy - sy) ** 2 + (gz - sz) ** 2) ** 0.5

    var xb = List[Float64]()
    for i in range(NB * 3):
        xb.append(Float64(env.d.xpos.data[i]))
    var xq = List[Float64]()
    for i in range(NB * 4):
        xq.append(Float64(env.d.xquat.data[i]))
    var sp2 = List[Float64]()
    for i in range(NS * 3):
        sp2.append(Float64(env.d.site_xpos.data[i]))
    # ⚠ `eval_goal`, NOT `META_IDX_GOAL_HELD` — that word is written by
    # `custom_reward_gpu` and this config's CPU reward is a constant zero.
    var holds = eval_goal(g, f, xb, xq, sp2, rsites)
    return (holds, goal_d, reach_d)


def main() raises:
    var args = argv()
    var task_name = String("so101_gather_bricks")
    var check_only = False
    var ckpt_arg = String("")
    var ai = 1
    while ai < len(args):
        var a = String(args[ai])
        if a == "--check":
            check_only = True
        elif a == "--ckpt" and ai + 1 < len(args):
            ckpt_arg = String(args[ai + 1])
            ai += 1
        elif ai == 1:
            task_name = a.copy()
        ai += 1
    if ckpt_arg:
        # a RUN ID -> its `checkpoints/last.ckpt` (what the SAC task driver
        # writes); a file is taken as is; anything else raises here.
        ckpt_arg = resolve_checkpoint(ckpt_arg, String("last"))
    seed_rng(0)

    print("=" * 70)
    print("task policy viewer —", task_name)
    print("=" * 70)
    if not check_only and not imgui_shim_available():
        print("  ⚠ no Dear ImGui shim — no sidebar, task fixed to argv.")
        print("    pixi run build-imgui")

    var f = load_family(String(FAMILY))
    var names = task_names()
    var cur = 0
    for i in range(len(names)):
        if names[i] == task_name:
            cur = i
    var t = load_task(String(TASK_DIR) + names[cur] + ".task")
    validate_task_against_family(t, f)

    var path = scene_path(f)
    var fmd = parse_model_runtime(path)
    var rsites = region_sites(f, fmd.site_names)
    var rects = region_rects(f)
    var rheights = region_half_heights(f)
    var g = bind_goal(parse_goal(t.goal), f, fmd.body_names, fmd.site_names)
    require_tier_a(g, t.name)

    var jt = List[Int]()
    var jq = List[Int]()
    var jv = List[Int]()
    for i in range(len(fmd.joints)):
        jt.append(fmd.joints[i].jnt_type)
        jq.append(fmd.joints[i].nq)
        jv.append(fmd.joints[i].nv)
    var addrs = free_slot_addresses(f, fmd.joint_names, jt, jq, jv)

    var env = EnvT()
    var nq = env.d.dims.get_nq()
    var nv = env.d.dims.get_nv()
    print("  family :", f.name, "|", len(f.slots), "slots,",
          f.n_free_slots(), "free")
    print("  task   :", t.name, "|", t.goal)
    print("  says   :", t.language)
    print("  scene  :", path, " nq", nq, " nv", nv, " obs", OBS_DIM)

    var pol = TaskPolicy(names[cur], ckpt_arg)
    var have_pol = pol.use(names[cur])

    # ── the scratch the reward's own distance function needs ──────────────
    # ⚠ THE SAME `curriculum` WORDS THE KERNEL READS, built from the family so
    # the readout cannot describe a different region from the reward.
    var t_meta = TensorImpl[DT].alloc(METADATA_SIZE)
    var t_cur = TensorImpl[DT].alloc(MODEL_CURRICULUM_SIZE)
    var t_xp = TensorImpl[DT].alloc(NB * 3)
    var t_xq = TensorImpl[DT].alloc(NB * 4)
    var t_sp = TensorImpl[DT].alloc(NS * 3)
    var cw = region_table_words(
        rsites[0], rects[0][0], rects[0][1], rects[0][2], rects[0][3],
        rheights[0],
    )
    for i in range(MODEL_CURRICULUM_SIZE):
        t_cur.data[i] = Scalar[DT](cw[i])

    var tape = encode_goal(g)
    var mask = active_mask(t, f)
    var iw = init_region_words(t, f)
    var sw = shaping_words(
        So101TabletopConfig.SHAPE_W_GOAL, So101TabletopConfig.SHAPE_W_REACH,
        So101TabletopConfig.GOAL_MARGIN, So101TabletopConfig.REACH_MARGIN,
    )

    var obs = List[Scalar[NN_DT]](length=OBS_DIM, fill=Scalar[NN_DT](0))
    var act = List[Scalar[NN_DT]](length=ACT_DIM, fill=Scalar[NN_DT](0))
    var lane = 0
    var episode = 1
    var step = 0
    var held = 0
    var paused = False
    var goal_d = 0.0
    var reach_d = 0.0
    var holds = False

    task_reset(env, f, t, rsites, addrs, lane, tape, mask, iw, sw, obs)
    var mm = measure(env, f, g, rsites, t_meta, t_cur, t_xp, t_xq, t_sp)
    holds = mm[0]
    goal_d = mm[1]
    reach_d = mm[2]

    if check_only:
        # ⚠⚠ A LANE SWEEP, NOT ONE EPISODE, and it is the check that matters:
        # it asks whether this viewer reproduces the rate the TRAINER measured.
        # A viewer with the wrong observation layout, the wrong action scale or
        # a missing `meta` channel drives a policy that looks plausible and
        # scores nothing — and one lane cannot tell that apart from a lane the
        # policy happens to fail. `sac_task_gpu.mojo` measured 0.5625 on
        # `gather` over 32 greedy lanes.
        comptime N_LANES = 12
        print("  policy loaded:", have_pol)
        var n_any = 0
        var n_end = 0
        var sum_goal = 0.0
        var sum_reach = 0.0
        for ln in range(N_LANES):
            lane = ln
            task_reset(
                env, f, t, rsites, addrs, lane, tape, mask, iw, sw, obs
            )
            var ever = False
            for _ in range(So101TabletopConfig.MAX_STEPS):
                if have_pol:
                    pol.agent.select_greedy_action(obs, act)
                var av = List[Float64]()
                for j in range(ACT_DIM):
                    av.append(Float64(act[j]))
                var res = env.step(ContAction[ACT_DIM].from_list(av))
                var ol = res[0].to_list()
                for i in range(OBS_DIM):
                    obs[i] = Scalar[NN_DT](ol[i])
                mm = measure(
                    env, f, g, rsites, t_meta, t_cur, t_xp, t_xq, t_sp
                )
                holds = mm[0]
                goal_d = mm[1]
                reach_d = mm[2]
                if holds:
                    ever = True
            if ever:
                n_any += 1
            if holds:
                n_end += 1
            sum_goal += goal_d
            sum_reach += reach_d
        print("  lanes            :", N_LANES)
        print("  met at ANY step  :", n_any, "/", N_LANES, "=",
              Float64(n_any) / Float64(N_LANES))
        print("  held at the END  :", n_end, "/", N_LANES, "=",
              Float64(n_end) / Float64(N_LANES))
        print("  mean final goal  :", sum_goal / Float64(N_LANES), "m")
        print("  mean final reach :", sum_reach / Float64(N_LANES), "m")
        # ⚠ ANTI-VACUITY. A viewer whose observation or action wiring is wrong
        # drives a policy that never achieves anything, and every task then
        # reads 0 — which is also what `lift` legitimately reads. `settle`'s
        # goal holds at RESET, so it must come back at 1.0 whatever the policy
        # does; a zero there is the wiring, not the task.
        if t.name == "so101_settle_brick" and n_any != N_LANES:
            raise Error(
                "task policy viewer: `so101_settle_brick` met its goal on only"
                " " + String(n_any) + " of " + String(N_LANES) + " lanes. Its"
                " goal HOLDS AT RESET by construction, so anything below all"
                " of them is this viewer's wiring — the observation layout,"
                " the action scale, or a `meta` channel that never got"
                " written — and not the policy."
            )
        print("=== CHECKED ===")
        return

    var src = read_model_source(path)
    var rf = build_render_fields(parse_xml_full(src[0], src[1]), src[0], src[1])
    var renderer = ModelRenderer[RfOnlyModelDef](
        width=1280, height=800, visual_radius_scale=1.0, show_velocity=False,
        title=String("task policy — ") + f.name,
        adopt_rf=Optional(rf.copy()),
    )
    renderer.init(None)
    renderer.request_free_camera()

    var have_ui = renderer.imgui_init()
    if have_ui:
        renderer.set_ui_sidebar_width(Int(SIDEBAR_W))
        renderer.set_show_hud(False)

    var positions = List[Vec3]()
    var quats = List[Quat]()

    while renderer.is_open():
        if renderer.check_quit():
            break

        if not paused:
            if have_pol:
                pol.agent.select_greedy_action(obs, act)
            var av = List[Float64]()
            for j in range(ACT_DIM):
                av.append(Float64(act[j]))
            var res = env.step(ContAction[ACT_DIM].from_list(av))
            var ol = res[0].to_list()
            for i in range(OBS_DIM):
                obs[i] = Scalar[NN_DT](ol[i])
            step += 1
            mm = measure(env, f, g, rsites, t_meta, t_cur, t_xp, t_xq, t_sp)
            holds = mm[0]
            goal_d = mm[1]
            reach_d = mm[2]
            if holds:
                held += 1

        var want_task = cur
        var want_reset = False
        if have_ui:
            renderer.imgui_new_frame()
            ig_begin_panel(
                String("task policy"), 0.0, 0.0, SIDEBAR_W,
                Float32(renderer.renderer.height),
            )
            ig_separator_text(String("family"))
            ig_text(f.name + "  (" + String(len(f.slots)) + " slots, "
                    + String(f.n_free_slots()) + " free)")

            # ⚠⚠ SELECTING A TASK RELOADS THE `.task` AND ITS CHECKPOINT. One
            # policy per task; a `gather` actor driving `lift` is a demo of
            # nothing. The MODEL is not rebuilt — the family's slot budget is
            # fixed, which is what makes the switch a data reload.
            ig_separator_text(String("task + its policy"))
            for i in range(len(names)):
                if ig_selectable(names[i], i == cur):
                    want_task = i
            ig_spacing()
            ig_text(String("says: ") + t.language)
            ig_text(String("goal: ") + t.goal)
            if have_pol:
                ig_text(String("policy: ") + pol.loaded_task)
            else:
                ig_text_colored(
                    String("NO CHECKPOINT — the arm is not driven"),
                    0.9, 0.5, 0.3,
                )

            ig_separator_text(String("goal"))
            if holds:
                ig_text_colored(
                    String("HOLDS  (") + String(held) + " of "
                    + String(step) + " steps)", 0.3, 0.9, 0.4,
                )
            else:
                ig_text_colored(String("not met"), 0.9, 0.5, 0.3)
            # ⚠ THE TWO DISTANCES THE REWARD IS MADE OF, in millimetres. The
            # goal one comes from `tape_distance_gpu` — the kernel's own
            # function — so it is a z-shortfall for `Above` and a box distance
            # for `On`, not a body-to-body separation for everything.
            ig_text(String("goal distance  ") + fixed(goal_d * 1000.0, 1)
                    + " mm")
            ig_text(String("gripper -> subject ")
                    + fixed(reach_d * 1000.0, 1) + " mm")

            ig_separator_text(String("episode"))
            ig_text(String("ep ") + String(episode) + "   step "
                    + String(step) + " / "
                    + String(So101TabletopConfig.MAX_STEPS))
            ig_text(String("lane ") + String(lane))
            if ig_button(String("reset (next lane)"), -1.0):
                want_reset = True
            if ig_button(String("pause / run"), -1.0):
                paused = not paused
            if ig_button(String("free camera"), -1.0):
                renderer.request_free_camera()
            ig_end()

        positions.clear()
        quats.clear()
        for b in range(NB):
            positions.append(Vec3(
                Float64(env.d.xpos.data[b * 3 + 0]),
                Float64(env.d.xpos.data[b * 3 + 1]),
                Float64(env.d.xpos.data[b * 3 + 2]),
            ))
            # ⚠ `Data.xquat` IS (x, y, z, W) AND `Quat` TAKES (W, x, y, z).
            quats.append(Quat(
                Float64(env.d.xquat.data[b * 4 + 3]),
                Float64(env.d.xquat.data[b * 4 + 0]),
                Float64(env.d.xquat.data[b * 4 + 1]),
                Float64(env.d.xquat.data[b * 4 + 2]),
            ))
        renderer.render(positions, quats)

        # ⚠⚠ THE SWITCH HAPPENS AFTER `render`, NEVER BEFORE.
        # `imgui_new_frame` opened a frame that only `render` closes; work that
        # can raise between them leaves it open and the NEXT `NewFrame`
        # asserts. `task_viewer.mojo` and `physics_studio` both document it.
        if want_task != cur:
            cur = want_task
            t = load_task(String(TASK_DIR) + names[cur] + ".task")
            validate_task_against_family(t, f)
            g = bind_goal(
                parse_goal(t.goal), f, fmd.body_names, fmd.site_names
            )
            require_tier_a(g, t.name)
            tape = encode_goal(g)
            mask = active_mask(t, f)
            iw = init_region_words(t, f)
            have_pol = pol.use(names[cur])
            print("  -> task:", t.name, "|", t.goal)
            step = 0
            lane += 1
            task_reset(env, f, t, rsites, addrs, lane, tape, mask, iw, sw, obs)
            mm = measure(env, f, g, rsites, t_meta, t_cur, t_xp, t_xq, t_sp)
            holds = mm[0]
            goal_d = mm[1]
            reach_d = mm[2]
            episode += 1
            held = 0
        elif want_reset or step >= So101TabletopConfig.MAX_STEPS:
            step = 0
            lane += 1
            task_reset(env, f, t, rsites, addrs, lane, tape, mask, iw, sw, obs)
            mm = measure(env, f, g, rsites, t_meta, t_cur, t_xp, t_xq, t_sp)
            holds = mm[0]
            goal_d = mm[1]
            reach_d = mm[2]
            episode += 1
            held = 0

    renderer.close()
    print("  closed after", episode, "episode(s)")

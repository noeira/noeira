"""Watch a PIXEL student (and the state teacher it learned from) drive the
so101_tower — ImGui sidebar, the overhead and wrist cameras as insets.

    pixi run build-imgui                                                   # ONCE
    pixi run mojo run -I . examples/so101/tower_pixel_student_viewer.mojo
    pixi run mojo run -I . examples/so101/tower_pixel_student_viewer.mojo so101_tower_cube_in_bowl
    pixi run mojo run -I . -D DAGGER_PX_32 examples/so101/tower_pixel_student_viewer.mojo \\
        so101_tower_cube_in_bowl                  # a 32x32 student needs a 32x32 build
    ... --seed 3                                  # another placement of the props
    ... --policy 82139be3                         # a policy by (part of) its run id
    ... --record gifs/so101_bowl.mp4 --episodes 2 # an unattended clip, then close
    ... --eye 0.75,-0.6,0.5 --target 0.2,0,0.05  # a fixed free-camera shot

The interactive counterpart of the DAgger driver's greedy eval
(`noeira/tasks/pixel_dagger_tower.mojo`). Two tasks — lift on the real
layouts (`so101_tower_lift_real_layout`) and `so101_tower_cube_in_bowl` —
selectable in the sidebar; the variant combo lists every pixel student found
for the task (newest first) AND the PPO teacher each was distilled from, so
student and teacher can be watched on the same placement.

## How the student sees here — the trainer's picture, on the CPU

Every step: the env's `qpos` (the observation's first `NQ` words) goes into
the rig's own `Data`, host forward kinematics, both cameras TRACED at 64 x 64
with one sample through the rig's renderer (`render_cpu`, the same code as the
GPU kernel), block-averaged to the student's 16 x 16 or 32 x 32
(`pixel_student.render_to_planes`), plus the six joint planes; the student's
delta action goes through `delta_action.delta_target` and is handed to the env
as its normalised absolute action. The teacher acts greedily on its own
normalised STATE observation (`obs_norm.txt` beside its checkpoint), through
the same delta rule. The insets (`1` overhead, `0` wrist) are the viewer's
rasteriser — the SCENE the cameras see, sharper than what the student gets.

## Where things are looked for

Students: `runs/*dagger-px-<task>*` and `projects/so101-tower/runs/*dagger-px-
<task>*` with `checkpoints/last.ckpt` (or the older `student.ckpt`), and the
promoted `projects/so101-tower/policies/pixel_*.ckpt` whose manifest names the
task. A student of another resolution than this build is listed and refused
with the define to build with. Teachers: the directory each student's config
names (`teacher=`), which must be on this machine (copy it from the box).

⚠ ON THE LAPTOP, FROM THE REPO ROOT. The reward on the CPU env is zero by
design (the family's reward is a GPU kernel); the status line prints the
distances the task is judged by instead.
"""

from std.os import listdir
from std.os.path import exists
from std.random import seed
from std.sys import argv

from max.gpu.host import DeviceContext

from noeira.envs.dm_control.viewer_core import (
    ActionSource, DRIVE_POLICY, ViewerState, run_view, task_index,
)
from noeira.io.json import load_json
from noeira.nn.constants import DT
from noeira.nn.core.checkpoint import load_params
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.ptr import mptr
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.physics3d.fields import Data, Model
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.render.imgui import imgui_shim_available
from noeira.render.renderer3d import Renderer3D
from noeira.tasks.delta_action import delta_target
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig
from noeira.tasks.pixel_student import (
    StudentNet, N_CAMS, OBS_PX, IN_DIM, ACT, JOINT_VEL, render_to_planes,
    OVERHEAD_RENDER_W, OVERHEAD_RENDER_H, WINDOWED,
    joints_to_planes, joint_vels_to_planes, student_act,
)
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.posed_reset import posed_qpos, task_meta_words
from noeira.tasks.ppo_family_driver import ActorNet, CriticNet, RunningMeanStd, OBS_CLIP
from noeira.deep_agents.ppo import PPOAgent
from noeira.tasks.so101_tower_rig import (
    RIG_DT, TOWER_MD, TowerRendererSized, make_tower_model,
    make_tower_renderer, tower_cameras,
)
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.spec import load_family
from noeira.utils.fmt import fixed

comptime FAMILY = "so101_tower"
comptime FAMILY_PATH = "noeira/tasks/families/so101_tower.family"
comptime OBS_DIM = So101TowerModel.OBS_DIM
comptime NQ = So101TowerModel.NQ
comptime GOAL_BASE = So101TowerConfig.OBS_GOAL_BASE
comptime RENDER = 64
comptime TeacherT = PPOAgent["cpu", ActorNet[OBS_DIM], CriticNet[OBS_DIM], OBS_DIM, ACT, 16, 16, 1, 1]


def task_names() -> List[String]:
    var t = List[String]()
    t.append(String("so101_tower_lift_real_layout"))
    t.append(String("so101_tower_cube_in_bowl"))
    return t^


def _cfg_value(path: String, key: String) -> String:
    """`key=` from a `metrics.config.kv`, or "" (a missing file too)."""
    try:
        with open(path, "r") as f:
            for ln in f.read().split("\n"):
                var s = String(ln)
                if s.startswith(key + "="):
                    return String(s[byte = key.byte_length() + 1 :])
    except:
        pass
    return String("")


struct Variant(Copyable, Movable):
    var label: String
    var is_teacher: Bool
    var ckpt: String
    var norm: String
    """The teacher's `obs_norm.txt`; unused for a student."""
    var refuse: String
    """Why this build cannot run it ("" if it can)."""
    var d_arm: Float64
    var d_grip: Float64
    """The delta scales it was trained with (its run config's)."""

    def __init__(out self, label: String, is_teacher: Bool, ckpt: String,
                 norm: String, refuse: String, d_arm: Float64 = 0.05,
                 d_grip: Float64 = 0.2):
        self.label = label
        self.is_teacher = is_teacher
        self.ckpt = ckpt
        self.norm = norm
        self.refuse = refuse
        self.d_arm = d_arm
        self.d_grip = d_grip


def find_variants(task: String) raises -> List[Variant]:
    """The task's pixel students, newest first, then their teachers (once
    each). Labels start with the task's short name: the viewer lists BOTH
    tasks' policies, because the task can be switched in the sidebar."""
    var short = String("lift") if task.find("lift") >= 0 else String("bowl")
    var dirs = List[String]()
    for root in [String("runs"), String("projects/so101-tower/runs")]:
        if not exists(root):
            continue
        for e in listdir(root):
            var n = String(e)
            if n.find("dagger-px-" + task) >= 0 and not n.startswith("eval-") and n.find("eval-dagger") < 0:
                dirs.append(root + "/" + n)
    # newest first (run dirs start with the date)
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            var ni = dirs[i][byte = dirs[i].rfind("/") + 1 :]
            var nj = dirs[j][byte = dirs[j].rfind("/") + 1 :]
            if String(nj) > String(ni):
                dirs[i], dirs[j] = dirs[j], dirs[i]
    var out = List[Variant]()
    var teachers = List[String]()
    for d in dirs:
        var ck = d + "/checkpoints/last.ckpt"
        if not exists(ck):
            ck = d + "/student.ckpt"
        if not exists(ck):
            continue
        var cfg = d + "/metrics.config.kv"
        var px_s = _cfg_value(cfg, "obs_px")
        var px = Int(px_s) if px_s.byte_length() > 0 else 16
        var refuse = String("")
        if px != OBS_PX:
            refuse = String("a ") + String(px) + "px student: build with" + (
                " -D DAGGER_PX_32" if px == 32 else " the default (16)"
            )
        var wn = _cfg_value(cfg, "window")
        if (wn == "workspace") != WINDOWED:
            refuse = String("camera window '") + (wn if wn else String("centre-square")) + "': build " + (
                "with -D DAGGER_WINDOW" if wn == "workspace" else "without -D DAGGER_WINDOW"
            )
        var pr = _cfg_value(cfg, "proprio")
        var want = String("q+qd") if JOINT_VEL else String("q")
        if (pr if pr.byte_length() > 0 else String("q")) != want:
            refuse = String("joint input '") + (pr if pr else String("q")) + "': build " + (
                "with -D DAGGER_JOINT_VEL" if pr == "q+qd" else "without -D DAGGER_JOINT_VEL"
            )
        var name = String(d[byte = d.rfind("/") + 1 :])
        var da_s = _cfg_value(cfg, "delta_arm")
        var dg_s = _cfg_value(cfg, "delta_gripper")
        out.append(Variant(
            short + " student " + String(name[byte = name.byte_length() - 8 :]) + " ("
            + String(px) + "px)" + (" — needs another build" if refuse else ""),
            False, ck, String(""), refuse,
            Float64(da_s) if da_s else 0.05, Float64(dg_s) if dg_s else 0.2,
        ))
        var t = _cfg_value(cfg, "teacher")
        var seen = False
        for u in teachers:
            if u == t:
                seen = True
        if t.byte_length() > 0 and not seen:
            teachers.append(t)
    for t in teachers:
        var ck = t + "/checkpoints/last.ckpt"
        var tn = String(t[byte = t.rfind("/") + 1 :])
        var refuse = String("") if exists(ck) else String("not on this machine: ") + t
        var tda = _cfg_value(t + "/metrics.config.kv", "delta_arm")
        var tdg = _cfg_value(t + "/metrics.config.kv", "delta_gripper")
        out.append(Variant(
            short + " teacher " + String(tn[byte = tn.byte_length() - 8 :])
            + ("" if not refuse else " (missing)"),
            True, ck, t + "/obs_norm.txt", refuse,
            Float64(tda) if tda else 0.05, Float64(tdg) if tdg else 0.2,
        ))
    return out^


struct PixelViewerPolicy(ActionSource, Movable):
    var variants: List[Variant]
    var current: Int
    var student: StudentNet
    var teacher: TeacherT
    var obs_rms: RunningMeanStd
    var ctx: DeviceContext
    var rm: Model[RIG_DT, TOWER_MD]
    var rd: Data[RIG_DT, TOWER_MD, 1]
    var r: TowerRendererSized[1, RENDER, RENDER, 1]
    """The wrist's square trace."""
    var r_o: TowerRendererSized[1, OVERHEAD_RENDER_W, OVERHEAD_RENDER_H, 1]
    """The overhead's: the full 4:3 frame with DAGGER_WINDOW, else square."""
    var cams: List[Int]
    var a_qa: List[Int]
    var a_da: List[Int]
    var lo: List[Float64]
    var hi: List[Float64]
    var xs: List[Scalar[DT]]
    var x: Tensor
    var y: Tensor
    var last_a: List[Float64]
    var reach_mm: Float64
    var goal_mm: Float64

    def __init__(out self) raises:
        self.variants = List[Variant]()
        for t in task_names():
            for v in find_variants(t):
                self.variants.append(v.copy())
        self.current = -1
        self.student = StudentNet.make["cpu", Kaiming](None)
        self.teacher = TeacherT()
        self.obs_rms = RunningMeanStd(OBS_DIM)
        self.ctx = DeviceContext()
        var f = load_family(String(FAMILY_PATH))
        var fmd = parse_model_runtime(scene_path(f))
        self.rm = make_tower_model(self.ctx)
        self.rd = Data[RIG_DT, TOWER_MD, 1]()
        self.r = make_tower_renderer[1, RENDER, RENDER, 1](self.ctx, fmd, self.rm)
        self.r_o = make_tower_renderer[1, OVERHEAD_RENDER_W, OVERHEAD_RENDER_H, 1](
            self.ctx, fmd, self.rm
        )
        var both = tower_cameras(fmd)
        self.cams = List[Int]()
        comptime if N_CAMS == 2:
            self.cams.append(both[0])
        self.cams.append(both[1])
        var jadr = List[Int]()
        var acc = 0
        for i in range(len(fmd.joints)):
            jadr.append(acc)
            acc += fmd.joints[i].nq
        var jdadr = List[Int]()
        var dacc = 0
        for i in range(len(fmd.joints)):
            jdadr.append(dacc)
            dacc += fmd.joints[i].nv
        self.a_qa = List[Int]()
        self.a_da = List[Int]()
        self.lo = List[Float64]()
        self.hi = List[Float64]()
        for i in range(ACT):
            self.a_qa.append(jadr[fmd.actuators[i].joint_id])
            self.a_da.append(jdadr[fmd.actuators[i].joint_id])
            self.lo.append(fmd.actuators[i].ctrl_min)
            self.hi.append(fmd.actuators[i].ctrl_max)
        self.xs = List[Scalar[DT]](length=IN_DIM, fill=Scalar[DT](0))
        self.x = Tensor.alloc(IN_DIM)
        self.y = Tensor.alloc(ACT)
        self.last_a = List[Float64](length=ACT, fill=0.0)
        self.reach_mm = -1.0
        self.goal_mm = -1.0
        print("  variants (both tasks — any policy can drive either task):")
        if len(self.variants) == 0:
            print("    ⚠ none — train a student (examples/tasks/dagger_tower_pixels.mojo)"
                  " or copy its run dir into runs/")
        for i in range(len(self.variants)):
            print("    ", i, self.variants[i].label,
                  ("   ⚠ " + self.variants[i].refuse) if self.variants[i].refuse else "")

    def obs_dim(self) -> Int:
        return OBS_DIM

    def act_dim(self) -> Int:
        return ACT

    def variant_labels(self) -> List[String]:
        var l = List[String]()
        for v in self.variants:
            l.append(v.label)
        return l^

    def choose(mut self, i: Int) raises:
        if i < 0 or i >= len(self.variants):
            raise Error("variant out of range: " + String(i))
        var v = self.variants[i].copy()
        if v.refuse:
            raise Error(v.label + ": " + v.refuse)
        if v.is_teacher:
            self.teacher.trainer.load_state(v.ckpt)
            self.obs_rms.load(v.norm)
        else:
            load_params["cpu"](self.student, v.ckpt, None)
        self.current = i
        print("  loaded", v.ckpt)

    def status(self) -> String:
        if self.current < 0:
            return String("no policy loaded")
        var s = self.variants[self.current].label
        if self.reach_mm >= 0.0:
            s += "   reach " + fixed(self.reach_mm, 1) + " mm   goal " + fixed(self.goal_mm, 1) + " mm"
        s += "   grip " + fixed(self.last_a[ACT - 1], 2)
        return s

    def act(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        greedy: Bool,
    ) raises:
        _ = greedy  # both policies are evaluated greedily, as in the evals
        if self.current < 0:
            for j in range(ACT):
                action_out[j] = Scalar[DT](0)
            return
        # the distances the task is judged by (the goal words, see
        # tower_policy_viewer.mojo)
        var rx = Float64(obs[GOAL_BASE + 3])
        var ry = Float64(obs[GOAL_BASE + 4])
        var rz = Float64(obs[GOAL_BASE + 5])
        self.reach_mm = ((rx * rx + ry * ry + rz * rz) ** 0.5) * 1000.0
        var gx = Float64(obs[GOAL_BASE + 6])
        var gy = Float64(obs[GOAL_BASE + 7])
        var gz = Float64(obs[GOAL_BASE + 8])
        self.goal_mm = ((gx * gx + gy * gy + gz * gz) ** 0.5) * 1000.0
        var a = List[Float64](length=ACT, fill=0.0)
        if self.variants[self.current].is_teacher:
            var on = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0))
            var src = List[Scalar[DT]](length=OBS_DIM, fill=Scalar[DT](0))
            for k in range(OBS_DIM):
                src[k] = obs[k]
            self.obs_rms.normalize_into(
                mptr(src.unsafe_ptr()), mptr(on.unsafe_ptr()), 1, OBS_DIM, OBS_CLIP
            )
            var ao = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0))
            self.teacher.trainer.select_greedy_action(on, ao)
            for j in range(ACT):
                a[j] = Float64(ao[j])
        else:
            # the trainer's picture of THIS state: qpos -> FK -> trace -> average
            for k in range(NQ):
                self.rd.qpos.data[k] = Scalar[RIG_DT](obs[k])
            forward_kinematics["cpu", RIG_DT, TOWER_MD, 1](self.rd, self.rm)
            for k in range(len(self.cams)):
                var rgb = List[Scalar[RIG_DT]]()
                var dep = List[Scalar[RIG_DT]]()
                var seg = List[Scalar[RIG_DT]]()
                if N_CAMS == 2 and k == 0:
                    # the overhead: its own trace (4:3 when windowed)
                    self.r_o.cam = self.cams[k]
                    self.r_o.render_cpu(self.rd, self.rm, rgb, dep, seg)
                    render_to_planes(rgb, OVERHEAD_RENDER_W, OVERHEAD_RENDER_H, k, self.xs)
                else:
                    self.r.cam = self.cams[k]
                    self.r.render_cpu(self.rd, self.rm, rgb, dep, seg)
                    render_to_planes(rgb, RENDER, RENDER, k, self.xs)
            var q = List[Float64](length=ACT, fill=0.0)
            for j in range(ACT):
                q[j] = Float64(obs[self.a_qa[j]])
            joints_to_planes(q, self.xs)
            # the observation is qpos (NQ) then qvel: the joint velocities
            # at their dof addresses (a no-op without DAGGER_JOINT_VEL)
            var qd = List[Float64](length=ACT, fill=0.0)
            for j in range(ACT):
                qd[j] = Float64(obs[NQ + self.a_da[j]])
            joint_vels_to_planes(qd, self.xs)
            for k in range(IN_DIM):
                self.x.data[k] = self.xs[k]
            self.student.forward["cpu", 1](TensorRefs[1](self.x), self.y, None)
            for j in range(ACT):
                a[j] = Float64(student_act(self.y.data[j], j, False))
        # the delta rule, then the env's normalised absolute action
        for j in range(ACT):
            self.last_a[j] = a[j]
            var vv = self.variants[self.current].copy()
            var tgt = delta_target(Float64(obs[self.a_qa[j]]), a[j], j,
                                   self.lo[j], self.hi[j], vv.d_arm, vv.d_grip)
            var mid = 0.5 * (self.lo[j] + self.hi[j])
            var half = 0.5 * (self.hi[j] - self.lo[j])
            action_out[j] = Scalar[DT]((tgt - mid) / half)


def main() raises:
    if not imgui_shim_available():
        print("Dear ImGui shim not built.  Run:  pixi run build-imgui")
        return
    var args = argv()
    var positional = List[String]()
    var place_seed = 0
    var want_policy = String("")
    var record_path = String("")
    var record_episodes = 1
    var eye = List[Float64]()
    var target = List[Float64]()
    var ai = 1
    while ai < len(args):
        var a = String(args[ai])
        if a == "--seed" and ai + 1 < len(args):
            place_seed = Int(String(args[ai + 1]))
            ai += 2
            continue
        if a == "--policy" and ai + 1 < len(args):
            want_policy = String(args[ai + 1])
            ai += 2
            continue
        if a == "--record" and ai + 1 < len(args):
            record_path = String(args[ai + 1])
            ai += 2
            continue
        if (a == "--eye" or a == "--target") and ai + 1 < len(args):
            var v = List[Float64]()
            for part in String(args[ai + 1]).split(","):
                v.append(Float64(String(part)))
            if len(v) != 3:
                print("  " + a + " takes x,y,z")
                return
            if a == "--eye":
                eye = v^
            else:
                target = v^
            ai += 2
            continue
        if a == "--episodes" and ai + 1 < len(args):
            record_episodes = Int(String(args[ai + 1]))
            ai += 2
            continue
        positional.append(a)
        ai += 1
    seed(place_seed)
    var task = positional[0].copy() if len(positional) > 0 else String("so101_tower_lift_real_layout")
    var ti = task_index(task, task_names())
    if ti < 0:
        print("unknown task:", task, "— this viewer registers:")
        for n in task_names():
            print("   ", n)
        return
    print("=" * 66)
    print("so101_tower — pixel student viewer (" + String(OBS_PX) + "px build) —", task)
    print("  insets: overhead (top), wrist (below) | status: reach / goal mm, gripper word")
    print("=" * 66)
    var src = PixelViewerPolicy()
    var domains = List[String]()
    domains.append(String(FAMILY))
    var td = List[Int]()
    for _ in range(len(task_names())):
        td.append(0)
    var st = ViewerState(ti, DRIVE_POLICY, 1.0, task_names(), domains^, td^)
    st.policy_variant = 0
    var want = String("lift") if task.find("lift") >= 0 else String("bowl")
    var labels = src.variant_labels()
    var picked = False
    for i in range(len(labels)):
        if want_policy.byte_length() > 0:
            if (labels[i].find(want_policy) >= 0
                    or src.variants[i].ckpt.find(want_policy) >= 0):
                if src.variants[i].refuse.byte_length() > 0:
                    print("  --policy", want_policy, ":", src.variants[i].refuse)
                    return
                st.policy_variant = Int32(i)
                picked = True
                break
        elif labels[i].startswith(want) and src.variants[i].refuse.byte_length() == 0:
            st.policy_variant = Int32(i)
            picked = True
            break
    if want_policy.byte_length() > 0 and not picked:
        print("  --policy", want_policy, ": no such variant (see the list above)")
        return
    st.record_path = record_path
    st.record_episodes = record_episodes
    st.free_camera_eye = eye^
    st.free_camera_target = target^
    st.free_camera = True
    st.episode_steps = So101TowerConfig.MAX_STEPS
    st.frame_ms = Int(Float64(So101TowerConfig.FRAME_SKIP) * So101TowerModel.TIMESTEP * 1000.0)
    st.pip_cameras = List[Int]()
    st.pip_cameras.append(1)
    st.pip_cameras.append(0)
    var pol = Pointer(to=src).as_unsafe_any_origin()
    while not st.quit:
        var name = task_names()[st.task]
        st.reset_qpos = posed_qpos[So101TowerPlacement](
            name, String(FAMILY), So101TowerConfig.SLOT_RADIUS,
            seed=UInt64(place_seed),
        )
        var mw = task_meta_words(
            name, String(FAMILY), So101TowerConfig.SHAPE_W_GOAL,
            So101TowerConfig.SHAPE_W_REACH, So101TowerConfig.GOAL_MARGIN,
            So101TowerConfig.REACH_MARGIN,
        )
        st.reset_meta_idx = mw[0].copy()
        st.reset_meta_val = mw[1].copy()
        run_view[So101TowerModel, So101TowerConfig, PixelViewerPolicy](name, st, pol)
    _ = src
    if st.handoff:
        Renderer3D.close_handoff(st.handoff.value().copy())
        st.handoff = None

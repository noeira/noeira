# +--------------------------------------------------------------------------+ #
# | SuperMimic's prompt interface: the suite, the step test, the honesty probe
# +--------------------------------------------------------------------------+ #
"""Score a trained prompt interface (`deep_agents/supermimic/prompt_eval.mojo`
says what each number means, `noeira-docs/SUPERMIMIC_BFM_HYBRID_PLAN.md` §0
what it gates).

    # on the device: NVIDIA only
    pixi run -e nvidia mojo build -I . -D SM_GPU=2 -D SM_PAPER=1 examples/g1/supermimic_prompt_eval.mojo -o smp_eval
    ./smp_eval --tracker final.ckpt --prompt prompt.ckpt --store stores/ [--suite] [--step-test]

    # the logic on the Mac: CPU env, few lanes
    pixi run mojo run -I . examples/g1/supermimic_prompt_eval.mojo

`-D SM_PROMPT_VEL=1` scores the velocity control, which can only be asked
velocities. The store must be the one the interface trained on: z is not
portable across pools.

⚠ A SEED store is confidential: private compute only.
"""

from std.builtin.sort import sort
from std.math import sqrt
from std.os import listdir
from std.os.path import isdir
from std.sys import argv, get_defined_int

from max.gpu.host import DeviceContext

from noeira.data.store import TrajectoryStore
from noeira.deep_agents.supermimic import SMConfig
from noeira.deep_agents.supermimic.env import SMReplayOf, SMTrackingEnv
from noeira.deep_agents.supermimic.env_gpu import SMReplayGpuOf, SMTrackingEnvGpu, ST_ENDING
from noeira.deep_agents.supermimic.heads import qapply_inv, yaw_quat
from noeira.deep_agents.supermimic.kinematics import Q4, qapply
from noeira.deep_agents.supermimic.lanes import talloc, tp
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.prompt import SMSpec, sm_spec_velocity
from noeira.deep_agents.supermimic.prompt_bank import sm_prompt_data
from noeira.deep_agents.supermimic.prompt_eval import (
    SM_E_SKIP, SM_E_STEP_STEPS, SM_E_STEP_TAIL, SM_E_SUITE_STEPS, SMEvalTally, SMEvalTask, sm_eval_tasks, sm_median, sm_stand_library,
    sm_step_commands,
)
from noeira.deep_agents.supermimic.prompt_imagination import SMPromptBatch, sm_imagine_prompt
from noeira.deep_agents.supermimic.prompt_policy import SMPromptConfig
from noeira.deep_agents.supermimic.prompt_trainer import SMPromptTrainer
from noeira.deep_agents.supermimic.robot import G1MjlabBatchedSim, G1MjlabRobot, SMDims
from noeira.deep_agents.supermimic.trainer import SuperMimicTrainer
from noeira.envs.robots.unitree_g1_pd import G1_N_DOF
from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl


comptime MODE = get_defined_int["SM_GPU", 0]()
comptime TARGET: StaticString = "gpu" if MODE != 0 else "cpu"
comptime PAPER = get_defined_int["SM_PAPER", 0]() != 0
comptime VEL = get_defined_int["SM_PROMPT_VEL", 0]() != 0
comptime LANES = get_defined_int["SM_LANES", 4096 if PAPER else 8]()
comptime Robot = G1MjlabRobot
comptime D = SMDims[Robot]
comptime H = 32
comptime C = SMConfig[
    TARGET, Robot,
    2048 if PAPER else 128, 64 if PAPER else 64, 2048 if PAPER else 128, 1024 if PAPER else 64, 512 if PAPER else 64, 256 if PAPER else 32,
    2048 if PAPER else 128, 1024 if PAPER else 64, 512 if PAPER else 64, 256 if PAPER else 32, 256 if PAPER else 64, 128 if PAPER else 32,
    LANES, 16, LANES, H,
]
comptime CMD = 3 if VEL else D.LATENT
comptime P = SMPromptConfig[C, CMD, 256 if PAPER else 32, 256 if PAPER else 32, 256 if PAPER else 32, 128 if PAPER else 32]
comptime Trainer = SMPromptTrainer[P]
comptime Sim = G1MjlabBatchedSim[LANES]
comptime NOISE_AMP = 0.1
comptime STAND_CLIPS = 64
comptime HONEST_T0 = 100


trait EvalEnv(Movable):
    def restart(mut self, frames: List[Int]) raises:
        ...

    def step(mut self, mut tr: Trainer, mut cmds: TensorImpl[DT], mut specs: TensorImpl[DT]) raises:
        """One step; then `read` returns its outcome."""
        ...

    def read(mut self, mut states: List[Scalar[DT]], mut ended: List[Bool], mut rews: List[Float64]) raises:
        """states [L·S], the lanes the env ended, their task reward."""
        ...

    def snapshot(mut self, mut ib: SMPromptBatch[Robot, CMD]) raises:
        """The lanes' current states, histories, last actions and contexts."""
        ...


struct CpuEval(EvalEnv):
    var env: SMTrackingEnv[Robot]
    var rb: SMReplayOf[Robot]
    var hist: TensorImpl[DT]
    var goals: TensorImpl[DT]

    def __init__(out self, var lib: SMMotionLibrary[Robot]) raises:
        self.env = SMTrackingEnv[Robot](LANES, lib^, dr=False, obs_noise=False, reset_noise=False, action_noise=0.0, max_episode_steps=0)
        self.env.prompt_from = 0
        self.rb = SMReplayOf[Robot](LANES, 8)
        self.hist = TensorImpl[DT].alloc(LANES * D.HIST * D.OBS)
        self.goals = TensorImpl[DT].alloc(LANES * D.GOAL)

    def __init__(out self, *, deinit move: Self):
        self.env = move.env^
        self.rb = move.rb^
        self.hist = move.hist^
        self.goals = move.goals^

    def restart(mut self, frames: List[Int]) raises:
        self.env.restart(frames)

    def step(mut self, mut tr: Trainer, mut cmds: TensorImpl[DT], mut specs: TensorImpl[DT]) raises:
        self.env.policy_inputs(self.hist, self.goals)
        var a = tr.act[LANES](self.hist, self.goals, cmds, 0)
        _ = self.env.step(a.data, self.rb, specs.data.unsafe_ptr().as_unsafe_any_origin())

    def read(mut self, mut states: List[Scalar[DT]], mut ended: List[Bool], mut rews: List[Float64]) raises:
        for l in range(LANES):
            ended[l] = self.env.lanes[l].ep_len == 0
            rews[l] = self.env.last_prompt_r[l]
            for k in range(D.S):
                states[l * D.S + k] = self.env.lanes[l].s[k]

    def snapshot(mut self, mut ib: SMPromptBatch[Robot, CMD]) raises:
        self.env.policy_inputs(self.hist, self.goals)
        for l in range(LANES):
            for k in range(D.S):
                ib.state.data[l * D.S + k] = self.env.lanes[l].s[k]
            for k in range(D.X):
                ib.xi.data[l * D.X + k] = self.env.lanes[l].xi[k]
            for k in range(Robot.NDOF):
                ib.prev_action.data[l * Robot.NDOF + k] = self.env.lanes[l].a_last[k]
            for hh in range(D.HIST):
                for k in range(D.OBS):
                    ib.hist0.data[(hh * LANES + l) * D.OBS + k] = self.hist.data[(l * D.HIST + hh) * D.OBS + k]


struct GpuEval(EvalEnv):
    var ctx: DeviceContext
    var env: SMTrackingEnvGpu[Robot, Sim]
    var rb: SMReplayGpuOf[Robot]
    var goals: TensorImpl[DT]

    def __init__(out self, ctx: DeviceContext, var lib: SMMotionLibrary[Robot]) raises:
        self.ctx = ctx
        self.env = SMTrackingEnvGpu[Robot, Sim](
            ctx, lib^, Sim.create(ctx), dr=False, obs_noise=False, reset_noise=False, action_noise=0.0, max_episode_steps=0,
        )
        self.env.prompt_from = 0
        self.rb = SMReplayGpuOf[Robot](ctx, LANES, 8)
        self.goals = talloc["gpu"](LANES * D.GOAL, Optional(ctx))

    def __init__(out self, *, deinit move: Self):
        self.ctx = move.ctx
        self.env = move.env^
        self.rb = move.rb^
        self.goals = move.goals^

    def restart(mut self, frames: List[Int]) raises:
        self.env.restart(self.ctx, frames)

    def step(mut self, mut tr: Trainer, mut cmds: TensorImpl[DT], mut specs: TensorImpl[DT]) raises:
        cmds.upload_resident(self.ctx)
        specs.upload_resident(self.ctx)
        var a = tr.act[LANES](self.env.hist, self.goals, cmds, 0, resident=True)
        _ = self.env.step(self.ctx, a, self.rb, tp["gpu"](specs))

    def read(mut self, mut states: List[Scalar[DT]], mut ended: List[Bool], mut rews: List[Float64]) raises:
        self.env.s.download(self.ctx)
        self.env.prompt_r.download(self.ctx)
        for l in range(LANES):
            ended[l] = (self.env.h_out_i[2 * l + 1] & ST_ENDING) != 0
            rews[l] = Float64(self.env.prompt_r.data[l])
        for i in range(LANES * D.S):
            states[i] = self.env.s.data[i]

    def snapshot(mut self, mut ib: SMPromptBatch[Robot, CMD]) raises:
        self.env.s.download(self.ctx)
        self.env.xi.download(self.ctx)
        self.env.a_last.download(self.ctx)
        self.env.hist.download(self.ctx)
        for l in range(LANES):
            for k in range(D.S):
                ib.state.data[l * D.S + k] = self.env.s.data[l * D.S + k]
            for k in range(D.X):
                ib.xi.data[l * D.X + k] = self.env.xi.data[l * D.X + k]
            for k in range(Robot.NDOF):
                ib.prev_action.data[l * Robot.NDOF + k] = self.env.a_last.data[l * Robot.NDOF + k]
            for hh in range(D.HIST):
                for k in range(D.OBS):
                    ib.hist0.data[(hh * LANES + l) * D.OBS + k] = self.env.hist.data[(l * D.HIST + hh) * D.OBS + k]


def _stand_frames(lib_starts: List[Int]) -> List[Int]:
    var out = List[Int](length=LANES, fill=0)
    for l in range(LANES):
        out[l] = lib_starts[l % len(lib_starts)]
    return out^


def suite[E: EvalEnv](mut e: E, mut tr: Trainer, tasks: List[SMEvalTask], frames: List[Int]) raises:
    """The suite, and the honesty probe at step `HONEST_T0` on the same lanes."""
    var nt = len(tasks)
    var task_of = List[Int](length=LANES, fill=0)
    # host-filled: `step` uploads them (`talloc` on the device has no host list)
    var cmds = TensorImpl[DT].alloc(LANES * CMD)
    var specs = TensorImpl[DT].alloc(LANES * SMSpec.SIZE)
    for l in range(LANES):
        var k = l % nt
        task_of[l] = k
        for i in range(CMD):
            cmds.data[l * CMD + i] = tasks[k].cmd[i]
        for i in range(SMSpec.SIZE):
            specs.data[l * SMSpec.SIZE + i] = tasks[k].spec[i]
    print("suite:", nt, "tasks,", LANES // nt, "lanes each,", SM_E_SUITE_STEPS, "steps")
    e.restart(frames)
    var tally = SMEvalTally(LANES, D.FK_BLOCK)
    var states = List[Scalar[DT]](length=LANES * D.S, fill=0)
    var ended = List[Bool](length=LANES, fill=False)
    var rews = List[Float64](length=LANES, fill=0.0)
    var ib = SMPromptBatch[Robot, CMD](LANES, H, 0)
    var imag = List[Float64](length=LANES, fill=0.0)
    var sim = List[Float64](length=LANES, fill=0.0)
    var live = List[Bool](length=LANES, fill=True)
    for t in range(SM_E_SUITE_STEPS):
        if t == HONEST_T0:
            e.snapshot(ib)
            for tt in range(H + 1):
                for l in range(LANES):
                    for i in range(CMD):
                        ib.cmds.data[(tt * LANES + l) * CMD + i] = cmds.data[l * CMD + i]
                    if tt < H:
                        for i in range(SMSpec.SIZE):
                            ib.specs.data[(tt * LANES + l) * SMSpec.SIZE + i] = specs.data[l * SMSpec.SIZE + i]
            comptime if TARGET == "gpu":
                ib.to_device(tr.base.ctx.value())
            var o = sm_imagine_prompt[P, LANES, TARGET](
                tr.base.wm, tr.base.pol, tr.enc, tr.critic, tr.base.tables, ib, C.DEF_GAMMA, 0.0, False, 0, tr.base.ctx,
            )
            comptime if TARGET == "gpu":
                o.to_host(tr.base.ctx.value())
            for tt in range(H):
                for l in range(LANES):
                    imag[l] += Float64(o.rewards.data[tt * LANES + l])
        e.step(tr, cmds, specs)
        e.read(states, ended, rews)
        for l in range(LANES):
            tally.step[Robot](tr.base.tables, l, t, states.unsafe_ptr().unsafe_offset(l * D.S).as_unsafe_any_origin(), ended[l], tasks[task_of[l]])
            if t >= SM_E_SKIP and not tally.ended[l]:
                tally.trained[l] += rews[l]
            if t >= HONEST_T0 and t < HONEST_T0 + H:
                sim[l] += rews[l]
                if ended[l]:
                    live[l] = False
    tally.report(tasks, task_of)
    print("--- honesty, reward over", H, "steps from step", HONEST_T0, ", the lanes alive throughout: imagined | simulated | mean |gap|")
    for g in range(4):
        var si = 0.0
        var ss = 0.0
        var sg = 0.0
        var n = 0
        for l in range(LANES):
            if live[l] and tasks[task_of[l]].family == g and not tally.ended[l]:
                si += imag[l]
                ss += sim[l]
                sg += abs(imag[l] - sim[l])
                n += 1
        if n > 0:
            print(" ", g, ":", si / Float64(n), "|", ss / Float64(n), "|", sg / Float64(n), "(", n, "lanes )")


def step_test[E: EvalEnv](mut e: E, mut tr: Trainer, cmd_of: List[Scalar[DT]], frames: List[Int]) raises:
    """Appendix G.2. `cmd_of` [672·CMD]: the interface's command for each."""
    var cl = sm_step_commands()
    var nc = min(len(cl), LANES // 4)
    # host-filled: `step` uploads them (`talloc` on the device has no host list)
    var cmds = TensorImpl[DT].alloc(LANES * CMD)
    var specs = TensorImpl[DT].alloc(LANES * SMSpec.SIZE)
    for l in range(LANES):
        var k = (l // 4) % nc
        for i in range(CMD):
            cmds.data[l * CMD + i] = cmd_of[k * CMD + i]
        sm_spec_velocity[DT](
            specs.data.unsafe_ptr().unsafe_offset(l * SMSpec.SIZE).as_unsafe_any_origin(),
            Scalar[DT](cl[k].vx), Scalar[DT](cl[k].vy), Scalar[DT](cl[k].wz),
        )
    print("step test:", nc, "commands, 4 trials each")
    e.restart(frames)
    var states = List[Scalar[DT]](length=LANES * D.S, fill=0)
    var ended = List[Bool](length=LANES, fill=False)
    var rews = List[Float64](length=LANES, fill=0.0)
    var fell = List[Bool](length=LANES, fill=False)
    var mv = List[Float64](length=LANES * 3, fill=0.0)
    for t in range(SM_E_STEP_STEPS):
        e.step(tr, cmds, specs)
        e.read(states, ended, rews)
        for l in range(LANES):
            fell[l] = fell[l] or ended[l]
            if t >= SM_E_STEP_STEPS - SM_E_STEP_TAIL and not fell[l]:
                var o = sm_step_velocity(states.unsafe_ptr().unsafe_offset(l * D.S).as_unsafe_any_origin())
                mv[l * 3] += o[0] / Float64(SM_E_STEP_TAIL)
                mv[l * 3 + 1] += o[1] / Float64(SM_E_STEP_TAIL)
                mv[l * 3 + 2] += o[2] / Float64(SM_E_STEP_TAIL)
    var per = List[Float64]()
    var all_fell = 0
    var n_fall = 0
    for k in range(nc):
        var errs = List[Float64]()
        for tr_i in range(4):
            var l = k * 4 + tr_i
            if l >= LANES:
                continue
            if fell[l]:
                n_fall += 1
                continue
            var dx = mv[l * 3] - cl[k].vx
            var dy = mv[l * 3 + 1] - cl[k].vy
            var dw = mv[l * 3 + 2] - cl[k].wz
            var nc2 = cl[k].vx * cl[k].vx + cl[k].vy * cl[k].vy + cl[k].wz * cl[k].wz
            errs.append(sqrt((dx * dx + dy * dy + dw * dw) / nc2))
        if len(errs) == 0:
            all_fell += 1
        else:
            per.append(sm_median(errs))
    print("median relative error", sm_median(per), "over", len(per), "commands (the paper: 0.16, distillation 0.49)")
    print("trials fallen", n_fall, "| commands fallen in every trial", all_fell)


def sm_step_velocity(s: Pointer[Scalar[DT], MutAnyOrigin]) -> Tuple[Float64, Float64, Float64]:
    """(vx, vy, wz): the heading frame's velocity and the world yaw rate."""
    var q = Q4[DT](s[unsafe_offset=3], s[unsafe_offset=4], s[unsafe_offset=5], s[unsafe_offset=6])
    var v = Q4[DT](s[unsafe_offset=D.NQ], s[unsafe_offset=D.NQ + 1], s[unsafe_offset=D.NQ + 2], 0)
    var w = Q4[DT](s[unsafe_offset=D.NQ + 3], s[unsafe_offset=D.NQ + 4], s[unsafe_offset=D.NQ + 5], 0)
    var vh = qapply_inv(yaw_quat(q), v)
    return (Float64(vh[0]), Float64(vh[1]), Float64(qapply(q, w)[2]))


def _opt(args: List[String], name: String, fallback: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == name:
            return String(args[i + 1])
    return fallback


def _flag(args: List[String], name: String) -> Bool:
    for i in range(len(args)):
        if args[i] == name:
            return True
    return False


def _store_paths(spec: String) raises -> List[String]:
    var out = List[String]()
    if isdir(spec):
        var names = List[String]()
        for n in listdir(spec):
            if String(n).endswith(".h5"):
                names.append(String(n))
        sort(names)
        for k in range(len(names)):
            out.append(spec + "/" + names[k])
    else:
        for part in spec.split(","):
            out.append(String(part))
    return out^


def _from_store(spec: String, max_clips: Int) raises -> SMMotionLibrary[Robot]:
    var paths = _store_paths(spec)
    var lens = List[Int]()
    var rows = List[Scalar[DT]]()
    for p in range(len(paths)):
        var st = TrajectoryStore(paths[p])
        var refc = st.load_column[DType.float32](String("ref"))
        var n_rows = 0
        for e in range(st.n_episodes()):
            if max_clips > 0 and len(lens) >= max_clips:
                break
            var l = Int(st.episodes.ep_len[e])
            lens.append(l)
            n_rows += l
        var o = len(rows)
        rows.resize(o + n_rows * D.S, 0)
        for i in range(n_rows * D.S):
            rows[o + i] = refc[i]
    var q0 = List[Float64](length=G1_N_DOF, fill=0.0)
    for j in range(G1_N_DOF):
        q0[j] = Robot.default_joint_pos(j)
    return SMMotionLibrary[Robot](rows^, lens, q0)


def run[E: EvalEnv](
    mut e: E, mut tr: Trainer, tasks: List[SMEvalTask], cmd_of: List[Scalar[DT]], frames: List[Int], do_suite: Bool, do_step: Bool,
) raises:
    if do_suite:
        suite(e, tr, tasks, frames)
    if do_step:
        step_test(e, tr, cmd_of, frames)


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))
    var tracker = _opt(args, String("--tracker"), String(""))
    var prompt = _opt(args, String("--prompt"), String(""))
    var store = _opt(args, String("--store"), String(""))
    var bank = Int(_opt(args, String("--bank"), String("4096" if PAPER else "64")))
    var pool = Int(_opt(args, String("--pool"), String("131072")))
    var z_beta = Float64(_opt(args, String("--z-beta"), String("1")))
    var z_ess = Float64(_opt(args, String("--z-ess"), String("0")))
    var max_clips = Int(_opt(args, String("--max-clips"), String("0")))
    var do_suite = _flag(args, String("--suite")) or not _flag(args, String("--step-test"))
    var do_step = _flag(args, String("--step-test"))
    if store == "":
        raise Error("give --store: z is computed over its frames")
    var lib = _from_store(store, max_clips)
    var ctx: Optional[DeviceContext] = None
    comptime if TARGET == "gpu":
        ctx = DeviceContext()
    var sim0 = Robot.SIM.create()
    var root_z = sim0.qpos(2)
    var base = SuperMimicTrainer[C].for_robot(Robot.SIM.create(), ctx)
    if tracker != "":
        print("tracker at iteration", base.load_state(tracker)[0])
    var data = sm_prompt_data[C, CMD](base, lib, bank, pool, z_beta, z_ess)
    var tr = Trainer(base^)
    if prompt != "":
        print("prompt interface at iteration", tr.load_state(prompt)[0])
    var tasks = sm_eval_tasks[CMD](data)
    # every step-test command as the interface reads it
    var cl = sm_step_commands()
    var cmd_of = List[Scalar[DT]](length=len(cl) * CMD, fill=0)
    if do_step:
        var spec = List[Scalar[DT]](length=SMSpec.SIZE, fill=0)
        for k in range(len(cl)):
            sm_spec_velocity[DT](spec.unsafe_ptr().as_unsafe_any_origin(), Scalar[DT](cl[k].vx), Scalar[DT](cl[k].vy), Scalar[DT](cl[k].wz))
            comptime if VEL:
                cmd_of[k * CMD] = Scalar[DT](cl[k].vx / 2.0)
                cmd_of[k * CMD + 1] = Scalar[DT](cl[k].vy / 1.32)
                cmd_of[k * CMD + 2] = Scalar[DT](cl[k].wz / 3.35)
            else:
                _ = data.z_of(spec.unsafe_ptr().as_unsafe_any_origin(), cmd_of.unsafe_ptr().unsafe_offset(k * CMD).as_unsafe_any_origin())
    var stand = sm_stand_library[Robot](STAND_CLIPS, SM_E_SUITE_STEPS + 40, NOISE_AMP, root_z)
    var starts = stand.clip_start.copy()
    var frames = _stand_frames(starts)
    comptime if MODE == 2:
        var e = GpuEval(ctx.value(), stand^)
        run(e, tr, tasks, cmd_of, frames, do_suite, do_step)
    else:
        var e = CpuEval(stand^)
        run(e, tr, tasks, cmd_of, frames, do_suite, do_step)

"""BFM-Zero G1 rollouts as FB-MEBE's prior rows — `noeira-docs/FB_MEBE_G1_PLAN.md`
§13.5 option 1.

The teacher (a BFM-Zero G1 checkpoint) drives `UnitreeG1Batched` — the env
FB-MEBE explores — under the base-command suite's own prompts, and every
transition is written to a `TrajectoryStore` in `fb/prior.mojo`'s
convention:

    obs     float32 [527]   the env's stored observation, pre-step
    action  float32 [29]    the teacher's action (clamped, + `--expl-std` noise)
    r_env   float32 [1]     `g1_reg_env_kernel` at its default weights — the
                            simulator half FB-MEBE's ring stores per row
    task    int32   [1]     the prompt's index in `--tasks`

and every episode ends with its post-step TERMINAL row (action 0, r_env 0,
task -1), so each transition has its real successor.

The z of a prompt is NOT inferred here: `bfm_zero_suite_eval --dump-z`
writes it on the laptop, from the LAFAN pool the teacher was trained on, so
the box needs no LAFAN. A lane holds its prompt `--hold` steps (staggered
per lane), then draws another; a fall (pelvis < 0.45 m or tilt > 60° for
0.5 s, FB-MEBE's rule) or the env's time limit ends the episode, and the
lane restarts from the stand keyframe with ±`--reset-noise` rad per joint,
as FB-MEBE's lanes do. The teacher's 401-D history tail is maintained per
lane by the BFM-Zero driver's kernels, reset per lane.

    pixi run -e nvidia mojo build -I . examples/g1/bfm_zero_prior_collect.mojo -o bfm_prior_collect
    ./bfm_prior_collect --ckpt projects/g1/policies/bfm_room.ckpt \\
        --z g1_teacher_z_s36000.txt --rows 1000000 --out g1_prior_s36000.h5

⚠ NVIDIA: like FB-MEBE's driver, the G1 fields path does not run on Metal
at 1024 lanes in useful time.
"""

from std.math import abs
from std.random import seed, random_float64
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from noeira.nn.constants import DT, TPB
from noeira.nn.core.ptr import mptr
from noeira.data.column import ColumnSpec
from noeira.data.store import TrajectoryStoreWriter
from noeira.deep_agents.fb import FBOnlineAgent
from noeira.deep_agents.fb.kernels import _blocks
from noeira.deep_agents.fb.bfm_towers import (
    BFMFTower, BFMBNetFiltered, BFMActorTowerFiltered,
)
from noeira.deep_agents.fb.g1_mebe import (
    G1RegWeights,
    G1M_COL_VX, G1M_COL_VY, G1M_COL_WZ, G1M_COL_H,
    g1_fall_kernel,
    g1_reset_noise_kernel,
    g1_reg_env_kernel,
)
from noeira.envs.robots import UnitreeG1Batched
from noeira.envs.robots.unitree_g1_xml import (
    UnitreeG1Model, UNITREE_G1_OBS_DIM, UNITREE_G1_STATE_DIM,
)
from noeira.envs.robots.unitree_g1_history import (
    UNITREE_G1_FULL_OBS_DIM, G1_ACTOR_EXTRA, G1_HIST_DIM, G1_N_ACT,
    G1_HIST_STEP,
    g1_pack_full_obs_kernel, g1_hist_push_kernel, g1_hist_reset_kernel,
    g1_scale_clip_kernel,
)
from noeira.envs.robots.unitree_g1_pd import G1_NORMALIZE_TO, G1_ACTION_CLIP
from noeira.envs.robots.g1_tracking_eval import G1_D, G1_H, G1_L, G1_HB


comptime N_ENVS: Int = 1024
comptime SP: Int = UNITREE_G1_OBS_DIM          # 527, what the store holds
comptime OBS: Int = UNITREE_G1_FULL_OBS_DIM    # 928, what the actor reads
comptime ACT: Int = UnitreeG1Model.ACTION_DIM  # 29
comptime D: Int = G1_D
comptime BATCH: Int = 256       # the trainer's scratch; nothing is trained
comptime CAP: Int = 2048
comptime EnvT = UnitreeG1Batched[N_ENVS]
comptime FNet = BFMFTower[OBS, ACT, D, G1_H, G1_L, D]
comptime BNet = BFMBNetFiltered[OBS, SP, D, G1_HB]
comptime ANet = BFMActorTowerFiltered[
    OBS, UNITREE_G1_STATE_DIM, G1_ACTOR_EXTRA, D, G1_H, G1_L, ACT
]
comptime Teacher = FBOnlineAgent[
    FNet, BNet, ANet, OBS, ACT, D, BATCH, CAP, N_ENVS, 256, 0, G1_ACTOR_EXTRA
]

comptime FALL_H: Float64 = 0.45
comptime FALL_GZ: Float64 = -0.5
comptime FALL_DELAY: Int = 25

# the suite's 13 velocity tasks and standing — the teacher's strength (§13.6)
comptime DEFAULT_TASKS = (
    "vx-0.5,vx0.0,vx0.5,vx1.0,vx1.5,vy-0.3,vy0.3,vx0.5_vy-0.3,vx0.5_vy0.3,"
    "wz-1.0,wz-0.5,wz0.5,wz1.0,h0.78"
)


def _flag(name: String, dflt: String) raises -> String:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def _read_z(path: String, prompt: String, ref names: List[String]) raises -> List[Float32]:
    """`len(names) * D` floats: the `z <task> <prompt> v_1 .. v_D` lines of
    `bfm_zero_suite_eval --dump-z`, in `names` order."""
    var out = List[Float32](length=len(names) * D, fill=0)
    var found = List[Bool](length=len(names), fill=False)
    var text: String
    with open(path, "r") as fh:
        text = fh.read()
    for line in text.split("\n"):
        var f = String(line).split(" ")
        if len(f) < 3 or String(f[0]) != "z" or String(f[2]) != prompt:
            continue
        for k in range(len(names)):
            if String(f[1]) == names[k]:
                if len(f) != 3 + D:
                    raise Error("z line for " + names[k] + ": " + String(len(f) - 3) + " values, want " + String(D))
                for j in range(D):
                    out[k * D + j] = Float32(atof(String(f[3 + j])))
                found[k] = True
    for k in range(len(names)):
        if not found[k]:
            raise Error("no '" + prompt + "' z for task " + names[k] + " in " + path)
    return out^


def main() raises:
    var ckpt = _flag(String("--ckpt"), String("projects/g1/policies/bfm_room.ckpt"))
    var z_path = _flag(String("--z"), String(""))
    var prompt = _flag(String("--prompt"), String("centred"))
    var out_path = _flag(String("--out"), String("g1_prior.h5"))
    var target = atol(_flag(String("--rows"), String(1_000_000)))
    var hold = atol(_flag(String("--hold"), String(150)))
    var expl_std = atof(_flag(String("--expl-std"), String(0.05)))
    var reset_noise = atof(_flag(String("--reset-noise"), String(0.1)))
    var seed_v = atol(_flag(String("--seed"), String(1)))
    var tasks_s = _flag(String("--tasks"), String(DEFAULT_TASKS))
    if z_path.byte_length() == 0:
        raise Error("--z FILE is required (bfm_zero_suite_eval --dump-z)")
    var names = List[String]()
    for t in tasks_s.split(","):
        names.append(String(t))
    var n_task = len(names)
    var zs = _read_z(z_path, prompt, names)
    seed(seed_v)

    print("=" * 72)
    print("BFM-Zero G1 teacher rollouts -> FB-MEBE prior rows (§13.5)")
    print("  teacher", ckpt, "  z", z_path, "(", prompt, ")")
    print("  tasks", n_task, ":", tasks_s)
    print("  rows", target, " hold", hold, " expl_std", expl_std, " reset noise ±", reset_noise,
          " lanes", N_ENVS, " ->", out_path)
    print("=" * 72)

    with DeviceContext() as ctx:
        var t = Teacher.make(ctx, normalize_obs=True, seed=UInt64(seed_v))
        t.load_state(ckpt)
        var env = EnvT(ctx)
        var rw = G1RegWeights.default()

        var prev_obs = ctx.enqueue_create_buffer[DT](N_ENVS * SP)
        var full_obs = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
        var h_last = ctx.enqueue_create_buffer[DT](N_ENVS * G1_N_ACT)
        var h_hist = ctx.enqueue_create_buffer[DT](N_ENVS * G1_HIST_DIM)
        var h_live = ctx.enqueue_create_buffer[DT](N_ENVS)
        var r_env = ctx.enqueue_create_buffer[DT](N_ENVS)
        var done_rec = ctx.enqueue_create_buffer[DT](N_ENVS)
        var fall_cnt = ctx.enqueue_create_buffer[DT](N_ENVS)
        var n_falls = ctx.enqueue_create_buffer[DT](N_ENVS)
        var z_dev = ctx.enqueue_create_buffer[DT](N_ENVS * D)
        var ones = ctx.enqueue_create_buffer[DT](N_ENVS)
        fall_cnt.enqueue_fill(Scalar[DT](0.0))
        n_falls.enqueue_fill(Scalar[DT](0.0))
        ones.enqueue_fill(Scalar[DT](1.0))
        var h_obs = ctx.enqueue_create_host_buffer[DT](N_ENVS * SP)
        var h_post = ctx.enqueue_create_host_buffer[DT](N_ENVS * SP)
        var h_act = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT)
        var h_r = ctx.enqueue_create_host_buffer[DT](N_ENVS)
        var h_done = ctx.enqueue_create_host_buffer[DT](N_ENVS)
        var h_z = ctx.enqueue_create_host_buffer[DT](N_ENVS * D)

        def _after_reset(s: UInt64) capturing raises:
            """Joint noise on the lanes the reset just touched, then FK,
            velocities and the obs — `fb_mebe_g1_gpu.mojo`'s sequence."""
            if reset_noise > 0.0:
                ctx.enqueue_function[g1_reset_noise_kernel[EnvT.NQ, N_ENVS]](
                    mptr(env.d.qpos.dev.value().unsafe_ptr()),
                    mptr(env._reset_mask.unsafe_ptr()),
                    Scalar[DT](reset_noise), s,
                    grid_dim=_blocks(N_ENVS), block_dim=TPB,
                )
            env._run_fields_fk(ctx)
            env._run_fields_vel(ctx)
            env._extract_obs_only(ctx)

        def _hist_reset(mask: DeviceBuffer[DT]) capturing raises:
            ctx.enqueue_function[g1_hist_reset_kernel[N_ENVS]](
                mptr(h_last.unsafe_ptr()), mptr(h_hist.unsafe_ptr()),
                mptr(h_live.unsafe_ptr()), mptr(mask.unsafe_ptr()),
                grid_dim=_blocks(N_ENVS * (G1_N_ACT + G1_HIST_DIM)), block_dim=TPB,
            )

        # ── per-lane state ────────────────────────────────────────────────
        var task = List[Int](length=N_ENVS, fill=0)
        var left = List[Int](length=N_ENVS, fill=0)
        def _draw(l: Int) capturing:
            task[l] = Int(random_float64() * Float64(n_task)) % n_task
            for j in range(D):
                h_z[l * D + j] = Scalar[DT](zs[task[l] * D + j])
        for l in range(N_ENVS):
            _draw(l)
            left[l] = 1 + Int(random_float64() * Float64(hold))   # staggered
        ctx.enqueue_copy(z_dev, h_z)
        var b_obs = List[List[Float32]]()
        var b_act = List[List[Float32]]()
        var b_r = List[List[Float32]]()
        var b_task = List[List[Int32]]()
        for _ in range(N_ENVS):
            b_obs.append(List[Float32]())
            b_act.append(List[Float32]())
            b_r.append(List[Float32]())
            b_task.append(List[Int32]())
        # per-task readouts: rows, Σvx, Σvy, Σωz, Σh, episodes ended by a fall
        var st = List[Float64](length=n_task * 6, fill=0.0)

        var cols = List[ColumnSpec]()
        cols.append(ColumnSpec(String("obs"), DType.float32, SP))
        cols.append(ColumnSpec(String("action"), DType.float32, ACT))
        cols.append(ColumnSpec(String("r_env"), DType.float32, 1))
        cols.append(ColumnSpec(String("task"), DType.int32, 1))
        var w = TrajectoryStoreWriter(
            out_path, cols^, env_id=String("builtin:unitree_g1"), seed=seed_v,
        )
        for k in range(n_task):
            w.add_task(k, names[k] + " " + prompt)
        var written = 0
        var episodes = 0

        def _write(l: Int, ref term: HostBuffer[DT]) capturing raises:
            """Lane `l`'s buffered transitions + its terminal row (the
            post-step obs in `term`) as one episode."""
            var n = len(b_r[l])
            if n == 0:
                return
            for k in range(SP):
                b_obs[l].append(Float32(term[l * SP + k]))
            for _ in range(ACT):
                b_act[l].append(Float32(0))
            b_r[l].append(Float32(0))
            b_task[l].append(Int32(-1))
            w.append[DType.float32](String("obs"), b_obs[l].unsafe_ptr().as_unsafe_any_origin(), n + 1)
            w.append[DType.float32](String("action"), b_act[l].unsafe_ptr().as_unsafe_any_origin(), n + 1)
            w.append[DType.float32](String("r_env"), b_r[l].unsafe_ptr().as_unsafe_any_origin(), n + 1)
            w.append[DType.int32](String("task"), b_task[l].unsafe_ptr().as_unsafe_any_origin(), n + 1)
            w.end_episode()
            written += n + 1
            episodes += 1
            b_obs[l].clear()
            b_act[l].clear()
            b_r[l].clear()
            b_task[l].clear()

        env.reset_batch[N_ENVS](Optional(ctx), UInt64(seed_v))
        env._done.enqueue_fill(Scalar[DT](1.0))
        env.selective_reset_batch[N_ENVS](Optional(ctx), UInt64(seed_v))
        _after_reset(UInt64(seed_v) * 1000003)
        h_last.enqueue_fill(Scalar[DT](0.0))
        h_hist.enqueue_fill(Scalar[DT](0.0))
        _hist_reset(ones)

        var t0 = perf_counter_ns()
        var s = 0
        while written < target:
            ctx.enqueue_copy(prev_obs, env._obs)
            ctx.enqueue_function[g1_pack_full_obs_kernel[N_ENVS]](
                mptr(env._obs.unsafe_ptr()), mptr(h_last.unsafe_ptr()),
                mptr(h_hist.unsafe_ptr()), mptr(full_obs.unsafe_ptr()),
                grid_dim=_blocks(N_ENVS * OBS), block_dim=TPB,
            )
            t._policy_into_z(
                mptr(full_obs.unsafe_ptr()), mptr(z_dev.unsafe_ptr()),
                mptr(env._action.unsafe_ptr()), expl_std,
            )
            # push(state_t, a_{t-1}) then last_action = a_t (the BFM driver's order)
            ctx.enqueue_function[g1_hist_push_kernel[N_ENVS]](
                mptr(env._obs.unsafe_ptr()), mptr(h_last.unsafe_ptr()),
                mptr(h_hist.unsafe_ptr()), mptr(h_live.unsafe_ptr()),
                grid_dim=_blocks(N_ENVS * G1_HIST_STEP), block_dim=TPB,
            )
            h_live.enqueue_fill(Scalar[DT](1.0))
            ctx.enqueue_function[g1_scale_clip_kernel[N_ENVS * G1_N_ACT]](
                mptr(h_last.unsafe_ptr()), mptr(env._action.unsafe_ptr()),
                Scalar[DT](G1_NORMALIZE_TO), Scalar[DT](G1_ACTION_CLIP),
                grid_dim=_blocks(N_ENVS * G1_N_ACT), block_dim=TPB,
            )
            env.step_batch[N_ENVS](Optional(ctx), UInt64(seed_v) + UInt64(s))
            ctx.enqueue_function[
                g1_reg_env_kernel[N_ENVS, EnvT.NQ, EnvT.NBODY, EnvT.MC]
            ](
                env.d.qpos.tt_dyn["gpu", type_of(env.d).L_QPOS](type_of(env.d).L_QPOS()),
                env.d.xquat.tt_dyn["gpu", type_of(env.d).L_B4](type_of(env.d).L_B4()),
                env.d.xvel.tt_dyn["gpu", type_of(env.d).L_B3](type_of(env.d).L_B3()),
                env.d.contacts.tt_dyn["gpu", type_of(env.d).L_CONTACTS](type_of(env.d).L_CONTACTS()),
                env.d.meta.tt_dyn["gpu", type_of(env.d).L_META](type_of(env.d).L_META()),
                mptr(r_env.unsafe_ptr()),
                Scalar[DT](rw.undesired), Scalar[DT](rw.feet_orient),
                Scalar[DT](rw.limits), Scalar[DT](rw.slide),
                grid_dim=_blocks(N_ENVS), block_dim=TPB,
            )
            ctx.enqueue_function[g1_fall_kernel[SP, N_ENVS]](
                mptr(env._obs.unsafe_ptr()), mptr(env._done.unsafe_ptr()),
                mptr(fall_cnt.unsafe_ptr()), mptr(done_rec.unsafe_ptr()),
                mptr(n_falls.unsafe_ptr()),
                Scalar[DT](FALL_H), Scalar[DT](FALL_GZ), Int32(FALL_DELAY),
                grid_dim=_blocks(N_ENVS), block_dim=TPB,
            )
            ctx.enqueue_copy(h_obs, prev_obs)
            ctx.enqueue_copy(h_act, env._action)
            ctx.enqueue_copy(h_r, r_env)
            ctx.enqueue_copy(h_done, done_rec)
            ctx.enqueue_copy(h_post, env._obs)   # post-step, pre-reset
            ctx.synchronize()

            var z_changed = False
            for l in range(N_ENVS):
                var k = task[l]
                for c in range(SP):
                    b_obs[l].append(Float32(h_obs[l * SP + c]))
                for c in range(ACT):
                    b_act[l].append(Float32(h_act[l * ACT + c]))
                b_r[l].append(Float32(h_r[l]))
                b_task[l].append(Int32(k))
                st[k * 6] += 1.0
                st[k * 6 + 1] += Float64(h_post[l * SP + G1M_COL_VX])
                st[k * 6 + 2] += Float64(h_post[l * SP + G1M_COL_VY])
                st[k * 6 + 3] += Float64(h_post[l * SP + G1M_COL_WZ])
                st[k * 6 + 4] += Float64(h_post[l * SP + G1M_COL_H])
                if h_done[l] > Scalar[DT](0.5):
                    if Float64(h_post[l * SP + G1M_COL_H]) < FALL_H:
                        st[k * 6 + 5] += 1.0
                    _write(l, h_post)
                    _draw(l)
                    left[l] = hold
                    z_changed = True
                else:
                    left[l] -= 1
                    if left[l] <= 0:
                        _draw(l)
                        left[l] = hold
                        z_changed = True
            if z_changed:
                ctx.enqueue_copy(z_dev, h_z)
            env.selective_reset_batch[N_ENVS](Optional(ctx), UInt64(seed_v) + UInt64(s))
            _after_reset(UInt64(seed_v) * 1000003 + UInt64(s + 1))
            _hist_reset(done_rec)
            s += 1
            if s % 100 == 0:
                var el = Float64(perf_counter_ns() - t0) / 1e9
                print("  step", s, " rows", written, "/", target, " episodes", episodes,
                      " ", Float64(s * N_ENVS) / el, "env st/s")

        # the lanes still running: their successor is the current obs
        ctx.enqueue_copy(h_post, env._obs)
        ctx.synchronize()
        for l in range(N_ENVS):
            _write(l, h_post)
        w.close()

        print("-" * 72)
        print("  wrote", written, "rows in", episodes, "episodes ->", out_path)
        print("TASK name rows mean_vx mean_vy mean_wz mean_h falls")
        for k in range(n_task):
            var n = max(st[k * 6], 1.0)
            print("TASK", names[k], Int(st[k * 6]), st[k * 6 + 1] / n, st[k * 6 + 2] / n,
                  st[k * 6 + 3] / n, st[k * 6 + 4] / n, Int(st[k * 6 + 5]))

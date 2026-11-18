"""The object interface's benchmark: every clip from its first frame, with the box.

`eval.sm_benchmark_run`'s protocol (each clip from frame 0 to its end or to a
fall; no noise, no randomisation, no step cap) over the simulator that carries
the object, with what `docs/OBJECT_INTERACTION_PLAN.md`'s O1.5 asks of the box:

    carry step   a step whose REFERENCE box is at least `SM_OBJ_LIFT` above the
                 clip's first reference height (the reference is carrying)
    held step    a carry step with the simulated box within `SM_OBJ_NEAR` of
                 its reference
    per clip     fell (the tracker's termination), carry steps, held steps,
                 the largest and the mean distance over the carry steps, and
                 the tracker's normalised return (its eight terms)

A clip SUCCEEDS (is "at the bar") when the robot did not fall, the reference
carries at some point, and the box was held on at least `SM_OBJ_HELD_SHARE`
of the carry steps. `strict` is the same with every carry step held.

The bar is not the carry. A run is CARRIED when the robot did not fall and the
simulated box is up on more than half of the carry's second half; its error
once in hand is the mean distance over that half
(`object_task.SMObjCarryTally`, the one rule, shared with the CPU trace
driver). Report the three side by side.

Tracking is reported twice: the tracker's normalised return (eight terms, all
14 bodies), and the six kernels with the body kernels taken over the trunk and
the legs only (`heads.sm_tracking_kernels_lane` under
`object_task.sm_trunk_legs_mask`), each over its maximum. The arms must leave
the reference to hold a box the reference pantomimes, so O1.5's tracking
criterion is the second, against the blind tracker's on the same clips. The
kernels are computed on the host from each lane's state after the step; a
run's last step (the lane is already reset) is not in them.

`trials` > 1 runs every clip that many times, each from a start perturbed by
the training's reset noise (the robot's root and joints; the box stays at its
reference). A pickup succeeds or fails as a whole, so ONE deterministic trial
per clip gives a score that moves by 20 points between neighbouring
checkpoints (`docs/OBJECT_INTERACTION_PLAN.md` §6, run 6); several trials are
what lets two close arms be ranked. `trials` = 1 is the deterministic
benchmark, unchanged.

With the object encoder at its zero start (or `gain` 0) this is the BLIND
tracker's score on the same clips: the number to beat.

`sm_obj_honesty` is O1.4's own gate: the imagined rollout's return against the
simulator's over the SAME starts. Lanes run their clips under the interface;
every `H + SM_OBJ_HONEST_GAP` steps each lane's live state (robot, history,
object) is the start of an imagined rollout of H steps, and the simulator then
takes those H steps. A window counts when its lane neither fell nor reached
its clip's end inside it. Per window: the tracker's eight terms and the
object's three, summed over the horizon on both sides (the simulator's object
terms are computed here, from its state, with the rollout's own lane
function), and the imagined box against the simulated one after 1, 8, 16 and
H steps.
"""

from std.math import ceildiv, sqrt

from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.agent_gpu import _k_start_inputs
from noeira.deep_agents.supermimic.env_gpu import SMReplayGpuOf, ST_ENDING, ST_TERMINATED
from noeira.deep_agents.supermimic.eval import SM_MAX_STEP_REWARD
from noeira.deep_agents.supermimic.heads import sm_tracking_kernels_lane
from noeira.deep_agents.supermimic.kinematics import FP, sm_fk_lane
from noeira.deep_agents.supermimic.lanes import talloc, tp
from noeira.deep_agents.supermimic.motion_library import SM_REF_SCRATCH, SMMotionLibrary
from noeira.deep_agents.supermimic.object_agent_gpu import SMObjSqueeze, _k_obj_err, _k_oref_lanes, _k_oref_starts
from noeira.deep_agents.supermimic.object_env_gpu import SMObjReplayGpu, SMObjTrackingEnvGpu
from noeira.deep_agents.supermimic.object_imagination import SMObjImagBatch, sm_imagine_obj
from noeira.deep_agents.supermimic.object_policy import SMObjConfigT
from noeira.deep_agents.supermimic.object_spec import SM_OBJ, SM_OBJ_CTX
from noeira.deep_agents.supermimic.object_task import (
    SM_OBJ_HELD_SHARE, SM_OBJ_LIFT, SM_OBJ_TERMS, SM_OREF, SM_ORW, SMObjCarryTally, sm_obj_carry_steps, sm_obj_carry_tally,
    sm_obj_reward_lane, sm_trunk_legs_mask,
)
from noeira.deep_agents.supermimic.object_trainer import SMObjTrainer
from noeira.deep_agents.supermimic.robot import SMDims, TrackingBatchedSim
from noeira.nn.constants import DT
from noeira.nn.core.ptr import mptr


comptime _TPB: Int = 64
comptime SM_OBJ_HONEST_FIRST: Int = 10      # the first window's start step
comptime SM_OBJ_HONEST_GAP: Int = 8         # steps between a window's end and the next one's start


@fieldwise_init
struct SMObjClipResult(Copyable, Movable):
    var clip: Int
    var trial: Int
    var frames: Int
    var steps: Int
    var norm_return: Float64      # the tracker's eight terms, over their per-step maximum
    var fell: Bool
    var carry_steps: Int
    var held_steps: Int
    var max_err_carry: Float64    # m, over the carry steps (0 without any)
    var mean_err_carry: Float64
    var late_steps: Int           # carry steps of the carry's second half that the run reached
    var late_up_steps: Int        # … with the simulated box up
    var mean_err_late: Float64    # m, over those steps: the error once in hand
    var kernels_all: Float64      # the six kernels over their maximum, all tracked bodies
    var kernels_trunk_legs: Float64   # … the body kernels over the trunk and the legs only

    def carried(self) -> Bool:
        return (not self.fell) and self.late_steps > 0 and 2 * self.late_up_steps > self.late_steps

    def success(self) -> Bool:
        return (not self.fell) and self.carry_steps > 0 and Float64(self.held_steps) >= SM_OBJ_HELD_SHARE * Float64(self.carry_steps)

    def strict(self) -> Bool:
        return (not self.fell) and self.carry_steps > 0 and self.held_steps == self.carry_steps


def sm_obj_benchmark[OC: SMObjConfigT, BS: TrackingBatchedSim](
    ctx: DeviceContext, mut trainer: SMObjTrainer[OC], var lib: SMMotionLibrary[OC.BASE.ROBOT], track: List[Scalar[DT]],
    orefs: List[Scalar[DT]], var sim: BS, max_clips: Int, verbose: Bool, squeeze_force: Float64,
    squeeze_ks: List[Float64], squeeze_arm0: Int, trials: Int = 1, rng_seed: Int = 0,
) raises -> List[SMObjClipResult]:
    """Every clip of `lib` (the first `max_clips` if > 0) with the trainer's
    interface over the device env's lanes. `track` and `orefs` as
    `SMObjAgentGpu`'s. `squeeze_force` > 0: `SMObjSqueeze` with that force on
    every lane (a probe of the scripted squeeze, not a score of the interface).
    `trials` > 1: every clip that many times, the starts perturbed by the
    reset noise (seeded by `rng_seed`); results in the order the runs end."""
    comptime R = OC.BASE.ROBOT
    comptime N = BS.N
    var n_clips = lib.num_clips if max_clips <= 0 else min(max_clips, lib.num_clips)
    var sq = SMObjSqueeze[R](ctx)
    if squeeze_force > 0.0:
        var forces = List[Float64](length=N, fill=squeeze_force)
        sq = SMObjSqueeze[R](ctx, N, lib.clip_len, track, SM_OBJ, forces, squeeze_ks, squeeze_arm0)
    if len(orefs) != lib.num_frames * SM_OREF:
        raise Error("sm_obj_benchmark: " + String(lib.num_frames) + " library frames, " + String(len(orefs) // SM_OREF) + " object reference rows")
    var orows = ctx.enqueue_create_buffer[DT](len(orefs))
    var ht = ctx.enqueue_create_host_buffer[DT](len(orefs))
    for i in range(len(orefs)):
        ht[i] = orefs[i]
    ctx.enqueue_copy(orows, ht)
    var env = SMObjTrackingEnvGpu[R, BS](
        ctx, lib^, track, sim^, dr=False, obs_noise=False, reset_noise=trials > 1, action_noise=0.0, rng_seed=rng_seed,
        max_episode_steps=0, friction_dr=False,
    )
    # the queue: run k is clip k mod clips, trial k div clips
    var n_runs = n_clips * max(trials, 1)
    var rb = SMReplayGpuOf[R](ctx, N, 4)               # the env writes rows; nobody samples them
    var orb = SMObjReplayGpu(ctx, N, 4)
    var current = List[Int](length=N, fill=0)
    var nxt = List[Int](length=N, fill=0)
    var active = List[Bool](length=N, fill=False)
    var frames0 = List[Int](length=N, fill=0)
    for l in range(N):
        current[l] = l
        nxt[l] = l + N
        active[l] = l < n_runs
        frames0[l] = env.env.lib.clip_start[min(l, n_runs - 1) % n_clips]
    env.restart(ctx, frames0)
    for l in range(N):
        env.env.next_frame[l] = env.env.lib.clip_start[min(nxt[l], n_runs - 1) % n_clips]
    var ret = List[Float64](length=N, fill=0.0)
    var steps = List[Int](length=N, fill=0)
    var clip_carry = List[Int](length=n_clips, fill=0)      # each clip's carry frames, known before its runs
    for c in range(n_clips):
        clip_carry[c] = sm_obj_carry_steps(orefs, env.env.lib.clip_start[c], env.env.lib.clip_len[c])
    var tally = List[SMObjCarryTally]()
    for l in range(N):
        tally.append(sm_obj_carry_tally(clip_carry[current[l] % n_clips]))
    # the six kernels on the host: all tracked bodies, and the trunk and the legs
    var mask_all = List[Scalar[DT]](length=R.N_TRACKED, fill=1)
    var mask_tl = sm_trunk_legs_mask[R]()
    var fk = List[Scalar[DT]](length=SMDims[R].FK_BLOCK, fill=0)
    var rf = List[Scalar[DT]](length=SMDims[R].REF, fill=0)
    var k6 = List[Scalar[DT]](length=6, fill=0)
    var k_all = List[Float64](length=N, fill=0.0)
    var k_tl = List[Float64](length=N, fill=0.0)
    var k_steps = List[Int](length=N, fill=0)
    var results = List[SMObjClipResult]()
    var goals = talloc["gpu"](N * SMDims[R].GOAL, Optional(ctx))
    var oref_l = talloc["gpu"](N * SM_OREF, Optional(ctx))
    var oerr = talloc["gpu"](N * 2, Optional(ctx))
    var remaining = n_runs
    var done_n = 0
    while remaining > 0:
        env.env.policy_inputs(ctx, goals)
        ctx.enqueue_function[_k_oref_lanes](
            Int64(N), mptr(env.env.frame.unsafe_ptr()), mptr(orows.unsafe_ptr()), tp["gpu"](oref_l), grid_dim=ceildiv(N, _TPB), block_dim=_TPB,
        )
        var a = trainer.act[N](env.env.hist, goals, env.env.s, env.prev_o, oref_l, resident=True)
        sq.apply(
            ctx, trainer.base.tables.tree_p["gpu"](), trainer.base.tables.head_p["gpu"](), mptr(env.env.frame.unsafe_ptr()),
            tp["gpu"](env.env.s), tp["gpu"](env.prev_o), tp["gpu"](a),
        )
        _ = env.step(ctx, a, rb, orb)
        # `env.o`: every lane's object after the step, before any reset; `oref_l`: the row the step aimed at
        ctx.enqueue_function[_k_obj_err](Int64(N), tp["gpu"](env.o), tp["gpu"](oref_l), tp["gpu"](oerr), grid_dim=ceildiv(N, _TPB), block_dim=_TPB)
        oerr.download(ctx)
        oref_l.download(ctx)
        env.o.download(ctx)
        env.env.s.download(ctx)
        for l in range(N):
            if not active[l]:
                continue
            var c = current[l] % n_clips
            ret[l] += env.env.last_reward[l]
            steps[l] += 1
            var z0 = Float64(orefs[env.env.lib.clip_start[c] * SM_OREF + 2])
            tally[l].note(Float64(oref_l.data[l * SM_OREF + 2]) - z0, Float64(env.o.data[l * SM_OBJ + 2]) - z0, Float64(oerr.data[2 * l]))
            var st = env.env.h_out_i[2 * l + 1]
            if (st & ST_ENDING) == 0:
                # `env.env.s` is this lane's state after the step (an ending lane's is already its next start)
                env.env.lib.reference_row(Int(env.env.h_out_i[2 * l]), trainer.base.tables, _hp(rf))
                var sp = _hp(env.env.s.data, l * SMDims[R].S)
                var pos = _hp(fk)
                sm_fk_lane[DT, SMDims[R].NB](
                    trainer.base.tables.tree_ptr(), sp, sp.unsafe_offset(SMDims[R].NQ), pos, pos.unsafe_offset(SMDims[R].NB * 3),
                    pos.unsafe_offset(SMDims[R].NB * 7), pos.unsafe_offset(SMDims[R].NB * 10),
                )
                k_all[l] += Float64(sm_tracking_kernels_lane[DT, R](trainer.base.tables.head_ptr(), pos, _hp(rf), _hp(mask_all), _hp(k6)))
                k_tl[l] += Float64(sm_tracking_kernels_lane[DT, R](trainer.base.tables.head_ptr(), pos, _hp(rf), _hp(mask_tl), _hp(k6)))
                k_steps[l] += 1
                continue
            var kn = SM_MAX_STEP_REWARD * Float64(max(k_steps[l], 1))
            results.append(SMObjClipResult(
                c, current[l] // n_clips, env.env.lib.clip_len[c], steps[l], ret[l] / (SM_MAX_STEP_REWARD * Float64(max(steps[l], 1))),
                (st & ST_TERMINATED) != 0, tally[l].carry, tally[l].held, tally[l].emax, tally[l].mean_err(),
                tally[l].late, tally[l].late_up, tally[l].mean_err_late(), k_all[l] / kn, k_tl[l] / kn,
            ))
            remaining -= 1
            done_n += 1
            if verbose and done_n % 20 == 0:
                print("  ", done_n, "/", n_runs, "runs")
            # the lane was just reset to its queued clip: advance the queue
            current[l] = nxt[l]
            nxt[l] += N
            active[l] = current[l] < n_runs
            env.env.next_frame[l] = env.env.lib.clip_start[min(nxt[l], n_runs - 1) % n_clips]
            ret[l] = 0.0
            steps[l] = 0
            tally[l] = sm_obj_carry_tally(clip_carry[min(current[l], n_runs - 1) % n_clips])
            k_all[l] = 0.0
            k_tl[l] = 0.0
            k_steps[l] = 0
    return results^


@fieldwise_init
struct SMObjHonesty(Copyable, Movable):
    """One bucket of windows (`sm_obj_honesty`): sums over the horizon, mean
    per window. `gap_*`: mean |imagined − simulated| per window."""
    var windows: Int
    var dropped: Int              # windows whose lane ended inside them
    var imag_track: Float64       # the tracker's eight terms
    var sim_track: Float64
    var gap_track: Float64
    var imag_pos: Float64         # the object's three terms
    var imag_ori: Float64
    var imag_hands: Float64
    var sim_pos: Float64
    var sim_ori: Float64
    var sim_hands: Float64
    var gap_total: Float64        # of the eleven terms' sum
    var err_1: Float64            # m, the imagined box against the simulated one after 1, 8, 16, H steps
    var err_8: Float64
    var err_16: Float64
    var err_h: Float64

    def imag_total(self) -> Float64:
        return self.imag_track + self.imag_pos + self.imag_ori + self.imag_hands

    def sim_total(self) -> Float64:
        return self.sim_track + self.sim_pos + self.sim_ori + self.sim_hands


def _hp(mut x: List[Scalar[DT]], at: Int = 0) -> FP[DT]:
    return x.unsafe_ptr().unsafe_offset(at).as_unsafe_any_origin()


def sm_obj_honesty[OC: SMObjConfigT, BS: TrackingBatchedSim](
    ctx: DeviceContext, mut trainer: SMObjTrainer[OC], var lib: SMMotionLibrary[OC.BASE.ROBOT], track: List[Scalar[DT]],
    orefs: List[Scalar[DT]], rw: List[Scalar[DT]], var sim: BS, steps: Int = 380,
) raises -> List[SMObjHonesty]:
    """Imagined against simulated return over the same starts: two buckets,
    [0] windows whose reference is NOT carrying at their start, [1] windows
    whose reference is (`SM_OBJ_LIFT`). No noise, no randomisation; lane l runs
    clip l mod clips and starts it again when it ends. `rw`: the object
    reward's data, the training run's (`object_task.sm_obj_reward_params`)."""
    comptime R = OC.BASE.ROBOT
    comptime D = SMDims[R]
    comptime N = BS.N
    comptime H = OC.BASE.HORIZON
    comptime n = R.NDOF
    if len(orefs) != lib.num_frames * SM_OREF:
        raise Error("sm_obj_honesty: " + String(lib.num_frames) + " library frames, " + String(len(orefs) // SM_OREF) + " object reference rows")
    if len(rw) != SM_ORW:
        raise Error("sm_obj_honesty: " + String(SM_ORW) + " reward numbers, got " + String(len(rw)))
    var n_clips = lib.num_clips
    var orows = ctx.enqueue_create_buffer[DT](len(orefs))
    var ht = ctx.enqueue_create_host_buffer[DT](len(orefs))
    for i in range(len(orefs)):
        ht[i] = orefs[i]
    ctx.enqueue_copy(orows, ht)
    var env = SMObjTrackingEnvGpu[R, BS](
        ctx, lib^, track, sim^, dr=False, obs_noise=False, reset_noise=False, action_noise=0.0, max_episode_steps=0, friction_dr=False,
    )
    var rb = SMReplayGpuOf[R](ctx, N, 4)
    var orb = SMObjReplayGpu(ctx, N, 4)
    var z0 = List[Float64](length=N, fill=0.0)          # each lane's clip: its first reference height
    var frames0 = List[Int](length=N, fill=0)
    for l in range(N):
        frames0[l] = env.env.lib.clip_start[l % n_clips]
        z0[l] = Float64(orefs[frames0[l] * SM_OREF + 2])
    env.restart(ctx, frames0)
    for l in range(N):
        env.env.next_frame[l] = frames0[l]

    var goals = talloc["gpu"](N * D.GOAL, Optional(ctx))
    var oref_l = talloc["gpu"](N * SM_OREF, Optional(ctx))
    var ib = SMObjImagBatch[R](N, H)
    for k in range(SM_ORW):
        ib.rw.data[k] = rw[k]
    ib.to_device(ctx)
    # `_k_start_inputs` builds a start's history from logged states; here the history is the env's own, copied
    # below, so the kernel gets zeros and what it writes to `hist0` is overwritten
    var zs = ctx.enqueue_create_buffer[DT](D.HIST * N * D.S)
    var za = ctx.enqueue_create_buffer[DT](D.HIST * N * n)
    var scratch = ctx.enqueue_create_buffer[DT](N * SM_REF_SCRATCH[R])
    zs.enqueue_fill(Scalar[DT](0))
    za.enqueue_fill(Scalar[DT](0))

    var rw_h = rw.copy()
    var fk = List[Scalar[DT]](length=D.FK_BLOCK, fill=0)
    var terms = List[Scalar[DT]](length=SM_OBJ_TERMS + 1, fill=0)
    var imag_track = List[Float64](length=N, fill=0.0)
    var imag_obj = List[Float64](length=N * SM_OBJ_TERMS, fill=0.0)
    var sim_track = List[Float64](length=N, fill=0.0)
    var sim_obj = List[Float64](length=N * SM_OBJ_TERMS, fill=0.0)
    var errs = List[Float64](length=N * 4, fill=0.0)
    var imag_objs = List[Scalar[DT]](length=(H + 1) * N * SM_OBJ, fill=0)
    var live = List[Bool](length=N, fill=False)
    var carrying = List[Bool](length=N, fill=False)
    var acc = List[Float64](length=2 * 14, fill=0.0)    # per bucket: the 14 sums of `SMObjHonesty` after its two counts
    var cnt = List[Int](length=2 * 2, fill=0)
    var w0 = -1                                          # the open window's start step

    for t in range(steps):
        env.env.policy_inputs(ctx, goals)
        ctx.enqueue_function[_k_oref_lanes](
            Int64(N), mptr(env.env.frame.unsafe_ptr()), mptr(orows.unsafe_ptr()), tp["gpu"](oref_l), grid_dim=ceildiv(N, _TPB), block_dim=_TPB,
        )
        if w0 < 0 and t >= SM_OBJ_HONEST_FIRST and (t - SM_OBJ_HONEST_FIRST) % (H + SM_OBJ_HONEST_GAP) == 0 and t + H <= steps:
            # ── a window opens: every lane's live state is an imagined start ──
            w0 = t
            ctx.enqueue_function[_k_start_inputs[R]](
                Int64(N), Int64(H), UInt32(0), Int64(0),
                mptr(env.env.lib.rows_dev.value().unsafe_ptr()), trainer.base.tables.tree_p["gpu"](), trainer.base.tables.head_p["gpu"](),
                mptr(env.env.lib.last_dev.value().unsafe_ptr()), mptr(env.env.frame.unsafe_ptr()),
                mptr(zs.unsafe_ptr()), mptr(za.unsafe_ptr()), tp["gpu"](env.env.xi), mptr(scratch.unsafe_ptr()),
                tp["gpu"](ib.base.hist0), tp["gpu"](ib.base.goals), tp["gpu"](ib.base.refs),
                grid_dim=ceildiv(N, _TPB), block_dim=_TPB,
            )
            ctx.enqueue_function[_k_oref_starts](
                Int64(N), Int64(H), mptr(env.env.lib.last_dev.value().unsafe_ptr()), mptr(env.env.frame.unsafe_ptr()),
                mptr(orows.unsafe_ptr()), tp["gpu"](ib.orefs), grid_dim=ceildiv(N, _TPB), block_dim=_TPB,
            )
            env.env.s.download(ctx)
            env.env.xi.download(ctx)
            env.env.a_last.download(ctx)
            env.env.hist.download(ctx)
            env.prev_o.download(ctx)
            env.octx.download(ctx)
            oref_l.download(ctx)
            for l in range(N):
                for k in range(D.S):
                    ib.base.state.data[l * D.S + k] = env.env.s.data[l * D.S + k]
                for k in range(D.X):
                    ib.base.xi.data[l * D.X + k] = env.env.xi.data[l * D.X + k]
                for k in range(n):
                    ib.base.prev_action.data[l * n + k] = env.env.a_last.data[l * n + k]
                for hh in range(D.HIST):
                    for k in range(D.OBS):
                        ib.base.hist0.data[(hh * N + l) * D.OBS + k] = env.env.hist.data[(l * D.HIST + hh) * D.OBS + k]
                for k in range(SM_OBJ):
                    ib.obj0.data[l * SM_OBJ + k] = env.prev_o.data[l * SM_OBJ + k]
                for k in range(SM_OBJ_CTX):
                    ib.octx.data[l * SM_OBJ_CTX + k] = env.octx.data[l * SM_OBJ_CTX + k]
                live[l] = True
                carrying[l] = Float64(oref_l.data[l * SM_OREF + 2]) >= z0[l] + SM_OBJ_LIFT
                imag_track[l] = 0.0
                sim_track[l] = 0.0
                for k in range(SM_OBJ_TERMS):
                    imag_obj[l * SM_OBJ_TERMS + k] = 0.0
                    sim_obj[l * SM_OBJ_TERMS + k] = 0.0
            ib.base.state.upload_resident(ctx)
            ib.base.xi.upload_resident(ctx)
            ib.base.prev_action.upload_resident(ctx)
            ib.base.hist0.upload_resident(ctx)
            ib.obj0.upload_resident(ctx)
            ib.octx.upload_resident(ctx)
            var o = sm_imagine_obj[OC, N, "gpu"](
                trainer.base.wm, trainer.base.pol, trainer.om, trainer.enc, trainer.critic, trainer.base.tables, ib,
                OC.BASE.DEF_GAMMA, trainer.gain, False, 0, Optional(ctx),
            )
            o.to_host(ctx)
            for tt in range(H):
                for l in range(N):
                    var ro = 0.0
                    for k in range(SM_OBJ_TERMS):
                        var x = Float64(o.oterms.data[(tt * N + l) * SM_OBJ_TERMS + k])
                        imag_obj[l * SM_OBJ_TERMS + k] += x
                        ro += x
                    imag_track[l] += Float64(o.base.rewards.data[tt * N + l]) - ro
            for i in range((H + 1) * N * SM_OBJ):
                imag_objs[i] = o.objs.data[i]

        var a = trainer.act[N](env.env.hist, goals, env.env.s, env.prev_o, oref_l, resident=True)
        _ = env.step(ctx, a, rb, orb)
        if w0 < 0:
            continue
        # ── inside a window: step kw = t − w0 + 1 of its H ──
        var kw = t - w0 + 1
        env.env.s.download(ctx)
        env.o.download(ctx)
        oref_l.download(ctx)
        for l in range(N):
            if not live[l]:
                continue
            if (env.env.h_out_i[2 * l + 1] & ST_ENDING) != 0:
                live[l] = False
                continue
            sim_track[l] += env.env.last_reward[l]
            var sp = _hp(env.env.s.data, l * D.S)
            var pos = _hp(fk)
            sm_fk_lane[DT, D.NB](trainer.base.tables.tree_ptr(), sp, sp.unsafe_offset(D.NQ), pos, pos.unsafe_offset(D.NB * 3), pos.unsafe_offset(D.NB * 7), pos.unsafe_offset(D.NB * 10))
            _ = sm_obj_reward_lane[DT, R](
                trainer.base.tables.head_ptr(), pos, _hp(env.o.data, l * SM_OBJ), _hp(oref_l.data, l * SM_OREF), _hp(rw_h), _hp(terms),
            )
            for j in range(SM_OBJ_TERMS):
                sim_obj[l * SM_OBJ_TERMS + j] += Float64(terms[j])
            var slot = 0 if kw == 1 else (1 if kw == 8 else (2 if kw == 16 else (3 if kw == H else -1)))
            if slot >= 0:
                var d2 = 0.0
                for j in range(3):
                    var d = Float64(env.o.data[l * SM_OBJ + j]) - Float64(imag_objs[(kw * N + l) * SM_OBJ + j])
                    d2 += d * d
                errs[l * 4 + slot] = sqrt(d2)
        if kw < H:
            continue
        # ── the window closes ──
        for l in range(N):
            var b = 1 if carrying[l] else 0
            if not live[l]:
                cnt[b * 2 + 1] += 1
                continue
            cnt[b * 2] += 1
            var it = imag_track[l]
            var stt = sim_track[l]
            var io = 0.0
            var so = 0.0
            for j in range(SM_OBJ_TERMS):
                io += imag_obj[l * SM_OBJ_TERMS + j]
                so += sim_obj[l * SM_OBJ_TERMS + j]
                acc[b * 14 + 3 + j] += imag_obj[l * SM_OBJ_TERMS + j]
                acc[b * 14 + 6 + j] += sim_obj[l * SM_OBJ_TERMS + j]
            acc[b * 14] += it
            acc[b * 14 + 1] += stt
            acc[b * 14 + 2] += abs(it - stt)
            acc[b * 14 + 9] += abs(it + io - stt - so)
            for j in range(4):
                acc[b * 14 + 10 + j] += errs[l * 4 + j]
        w0 = -1

    var out = List[SMObjHonesty]()
    for b in range(2):
        var m = 1.0 / Float64(max(cnt[b * 2], 1))
        var q = b * 14
        out.append(SMObjHonesty(
            cnt[b * 2], cnt[b * 2 + 1], acc[q] * m, acc[q + 1] * m, acc[q + 2] * m, acc[q + 3] * m, acc[q + 4] * m, acc[q + 5] * m,
            acc[q + 6] * m, acc[q + 7] * m, acc[q + 8] * m, acc[q + 9] * m, acc[q + 10] * m, acc[q + 11] * m, acc[q + 12] * m, acc[q + 13] * m,
        ))
    return out^

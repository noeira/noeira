"""SuperMimic M2: the motion library and the return-based reset sampler.

    uv run --no-project --python 3.12 --with torch --with mujoco==3.12.0 --with numpy \\
        python tools/supermimic/lib_oracle.py --out /tmp/sm_lib
    pixi run mojo run -I . tests/deep_agents/supermimic/test_supermimic_lib.mojo /tmp/sm_lib

The oracle is the reproduction's compact `MotionLibrary` and its
`MultiClipMotionCommand._sample_frames` / `_finalize_episodes`, verbatim, on a
synthetic packed library of four clips (73 / 120 / 50 / 31 frames).

WHAT EACH SECTION CAN CATCH
===========================
[1] the bins (start, end, clip; short last bins), `last_frame_of` and
    `bin_of_frame` on every frame: exact.
[2] the 680-d goal window on every frame, clamped at each clip's end
    (the last 9 frames of every clip exercise the clamp): 1e-6 of scale.
[3] the heads' reference row (189) on every frame: the tracked bodies by OUR
    FK of the reference state against the reproduction's FK on our model.
[4] the sampler's bin distribution and its memory update (1e-6), and the
    frames it picks for injected (bin, u) draws: exact.
[5] `_finalize_episodes` on six episodes — terminated, clip-ended and
    truncated (the last 100 steps dropped) — into the bin returns: 1e-5.
[6] the replay ring after the reference's own sequence of 335 adds (a ring
    that wraps, terminal rows): the k = 8 window mask and the start mask on
    every row, and the windows / starts at the reference's picks (states,
    executed actions, ξ, labels, frames, backfilled history and its
    actions): exact.

Planted mutants: a disabled truncation rule fails [5], an unclamped goal
window fails [2]. Dropping the `ep_start` test from the history's previous-
action rule PASSES: the recorder bumps the episode id after every terminal
row, so the row before an episode's first row always carries another id and
the id test alone excludes it. The term is redundant under that protocol,
not untested by accident.
"""

from std.sys import argv

from noeira.deep_agents.supermimic.g1_fk import g1_fk_tree
from noeira.deep_agents.supermimic.heads import SM_REF_DIM, g1_heads_table
from noeira.deep_agents.supermimic.motion_library import (
    SMEpisodeLog, SMMotionLibrary, SMResetSampler,
)
from noeira.deep_agents.supermimic.replay import SMReplayBuffer
from noeira.envs.robots import UnitreeG1
from noeira.io.npy import load_npy_2d_f64
from noeira.nn.constants import DT


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def main() raises:
    var av = argv()
    var dump = String(av[1]) if len(av) > 1 else String("/tmp/sm_lib")
    var t = Tally()
    var env = UnitreeG1[DType.float64]()
    _ = env.reset()

    var refv = load_npy_2d_f64(dump + "/l_ref.npy", 71).values.copy()
    var lensf = load_npy_2d_f64(dump + "/l_lens.npy", 1).values.copy()
    var lens = List[Int]()
    for i in range(len(lensf)):
        lens.append(Int(lensf[i]))
    var q0 = load_npy_2d_f64(dump + "/l_q0.npy", 29).values.copy()
    var rows = List[Scalar[DT]](length=len(refv), fill=0)
    for i in range(len(refv)):
        rows[i] = Scalar[DT](refv[i])
    var lib = SMMotionLibrary(rows^, lens, q0)
    var F = lib.num_frames

    print("\n[1] bins and frame queries,", F, "frames")
    var bins = load_npy_2d_f64(dump + "/l_bins.npy", 3)
    var bad = 0 if bins.rows == lib.num_bins else 1
    for b in range(min(bins.rows, lib.num_bins)):
        if lib.bin_start[b] != Int(bins.values[b * 3]) or lib.bin_end[b] != Int(bins.values[b * 3 + 1]) or lib.bin_clip[b] != Int(bins.values[b * 3 + 2]):
            bad += 1
    t.truth(bad == 0, String(lib.num_bins) + " bins: start, end, clip")
    var last = load_npy_2d_f64(dump + "/l_last.npy", 1).values.copy()
    var bof = load_npy_2d_f64(dump + "/l_bin_of.npy", 1).values.copy()
    var bad2 = 0
    for f in range(F):
        if lib.last_frame_of(f) != Int(last[f]) or lib.bin_of_frame(f) != Int(bof[f]):
            bad2 += 1
    t.truth(bad2 == 0, "last_frame_of and bin_of_frame on every frame")

    print("\n[2] the goal window")
    var wg = load_npy_2d_f64(dump + "/l_goal.npy", 680).values.copy()
    var g = List[Scalar[DT]](length=680, fill=0)
    var eg = 0.0
    var sg = 0.0
    for f in range(F):
        lib.goal_window(f, g.unsafe_ptr().as_unsafe_any_origin())
        for k in range(680):
            eg = max(eg, abs(Float64(g[k]) - wg[f * 680 + k]))
            sg = max(sg, abs(wg[f * 680 + k]))
    t.truth(eg <= 1e-6 * sg, "every frame: max err " + String(eg) + " (scale " + String(sg) + ")")

    print("\n[3] the reference rows (our FK)")
    var tree = g1_fk_tree[DT](env)
    var table = g1_heads_table[DT](env, q0)
    var wr = load_npy_2d_f64(dump + "/l_refrow.npy", SM_REF_DIM).values.copy()
    var rr = List[Scalar[DT]](length=SM_REF_DIM, fill=0)
    var er = 0.0
    var srr = 0.0
    for f in range(F):
        lib.reference_row(f, tree, table, rr.unsafe_ptr().as_unsafe_any_origin())
        for k in range(SM_REF_DIM):
            er = max(er, abs(Float64(rr[k]) - wr[f * SM_REF_DIM + k]))
            srr = max(srr, abs(wr[f * SM_REF_DIM + k]))
    t.truth(er <= 1e-5 * srr, "every frame: max err " + String(er) + " (scale " + String(srr) + ")")

    print("\n[4] the sampler")
    var sm = SMResetSampler(lib.num_bins)
    var r0 = load_npy_2d_f64(dump + "/s_return0.npy", 1).values.copy()
    var m0 = load_npy_2d_f64(dump + "/s_memory0.npy", 1).values.copy()
    for b in range(lib.num_bins):
        sm.bin_return[b] = r0[b]
        sm.bin_memory[b] = m0[b]
    var p = sm.probabilities()
    var wp = load_npy_2d_f64(dump + "/s_p.npy", 1).values.copy()
    var m1 = load_npy_2d_f64(dump + "/s_memory1.npy", 1).values.copy()
    var ep = 0.0
    var em = 0.0
    for b in range(lib.num_bins):
        ep = max(ep, abs(p[b] - wp[b]))
        em = max(em, abs(sm.bin_memory[b] - m1[b]))
    t.truth(ep <= 1e-6 and em <= 1e-6, "bin distribution (max err " + String(ep) + ") and memory update (" + String(em) + ")")
    var sb = load_npy_2d_f64(dump + "/s_b.npy", 1).values.copy()
    var su = load_npy_2d_f64(dump + "/s_u.npy", 1).values.copy()
    var sf = load_npy_2d_f64(dump + "/s_frames.npy", 1).values.copy()
    var bad3 = 0
    for i in range(len(sb)):
        if SMResetSampler.frame_from(lib, Int(sb[i]), su[i]) != Int(sf[i]):
            bad3 += 1
    t.truth(bad3 == 0, String(len(sb)) + " injected draws pick the reference's frames")

    print("\n[5] finalize")
    var fl = load_npy_2d_f64(dump + "/f_lens.npy", 1).values.copy()
    var ff = load_npy_2d_f64(dump + "/f_flags.npy", 2).values.copy()
    var tmax = 0
    for e in range(len(fl)):
        tmax = max(tmax, Int(fl[e]))
    var frw = load_npy_2d_f64(dump + "/f_rewards.npy", tmax).values.copy()
    var fbn = load_npy_2d_f64(dump + "/f_bins.npy", tmax).values.copy()
    var eps = List[SMEpisodeLog]()
    for e in range(len(fl)):
        var lg = SMEpisodeLog()
        for k in range(Int(fl[e])):
            lg.rewards.append(frw[e * tmax + k])
            lg.bins.append(Int(fbn[e * tmax + k]))
        lg.terminated = ff[e * 2] > 0.5
        lg.clip_end = ff[e * 2 + 1] > 0.5
        eps.append(lg^)
    var fr0 = load_npy_2d_f64(dump + "/f_return0.npy", 1).values.copy()
    var fr1 = load_npy_2d_f64(dump + "/f_return1.npy", 1).values.copy()
    var sm2 = SMResetSampler(lib.num_bins)
    for b in range(lib.num_bins):
        sm2.bin_return[b] = fr0[b]
    sm2.finalize(eps)
    var efin = 0.0
    var moved = 0
    for b in range(lib.num_bins):
        efin = max(efin, abs(sm2.bin_return[b] - fr1[b]) / max(1.0, abs(fr1[b])))
        if abs(fr1[b] - fr0[b]) > 1e-6:
            moved += 1
    t.truth(efin <= 1e-5 and moved > 0, String(moved) + " bins moved; max rel err " + String(efin))


    print("\n[6] the replay ring")
    var calls = load_npy_2d_f64(dump + "/r_calls.npy", 7 + 71 + 29 + 29 + 33)
    var rb = SMReplayBuffer(3, 40)
    var W = 7 + 71 + 29 + 29 + 33
    for c in range(calls.rows):
        var o = c * W
        var sv = List[Scalar[DT]](length=71, fill=0)
        var av2 = List[Scalar[DT]](length=29, fill=0)
        var ae = List[Scalar[DT]](length=29, fill=0)
        var xv = List[Scalar[DT]](length=33, fill=0)
        for k in range(71):
            sv[k] = Scalar[DT](calls.values[o + 7 + k])
        for k in range(29):
            av2[k] = Scalar[DT](calls.values[o + 78 + k])
            ae[k] = Scalar[DT](calls.values[o + 107 + k])
        for k in range(33):
            xv[k] = Scalar[DT](calls.values[o + 136 + k])
        var e = Int(calls.values[o])
        rb.add(e, sv, av2, ae, xv, calls.values[o + 1], calls.values[o + 2], Int(calls.values[o + 3]),
               calls.values[o + 4] > 0.5, calls.values[o + 5] > 0.5)
        if calls.values[o + 6] > 0.5:
            rb.end_episode(e)
    var wm_ref = load_npy_2d_f64(dump + "/r_wmask.npy", 1).values.copy()
    var sm_ref = load_npy_2d_f64(dump + "/r_smask.npy", 1).values.copy()
    var wmask = rb.window_mask(8)
    var smask = rb.start_mask(10)
    var bad_w = 0
    var bad_s = 0
    var nw = 0
    for i in range(len(wmask)):
        if wmask[i] != (wm_ref[i] > 0.5):
            bad_w += 1
        if smask[i] != (sm_ref[i] > 0.5):
            bad_s += 1
        if wmask[i]:
            nw += 1
    t.truth(bad_w == 0 and bad_s == 0 and nw > 0, "window mask (" + String(nw) + " rows) and start mask on all 120 rows")
    var wpk = load_npy_2d_f64(dump + "/r_wpick.npy", 1).values.copy()
    var idx = List[Int]()
    for i in range(len(wpk)):
        idx.append(Int(wpk[i]))
    var w = rb.windows_at(idx, 8)
    var ew = 0.0
    var r_st = load_npy_2d_f64(dump + "/r_w_states.npy", 71).values.copy()
    var r_ae = load_npy_2d_f64(dump + "/r_w_a_exec.npy", 29).values.copy()
    var r_xi = load_npy_2d_f64(dump + "/r_w_xi.npy", 33).values.copy()
    var r_dn = load_npy_2d_f64(dump + "/r_w_done.npy", 1).values.copy()
    var r_fr = load_npy_2d_f64(dump + "/r_w_frame.npy", 1).values.copy()
    for i in range(len(r_st)):
        ew = max(ew, abs(Float64(w.states[i]) - r_st[i]))
    for i in range(len(r_ae)):
        ew = max(ew, abs(Float64(w.a_exec[i]) - r_ae[i]))
    for i in range(len(r_xi)):
        ew = max(ew, abs(Float64(w.xi[i]) - r_xi[i]))
    for i in range(len(r_dn)):
        ew = max(ew, abs(Float64(w.done[i]) - r_dn[i]) + abs(Float64(w.frame[i]) - r_fr[i]))
    t.truth(ew == 0.0, String(len(idx)) + " windows: states, executed actions, xi, labels, frames (max err " + String(ew) + ")")
    var sp = load_npy_2d_f64(dump + "/r_spick.npy", 1).values.copy()
    var sidx = List[Int]()
    for i in range(len(sp)):
        sidx.append(Int(sp[i]))
    var stt = rb.starts_at(sidx, 10)
    var es = 0.0
    var r_ss = load_npy_2d_f64(dump + "/r_s_state.npy", 71).values.copy()
    var r_pa = load_npy_2d_f64(dump + "/r_s_prev_action.npy", 29).values.copy()
    var r_hs = load_npy_2d_f64(dump + "/r_s_hist_states.npy", 71).values.copy()
    var r_ha = load_npy_2d_f64(dump + "/r_s_hist_actions.npy", 29).values.copy()
    for i in range(len(r_ss)):
        es = max(es, abs(Float64(stt.state[i]) - r_ss[i]))
    for i in range(len(r_pa)):
        es = max(es, abs(Float64(stt.prev_action[i]) - r_pa[i]))
    for i in range(len(r_hs)):
        es = max(es, abs(Float64(stt.hist_states[i]) - r_hs[i]))
    for i in range(len(r_ha)):
        es = max(es, abs(Float64(stt.hist_actions[i]) - r_ha[i]))
    t.truth(es == 0.0, String(len(sidx)) + " starts: state, previous action, backfilled history and its actions (max err " + String(es) + ")")

    print("\n===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_supermimic_lib: " + String(t.fails) + " failed")

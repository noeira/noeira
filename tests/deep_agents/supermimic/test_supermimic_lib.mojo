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
[6] the sampler's draw statistics (`note_draw` / `take_stats`) on 20 bins
    of known returns, against hand-computed values: the drawn bins' mean,
    the all-bins mean, the hard share against the PREVIOUS window's bottom
    decile, and the counters cleared after a take.
[7] realmimic's count estimator (`bin_estimator="count"`) against its own
    `_finalize_episodes` on six episodes (terminated, clip-ended, timed out,
    one shorter than the 50-step continuation window), bins at 0, 3 and 25
    visits: the returns (1e-5) and the visit counts (exact); a fresh count
    sampler starts every bin at 0, unvisited.
[8] seeding the count estimator from a benchmark: a failed clip's bins at 0,
    another's at G_MAX x its return, each one visit; the hard term then puts
    most of its mass on the failed clip.

Planted mutants: a disabled truncation rule fails [5], an unclamped goal
window fails [2]; a dropped continuation and a fixed EMA α each fail [7]. (The replay ring's gate moved to
`tests/deep_agents/data/test_episodic_replay.mojo`.)
"""

from std.sys import argv

from noeira.deep_agents.supermimic.robot import G1Robot, G1Sim, SMDims
from noeira.deep_agents.supermimic.tables import sm_tables
from noeira.deep_agents.supermimic.motion_library import (
    SM_G_MAX, SMEpisodeLog, SMMotionLibrary, SMResetSampler,
)
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
    var sim = G1Sim.create()
    comptime SM_REF_DIM = SMDims[G1Robot].REF

    var refv = load_npy_2d_f64(dump + "/l_ref.npy", 71).values.copy()
    var lensf = load_npy_2d_f64(dump + "/l_lens.npy", 1).values.copy()
    var lens = List[Int]()
    for i in range(len(lensf)):
        lens.append(Int(lensf[i]))
    var q0 = load_npy_2d_f64(dump + "/l_q0.npy", 29).values.copy()
    var rows = List[Scalar[DT]](length=len(refv), fill=0)
    for i in range(len(refv)):
        rows[i] = Scalar[DT](refv[i])
    var lib = SMMotionLibrary[G1Robot](rows^, lens, q0)
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
    var tb = sm_tables[DT, G1Robot](sim, q0)
    var wr = load_npy_2d_f64(dump + "/l_refrow.npy", SM_REF_DIM).values.copy()
    var rr = List[Scalar[DT]](length=SM_REF_DIM, fill=0)
    var er = 0.0
    var srr = 0.0
    for f in range(F):
        lib.reference_row(f, tb, rr.unsafe_ptr().as_unsafe_any_origin())
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
        if SMResetSampler.frame_from[G1Robot](lib, Int(sb[i]), su[i]) != Int(sf[i]):
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

    print("\n[6] draw statistics")
    var sm3 = SMResetSampler(20)
    for b in range(20):
        sm3.bin_return[b] = 0.05 * SM_G_MAX * Float64(b)         # 0, 0.05, ..., 0.95 of G_MAX
    var s0 = sm3.take_stats()                                       # sets the decile: bins 0, 1 are below it
    t.truth(s0.resets == 0 and s0.reset_hard_share == 0.0, "no draws: zero resets, zero hard share")
    var draws: List[Int] = [0, 1, 10, 19]
    for k in range(len(draws)):
        sm3.note_draw(draws[k])
    var s1 = sm3.take_stats()
    t.truth(s1.resets == 4, "4 draws counted (" + String(s1.resets) + ")")
    t.truth(abs(s1.reset_bin_return - 0.375) < 1e-12, "drawn bins' mean 0.375 (" + String(s1.reset_bin_return) + ")")
    t.truth(abs(s1.bin_return_mean - 0.475) < 1e-12, "all bins' mean 0.475 (" + String(s1.bin_return_mean) + ")")
    t.truth(abs(s1.reset_hard_share - 0.5) < 1e-12, "hard share 2 / 4 (" + String(s1.reset_hard_share) + ")")
    var s2 = sm3.take_stats()
    t.truth(s2.resets == 0 and s2.reset_hard_share == 0.0, "counters cleared after a take")

    print("\n[7] the count estimator (realmimic)")
    var fresh = SMResetSampler(lib.num_bins, True)
    var zero = True
    for b in range(lib.num_bins):
        zero = zero and fresh.bin_return[b] == 0.0 and fresh.bin_count[b] == 0.0
    t.truth(zero and fresh.untouched_share() == 1.0, "a fresh count sampler starts every bin at 0, unvisited")
    var cl = load_npy_2d_f64(dump + "/c_lens.npy", 1).values.copy()
    var cf = load_npy_2d_f64(dump + "/c_flags.npy", 2).values.copy()
    var cmax = 0
    for e in range(len(cl)):
        cmax = max(cmax, Int(cl[e]))
    var crw = load_npy_2d_f64(dump + "/c_rewards.npy", cmax).values.copy()
    var cbn = load_npy_2d_f64(dump + "/c_bins.npy", cmax).values.copy()
    var ceps = List[SMEpisodeLog]()
    for e in range(len(cl)):
        var lg = SMEpisodeLog()
        for k in range(Int(cl[e])):
            lg.rewards.append(crw[e * cmax + k])
            lg.bins.append(Int(cbn[e * cmax + k]))
        lg.terminated = cf[e * 2] > 0.5
        lg.clip_end = cf[e * 2 + 1] > 0.5
        ceps.append(lg^)
    var cr0 = load_npy_2d_f64(dump + "/c_return0.npy", 1).values.copy()
    var cc0 = load_npy_2d_f64(dump + "/c_count0.npy", 1).values.copy()
    var cr1 = load_npy_2d_f64(dump + "/c_return1.npy", 1).values.copy()
    var cc1 = load_npy_2d_f64(dump + "/c_count1.npy", 1).values.copy()
    var smc = SMResetSampler(lib.num_bins, True)
    var regimes = List[Bool](length=3, fill=False)
    for b in range(lib.num_bins):
        smc.bin_return[b] = cr0[b]
        smc.bin_count[b] = cc0[b]
        regimes[0] = regimes[0] or cc0[b] == 0.0
        regimes[1] = regimes[1] or (cc0[b] > 0.0 and cc0[b] < 20.0)
        regimes[2] = regimes[2] or cc0[b] >= 20.0
    t.truth(regimes[0] and regimes[1] and regimes[2], "the fixture holds unvisited, young (< 1/α) and old (≥ 1/α) bins")
    smc.finalize(ceps)
    var ecnt = 0.0
    var cnt_ok = True
    for b in range(lib.num_bins):
        ecnt = max(ecnt, abs(smc.bin_return[b] - cr1[b]) / max(1.0, abs(cr1[b])))
        cnt_ok = cnt_ok and smc.bin_count[b] == cc1[b]
    t.truth(ecnt <= 1e-5, "the bin returns fold as realmimic's: max rel err " + String(ecnt))
    t.truth(cnt_ok, "one visit counted per touched bin")
    t.truth(smc.untouched_share() == 0.0, "every bin of the fixture was visited")

    print("\n[8] seeded from a benchmark")
    var sd = SMResetSampler(lib.num_bins, True)
    var rets: List[Float64] = [0.9, 0.5, 0.95, 1.2]
    var fl8: List[Bool] = [False, False, True, False]
    sd.seed_from_benchmark(lib.bin_clip, rets, fl8)
    var seeded = True
    for b in range(lib.num_bins):
        var c = lib.bin_clip[b]
        var want = 0.0 if fl8[c] else min(SM_G_MAX * rets[c], SM_G_MAX)
        seeded = seeded and abs(sd.bin_return[b] - want) < 1e-12 and sd.bin_count[b] == 1.0
    t.truth(seeded, "failed clip's bins at 0, the others at G_MAX x return (capped), one visit each")
    var p8 = sd.probabilities()
    var on_failed = 0.0
    var n_failed_bins = 0
    for b in range(lib.num_bins):
        if fl8[lib.bin_clip[b]]:
            on_failed += p8[b]
            n_failed_bins += 1
    var uniform_share = Float64(n_failed_bins) / Float64(lib.num_bins)
    print("  the failed clip holds", n_failed_bins, "of", lib.num_bins, "bins and", on_failed, "of the draw mass")
    t.truth(on_failed > uniform_share + 0.3, "the draws go to the failed clip well beyond its uniform share")

    print("\n===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_supermimic_lib: " + String(t.fails) + " failed")

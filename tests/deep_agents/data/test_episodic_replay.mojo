"""The shared per-env episodic window replay (`deep_agents/data/episodic_replay`).

    uv run --no-project --python 3.12 --with torch --with mujoco==3.12.0 --with numpy \\
        python tools/supermimic/lib_oracle.py --out /tmp/sm_lib
    pixi run mojo run -I . tests/deep_agents/data/test_episodic_replay.mojo /tmp/sm_lib

The oracle is the SuperMimic reproduction's `ReplayBuffer` run verbatim on a
recorded sequence of 335 adds (3 envs, a 40-row ring per env that wraps,
episodes of 3-25 steps closed by a terminal row), its masks captured and its
picks recorded (`r_*` files of `lib_oracle.py`).

[1] the k = 8 window mask and the start mask on every row, and the windows /
    starts at the reference's picks (states, executed actions, context,
    labels, frames, backfilled history and its actions): exact.

Planted mutant: dropping `ep_start` from the history's previous-action rule
PASSES — the episode id is bumped after every terminal row, so the row before
an episode's first row always carries another id and the id test alone
excludes it. Redundant under that protocol, not untested by accident.
"""

from std.sys import argv

from noeira.deep_agents.data import EpisodicReplay
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
    print("\n[1] the reference's own add sequence, then masks and gathers")
    var calls = load_npy_2d_f64(dump + "/r_calls.npy", 7 + 71 + 29 + 29 + 33)
    var rb = EpisodicReplay[71, 29, 33](3, 40)
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
        raise Error("test_episodic_replay: " + String(t.fails) + " failed")

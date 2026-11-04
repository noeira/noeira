"""EpisodicReplayGpu against EpisodicReplay (the CPU ring, itself gated
against the SuperMimic reproduction's ReplayBuffer).

The same random episodes (6 envs, capacity 24, 70 steps: every ring wraps
several times; ~15 % of steps end an episode and write a terminal row) go
into both rings. Then:

  [1] eligible rows: windows k = 1, 3, 8 and starts h = 4, 10 — the device's
      compacted lists equal the CPU masks exactly
  [2] draws: 400 per query, all eligible, and every eligible env drawn from
  [3] gathers at the same rows: windows (k = 8) and starts (h = 10), all
      seven / six outputs bit-identical

Mutants (planted 2026-11-03): the write-pointer bound `>= k` → `> k` fails
[1] three times. Dropping the episode-start test from the starts'
previous-action rule PASSES, as on the CPU ring (test_episodic_replay): under
the recorder's protocol an episode-start row always follows the previous
episode's terminal row (another episode id) or an empty slot, so the rule is
redundant there.

Run: pixi run -e apple mojo run -I . tests/deep_agents/data/test_episodic_replay_gpu.mojo
"""

from std.random import random_float64, seed

from max.gpu.host import DeviceContext

from noeira.deep_agents.data import EpisodicReplay
from noeira.deep_agents.data.episodic_replay_gpu import (
    EpisodicReplayGpu, META_ENDING, META_EP_START, META_FRAME0, META_FRAME1,
)
from noeira.nn.constants import DT
from noeira.nn.core.ptr import mptr


comptime S = 5
comptime A = 3
comptime X = 2
comptime E = 6
comptime T = 24
comptime STEPS = 70
comptime I32 = DType.int32


struct Tally:
    var ok: Int
    var total: Int

    def __init__(out self):
        self.ok = 0
        self.total = 0

    def check(mut self, cond: Bool, msg: String):
        self.total += 1
        if cond:
            self.ok += 1
            print("  ok:", msg)
        else:
            print("  FAIL:", msg)


def _rl(n: Int) -> List[Scalar[DT]]:
    var l = List[Scalar[DT]](length=n, fill=0)
    for i in range(n):
        l[i] = Scalar[DT](random_float64(-1, 1))
    return l^


def main() raises:
    var ctx = DeviceContext()
    var tl = Tally()
    seed(5)
    var cpu = EpisodicReplay[S, A, X](E, T)
    var gpu = EpisodicReplayGpu[S, A, X](ctx, E, T)
    var hs0 = ctx.enqueue_create_host_buffer[DT](E * S)
    var ha0 = ctx.enqueue_create_host_buffer[DT](E * A)
    var hae0 = ctx.enqueue_create_host_buffer[DT](E * A)
    var hx0 = ctx.enqueue_create_host_buffer[DT](E * X)
    var hrd = ctx.enqueue_create_host_buffer[DT](E * 2)
    var hs1 = ctx.enqueue_create_host_buffer[DT](E * S)
    var hm = ctx.enqueue_create_host_buffer[I32](E * 4)
    var ds0 = ctx.enqueue_create_buffer[DT](E * S)
    var da0 = ctx.enqueue_create_buffer[DT](E * A)
    var dae0 = ctx.enqueue_create_buffer[DT](E * A)
    var dx0 = ctx.enqueue_create_buffer[DT](E * X)
    var drd = ctx.enqueue_create_buffer[DT](E * 2)
    var ds1 = ctx.enqueue_create_buffer[DT](E * S)
    var dm = ctx.enqueue_create_buffer[I32](E * 4)
    ctx.synchronize()
    var ep_start = List[Bool](length=E, fill=True)
    var frame = List[Int](length=E, fill=0)
    var n_end = 0
    for _ in range(STEPS):
        for e in range(E):
            var s0 = _rl(S)
            var a0 = _rl(A)
            var ae0 = _rl(A)
            var x0 = _rl(X)
            var s1 = _rl(S)
            var r = random_float64(0, 1)
            var ending = random_float64() < 0.15
            var lbl = 1.0 if (ending and random_float64() < 0.5) else 0.0
            frame[e] += 1
            cpu.add(e, s0, a0, ae0, x0, r, lbl, frame[e], False, ep_start[e])
            if ending:
                var z = List[Scalar[DT]](length=A, fill=0)
                cpu.add(e, s1, z, z, x0, 0.0, lbl, frame[e] + 1, True, False)
                cpu.end_episode(e)
                n_end += 1
            for k in range(S):
                hs0[e * S + k] = s0[k]
                hs1[e * S + k] = s1[k]
            for k in range(A):
                ha0[e * A + k] = a0[k]
                hae0[e * A + k] = ae0[k]
            for k in range(X):
                hx0[e * X + k] = x0[k]
            hrd[e * 2] = Scalar[DT](r)
            hrd[e * 2 + 1] = Scalar[DT](lbl)
            hm[e * 4 + META_FRAME0] = Int32(frame[e])
            hm[e * 4 + META_FRAME1] = Int32(frame[e] + 1)
            hm[e * 4 + META_EP_START] = 1 if ep_start[e] else 0
            hm[e * 4 + META_ENDING] = 1 if ending else 0
            ep_start[e] = ending
            if ending:
                frame[e] = Int(random_float64(0, 100))
        ctx.enqueue_copy(ds0, hs0)
        ctx.enqueue_copy(da0, ha0)
        ctx.enqueue_copy(dae0, hae0)
        ctx.enqueue_copy(dx0, hx0)
        ctx.enqueue_copy(drd, hrd)
        ctx.enqueue_copy(ds1, hs1)
        ctx.enqueue_copy(dm, hm)
        gpu.add_step(ctx, mptr(ds0.unsafe_ptr()), mptr(da0.unsafe_ptr()), mptr(dae0.unsafe_ptr()), mptr(dx0.unsafe_ptr()), mptr(drd.unsafe_ptr()), mptr(ds1.unsafe_ptr()), mptr(dm.unsafe_ptr()))
        ctx.synchronize()        # the host staging buffers are refilled next step
    print("filled:", STEPS, "steps,", n_end, "episodes ended (terminal rows)")

    print("[1] eligible rows, device compaction vs the CPU masks")
    print("[2] draws")
    comptime NDRAW = 400
    var didx = ctx.enqueue_create_buffer[I32](NDRAW)
    var hidx = ctx.enqueue_create_host_buffer[I32](NDRAW)
    var helig = ctx.enqueue_create_host_buffer[I32](E * T)
    var hcount = ctx.enqueue_create_host_buffer[I32](E)
    var queries: List[Int] = [1, 3, 8, -4, -10]          # k > 0: windows, −h: starts
    for qi in range(len(queries)):
        var q = queries[qi]
        var starts = q < 0
        var kk = -q if starts else q
        var mask = cpu.start_mask(kk) if starts else cpu.window_mask(kk)
        if starts:
            gpu.sample_starts(ctx, kk, NDRAW, UInt64(100 + qi), mptr(didx.unsafe_ptr()))
        else:
            gpu.sample_windows(ctx, kk, NDRAW, UInt64(100 + qi), mptr(didx.unsafe_ptr()))
        ctx.enqueue_copy(helig, gpu.elig)
        ctx.enqueue_copy(hcount, gpu.count)
        ctx.enqueue_copy(hidx, didx)
        ctx.synchronize()
        var gm = List[Bool](length=E * T, fill=False)
        var ordered = True
        for e in range(E):
            var last = -1
            for c in range(Int(hcount[e])):
                var row = Int(helig[e * T + c])
                gm[row] = True
                ordered = ordered and row > last and row // T == e
                last = row
        var same = ordered
        var n_el = 0
        for i in range(E * T):
            same = same and gm[i] == mask[i]
            if mask[i]:
                n_el += 1
        var name = ("start h=" if starts else "window k=") + String(kk)
        tl.check(same and n_el > 0, name + ": " + String(n_el) + " eligible rows, identical, in ring order")
        var all_ok = True
        var hit = List[Bool](length=E, fill=False)
        for i in range(NDRAW):
            var r = Int(hidx[i])
            all_ok = all_ok and r >= 0 and r < E * T and mask[r]
            if r >= 0:
                hit[r // T] = True
        var envs_ok = True
        for e in range(E):
            var any = False
            for t in range(T):
                any = any or mask[e * T + t]
            envs_ok = envs_ok and (hit[e] or not any)
        tl.check(all_ok and envs_ok, name + ": " + String(NDRAW) + " draws all eligible, every eligible env drawn")

    print("[3] gathers at the same rows")
    comptime K = 8
    comptime HH = 10
    comptime N = 64
    var mk = cpu.window_mask(K)
    var idx = EpisodicReplay[S, A, X].sample_indices(mk, N)
    var hi = ctx.enqueue_create_host_buffer[I32](N)
    var di = ctx.enqueue_create_buffer[I32](N)
    ctx.synchronize()
    for i in range(N):
        hi[i] = Int32(idx[i])
    ctx.enqueue_copy(di, hi)
    var w = cpu.windows_at(idx, K)
    var o_s = ctx.enqueue_create_buffer[DT]((K + 1) * N * S)
    var o_ae = ctx.enqueue_create_buffer[DT](K * N * A)
    var o_a = ctx.enqueue_create_buffer[DT](K * N * A)
    var o_x = ctx.enqueue_create_buffer[DT](N * X)
    var o_r = ctx.enqueue_create_buffer[DT](K * N)
    var o_d = ctx.enqueue_create_buffer[DT](K * N)
    var o_f = ctx.enqueue_create_buffer[I32](K * N)
    gpu.windows_at(ctx, mptr(di.unsafe_ptr()), N, K, mptr(o_s.unsafe_ptr()), mptr(o_ae.unsafe_ptr()), mptr(o_a.unsafe_ptr()), mptr(o_x.unsafe_ptr()), mptr(o_r.unsafe_ptr()), mptr(o_d.unsafe_ptr()), mptr(o_f.unsafe_ptr()))
    var h_s = ctx.enqueue_create_host_buffer[DT]((K + 1) * N * S)
    var h_ae = ctx.enqueue_create_host_buffer[DT](K * N * A)
    var h_a = ctx.enqueue_create_host_buffer[DT](K * N * A)
    var h_x = ctx.enqueue_create_host_buffer[DT](N * X)
    var h_r = ctx.enqueue_create_host_buffer[DT](K * N)
    var h_d = ctx.enqueue_create_host_buffer[DT](K * N)
    var h_f = ctx.enqueue_create_host_buffer[I32](K * N)
    ctx.enqueue_copy(h_s, o_s)
    ctx.enqueue_copy(h_ae, o_ae)
    ctx.enqueue_copy(h_a, o_a)
    ctx.enqueue_copy(h_x, o_x)
    ctx.enqueue_copy(h_r, o_r)
    ctx.enqueue_copy(h_d, o_d)
    ctx.enqueue_copy(h_f, o_f)
    ctx.synchronize()
    var bad = 0
    for i in range(len(w.states)):
        bad += 1 if h_s[i] != w.states[i] else 0
    for i in range(len(w.a_exec)):
        bad += 1 if h_ae[i] != w.a_exec[i] or h_a[i] != w.a_clean[i] else 0
    for i in range(len(w.xi)):
        bad += 1 if h_x[i] != w.xi[i] else 0
    for i in range(len(w.reward)):
        bad += 1 if h_r[i] != w.reward[i] or h_d[i] != w.done[i] or Int(h_f[i]) != w.frame[i] else 0
    tl.check(bad == 0, "windows k=8 at 64 rows: states, a_exec, a_clean, xi, reward, done, frame bit-identical (" + String(bad) + " mismatches)")

    var ms = cpu.start_mask(HH)
    var sidx = EpisodicReplay[S, A, X].sample_indices(ms, N)
    for i in range(N):
        hi[i] = Int32(sidx[i])
    ctx.enqueue_copy(di, hi)
    var st = cpu.starts_at(sidx, HH)
    var p_s = ctx.enqueue_create_buffer[DT](N * S)
    var p_x = ctx.enqueue_create_buffer[DT](N * X)
    var p_f = ctx.enqueue_create_buffer[I32](N)
    var p_pa = ctx.enqueue_create_buffer[DT](N * A)
    var p_hs = ctx.enqueue_create_buffer[DT](HH * N * S)
    var p_ha = ctx.enqueue_create_buffer[DT](HH * N * A)
    gpu.starts_at(ctx, mptr(di.unsafe_ptr()), N, HH, mptr(p_s.unsafe_ptr()), mptr(p_x.unsafe_ptr()), mptr(p_f.unsafe_ptr()), mptr(p_pa.unsafe_ptr()), mptr(p_hs.unsafe_ptr()), mptr(p_ha.unsafe_ptr()))
    var q_s = ctx.enqueue_create_host_buffer[DT](N * S)
    var q_x = ctx.enqueue_create_host_buffer[DT](N * X)
    var q_f = ctx.enqueue_create_host_buffer[I32](N)
    var q_pa = ctx.enqueue_create_host_buffer[DT](N * A)
    var q_hs = ctx.enqueue_create_host_buffer[DT](HH * N * S)
    var q_ha = ctx.enqueue_create_host_buffer[DT](HH * N * A)
    ctx.enqueue_copy(q_s, p_s)
    ctx.enqueue_copy(q_x, p_x)
    ctx.enqueue_copy(q_f, p_f)
    ctx.enqueue_copy(q_pa, p_pa)
    ctx.enqueue_copy(q_hs, p_hs)
    ctx.enqueue_copy(q_ha, p_ha)
    ctx.synchronize()
    bad = 0
    var zeros = 0
    for i in range(N * S):
        bad += 1 if q_s[i] != st.state[i] else 0
    for i in range(N * X):
        bad += 1 if q_x[i] != st.xi[i] else 0
    for i in range(N):
        bad += 1 if Int(q_f[i]) != st.frame[i] else 0
    for i in range(N * A):
        bad += 1 if q_pa[i] != st.prev_action[i] else 0
    for i in range(HH * N * S):
        bad += 1 if q_hs[i] != st.hist_states[i] else 0
    for i in range(HH * N * A):
        bad += 1 if q_ha[i] != st.hist_actions[i] else 0
        zeros += 1 if st.hist_actions[i] == 0 else 0
    tl.check(bad == 0 and zeros > 0, "starts h=10 at 64 rows: state, xi, frame, prev_action, history states / actions bit-identical (" + String(bad) + " mismatches; " + String(zeros // A) + " zeroed pre-episode actions exercised)")

    print("\n===", tl.ok, "/", tl.total, "passed ===")
    if tl.ok != tl.total:
        raise Error("test_episodic_replay_gpu: " + String(tl.total - tl.ok) + " failed")

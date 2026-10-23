"""`fb/prior.mojo` gate — a teacher's store turned into prior rows (CPU).

`noeira-docs/FB_MEBE_G1_PLAN.md` §13.5 option 1. Three episodes of 4, 1 and
3 rows (each ending with its terminal row, the collector's convention):

  [1] layout: 5 transitions (3 + 0 + 2; the 1-row episode holds none), 2
      terminal rows AFTER them; every table row is the store row it should
      be, and `nxt` points at the next transition or at the episode's own
      terminal row — never across episodes.
  [2] the regularity reward, row by row, against a hand recomputation of
      `reg_reward_kernel`'s rule: q̇ finite difference to the SUCCESSOR,
      a_prev = 0 on each episode's first row, and the angular-velocity price
      read on the successor. Each term is checked to be non-zero somewhere,
      so a dropped term cannot pass.
  [3] a store round trip: `load_prior_rows` over a written store gives the
      same table, actions and rewards as `prior_rows` over the arrays.
  [4] a store of another width is refused.
  [5] `windows(4)`, the rows the prior-row BC encodes its z from: each
      transition's next 4 successors along `nxt`, holding the episode's own
      terminal row once reached; never a row of another episode.

Run:
    pixi run mojo run -I . tests/deep_agents/test_fb_prior.mojo
"""

from std.math import abs, sin, cos
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.data.column import ColumnSpec
from noeira.data.store import TrajectoryStoreWriter
from noeira.io.fileio import remove_file
from noeira.deep_agents.fb.online_mebe import RegRewardConfig
from noeira.deep_agents.fb.prior import (
    PriorPrice, PriorRows, prior_rows, load_prior_rows,
)


comptime W = 8
comptime ACT = 3
comptime QD_OFF = 2
comptime QD_N = 2
comptime ANG_OFF = 5
comptime DT_CTRL = 0.02
comptime OUT = "/tmp/noeira_test_fb_prior.h5"


def _obs(s: Int, k: Int) -> Float32:
    return Float32(sin(0.3 * Float64(s) + 0.7 * Float64(k)))


def _act(s: Int, k: Int) -> Float32:
    return Float32(cos(0.5 * Float64(s) - 0.9 * Float64(k)))


def _renv(s: Int) -> Float32:
    return Float32(-0.1 * Float64(s + 1))


def _arrays(
    mut obs: List[Float32], mut act: List[Float32], mut r_env: List[Float32],
    mut off: List[Int], mut ln: List[Int],
):
    var lens: List[Int] = [4, 1, 3]
    var n = 0
    for e in range(len(lens)):
        off.append(n)
        ln.append(lens[e])
        n += lens[e]
    obs = List[Float32](length=n * W, fill=0)
    act = List[Float32](length=n * ACT, fill=0)
    r_env = List[Float32](length=n, fill=0)
    for s in range(n):
        for k in range(W):
            obs[s * W + k] = _obs(s, k)
        for k in range(ACT):
            act[s * ACT + k] = _act(s, k)
        r_env[s] = _renv(s)


def _reg() -> RegRewardConfig:
    return RegRewardConfig(
        qd_off=QD_OFF, qd_n=QD_N, dt=DT_CTRL, w_acc=1e-3, w_ar=0.1, w_env=1.0,
    )


def _price() -> PriorPrice:
    return PriorPrice(ANG_OFF, 0.25, 0.02)


def test_layout() raises:
    print("[1] layout: transitions first, terminal rows after, nxt inside episodes ...")
    var obs = List[Float32]()
    var act = List[Float32]()
    var r_env = List[Float32]()
    var off = List[Int]()
    var ln = List[Int]()
    _arrays(obs, act, r_env, off, ln)
    var p = prior_rows(obs, act, r_env, off, ln, W, ACT, _reg(), _price())
    print("      n_valid", p.n_valid, " n_term", p.n_term, " nxt", p.nxt[0], p.nxt[1],
          p.nxt[2], p.nxt[3], p.nxt[4])
    assert_true(p.n_valid == 5 and p.n_term == 2, "wrong transition / terminal counts")
    # table row -> store row
    var src: List[Int] = [0, 1, 2, 5, 6, 3, 7]
    var want_nxt: List[Int] = [1, 2, 5, 4, 6]
    var bad = 0
    for r in range(len(src)):
        for k in range(W):
            if p.obs.data[r * W + k] != Scalar[DT](_obs(src[r], k)):
                bad += 1
    for r in range(p.n_valid):
        for k in range(ACT):
            if p.act.data[r * ACT + k] != Scalar[DT](_act(src[r], k)):
                bad += 1
        if p.nxt[r] != want_nxt[r]:
            bad += 1
    print("      mismatched entries:", bad)
    assert_true(bad == 0, "prior table / actions / nxt not laid out as specified")


def test_reward() raises:
    print("[2] regularity reward == hand recomputation, every term live ...")
    var obs = List[Float32]()
    var act = List[Float32]()
    var r_env = List[Float32]()
    var off = List[Int]()
    var ln = List[Int]()
    _arrays(obs, act, r_env, off, ln)
    var reg = _reg()
    var pr = _price()
    var p = prior_rows(obs, act, r_env, off, ln, W, ACT, reg, pr)
    var src: List[Int] = [0, 1, 2, 5, 6]
    var first: List[Bool] = [True, False, False, True, False]
    var worst = 0.0
    var acc_live = 0
    var ar_live = 0
    var price_live = 0
    for r in range(p.n_valid):
        var s = src[r]
        var acc = 0.0
        for k in range(QD_N):
            var dq = (Float64(_obs(s + 1, QD_OFF + k)) - Float64(_obs(s, QD_OFF + k))) / DT_CTRL
            acc += dq * dq
        var ar = 0.0
        for k in range(ACT):
            var ap = 0.0 if first[r] else Float64(_act(s - 1, k))
            ar += (Float64(_act(s, k)) - ap) ** 2
        var wx = Float64(_obs(s + 1, ANG_OFF))
        var wy = Float64(_obs(s + 1, ANG_OFF + 1))
        var wz = Float64(_obs(s + 1, ANG_OFF + 2))
        var price = pr.w_yaw * wz * wz + pr.w_angxy * (wx * wx + wy * wy)
        var want = -reg.w_acc * acc - reg.w_ar * ar + reg.w_env * (Float64(_renv(s)) - price)
        var err = abs(Float64(p.rew.data[r]) - want) / max(1.0, abs(want))
        if err > worst:
            worst = err
        if reg.w_acc * acc > 1e-3:
            acc_live += 1
        if reg.w_ar * ar > 1e-3:
            ar_live += 1
        if price > 1e-3:
            price_live += 1
    print("      worst rel |Δ|", worst, "  rows with a live acc / ar / price term:",
          acc_live, ar_live, price_live)
    assert_true(worst < 1e-5, "prior reward differs from the hand recomputation")
    assert_true(acc_live > 0 and ar_live > 0 and price_live > 0, "a reward term is dead in the fixture")


def _write_store(path: String, w: Int) raises:
    var obs = List[Float32]()
    var act = List[Float32]()
    var r_env = List[Float32]()
    var off = List[Int]()
    var ln = List[Int]()
    _arrays(obs, act, r_env, off, ln)
    var cols = List[ColumnSpec]()
    cols.append(ColumnSpec(String("obs"), DType.float32, w))
    cols.append(ColumnSpec(String("action"), DType.float32, ACT))
    cols.append(ColumnSpec(String("r_env"), DType.float32, 1))
    cols.append(ColumnSpec(String("task"), DType.int32, 1))
    var wr = TrajectoryStoreWriter(path, cols^, env_id=String("test"))
    var bo = List[Float32](length=8 * w, fill=0)
    var ba = List[Float32](length=8 * ACT, fill=0)
    var br = List[Float32](length=8, fill=0)
    var bt = List[Int32](length=8, fill=0)
    for e in range(len(off)):
        for i in range(ln[e]):
            var s = off[e] + i
            for k in range(w):
                bo[i * w + k] = _obs(s, k) if k < W else Float32(0)
            for k in range(ACT):
                ba[i * ACT + k] = act[s * ACT + k]
            br[i] = r_env[s]
            bt[i] = Int32(e)
        wr.append[DType.float32](String("obs"), bo.unsafe_ptr().as_unsafe_any_origin(), ln[e])
        wr.append[DType.float32](String("action"), ba.unsafe_ptr().as_unsafe_any_origin(), ln[e])
        wr.append[DType.float32](String("r_env"), br.unsafe_ptr().as_unsafe_any_origin(), ln[e])
        wr.append[DType.int32](String("task"), bt.unsafe_ptr().as_unsafe_any_origin(), ln[e])
        wr.end_episode()
    wr.close()


def test_round_trip() raises:
    print("[3] store round trip == arrays ...")
    _write_store(String(OUT), W)
    var a = load_prior_rows(String(OUT), W, ACT, _reg(), _price())
    var obs = List[Float32]()
    var act = List[Float32]()
    var r_env = List[Float32]()
    var off = List[Int]()
    var ln = List[Int]()
    _arrays(obs, act, r_env, off, ln)
    var b = prior_rows(obs, act, r_env, off, ln, W, ACT, _reg(), _price())
    var bad = 0
    if a.n_valid != b.n_valid or a.n_term != b.n_term:
        bad += 1
    else:
        for i in range((b.n_valid + b.n_term) * W):
            if a.obs.data[i] != b.obs.data[i]:
                bad += 1
        for i in range(b.n_valid * ACT):
            if a.act.data[i] != b.act.data[i]:
                bad += 1
        for i in range(b.n_valid):
            if a.rew.data[i] != b.rew.data[i] or a.nxt[i] != b.nxt[i]:
                bad += 1
    print("      mismatches:", bad)
    remove_file(String(OUT))
    assert_true(bad == 0, "the store round trip changed the prior rows")


def test_width_refused() raises:
    print("[4] a store of another width is refused ...")
    _write_store(String(OUT), W + 1)
    var refused = False
    try:
        _ = load_prior_rows(String(OUT), W, ACT, _reg(), _price())
    except e:
        refused = True
        print("      refused:", e)
    remove_file(String(OUT))
    assert_true(refused, "a store of the wrong width was accepted")


def test_windows() raises:
    print("[5] windows: successors along nxt, held at the episode's terminal row ...")
    var obs = List[Float32]()
    var act = List[Float32]()
    var r_env = List[Float32]()
    var off = List[Int]()
    var ln = List[Int]()
    _arrays(obs, act, r_env, off, ln)
    var p = prior_rows(obs, act, r_env, off, ln, W, ACT, _reg(), _price())
    var w = p.windows(4)
    # transitions 0..2 (episode 0, terminal 5), 3..4 (episode 2, terminal 6)
    var want: List[Int] = [
        1, 2, 5, 5,
        2, 5, 5, 5,
        5, 5, 5, 5,
        4, 6, 6, 6,
        6, 6, 6, 6,
    ]
    var bad = 0
    for i in range(len(want)):
        if w[i] != want[i]:
            bad += 1
    print("      window rows:", w[0], w[1], w[2], w[3], "|", w[12], w[13], w[14], w[15],
          "  mismatches", bad)
    assert_true(len(w) == len(want) and bad == 0, "windows do not follow nxt inside the episode")
    var w1 = p.windows(1)
    var bad1 = 0
    for r in range(p.n_valid):
        if w1[r] != p.nxt[r]:
            bad1 += 1
    assert_true(bad1 == 0, "windows(1) is not nxt")


def main() raises:
    test_layout()
    test_reward()
    test_round_trip()
    test_width_refused()
    test_windows()
    print("ALL PASS — fb prior rows")

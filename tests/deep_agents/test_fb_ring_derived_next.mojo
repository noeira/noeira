""""`s'` derived from the ring instead of stored — the gates that are not vacuous.

`docs/BFM_ZERO_G1_REPRODUCTION.md` §12.23 dropped the `next_obs` column from
the FB replay ring. `next_obs` is now `r_obs[(row + LANES) % CAP]`, which is
correct for every row EXCEPT the last one before the env resets; those are
flagged in `r_bnd` and remapped away at draw time.

That buys 5360 B -> 3256 B per transition. It also makes three silent failures
possible that could not happen while the successor was stored:

  * a pair that STRADDLES A RESET — `s` from the end of one episode, `s'` the
    reset state of the next. The loss would be fitting a teleport;
  * a row whose successor has NOT BEEN WRITTEN YET (the newest `LANES` rows),
    which before the first wrap is uninitialised memory and after it is a row
    from `CAP` rows ago;
  * the derivation itself being off by a lane or a step, which no amount of
    self-consistency will show.

⚠ A gate asserting "the gathered `s'` equals `r_obs` of the next row" is TRUE
BY CONSTRUCTION and observes nothing. What this file does instead: it records
a synthetic rollout in which EVERY observation encodes its own `(step, lane)`,
so a gathered row can be decoded back to the transition it came from, and
checks the decoded pairs against the host's record of what was fed in.

Non-vacuity is a POSITIVE CONTROL, not an argument: the same draw is run twice,
once with the real `r_bnd` and once with `r_bnd` all-zero. The all-zero run
MUST return boundary rows (otherwise the fixture never put one where the draw
could reach it, and "no boundary was drawn" means nothing) and the real run
MUST return none. Same seed, same offset, so the two differ only in the remap.

Nothing else instantiates these kernels where they can be built: their only
caller is `FBOnlineAgent`, reachable on Metal only through a full agent with
networks attached. A generic Mojo kernel is type-checked when instantiated, so
an edit to the ring could otherwise reach the box as a compile error.

Run: pixi run -e apple mojo run -I . tests/deep_agents/test_fb_ring_derived_next.mojo
"""

from std.testing import assert_true
from layout import Layout, LayoutTensor
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.ptr import mptr
from noeira.data.resident import IDX_DT
from noeira.deep_agents.fb.online import (
    ring_store_kernel, ring_indices_kernel, ring_next_idx_kernel,
)
from noeira.deep_agents.fb.kernels import gather_rows_kernel

comptime OBS = 3
comptime ACT = 2
comptime D = 4
comptime T_EP = 5          # reset every 5 steps, as the G1 driver does at 500
comptime DRAWS = 256
comptime SEED = UInt64(12345)


def _blk(n: Int) -> Int:
    return (n + TPB - 1) // TPB


def _encode[LANES: Int](step: Int, lane: Int, k: Int) -> Float64:
    """One float that names the transition it belongs to.

    `(step * LANES + lane) * 10 + k` — at most 923 here, exact in float32, and
    a wrong lane or a wrong step lands on a value no other row carries. A
    fixture of constants (or `Deterministic` init) has made gates in this track
    vacuous twice: with every row equal, a gather from the wrong row passes.
    """
    return Float64((step * LANES + lane) * 10 + k)


struct Drawn(Copyable, Movable):
    var step: Int
    var lane: Int

    def __init__(out self, step: Int, lane: Int):
        self.step = step
        self.lane = lane


def _decode[LANES: Int](v: Float64, k: Int) raises -> Drawn:
    var id = Int(v + 0.5) // 10
    var got_k = Int(v + 0.5) % 10
    if got_k != k:
        raise Error(
            "gathered element " + String(k) + " carries element "
            + String(got_k) + ": the gather is off within the row"
        )
    return Drawn(id // LANES, id % LANES)


def _case[
    CAP: Int, LANES: Int, NSTEPS: Int, PRE: Int
](c: DeviceContext) raises:
    print("  ── CAP", CAP, " LANES", LANES, " CAP % LANES =", CAP % LANES,
          " steps", NSTEPS, "(", NSTEPS * LANES, "rows into", CAP, ")")

    var r_obs = c.enqueue_create_buffer[DT](CAP * OBS)
    var r_act = c.enqueue_create_buffer[DT](CAP * ACT)
    var r_z = c.enqueue_create_buffer[DT](CAP * D)
    var r_term = c.enqueue_create_buffer[DT](CAP)
    var r_bnd = c.enqueue_create_buffer[DT](CAP)
    # `r_age` (steps since the lane's reset) is what the DERIVED observation
    # tail reads, not this gate — but `ring_store_kernel` writes it, so it has
    # to exist. §12.36 added the column; this file's job is still `s'`.
    var r_age = c.enqueue_create_buffer[DT](CAP)
    r_obs.enqueue_fill(Scalar[DT](-1.0))
    r_bnd.enqueue_fill(Scalar[DT](1.0))

    var obs_src = c.enqueue_create_buffer[DT](LANES * OBS)
    var h_obs = c.enqueue_create_host_buffer[DT](LANES * OBS)
    var act_src = c.enqueue_create_buffer[DT](LANES * ACT)
    var z_src = c.enqueue_create_buffer[DT](LANES * D)
    var term_src = c.enqueue_create_buffer[DT](LANES)
    act_src.enqueue_fill(Scalar[DT](0.0))
    z_src.enqueue_fill(Scalar[DT](0.0))
    term_src.enqueue_fill(Scalar[DT](0.0))

    comptime W = OBS + ACT + D + 3
    var pos = 0
    var size = 0
    for s in range(NSTEPS):
        for l in range(LANES):
            for k in range(OBS):
                h_obs[l * OBS + k] = Scalar[DT](_encode[LANES](s, l, k))
        c.enqueue_copy(obs_src, h_obs)
        # the driver's rule: the reset runs at the START of a step, so it is
        # step `s` with `(s + 1) % T_EP == 0` whose successor is post-reset
        var bnd = 1 if (s + 1) % T_EP == 0 else 0
        c.enqueue_function[ring_store_kernel[OBS, ACT, D, CAP, LANES]](
            mptr(obs_src.unsafe_ptr()), mptr(act_src.unsafe_ptr()),
            mptr(term_src.unsafe_ptr()), mptr(z_src.unsafe_ptr()),
            mptr(r_obs.unsafe_ptr()), mptr(r_act.unsafe_ptr()),
            mptr(r_term.unsafe_ptr()), mptr(r_bnd.unsafe_ptr()),
            mptr(r_age.unsafe_ptr()), mptr(r_z.unsafe_ptr()),
            Int32(pos), Int32(bnd), Int32(s if s < 5 else 5),
            grid_dim=_blk(LANES * W), block_dim=TPB,
        )
        # ⚠ the H2D copy above is ENQUEUED: without this the next iteration
        # overwrites `h_obs` before the copy runs and every row in the ring
        # ends up carrying the LAST step's observation. That failure looks
        # exactly like a broken derivation (`s'` one step off, every draw).
        c.synchronize()
        pos = (pos + LANES) % CAP
        size = size + LANES
        if size > CAP:
            size = CAP

    print("  after the rollout: size", size, " pos", pos,
          " (wrapped" if size == CAP else " (not wrapped", ")")

    var size_d = c.enqueue_create_buffer[DType.int32](1)
    size_d.enqueue_fill(Int32(size))
    var pos_d = c.enqueue_create_buffer[DType.int32](1)
    pos_d.enqueue_fill(Int32(pos))
    var off_d = c.enqueue_create_buffer[DType.uint64](1)
    off_d.enqueue_fill(UInt64(0))
    var size_lt = LayoutTensor[DType.int32, Layout.row_major(1)](size_d)
    var pos_lt = LayoutTensor[DType.int32, Layout.row_major(1)](pos_d)
    var off_lt = LayoutTensor[DType.uint64, Layout.row_major(1)](off_d)

    # ── the all-zero-`r_bnd` POSITIVE CONTROL ────────────────────────────
    # Identical seed and offset, so the raw uniform and therefore `j` are the
    # same; the only difference is whether the remap can fire. If this run
    # draws no boundary row, the fixture never put one within reach and the
    # real run's "no boundary" proves nothing.
    var zero_bnd = c.enqueue_create_buffer[DT](CAP)
    zero_bnd.enqueue_fill(Scalar[DT](0.0))
    var idx_ctl = c.enqueue_create_buffer[IDX_DT](DRAWS)
    c.enqueue_function[ring_indices_kernel[DRAWS, CAP, LANES]](
        LayoutTensor[IDX_DT, Layout.row_major(DRAWS)](idx_ctl),
        size_lt, pos_lt, mptr(zero_bnd.unsafe_ptr()), SEED, off_lt,
        grid_dim=_blk(DRAWS), block_dim=TPB,
    )

    var idx = c.enqueue_create_buffer[IDX_DT](DRAWS)
    var idx_n = c.enqueue_create_buffer[IDX_DT](DRAWS)
    c.enqueue_function[ring_indices_kernel[DRAWS, CAP, LANES]](
        LayoutTensor[IDX_DT, Layout.row_major(DRAWS)](idx),
        size_lt, pos_lt, mptr(r_bnd.unsafe_ptr()), SEED, off_lt,
        grid_dim=_blk(DRAWS), block_dim=TPB,
    )
    c.enqueue_function[ring_next_idx_kernel[DRAWS, CAP, LANES]](
        mptr(idx.unsafe_ptr()), mptr(idx_n.unsafe_ptr()),
        grid_dim=_blk(DRAWS), block_dim=TPB,
    )
    var bs = c.enqueue_create_buffer[DT](DRAWS * OBS)
    var bsn = c.enqueue_create_buffer[DT](DRAWS * OBS)
    c.enqueue_function[gather_rows_kernel[OBS, DRAWS]](
        mptr(r_obs.unsafe_ptr()), mptr(idx.unsafe_ptr()), mptr(bs.unsafe_ptr()),
        grid_dim=_blk(DRAWS * OBS), block_dim=TPB,
    )
    c.enqueue_function[gather_rows_kernel[OBS, DRAWS]](
        mptr(r_obs.unsafe_ptr()), mptr(idx_n.unsafe_ptr()), mptr(bsn.unsafe_ptr()),
        grid_dim=_blk(DRAWS * OBS), block_dim=TPB,
    )

    var h_bnd = c.enqueue_create_host_buffer[DT](CAP)
    var h_idx = c.enqueue_create_host_buffer[IDX_DT](DRAWS)
    var h_ctl = c.enqueue_create_host_buffer[IDX_DT](DRAWS)
    var h_bs = c.enqueue_create_host_buffer[DT](DRAWS * OBS)
    var h_bsn = c.enqueue_create_host_buffer[DT](DRAWS * OBS)
    c.enqueue_copy(h_bnd, r_bnd)
    c.enqueue_copy(h_idx, idx)
    c.enqueue_copy(h_ctl, idx_ctl)
    c.enqueue_copy(h_bs, bs)
    c.enqueue_copy(h_bsn, bsn)
    c.synchronize()

    # ── the window: which rows the draw is allowed to return ─────────────
    var base = (pos - size) % CAP
    if base < 0:
        base += CAP
    var n = size - LANES
    var in_window = List[Bool](length=CAP, fill=False)
    var bnd_in_window = 0
    for j in range(n):
        var row = (base + j) % CAP
        in_window[row] = True
        if Float64(h_bnd[row]) != 0.0:
            bnd_in_window += 1
    print("  window: base", base, " n", n, " boundary rows inside it",
          bnd_in_window)
    assert_true(
        bnd_in_window > 0,
        "vacuous fixture: the sampling window holds no boundary row, so"
        " 'no boundary was drawn' would be true of a kernel with no remap",
    )

    # ── the positive control ─────────────────────────────────────────────
    var ctl_bnd = 0
    for i in range(DRAWS):
        if Float64(h_bnd[Int(h_ctl[i])]) != 0.0:
            ctl_bnd += 1
    print("  positive control (r_bnd all zero):", ctl_bnd, "/", DRAWS,
          "draws landed on a boundary row")
    assert_true(
        ctl_bnd > 0,
        "the control drew no boundary row: the remap has nothing to remove"
        " and this file's main assertion is unfalsifiable",
    )

    # ── the real draw ────────────────────────────────────────────────────
    var out_of_window = 0
    var drew_bnd = 0
    var wrong_lane = 0
    var wrong_step = 0
    var straddled = 0
    var unwritten = 0
    for i in range(DRAWS):
        var row = Int(h_idx[i])
        if not in_window[row]:
            out_of_window += 1
        if Float64(h_bnd[row]) != 0.0:
            drew_bnd += 1
        var d = _decode[LANES](Float64(h_bs[i * OBS]), 0)
        var dn = _decode[LANES](Float64(h_bsn[i * OBS]), 0)
        if Float64(h_bs[i * OBS]) < 0.0 or Float64(h_bsn[i * OBS]) < 0.0:
            unwritten += 1
            continue
        # THE DERIVATION, against the host's record of what was fed in:
        # the successor must be the SAME lane, ONE step later.
        if dn.lane != d.lane:
            wrong_lane += 1
        if dn.step != d.step + 1:
            wrong_step += 1
        # and the pair must not span a reset
        if (d.step + 1) % T_EP == 0:
            straddled += 1

    print("  draws", DRAWS, " outside the window", out_of_window,
          " on a boundary row", drew_bnd, " unwritten", unwritten)
    print("  derived s': wrong lane", wrong_lane, " wrong step", wrong_step,
          " straddling a reset", straddled)

    assert_true(
        out_of_window == 0,
        "a draw returned a row outside [base, base + size - LANES): either"
        " the newest rows (no successor written) or a row the ring has"
        " already overwritten",
    )
    assert_true(
        unwritten == 0,
        "a gathered row is still at its fill value: the draw reached a row"
        " the rollout never wrote",
    )
    assert_true(
        drew_bnd == 0,
        "a boundary row survived the draw: its successor is the post-reset"
        " observation of the NEXT episode, not its own next state",
    )
    assert_true(
        wrong_lane == 0,
        "the derived s' came from a different lane than s — `(row + LANES)"
        " % CAP` does not track the lane through the write order",
    )
    assert_true(
        wrong_step == 0,
        "the derived s' is not one env step after s",
    )
    assert_true(
        straddled == 0,
        "a sampled pair spans an episode reset",
    )

    # ── the same, BEFORE the ring wraps ──────────────────────────────────
    # `base` is 0 there and `pos == size`, a different branch of the same two
    # lines; the wrapped case above would pass with the pre-wrap arithmetic
    # hard-coded and vice versa.
    var p2_obs = c.enqueue_create_buffer[DT](CAP * OBS)
    var p2_bnd = c.enqueue_create_buffer[DT](CAP)
    p2_obs.enqueue_fill(Scalar[DT](-1.0))
    p2_bnd.enqueue_fill(Scalar[DT](1.0))
    var pos2 = 0
    var size2 = 0
    for s in range(PRE):
        for l in range(LANES):
            for k in range(OBS):
                h_obs[l * OBS + k] = Scalar[DT](_encode[LANES](s, l, k))
        c.enqueue_copy(obs_src, h_obs)
        c.enqueue_function[ring_store_kernel[OBS, ACT, D, CAP, LANES]](
            mptr(obs_src.unsafe_ptr()), mptr(act_src.unsafe_ptr()),
            mptr(term_src.unsafe_ptr()), mptr(z_src.unsafe_ptr()),
            mptr(p2_obs.unsafe_ptr()), mptr(r_act.unsafe_ptr()),
            mptr(r_term.unsafe_ptr()), mptr(p2_bnd.unsafe_ptr()),
            mptr(r_age.unsafe_ptr()), mptr(r_z.unsafe_ptr()),
            Int32(pos2), Int32(1 if (s + 1) % T_EP == 0 else 0),
            Int32(s if s < 5 else 5),
            grid_dim=_blk(LANES * W), block_dim=TPB,
        )
        c.synchronize()
        pos2 = (pos2 + LANES) % CAP
        size2 += LANES

    size_d.enqueue_fill(Int32(size2))
    pos_d.enqueue_fill(Int32(pos2))
    off_d.enqueue_fill(UInt64(0))
    c.enqueue_function[ring_indices_kernel[DRAWS, CAP, LANES]](
        LayoutTensor[IDX_DT, Layout.row_major(DRAWS)](idx),
        size_lt, pos_lt, mptr(p2_bnd.unsafe_ptr()), SEED, off_lt,
        grid_dim=_blk(DRAWS), block_dim=TPB,
    )
    c.enqueue_function[ring_next_idx_kernel[DRAWS, CAP, LANES]](
        mptr(idx.unsafe_ptr()), mptr(idx_n.unsafe_ptr()),
        grid_dim=_blk(DRAWS), block_dim=TPB,
    )
    c.enqueue_function[gather_rows_kernel[OBS, DRAWS]](
        mptr(p2_obs.unsafe_ptr()), mptr(idx.unsafe_ptr()), mptr(bs.unsafe_ptr()),
        grid_dim=_blk(DRAWS * OBS), block_dim=TPB,
    )
    c.enqueue_function[gather_rows_kernel[OBS, DRAWS]](
        mptr(p2_obs.unsafe_ptr()), mptr(idx_n.unsafe_ptr()), mptr(bsn.unsafe_ptr()),
        grid_dim=_blk(DRAWS * OBS), block_dim=TPB,
    )
    c.enqueue_copy(h_idx, idx)
    c.enqueue_copy(h_bs, bs)
    c.enqueue_copy(h_bsn, bsn)
    var h_bnd2 = c.enqueue_create_host_buffer[DT](CAP)
    c.enqueue_copy(h_bnd2, p2_bnd)
    c.synchronize()

    var pre_bad = 0
    var pre_bnd = 0
    var pre_unwritten = 0
    var pre_max = 0
    var pre_bnd_in_window = 0
    for row in range(size2 - LANES):
        if Float64(h_bnd2[row]) != 0.0:
            pre_bnd_in_window += 1
    assert_true(
        pre_bnd_in_window > 0,
        "vacuous: the pre-wrap window holds no boundary row, so its"
        " 'no boundary drawn' assertion cannot fail",
    )
    for i in range(DRAWS):
        var row = Int(h_idx[i])
        if row > pre_max:
            pre_max = row
        if Float64(h_bnd2[row]) != 0.0:
            pre_bnd += 1
        if Float64(h_bs[i * OBS]) < 0.0 or Float64(h_bsn[i * OBS]) < 0.0:
            pre_unwritten += 1
            continue
        var d = _decode[LANES](Float64(h_bs[i * OBS]), 0)
        var dn = _decode[LANES](Float64(h_bsn[i * OBS]), 0)
        if dn.lane != d.lane or dn.step != d.step + 1:
            pre_bad += 1
    print("  pre-wrap: size", size2, " pos", pos2, " highest row drawn",
          pre_max, "(bound", size2 - LANES - 1, ") boundary rows in window",
          pre_bnd_in_window, " drawn", pre_bnd,
          " unwritten", pre_unwritten, " wrong successor", pre_bad)
    assert_true(
        pre_max <= size2 - LANES - 1,
        "before the wrap the draw reached into rows whose successor has not"
        " been written yet",
    )
    assert_true(pre_unwritten == 0, "a pre-wrap draw gathered an unwritten row")
    assert_true(pre_bnd == 0, "a pre-wrap draw returned a boundary row")
    assert_true(pre_bad == 0, "a pre-wrap derived s' is not the lane's next step")

    print("    OK")


def main() raises:
    var c = DeviceContext()
    print("FB replay ring: `s'` derived from `(row + LANES) % CAP`")
    print("  OBS", OBS, " ACT", ACT, " D", D, " T_EP", T_EP, " draws", DRAWS)
    # ⚠ CAP % LANES == 0 is the EASY case and it is NOT the production one:
    # `CAP = 4_000_000` with 1024 lanes leaves 3906.25 blocks, so a write
    # block STRADDLES the wrap point and the `size` clamp fires PARTIALLY
    # (size rises by less than LANES on that one step) while `pos` advances by
    # a full LANES. `base = (pos - size) mod CAP` and the newest-LANES
    # exclusion both have to survive that. A divisible fixture never asks.
    _case[40, 4, 23, 7](c)      # divisible: 10 blocks per lap, clean wrap
    _case[42, 4, 25, 7](c)      # NOT divisible: 10.5 blocks, partial clamp
    _case[1000, 64, 40, 12](c)  # NOT divisible: 15.625 blocks, wider rows
    print("FB_RING_DERIVED_NEXT OK")

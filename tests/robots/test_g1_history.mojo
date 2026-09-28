""""`last_action` + `history_actor` against the G2-validated oracle.

The 401 dims our actor never had (docs §12.34). The layout is KEY-MAJOR with
NEWEST FIRST inside each key, which is the part that produces a vector of the
right SIZE and the wrong MEANING if transcribed step-major — nothing but a
value check can see that, so this file is a value check.

The oracle (`tools/g1/bfm_zero_tracking_oracle.py`) is the authority, not the
yaml: G2 ran the RELEASED ONNX actor through its 721-vector and reproduced the
reference's own numbers to three decimals on 37 of 39 segments (§11). If our
kernels agree with it, they agree with the thing that passed.

Checks:
  * the constants agree with the oracle's own (`HIST_KEYS` / `HIST_DIMS` /
    `HIST_LEN`, `ACTOR_OBS_DIM`);
  * a SEQUENCE of pushes, with distinct values per step so a wrong step index
    is visible, reproduces the oracle's `hist` vector exactly;
  * the reset row is NEVER pushed, and a reset zeroes both buffers;
  * NON-VACUITY: the fixture's steps must differ, and the assembled vector
    must actually change between steps — a frozen buffer would pass a
    same-value comparison.

Run: pixi run -e apple mojo run -I . tests/robots/test_g1_history.mojo
"""

from std.testing import assert_true
from std.python import Python, PythonObject
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.ptr import mptr
from noeira.envs.robots.unitree_g1_xml import (
    UNITREE_G1_OBS_DIM, UNITREE_G1_STATE_DIM,
)
from noeira.envs.robots.unitree_g1_history import (
    G1_HIST_LEN, G1_HIST_DIM, G1_HIST_STEP, G1_N_ACT, G1_ACTOR_EXTRA,
    UNITREE_G1_FULL_OBS_DIM, G1_LAST_ACTION_DIM,
    G1_S_DOFPOS, G1_S_DOFVEL, G1_S_GRAV, G1_S_ANGVEL,
    g1_hist_push_kernel, g1_hist_reset_kernel, g1_pack_full_obs_kernel,
    g1_hist_gather_kernel, g1_build_tail_spec,
)
from noeira.deep_agents.fb.kernels import derive_tail_kernel
from noeira.data.resident import IDX_DT

comptime LANES = 2
comptime NSTEP = 6


def _blk(n: Int) -> Int:
    return (n + TPB - 1) // TPB


def main() raises:
    var c = DeviceContext()
    print("G1 last_action + history_actor vs the oracle")

    var sys = Python.import_module("sys")
    _ = sys.path.insert(0, "tools/g1")
    var orc = Python.import_module("bfm_zero_tracking_oracle")
    var np = Python.import_module("numpy")

    # ── [1] the constants ────────────────────────────────────────────
    var o_len = Int(Float64(py=orc.HIST_LEN))
    var o_keys = orc.HIST_KEYS
    var o_dims = orc.HIST_DIMS
    var o_step = 0
    for k in o_keys:
        o_step += Int(Float64(py=o_dims[k]))
    var o_actor = Int(Float64(py=orc.ACTOR_OBS_DIM))
    print("  oracle: HIST_LEN", o_len, " per-step", o_step,
          " ACTOR_OBS_DIM", o_actor)
    print("  ours  : HIST_LEN", G1_HIST_LEN, " per-step", G1_HIST_STEP,
          " state+last+hist+z", UNITREE_G1_STATE_DIM + G1_ACTOR_EXTRA + 256)
    assert_true(o_len == G1_HIST_LEN, "HIST_LEN disagrees with the oracle")
    assert_true(o_step == G1_HIST_STEP, "per-step width disagrees")
    assert_true(
        o_actor == UNITREE_G1_STATE_DIM + G1_ACTOR_EXTRA + 256,
        "the actor input width disagrees with the oracle's ACTOR_OBS_DIM",
    )

    # ── the fixture: distinct values per (step, lane, element) ───────
    var h_obs = c.enqueue_create_host_buffer[DT](LANES * UNITREE_G1_OBS_DIM)
    var d_obs = c.enqueue_create_buffer[DT](LANES * UNITREE_G1_OBS_DIM)
    var h_act = c.enqueue_create_host_buffer[DT](LANES * G1_N_ACT)
    var d_act = c.enqueue_create_buffer[DT](LANES * G1_N_ACT)
    var d_hist = c.enqueue_create_buffer[DT](LANES * G1_HIST_DIM)
    var d_live = c.enqueue_create_buffer[DT](LANES)
    var d_mask = c.enqueue_create_buffer[DT](LANES)
    var d_full = c.enqueue_create_buffer[DT](LANES * UNITREE_G1_FULL_OBS_DIM)
    d_hist.enqueue_fill(Scalar[DT](0.0))
    d_act.enqueue_fill(Scalar[DT](0.0))
    d_live.enqueue_fill(Scalar[DT](0.0))   # the reset row is never pushed
    d_mask.enqueue_fill(Scalar[DT](0.0))

    # the oracle's buffers, mirrored on the host
    var hist_py = Python.dict()
    for k in o_keys:
        hist_py[k] = np.zeros(Python.tuple(o_len, o_dims[k]))
    var last_py = np.zeros(G1_N_ACT)

    def _val(step: Int, lane: Int, i: Int) -> Float64:
        # Distinct across all three so a wrong step, lane or element shows.
        # ⚠ Kept SMALL on purpose: the buffers are float32 and the oracle is
        # float64, so at magnitude ~500 the representation gap (3.7e-06) sits
        # above an element-sized error and the comparison stops discriminating.
        # Here the smallest signal (1e-4, one element) is ~200x float32's
        # resolution at this magnitude.
        return Float64(step) * 1.0 + Float64(lane) * 0.25 + Float64(i) * 0.0001

    var changed = 0
    var prev_first = Float64(-1e30)
    for s in range(NSTEP):
        for l in range(LANES):
            for i in range(UNITREE_G1_OBS_DIM):
                h_obs[l * UNITREE_G1_OBS_DIM + i] = Scalar[DT](_val(s, l, i))
            for i in range(G1_N_ACT):
                h_act[l * G1_N_ACT + i] = Scalar[DT](_val(s, l, i) + 0.5)
        c.enqueue_copy(d_obs, h_obs)
        c.synchronize()

        # ⚠ `d_act` still holds step s-1's action here, and MUST: the
        # reference pushes before updating `last_action`, so the newest
        # `actions` entry is the action applied at step t-1. Uploading this
        # step's action first is an off-by-one that shows ONLY in the 116
        # `actions` dims — which is how this fixture caught it.
        # push (skipped on the reset row by `live`)
        c.enqueue_function[g1_hist_push_kernel[LANES]](
            mptr(d_obs.unsafe_ptr()), mptr(d_act.unsafe_ptr()),
            mptr(d_hist.unsafe_ptr()), mptr(d_live.unsafe_ptr()),
            grid_dim=_blk(LANES * G1_HIST_STEP), block_dim=TPB,
        )
        c.synchronize()
        # the oracle's equivalent: push only from step 1
        if s >= 1:
            for k in o_keys:
                var buf = hist_py[k]
                var d = Int(Float64(py=o_dims[k]))
                for j in range(o_len - 1, 0, -1):
                    for i in range(d):
                        buf[Python.tuple(j, i)] = buf[Python.tuple(j - 1, i)]
                for i in range(d):
                    if String(k) == "actions":
                        buf[Python.tuple(0, i)] = last_py[i]
                    else:
                        var so = (
                            G1_S_ANGVEL if String(k) == "base_ang_vel"
                            else G1_S_DOFPOS if String(k) == "dof_pos"
                            else G1_S_DOFVEL if String(k) == "dof_vel"
                            else G1_S_GRAV
                        )
                        buf[Python.tuple(0, i)] = _val(s, 0, so + i)
        # NOW update last_action, on both sides, as the reference does
        c.enqueue_copy(d_act, h_act)
        c.synchronize()
        for i in range(G1_N_ACT):
            last_py[i] = _val(s, 0, i) + 0.5
        # after the first push attempt every lane is live
        d_live.enqueue_fill(Scalar[DT](1.0))

        # compare lane 0's history against the oracle's concatenation
        var h_hist = c.enqueue_create_host_buffer[DT](LANES * G1_HIST_DIM)
        c.enqueue_copy(h_hist, d_hist)
        c.synchronize()
        var parts = Python.list()
        for k in o_keys:
            _ = parts.append(hist_py[k].reshape(-1))
        var flat = np.concatenate(parts)
        var worst = 0.0
        for i in range(G1_HIST_DIM):
            var g = Float64(h_hist[i])
            var w = Float64(py=flat[i])
            var e = abs(g - w)
            if e > worst:
                worst = e
        if s == NSTEP - 1:
            print("  step", s, " worst |ours - oracle| over the 372 =", worst)
        assert_true(
            worst < 1e-5,
            "history disagrees with the oracle at step " + String(s)
            + ": worst " + String(worst) + " — the layout is KEY-MAJOR with"
            " newest first; a step-major transcription has the right size",
        )
        var f0 = Float64(h_hist[0])
        if f0 != prev_first:
            changed += 1
        prev_first = f0

    assert_true(
        changed >= NSTEP - 2,
        "the history did not change between steps — a frozen buffer would"
        " pass the comparison above if the oracle mirror were frozen too",
    )

    # ── the packed 928 ───────────────────────────────────────────────
    c.enqueue_function[g1_pack_full_obs_kernel[LANES]](
        mptr(d_obs.unsafe_ptr()), mptr(d_act.unsafe_ptr()),
        mptr(d_hist.unsafe_ptr()), mptr(d_full.unsafe_ptr()),
        grid_dim=_blk(LANES * UNITREE_G1_FULL_OBS_DIM), block_dim=TPB,
    )
    var h_full = c.enqueue_create_host_buffer[DT](LANES * UNITREE_G1_FULL_OBS_DIM)
    var h_hist2 = c.enqueue_create_host_buffer[DT](LANES * G1_HIST_DIM)
    c.enqueue_copy(h_full, d_full)
    c.enqueue_copy(h_hist2, d_hist)
    c.synchronize()
    var bad = 0
    for l in range(LANES):
        var b = l * UNITREE_G1_FULL_OBS_DIM
        for i in range(UNITREE_G1_OBS_DIM):
            if abs(Float64(h_full[b + i])
                   - Float64(h_obs[l * UNITREE_G1_OBS_DIM + i])) > 1e-9:
                bad += 1
        for i in range(G1_N_ACT):
            if abs(Float64(h_full[b + UNITREE_G1_OBS_DIM + i])
                   - Float64(h_act[l * G1_N_ACT + i])) > 1e-9:
                bad += 1
        for i in range(G1_HIST_DIM):
            if abs(Float64(h_full[b + UNITREE_G1_OBS_DIM + G1_N_ACT + i])
                   - Float64(h_hist2[l * G1_HIST_DIM + i])) > 1e-9:
                bad += 1
    print("  packed 928: mismatched elements", bad, "of",
          LANES * UNITREE_G1_FULL_OBS_DIM)
    assert_true(bad == 0, "the packed observation is not [527 | 29 | 372]")

    # ── reset zeroes both buffers and skips the next push ────────────
    var h_mask = c.enqueue_create_host_buffer[DT](LANES)
    h_mask[0] = Scalar[DT](1.0)
    h_mask[1] = Scalar[DT](0.0)      # lane 1 NOT reset: the mask must be read
    c.enqueue_copy(d_mask, h_mask)
    c.enqueue_function[g1_hist_reset_kernel[LANES]](
        mptr(d_act.unsafe_ptr()), mptr(d_hist.unsafe_ptr()),
        mptr(d_live.unsafe_ptr()), mptr(d_mask.unsafe_ptr()),
        grid_dim=_blk(LANES * (G1_N_ACT + G1_HIST_DIM)), block_dim=TPB,
    )
    var h_h3 = c.enqueue_create_host_buffer[DT](LANES * G1_HIST_DIM)
    var h_l3 = c.enqueue_create_host_buffer[DT](LANES)
    c.enqueue_copy(h_h3, d_hist)
    c.enqueue_copy(h_l3, d_live)
    c.synchronize()
    var z0 = 0
    var nz1 = 0
    for i in range(G1_HIST_DIM):
        if Float64(h_h3[i]) == 0.0:
            z0 += 1
        if Float64(h_h3[G1_HIST_DIM + i]) != 0.0:
            nz1 += 1
    print("  reset: lane 0 zeroed", z0, "/", G1_HIST_DIM,
          "  lane 1 untouched", nz1, "/", G1_HIST_DIM,
          "  live =", Float64(h_l3[0]), Float64(h_l3[1]))
    assert_true(z0 == G1_HIST_DIM, "reset did not zero the history")
    assert_true(nz1 > 0, "reset zeroed a lane its mask did not select")
    assert_true(
        Float64(h_l3[0]) == 0.0 and Float64(h_l3[1]) == 1.0,
        "reset must clear `live` for the reset lane ONLY — that is what makes"
        " the reset observation never pushed",
    )

    # ── [6] DERIVE-FROM-RING == PUSH-INCREMENTALLY ───────────────────
    # The training side reads the 401 out of the replay ring instead of
    # storing it (L1, docs §12.36): 4 bytes per row instead of 1604. The only
    # check that means anything is that it reproduces the incremental path
    # BIT-FOR-BIT on the same rollout, including the age rule at the start of
    # an episode — which is where an off-by-one hides.
    print("[6] derive-from-ring == push-incrementally ...")
    comptime RCAP = LANES * 12
    var r_obs = c.enqueue_create_buffer[DT](RCAP * UNITREE_G1_OBS_DIM)
    var r_act = c.enqueue_create_buffer[DT](RCAP * G1_N_ACT)
    var r_age = c.enqueue_create_buffer[DT](RCAP)
    var h_robs = c.enqueue_create_host_buffer[DT](RCAP * UNITREE_G1_OBS_DIM)
    var h_ract = c.enqueue_create_host_buffer[DT](RCAP * G1_N_ACT)
    var h_rage = c.enqueue_create_host_buffer[DT](RCAP)

    # replay the SAME rollout, writing the ring exactly as `ring_store_kernel`
    # would (lane l of step s at row (s*LANES + l) % RCAP), and keeping the
    # incremental buffers in step beside it
    d_hist.enqueue_fill(Scalar[DT](0.0))
    d_act.enqueue_fill(Scalar[DT](0.0))
    d_live.enqueue_fill(Scalar[DT](0.0))
    var inc = c.enqueue_create_host_buffer[DT](LANES * G1_ACTOR_EXTRA)
    var want = List[Float64]()
    comptime NS2 = 10
    for s in range(NS2):
        for l in range(LANES):
            for i in range(UNITREE_G1_OBS_DIM):
                h_obs[l * UNITREE_G1_OBS_DIM + i] = Scalar[DT](_val(s, l, i))
            for i in range(G1_N_ACT):
                # the RAW action; the ring stores this and the derive scales it
                h_act[l * G1_N_ACT + i] = Scalar[DT](_val(s, l, i) * 0.1)
            var row = (s * LANES + l) % RCAP
            for i in range(UNITREE_G1_OBS_DIM):
                h_robs[row * UNITREE_G1_OBS_DIM + i] = h_obs[
                    l * UNITREE_G1_OBS_DIM + i
                ]
            for i in range(G1_N_ACT):
                h_ract[row * G1_N_ACT + i] = h_act[l * G1_N_ACT + i]
            h_rage[row] = Scalar[DT](Float64(s if s < 5 else 5))
        c.enqueue_copy(d_obs, h_obs)
        c.synchronize()
        c.enqueue_function[g1_hist_push_kernel[LANES]](
            mptr(d_obs.unsafe_ptr()), mptr(d_act.unsafe_ptr()),
            mptr(d_hist.unsafe_ptr()), mptr(d_live.unsafe_ptr()),
            grid_dim=_blk(LANES * G1_HIST_STEP), block_dim=TPB,
        )
        c.synchronize()
        # last_action is the SCALED CLIPPED action, as the PD chain consumed it
        for l in range(LANES):
            for i in range(G1_N_ACT):
                var a = Float64(h_act[l * G1_N_ACT + i]) * 5.0
                if a > 5.0:
                    a = 5.0
                if a < -5.0:
                    a = -5.0
                h_act[l * G1_N_ACT + i] = Scalar[DT](a)
        c.enqueue_copy(d_act, h_act)
        c.synchronize()
        d_live.enqueue_fill(Scalar[DT](1.0))
        # snapshot the incremental [last_action | history] AT THE NEXT step's
        # read point, which is what the derive for row (s+1) must reproduce
        var h_h = c.enqueue_create_host_buffer[DT](LANES * G1_HIST_DIM)
        c.enqueue_copy(h_h, d_hist)
        c.synchronize()
        if s + 1 < NS2:
            for l in range(LANES):
                for i in range(G1_N_ACT):
                    want.append(Float64(h_act[l * G1_N_ACT + i]))
                for i in range(G1_HIST_DIM):
                    want.append(Float64(h_h[l * G1_HIST_DIM + i]))

    c.enqueue_copy(r_obs, h_robs)
    c.enqueue_copy(r_act, h_ract)
    c.enqueue_copy(r_age, h_rage)
    c.synchronize()

    # derive for every (step, lane) from step 1 on, and compare
    comptime NDRAW = (NS2 - 1) * LANES
    var d_idx = c.enqueue_create_buffer[IDX_DT](NDRAW)
    var h_idx = c.enqueue_create_host_buffer[IDX_DT](NDRAW)
    var q = 0
    for s in range(1, NS2):
        for l in range(LANES):
            h_idx[q] = Scalar[IDX_DT]((s * LANES + l) % RCAP)
            q += 1
    c.enqueue_copy(d_idx, h_idx)
    var d_tail = c.enqueue_create_buffer[DT](NDRAW * G1_ACTOR_EXTRA)
    c.enqueue_function[g1_hist_gather_kernel[NDRAW, RCAP, LANES, G1_N_ACT]](
        mptr(r_obs.unsafe_ptr()), mptr(r_act.unsafe_ptr()),
        mptr(r_age.unsafe_ptr()), mptr(d_idx.unsafe_ptr()),
        Scalar[DT](5.0), Scalar[DT](5.0), mptr(d_tail.unsafe_ptr()),
        grid_dim=_blk(NDRAW * G1_ACTOR_EXTRA), block_dim=TPB,
    )
    var h_tail = c.enqueue_create_host_buffer[DT](NDRAW * G1_ACTOR_EXTRA)
    c.enqueue_copy(h_tail, d_tail)
    c.synchronize()
    var wrong = 0
    var nonzero = 0
    var worst2 = 0.0
    for i in range(NDRAW * G1_ACTOR_EXTRA):
        var g = Float64(h_tail[i])
        var w = want[i]
        if abs(g - w) > 1e-5:
            wrong += 1
        if w != 0.0:
            nonzero += 1
        if abs(g - w) > worst2:
            worst2 = abs(g - w)
    print("      compared", NDRAW * G1_ACTOR_EXTRA, "elements over",
          NS2 - 1, "steps x", LANES, "lanes   wrong", wrong,
          "  worst", worst2)
    assert_true(
        wrong == 0,
        "derive-from-ring disagrees with the incremental path at "
        + String(wrong) + " elements (worst " + String(worst2) + ") — the"
        " lane-aligned back-step or the age rule is off",
    )
    assert_true(
        nonzero > NDRAW * G1_ACTOR_EXTRA // 3,
        "vacuous: most of the expected tail is zero, so agreeing with it"
        " proves little — the fixture must run past the age ramp",
    )
    print("      OK")

    # ── [7] the GENERIC derive == the G1-specific one ────────────────
    # `derive_tail_kernel` fills a tail it cannot interpret, from a spec the
    # env builds (§12.36). That indirection is what keeps the G1's observation
    # layout out of `online.mojo`, which the walker path also runs — and it is
    # exactly the kind of table that can be built plausibly and wrongly, so it
    # is checked against the SAME incremental expectation, not against the
    # G1-specific kernel it replaces.
    print("[7] generic derive_tail_kernel from the G1 spec ...")
    var spec = List[Int32]()
    g1_build_tail_spec(spec)
    print("      spec entries", len(spec) // 4, "of", G1_ACTOR_EXTRA)
    assert_true(
        len(spec) == G1_ACTOR_EXTRA * 4,
        "the spec must carry 4 int32 per tail element, got "
        + String(len(spec)),
    )
    var d_spec = c.enqueue_create_buffer[DType.int32](G1_ACTOR_EXTRA * 4)
    var h_spec = c.enqueue_create_host_buffer[DType.int32](G1_ACTOR_EXTRA * 4)
    for i in range(G1_ACTOR_EXTRA * 4):
        h_spec[i] = spec[i]
    c.enqueue_copy(d_spec, h_spec)

    comptime OBSF2 = UNITREE_G1_OBS_DIM + G1_ACTOR_EXTRA
    var d_batch = c.enqueue_create_buffer[DT](NDRAW * OBSF2)
    d_batch.enqueue_fill(Scalar[DT](-7.0))      # poison: unwritten stays visible
    c.enqueue_function[
        derive_tail_kernel[
            NDRAW, RCAP, LANES, UNITREE_G1_OBS_DIM, G1_N_ACT,
            G1_ACTOR_EXTRA, OBSF2,
        ]
    ](
        mptr(r_obs.unsafe_ptr()), mptr(r_act.unsafe_ptr()),
        mptr(r_age.unsafe_ptr()), mptr(d_idx.unsafe_ptr()),
        d_spec.unsafe_ptr(),
        Scalar[DT](5.0), Scalar[DT](5.0),
        mptr(d_batch.unsafe_ptr()), Int32(0),
        grid_dim=_blk(NDRAW * G1_ACTOR_EXTRA), block_dim=TPB,
    )
    var h_batch = c.enqueue_create_host_buffer[DT](NDRAW * OBSF2)
    c.enqueue_copy(h_batch, d_batch)
    c.synchronize()
    var gwrong = 0
    var head_touched = 0
    for i in range(NDRAW):
        for k in range(G1_ACTOR_EXTRA):
            var g = Float64(h_batch[i * OBSF2 + UNITREE_G1_OBS_DIM + k])
            var w = want[i * G1_ACTOR_EXTRA + k]
            if abs(g - w) > 1e-5:
                gwrong += 1
        # the 527 HEAD is the caller's business and must be left alone
        for k in range(0, UNITREE_G1_OBS_DIM, 37):
            if Float64(h_batch[i * OBSF2 + k]) != -7.0:
                head_touched += 1
    print("      wrong tail elements", gwrong, "of", NDRAW * G1_ACTOR_EXTRA,
          "   head columns disturbed", head_touched)
    assert_true(
        gwrong == 0,
        "the generic derive disagrees with the incremental path at "
        + String(gwrong) + " elements — the spec table is wrong",
    )
    assert_true(
        head_touched == 0,
        "derive_tail_kernel wrote into the 527 head, which the plain row"
        " gather owns",
    )
    print("      OK")

    print("G1_HISTORY OK")

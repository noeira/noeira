"""Real-Time Chunking (`rtc.mojo`) on the real sampler.

  1. **The weights are LeRobot's.** `get_prefix_weights` builds its ramp from
     `linspace(1, 0, n + 2)[1:-1]`; the values below are that call's output,
     worked by hand, not a re-run of our formula. Plus the clip of `e` to
     the leftover.
  2. **The guidance weight is LeRobot's**, including the first step (time =
     1, +inf) clipping to beta.
  3. **Disarmed, the sampler is today's sampler — bit for bit.** RTC is an
     option on the deploy path; off, nothing may move.
  4. **Armed with all-zero weights, also bit for bit.** `v - g*0 = v`
     exactly, so any drift means the kernel touches what it must not.
  5. **The frozen prefix lands on the committed chunk.** Armed from a chunk
     whose actions from index `k` on are a known target, the first `d` steps
     of the new chunk must end far closer to that target than an unguided
     sample does — which also pins the `k` shift (an unshifted copy guides
     toward the wrong actions and fails the ratio).
  6. **The CPU `apply` and the GPU kernel agree** on the same inputs (two
     separately written paths).

Run:
  pixi run -e apple mojo run -I . tests/deep_agents/smolvla/test_rtc.mojo
"""

from std.math import abs
from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.linear import Linear
from noeira.deep_agents.smolvla.text import (
    SmolVLMTextLayers, SMOLLM_DIM, SMOLLM_FF, SMOLLM_KV_W, SMOLLM_LAYERS,
    SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM,
)
from noeira.deep_agents.smolvla.expert import SmolVLAExpert, EXPERT_FF
from noeira.deep_agents.smolvla.heads import SMOLVLA_EXPERT_W
from noeira.deep_agents.smolvla.kv_cache import SmolVLAKVCache
from noeira.deep_agents.smolvla.fused import SmolVLAPrefill, SmolVLADenoise
from noeira.deep_agents.smolvla.policy import SmolVLAActionSampler
from noeira.deep_agents.smolvla.attn_mask import (
    att_2d_mask, att_2d_mask_square, smolvla_ar,
)
from noeira.deep_agents.smolvla.rtc import (
    RtcGuide, RTC_ZEROS, RTC_ONES, RTC_LINEAR, RTC_EXP,
    rtc_prefix_weights, rtc_guidance_weight,
)

comptime P = 24
comptime CHUNK = 8
comptime ADIM = 32
comptime B = 1
comptime L = SMOLLM_LAYERS
comptime W = SMOLLM_DIM
comptime KVW = SMOLLM_KV_W
comptime EW = SMOLVLA_EXPERT_W
comptime STEPS = 10
comptime XN = B * CHUNK * ADIM

comptime Tower = SmolVLMTextLayers[L, W, SMOLLM_FF, KVW]
comptime Expert = SmolVLAExpert[L, EW, EXPERT_FF, W, KVW, 2]
comptime Cache = SmolVLAKVCache[L, P, CHUNK, SMOLLM_KV_HEADS, SMOLLM_HEAD_DIM, B]
comptime Pre = SmolVLAPrefill[P, CHUNK, B]
comptime Den = SmolVLADenoise[P, CHUNK, B]
comptime Sam = SmolVLAActionSampler[CHUNK, ADIM, EW, STEPS, B]


def _close(a: Float64, b: Float64, tol: Float64 = 1e-12) -> Bool:
    return abs(a - b) <= tol


def _check_weights() raises:
    # LINEAR, d = 3, e = 8, total = 10: start = 3, five ramp entries =
    # linspace(1, 0, 7)[1:-1] = 5/6 4/6 3/6 2/6 1/6, then zeros.
    var w = rtc_prefix_weights(3, 8, 10, RTC_LINEAR)
    var want: List[Float64] = [
        1.0, 1.0, 1.0, 5.0 / 6.0, 4.0 / 6.0, 3.0 / 6.0, 2.0 / 6.0,
        1.0 / 6.0, 0.0, 0.0,
    ]
    for i in range(10):
        assert_true(_close(w[i], want[i]), "LINEAR weight " + String(i))
    # EXP: c * expm1(c) / (e - 1) on the same ramp.
    var we = rtc_prefix_weights(3, 8, 10, RTC_EXP)
    var c = 4.0 / 6.0
    assert_true(
        _close(we[4], c * (1.9477340410546757 - 1.0) / 1.718281828459045, 1e-9),
        "EXP weight 4",
    )
    assert_true(we[2] == 1.0 and we[8] == 0.0, "EXP ends")
    # ZEROS depends on d only, ONES on e only.
    var wz = rtc_prefix_weights(3, 8, 10, RTC_ZEROS)
    var wo = rtc_prefix_weights(3, 8, 10, RTC_ONES)
    for i in range(10):
        assert_true(wz[i] == (1.0 if i < 3 else 0.0), "ZEROS " + String(i))
        assert_true(wo[i] == (1.0 if i < 8 else 0.0), "ONES " + String(i))
    # d >= e: no ramp, ones up to e.
    var wde = rtc_prefix_weights(6, 4, 10, RTC_LINEAR)
    for i in range(10):
        assert_true(wde[i] == (1.0 if i < 4 else 0.0), "d >= e " + String(i))
    print("  [1] prefix weights: LINEAR / EXP / ZEROS / ONES / d>=e as LeRobot")

    # Guidance weight: time = 1 -> beta; 0.5 -> 2; 0.9 -> 0.82/0.09.
    assert_true(rtc_guidance_weight(1.0, 10.0) == 10.0, "g(1) = beta")
    assert_true(_close(rtc_guidance_weight(0.5, 10.0), 2.0), "g(0.5) = 2")
    assert_true(
        _close(rtc_guidance_weight(0.9, 10.0), 0.82 / 0.09, 1e-12), "g(0.9)"
    )
    assert_true(rtc_guidance_weight(0.9, 5.0) == 5.0, "g(0.9) clips at 5")
    # The same through LeRobot's own expression, tau = 1 - time.
    var tau = 1.0 - 0.3
    var inv_r2 = ((1 - tau) ** 2 + tau**2) / ((1 - tau) ** 2)
    assert_true(
        _close(rtc_guidance_weight(0.3, 100.0), (1 - tau) / tau * inv_r2, 1e-12),
        "g(0.3) vs LeRobot's form",
    )
    print("  [2] guidance weight: beta at t=1, 2 at t=0.5, LeRobot's form at 0.3")


def _fill_noise(mut t: Tensor, seed: Int, d: DeviceContext) raises:
    t = Tensor.alloc(XN)
    for i in range(XN):
        t.data[i] = Scalar[DT]((((i + seed) * 37) % 19) - 9) * 0.1
    t.upload(d)


def main() raises:
    print("=" * 70)
    print("SmolVLA — Real-Time Chunking")
    print("=" * 70)
    _check_weights()

    var d = DeviceContext()
    var ar_pre = smolvla_ar(8, 15, 1, 0)
    var ar_full = smolvla_ar(8, 15, 1, CHUNK)
    var mask_pre = att_2d_mask_square(ar_pre)
    var mask_self = att_2d_mask(ar_full, P, P + CHUNK, 0, P + CHUNK)
    var mask_cross = att_2d_mask(ar_full, P, P + CHUNK, 0, P)

    var tower = Tower.make["gpu", Deterministic](Optional(d))
    var expert = Expert.make["gpu", Deterministic](Optional(d))
    var cache = Cache.make["gpu"](Optional(d))
    var pre = Pre.make["gpu"](mask_pre, Optional(d))
    var den = Den.make["gpu"](mask_self, mask_cross, Optional(d))
    var sam = Sam.make["gpu"](Optional(d))
    var action_in = Linear[ADIM, EW].make["gpu", Deterministic](Optional(d))
    var tmlp_in = Linear[2 * EW, EW].make["gpu", Deterministic](Optional(d))
    var tmlp_out = Linear[EW, EW].make["gpu", Deterministic](Optional(d))
    var action_out = Linear[EW, ADIM].make["gpu", Deterministic](Optional(d))

    var xp = Tensor.alloc(B * P * W)
    for i in range(B * P * W):
        xp.data[i] = Scalar[DT](((i * 29) % 17) - 8) * 0.05
    xp.upload(d)
    var pout = Tensor.alloc(B * P * W)
    pre.run["gpu"](tower, cache, xp, pout, Optional(d))

    # ── baseline: never armed ────────────────────────────────────────────
    var noise = Tensor()
    _fill_noise(noise, 0, d)
    var base = Tensor.alloc(XN)
    sam.sample["gpu", P](
        expert, cache, den, action_in, tmlp_in, tmlp_out, action_out,
        noise, base, Optional(d),
    )
    base.download(d)

    # ── 3. disarmed after an arm: bit for bit ────────────────────────────
    var target = Tensor.alloc(XN)
    for i in range(XN):
        target.data[i] = Scalar[DT](((i * 13) % 11) - 5) * 0.3
    sam.rtc.arm["gpu"](target, 0, 3, CHUNK, RTC_ONES, 10.0, Optional(d))
    sam.rtc.disarm()
    _fill_noise(noise, 0, d)
    var off = Tensor.alloc(XN)
    sam.sample["gpu", P](
        expert, cache, den, action_in, tmlp_in, tmlp_out, action_out,
        noise, off, Optional(d),
    )
    off.download(d)
    var diff = 0
    for i in range(XN):
        if off.data[i] != base.data[i]:
            diff += 1
    print("  [3] disarmed vs never armed: differing", diff, "/", XN)
    assert_equal(diff, 0, "a disarmed RTC changed the chunk")

    # ── 4. armed, zero weights (ZEROS with d = 0): bit for bit ───────────
    sam.rtc.arm["gpu"](target, 0, 0, CHUNK, RTC_ZEROS, 10.0, Optional(d))
    _fill_noise(noise, 0, d)
    var zero = Tensor.alloc(XN)
    sam.sample["gpu", P](
        expert, cache, den, action_in, tmlp_in, tmlp_out, action_out,
        noise, zero, Optional(d),
    )
    zero.download(d)
    diff = 0
    for i in range(XN):
        if zero.data[i] != base.data[i]:
            diff += 1
    print("  [4] armed with zero weights: differing", diff, "/", XN)
    assert_equal(diff, 0, "zero weights changed the chunk")

    # ── 5. the frozen prefix lands on the committed chunk, shifted by k ──
    # `last[k + j] = goal[j]`: an unshifted copy would guide toward
    # last[j] = something else.
    comptime K = 3
    comptime DFRZ = 3
    var goal = Tensor.alloc(XN)
    var last = Tensor.alloc(XN)
    for j in range(CHUNK):
        for a in range(ADIM):
            var i = j * ADIM + a
            goal.data[i] = Scalar[DT]((((i * 7) % 13) - 6)) * 0.25
    for j in range(CHUNK):
        for a in range(ADIM):
            var src = j - K
            last.data[j * ADIM + a] = (
                goal.data[src * ADIM + a] if src >= 0
                else Scalar[DT](-4.0)
            )
    sam.rtc.arm["gpu"](last, K, DFRZ, CHUNK, RTC_LINEAR, 10.0, Optional(d))
    # The leftover is CHUNK - K = 5 long, so e clips to 5: weights 1,1,1 on
    # the frozen steps, the ramp (5-i)/3 at i = 3, 4, zeros from 5 on.
    assert_equal(sam.rtc.e, CHUNK - K, "e must clip to the leftover")
    assert_true(
        Float64(sam.rtc.w.data[4]) > 0.3 and sam.rtc.w.data[5] == 0,
        "weights after the clip",
    )
    _fill_noise(noise, 0, d)
    var guided = Tensor.alloc(XN)
    sam.sample["gpu", P](
        expert, cache, den, action_in, tmlp_in, tmlp_out, action_out,
        noise, guided, Optional(d),
    )
    guided.download(d)
    sam.rtc.disarm()
    var e_g = 0.0
    var e_u = 0.0
    var e_alt = 0.0
    for j in range(DFRZ):
        for a in range(ADIM):
            var i = j * ADIM + a
            e_g += abs(Float64(guided.data[i] - goal.data[i]))
            e_u += abs(Float64(base.data[i] - goal.data[i]))
            # the UNSHIFTED leftover, last[j]: where a missing k lands
            e_alt += abs(Float64(guided.data[i] - last.data[i]))
    var nan = 0
    for i in range(XN):
        if guided.data[i] != guided.data[i]:
            nan += 1
    print(
        "  [5] frozen steps |chunk - committed|: guided", e_g / (DFRZ * ADIM),
        " unguided", e_u / (DFRZ * ADIM), " ratio", e_g / e_u, " nan", nan,
    )
    assert_true(nan == 0, "RTC produced NaN")
    print(
        "      vs the unshifted leftover last[0:d]:", e_alt / (DFRZ * ADIM),
        " (goal / unshifted", e_g / e_alt, ")",
    )
    assert_true(
        e_g < 0.15 * e_u,
        "the frozen prefix did not land on the committed chunk",
    )
    # ⚠ The ratio above alone is VACUOUS for the shift: the unguided chunk
    # is ~40 away, so guidance toward ANY nearby target passes it (a mutant
    # dropping k did, 0.09). The shift is pinned by landing on last[k:]
    # rather than on last[0:].
    assert_true(
        e_g < 0.25 * e_alt,
        "the prefix landed nearer the UNSHIFTED leftover — the k shift is"
        " wrong",
    )

    # ── 6. CPU apply vs the GPU kernel ───────────────────────────────────
    var gcpu = RtcGuide[CHUNK, ADIM, B]()
    var ggpu = RtcGuide[CHUNK, ADIM, B]()
    gcpu.arm["cpu"](last, K, DFRZ, CHUNK, RTC_EXP, 7.0)
    ggpu.arm["gpu"](last, K, DFRZ, CHUNK, RTC_EXP, 7.0, Optional(d))
    var xc = Tensor.alloc(XN)
    var vc = Tensor.alloc(XN)
    for i in range(XN):
        xc.data[i] = Scalar[DT](((i * 17) % 23) - 11) * 0.07
        vc.data[i] = Scalar[DT](((i * 5) % 29) - 14) * 0.05
    var xg = Tensor.alloc(XN)
    var vg = Tensor.alloc(XN)
    for i in range(XN):
        xg.data[i] = xc.data[i]
        vg.data[i] = vc.data[i]
    xg.upload(d)
    vg.upload(d)
    gcpu.apply["cpu"](xc, vc, 0.7)
    ggpu.apply["gpu"](xg, vg, 0.7, Optional(d))
    vg.download(d)
    var worst = 0.0
    for i in range(XN):
        worst = max(worst, abs(Float64(vg.data[i] - vc.data[i])))
    print("  [6] CPU apply vs GPU kernel: max |diff|", worst)
    assert_true(worst < 1e-5, "CPU and GPU RTC disagree")

    print()
    print("PASSED — RTC: weights, guidance, off = identity, prefix lands")

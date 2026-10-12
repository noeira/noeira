# +--------------------------------------------------------------------------+ #
# | The grouped arena covers the WHOLE trainable set, and the clip is global
# +--------------------------------------------------------------------------+ #
"""`adopt_multi` and `clip_trainables`, against the two ways they fail quietly.

    pixi run -e apple mojo run -I . \\
        tests/deep_agents/smolvla/test_finetune_arena.mojo

`Adam.adopt` says "call ONCE" and means it — it resets `total` and `_off` and
reallocates. Calling it per component therefore leaves only the LAST one in
the arena **while `adopted` reads True**, so the grouped step updates one
fifth of the model and every other component silently stops training. That is
leg [1], and it is a count: the arena's total must equal the five walks laid
end to end, each param starting on a `PARAM_ALIGN` boundary.

`configuration_smolvla.py` sets `optimizer_grad_clip_norm = 10`. Clipping each
component to 10 INDEPENDENTLY is not clipping their joint norm to 10 — five
components each at the limit have a joint norm of sqrt(5)x it — so a
per-component clip is a different algorithm that happens to have the same
parameter. Leg [3] measures the joint norm before and after and leg [4] shows
the per-component answer differs.

⚠ Leg [2] is the equivalence that makes the arena safe to turn on: adopted and
un-adopted must take the SAME optimizer steps. A faster path that trains a
slightly different model is not an optimisation.
"""

from std.math import abs, sqrt
from std.testing import assert_true, assert_equal
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.initializer import Deterministic
from noeira.nn.core.named_params import named_params
from noeira.nn.optimizer.adam import Adam
from noeira.nn.optimizer.param_arena import align_param_off
from noeira.nn.primitives.linear import Linear
from noeira.deep_agents.smolvla.text import SMOLLM_THETA
from noeira.deep_agents.smolvla.expert import SmolVLAExpert
from noeira.deep_agents.smolvla.kv_cache import SmolVLAKVCache
from noeira.deep_agents.smolvla.fused import SmolVLADenoise
from noeira.deep_agents.smolvla.train_step import SmolVLATrainStep
from noeira.deep_agents.smolvla.finetune import (
    zero_trainable_grads, adam_step_trainables, adopt_trainables,
    clip_trainables,
)
from noeira.deep_agents.smolvla.flow_loss import build_xt_ut
from noeira.deep_agents.smolvla.attn_mask import att_2d_mask, smolvla_ar

comptime P = 6
comptime CHUNK = 3
comptime B = 1
comptime L = 2
comptime EW = 8
comptime EFF = 12
comptime W = 8
comptime HEADS = 2
comptime NKV = 1
comptime HD = 4
comptime KVW = NKV * HD
comptime ADIM = 6
comptime ADIM_REAL = 3
comptime XN = B * CHUNK * ADIM
comptime PKV = B * P * KVW
comptime STEPS = 12
comptime LR = Scalar[DT](3.0e-3)
comptime CLIP = Scalar[DT](0.05)
"""⚠ Small ON PURPOSE. The reference's 10 would never bind on this fixture,
and a clip that never fires is a clip that is not tested."""

comptime Expert = SmolVLAExpert[L, EW, EFF, W, KVW, 2]
comptime Cache = SmolVLAKVCache[L, P, CHUNK, NKV, HD, B]
comptime Den = SmolVLADenoise[
    P, CHUNK, B, L, EW, EFF, W, HEADS, NKV, HD, SMOLLM_THETA, 2, KVW, True
]
comptime Step = SmolVLATrainStep[
    CHUNK, ADIM_REAL, ADIM, EW, B, L, EFF, W, HEADS, NKV, HD, SMOLLM_THETA,
    KVW,
]
comptime AIn = Linear[ADIM, EW]
comptime TIn = Linear[2 * EW, EW]
comptime TOut = Linear[EW, EW]
comptime AOut = Linear[EW, ADIM]
comptime SProj = Linear[32, 960]


def _count(
    mut e: Expert, mut ai: AIn, mut ti: TIn, mut to: TOut, mut ao: AOut,
    d: DeviceContext,
) raises -> Int:
    """Elements across all five walks — what the arena must equal.

    ⚠ The arena starts every param on a `PARAM_ALIGN` boundary, so this is one
    running offset over the five walks in adoption order, each param's start
    rounded up by `align_param_off` — not the bare sum of the sizes, which
    the arena exceeds by its padding."""
    var n = 0
    var a = named_params["gpu"](e, Optional(d))
    for i in range(len(a)):
        n = align_param_off(n) + a[i].size
    var b = named_params["gpu"](ai, Optional(d))
    for i in range(len(b)):
        n = align_param_off(n) + b[i].size
    var c = named_params["gpu"](ti, Optional(d))
    for i in range(len(c)):
        n = align_param_off(n) + c[i].size
    var f = named_params["gpu"](to, Optional(d))
    for i in range(len(f)):
        n = align_param_off(n) + f[i].size
    var g = named_params["gpu"](ao, Optional(d))
    for i in range(len(g)):
        n = align_param_off(n) + g[i].size
    return n


def main() raises:
    print("=" * 70)
    print("SmolVLA trainable-set arena + global grad clip")
    print("=" * 70)

    var d = DeviceContext()
    var ar = smolvla_ar(3, 2, 1, CHUNK)
    var ms = att_2d_mask(ar, P, P + CHUNK, 0, P + CHUNK)
    var mc = att_2d_mask(ar, P, P + CHUNK, 0, P)

    var kp = Tensor.alloc(PKV)
    var vp = Tensor.alloc(PKV)
    var noise = Tensor.alloc(XN)
    var acts = Tensor.alloc(XN)
    for i in range(XN):
        noise.data[i] = Scalar[DT](((i * 37) % 19) - 9) * 0.07
        acts.data[i] = Scalar[DT](0)
    for t in range(CHUNK):
        for dd in range(ADIM_REAL):
            acts.data[t * ADIM + dd] = Scalar[DT](
                ((t * 5 + dd * 3) % 7) - 3
            ) * 0.2
    var times_t = Tensor.alloc(B)
    var tl = List[Float64]()
    for b in range(B):
        times_t.data[b] = Scalar[DT](0.37)
        tl.append(0.37)
    var x_h = Tensor.alloc(XN)
    var u_h = Tensor.alloc(XN)
    build_xt_ut["cpu", B, CHUNK * ADIM](noise, acts, times_t, x_h, u_h, None)
    var valid_h = Tensor.alloc(B * CHUNK)
    for i in range(B * CHUNK):
        valid_h.data[i] = Scalar[DT](1.0)
    comptime N_VALID = B * CHUNK

    # ── the ADOPTED run ──────────────────────────────────────────────────
    var e = Expert.make["gpu", Deterministic](Optional(d))
    var c = Cache.make["gpu"](Optional(d))
    var den = Den.make["gpu"](ms, mc, Optional(d))
    var st = Step.make["gpu"](Optional(d))
    var ai = AIn.make["gpu", Deterministic](Optional(d))
    var ti = TIn.make["gpu", Deterministic](Optional(d))
    var to = TOut.make["gpu", Deterministic](Optional(d))
    var ao = AOut.make["gpu", Deterministic](Optional(d))
    var sp = SProj.make["gpu", Deterministic](Optional(d))
    var want_total = _count(e, ai, ti, to, ao, d)

    var opt = Adam(lr=LR)
    adopt_trainables["gpu", L, EW, EFF, W, KVW, ADIM](
        opt, e, ai, ti, to, ao, sp, Optional(d)
    )
    print("  [1] arena holds", opt.arena.total, "elements; the five walks"
          " count", want_total)
    assert_true(want_total > 0, "no parameters found — leg [1] is vacuous")
    assert_equal(
        opt.arena.total, want_total,
        "the arena does not cover the whole trainable set — `adopt` called"
        " per component leaves only the LAST one adopted, and the grouped"
        " step then trains one fifth of the model in silence",
    )

    for l in range(L):
        for i in range(PKV):
            kp.data[i] = Scalar[DT](((i * 31 + l * 7) % 13) - 6) * 0.11
            vp.data[i] = Scalar[DT](((i * 17 + l * 5) % 11) - 5) * 0.09
        kp.upload(d)
        vp.upload(d)
        c.write_prefix["gpu"](l, kp, vp, Optional(d))
    var x_t = Tensor.alloc(XN)
    var u_t = Tensor.alloc(XN)
    for i in range(XN):
        x_t.data[i] = x_h.data[i]
        u_t.data[i] = u_h.data[i]
    x_t.upload(d)
    u_t.upload(d)
    var valid = Tensor.alloc(B * CHUNK)
    for i in range(B * CHUNK):
        valid.data[i] = valid_h.data[i]
    valid.upload(d)
    st.set_times["gpu"](tl, Optional(d))

    var loss_a = 0.0
    for _ in range(STEPS):
        zero_trainable_grads["gpu", L, EW, EFF, W, KVW, ADIM](
            opt, e, ai, ti, to, ao, sp, Optional(d)
        )
        loss_a = st.run["gpu", P](
            e, c, den, ai, ti, to, ao, x_t, u_t, valid, N_VALID, Optional(d)
        )
        adam_step_trainables["gpu", L, EW, EFF, W, KVW, ADIM](
            opt, e, ai, ti, to, ao, sp, Optional(d)
        )
    d.synchronize()

    # ── the UN-ADOPTED run, same everything ──────────────────────────────
    var e2 = Expert.make["gpu", Deterministic](Optional(d))
    var c2 = Cache.make["gpu"](Optional(d))
    var den2 = Den.make["gpu"](ms, mc, Optional(d))
    var st2 = Step.make["gpu"](Optional(d))
    var ai2 = AIn.make["gpu", Deterministic](Optional(d))
    var ti2 = TIn.make["gpu", Deterministic](Optional(d))
    var to2 = TOut.make["gpu", Deterministic](Optional(d))
    var ao2 = AOut.make["gpu", Deterministic](Optional(d))
    var sp2 = SProj.make["gpu", Deterministic](Optional(d))
    for l in range(L):
        for i in range(PKV):
            kp.data[i] = Scalar[DT](((i * 31 + l * 7) % 13) - 6) * 0.11
            vp.data[i] = Scalar[DT](((i * 17 + l * 5) % 11) - 5) * 0.09
        kp.upload(d)
        vp.upload(d)
        c2.write_prefix["gpu"](l, kp, vp, Optional(d))
    st2.set_times["gpu"](tl, Optional(d))
    var opt2 = Adam(lr=LR)          # NOT adopted
    var loss_b = 0.0
    for _ in range(STEPS):
        zero_trainable_grads["gpu", L, EW, EFF, W, KVW, ADIM](
            opt2, e2, ai2, ti2, to2, ao2, sp2, Optional(d)
        )
        loss_b = st2.run["gpu", P](
            e2, c2, den2, ai2, ti2, to2, ao2, x_t, u_t, valid, N_VALID,
            Optional(d)
        )
        adam_step_trainables["gpu", L, EW, EFF, W, KVW, ADIM](
            opt2, e2, ai2, ti2, to2, ao2, sp2, Optional(d)
        )
    d.synchronize()

    print("  [2] after", STEPS, "steps: adopted", loss_a, " un-adopted",
          loss_b, " rel", abs(loss_a - loss_b) / abs(loss_b))
    assert_true(
        loss_b < 0.9 * loss_a + 0.9 * loss_b,
        "neither run learned, so leg [2] compares two frozen models",
    )
    assert_true(
        abs(loss_a - loss_b) / abs(loss_b) < 1.0e-4,
        "the grouped arena takes DIFFERENT steps from the per-parameter walk"
        " — a faster path that trains another model is not an optimisation",
    )

    # ── [3] the clip is GLOBAL ───────────────────────────────────────────
    zero_trainable_grads["gpu", L, EW, EFF, W, KVW, ADIM](
        opt, e, ai, ti, to, ao, sp, Optional(d)
    )
    _ = st.run["gpu", P](
        e, c, den, ai, ti, to, ao, x_t, u_t, valid, N_VALID, Optional(d)
    )
    d.synchronize()
    opt.arena.grd.download(d)
    var pre = 0.0
    for i in range(opt.arena.total):
        pre += Float64(opt.arena.grd.data[i]) * Float64(
            opt.arena.grd.data[i]
        )
    pre = sqrt(pre)

    var reported = clip_trainables["gpu"](opt, CLIP, Optional(d))
    d.synchronize()
    opt.arena.grd.download(d)
    var post = 0.0
    for i in range(opt.arena.total):
        post += Float64(opt.arena.grd.data[i]) * Float64(
            opt.arena.grd.data[i]
        )
    post = sqrt(post)
    print("  [3] joint grad norm", pre, "-> ", post, "  (limit", CLIP,
          ", reported pre-clip", reported, ")")
    assert_true(
        pre > Float64(CLIP),
        "the gradient norm is already under the limit, so the clip never"
        " fired and this leg proves nothing",
    )
    assert_true(
        abs(post - Float64(CLIP)) / Float64(CLIP) < 1.0e-3,
        "the clipped JOINT norm is not the limit",
    )
    assert_true(
        abs(Float64(reported) - pre) / pre < 1.0e-3,
        "the reported pre-clip norm is not the joint norm",
    )

    print()
    print("PASSED — one arena over all " + String(want_total) + " elements,"
          " same steps as the per-parameter walk, and a global clip")

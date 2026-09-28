# +--------------------------------------------------------------------------+ #
# | The fine-tuned weights survive a save and a load
# +--------------------------------------------------------------------------+ #
"""Round-trip the trainable set, and prove the load actually did something.

    pixi run mojo run -I . \\
        tests/deep_agents/smolvla/test_finetune_checkpoint.mojo

A 73-minute fine-tune that keeps nothing is a measurement, not a model. This
gates the path that keeps it.

⚠ **Only the TRAINABLE set is saved** — the SigLIP tower, the sixteen VLM
layers, the connector and the token embedding are frozen and already on disk
as `lerobot/smolvla_base`. Saving them again would triple the file and, worse,
create a second copy that could silently disagree with the base it was
fine-tuned from.

## What a checkpoint gate has to prove, and what it is tempting to prove

The tempting version is "save, load, no error". That passes on a file the
loader ignored. So:

  [1] the weights come back **bit-identical**, every one of them;
  [2] a model loaded from the file produces the **same loss** as the model
      that wrote it — which is the property anyone actually wants;
  [3] ⚠ the destination was **DIFFERENT before the load**. Without this,
      legs [1] and [2] pass on a load that did nothing at all, because the
      fixture's two models started life identical — `Deterministic` gives the
      same weights every time. The test TRAINS one of them first so there is
      something for the file to carry.

⚠ Leg [4] checks the moments too. `save_moments=True` is what makes a resume
exact rather than a restart with a cold optimizer, and a cold optimizer is
precisely what damages a pretrained model on its first step — the runner's
whole warmup story. A checkpoint that silently dropped them would resume into
that. Leg [5] checks Adam's step count and bias-correction powers, which the
moments are meaningless without: restored under `t = 0`, settled moments are
divided by the step-1 corrections.
"""

from std.math import abs
from std.os import remove
from std.pathlib import Path
from std.testing import assert_true, assert_equal

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.initializer import Deterministic
from noeira.nn.optimizer.adam import Adam
from noeira.nn.core.checkpoint import CheckpointScalars
from noeira.nn.primitives.linear import Linear
from noeira.deep_agents.smolvla.text import SMOLLM_THETA
from noeira.deep_agents.smolvla.expert import SmolVLAExpert
from noeira.deep_agents.smolvla.kv_cache import SmolVLAKVCache
from noeira.deep_agents.smolvla.fused import SmolVLADenoise
from noeira.deep_agents.smolvla.train_step import SmolVLATrainStep
from noeira.deep_agents.smolvla.finetune import (
    zero_trainable_grads, adam_step_trainables, save_trainables,
    load_trainables,
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
comptime TRAIN_STEPS = 25
comptime LR = Scalar[DT](3.0e-3)

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
comptime CKPT = String("/tmp/smolvla_ckpt_fixture.ckpt")

comptime N_GROUPS = 6


def _gsize(g: Int) -> Int:
    if g == 0: return EW * W
    if g == 1: return EW * EFF
    if g == 2: return EW
    if g == 3: return KVW * KVW
    if g == 4: return ADIM * EW
    return EW * ADIM


def _gval(
    g: Int, t: Int, mut e: Expert, mut ai: AIn, mut ao: AOut
) raises -> Scalar[DT]:
    if g == 0: return e.self_layers[0].q.weight.val.data[t]
    if g == 1: return e.self_layers[0].mlp.gate.weight.val.data[t]
    if g == 2: return e.self_layers[0].input_layernorm.gamma.val.data[t]
    if g == 3: return e.cross_layers[0].k.weight.val.data[t]
    if g == 4: return ai.weight.val.data[t]
    return ao.weight.val.data[t]


def _gm(g: Int, t: Int, mut e: Expert, mut ai: AIn) raises -> Scalar[DT]:
    """One of Adam's first moments, for leg [4]."""
    if g == 0: return e.self_layers[0].q.weight.m.data[t]
    return ai.weight.m.data[t]


def main() raises:
    print("=" * 70)
    print("SmolVLA fine-tune checkpoint round-trip")
    print("=" * 70)

    var ar = smolvla_ar(3, 2, 1, CHUNK)
    var ms = att_2d_mask(ar, P, P + CHUNK, 0, P + CHUNK)
    var mc = att_2d_mask(ar, P, P + CHUNK, 0, P)

    var e = Expert.make["cpu", Deterministic]()
    var c = Cache.make["cpu"]()
    var den = Den.make["cpu"](ms, mc, None)
    var st = Step.make["cpu"](None)
    var ai = AIn.make["cpu", Deterministic]()
    var ti = TIn.make["cpu", Deterministic]()
    var to = TOut.make["cpu", Deterministic]()
    var ao = AOut.make["cpu", Deterministic]()
    var sp = SProj.make["cpu", Deterministic]()

    var kp = Tensor.alloc(PKV)
    var vp = Tensor.alloc(PKV)
    for l in range(L):
        for i in range(PKV):
            kp.data[i] = Scalar[DT](((i * 31 + l * 7) % 13) - 6) * 0.11
            vp.data[i] = Scalar[DT](((i * 17 + l * 5) % 11) - 5) * 0.09
        c.write_prefix["cpu"](l, kp, vp)

    var noise = Tensor.alloc(XN)
    var acts = Tensor.alloc(XN)
    for i in range(XN):
        noise.data[i] = Scalar[DT](((i * 37) % 19) - 9) * 0.07
        acts.data[i] = Scalar[DT](0)
    for t in range(CHUNK):
        for d in range(ADIM_REAL):
            acts.data[t * ADIM + d] = Scalar[DT](((t * 5 + d * 3) % 7) - 3) * 0.2
    var times_t = Tensor.alloc(B)
    var tl = List[Float64]()
    for b in range(B):
        times_t.data[b] = Scalar[DT](0.37)
        tl.append(0.37)
    var x_t = Tensor.alloc(XN)
    var u_t = Tensor.alloc(XN)
    build_xt_ut["cpu", B, CHUNK * ADIM](noise, acts, times_t, x_t, u_t, None)
    var valid = Tensor.alloc(B * CHUNK)
    for i in range(B * CHUNK):
        valid.data[i] = Scalar[DT](1.0)
    comptime N_VALID = B * CHUNK
    st.set_times["cpu"](tl, None)

    # ── train it, so the file has something to carry ─────────────────────
    var opt = Adam(lr=LR)
    var trained_loss = 0.0
    for _ in range(TRAIN_STEPS):
        zero_trainable_grads["cpu", L, EW, EFF, W, KVW, ADIM](
            opt, e, ai, ti, to, ao, sp, None
        )
        trained_loss = st.run["cpu", P](
            e, c, den, ai, ti, to, ao, x_t, u_t, valid, N_VALID, None
        )
        adam_step_trainables["cpu", L, EW, EFF, W, KVW, ADIM](
            opt, e, ai, ti, to, ao, sp, None
        )
    # ⚠ ONE more forward, because `trained_loss` above is the loss BEFORE the
    # last `adam_step` and the file about to be written holds the weights
    # AFTER it. Comparing those two reads as a broken checkpoint — it was the
    # first thing this gate reported, at a difference of 2.8e-03 — and the
    # checkpoint was fine. The loss to compare against is the one at the
    # weights actually saved.
    zero_trainable_grads["cpu", L, EW, EFF, W, KVW, ADIM](
        opt, e, ai, ti, to, ao, sp, None
    )
    trained_loss = st.run["cpu", P](
        e, c, den, ai, ti, to, ao, x_t, u_t, valid, N_VALID, None
    )
    print("  [1] trained", TRAIN_STEPS, "steps, loss at the SAVED weights",
          trained_loss)

    if Path(CKPT).exists():
        remove(CKPT)
    var opt_sc = CheckpointScalars()
    opt.put_step_state(opt_sc, "opt")
    save_trainables["cpu", L, EW, EFF, W, KVW, ADIM](
        CKPT, e, ai, ti, to, ao, sp, True, None, scalars=opt_sc
    )
    assert_true(Path(CKPT).exists(), "no checkpoint file was written")

    var saved = List[Scalar[DT]]()
    for g in range(N_GROUPS):
        for t in range(_gsize(g)):
            saved.append(_gval(g, t, e, ai, ao))
    var saved_m = List[Scalar[DT]]()
    for g in range(2):
        for t in range(_gsize(g) if g == 0 else ADIM * EW):
            saved_m.append(_gm(g, t, e, ai))

    # ── a FRESH model, at the initialiser ────────────────────────────────
    var e2 = Expert.make["cpu", Deterministic]()
    var ai2 = AIn.make["cpu", Deterministic]()
    var ti2 = TIn.make["cpu", Deterministic]()
    var to2 = TOut.make["cpu", Deterministic]()
    var ao2 = AOut.make["cpu", Deterministic]()
    var sp2 = SProj.make["cpu", Deterministic]()

    # ⚠ LEG [3] FIRST: the destination must DIFFER before the load, or
    # everything below passes on a load that did nothing. `Deterministic`
    # gives both models the same weights, which is exactly why the first one
    # was trained.
    var pre_diff = 0
    var k = 0
    for g in range(N_GROUPS):
        for t in range(_gsize(g)):
            if _gval(g, t, e2, ai2, ao2) != saved[k + t]:
                pre_diff += 1
        k += _gsize(g)
    print("  [3] before the load, the fresh model differs in", pre_diff,
          "of", len(saved), "weights")
    assert_true(
        pre_diff > len(saved) // 2,
        "the fresh model already matches the saved one, so a load that did"
        " NOTHING would pass legs [1] and [2]",
    )

    var back_sc = load_trainables["cpu", L, EW, EFF, W, KVW, ADIM](
        CKPT, e2, ai2, ti2, to2, ao2, sp2, None
    )

    # ── [1] bit-identical weights ────────────────────────────────────────
    var wdiff = 0
    k = 0
    for g in range(N_GROUPS):
        for t in range(_gsize(g)):
            if _gval(g, t, e2, ai2, ao2) != saved[k + t]:
                wdiff += 1
        k += _gsize(g)
    print("  [1] after the load: compared", len(saved), " differing", wdiff)
    assert_true(wdiff == 0, "the reloaded weights are not the saved ones")

    # ── [2] and it computes the same loss ────────────────────────────────
    var c2 = Cache.make["cpu"]()
    for l in range(L):
        for i in range(PKV):
            kp.data[i] = Scalar[DT](((i * 31 + l * 7) % 13) - 6) * 0.11
            vp.data[i] = Scalar[DT](((i * 17 + l * 5) % 11) - 5) * 0.09
        c2.write_prefix["cpu"](l, kp, vp)
    var den2 = Den.make["cpu"](ms, mc, None)
    var st2 = Step.make["cpu"](None)
    st2.set_times["cpu"](tl, None)
    zero_trainable_grads["cpu", L, EW, EFF, W, KVW, ADIM](
        opt, e2, ai2, ti2, to2, ao2, sp2, None
    )
    var reloaded_loss = st2.run["cpu", P](
        e2, c2, den2, ai2, ti2, to2, ao2, x_t, u_t, valid, N_VALID, None
    )
    print("  [2] loss from the reloaded model", reloaded_loss, " vs",
          trained_loss, " diff", abs(reloaded_loss - trained_loss))
    assert_true(
        abs(reloaded_loss - trained_loss) < 1.0e-12,
        "the reloaded model does not compute what the saved one did",
    )

    # ── [4] Adam's moments came back too ─────────────────────────────────
    var mdiff = 0
    var mn = 0
    var mnz = 0
    k = 0
    for g in range(2):
        var n = _gsize(g) if g == 0 else ADIM * EW
        for t in range(n):
            mn += 1
            if _gm(g, t, e2, ai2) != saved_m[k + t]:
                mdiff += 1
            if saved_m[k + t] != Scalar[DT](0):
                mnz += 1
        k += n
    print("  [4] Adam first moments: compared", mn, " differing", mdiff,
          " (nonzero in the source:", mnz, ")")
    assert_true(
        mnz > 0,
        "every saved moment is zero, so leg [4] cannot tell a restored"
        " optimizer from a cold one",
    )
    assert_true(
        mdiff == 0,
        "the optimizer moments did not survive — a resume would restart with"
        " a COLD optimizer, whose first step is what damages a pretrained"
        " model",
    )

    # ── [5] and so did the step count the moments were accumulated under ──
    var opt2 = Adam(lr=LR)
    opt2.take_step_state(back_sc, "opt")
    print("  [5] Adam step: saved", opt.t, " restored", opt2.t)
    assert_true(opt.t > 0, "the source optimizer never stepped")
    assert_true(
        opt2.t == opt.t and opt2.bc1 == opt.bc1 and opt2.bc2 == opt.bc2,
        "Adam's step state did not survive — the restored moments would be"
        " divided by the step-1 bias corrections",
    )

    remove(CKPT)
    print()
    print("PASSED — weights, loss, moments and Adam's step all survive")

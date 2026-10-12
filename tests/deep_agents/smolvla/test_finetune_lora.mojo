# +--------------------------------------------------------------------------+ #
# | SmolVLA fine-tune with LoRA: the frozen base stays frozen, the rest trains
# +--------------------------------------------------------------------------+ #
"""The two LoRA regimes of `examples/so101/smolvla_so101_finetune.mojo`, at
toy size on the GPU, through the same helpers the program calls.

    pixi run -e apple  mojo run -I . tests/deep_agents/smolvla/test_finetune_lora.mojo
    pixi run -e nvidia mojo run -I . tests/deep_agents/smolvla/test_finetune_lora.mojo

  [1] LoRA on the EXPERT (`-D SMOLVLA_EXPERT_R`): the expert's base frozen by
      its base walk, `adopt_trainables[E_R]` adopts exactly the adapters and
      the four projections, and after 12 steps (arena, global clip) every
      base value is bit-unchanged, every adapter's A and B and every
      projection moved, the loss fell, and the optimizer's version bump left
      the frozen weights' versions alone while it advanced the projections'
  [2] LoRA on the TEXT TOWER (`-D SMOLVLA_TOWER_R`): the expert trains in
      full, the tower's base is frozen by `freeze_backbone`'s base walk, and
      its adapters train through the prefill's backward.
      `adopt_trainables_tower` adopts expert + projections + adapters; after
      8 steps the tower base is bit-unchanged and its adapters and the expert
      moved; `save_trainables_tower` -> `load_trainables_tower` into fresh
      models restores every value bit for bit
"""

from std.math import abs
from std.random import seed
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.initializer import Deterministic
from noeira.nn.core.named_params import named_params
from noeira.nn.core.param import ParamVisitor, ParamVisitorRT, ParamWalkable
from noeira.nn.optimizer.adam import Adam
from noeira.nn.optimizer.param_arena import align_param_off
from noeira.nn.primitives.linear import Linear
from noeira.nn.peft.freeze import FreezeParams
from noeira.nn.peft.lora import LoRAAdapter
from noeira.deep_agents.smolvla.text import SmolVLMTextLayers, SMOLLM_THETA
from noeira.deep_agents.smolvla.expert import SmolVLAExpert
from noeira.deep_agents.smolvla.kv_cache import SmolVLAKVCache
from noeira.deep_agents.smolvla.fused import SmolVLAPrefill, SmolVLADenoise
from noeira.deep_agents.smolvla.train_step import SmolVLATrainStep
from noeira.deep_agents.smolvla.finetune import (
    zero_trainable_grads, adam_step_trainables, adopt_trainables,
    clip_trainables, adopt_trainables_tower, save_trainables_tower,
    load_trainables_tower,
)
from noeira.deep_agents.smolvla.flow_loss import build_xt_ut
from noeira.deep_agents.smolvla.attn_mask import (
    att_2d_mask, att_2d_mask_square, smolvla_ar,
)

comptime P = 6
comptime CHUNK = 3
comptime B = 1
comptime L = 2
comptime EW = 8
comptime EFF = 12
comptime W = 8
comptime FF_T = 12
comptime HEADS = 2
comptime NKV = 1
comptime HD = 4
comptime KVW = NKV * HD
comptime ADIM = 6
comptime ADIM_REAL = 3
comptime XN = B * CHUNK * ADIM
comptime XN_P = B * P * W
comptime PKV = B * P * KVW
comptime R = 3
comptime LR = Scalar[DT](3.0e-3)
comptime CLIP = Scalar[DT](10.0)

comptime AIn = Linear[ADIM, EW]
comptime TIn = Linear[2 * EW, EW]
comptime TOut = Linear[EW, EW]
comptime AOut = Linear[EW, ADIM]
comptime SProj = Linear[32, W]
comptime Cache = SmolVLAKVCache[L, P, CHUNK, NKV, HD, B]


struct Snap(Movable, ParamVisitor, ParamVisitorRT):
    """Every visited value (downloaded first), in walk order."""

    var vals: List[Scalar[DT]]

    def __init__(out self):
        self.vals = List[Scalar[DT]]()

    def visit_rt[target: StaticString](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, n: Int, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "gpu":
            param.download(ctx.value())
        for i in range(n):
            self.vals.append(param.data[i])

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.visit_rt[target](name, param, grad, m, v, N, apply_decay, ctx)


def _snap[M: ParamWalkable](mut m: M, c: DeviceContext) raises -> List[Scalar[DT]]:
    var s = Snap()
    m.for_each_param["gpu"](s, Optional(c))
    return s.vals.copy()


def _diff(ref a: List[Scalar[DT]], ref b: List[Scalar[DT]]) -> Int:
    var n = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            n += 1
    return n


def _arena_count[M: ParamWalkable](mut m: M, start: Int, c: DeviceContext) raises -> Int:
    """The arena's running offset after `m`'s trainable params, from `start`."""
    var n = start
    var nps = named_params["gpu"](m, Optional(c))
    for i in range(len(nps)):
        if nps[i].trainable:
            n = align_param_off(n) + nps[i].size
    return n


struct Data(Movable):
    var x_t: Tensor
    var u_t: Tensor
    var valid: Tensor
    var tl: List[Float64]

    def __init__(out self, c: DeviceContext) raises:
        var noise = Tensor.alloc(XN)
        var acts = Tensor.alloc(XN)
        for i in range(XN):
            noise.data[i] = Scalar[DT](((i * 37) % 19) - 9) * 0.07
        for t in range(CHUNK):
            for dd in range(ADIM_REAL):
                acts.data[t * ADIM + dd] = Scalar[DT](((t * 5 + dd * 3) % 7) - 3) * 0.2
        var times_t = Tensor.alloc(B)
        self.tl = List[Float64]()
        for b in range(B):
            times_t.data[b] = Scalar[DT](0.37)
            self.tl.append(0.37)
        var x_h = Tensor.alloc(XN)
        var u_h = Tensor.alloc(XN)
        build_xt_ut["cpu", B, CHUNK * ADIM](noise, acts, times_t, x_h, u_h, None)
        self.x_t = Tensor.alloc(XN)
        self.u_t = Tensor.alloc(XN)
        for i in range(XN):
            self.x_t.data[i] = x_h.data[i]
            self.u_t.data[i] = u_h.data[i]
        self.x_t.upload(c)
        self.u_t.upload(c)
        self.valid = Tensor.alloc(B * CHUNK)
        for i in range(B * CHUNK):
            self.valid.data[i] = Scalar[DT](1.0)
        self.valid.upload(c)


def _ab[
    IN: Int, OUT: Int, RR: Int, //
](mut a: LoRAAdapter[IN, OUT, RR], c: DeviceContext) raises -> Tuple[
    List[Scalar[DT]], List[Scalar[DT]]
]:
    """An adapter's A and B, downloaded."""
    a.lora_A.val.download(c)
    a.lora_B.val.download(c)
    var av = List[Scalar[DT]]()
    var bv = List[Scalar[DT]]()
    for i in range(a.A_SIZE):
        av.append(a.lora_A.val.data[i])
    for i in range(a.B_SIZE):
        bv.append(a.lora_B.val.data[i])
    return (av^, bv^)


# ── [1] LoRA on the expert ──────────────────────────────────────────────
def leg_expert(c: DeviceContext) raises:
    comptime Expert = SmolVLAExpert[L, EW, EFF, W, KVW, 2, R]
    comptime Den = SmolVLADenoise[
        P, CHUNK, B, L, EW, EFF, W, HEADS, NKV, HD, SMOLLM_THETA, 2, KVW, True, R
    ]
    comptime Step = SmolVLATrainStep[
        CHUNK, ADIM_REAL, ADIM, EW, B, L, EFF, W, HEADS, NKV, HD,
        SMOLLM_THETA, KVW, R,
    ]
    var ctx = Optional(c)
    var ar = smolvla_ar(3, 2, 1, CHUNK)
    var ms = att_2d_mask(ar, P, P + CHUNK, 0, P + CHUNK)
    var mc = att_2d_mask(ar, P, P + CHUNK, 0, P)
    seed(11)
    var e = Expert.make["gpu", Deterministic](ctx)
    var cache = Cache.make["gpu"](ctx)
    var den = Den.make["gpu"](ms, mc, ctx)
    var st = Step.make["gpu"](ctx)
    var ai = AIn.make["gpu", Deterministic](ctx)
    var ti = TIn.make["gpu", Deterministic](ctx)
    var to = TOut.make["gpu", Deterministic](ctx)
    var ao = AOut.make["gpu", Deterministic](ctx)
    var sp = SProj.make["gpu", Deterministic](ctx)
    var f = FreezeParams()
    e.for_each_base_param["gpu"](f, ctx)
    e.set_lora_alpha(6.0)
    var want = _arena_count(e, 0, c)
    want = _arena_count(ai, want, c)
    want = _arena_count(ti, want, c)
    want = _arena_count(to, want, c)
    want = _arena_count(ao, want, c)

    var base0 = Snap()
    e.for_each_base_param["gpu"](base0, ctx)
    var all0 = _snap(e, c)
    var proj0 = _snap(ai, c)
    var q0 = _ab(e.self_layers[0].q_lora, c)
    var dn0 = _ab(e.cross_layers[0].mlp.down_lora, c)
    var v_frozen0 = e.self_layers[0].q.weight.val.version
    var v_proj0 = ai.weight.val.version

    var opt = Adam(lr=LR)
    adopt_trainables["gpu", L, EW, EFF, W, KVW, ADIM, E_R=R](
        opt, e, ai, ti, to, ao, sp, ctx
    )
    var kp = Tensor.alloc(PKV)
    var vp = Tensor.alloc(PKV)
    for l in range(L):
        for i in range(PKV):
            kp.data[i] = Scalar[DT](((i * 31 + l * 7) % 13) - 6) * 0.11
            vp.data[i] = Scalar[DT](((i * 17 + l * 5) % 11) - 5) * 0.09
        kp.upload(c)
        vp.upload(c)
        cache.write_prefix["gpu"](l, kp, vp, ctx)
    var d = Data(c)
    st.set_times["gpu"](d.tl, ctx)
    var l0 = 0.0
    var l1 = 0.0
    for s in range(12):
        zero_trainable_grads["gpu", L, EW, EFF, W, KVW, ADIM, E_R=R](
            opt, e, ai, ti, to, ao, sp, ctx
        )
        var loss = st.run["gpu", P](
            e, cache, den, ai, ti, to, ao, d.x_t, d.u_t, d.valid, B * CHUNK, ctx
        )
        if s == 0:
            l0 = loss
        l1 = loss
        _ = clip_trainables["gpu"](opt, CLIP, ctx)
        adam_step_trainables["gpu", L, EW, EFF, W, KVW, ADIM, E_R=R](
            opt, e, ai, ti, to, ao, sp, ctx
        )
    c.synchronize()
    var base1 = Snap()
    e.for_each_base_param["gpu"](base1, ctx)
    var all1 = _snap(e, c)
    var proj1 = _snap(ai, c)
    var base_moved = _diff(base0.vals, base1.vals)
    var adapters_moved = _diff(all0, all1)  # only adapters can differ here
    var proj_moved = _diff(proj0, proj1)
    var q1 = _ab(e.self_layers[0].q_lora, c)
    var dn1 = _ab(e.cross_layers[0].mlp.down_lora, c)
    # B starts at zero, so it must have left it; A starts random, so it must
    # have moved from where it started (it moves only once B is nonzero).
    var na = _diff(q0[0], q1[0]) + _diff(dn0[0], dn1[0])
    var nb = _diff(q0[1], q1[1]) + _diff(dn0[1], dn1[1])
    var v_frozen1 = e.self_layers[0].q.weight.val.version
    var v_proj1 = ai.weight.val.version
    print("  [1] expert LoRA: arena", opt.arena.total, "(want", want, "); 12 steps loss",
          l0, "->", l1, "; base values moved", base_moved, "of", len(base0.vals),
          "; expert values moved (adapters only)", adapters_moved, "; action_in moved",
          proj_moved, "; A values moved (self q + cross down)", na, ", B values moved", nb,
          "; frozen q version", v_frozen0, "->", v_frozen1, ", action_in version",
          v_proj0, "->", v_proj1)
    assert_true(opt.arena.total == want, "the arena does not hold exactly adapters + projections")
    assert_true(base_moved == 0, "a frozen expert base value moved")
    assert_true(adapters_moved > 0 and proj_moved > 0, "the adapters or the projections did not train")
    assert_true(na > 0 and nb > 0, "an adapter's A or B did not move")
    assert_true(l1 < l0, "the loss did not fall")
    assert_true(v_frozen1 == v_frozen0, "the optimizer bumped a frozen weight's version")
    assert_true(v_proj1 > v_proj0, "the optimizer did not bump a trained weight's version")


# ── [2] LoRA on the text tower ──────────────────────────────────────────
def leg_tower(c: DeviceContext) raises:
    comptime Tower = SmolVLMTextLayers[L, W, FF_T, KVW, R]
    comptime Expert = SmolVLAExpert[L, EW, EFF, W, KVW, 2]
    comptime Pre = SmolVLAPrefill[
        P, CHUNK, B, L, W, FF_T, HEADS, NKV, HD, SMOLLM_THETA, True, R
    ]
    comptime Den = SmolVLADenoise[
        P, CHUNK, B, L, EW, EFF, W, HEADS, NKV, HD, SMOLLM_THETA, 2, KVW, True
    ]
    comptime Step = SmolVLATrainStep[
        CHUNK, ADIM_REAL, ADIM, EW, B, L, EFF, W, HEADS, NKV, HD,
        SMOLLM_THETA, KVW,
    ]
    var ctx = Optional(c)
    var ar_pre = smolvla_ar(3, 2, 1, 0)
    var ar = smolvla_ar(3, 2, 1, CHUNK)
    var mask_pre = att_2d_mask_square(ar_pre)
    var ms = att_2d_mask(ar, P, P + CHUNK, 0, P + CHUNK)
    var mc = att_2d_mask(ar, P, P + CHUNK, 0, P)
    seed(12)
    var tower = Tower.make["gpu", Deterministic](ctx)
    var e = Expert.make["gpu", Deterministic](ctx)
    var cache = Cache.make["gpu"](ctx)
    var pre = Pre.make["gpu"](mask_pre, ctx)
    var den = Den.make["gpu"](ms, mc, ctx)
    var st = Step.make["gpu"](ctx)
    var ai = AIn.make["gpu", Deterministic](ctx)
    var ti = TIn.make["gpu", Deterministic](ctx)
    var to = TOut.make["gpu", Deterministic](ctx)
    var ao = AOut.make["gpu", Deterministic](ctx)
    var f = FreezeParams()
    tower.for_each_base_param["gpu"](f, ctx)
    tower.set_lora_alpha(6.0)
    var want = _arena_count(e, 0, c)
    want = _arena_count(ai, want, c)
    want = _arena_count(ti, want, c)
    want = _arena_count(to, want, c)
    want = _arena_count(ao, want, c)
    want = _arena_count(tower, want, c)

    var tb0 = Snap()
    tower.for_each_base_param["gpu"](tb0, ctx)
    var tall0 = _snap(tower, c)
    var e0 = _snap(e, c)
    var opt = Adam(lr=LR)
    adopt_trainables_tower["gpu", L, EW, EFF, W, KVW, ADIM](
        opt, e, ai, ti, to, ao, tower, ctx
    )
    var x = Tensor.alloc(XN_P)
    for i in range(XN_P):
        x.data[i] = Scalar[DT](((i * 29) % 17) - 8) * 0.06
    x.upload(c)
    var po = Tensor.alloc_gpu(c, XN_P)
    var gx = Tensor.alloc_gpu(c, XN_P)
    var d = Data(c)
    st.set_times["gpu"](d.tl, ctx)
    var sp = SProj.make["gpu", Deterministic](ctx)
    var l0 = 0.0
    var l1 = 0.0
    for s in range(8):
        zero_trainable_grads["gpu", L, EW, EFF, W, KVW, ADIM](
            opt, e, ai, ti, to, ao, sp, ctx
        )
        cache.reset()
        pre.run["gpu"](tower, cache, x, po, ctx)
        var loss = st.run["gpu", P](
            e, cache, den, ai, ti, to, ao, d.x_t, d.u_t, d.valid, B * CHUNK, ctx
        )
        pre.backward["gpu"](tower, st.pool[Step.GCK], st.pool[Step.GCV], gx, ctx)
        if s == 0:
            l0 = loss
        l1 = loss
        _ = clip_trainables["gpu"](opt, CLIP, ctx)
        adam_step_trainables["gpu", L, EW, EFF, W, KVW, ADIM](
            opt, e, ai, ti, to, ao, sp, ctx
        )
    c.synchronize()
    var tb1 = Snap()
    tower.for_each_base_param["gpu"](tb1, ctx)
    var tall1 = _snap(tower, c)
    var e1 = _snap(e, c)
    var base_moved = _diff(tb0.vals, tb1.vals)
    var adapters_moved = _diff(tall0, tall1)
    var expert_moved = _diff(e0, e1)

    # checkpoint round trip into fresh models
    var path = String("/tmp/noeira_test_finetune_lora_tower.ckpt")
    save_trainables_tower["gpu", L, EW, EFF, W, KVW, ADIM](
        path, e, ai, ti, to, ao, tower, True, ctx
    )
    var tower2 = Tower.make["gpu", Deterministic](ctx)
    var e2 = Expert.make["gpu", Deterministic](ctx)
    var ai2 = AIn.make["gpu", Deterministic](ctx)
    var ti2 = TIn.make["gpu", Deterministic](ctx)
    var to2 = TOut.make["gpu", Deterministic](ctx)
    var ao2 = AOut.make["gpu", Deterministic](ctx)
    var f2 = FreezeParams()
    tower2.for_each_base_param["gpu"](f2, ctx)
    _ = load_trainables_tower["gpu", L, EW, EFF, W, KVW, ADIM](
        path, e2, ai2, ti2, to2, ao2, tower2, ctx
    )
    var rt = (
        _diff(_snap(tower2, c), tall1) + _diff(_snap(e2, c), e1)
        + _diff(_snap(ai2, c), _snap(ai, c)) + _diff(_snap(ao2, c), _snap(ao, c))
    )
    print("  [2] tower LoRA: arena", opt.arena.total, "(want", want, "); 8 steps loss",
          l0, "->", l1, "; tower base moved", base_moved, "of", len(tb0.vals),
          "; tower values moved (adapters only)", adapters_moved, "; expert moved",
          expert_moved, "; checkpoint round trip mismatches", rt)
    assert_true(opt.arena.total == want, "the arena does not hold expert + projections + tower adapters")
    assert_true(base_moved == 0, "a frozen text-tower value moved")
    assert_true(adapters_moved > 0, "the tower's adapters did not train")
    assert_true(expert_moved > 0, "the expert did not train")
    assert_true(rt == 0, "the tower checkpoint did not round-trip")


def main() raises:
    print("SmolVLA fine-tune with LoRA (GPU)")
    var c = DeviceContext()
    leg_expert(c)
    leg_tower(c)
    print("PASSED")

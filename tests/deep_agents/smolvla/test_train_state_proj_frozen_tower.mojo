# +--------------------------------------------------------------------------+ #
# | train_state_proj through a FROZEN text tower: the same training, bit for bit
# +--------------------------------------------------------------------------+ #
"""Freezing the VLM text tower changes nothing a fine-tune computes.

    pixi run mojo run -I . \\
        tests/deep_agents/smolvla/test_train_state_proj_frozen_tower.mojo

`train_state_proj = True` is the one regime where the text tower sits inside
the training step: `state_proj`'s gradient arrives only through
`SmolVLAPrefill.backward`, a full backward through every frozen layer. Before
`Linear.freeze` that backward formed and accumulated a weight gradient in
every tower linear (never read, never zeroed); frozen, it forms grad_input
only. The training must not notice.

Ten Adam steps (weight decay on) of the toy chain `test_state_proj_grad.mojo`
gates, `state_proj -> prefill -> cache -> one denoising step -> loss`, training
the expert and `state_proj`, twice: once with the tower frozen
(`FreezeParams`), once with it trainable but never stepped (today's regime).

  [1] the loss at every step is BIT-identical across the two runs, and it moves
  [2] `state_proj` and every expert parameter end BIT-identical, and moved
  [3] the tower's values never move in either run, and the frozen tower has
      no gradient buffer left
"""

from std.math import abs
from std.testing import assert_true

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.core.param import ParamVisitor, ParamVisitorRT, walk_params
from noeira.nn.optimizer.adam import Adam
from noeira.nn.primitives.linear import Linear
from noeira.nn.peft.freeze import FreezeParams
from noeira.deep_agents.smolvla.text import SmolVLMTextLayers, SMOLLM_THETA
from noeira.deep_agents.smolvla.expert import SmolVLAExpert
from noeira.deep_agents.smolvla.kv_cache import SmolVLAKVCache
from noeira.deep_agents.smolvla.fused import SmolVLAPrefill, SmolVLADenoise
from noeira.deep_agents.smolvla.finetune import state_proj_backward
from noeira.deep_agents.smolvla.attn_mask import (
    att_2d_mask, att_2d_mask_square, smolvla_ar,
)

comptime P = 6
comptime S = 3
comptime B = 1
comptime L = 2
comptime W = 8
comptime FF_T = 12
comptime EW = 8
comptime EFF = 12
comptime HEADS = 2
comptime NKV = 1
comptime HD = 4
comptime KVW = NKV * HD
comptime SDIM = 32
comptime XN_P = B * P * W
comptime XN_S = B * S * EW
comptime CACHE_N = L * B * P * KVW
comptime STEPS = 10

comptime Tower = SmolVLMTextLayers[L, W, FF_T, KVW]
comptime Expert = SmolVLAExpert[L, EW, EFF, W, KVW, 2]
comptime Cache = SmolVLAKVCache[L, P, S, NKV, HD, B]
comptime Pre = SmolVLAPrefill[
    P, S, B, L, W, FF_T, HEADS, NKV, HD, SMOLLM_THETA, True
]
comptime Den = SmolVLADenoise[
    P, S, B, L, EW, EFF, W, HEADS, NKV, HD, SMOLLM_THETA, 2, KVW, True
]
comptime SProj = Linear[SDIM, W]


struct Values(Movable, ParamVisitor, ParamVisitorRT):
    """Every visited value, in walk order, plus how many params had a
    gradient buffer."""

    var vals: List[Scalar[DT]]
    var with_grad: Int

    def __init__(out self):
        self.vals = List[Scalar[DT]]()
        self.with_grad = 0

    def visit_rt[target: StaticString](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, n: Int, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        for i in range(n):
            self.vals.append(param.data[i])
        if grad.n > 0:
            self.with_grad += 1

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.visit_rt[target](name, param, grad, m, v, N, apply_decay, ctx)


struct Run(Movable):
    var losses: List[Float64]
    var sp: List[Scalar[DT]]
    var expert: List[Scalar[DT]]
    var tower0: List[Scalar[DT]]
    var tower1: List[Scalar[DT]]
    var tower_grads: Int

    def __init__(out self):
        self.losses = List[Float64]()
        self.sp = List[Scalar[DT]]()
        self.expert = List[Scalar[DT]]()
        self.tower0 = List[Scalar[DT]]()
        self.tower1 = List[Scalar[DT]]()
        self.tower_grads = 0


def _tower_values(mut tower: Tower) raises -> Values:
    var v = Values()
    tower.for_each_param["cpu"](v, None)
    return v^


def train(frozen: Bool) raises -> Run:
    var ar_pre = smolvla_ar(3, 2, 1, 0)
    var ar_full = smolvla_ar(3, 2, 1, S)
    var mask_pre = att_2d_mask_square(ar_pre)
    var mask_self = att_2d_mask(ar_full, P, P + S, 0, P + S)
    var mask_cross = att_2d_mask(ar_full, P, P + S, 0, P)
    var tower = Tower.make["cpu", Deterministic]()
    var e = Expert.make["cpu", Deterministic]()
    var c = Cache.make["cpu"]()
    var pre = Pre.make["cpu"](mask_pre, None)
    var den = Den.make["cpu"](mask_self, mask_cross, None)
    var sp = SProj.make["cpu", Deterministic]()
    if frozen:
        var f = FreezeParams()
        tower.for_each_param["cpu"](f, None)

    var base = List[Scalar[DT]](unsafe_uninit_length=XN_P)
    for i in range(XN_P):
        base[i] = Scalar[DT](((i * 29) % 17) - 8) * 0.06
    var st = Tensor.alloc(B * SDIM)
    for j in range(SDIM):
        st.data[j] = Scalar[DT](Float32(j) * 0.3 - 0.9) if j < 6 else Scalar[DT](0)
    var xs = Tensor.alloc(XN_S)
    for i in range(XN_S):
        xs.data[i] = Scalar[DT](((i * 37) % 19) - 9) * 0.05
    var grad_out = Tensor.alloc(XN_S)
    for i in range(XN_S):
        grad_out.data[i] = Scalar[DT](Float32(((i * 23) % 7) - 3) * 0.3)

    var tok = Tensor.alloc(B * W)
    var x = Tensor.alloc(XN_P)
    var po = Tensor.alloc(XN_P)
    var out = Tensor.alloc(XN_S)
    var gx_s = Tensor.alloc(XN_S)
    var gck = Tensor.alloc(CACHE_N)
    var gcv = Tensor.alloc(CACHE_N)
    var grad_x = Tensor.alloc(XN_P)
    var g_tok = Tensor.alloc(B * W)
    var g_state = Tensor.alloc(B * SDIM)
    var opt = Adam(lr=1e-2, wd=0.1)

    var r = Run()
    r.tower0 = _tower_values(tower).vals.copy()
    for _ in range(STEPS):
        # forward: state_proj -> the last prefix token -> prefill -> denoise
        sp.forward["cpu", B](TensorRefs[1](st), tok, None)
        for i in range(XN_P):
            x.data[i] = base[i]
        for j in range(W):
            x.data[(P - 1) * W + j] = tok.data[j]
        c.reset()
        pre.run["cpu"](tower, c, x, po, None)
        den.step["cpu"](e, c, xs, out, None)
        var loss = 0.0
        for i in range(XN_S):
            loss += Float64(out.data[i]) * Float64(grad_out.data[i])
        r.losses.append(loss)
        # backward: expert -> cache -> the 16 (here 2) frozen layers -> state_proj
        e.zero_grad["cpu"](None)
        sp.zero_grad["cpu"](None)
        den.backward["cpu"](e, c, grad_out, gx_s, gck, gcv, None)
        pre.backward["cpu"](tower, gck, gcv, grad_x, None)
        state_proj_backward["cpu", B, P, W, SDIM](sp, st, grad_x, g_tok, g_state, None)
        opt.begin_step()
        walk_params["cpu"](e, opt, None)
        walk_params["cpu"](sp, opt, None)

    var vs = Values()
    walk_params["cpu"](sp, vs, None)
    r.sp = vs.vals.copy()
    var ve = Values()
    walk_params["cpu"](e, ve, None)
    r.expert = ve.vals.copy()
    var vt = _tower_values(tower)
    r.tower1 = vt.vals.copy()
    r.tower_grads = vt.with_grad
    return r^


def _diff(ref a: List[Scalar[DT]], ref b: List[Scalar[DT]]) -> Int:
    var n = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            n += 1
    return n


def main() raises:
    print("train_state_proj, 10 Adam steps, text tower frozen vs not")
    var fz = train(True)
    var tr = train(False)

    var loss_bad = 0
    for i in range(STEPS):
        if fz.losses[i] != tr.losses[i]:
            loss_bad += 1
    var moved = abs(fz.losses[STEPS - 1] - fz.losses[0])
    print("  [1] losses differ at", loss_bad, "of", STEPS, "steps;  L0",
          fz.losses[0], " L9", fz.losses[STEPS - 1])
    var init = train_init_values()
    var sp_bad = _diff(fz.sp, tr.sp)
    var ex_bad = _diff(fz.expert, tr.expert)
    var sp_moved = _diff(fz.sp, init[0])
    var ex_moved = _diff(fz.expert, init[1])
    print("  [2] state_proj mismatches", sp_bad, "of", len(fz.sp), " (moved",
          sp_moved, ")   expert mismatches", ex_bad, "of", len(fz.expert),
          " (moved", ex_moved, ")")
    var t_fz = _diff(fz.tower0, fz.tower1)
    var t_tr = _diff(tr.tower0, tr.tower1)
    print("  [3] tower values moved: frozen run", t_fz, ", trainable run", t_tr,
          ";  tower params with a gradient buffer: frozen", fz.tower_grads,
          ", trainable", tr.tower_grads)
    assert_true(loss_bad == 0, "freezing the tower changed the loss")
    assert_true(moved > 1e-6, "the loss did not move: [1] is vacuous")
    assert_true(sp_bad == 0 and ex_bad == 0, "freezing the tower changed what trained")
    assert_true(sp_moved > 0 and ex_moved > 0, "nothing trained: [2] is vacuous")
    assert_true(t_fz == 0 and t_tr == 0, "a tower weight moved")
    assert_true(fz.tower_grads == 0, "the frozen tower kept a gradient buffer")
    assert_true(tr.tower_grads > 0, "the control tower had no gradient buffers")
    print("PASSED")


def train_init_values() raises -> Tuple[List[Scalar[DT]], List[Scalar[DT]]]:
    """The untrained `state_proj` and expert, for the 'it moved' checks."""
    var sp = SProj.make["cpu", Deterministic]()
    var e = Expert.make["cpu", Deterministic]()
    var vs = Values()
    walk_params["cpu"](sp, vs, None)
    var ve = Values()
    walk_params["cpu"](e, ve, None)
    return (vs.vals.copy(), ve.vals.copy())

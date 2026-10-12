"""B on a GATHERED projection — `WalkerBNetProj` (FB_MEBE_G1_PLAN.md §2 B1).

The projection is a composition (`Parallel[Slice, Slice]` in front of the
B MLP), so the gate checks the COMPOSITION against the bare MLP fed the
hand-gathered columns, with the same weights:

  [1] forward: `B_proj(s) == B_mlp(s[ϑ])` bit for bit on random rows.
  [2] vjp: the input gradient on the gathered columns equals the bare MLP's,
      and is EXACTLY zero on every column B does not read — a slice in the
      wrong place would put gradient (and so B's dependence) elsewhere.
  [3] the parameter gradients agree (the projection changes what B sees,
      not how it learns).

Run: pixi run mojo run -I . tests/deep_agents/test_fb_proj_b.mojo
"""

from std.random import random_float64, seed
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward, call_vjp
from noeira.nn.core.initializer import Xavier
from noeira.nn.core.param import ParamVisitor
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import ReLU
from noeira.nn.primitives.layer_norm_no_affine import LayerNormNoAffine
from noeira.deep_agents.fb.walker_nets import (
    WalkerBNetProj, WALKER_OBS, WALKER_D, WALKER_BHID, WALKER_PROJ,
)

comptime N: Int = 8
comptime Mlp = Sequential[
    Linear[WALKER_PROJ, WALKER_BHID], ReLU[WALKER_BHID],
    Linear[WALKER_BHID, WALKER_D], LayerNormNoAffine[WALKER_D],
]


struct _Grab(ParamVisitor):
    var vals: List[List[Scalar[DT]]]
    var grads: List[List[Scalar[DT]]]

    def __init__(out self):
        self.vals = List[List[Scalar[DT]]]()
        self.grads = List[List[Scalar[DT]]]()

    def visit[target: StaticString, NP: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        var g = List[Scalar[DT]](capacity=NP)
        var gg = List[Scalar[DT]](capacity=NP)
        for i in range(NP):
            g.append(param.data[i])
            gg.append(grad.data[i])
        self.vals.append(g^)
        self.grads.append(gg^)


struct _Put(ParamVisitor):
    var vals: List[List[Scalar[DT]]]
    var k: Int

    def __init__(out self, var vals: List[List[Scalar[DT]]]):
        self.vals = vals^
        self.k = 0

    def visit[target: StaticString, NP: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        assert_true(len(self.vals[self.k]) == NP, "param sizes differ")
        for i in range(NP):
            param.data[i] = self.vals[self.k][i]
        self.k += 1


def main() raises:
    var COLS: List[Int] = [0, 1, 14, 15, 16, 17]
    seed(20261012)
    var proj = WalkerBNetProj.make["cpu", Xavier](None)
    var mlp = Mlp.make["cpu", Xavier](None)
    var g0 = _Grab()
    proj.for_each_param["cpu"](g0, None)
    var put = _Put(g0.vals.copy())
    mlp.for_each_param["cpu"](put, None)
    assert_true(put.k == len(g0.vals), "param walks differ in length")

    var x = Tensor.alloc(N * WALKER_OBS)
    for i in range(N * WALKER_OBS):
        x.data[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
    var xs = Tensor.alloc(N * WALKER_PROJ)
    for i in range(N):
        for k in range(WALKER_PROJ):
            xs.data[i * WALKER_PROJ + k] = x.data[i * WALKER_OBS + COLS[k]]

    print("[1] forward bit-identical ...")
    var y1 = Tensor()
    var y2 = Tensor()
    call_forward["cpu", N](proj, TensorRefs[1, MutAnyOrigin](x), y1, None)
    call_forward["cpu", N](mlp, TensorRefs[1, MutAnyOrigin](xs), y2, None)
    var diff = 0
    for i in range(N * WALKER_D):
        if y1.data[i] != y2.data[i]:
            diff += 1
    print("      differing outputs:", diff, "/", N * WALKER_D)
    assert_true(diff == 0, "projected B differs from the MLP on gathered columns")

    print("[2]+[3] vjp: input grads on ϑ equal, zero elsewhere; param grads equal ...")
    var cot = Tensor.alloc(N * WALKER_D)
    for i in range(N * WALKER_D):
        cot.data[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
    var gx1 = Tensor.alloc(N * WALKER_OBS)
    var gx2 = Tensor.alloc(N * WALKER_PROJ)
    proj.zero_grad["cpu"](None)
    mlp.zero_grad["cpu"](None)
    call_vjp["cpu", N](
        proj, TensorRefs[1, MutAnyOrigin](x), cot, TensorRefs[1, MutAnyOrigin](gx1), None
    )
    call_vjp["cpu", N](
        mlp, TensorRefs[1, MutAnyOrigin](xs), cot, TensorRefs[1, MutAnyOrigin](gx2), None
    )
    var bad_sel = 0
    var nonzero_off = 0
    for i in range(N):
        for c in range(WALKER_OBS):
            var sel = -1
            for k in range(WALKER_PROJ):
                if COLS[k] == c:
                    sel = k
            var g = gx1.data[i * WALKER_OBS + c]
            if sel >= 0:
                if g != gx2.data[i * WALKER_PROJ + sel]:
                    bad_sel += 1
            elif g != Scalar[DT](0.0):
                nonzero_off += 1
    print("      selected-column mismatches", bad_sel, "  nonzero off-ϑ grads", nonzero_off)
    assert_true(bad_sel == 0, "input gradient on ϑ differs")
    assert_true(nonzero_off == 0, "gradient leaked onto a column B does not read")
    var a = _Grab()
    var b = _Grab()
    proj.for_each_param["cpu"](a, None)
    mlp.for_each_param["cpu"](b, None)
    var pg = 0
    for k in range(len(a.grads)):
        for j in range(len(a.grads[k])):
            if a.grads[k][j] != b.grads[k][j]:
                pg += 1
    print("      differing param grads:", pg)
    assert_true(pg == 0, "parameter gradients differ")
    print("ALL PASS — projected B")

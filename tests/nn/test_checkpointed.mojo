"""Checkpointed — recomputing a block in the vjp changes nothing but memory.

`Repeat[2, Checkpointed[TransformerBlock]]` (dim 192: every Linear takes the
K-padded path; attention; LayerNorm; residuals) runs the same input twice,
checkpointing OFF then ON: the output and every parameter gradient must be
BIT-identical (the recompute runs the same kernels on the same inputs), on CPU
and GPU. And with it ON the blocks must hold nothing after the forward: the
gradient pass then has to rebuild every buffer it reads — a buffer
`release_buffers` dropped but the vjp still needed would show here as garbage.
"""

from std.random import seed, random_float64
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.initializer import Kaiming
from noeira.nn import Repeat, Checkpointed
from noeira.nn.models.transformer import TransformerBlock


comptime S = 17
comptime D = 192
comptime B = 4
comptime Net = Repeat[2, Checkpointed[TransformerBlock[D, 3, S, 4 * D]]]


struct _Grads(ParamVisitor):
    var g: List[Scalar[DT]]

    def __init__(out self):
        self.g = List[Scalar[DT]]()

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "gpu":
            grad.download(ctx.value())
        for i in range(N):
            self.g.append(grad.data[i])


def _pass[target: StaticString](
    mut net: Net, x: List[Scalar[DT]], gy: List[Scalar[DT]], ctx: Optional[DeviceContext]
) raises -> Tuple[List[Scalar[DT]], List[Scalar[DT]]]:
    var xin = Tensor.alloc(B * S * D)
    var g = Tensor.alloc(B * S * D)
    for i in range(B * S * D):
        xin.data[i] = x[i]
        g.data[i] = gy[i]
    comptime if target == "gpu":
        xin.upload(ctx.value())
        g.upload(ctx.value())
    var y = Tensor()
    var gx = Tensor()
    net.zero_grad[target](ctx)
    net.forward[target, B](TensorRefs[1](xin), y, ctx)
    net.vjp[target, B](TensorRefs[1](xin), g, TensorRefs[1](gx), ctx)
    comptime if target == "gpu":
        ctx.value().synchronize()
        y.download(ctx.value())
        gx.download(ctx.value())
    var out = List[Scalar[DT]]()
    for i in range(B * S * D):
        out.append(y.data[i])
    for i in range(B * S * D):
        out.append(gx.data[i])
    var gv = _Grads()
    net.for_each_param[target](gv, ctx)
    var grads = gv.g.copy()
    return (out^, grads^)


def _run[target: StaticString](ctx: Optional[DeviceContext]) raises -> Int:
    seed(7)
    var net = Net.make[target, Kaiming](ctx)
    var x = List[Scalar[DT]]()
    var gy = List[Scalar[DT]]()
    for _ in range(B * S * D):
        x.append(Scalar[DT](random_float64(-1, 1)))
        gy.append(Scalar[DT](random_float64(-1, 1)))
    var off = _pass[target](net, x, gy, ctx)
    net.set_attr["checkpoint"](Scalar[DT](1))
    var on = _pass[target](net, x, gy, ctx)
    # twice with it on: buffers released by the first ON pass come back
    var on2 = _pass[target](net, x, gy, ctx)
    var fails = 0
    var n_act = 0
    var n_grad = 0
    for i in range(len(off[0])):
        if off[0][i] != on[0][i] or off[0][i] != on2[0][i]:
            n_act += 1
    for i in range(len(off[1])):
        if off[1][i] != on[1][i] or off[1][i] != on2[1][i]:
            n_grad += 1
    print("  --", target, ":", len(off[0]), "output + input-grad values,", len(off[1]),
          "param grads; differing:", n_act, "/", n_grad)
    if n_act > 0 or n_grad > 0:
        fails += 1
    return fails


def main() raises:
    print("Checkpointed: recompute-in-vjp is bit-identical to keeping activations")
    var fails = _run["cpu"](None)
    var c = DeviceContext()
    fails += _run["gpu"](Optional(c))
    if fails > 0:
        raise Error("FAIL: checkpointing changed the result")
    print("PASS")

"""Checkpoint save/load round-trip + Adam.step adapter (CPU + GPU).

- Train a Sequential MLP via the model-walking `opt.step(model)` adapter.
- save_params → fresh model → load_params; the loaded model's forward output
  must match the trained model's (positional param round-trip).

Run: pixi run -e apple mojo run -I . tests/nn/test_checkpoint_adam_storage.mojo
"""

from std.testing import assert_true
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.core.checkpoint import save_params, load_params
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import ReLU
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.optimizer.adam import Adam
from noeira.nn.loss.mse import mse_forward, mse_backward


comptime B = 4
comptime IN = 3
comptime H = 6
comptime OUT = 2
comptime MODEL = Sequential[Linear[IN, H], ReLU[H], Linear[H, OUT]]


def _check[target: StaticString](path: String, ctx: Optional[DeviceContext]) raises -> Bool:
    comptime TOL = Scalar[DT](1e-5)
    var model = MODEL.make[target, Deterministic](ctx)
    var opt = Adam(lr=0.05)

    var x = Tensor.alloc(B * IN)
    var tgt = Tensor.alloc(B * OUT)
    for i in range(B * IN):
        x.data[i] = Scalar[DT]((i % 5) - 2) * 0.5
    for i in range(B * OUT):
        tgt.data[i] = Scalar[DT](1) if (i % 2 == 0) else Scalar[DT](-1)
    var pred = Tensor.alloc(B * OUT)
    var grad = Tensor.alloc(B * OUT)
    var gi = Tensor.alloc(B * IN)

    comptime if target == "gpu":
        x.upload(ctx.value()); tgt.upload(ctx.value())

    # train via the opt.step(model) adapter
    for _ in range(40):
        model.zero_grad[target](ctx)
        model.forward[target, B](TensorRefs[1](x), pred, ctx)
        mse_backward[target, B, OUT](pred, tgt, grad, ctx)
        model.vjp[target, B](TensorRefs[1](x), grad, TensorRefs[1](gi), ctx)
        opt.step[target](model, ctx)

    # reference forward output of the TRAINED model
    model.forward[target, B](TensorRefs[1](x), pred, ctx)
    comptime if target == "gpu":
        pred.download(ctx.value())
    var ref_out = List[Scalar[DT]](length=B * OUT, fill=Scalar[DT](0))
    for i in range(B * OUT):
        ref_out[i] = pred.data[i]

    # save → fresh model → load → forward must match
    save_params[target](model, path, ctx)
    var fresh = MODEL.make[target, Deterministic](ctx)
    load_params[target](fresh, path, ctx)
    var pred2 = Tensor.alloc(B * OUT)
    fresh.forward[target, B](TensorRefs[1](x), pred2, ctx)
    comptime if target == "gpu":
        pred2.download(ctx.value())

    var ok = True
    for i in range(B * OUT):
        if abs(pred2.data[i] - ref_out[i]) > TOL:
            ok = False
    return ok


def main() raises:
    print("=" * 70)
    print("Checkpoint round-trip + Adam.step adapter")
    print("=" * 70)
    var oc = _check["cpu"]("/tmp/storage_ckpt_cpu.txt", None)
    print("  CPU:", "OK" if oc else "FAIL")
    var og = True
    comptime if has_accelerator():
        var c = DeviceContext()
        og = _check["gpu"]("/tmp/storage_ckpt_gpu.txt", Optional(c))
        print("  GPU:", "OK" if og else "FAIL")
    else:
        print(
            "  GPU: SKIPPED (no accelerator;"
            " run with -e apple / -e nvidia)"
        )
    assert_true(oc and og, "checkpoint round-trip")
    print("CHECKPOINT + ADAM.STEP OK")

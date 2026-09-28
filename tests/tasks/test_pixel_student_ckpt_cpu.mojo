"""A pixel student's checkpoint gives the SAME actions on the CPU as on the
device it was trained on.

    pixi run mojo run -I . tests/tasks/test_pixel_student_ckpt_cpu.mojo [CKPT]

The trainer saves from the GPU (`save_params["gpu"]`); the real-arm deploy
loads on the CPU (`load_params["cpu"]`, batch 1). A save of the wrong copy,
a load that fills nothing, or a CPU conv that lays out its weights
differently would each leave a policy that acts — just not as trained, and
on a real arm. With no path, a freshly initialised net is saved and
reloaded (the round trip alone); with one (e.g.
projects/so101-tower/policies/pixel_lift.ckpt), THAT checkpoint on both
devices. The inputs are random observations of the student's shape.
"""

from std.random import random_float64, seed
from std.sys import argv
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.checkpoint import load_params, save_params
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.tasks.pixel_student import StudentNet, IN_DIM, ACT

comptime B = 8


def main() raises:
    seed(11)
    var a = argv()
    var path = String(a[1]) if len(a) > 1 else String("")
    var ctx = DeviceContext()
    var gpu = StudentNet.make["gpu", Kaiming](Optional(ctx))
    if path.byte_length() == 0:
        path = String("/tmp/pixel_student_ckpt_roundtrip.ckpt")
        save_params["gpu"](gpu, path, Optional(ctx))
        print("  no checkpoint given: a fresh net saved to", path)
    else:
        load_params["gpu"](gpu, path, Optional(ctx))
    var cpu = StudentNet.make["cpu", Kaiming](None)
    load_params["cpu"](cpu, path, None)
    var xg = Tensor.alloc(B * IN_DIM)
    for k in range(B * IN_DIM):
        xg.data[k] = Scalar[DT](random_float64(-0.5, 0.5))
    var yg = Tensor.alloc(B * ACT)
    xg.upload(ctx)
    yg.upload(ctx)
    gpu.forward["gpu", B](TensorRefs[1](xg), yg, Optional(ctx))
    yg.download(ctx)
    ctx.synchronize()
    var worst = 0.0
    var mag = 0.0
    var x1 = Tensor.alloc(IN_DIM)
    var y1 = Tensor.alloc(ACT)
    for b in range(B):
        for k in range(IN_DIM):
            x1.data[k] = xg.data[b * IN_DIM + k]
        cpu.forward["cpu", 1](TensorRefs[1](x1), y1, None)
        for j in range(ACT):
            var d = abs(Float64(y1.data[j]) - Float64(yg.data[b * ACT + j]))
            if d > worst:
                worst = d
            mag += abs(Float64(yg.data[b * ACT + j]))
    mag /= Float64(B * ACT)
    print("  checkpoint", path)
    print("  max |cpu - gpu| over", B, "observations x", ACT, "actions =",
          worst, "(mean |action|", mag, ")")
    assert_true(mag > 1e-3, "the network outputs ~0: nothing was loaded")
    # ⚠ TF32 on CUDA, fp32 on the CPU: agreement to ~1e-3 of an action in
    # [-1, 1] is the same policy; a layout or load error is O(1).
    assert_true(worst < 2e-2, "the CPU forward does not reproduce the GPU's")
    print("PIXEL STUDENT CKPT CPU OK")

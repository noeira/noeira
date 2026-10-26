"""OneHotKLLoss[STOCH, CLASSES] storage Module — correctness gate.

DreamerV3 dynamics/representation KL as a storage `Module` (ARITY=2,
OUT_DIM=2; output `[B,2]` = `[dyn, rep]`). The op carries an ASYMMETRIC
stop-gradient: the dyn cotangent (output channel 0) flows ONLY to the prior
logits (grad_inputs[1]), the rep cotangent (channel 1) ONLY to the post
logits (grad_inputs[0]). This test is the critical correctness check.

  1. Finite-difference grad check (CPU) for BOTH input branches:
       - perturb the POST logits, set go=[0, 1] (rep cotangent only) →
         compares the FD of Σ rep against grad_inputs[0] (grad_post).
       - perturb the PRIOR logits, set go=[1, 0] (dyn cotangent only) →
         compares the FD of Σ dyn against grad_inputs[1] (grad_prior).
     Routing the cotangent to ONE channel at a time isolates each branch and
     catches an asymmetric-routing mistake (e.g. dyn→post / rep→prior swap).
  2. CPU vs GPU parity (forward + vjp), max abs diff < 1e-4.

Run:
  rm -f noeira.mojoc && pixi run -e apple mojo run -I . tests/nn/test_onehot_kl_storage.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_pack import TensorPack
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Zero
from noeira.deep_agents.dreamerv3.onehot_kl import OneHotKLLoss


comptime STOCH = 3
comptime CLASSES = 4
comptime SC = STOCH * CLASSES
comptime B = 5
comptime N = B * SC


def _fill(mut inp: TensorPack[2]) raises:
    inp[0].ensure(N)
    inp[1].ensure(N)
    for i in range(N):
        inp[0].data[i] = Scalar[DT]((i % 7) - 3) * 0.3
        inp[1].data[i] = Scalar[DT]((i % 5) - 2) * 0.45


def _scalar_loss[CH: Int](out_t: Tensor) -> Scalar[DT]:
    # Σ over the chosen output channel (CH=0 dyn, CH=1 rep).
    var s: Scalar[DT] = 0.0
    for b in range(B):
        s += out_t.data[b * 2 + CH]
    return s


def test_fd_post_branch() raises:
    # rep cotangent only (go = [0, 1]) → grad_post = grad_inputs[0].
    print("FD grad check — POST branch (rep cotangent) ...")
    var m = OneHotKLLoss[STOCH, CLASSES].make["cpu", INIT=Zero]()
    var inp = TensorPack[2]()
    _fill(inp)
    var go = Tensor.alloc(B * 2)
    for b in range(B):
        go.data[b * 2] = Scalar[DT](0.0)      # dyn cotangent off
        go.data[b * 2 + 1] = Scalar[DT](1.0)  # rep cotangent on
    var gpk = TensorPack[2]()
    m.vjp["cpu", B](
        TensorRefs[2](inp[0], inp[1]), go, TensorRefs[2](gpk[0], gpk[1])
    )

    var eps = Scalar[DT](1e-3)
    var op = Tensor.alloc(B * 2)
    var om = Tensor.alloc(B * 2)
    var maxd = Scalar[DT](0)
    for idx in range(N):
        var saved = inp[0].data[idx]
        inp[0].data[idx] = saved + eps
        m.forward["cpu", B](TensorRefs[2](inp[0], inp[1]), op)
        inp[0].data[idx] = saved - eps
        m.forward["cpu", B](TensorRefs[2](inp[0], inp[1]), om)
        inp[0].data[idx] = saved
        # rep cotangent → Σ rep channel.
        var fd = (_scalar_loss[1](op) - _scalar_loss[1](om)) / (
            Scalar[DT](2) * eps
        )
        var d = abs(fd - gpk[0].data[idx])
        if d > maxd:
            maxd = d
    print("  max |fd - grad_post| =", maxd)
    assert_true(maxd < Scalar[DT](1e-2), "POST branch FD")
    print("  ok")


def test_fd_prior_branch() raises:
    # dyn cotangent only (go = [1, 0]) → grad_prior = grad_inputs[1].
    print("FD grad check — PRIOR branch (dyn cotangent) ...")
    var m = OneHotKLLoss[STOCH, CLASSES].make["cpu", INIT=Zero]()
    var inp = TensorPack[2]()
    _fill(inp)
    var go = Tensor.alloc(B * 2)
    for b in range(B):
        go.data[b * 2] = Scalar[DT](1.0)      # dyn cotangent on
        go.data[b * 2 + 1] = Scalar[DT](0.0)  # rep cotangent off
    var gpk = TensorPack[2]()
    m.vjp["cpu", B](
        TensorRefs[2](inp[0], inp[1]), go, TensorRefs[2](gpk[0], gpk[1])
    )

    var eps = Scalar[DT](1e-3)
    var op = Tensor.alloc(B * 2)
    var om = Tensor.alloc(B * 2)
    var maxd = Scalar[DT](0)
    for idx in range(N):
        var saved = inp[1].data[idx]
        inp[1].data[idx] = saved + eps
        m.forward["cpu", B](TensorRefs[2](inp[0], inp[1]), op)
        inp[1].data[idx] = saved - eps
        m.forward["cpu", B](TensorRefs[2](inp[0], inp[1]), om)
        inp[1].data[idx] = saved
        # dyn cotangent → Σ dyn channel.
        var fd = (_scalar_loss[0](op) - _scalar_loss[0](om)) / (
            Scalar[DT](2) * eps
        )
        var d = abs(fd - gpk[1].data[idx])
        if d > maxd:
            maxd = d
    print("  max |fd - grad_prior| =", maxd)
    assert_true(maxd < Scalar[DT](1e-2), "PRIOR branch FD")
    print("  ok")


def test_cpu_gpu_parity() raises:
    print("CPU/GPU parity (forward + vjp) ...")
    var c = DeviceContext()
    var inp = TensorPack[2]()
    _fill(inp)
    var go = Tensor.alloc(B * 2)
    for b in range(B):
        go.data[b * 2] = Scalar[DT](0.3 + 0.1 * Float64(b))
        go.data[b * 2 + 1] = Scalar[DT](0.7 - 0.05 * Float64(b))

    # CPU reference.
    var mc = OneHotKLLoss[STOCH, CLASSES].make["cpu", INIT=Zero]()
    var out_cpu = Tensor.alloc(B * 2)
    mc.forward["cpu", B](TensorRefs[2](inp[0], inp[1]), out_cpu)
    var gcpu = TensorPack[2]()
    mc.vjp["cpu", B](
        TensorRefs[2](inp[0], inp[1]), go, TensorRefs[2](gcpu[0], gcpu[1])
    )

    # GPU.
    var mg = OneHotKLLoss[STOCH, CLASSES].make["gpu", INIT=Zero](ctx=c)
    inp[0].upload(c); inp[1].upload(c); go.upload(c)
    var out_g = Tensor.alloc_gpu(c, B * 2)
    mg.forward["gpu", B](TensorRefs[2](inp[0], inp[1]), out_g, ctx=c)
    var gg = TensorPack[2]()
    mg.vjp["gpu", B](
        TensorRefs[2](inp[0], inp[1]), go, TensorRefs[2](gg[0], gg[1]), ctx=c
    )
    out_g.download(c); gg[0].download(c); gg[1].download(c)

    var maxd = Scalar[DT](0)
    for i in range(B * 2):
        var d = abs(out_g.data[i] - out_cpu.data[i])
        if d > maxd: maxd = d
    for i in range(N):
        var dp = abs(gg[0].data[i] - gcpu[0].data[i])
        if dp > maxd: maxd = dp
        var dq = abs(gg[1].data[i] - gcpu[1].data[i])
        if dq > maxd: maxd = dq
    print("  max CPU/GPU Δ =", maxd)
    assert_true(maxd < Scalar[DT](1e-4), "CPU/GPU parity")
    print("  ok")


def main() raises:
    print("=" * 60)
    print("OneHotKLLoss storage Module")
    print("=" * 60)
    test_fd_post_branch()
    test_fd_prior_branch()
    test_cpu_gpu_parity()
    print("ALL PASSED")

"""Reusable two-hot CE + decode helpers (storage loss/two_hot.mojo).

Validates the fused two-hot soft-cross-entropy (forward + backward) and the
symexp decode (forward + backward) that TD-MPC2 (and, on migration, DreamerV3)
share as ComputeGraph nodes. The graph-node helpers take their operands as a
`TensorRefs` (one shared origin) so a node can pass pooled inputs + write a
pooled output without a §B0 ref/mut aliasing clash; the test mirrors that by
backing the operands with a `TensorPack`.

  1. CE backward vs central finite differences of the CE forward (CPU).
  2. decode backward vs central finite differences of decode forward (CPU).
  3. CPU vs GPU parity for all four (CE fwd/bwd, decode fwd/bwd).

Run:
  pixi run mojo run -I . tests/nn/test_two_hot_ce_storage.mojo
  pixi run -e apple mojo run -I . tests/nn/test_two_hot_ce_storage.mojo
"""

from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_pack import TensorPack
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.loss.two_hot import (
    fill_bins,
    two_hot_ce_loss_batch,
    two_hot_ce_backward_batch,
    two_hot_decode_batch,
    decode_value_backward_batch,
    two_hot_ce_fwd_kernel,
    two_hot_ce_bwd_kernel,
    decode_value_fwd_kernel,
    decode_value_bwd_kernel,
)


comptime B = 4
comptime BINS = 7
comptime VMIN = -5
comptime VMAX = 5


def _fill_ce(mut inp: TensorPack[2]) raises:
    inp[0].ensure(B * BINS)
    inp[1].ensure(B)
    for i in range(B * BINS):
        inp[0].data[i] = Scalar[DT]((i % 5) - 2) * 0.4
    for b in range(B):
        inp[1].data[b] = Scalar[DT]((b % 4) - 1) * 1.7


def test_ce_backward_fd() raises:
    print("two-hot CE backward vs finite-diff (CPU) ...")
    var inp = TensorPack[2]()
    _fill_ce(inp)
    var bins = Tensor.alloc(BINS)
    fill_bins[BINS](Scalar[DT](VMIN), Scalar[DT](VMAX), bins)
    var go = Tensor.alloc(B)
    for b in range(B):
        go.data[b] = Scalar[DT](1.0)
    var gpk = TensorPack[2]()
    two_hot_ce_backward_batch[B, BINS, True](
        TensorRefs[2](inp[0], inp[1]), bins, go, TensorRefs[2](gpk[0], gpk[1])
    )

    var eps = Scalar[DT](1e-3)
    var out_p = Tensor.alloc(B)
    var out_m = Tensor.alloc(B)
    var maxd = Scalar[DT](0)
    for b in range(B):
        for c in range(BINS):
            var idx = b * BINS + c
            var saved = inp[0].data[idx]
            inp[0].data[idx] = saved + eps
            two_hot_ce_loss_batch[B, BINS, True](
                TensorRefs[2](inp[0], inp[1]), bins, out_p
            )
            inp[0].data[idx] = saved - eps
            two_hot_ce_loss_batch[B, BINS, True](
                TensorRefs[2](inp[0], inp[1]), bins, out_m
            )
            inp[0].data[idx] = saved
            var fd = (out_p.data[b] - out_m.data[b]) / (Scalar[DT](2) * eps)
            var d = abs(fd - gpk[0].data[idx])
            if d > maxd:
                maxd = d
    print("  max |fd - analytic| =", maxd)
    assert_true(maxd < Scalar[DT](2e-3), "CE backward FD")
    print("  ok")


def test_decode_backward_fd() raises:
    print("decode backward vs finite-diff (CPU) ...")
    var inp = TensorPack[1]()
    inp[0].ensure(B * BINS)
    for i in range(B * BINS):
        inp[0].data[i] = Scalar[DT]((i % 5) - 2) * 0.4
    var bins = Tensor.alloc(BINS)
    fill_bins[BINS](Scalar[DT](VMIN), Scalar[DT](VMAX), bins)
    var go = Tensor.alloc(B)
    for b in range(B):
        go.data[b] = Scalar[DT](1.0)
    var gpk = TensorPack[1]()
    decode_value_backward_batch[B, BINS](
        TensorRefs[1](inp[0]), bins, go, TensorRefs[1](gpk[0])
    )

    var eps = Scalar[DT](1e-3)
    var v_p = Tensor.alloc(B)
    var v_m = Tensor.alloc(B)
    var maxd = Scalar[DT](0)
    for b in range(B):
        for c in range(BINS):
            var idx = b * BINS + c
            var saved = inp[0].data[idx]
            inp[0].data[idx] = saved + eps
            two_hot_decode_batch[B, BINS](TensorRefs[1](inp[0]), bins, v_p)
            inp[0].data[idx] = saved - eps
            two_hot_decode_batch[B, BINS](TensorRefs[1](inp[0]), bins, v_m)
            inp[0].data[idx] = saved
            var fd = (v_p.data[b] - v_m.data[b]) / (Scalar[DT](2) * eps)
            var d = abs(fd - gpk[0].data[idx])
            if d > maxd:
                maxd = d
    print("  max |fd - analytic| =", maxd)
    assert_true(maxd < Scalar[DT](5e-3), "decode backward FD")
    print("  ok")


def test_cpu_gpu_parity() raises:
    print("CPU/GPU parity (CE fwd/bwd, decode fwd/bwd) ...")
    var c = DeviceContext()
    var inp = TensorPack[2]()
    _fill_ce(inp)
    var bins = Tensor.alloc(BINS)
    fill_bins[BINS](Scalar[DT](VMIN), Scalar[DT](VMAX), bins)
    var go = Tensor.alloc(B)
    for b in range(B):
        go.data[b] = Scalar[DT](0.3 + 0.1 * Float64(b))

    # CPU references (TensorRefs over the pack).
    var ce_cpu = Tensor.alloc(B)
    two_hot_ce_loss_batch[B, BINS, True](
        TensorRefs[2](inp[0], inp[1]), bins, ce_cpu
    )
    var gce = TensorPack[2]()
    two_hot_ce_backward_batch[B, BINS, True](
        TensorRefs[2](inp[0], inp[1]), bins, go, TensorRefs[2](gce[0], gce[1])
    )
    var dec_cpu = Tensor.alloc(B)
    two_hot_decode_batch[B, BINS](TensorRefs[1](inp[0]), bins, dec_cpu)
    var gdec = TensorPack[1]()
    decode_value_backward_batch[B, BINS](
        TensorRefs[1](inp[0]), bins, go, TensorRefs[1](gdec[0])
    )

    # GPU (kernels take plain LayoutTensor views).
    inp[0].upload(c); inp[1].upload(c); bins.upload(c); go.upload(c)
    comptime nb = (B + TPB - 1) // TPB
    var ce_g = Tensor.alloc_gpu(c, B)
    c.enqueue_function[two_hot_ce_fwd_kernel[B, BINS, True]](
        inp[0].tt["gpu", B * BINS](),
        inp[1].tt["gpu", B](),
        bins.tt["gpu", BINS](),
        ce_g.tt["gpu", B](),
        grid_dim=nb, block_dim=TPB,
    )
    var gce_g = Tensor.alloc_gpu(c, B * BINS)
    c.enqueue_function[two_hot_ce_bwd_kernel[B, BINS, True]](
        go.tt["gpu", B](),
        inp[0].tt["gpu", B * BINS](),
        inp[1].tt["gpu", B](),
        bins.tt["gpu", BINS](),
        gce_g.tt["gpu", B * BINS](),
        grid_dim=nb, block_dim=TPB,
    )
    var dec_g = Tensor.alloc_gpu(c, B)
    c.enqueue_function[decode_value_fwd_kernel[B, BINS]](
        inp[0].tt["gpu", B * BINS](),
        bins.tt["gpu", BINS](),
        dec_g.tt["gpu", B](),
        grid_dim=nb, block_dim=TPB,
    )
    var gdec_g = Tensor.alloc_gpu(c, B * BINS)
    c.enqueue_function[decode_value_bwd_kernel[B, BINS]](
        go.tt["gpu", B](),
        inp[0].tt["gpu", B * BINS](),
        bins.tt["gpu", BINS](),
        gdec_g.tt["gpu", B * BINS](),
        grid_dim=nb, block_dim=TPB,
    )
    ce_g.download(c); gce_g.download(c); dec_g.download(c); gdec_g.download(c)

    var m = Scalar[DT](0)
    for b in range(B):
        if abs(ce_g.data[b] - ce_cpu.data[b]) > m: m = abs(ce_g.data[b] - ce_cpu.data[b])
        if abs(dec_g.data[b] - dec_cpu.data[b]) > m: m = abs(dec_g.data[b] - dec_cpu.data[b])
    for i in range(B * BINS):
        if abs(gce_g.data[i] - gce[0].data[i]) > m: m = abs(gce_g.data[i] - gce[0].data[i])
        if abs(gdec_g.data[i] - gdec[0].data[i]) > m: m = abs(gdec_g.data[i] - gdec[0].data[i])
    print("  max CPU/GPU Δ =", m)
    assert_true(m < Scalar[DT](2e-5), "CPU/GPU parity")
    print("  ok")


def main() raises:
    print("=" * 60)
    print("Reusable two-hot CE + decode helpers")
    print("=" * 60)
    test_ce_backward_fd()
    test_decode_backward_fd()
    test_cpu_gpu_parity()
    print("ALL PASSED")

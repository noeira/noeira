"""PyTorch / cuDNN twin of `benchmarks/bench_conv2d_paths_gpu.mojo`: one
`nn.Conv2d` (bias on), batch 100, fp32 storage with TF32 convolutions,
`cudnn.benchmark` on. Per shape, the GPU time of ONE forward, and of one
forward + backward (input, weight and bias grads), inside a captured CUDA
graph of R back-to-back calls, averaged over NREP replays.

    python tools/nn/bench_conv_torch.py
"""

import time
import torch

R, NREP = 20, 20
SHAPES = [  # name, IC, OC, K, S, P, H, W, B, NHWC — the shapes of benchmarks/bench_conv2d_paths_gpu.mojo
    ("resnet_stem", 3, 16, 3, 1, 1, 32, 32, 100, False),
    ("resnet_s1", 16, 16, 3, 1, 1, 32, 32, 100, False),
    ("resnet_down2", 16, 32, 3, 2, 1, 32, 32, 100, False),
    ("resnet_short2", 16, 32, 1, 2, 0, 32, 32, 100, False),
    ("resnet_s2", 32, 32, 3, 1, 1, 16, 16, 100, False),
    ("resnet_down3", 32, 64, 3, 2, 1, 16, 16, 100, False),
    ("resnet_short3", 32, 64, 1, 2, 0, 16, 16, 100, False),
    ("resnet_s3", 64, 64, 3, 1, 1, 8, 8, 100, False),
    ("cnn_c1", 3, 32, 3, 1, 1, 32, 32, 100, False),
    ("cnn_c2", 32, 32, 3, 1, 1, 32, 32, 100, False),
    ("cnn_c3", 32, 64, 3, 1, 1, 16, 16, 100, False),
    ("cnn_c4", 64, 64, 3, 1, 1, 16, 16, 100, False),
    ("cnn_c5", 64, 128, 3, 1, 1, 8, 8, 100, False),
    ("cnn_c6", 128, 128, 3, 1, 1, 8, 8, 100, False),
    ("nature_c1_b32", 4, 32, 8, 4, 0, 84, 84, 32, False),
    ("nature_c1_b256", 4, 32, 8, 4, 0, 84, 84, 256, False),
    ("nature_c1_b64", 4, 32, 8, 4, 0, 84, 84, 64, False),
    ("nature_c2_b32", 32, 64, 4, 2, 0, 20, 20, 32, False),
    ("nature_c2_b256", 32, 64, 4, 2, 0, 20, 20, 256, False),
    ("nature_c3_b32", 64, 64, 3, 1, 0, 9, 9, 32, False),
    ("nature_c3_b256", 64, 64, 3, 1, 0, 9, 9, 256, False),
    ("nature_c1_nhwc", 4, 32, 8, 4, 0, 84, 84, 64, True),
    ("ez_stem", 12, 32, 3, 2, 1, 96, 96, 256, True),
    ("ez_48", 32, 32, 3, 1, 1, 48, 48, 256, True),
    ("ez_down48", 32, 64, 3, 2, 1, 48, 48, 256, True),
    ("ez_24", 64, 64, 3, 1, 1, 24, 24, 256, True),
    ("ez_12", 64, 64, 3, 1, 1, 12, 12, 256, True),
    ("ez_6_nhwc", 64, 64, 3, 1, 1, 6, 6, 256, True),
    ("ez_6", 64, 64, 3, 1, 1, 6, 6, 256, False),
    ("ez_dyn80", 80, 64, 3, 1, 1, 6, 6, 256, False),
    ("ez_act1x1", 1, 16, 1, 1, 0, 6, 6, 256, False),
    ("ez_head1x1", 64, 16, 1, 1, 0, 6, 6, 256, False),
    ("ez_6_act4", 64, 64, 3, 1, 1, 6, 6, 4, False),
    ("c4_stem64", 3, 64, 3, 1, 1, 6, 7, 128, False),
    ("c4_res64", 64, 64, 3, 1, 1, 6, 7, 128, False),
    ("c4_res64_b64", 64, 64, 3, 1, 1, 6, 7, 64, False),
    ("c4_dyn80", 80, 64, 3, 1, 1, 6, 7, 128, False),
    ("c4_head1x1", 64, 16, 1, 1, 0, 6, 7, 128, False),
    ("az_stem", 3, 128, 3, 1, 1, 6, 7, 128, False),
    ("az_res", 128, 128, 3, 1, 1, 6, 7, 128, False),
    ("az_res_b64", 128, 128, 3, 1, 1, 6, 7, 64, False),
    ("pix_c1", 12, 32, 3, 1, 1, 16, 16, 1024, False),
    ("pix_c2", 32, 64, 3, 2, 1, 16, 16, 1024, False),
    ("pix_c3", 64, 64, 3, 2, 1, 8, 8, 1024, False),
    ("act_stem", 3, 64, 7, 2, 3, 240, 320, 32, False),
    ("act_l1", 64, 64, 3, 1, 1, 60, 80, 32, False),
    ("act_down2", 64, 128, 3, 2, 1, 60, 80, 32, False),
    ("act_short2", 64, 128, 1, 2, 0, 60, 80, 32, False),
    ("act_l2", 128, 128, 3, 1, 1, 30, 40, 32, False),
    ("act_down3", 128, 256, 3, 2, 1, 30, 40, 32, False),
    ("act_short3", 128, 256, 1, 2, 0, 30, 40, 32, False),
    ("act_l3", 256, 256, 3, 1, 1, 15, 20, 32, False),
    ("act_down4", 256, 512, 3, 2, 1, 15, 20, 32, False),
    ("act_short4", 256, 512, 1, 2, 0, 15, 20, 32, False),
    ("act_l4", 512, 512, 3, 1, 1, 8, 10, 32, False),
    ("act_stem_deploy", 3, 64, 7, 2, 3, 240, 320, 2, False),
    ("act_l2_deploy", 128, 128, 3, 1, 1, 30, 40, 2, False),
    ("libero_stem", 3, 64, 7, 2, 3, 128, 128, 32, False),
    ("libero_l1", 64, 64, 3, 1, 1, 32, 32, 32, False),
    ("libero_l2", 128, 128, 3, 1, 1, 16, 16, 32, False),
    ("libero_l3", 256, 256, 3, 1, 1, 8, 8, 32, False),
    ("mnist_c1", 1, 16, 5, 2, 0, 28, 28, 100, False),
    ("mnist_c2", 16, 32, 5, 2, 0, 12, 12, 100, False),
    ("vit_patch", 3, 192, 4, 4, 0, 32, 32, 128, False),
    ("lewm_patch", 3, 192, 14, 14, 0, 224, 224, 512, False),
    ("dv3_enc1", 1, 64, 5, 1, 2, 96, 96, 16, False),
    ("dv3_enc2", 64, 96, 5, 1, 2, 48, 48, 16, False),
    ("dv3_enc3", 96, 128, 5, 1, 2, 24, 24, 16, False),
    ("dv3_enc4", 128, 128, 5, 1, 2, 12, 12, 16, False),
    ("dv3_dec3", 96, 64, 5, 1, 2, 48, 48, 16, False),
    ("dv3_dec_out", 64, 1, 5, 1, 2, 96, 96, 16, False),
    ("dv3s_enc1", 4, 48, 4, 2, 1, 96, 96, 16, False),
    ("dv3s_enc2", 48, 96, 4, 2, 1, 48, 48, 16, False),
    ("dv3s_enc3", 96, 192, 4, 2, 1, 24, 24, 16, False),
    ("dv3s_enc4", 192, 384, 4, 2, 1, 12, 12, 16, False),
    ("dv3_enc1_act", 1, 64, 5, 1, 2, 96, 96, 1, False),
    ("d4_stem", 3, 16, 3, 1, 1, 64, 64, 80, False),
    ("d4_s1", 16, 16, 3, 1, 1, 64, 64, 80, False),
    ("d4_down2", 16, 32, 3, 2, 1, 64, 64, 80, False),
    ("d4_s2", 32, 32, 3, 1, 1, 32, 32, 80, False),
]


def time_graph(fn, stream):
    """Warm up, capture R calls and time NREP replays, all on `stream` (the
    tensors were created on it too: autograd syncs with the stream its inputs
    came from, and the legacy default stream cannot depend on a capture)."""
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
        stream.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            for _ in range(R):
                fn()
        for _ in range(3):
            g.replay()
        stream.synchronize()
        t0 = time.perf_counter()
        for _ in range(NREP):
            g.replay()
        stream.synchronize()
    return (time.perf_counter() - t0) * 1e6 / (NREP * R)


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    print("torch", torch.__version__, "cudnn", torch.backends.cudnn.version())
    st = torch.cuda.Stream()
    for name, ic, oc, k, s, p, h, w, B, nhwc in SHAPES:
        fmt = torch.channels_last if nhwc else torch.contiguous_format
        with torch.cuda.stream(st):
            conv = torch.nn.Conv2d(ic, oc, k, s, p).to(dev, memory_format=fmt)
            x = torch.randn(B, ic, h, w, device=dev).to(memory_format=fmt).requires_grad_()
            with torch.no_grad():
                go = torch.randn_like(conv(x))

        def fwd():
            with torch.no_grad():
                conv(x)

        params = [x, conv.weight, conv.bias]

        def fwdbwd():
            # autograd.grad (input, weight and bias grads) instead of
            # .backward(): accumulating into .grad inside a capture makes the
            # legacy stream depend on the capturing one.
            out = conv(x)
            torch.autograd.grad(out, params, go)

        f = time_graph(fwd, st)
        fb = time_graph(fwdbwd, st)
        print(f"ROW path=torch name={name} ic={ic} oc={oc} k={k} s={s} p={p} h={h} w={w} b={B} nhwc={nhwc} fwd_us={f:.3f} fwdbwd_us={fb:.3f}")


if __name__ == "__main__":
    main()

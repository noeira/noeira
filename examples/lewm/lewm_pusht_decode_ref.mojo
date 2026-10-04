"""What a trained LeWM sees and imagines — decoded predictions as PNGs.

docs/LEWM_REOPEN_PLAN.md P6. LeWM has no decoder (the paper trains one only to
visualise, on the FROZEN encoder: "Decoder (Visualization Only)"). This:

1. loads a checkpoint (`--dump`: one of the training run's `epoch_<k>/`, or
   the published weights' converted dump) into the reference encoder and the
   planning rollout (`ref_rollout`);
2. trains the reconstruction probe (`decoder.mojo`: learned per-patch queries
   reading the 192-d embedding) for `--dec-steps` on random dataset frames,
   the encoder frozen and in eval mode;
3. for `--windows` validation windows of 6 frames (history 1 + 5 predictions,
   frameskip 5), writes `<out>/pred_w<k>.png`, three rows of 6 frames:

     real            the dataset frames t = 0 .. 5
     decoded real    decoder(encoder(frame t))  — what the embedding keeps
     decoded imagined decoder(predictor rollout from frame 0 with the
                     recorded actions)  — what the model expects to see

   and prints, per step, ‖predicted − encoded‖ / ‖encoded‖ in latent space.

    pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_decode_ref.mojo \\
        --dump /workspace/lewm_train/epoch_7 --out /workspace/lewm_viz/epoch_7
"""

from std.sys import argv
from std.math import sqrt, isnan
from std.os import makedirs
from std.random import seed, random_ui64
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.nn.datasets.lewm_pusht import LewmPushTExpert
from noeira.deep_agents.act.refload import RefDump
from noeira.experimental.lewm.ref_load import load_ref
from noeira.experimental.lewm.ref_rollout import (
    LeWMRefRollout, RefEncoder, encode_ref, REF_EMB, REF_ACT,
)
from noeira.experimental.lewm.decoder import patchify, unpatchify
from noeira.experimental.lewm.decoder_trainer import LeWMDecoderTrainer
from noeira.io.png import save_png


comptime IMG = 224
comptime HW = IMG * IMG
comptime IMG_DIM = 3 * HW
comptime NB = 64                    # frames per decoder batch
comptime WIN = 6                    # frames per visualised window
comptime HORIZON = WIN - 1
comptime PATCH_D = 16
comptime N_Q = (IMG // PATCH_D) * (IMG // PATCH_D)
comptime PATCH_PX = 3 * PATCH_D * PATCH_D
comptime Decoder = LeWMDecoderTrainer[REF_EMB, 192, N_Q, PATCH_PX, 768, 4, NB, "gpu"]
comptime GAP = 4


def _p(b: DeviceBuffer[DT]) -> Pointer[Scalar[DT], MutAnyOrigin]:
    return rebind[Pointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _frames(
    u8: List[UInt8], n: Int, mut enc_in: List[Scalar[DT]], mut img01: List[Scalar[DT]]
):
    """n HWC uint8 frames -> ImageNet-normalised CHW (encoder) and [0, 1]
    CHW (decoder target / display)."""
    var mean: List[Float32] = [0.485, 0.456, 0.406]
    var std: List[Float32] = [0.229, 0.224, 0.225]
    for f in range(n):
        for p in range(HW):
            for c in range(3):
                var x = Float32(u8[(f * HW + p) * 3 + c]) / Float32(255.0)
                img01[f * IMG_DIM + c * HW + p] = rebind[Scalar[DT]](x)
                enc_in[f * IMG_DIM + c * HW + p] = rebind[Scalar[DT]]((x - mean[c]) / std[c])


def _decode(
    mut dec: Decoder, embs: List[Scalar[DT]], mut emb_dev: DeviceBuffer[DT],
    c: DeviceContext,
) raises -> List[Scalar[DT]]:
    """NB embeddings -> NB images, CHW [0, 1]-ish (host)."""
    with emb_dev.map_to_host() as h:
        for i in range(NB * REF_EMB):
            h[i] = embs[i] if i < len(embs) else Scalar[DT](0)
    var recon = List[Scalar[DT]](length=NB * N_Q * PATCH_PX, fill=Scalar[DT](0))
    dec.recon_into(
        TileTensor(_p(emb_dev), row_major[NB, REF_EMB]()),
        rebind[Pointer[Scalar[DT], MutAnyOrigin]](recon.unsafe_ptr()),
    )
    var img = List[Scalar[DT]](length=NB * IMG_DIM, fill=Scalar[DT](0))
    unpatchify["cpu", NB, 3, IMG, PATCH_D](
        None,
        rebind[Pointer[Scalar[DT], MutAnyOrigin]](recon.unsafe_ptr()),
        rebind[Pointer[Scalar[DT], MutAnyOrigin]](img.unsafe_ptr()),
    )
    return img^


def _put(
    mut canvas: List[UInt8], cw: Int, row: Int, col: Int,
    src: List[Scalar[DT]], off: Int,
):
    """One CHW [0, 1] image into tile (row, col) of an HWC uint8 canvas."""
    var x0 = col * (IMG + GAP)
    var y0 = row * (IMG + GAP)
    for y in range(IMG):
        for x in range(IMG):
            for c in range(3):
                var v = Float64(src[off + c * HW + y * IMG + x])
                v = min(1.0, max(0.0, v))
                canvas[((y0 + y) * cw + x0 + x) * 3 + c] = UInt8(Int(v * 255.0 + 0.5))


def main() raises:
    var dump = String("/workspace/lewm_train/epoch_7")
    var h5 = String("/workspace/lewm_session_a/stablewm/pusht_expert_train.h5")
    var split = String("/workspace/lewm_split")
    var stats_dir = String("/workspace/lewm_parity128")
    var out_dir = String("/workspace/lewm_viz")
    var dec_steps = 4000
    var n_win = 8
    var args = argv()
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--dump":
            dump = String(args[i + 1]); i += 1
        elif a == "--h5":
            h5 = String(args[i + 1]); i += 1
        elif a == "--split":
            split = String(args[i + 1]); i += 1
        elif a == "--stats":
            stats_dir = String(args[i + 1]); i += 1
        elif a == "--out":
            out_dir = String(args[i + 1]); i += 1
        elif a == "--dec-steps":
            dec_steps = Int(String(args[i + 1])); i += 1
        elif a == "--windows":
            n_win = Int(String(args[i + 1])); i += 1
        else:
            raise Error("unknown argument " + a)
        i += 1
    makedirs(out_dir, exist_ok=True)

    var c = DeviceContext()
    var ctx = Optional(c)
    var enc = RefEncoder.make["gpu", Kaiming](ctx)
    _ = load_ref["gpu"](enc, dump, String("emb.0."), ctx)
    var roll = LeWMRefRollout["gpu", 1, HORIZON](dump, ctx)
    var dec = Decoder.make(lr=Scalar[DT](1e-3), ctx=ctx)
    var st = RefDump(stats_dir)
    var a_mean = st.get(String("run.action_mean"))
    var a_std = st.get(String("run.action_std"))
    print("decoder probe on", dump)

    # ── 1. the decoder, on frozen embeddings of random frames ─────────────
    var frames = LewmPushTExpert(frameskip=1, num_steps=1, path=h5)
    var u8 = List[UInt8](length=NB * IMG_DIM, fill=0)
    var dense = List[UInt8](length=IMG_DIM, fill=0)
    var araw = List[Float32](length=2, fill=0)
    var enc_in = List[Scalar[DT]](length=NB * IMG_DIM, fill=Scalar[DT](0))
    var img01 = List[Scalar[DT]](length=NB * IMG_DIM, fill=Scalar[DT](0))
    var tgt = List[Scalar[DT]](length=NB * N_Q * PATCH_PX, fill=Scalar[DT](0))
    var emb_dev = c.enqueue_create_buffer[DT](NB * REF_EMB)
    var tgt_dev = c.enqueue_create_buffer[DT](NB * N_Q * PATCH_PX)
    seed(7)
    var t0 = perf_counter_ns()
    dec.reset_loss_accum()
    for step in range(1, dec_steps + 1):
        for f in range(NB):
            frames.sample_clip_pixels_uint8(
                Int(random_ui64(0, UInt64(len(frames) - 1))),
                rebind[Pointer[Scalar[DType.uint8], MutAnyOrigin]](u8.unsafe_ptr() + f * IMG_DIM),
                rebind[Pointer[Scalar[DType.float32], MutAnyOrigin]](araw.unsafe_ptr()),
                rebind[Pointer[Scalar[DType.uint8], MutAnyOrigin]](dense.unsafe_ptr()),
            )
        _frames(u8, NB, enc_in, img01)
        var emb = encode_ref["gpu", NB](enc, enc_in, ctx)
        patchify["cpu", NB, 3, IMG, PATCH_D](
            None,
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](img01.unsafe_ptr()),
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](tgt.unsafe_ptr()),
        )
        with emb_dev.map_to_host() as h:
            for k in range(NB * REF_EMB):
                h[k] = emb[k]
        with tgt_dev.map_to_host() as h:
            for k in range(NB * N_Q * PATCH_PX):
                h[k] = tgt[k]
        _ = dec.train_step(
            TileTensor(_p(emb_dev), row_major[NB, REF_EMB]()),
            TileTensor(_p(tgt_dev), row_major[NB, N_Q * PATCH_PX]()),
        )
        if step % 250 == 0 or step == dec_steps:
            var ml = dec.read_loss_accum()
            dec.reset_loss_accum()
            print("  decoder step", step, "/", dec_steps, " recon mse", ml, " ",
                  Float32(Float64(perf_counter_ns() - t0) / 1e9), "s")

    # ── 2. validation windows: real / decoded real / decoded imagined ─────
    var win = LewmPushTExpert(frameskip=5, num_steps=WIN, path=h5)
    var ds4 = LewmPushTExpert(frameskip=5, num_steps=4, path=h5)  # the split's clip list
    var val = RefDump(split).get(String("split.val"))
    var wu8 = List[UInt8](length=WIN * IMG_DIM, fill=0)
    var wdense = List[UInt8](length=WIN * 5 * IMG_DIM, fill=0)
    var wact = List[Float32](length=WIN * 10, fill=0)
    var w_in = List[Scalar[DT]](length=NB * IMG_DIM, fill=Scalar[DT](0))
    var w01 = List[Scalar[DT]](length=NB * IMG_DIM, fill=Scalar[DT](0))
    var cw = WIN * (IMG + GAP) - GAP
    var chh = 3 * (IMG + GAP) - GAP
    var err_sum = List[Float64](length=WIN, fill=0.0)
    for k in range(n_win):
        # a held-out clip (4 frames, `split.val`) -> the 6-frame window at
        # the same (episode, start); skip the clips too close to an
        # episode's end for 6 frames
        var w_idx = -1
        var probe = k * 997
        while w_idx < 0:
            var v = Int(val[probe % len(val)])
            var ep = ds4.clip_ep_idx[v]
            var s0 = ds4.clip_start[v]
            for j in range(len(win)):
                if win.clip_ep_idx[j] == ep and win.clip_start[j] == s0:
                    w_idx = j
                    break
            probe += 1
        win.sample_clip_pixels_uint8(
            w_idx,
            rebind[Pointer[Scalar[DType.uint8], MutAnyOrigin]](wu8.unsafe_ptr()),
            rebind[Pointer[Scalar[DType.float32], MutAnyOrigin]](wact.unsafe_ptr()),
            rebind[Pointer[Scalar[DType.uint8], MutAnyOrigin]](wdense.unsafe_ptr()),
        )
        _frames(wu8, WIN, w_in, w01)
        var e_all = encode_ref["gpu", NB](enc, w_in, ctx)  # rows >= WIN are padding
        var start = List[Scalar[DT]](capacity=REF_EMB)
        for d in range(REF_EMB):
            start.append(e_all[d])
        var acts = List[Scalar[DT]](capacity=HORIZON * REF_ACT)
        for j in range(HORIZON * REF_ACT):
            var z = (wact[j] - Float32(a_mean[j % 2])) / Float32(a_std[j % 2])
            acts.append(Scalar[DT](0) if isnan(z) else rebind[Scalar[DT]](z))
        var pred = roll.rollout(start, acts)  # (1, WIN, D), entry 0 = start
        var both = List[Scalar[DT]](capacity=NB * REF_EMB)
        for j in range(WIN * REF_EMB):
            both.append(e_all[j])
        for j in range(WIN * REF_EMB):
            both.append(pred[j])
        var dimg = _decode(dec, both, emb_dev, c)
        var canvas = List[UInt8](length=cw * chh * 3, fill=255)
        var line = String("  window ") + String(k) + " (clip " + String(w_idx) + ")  ‖pred-enc‖/‖enc‖:"
        for t in range(WIN):
            _put(canvas, cw, 0, t, w01, t * IMG_DIM)
            _put(canvas, cw, 1, t, dimg, t * IMG_DIM)
            _put(canvas, cw, 2, t, dimg, (WIN + t) * IMG_DIM)
            var num = 0.0
            var den = 0.0
            for d in range(REF_EMB):
                var r = Float64(e_all[t * REF_EMB + d])
                num += (Float64(pred[t * REF_EMB + d]) - r) ** 2
                den += r * r
            var e = sqrt(num / den)
            err_sum[t] += e
            line += " " + String(Float32(e))
        print(line)
        save_png(out_dir + "/pred_w" + String(k) + ".png", canvas, cw, chh, 3)
    var mean_line = String("  mean over windows, steps 0..5:")
    for t in range(WIN):
        mean_line += " " + String(Float32(err_sum[t] / Float64(n_win)))
    print(mean_line)
    print("wrote", n_win, "PNGs to", out_dir, "(rows: real / decoded real / decoded imagined)")

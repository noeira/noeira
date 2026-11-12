"""`noeira.perception.rfdetr.RfDetr` against Roboflow's `RFDETRSegSmall` (CPU,
fp32), stage by stage, on a real photo (NYU labelled 921, 640 x 480).

    pixi run -e detect ~/.cache/noeira/venvs/detcand/bin/python tools/perception/dump_rfdetr_reference.py
    pixi run -e detect python -I tools/perception/extract_rfdetr.py
    pixi run mojo run -I . tests/perception/test_rfdetr_reference_gpu.mojo [ref dir]

Gates (relative error = max |mine - ref| / max |ref|): the pixels, the windowed
embeddings, every backbone block, the four taps, the projector's output, the
two-stage logits, the reference boxes, `query_pos`, every decoder layer, the
normalised decoder outputs, the logits, boxes and mask logits; then the
detections themselves (same labels in the same order, scores, boxes, mask
IoU). Three DEFECTS must move the projector's output or the logits: no global
attention, the deformable offsets' x / y swapped, the C2f bottlenecks with a
residual.
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.ai.picture import Picture
from noeira.io.npy import load_npy_f64
from noeira.nn.constants import DT
from noeira.perception.rfdetr import RfDetr
from noeira.perception.sam3.ops import SAM3_TF32


comptime BOUND: Float64 = 1e-4
comptime STRICT: Bool = not SAM3_TF32
"""fp32: every stage gated at BOUND. TF32 / bf16 (`-D SAM3_TF32=1` / `SAM3_BF16=1`):
the stages are PRINTED (their drift), not gated — a near-tie in the two-stage
top-100 swaps query slots, so a per-slot comparison measures the slot order,
not the answer — and the DETECTIONS are gated: the same labels in the same
order, score |d| <= 0.02, box |d| <= 2 px, mask IoU >= 0.99."""


def check(mut fails: Int, ok: Bool, what: String, gated: Bool = True):
    if not gated:
        print("  (drift)", what)
        return
    if ok:
        print("  PASS ", what)
    else:
        print("  FAIL ", what)
        fails += 1


def down(ctx: DeviceContext, x: DeviceBuffer[DT], n: Int) raises -> List[Float64]:
    var h = ctx.enqueue_create_host_buffer[DT](n)
    ctx.enqueue_copy(h, x.create_sub_buffer[DT](0, n))
    ctx.synchronize()
    var out = List[Float64](capacity=n)
    for i in range(n):
        out.append(Float64(h[i]))
    return out^


def rel(a: List[Float64], b: List[Float64]) -> Float64:
    var num = 0.0
    var den = 0.0
    for i in range(min(len(a), len(b))):
        num = max(num, abs(a[i] - b[i]))
        den = max(den, abs(b[i]))
    if len(a) != len(b):
        return 1e9
    return num / max(den, 1e-30)


def f64(v: List[Float32]) -> List[Float64]:
    var o = List[Float64](capacity=len(v))
    for x in v:
        o.append(Float64(x))
    return o^


def chw_to_tm(v: List[Float64], c: Int, n: Int) -> List[Float64]:
    """[c, n] -> [n, c] (a reference feature map to the port's token-major)."""
    var o = List[Float64](length=c * n, fill=0.0)
    for ch in range(c):
        for i in range(n):
            o[i * c + ch] = v[ch * n + i]
    return o^


def taps_to_tm(v: List[Float64]) -> List[Float64]:
    """bb_taps [4, 384, 32 x 32] -> [1024, 4 x 384]."""
    var o = List[Float64](length=4 * 384 * 1024, fill=0.0)
    for t in range(4):
        for ch in range(384):
            for i in range(1024):
                o[i * 1536 + t * 384 + ch] = v[(t * 384 + ch) * 1024 + i]
    return o^


def gate(mut fails: Int, what: String, mine: List[Float64], ref_v: List[Float64], bound: Float64 = BOUND) -> Float64:
    var e = rel(mine, ref_v)
    check(fails, e <= bound, what + "  rel err = " + String(e), STRICT)
    return e


def main() raises:
    var fails = 0
    var args = argv()
    var R = String(args[1]) if len(args) > 1 else getenv("HOME") + "/.cache/noeira/rfdetr/ref"
    var I = getenv("HOME") + "/.cache/noeira/sam3/ref/image.npy"
    print("RF-DETR-Seg-Small vs rfdetr (CPU fp32), NYU frame 921")
    var t0 = perf_counter_ns()
    var det = RfDetr(DeviceContext())
    ref ctx = det.wts.ctx
    print("  weights loaded in", (perf_counter_ns() - t0) // 1_000_000, "ms")
    var im = load_npy_f64(I)
    var rgb = List[UInt8](capacity=len(im))
    for v in im:
        rgb.append(UInt8(Int(v)))
    var pic = Picture(rgb^, 480, 640)

    var px = det.pixel_values(pic)
    _ = gate(fails, "pixels (stretch to 384, normalise)", down(ctx, px, 3 * 384 * 384), load_npy_f64(R + "/pixel_values.npy"), 1e-6)
    var keep = List[DeviceBuffer[DT]]()
    var t1 = perf_counter_ns()
    var src = det.backbone(px, keep, True)
    ctx.synchronize()
    print("  backbone + projector (with taps):", (perf_counter_ns() - t1) // 1_000_000, "ms")
    _ = gate(fails, "windowed embeddings [4 x 257, 384]", down(ctx, keep[0], 1028 * 384), load_npy_f64(R + "/bb_embed.npy"), 1e-6)
    var worst = 0.0
    var worst_i = -1
    for i in range(12):
        var e = rel(down(ctx, keep[1 + i], 1028 * 384), load_npy_f64(R + "/bb_layer" + String(i) + ".npy"))
        if e > worst:
            worst = e
            worst_i = i
    check(fails, worst <= BOUND, String("backbone blocks, worst ") + String(worst_i) + "  rel err = " + String(worst), STRICT)
    _ = gate(fails, "four taps [1024, 1536]", down(ctx, keep[13], 1024 * 1536), taps_to_tm(load_npy_f64(R + "/bb_taps.npy")))
    var src_ref = chw_to_tm(load_npy_f64(R + "/src.npy"), 256, 1024)
    _ = gate(fails, "projector output (memory) [1024, 256]", down(ctx, src, 1024 * 256), src_ref)

    var t2 = perf_counter_ns()
    var raw = det.decode(src)
    ctx.synchronize()
    print("  two-stage + decoder + heads + masks:", (perf_counter_ns() - t2) // 1_000_000, "ms")
    _ = gate(fails, "two-stage logits [1024, 91]", f64(det.enc_logits), load_npy_f64(R + "/enc_cls.npy"))
    _ = gate(fails, "reference boxes [100, 4] (top-100 order included)", f64(det.dec_ref), load_npy_f64(R + "/dec_ref.npy"))
    _ = gate(fails, "query_pos [100, 256]", down(ctx, det.query_pos, 100 * 256), load_npy_f64(R + "/query_pos.npy"))
    for l in range(4):
        _ = gate(fails, String("decoder layer ") + String(l), down(ctx, det.dec_layers[l], 100 * 256), load_npy_f64(R + "/dec_layer" + String(l) + ".npy"))
    var hs_ref = load_npy_f64(R + "/hs.npy")
    var hs_mine = List[Float64]()
    for l in range(4):
        hs_mine.extend(down(ctx, raw.hs[l], 100 * 256))
    _ = gate(fails, "normalised decoder outputs [4, 100, 256]", hs_mine, hs_ref)
    _ = gate(fails, "logits [100, 91]", f64(raw.logits), load_npy_f64(R + "/pred_logits.npy"))
    # the boxes are the end of the cascade through an exp() of the deltas: 3e-4 (8.7e-5 on the M1 and the
    # Orin with host preprocessing, 1.15e-4 once the pixels went exact on the device)
    _ = gate(fails, "boxes [100, 4]", f64(raw.boxes), load_npy_f64(R + "/pred_boxes.npy"), 3e-4)
    _ = gate(fails, "mask logits [100, 96 x 96]", down(ctx, raw.masks, 100 * 96 * 96), load_npy_f64(R + "/pred_masks.npy"))

    var found = det.post_process(raw, 480, 640, Float32(0.3))
    var rs = load_npy_f64(R + "/post_scores.npy")
    var rl = load_npy_f64(R + "/post_labels.npy")
    var rb = load_npy_f64(R + "/post_boxes.npy")
    var rm = load_npy_f64(R + "/post_masks.npy")
    var same = len(found) == len(rs)
    var ds = 0.0
    var db = 0.0
    var worst_iou = 1.0
    for k in range(min(len(found), len(rs))):
        same = same and found[k].label == Int(rl[k])
        ds = max(ds, abs(Float64(found[k].score) - rs[k]))
        for c in range(4):
            db = max(db, abs(Float64(found[k].box[c]) - rb[k * 4 + c]))
        var inter = 0
        var uni = 0
        for p in range(480 * 640):
            var a = found[k].mask[p] != 0
            var b = rm[k * 480 * 640 + p] != 0.0
            inter += 1 if (a and b) else 0
            uni += 1 if (a or b) else 0
        worst_iou = min(worst_iou, Float64(inter) / Float64(max(uni, 1)))
    var labels = String("")
    for k in range(len(found)):
        labels += String(found[k].label) + ":" + String(found[k].score) + " "
    comptime S_B = 1e-4 if STRICT else 0.02
    comptime B_B = 0.05 if STRICT else 2.0
    comptime I_B = 0.999 if STRICT else 0.99
    check(fails, same and ds <= S_B and db <= B_B and worst_iou >= I_B,
          String("detections (") + labels + "): labels and order as rfdetr's (" + String(len(rs)) + "), score |d| "
          + String(ds) + ", box |d| " + String(db) + " px, mask IoU >= " + String(worst_iou))
    var unused = det.wts.unused()
    var stray = 0
    for ref n in unused:
        # the two-stage heads of groups 1..12 (group DETR, training only; group 0 is read) and mask_token
        var training_only = n.startswith("transformer.enc_out") or n == "backbone.0.encoder.encoder.embeddings.mask_token"
        if not training_only:
            stray += 1
            print("        unused:", n)
    check(fails, stray == 0, String(len(unused)) + " tensors unused, all of them the training-only two-stage groups 1..12 and mask_token")

    # defects
    det.debug = 1
    var none = List[DeviceBuffer[DT]]()
    var e1 = rel(down(ctx, det.backbone(px, none), 1024 * 256), src_ref)
    check(fails, e1 > 0.05, String("defect caught: no global attention  projector output rel err = ") + String(e1))
    det.debug = 4
    var e4 = rel(down(ctx, det.backbone(px, none), 1024 * 256), src_ref)
    check(fails, e4 > 0.05, String("defect caught: C2f bottlenecks with a residual  rel err = ") + String(e4))
    det.debug = 2
    var raw2 = det.decode(src)
    var e2 = rel(f64(raw2.logits), load_npy_f64(R + "/pred_logits.npy"))
    check(fails, e2 > 0.05, String("defect caught: deformable offsets x / y swapped  logits rel err = ") + String(e2))
    det.debug = 0

    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")

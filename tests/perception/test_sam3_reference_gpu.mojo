"""SAM 3 end to end (`noeira.perception.sam3.Sam3`) against transformers'
`Sam3Processor` + `Sam3Model` + `post_process_instance_segmentation`
(threshold 0.3, mask threshold 0.5, the picture's own size).

    pixi run -e apple mojo run -I . tests/perception/test_sam3_reference_gpu.mojo [ref dir]

From the reference's raw picture (an NYU room, 640 x 480): the processor's
pixel values exactly (torch's uint8 antialiased resize), then per word the
same kept instances — scores within 1e-4, boxes within 0.05 px, each mask's
pixels equal but for a few at the 0.5 cut (fraction <= 1e-4) — and the IoU.
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.ai.picture import Picture
from noeira.io.npy import load_npy_f64
from noeira.nn.constants import DT
from noeira.perception.sam3 import Sam3
from noeira.perception.sam3.ops import SAM3_TF32


comptime SCORE_BOUND: Float64 = 1e-2 if SAM3_TF32 else 1e-4
comptime BOX_BOUND: Float64 = 2.0 if SAM3_TF32 else 0.05
comptime MASK_FLIP: Float64 = 1e-2 if SAM3_TF32 else 1e-4


def check(mut fails: Int, ok: Bool, what: String):
    print("  PASS " if ok else "  FAIL ", what)
    if not ok:
        fails += 1


def main() raises:
    var args = argv()
    var ref_dir = String(args[1]) if len(args) > 1 else getenv("HOME") + "/.cache/noeira/sam3/ref"
    var fails = 0
    print("SAM 3 end to end vs transformers", "(TF32)" if SAM3_TF32 else "(fp32)")
    var t0 = perf_counter_ns()
    var sam = Sam3(DeviceContext())
    print("  loaded in", (perf_counter_ns() - t0) // 1_000_000, "ms")
    var im = load_npy_f64(ref_dir + "/image.npy")
    var rgb = List[UInt8](capacity=len(im))
    for v in im:
        rgb.append(UInt8(Int(v)))
    var pic = Picture(rgb^, 480, 640)
    # the pixel values, exactly
    var pv = sam.pixel_values(pic)
    var hp = sam.vit.wts.ctx.enqueue_create_host_buffer[DT](3 * 1008 * 1008)
    sam.vit.wts.ctx.enqueue_copy(hp, pv)
    sam.vit.wts.ctx.synchronize()
    var want_pv = load_npy_f64(ref_dir + "/pixel_values.npy")
    var diff = 0
    for i in range(len(want_pv)):
        if Float64(hp[i]) != Float64(Float32(want_pv[i])):
            diff += 1
    check(fails, diff == 0, "pixel values: " + String(diff) + " of " + String(len(want_pv)) + " differ")
    var t1 = perf_counter_ns()
    var img = sam.encode_image(pic)
    sam.vit.wts.ctx.synchronize()
    print("  encode_image:", (perf_counter_ns() - t1) // 1_000_000, "ms")
    var words: List[String] = ["door", "table", "sofa", "bed", "a red chair"]
    for w in words:
        var dir = ref_dir + "/words/" + w.replace(" ", "_")
        var t2 = perf_counter_ns()
        var got = sam.segment(img, w)
        var ms = (perf_counter_ns() - t2) // 1_000_000
        var ws = load_npy_f64(dir + "/post_scores.npy")
        var ok = len(got) == len(ws)
        var es = 0.0
        var eb = 0.0
        var worst_flip = 0.0
        var worst_iou = 1.0
        if ok and len(ws) > 0:
            var wb = load_npy_f64(dir + "/post_boxes.npy")
            var wm = load_npy_f64(dir + "/post_masks.npy")
            var npx = 480 * 640
            for k in range(len(ws)):
                es = max(es, abs(Float64(got[k].score) - ws[k]))
                for j in range(4):
                    eb = max(eb, abs(Float64(got[k].box[j]) - wb[k * 4 + j]))
                var flips = 0
                var inter = 0
                var uni = 0
                for i in range(npx):
                    var a = Int(got[k].mask[i])
                    var b = Int(wm[k * npx + i])
                    if a != b:
                        flips += 1
                    if a == 1 and b == 1:
                        inter += 1
                    if a == 1 or b == 1:
                        uni += 1
                worst_flip = max(worst_flip, Float64(flips) / Float64(npx))
                worst_iou = min(worst_iou, Float64(inter) / Float64(uni) if uni > 0 else 1.0)
        var detail = String("")
        if len(got) > 0:
            detail = ", score " + String(got[0].score) + ", box [" + String(Int(got[0].box[0])) + " " + String(Int(got[0].box[1])) + " " + String(Int(got[0].box[2])) + " " + String(Int(got[0].box[3])) + "]"
        check(fails, ok and es <= SCORE_BOUND and eb <= BOX_BOUND and worst_flip <= MASK_FLIP,
              repr(w) + ": " + String(len(got)) + " kept (reference " + String(len(ws)) + ")" + detail + "; score |d| " + String(es)
              + ", box |d| " + String(eb) + " px, mask pixels flipped " + String(worst_flip) + ", IoU " + String(worst_iou)
              + "  (" + String(ms) + " ms)")
    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")

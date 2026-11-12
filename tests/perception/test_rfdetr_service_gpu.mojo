"""`RfDetrService` (`noeira.perception.rfdetr.service`) on a REAL RGB-D frame:
RF-DETR in this process on its own thread, its masks lifted by the frame's
registered depth — against `rfdetr`'s own detections (CPU) lifted by the same rule.

    pixi run -e detect ~/.cache/noeira/venvs/detcand/bin/python tools/perception/dump_rfdetr_reference.py
    (and SAM 3's places reference for the depth: tools/perception/dump_sam3_places_reference.py)
    pixi run mojo run -I . tests/perception/test_rfdetr_service_gpu.mojo

The frame: NYU labelled 921 (the depth registered to the RGB). Gates:
  1. READY; one request (picture + "bed", "sofa", "table", "fridge"); the
     caller's `start` / `poll` never block long (< 5 ms);
  2. per word the detections rfdetr made of its COCO class at 0.3 (bed 1,
     couch 1, dining table 1, refrigerator 0), mask IoU >= 0.999 against
     rfdetr's masks (0.99 in a TF32 / bf16 build);
  3. each detection's 3D place (`mask_place`) within 1 mm of rfdetr's mask's
     (5 mm in TF32 / bf16);
  4. a word COCO does not have ("door") fails at `start`, naming why.
"""

from std.os import getenv
from std.time import perf_counter_ns
from std.math import sqrt

from noeira.ai.picture import Picture
from noeira.ai.local_service import SVC_LOADING, SVC_READY
from noeira.core.concurrent.thread import sleep_us
from noeira.io.json import load_json
from noeira.io.npy import load_npy_f64
from noeira.perception.depth import DepthCameraSpec
from noeira.perception.detections import mask_place
from noeira.perception.rfdetr.service import RfDetrService, coco_id
from noeira.perception.sam3.ops import SAM3_TF32


comptime IOU_BOUND: Float64 = 0.99 if SAM3_TF32 else 0.999
comptime PLACE_BOUND: Float64 = 5e-3 if SAM3_TF32 else 1e-3
"""fp32: rfdetr's masks to the pixel. TF32 / bf16: the detections' masks drift by a
few edge pixels (IoU 0.998 measured on the Orin), a place by < 1 mm."""
comptime W: Int = 640
comptime H: Int = 480


def check(mut fails: Int, ok: Bool, what: String):
    if ok:
        print("  PASS ", what)
    else:
        print("  FAIL ", what)
        fails += 1


def main() raises:
    var fails = 0
    var home = getenv("HOME")
    var R = home + "/.cache/noeira/rfdetr/ref"
    var S = home + "/.cache/noeira/sam3/ref"
    print("RfDetrService on a real RGB-D frame (NYU 921)")
    var im = load_npy_f64(S + "/image.npy")
    var rgb = List[UInt8](capacity=len(im))
    for v in im:
        rgb.append(UInt8(Int(v)))
    var pic = Picture(rgb^, H, W)
    var depth = load_npy_f64(S + "/depth.npy")
    var doc = load_json(S + "/places.json")
    var intr = doc.field(doc.root(), "intrinsics")
    var spec = DepthCameraSpec()
    spec.width = W
    spec.height = H
    spec.fx = doc.number(doc.at(intr, 0))
    spec.fy = doc.number(doc.at(intr, 1))
    spec.cx = doc.number(doc.at(intr, 2))
    spec.cy = doc.number(doc.at(intr, 3))
    spec.validate()

    var t0 = perf_counter_ns()
    var det = RfDetrService()
    while det.state() == SVC_LOADING and perf_counter_ns() - t0 < 300_000_000_000:
        _ = sleep_us(10_000)
    check(fails, det.state() == SVC_READY, String("READY after ") + String((perf_counter_ns() - t0) // 1_000_000)
          + " ms (load + one warm call: " + String(det.load_ms()) + " ms)")

    var words: List[String] = ["bed", "sofa", "table", "fridge"]
    var worst_us = 0
    var f0 = perf_counter_ns()
    det.start(pic, words)
    worst_us = max(worst_us, (perf_counter_ns() - f0) // 1000)
    var polls = 0
    while True:
        polls += 1
        f0 = perf_counter_ns()
        var done = det.poll()
        worst_us = max(worst_us, (perf_counter_ns() - f0) // 1000)
        if done:
            break
        _ = sleep_us(1_000)
    var found = det.result()
    check(fails, worst_us < 5000, String("one request: ") + String(len(found)) + " detections, model "
          + String(det.latency_ms()) + " ms; worst start/poll " + String(worst_us) + " us over " + String(polls) + " polls")

    var rl = load_npy_f64(R + "/post_labels.npy")
    var rm = load_npy_f64(R + "/post_masks.npy")
    for wi in range(len(words)):
        var cid = coco_id(words[wi])
        var mine = List[Int]()
        for k in range(len(found)):
            if found[k].word == wi:
                mine.append(k)
        var theirs = List[Int]()
        for k in range(len(rl)):
            if Int(rl[k]) == cid:
                theirs.append(k)
        check(fails, len(mine) == len(theirs), String("'") + words[wi] + "' (COCO " + String(cid) + "): "
              + String(len(mine)) + " detections (rfdetr " + String(len(theirs)) + ")")
        for j in range(min(len(mine), len(theirs))):
            ref d = found[mine[j]]
            var rmask = List[UInt8](capacity=W * H)
            var inter = 0
            var uni = 0
            for p in range(W * H):
                var b = rm[theirs[j] * W * H + p] != 0.0
                rmask.append(UInt8(1) if b else UInt8(0))
                var a = d.mask[p] != 0
                inter += 1 if (a and b) else 0
                uni += 1 if (a or b) else 0
            var iou = Float64(inter) / Float64(max(uni, 1))
            var pa = mask_place(spec, depth, d.mask)
            var pb = mask_place(spec, depth, rmask)
            var e = 1e9
            if len(pa) == 3 and len(pb) == 3:
                e = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2 + (pa[2] - pb[2]) ** 2)
            check(fails, iou >= IOU_BOUND and e <= PLACE_BOUND, String("'") + words[wi] + "' #" + String(j) + ": score "
                  + String(d.score) + ", mask IoU " + String(iou) + ", place (" + String(pa[0] if len(pa) == 3 else 0.0)
                  + ", " + String(pa[1] if len(pa) == 3 else 0.0) + ", " + String(pa[2] if len(pa) == 3 else 0.0)
                  + ") m, |d| vs rfdetr's mask " + String(e) + " m")

    var refused = False
    var why = String("")
    try:
        var bad: List[String] = ["door"]
        det.start(pic, bad)
    except e:
        refused = True
        why = String(e)
    check(fails, refused and why.find("COCO") >= 0, String("'door' refused at start: ") + why)

    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")

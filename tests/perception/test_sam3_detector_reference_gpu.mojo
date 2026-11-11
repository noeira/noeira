"""`Sam3Service` (`noeira.perception.detector`) on a REAL RGB-D frame: SAM 3 in
this process on its own thread, its masks lifted by the frame's registered
depth to 3D places — against transformers' masks lifted by the same rule.

    pixi run -e detect python tools/perception/dump_sam3_reference.py --image .../nyu_0921.png
    pixi run -e detect python tools/perception/dump_sam3_places_reference.py --mat .../nyu_depth_v2_labeled.mat --index 921
    SAM3_CHECKPOINT=... ./build/test_sam3_detector [ref dir]       (build with -D SAM3_BF16=1 / -D SAM3_TF32=1 or fp32)

The frame: NYU Depth v2 labelled frame 921 (Kinect, 640 x 480, depth registered
to the RGB, NYU's intrinsics). Gates:
  1. the service: READY, one request (picture + the 5 words), a reply; the
     caller's `start` / `poll` never block long (worst call printed, < 5 ms);
  2. per word the SAME number of detections as transformers' post-processing;
  3. each detection's 3D place (`mask_place`: per-axis median of its depth
     points) within PLACE_BOUND of transformers' mask's place (`places.json`);
  4. `detections_endpoints_world` at stride 1 labels exactly the depth pixels
     of the masks (non-overlapping here), and their place map's footprint is
     printed;
  5. a DEFECT is caught: the masks mirrored left-right move the places > 0.3 m.
Printed, not gated: each place against NYU's true instance (the detector's own
error — 0.7 / 1.2 / 3.4 cm for door / sofa / table with transformers' masks).
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from std.math import sqrt

from noeira.ai.picture import Picture
from noeira.ai.local_service import SVC_LOADING, SVC_READY
from noeira.core.concurrent.thread import sleep_us
from noeira.io.json import load_json, J_NULL
from noeira.io.npy import load_npy_f64
from noeira.math3d import Vec3, Quat
from noeira.perception.depth import DepthCameraSpec
from noeira.perception.places import PlaceMap
from noeira.perception.detector import Sam3Service, Detection, mask_place, detections_endpoints_world
from noeira.perception.sam3.ops import SAM3_TF32, SAM3_BF16


comptime PLACE_BOUND: Float64 = 0.01 if SAM3_TF32 else 1e-3
"""Metres. fp32: the masks are transformers' (pixels flipped 0), so the medians
match to the depth's float32. TF32 / bf16 flip up to ~1e-4 of the pixels: a
median moves by a neighbour's depth at most."""
comptime W: Int = 640
comptime H: Int = 480


def check(mut fails: Int, ok: Bool, what: String):
    if ok:
        print("  PASS ", what)
    else:
        print("  FAIL ", what)
        fails += 1


def dist(a: List[Float64], b: List[Float64]) -> Float64:
    var s = 0.0
    for k in range(3):
        s += (a[k] - b[k]) * (a[k] - b[k])
    return sqrt(s)


def rep103(doc_place: List[Float64]) -> List[Float64]:
    """OpenCV (x right, y down, z forward) -> REP-103 (x forward, y left, z up)."""
    var out = List[Float64]()
    out.append(doc_place[2])
    out.append(-doc_place[0])
    out.append(-doc_place[1])
    return out^


def main() raises:
    var fails = 0
    var args = argv()
    var ref_dir = String(args[1]) if len(args) > 1 else getenv("HOME") + "/.cache/noeira/sam3/ref"
    print("Sam3Service on a real RGB-D frame", "(bf16)" if SAM3_BF16 else ("(TF32)" if SAM3_TF32 else "(fp32)"))
    var im = load_npy_f64(ref_dir + "/image.npy")
    var rgb = List[UInt8](capacity=len(im))
    for v in im:
        rgb.append(UInt8(Int(v)))
    var pic = Picture(rgb^, H, W)
    var depth = load_npy_f64(ref_dir + "/depth.npy")
    var doc = load_json(ref_dir + "/places.json")
    var root = doc.root()
    var intr = doc.field(root, "intrinsics")
    var spec = DepthCameraSpec()
    spec.name = String("nyu-kinect")
    spec.width = W
    spec.height = H
    spec.fx = doc.number(doc.at(intr, 0))
    spec.fy = doc.number(doc.at(intr, 1))
    spec.cx = doc.number(doc.at(intr, 2))
    spec.cy = doc.number(doc.at(intr, 3))
    spec.validate()
    var wnode = doc.field(root, "words")
    var words = List[String]()
    for i in range(doc.size(wnode)):
        words.append(doc.key_at(wnode, i))

    # 1. the service
    var t0 = perf_counter_ns()
    var det = Sam3Service()
    while det.state() == SVC_LOADING and perf_counter_ns() - t0 < 600_000_000_000:
        _ = sleep_us(10_000)
    check(fails, det.state() == SVC_READY, String("READY after ") + String((perf_counter_ns() - t0) // 1_000_000)
          + " ms (load + one warm call: " + String(det.load_ms()) + " ms)")
    var worst_us = 0
    var polls = 0
    var t1 = perf_counter_ns()
    var f0 = perf_counter_ns()
    det.start(pic, words)
    worst_us = max(worst_us, (perf_counter_ns() - f0) // 1000)
    while True:
        polls += 1
        f0 = perf_counter_ns()
        var done = det.poll()
        worst_us = max(worst_us, (perf_counter_ns() - f0) // 1000)
        if done:
            break
        _ = sleep_us(2_000)
    var wall_ms = (perf_counter_ns() - t1) // 1_000_000
    var found = det.result()
    check(fails, worst_us < 5000, String("one request: ") + String(len(found)) + " detections for "
          + String(len(words)) + " words in " + String(wall_ms) + " ms (model " + String(det.latency_ms())
          + " ms); worst start/poll " + String(worst_us) + " us over " + String(polls) + " polls")

    # 2-3. counts and places, word by word
    var label_of_word = List[Int]()
    for wi in range(len(words)):
        label_of_word.append(wi)
        var wn = doc.field(wnode, words[wi])
        var refs = doc.field(wn, "sam3")
        var truth = doc.field(wn, "truth")
        var mine = List[Int]()
        for k in range(len(found)):
            if found[k].word == wi:
                mine.append(k)
        check(fails, len(mine) == doc.size(refs), String("'") + words[wi] + "': " + String(len(mine))
              + " detections (transformers " + String(doc.size(refs)) + ")")
        for j in range(min(len(mine), doc.size(refs))):
            var p = mask_place(spec, depth, found[mine[j]].mask)
            var rn = doc.at(refs, j)
            if doc.kind_of(rn) == J_NULL:
                check(fails, len(p) == 0, String("'") + words[wi] + "' #" + String(j) + ": no place (too few depth points), as the reference")
                continue
            var rp = List[Float64]()
            for a in range(3):
                rp.append(doc.number(doc.at(rn, a)))
            var r = rep103(rp)
            var e = dist(p, r) if len(p) == 3 else 1e9
            var line = String("'") + words[wi] + "' #" + String(j) + ": place (" + String(p[0] if len(p) == 3 else 0.0)
            line += String(", ") + String(p[1] if len(p) == 3 else 0.0) + ", " + String(p[2] if len(p) == 3 else 0.0)
            line += String(") m, |d| vs transformers' mask ") + String(e) + " m (bound " + String(PLACE_BOUND) + ")"
            if doc.kind_of(truth) != J_NULL and doc.size(truth) > 0:
                var best = 1e9
                for t in range(doc.size(truth)):
                    var tn = doc.at(truth, t)
                    if doc.kind_of(tn) == J_NULL:
                        continue
                    var tp = List[Float64]()
                    for a in range(3):
                        tp.append(doc.number(doc.at(tn, a)))
                    if len(p) == 3:
                        best = min(best, dist(p, rep103(tp)))
                line += String("; vs NYU's true instance ") + String(best) + " m"
            check(fails, e <= PLACE_BOUND, line)

    # 4. into a place map, the camera at the origin (identity pose)
    var ends = List[Float64]()
    var hits = List[Bool]()
    var labels = List[Int]()
    detections_endpoints_world(Vec3[DType.float64](0.0, 0.0, 0.0), Quat[DType.float64](1.0, 0.0, 0.0, 0.0), spec, depth,
                               found, label_of_word, 1, ends, hits, labels)
    var want = 0
    for ref d in found:
        for p in range(W * H):
            if d.mask[p] != 0 and depth[p] > 0.0:
                want += 1
    var got = 0
    for l in labels:
        if l >= 0:
            got += 1
    check(fails, got == want, String("labelled depth pixels ") + String(got) + " = the masks' pixels with depth " + String(want))
    var pm = PlaceMap(0.0, -5.0, 0.05, 160, 200, len(words), -2.0, 2.0)
    pm.add_scan(ends, hits, labels)
    for wi in range(len(words)):
        var b = pm.footprint(wi)
        if b.n > 0:
            print("        footprint '", words[wi], "': centre (", b.cx, ",", b.cy, ") m, half extents", b.hu, "x", b.hv, "m,", b.n, "cells")

    # 5. a defect: the masks mirrored left-right
    var moved = 0.0
    for ref d in found:
        var m = List[UInt8](length=W * H, fill=0)
        for v in range(H):
            for u in range(W):
                m[v * W + u] = d.mask[v * W + (W - 1 - u)]
        var a = mask_place(spec, depth, d.mask)
        var b = mask_place(spec, depth, m)
        if len(a) == 3 and len(b) == 3:
            moved = max(moved, dist(a, b))
    check(fails, moved > 0.3, String("defect caught: masks mirrored left-right move a place by ") + String(moved) + " m")

    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")

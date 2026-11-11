"""The Mojo RF-DETR (`noeira.perception.rfdetr`) over a directory of frames,
its detections written for `detector_eval.py --detectors precomputed:<out>`.

    pixi run -e detect python tools/perception/detector_eval.py --mat <nyu .mat> --out X --export-frames ~/.cache/noeira/rfdetr/nyu_eval_frames
    mkdir -p <out> && pixi run mojo run -I . tools/perception/rfdetr_frames.mojo <frames dir> <out dir> [threshold 0.01]
    pixi run -e detect python tools/perception/detector_eval.py --mat <nyu .mat> --out runs/p3/detector_eval --detectors precomputed:<out dir> --tag mojo-rfdetr

Per frame `<frames>/nyu_<i>.npy` (uint8 640 x 480 RGB, listed in `frames.txt`)
-> `<out>/nyu_<i>.det`: `[u32 count]` then per detection `[u32 COCO id][f32
score][f32 x0 y0 x1 y1][u32 h][u32 w][mask bits, row-major, LSB first]`. Only
the COCO classes the eval maps (refrigerator, dining table, couch) — the
others' masks are never made.
"""

from std.sys import argv
from std.time import perf_counter_ns
from std.memory import bitcast
from max.gpu.host import DeviceContext

from noeira.ai.picture import Picture
from noeira.io.fileio import write_file_atomic
from noeira.io.npy import load_npy_f64
from noeira.perception.rfdetr import RfDetr


def put_u32(mut b: List[UInt8], v: Int):
    for k in range(4):
        b.append(UInt8((v >> (8 * k)) & 0xFF))


def put_f32(mut b: List[UInt8], x: Float32):
    put_u32(b, Int(bitcast[DType.uint32](x)))


def main() raises:
    var args = argv()
    if len(args) < 3:
        raise Error("usage: rfdetr_frames <frames dir> <out dir> [threshold]")
    var src = String(args[1])
    var dst = String(args[2])
    var thr = Float32(Float64(String(args[3]))) if len(args) > 3 else Float32(0.01)
    var lines = open(src + "/frames.txt", "r").read().split("\n")
    var frames = List[Int]()
    for ref l in lines:
        if l.byte_length() > 0:
            frames.append(Int(String(l)))
    var det = RfDetr(DeviceContext())
    var keep: List[Int] = [82, 67, 63]
    var t0 = perf_counter_ns()
    var total = 0
    for k in range(len(frames)):
        var i = frames[k]
        var digits = String(i)
        var name = String("nyu_")
        for _ in range(4 - digits.byte_length()):
            name += "0"
        name += digits
        var im = load_npy_f64(src + "/" + name + ".npy")
        var rgb = List[UInt8](capacity=len(im))
        for v in im:
            rgb.append(UInt8(Int(v)))
        var found = det.detect(Picture(rgb^, 480, 640), thr, keep)
        var b = List[UInt8]()
        put_u32(b, len(found))
        for ref d in found:
            put_u32(b, d.label)
            put_f32(b, d.score)
            for c in range(4):
                put_f32(b, d.box[c])
            put_u32(b, 480)
            put_u32(b, 640)
            var at = len(b)
            b.resize(at + (480 * 640 + 7) // 8, 0)
            for p in range(480 * 640):
                if d.mask[p] != 0:
                    b[at + p // 8] |= UInt8(1) << UInt8(p % 8)
        write_file_atomic(dst + "/" + name + ".det", b)
        total += len(found)
        if (k + 1) % 50 == 0:
            print("  ", k + 1, "/", len(frames), "frames,", (perf_counter_ns() - t0) // 1_000_000 // (k + 1), "ms a frame,", total, "detections")
    print("done:", len(frames), "frames,", total, "detections,", (perf_counter_ns() - t0) // 1_000_000 // max(len(frames), 1), "ms a frame")

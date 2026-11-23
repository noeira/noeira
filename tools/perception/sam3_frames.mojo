"""The Mojo SAM 3 (`noeira.perception.sam3`) over a directory of frames, its
detections written for `detector_eval.py --detectors words:<out>`
(docs/NN_MODEL_PORTS_PLAN.md §12.9, T5's comparison with d1's boxes).

    build on the Orin: build/jetson_build.sh s3f tools/perception/sam3_frames.mojo -D SAM3_BF16=1
    build/jetson_run.sh s3f <frames dir> <out dir> [threshold 0.05]

Per frame `<frames>/nyu_<i>.npy` (uint8 480 x 640 RGB, listed in `frames.txt`):
the picture encoded once, then each word ("fridge", "table", "door", "sofa")
segmented, every instance above `threshold` (0.05, as `detector_eval.py`'s own
SAM 3 run) written with its score, box and mask. `<out>/timing.txt`: each
frame's encode and per-word times.
"""

from std.sys import argv
from std.time import perf_counter_ns
from std.memory import bitcast
from max.gpu.host import DeviceContext

from noeira.ai.picture import Picture
from noeira.io.fileio import write_file_atomic
from noeira.io.npy import load_npy_f64
from noeira.perception.sam3.sam3 import Sam3


comptime WORDS: List[String] = ["fridge", "table", "door", "sofa"]


def put_u32(mut b: List[UInt8], v: Int):
    for k in range(4):
        b.append(UInt8((v >> (8 * k)) & 0xFF))


def put_f32(mut b: List[UInt8], x: Float32):
    put_u32(b, Int(bitcast[DType.uint32](x)))


def main() raises:
    var args = argv()
    if len(args) < 3:
        raise Error("usage: sam3_frames <frames dir> <out dir> [threshold]")
    var src = String(args[1])
    var dst = String(args[2])
    var thr = Float32(Float64(String(args[3]))) if len(args) > 3 else Float32(0.05)
    var lines = open(src + "/frames.txt", "r").read().split("\n")
    var frames = List[Int]()
    for ref l in lines:
        if l.byte_length() > 0:
            frames.append(Int(String(l)))
    var t_load = perf_counter_ns()
    var sam = Sam3(DeviceContext())
    print("SAM 3 loaded in", (perf_counter_ns() - t_load) // 1_000_000, "ms;", len(frames), "frames, threshold", thr)
    var words = materialize[WORDS]()
    var timing = String("")
    var enc_ms = 0.0
    var word_ms = 0.0
    var total = 0
    for k in range(len(frames)):
        var fi = frames[k]
        var digits = String(fi)
        var name = String("nyu_")
        for _ in range(4 - digits.byte_length()):
            name += "0"
        name += digits
        var im = load_npy_f64(src + "/" + name + ".npy")
        var rgb = List[UInt8](capacity=len(im))
        for v in im:
            rgb.append(UInt8(Int(v)))
        var pic = Picture(rgb^, 480, 640)
        var t0 = perf_counter_ns()
        var img = sam.encode_image(pic)
        sam.vit.wts.ctx.synchronize()
        var e = Float64(perf_counter_ns() - t0) / 1e6
        enc_ms += e
        var b = List[UInt8]()
        var n_at = len(b)
        put_u32(b, 0)
        var n = 0
        var line = name + " encode " + String(e)
        for wi in range(len(words)):
            var t1 = perf_counter_ns()
            var found = sam.segment(img, words[wi], thr)
            var wm = Float64(perf_counter_ns() - t1) / 1e6
            word_ms += wm
            line += " " + words[wi] + " " + String(wm)
            for ref d in found:
                put_u32(b, wi)
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
                n += 1
        for kk in range(4):
            b[n_at + kk] = UInt8((n >> (8 * kk)) & 0xFF)
        total += n
        write_file_atomic(dst + "/" + name + ".det", b)
        timing += line + "\n"
        if (k + 1) % 50 == 0:
            print("  ", k + 1, "/", len(frames), "frames: encode", Int(enc_ms / Float64(k + 1)), "ms, 4 words",
                  Int(word_ms / Float64(k + 1)), "ms a frame;", total, "detections")
    var tb = List[UInt8]()
    for x in timing.as_bytes():
        tb.append(x)
    write_file_atomic(dst + "/timing.txt", tb)
    print("done:", len(frames), "frames: encode", Int(enc_ms / Float64(len(frames))), "ms, 4 words",
          Int(word_ms / Float64(len(frames))), "ms a frame;", total, "detections")

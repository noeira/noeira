"""d1-2B's `boxes` over a directory of frames, its detections written for
`detector_eval.py --detectors words:<out>` (docs/NN_MODEL_PORTS_PLAN.md §12.9,
T5's comparison with SAM 3).

    build on the Orin: build/jetson_build.sh d1f tools/perception/d1_frames.mojo -D LFM3_BF16=1
    build/jetson_run.sh d1f <frames dir> <out dir> [--w8a8] [--max 6]
    pixi run -e detect python tools/perception/detector_eval.py --mat <nyu .mat> --out runs/p3/detector_eval \\
        --detectors words:<out dir> --tag d1-orin

Per frame `<frames>/nyu_<i>.npy` (uint8 480 x 640 RGB, listed in `frames.txt`):
ONE request with a `boxes` question per word ("Every fridge.", "Every table.",
"Every door.", "Every sofa.", `max` blocks each) over the picture, the state
"The robot's head camera sees this." Every block is written, not only those d1
calls present: its box (the edges' medians, as `answers.answer_boxes` reads
them) with its presence P(count >= k) through `found:boxes`, kept
non-increasing, as the score — so AP sees d1's whole ranking. No masks (h = w
= 0). `<out>/timing.txt`: each frame's request time.
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from std.memory import bitcast
from max.gpu.host import DeviceContext

from noeira.ai.picture import Picture
from noeira.io.fileio import write_file_atomic
from noeira.io.npy import load_npy_f64
from noeira.llm.d1 import D1
from noeira.llm.d1.answers import medians


comptime WORDS: List[String] = ["fridge", "table", "door", "sofa"]


def put_u32(mut b: List[UInt8], v: Int):
    for k in range(4):
        b.append(UInt8((v >> (8 * k)) & 0xFF))


def put_f32(mut b: List[UInt8], x: Float32):
    put_u32(b, Int(bitcast[DType.uint32](x)))


def main() raises:
    var args = argv()
    if len(args) < 3:
        raise Error("usage: d1_frames <frames dir> <out dir> [--w8a8] [--max N] [--model DIR]")
    var src = String(args[1])
    var dst = String(args[2])
    var w8a8 = False
    var blocks = 6
    var model = String("")
    var i = 3
    while i < len(args):
        var a = String(args[i])
        if a == "--w8a8":
            w8a8 = True
        elif a == "--max":
            blocks = Int(String(args[i + 1]))
            i += 1
        elif a == "--model":
            model = String(args[i + 1])
            i += 1
        i += 1
    if model.byte_length() == 0:
        model = getenv("HOME") + "/.cache/noeira/local-ai/d1/d1-2B"
    var words = materialize[WORDS]()
    var q = String("{")
    for k in range(len(words)):
        q += String('"') + words[k] + '": {"type": "boxes", "instructions": "Every ' + words[k] + '.", "max": ' + String(blocks) + "}"
        q += ", " if k + 1 < len(words) else ""
    q += "}"
    var lines = open(src + "/frames.txt", "r").read().split("\n")
    var frames = List[Int]()
    for ref l in lines:
        if l.byte_length() > 0:
            frames.append(Int(String(l)))
    var t_load = perf_counter_ns()
    var d1 = D1.load(DeviceContext(), model, vision=True, w8a8=w8a8)
    print("d1 loaded in", (perf_counter_ns() - t_load) // 1_000_000, "ms;", len(frames), "frames,", blocks, "blocks a word,",
          "W8A8" if w8a8 else "no W8A8")
    var timing = String("")
    var total_ms = 0.0
    var drawn = 0
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
        var pics = List[Picture]()
        pics.append(Picture(rgb^, 480, 640))
        var t0 = perf_counter_ns()
        _ = d1.system_one(String("The robot's head camera sees this."), False, q, pics)
        var ms = Float64(perf_counter_ns() - t0) / 1e6
        total_ms += ms
        timing += name + " " + String(ms) + "\n"
        var b = List[UInt8]()
        var n = len(words) * blocks
        put_u32(b, n)
        for wi in range(len(words)):
            ref slots = d1.last_probs[wi]
            var below = 1.0
            for blk in range(blocks):
                below = min(below, d1.calibration.through("found:boxes", slots[5 * blk][0]))
                var edges = List[List[Float64]]()
                for j in range(1, 5):
                    edges.append(slots[5 * blk + j].copy())
                var r = medians(edges, d1.record.box_edges)
                if below >= 0.5:
                    drawn += 1
                put_u32(b, wi)
                put_f32(b, Float32(below))
                put_f32(b, Float32(r.left * 640.0))
                put_f32(b, Float32(r.top * 480.0))
                put_f32(b, Float32(r.right * 640.0))
                put_f32(b, Float32(r.bottom * 480.0))
                put_u32(b, 0)
                put_u32(b, 0)
        write_file_atomic(dst + "/" + name + ".det", b)
        if (k + 1) % 50 == 0:
            print("  ", k + 1, "/", len(frames), "frames,", Int(total_ms / Float64(k + 1)), "ms a frame,", drawn,
                  "boxes d1 draws (presence >= 0.5)")
    var tb = List[UInt8]()
    for x in timing.as_bytes():
        tb.append(x)
    write_file_atomic(dst + "/timing.txt", tb)
    print("done:", len(frames), "frames,", Int(total_ms / Float64(len(frames))), "ms a frame (4 words),", drawn, "boxes drawn")

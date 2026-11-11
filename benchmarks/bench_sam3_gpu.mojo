"""SAM 3 (`noeira.perception.sam3.Sam3`) timed: one picture encoded, then words
segmented — warm (one untimed round first).

    build-jetson benchmarks/bench_sam3_gpu.mojo [-D SAM3_TF32=1] -o build/bench_sam3
    SAM3_CHECKPOINT=... ./build/bench_sam3 [ref dir] [reps]

The picture is the reference's (`tools/perception/dump_sam3_reference.py`, 640 x 480).
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.ai.picture import Picture
from noeira.io.npy import load_npy_f64
from noeira.perception.sam3 import Sam3
from noeira.perception.sam3.ops import SAM3_TF32


def main() raises:
    var args = argv()
    var ref_dir = String(args[1]) if len(args) > 1 else getenv("HOME") + "/.cache/noeira/sam3/ref"
    var reps = Int(String(args[2])) if len(args) > 2 else 3
    var sam = Sam3(DeviceContext())
    var im = load_npy_f64(ref_dir + "/image.npy")
    var rgb = List[UInt8](capacity=len(im))
    for v in im:
        rgb.append(UInt8(Int(v)))
    var pic = Picture(rgb^, 480, 640)
    var words: List[String] = ["door", "table", "sofa", "bed"]
    var img = sam.encode_image(pic)
    for w in words:
        _ = sam.segment(img, w)
    var enc_ms = 0.0
    var seg_ms = 0.0
    for _ in range(reps):
        var t0 = perf_counter_ns()
        img = sam.encode_image(pic)
        sam.vit.wts.ctx.synchronize()
        enc_ms += Float64(perf_counter_ns() - t0) / 1e6
        for w in words:
            var t1 = perf_counter_ns()
            _ = sam.segment(img, w)
            seg_ms += Float64(perf_counter_ns() - t1) / 1e6
    print("SAM 3", "(TF32)" if SAM3_TF32 else "(fp32)", ": encode_image", enc_ms / Float64(reps), "ms, segment",
          seg_ms / Float64(reps * len(words)), "ms a word")

"""RF-DETR-Seg-Small (`noeira.perception.rfdetr.RfDetr`) timed by stage on one
640 x 480 frame (NYU 921, SAM 3's reference picture): the host preprocessing,
the backbone + projector, the two-stage + decoder + heads + masks, the host
post-processing at 0.3 — warm (two untimed runs first), synchronised after
each stage.

    pixi run mojo run -I . benchmarks/bench_rfdetr_gpu.mojo [reps 20]       (or build-jetson ... on the Orin)
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer

from noeira.ai.picture import Picture
from noeira.io.npy import load_npy_f64
from noeira.nn.constants import DT
from noeira.perception.rfdetr import RfDetr
from noeira.perception.sam3.ops import SAM3_TF32, SAM3_BF16
from max.gpu.host import DeviceContext


def main() raises:
    var args = argv()
    var reps = Int(String(args[1])) if len(args) > 1 else 20
    var im = load_npy_f64(getenv("HOME") + "/.cache/noeira/sam3/ref/image.npy")
    var rgb = List[UInt8](capacity=len(im))
    for v in im:
        rgb.append(UInt8(Int(v)))
    var pic = Picture(rgb^, 480, 640)
    var det = RfDetr(DeviceContext())
    ref ctx = det.wts.ctx
    var none = List[DeviceBuffer[DT]]()
    for _ in range(2):
        _ = det.detect(pic, Float32(0.3))
    ctx.synchronize()
    var t_pre = 0.0
    var t_bb = 0.0
    var t_dec = 0.0
    var t_post = 0.0
    var n_found = 0
    for _ in range(reps):
        var t0 = perf_counter_ns()
        var px = det.pixel_values(pic)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var src = det.backbone(px, none)
        ctx.synchronize()
        var t2 = perf_counter_ns()
        var raw = det.decode(src)
        ctx.synchronize()
        var t3 = perf_counter_ns()
        var found = det.post_process(raw, 480, 640, Float32(0.3))
        var t4 = perf_counter_ns()
        n_found = len(found)
        t_pre += Float64(t1 - t0) / 1e6
        t_bb += Float64(t2 - t1) / 1e6
        t_dec += Float64(t3 - t2) / 1e6
        t_post += Float64(t4 - t3) / 1e6
    var r = Float64(reps)
    print("RF-DETR-Seg-Small", "(bf16)" if SAM3_BF16 else ("(TF32)" if SAM3_TF32 else "(fp32)"), ":",
          (t_pre + t_bb + t_dec + t_post) / r, "ms a frame,", n_found, "detections at 0.3")
    print("   preprocess (host)", t_pre / r, "ms")
    print("   backbone + projector", t_bb / r, "ms")
    print("   two-stage + decoder + heads + masks", t_dec / r, "ms")
    print("   post-process (host)", t_post / r, "ms")

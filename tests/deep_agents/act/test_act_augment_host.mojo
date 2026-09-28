"""ACT image augmentation on the HOST sampler (`ACTDataset.set_augment`).

The device path is gated in `test_act_augment_gpu.mojo` (device == host twin
to one byte). This gates the host sampler's WIRING of the same twin:

  1. with augmentation on, the ROWS do not move: qpos and actions equal an
     un-augmented twin's at the same seed (augmentation has its own stream),
     while the images differ (the positive control);
  2. validation batches are untouched;
  3. every augmented slot and camera equals `augment_camera_u8` +
     `normalize_camera_chw` of ITS stored row under ITS record, exactly — a
     slot or camera mix-up fails here;
  4. the records are inside the config's ranges and differ per (slot, cam).

CPU only; needs a converted store, like `test_act_dataset.mojo`.
"""

from std.sys import exit
from std.python import Python, PythonObject

from noeira.nn.constants import DT
from noeira.deep_agents.act.config import (
    SO101_ADIM, SO101_IMG_H, SO101_IMG_W, SO101_N_CAM, SO101_QPOS,
)
from noeira.deep_agents.act.data import ACTDataset
from noeira.deep_agents.act.inference import normalize_camera_chw
from noeira.deep_agents.act.augment import (
    ImageAugConfig, augment_camera_u8, W_BRIGHT, W_CONTRAST, W_ENABLED,
)

comptime QPOS = SO101_QPOS
comptime ADIM = SO101_ADIM
comptime N_CAM = SO101_N_CAM
comptime IMG_H = SO101_IMG_H
comptime IMG_W = SO101_IMG_W
comptime K = 8
comptime B = 4
comptime CAM = 3 * IMG_H * IMG_W
comptime IMG = N_CAM * CAM
comptime HDS = ACTDataset[QPOS, ADIM, N_CAM, IMG_H, IMG_W]


def store_path() raises -> String:
    var os = Python.import_module("os")
    var env = String(os.environ.get(PythonObject("ACT_STORE"), PythonObject("")))
    if env.byte_length() > 0:
        return env
    var glob = Python.import_module("glob")
    var home = String(os.path.expanduser(PythonObject("~")))
    var hits = glob.glob(PythonObject(
        home + "/.cache/noeira/act_so101/*_" + String(IMG_H) + "x" + String(IMG_W) + ".h5"
    ))
    if Int(String(Python.import_module("builtins").len(hits))) == 0:
        raise Error("no ACT store — set ACT_STORE")
    return String(hits[0])


def check(mut fails: Int, name: String, ok: Bool, detail: String):
    if ok:
        print("  PASS  " + name + "  " + detail)
    else:
        fails += 1
        print("  FAIL  " + name + "  " + detail)


def maxdiff(ref a: List[Scalar[DT]], ref b: List[Scalar[DT]]) -> Float64:
    var m = 0.0
    for i in range(len(a)):
        m = max(m, abs(Float64(a[i]) - Float64(b[i])))
    return m


def main() raises:
    var fails = 0
    print("ACT host-sampler augmentation gate")
    var path = store_path()
    var plain = HDS(path, seed=11, max_image_bytes=0)
    var aug = HDS(path, seed=11, max_image_bytes=0)
    var cfg = ImageAugConfig.default()
    aug.set_augment(cfg)

    var q0 = List[Scalar[DT]]()
    var i0 = List[Scalar[DT]]()
    var a0 = List[Scalar[DT]]()
    var v0 = List[Scalar[DT]]()
    var q1 = List[Scalar[DT]]()
    var i1 = List[Scalar[DT]]()
    var a1 = List[Scalar[DT]]()
    var v1 = List[Scalar[DT]]()

    # ── 1. training batches: same rows, different pixels ────────────────
    plain.sample_batch[K, B](False, q0, i0, a0, v0)
    aug.sample_batch[K, B](False, q1, i1, a1, v1)
    check(fails, "1a same rows with augmentation on", plain.last_rows == aug.last_rows, "")
    check(fails, "1b qpos identical", maxdiff(q0, q1) == 0.0, "")
    check(fails, "1c actions + mask identical",
          maxdiff(a0, a1) == 0.0 and maxdiff(v0, v1) == 0.0, "")
    var di = maxdiff(i0, i1)
    check(fails, "1d images differ (positive control)", di > 0.05,
          "maxdiff " + String(di))

    # ── 3. every slot/camera == the twin on ITS row under ITS record ────
    var row = List[Scalar[DType.uint8]]()
    var cam = List[Scalar[DType.uint8]](length=CAM, fill=0)
    var want = List[Scalar[DT]](length=B * IMG, fill=Scalar[DT](0))
    for b in range(B):
        aug.image_row_u8(aug.last_rows[b], row)
        for c in range(N_CAM):
            augment_camera_u8[IMG_H, IMG_W](
                row, c * CAM, aug.aug_last[b * N_CAM + c], cam, 0
            )
            normalize_camera_chw[IMG_H, IMG_W](cam, 0, want, b * IMG + c * CAM)
    var dw = maxdiff(i1, want)
    check(fails, "3 each slot/camera == augment_camera_u8 + normalise of its row",
          dw == 0.0, "maxdiff " + String(dw))

    # ── 4. the records ──────────────────────────────────────────────────
    var ok_range = len(aug.aug_last) == B * N_CAM
    var distinct = False
    for r in range(len(aug.aug_last)):
        var p = aug.aug_last[r]
        if p[W_ENABLED] != 1.0 or abs(Float64(p[W_BRIGHT])) > Float64(cfg.brightness) + 1e-6:
            ok_range = False
        if abs(Float64(p[W_CONTRAST]) - 1.0) > Float64(cfg.contrast) + 1e-6:
            ok_range = False
        if r > 0 and p[W_BRIGHT] != aug.aug_last[0][W_BRIGHT]:
            distinct = True
    check(fails, "4a one record per (slot, camera), inside the ranges", ok_range,
          String(len(aug.aug_last)) + " records")
    check(fails, "4b records differ per (slot, camera)", distinct, "")

    # ── 2. validation is untouched ──────────────────────────────────────
    plain.rng = UInt64(0x5DEECE66D)
    aug.rng = UInt64(0x5DEECE66D)
    plain.sample_batch[K, B](True, q0, i0, a0, v0)
    aug.sample_batch[K, B](True, q1, i1, a1, v1)
    var dv = maxdiff(i0, i1)
    check(fails, "2 validation batches are not augmented", dv == 0.0,
          "maxdiff " + String(dv))

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILED")
        exit(1)

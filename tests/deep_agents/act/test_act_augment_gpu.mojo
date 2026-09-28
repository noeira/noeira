"""ACT image augmentation — the device kernels against the host twin.

`docs/DOMAIN_RANDOMIZATION_PLAN.md` Phase 1, gates G1a and G1b:

  G1a  augmentation OFF is the original gather, bit for bit — both the
       `set_augment(off)` path and an identity record through the augmented
       kernel.
  G1b  given the SAME records, the device kernel matches
       `augment_camera_u8` + the shared `normalize_camera_chw` to one
       quantisation step; two draws differ; every drawn record is inside its
       config's ranges; a pinned offset reproduces the batch; validation
       batches are never augmented.

Every "differs" check has a positive control (an augmented frame is NOT the
stored one) so a kernel that silently ignored its records would fail here.
Needs a converted store, like `test_act_dataset_gpu.mojo`.
"""

from std.sys import exit
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.deep_agents.act.config import (
    SO101_ADIM,
    SO101_IMG_H,
    SO101_IMG_W,
    SO101_N_CAM,
    SO101_QPOS,
)
from noeira.deep_agents.act.data import ACTDataset
from noeira.deep_agents.act.data_gpu import ACTDeviceDataset
from noeira.deep_agents.act.inference import normalize_camera_chw
from noeira.deep_agents.act.augment import (
    AUG_UNIFORMS,
    AUG_WORDS,
    ImageAugConfig,
    W_BRIGHT,
    W_CONTRAST,
    W_CUT_X0,
    W_DX,
    W_DY,
    W_ENABLED,
    W_GAIN_R,
    W_GAMMA,
    W_NOISE,
    aug_param,
    augment_camera_u8,
)

from std.python import Python, PythonObject

comptime QPOS = SO101_QPOS
comptime ADIM = SO101_ADIM
comptime N_CAM = SO101_N_CAM
comptime IMG_H = SO101_IMG_H
comptime IMG_W = SO101_IMG_W
comptime K = 8
comptime B = 3
comptime CAM_ELEMS = 3 * IMG_H * IMG_W
comptime IMG_ELEMS = N_CAM * CAM_ELEMS
comptime NREC = B * N_CAM * AUG_WORDS

comptime HDS = ACTDataset[QPOS, ADIM, N_CAM, IMG_H, IMG_W]
comptime DDS = ACTDeviceDataset[QPOS, ADIM, N_CAM, IMG_H, IMG_W]


def store_path() raises -> String:
    var os = Python.import_module("os")
    var env = String(os.environ.get(PythonObject("ACT_STORE"), PythonObject("")))
    if env.byte_length() > 0:
        return env
    var glob = Python.import_module("glob")
    var home = String(os.path.expanduser(PythonObject("~")))
    var pat = (
        home + "/.cache/noeira/act_so101/*_" + String(IMG_H) + "x"
        + String(IMG_W) + ".h5"
    )
    var hits = glob.glob(PythonObject(pat))
    var n = Int(String(Python.import_module("builtins").len(hits)))
    if n == 0:
        raise Error("no ACT store at " + pat + " — set ACT_STORE")
    return String(hits[0])


def check(mut fails: Int, name: String, ok: Bool, detail: String):
    if ok:
        print("  PASS  " + name + "  " + detail)
    else:
        fails += 1
        print("  FAIL  " + name + "  " + detail)


def _download(mut t: Tensor, ctx: DeviceContext, n: Int) raises -> List[Scalar[DT]]:
    t.download(ctx)
    var out = List[Scalar[DT]](length=n, fill=Scalar[DT](0))
    for i in range(n):
        out[i] = t.data[i]
    return out^


def _maxdiff(ref a: List[Scalar[DT]], ref b: List[Scalar[DT]]) -> Float64:
    var m = Float64(0.0)
    for i in range(len(a)):
        var d = abs(Float64(a[i]) - Float64(b[i]))
        if d > m:
            m = d
    return m


def _splitmix(mut s: UInt64) -> UInt64:
    s += UInt64(0x9E3779B97F4A7C15)
    var z = s
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def main() raises:
    var fails = 0
    var ctx = DeviceContext()
    print("ACT image-augmentation gate")
    print("  device: " + String(ctx.name()))

    var host = HDS(store_path(), seed=11, max_image_bytes=0)
    var dev = DDS.upload_from[B](host, ctx, seed=11)

    var ep0 = host.train_eps[0]
    var len0 = host.store.episodes.length_of(ep0)
    var st0 = host.store.episodes.start_of(ep0)
    var rows = List[Int]()
    var nreals = List[Int]()
    for ts in [0, len0 // 2, len0 - 2]:
        rows.append(st0 + ts)
        var rem = len0 - ts
        nreals.append(K if rem > K else rem)

    var q = Tensor()
    var im = Tensor()
    var a = Tensor()
    var v = Tensor()

    # ── the un-augmented reference ──────────────────────────────────────
    dev.gather_at[B, K](rows, nreals, q, im, a, v, ctx)
    ctx.synchronize()
    var plain = _download(im, ctx, B * IMG_ELEMS)

    # ── G1a: an identity record through the AUGMENTED kernel ────────────
    var ident = List[Scalar[DT]](length=NREC, fill=Scalar[DT](0))
    var u0 = SIMD[DType.float32, AUG_UNIFORMS](0.5)
    var p_id = aug_param(ImageAugConfig.off(), u0, IMG_H, IMG_W)
    for r in range(B * N_CAM):
        for w in range(AUG_WORDS):
            ident[r * AUG_WORDS + w] = Scalar[DT](p_id[w])
    dev.gather_at_augmented[B, K](rows, nreals, ident, q, im, a, v, ctx)
    ctx.synchronize()
    var through_id = _download(im, ctx, B * IMG_ELEMS)
    var d_id = _maxdiff(plain, through_id)
    check(fails, "G1a identity record == original gather (bit-exact)",
          d_id == 0.0, "maxdiff " + String(d_id))

    # ── G1b: explicit records, device vs host twin ──────────────────────
    # Records from the DEFAULT config; slot (0, cam 0) forced to cut out so
    # the cutout branch is exercised, slot (2, cam 1) left disabled.
    var cfg = ImageAugConfig.default()
    var cut_cfg = cfg
    cut_cfg.cutout_prob = 1.0
    var recs = List[Scalar[DT]](length=NREC, fill=Scalar[DT](0))
    var s: UInt64 = 0xDA7A5EED
    var any_shift = False
    for r in range(B * N_CAM):
        var u = SIMD[DType.float32, AUG_UNIFORMS](0.0)
        for j in range(AUG_UNIFORMS):
            u[j] = Float32(Int(_splitmix(s) >> 40)) * Float32(1.0 / 16777216.0)
        var c = cut_cfg if r == 0 else cfg
        var p = aug_param(c, u, IMG_H, IMG_W)
        if r == B * N_CAM - 1:
            p = p_id
        if p[W_DX] != 0.0 or p[W_DY] != 0.0:
            any_shift = True
        for w in range(AUG_WORDS):
            recs[r * AUG_WORDS + w] = Scalar[DT](p[w])
    check(fails, "the test records include a shift and a cutout",
          any_shift and recs[W_CUT_X0] >= 0.0, "")

    dev.gather_at_augmented[B, K](rows, nreals, recs, q, im, a, v, ctx)
    ctx.synchronize()
    var d_aug = _download(im, ctx, B * IMG_ELEMS)

    var h_aug = List[Scalar[DT]](length=B * IMG_ELEMS, fill=Scalar[DT](0))
    var row_u8 = List[Scalar[DType.uint8]]()
    var aug_u8 = List[Scalar[DType.uint8]](length=IMG_ELEMS, fill=0)
    for b in range(B):
        host.image_row_u8(rows[b], row_u8)
        for c in range(N_CAM):
            var p = SIMD[DType.float32, AUG_WORDS](0.0)
            for w in range(AUG_WORDS):
                p[w] = Float32(recs[(b * N_CAM + c) * AUG_WORDS + w])
            augment_camera_u8[IMG_H, IMG_W](
                row_u8, c * CAM_ELEMS, p, aug_u8, c * CAM_ELEMS
            )
            normalize_camera_chw[IMG_H, IMG_W](
                aug_u8, c * CAM_ELEMS, h_aug, b * IMG_ELEMS + c * CAM_ELEMS
            )

    # One quantisation step in normalized units is 1/255/std; the smallest
    # ImageNet std is 0.224, so a one-byte flip is <= 0.0176.
    var step = 1.0 / 255.0 / 0.224 + 1e-5
    var n_off = 0
    var dmax = Float64(0.0)
    for i in range(B * IMG_ELEMS):
        var d = abs(Float64(d_aug[i]) - Float64(h_aug[i]))
        if d > 1e-5:
            n_off += 1
        if d > dmax:
            dmax = d
    var frac = Float64(n_off) / Float64(B * IMG_ELEMS)
    check(fails, "G1b device == host twin to one byte", dmax <= step,
          "maxdiff " + String(dmax) + " (one byte = " + String(step) + ")")
    check(fails, "G1b bytes that differ at all are rare", frac < 1e-3,
          String(n_off) + " of " + String(B * IMG_ELEMS))

    # Positive control: the augmented slots are NOT the stored frame, and the
    # disabled slot IS.
    var cam_diff = List[Float64]()
    for r in range(B * N_CAM):
        var m = Float64(0.0)
        for e in range(CAM_ELEMS):
            var i = r * CAM_ELEMS + e
            m += abs(Float64(d_aug[i]) - Float64(plain[i]))
        cam_diff.append(m / Float64(CAM_ELEMS))
    var all_moved = True
    for r in range(B * N_CAM - 1):
        if cam_diff[r] < 0.01:
            all_moved = False
    check(fails, "every enabled record changed its camera", all_moved,
          "min mean|d| " + String(cam_diff[0]))
    check(fails, "the disabled record left its camera untouched",
          cam_diff[B * N_CAM - 1] == 0.0, String(cam_diff[B * N_CAM - 1]))

    # ── G1a: `sample` with augmentation OFF is the original gather ──────
    var qo = Tensor()
    var ao = Tensor()
    dev.set_augment(ImageAugConfig.off())
    dev.set_offset(ctx, 777)
    dev.sample[B, K](False, qo, im, ao, v, ctx)
    ctx.synchronize()
    var s_off = _download(im, ctx, B * IMG_ELEMS)
    dev.g.download(ctx)
    dev.n_real.download(ctx)
    var drawn = List[Int]()
    var drawn_n = List[Int]()
    for b in range(B):
        drawn.append(Int(dev.g.data[b]))
        drawn_n.append(Int(dev.n_real.data[b]))
    dev.gather_at[B, K](drawn, drawn_n, q, im, a, v, ctx)
    ctx.synchronize()
    var ref_off = _download(im, ctx, B * IMG_ELEMS)
    var d_off = _maxdiff(s_off, ref_off)
    check(fails, "G1a sample(aug off) == gather_at(same rows)", d_off == 0.0,
          "maxdiff " + String(d_off))

    # ── G1b: draws — reproducible, different, in range ──────────────────
    dev.set_augment(cfg)
    dev.set_offset(ctx, 777)
    dev.sample[B, K](False, qo, im, ao, v, ctx)
    ctx.synchronize()
    var s_a = _download(im, ctx, B * IMG_ELEMS)
    var p_a = _download(dev.aug_params, ctx, NREC)
    dev.set_offset(ctx, 777)
    dev.sample[B, K](False, qo, im, ao, v, ctx)
    ctx.synchronize()
    var s_a2 = _download(im, ctx, B * IMG_ELEMS)
    var d_rep = _maxdiff(s_a, s_a2)
    check(fails, "a pinned offset reproduces the augmented batch",
          d_rep == 0.0, "maxdiff " + String(d_rep))
    var d_vs_off = _maxdiff(s_a, s_off)
    check(fails, "same rows, augmented != un-augmented", d_vs_off > 0.05,
          "maxdiff " + String(d_vs_off))

    dev.sample[B, K](False, qo, im, ao, v, ctx)
    ctx.synchronize()
    var p_b = _download(dev.aug_params, ctx, NREC)
    var d_draw = _maxdiff(p_a, p_b)
    check(fails, "two successive draws give different records",
          d_draw > 1e-3, "maxdiff " + String(d_draw))

    var in_range = True
    var cams_differ = False
    for r in range(B * N_CAM):
        var o = r * AUG_WORDS
        if abs(Float64(p_a[o + W_BRIGHT])) > Float64(cfg.brightness) + 1e-6:
            in_range = False
        if abs(Float64(p_a[o + W_CONTRAST]) - 1.0) > Float64(cfg.contrast) + 1e-6:
            in_range = False
        if abs(Float64(p_a[o + W_GAIN_R]) - 1.0) > Float64(cfg.gain) + 1e-6:
            in_range = False
        if p_a[o + W_GAMMA] <= 0.0:
            in_range = False
        if Float64(p_a[o + W_NOISE]) > Float64(cfg.noise_sigma) + 1e-6:
            in_range = False
        if abs(Float64(p_a[o + W_DX])) > Float64(cfg.max_shift):
            in_range = False
        if p_a[o + W_ENABLED] != 1.0:
            in_range = False
        if r > 0 and p_a[o + W_BRIGHT] != p_a[W_BRIGHT]:
            cams_differ = True
    check(fails, "every drawn record is inside the config's ranges",
          in_range, "")
    check(fails, "records are independent per (slot, camera)", cams_differ, "")

    # ── validation is never augmented ───────────────────────────────────
    dev.set_offset(ctx, 4242)
    dev.sample[B, K](True, qo, im, ao, v, ctx)
    ctx.synchronize()
    var val_on = _download(im, ctx, B * IMG_ELEMS)
    dev.set_augment(ImageAugConfig.off())
    dev.set_offset(ctx, 4242)
    dev.sample[B, K](True, qo, im, ao, v, ctx)
    ctx.synchronize()
    var val_off = _download(im, ctx, B * IMG_ELEMS)
    var d_val = _maxdiff(val_on, val_off)
    check(fails, "validation batches ignore set_augment", d_val == 0.0,
          "maxdiff " + String(d_val))

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILED")
        exit(1)

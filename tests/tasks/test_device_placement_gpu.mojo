"""THE TOWER RESET ON THE DEVICE — `reset_task_slots` in float32 on a GPU.

    pixi run -e apple mojo run -I . tests/tasks/test_device_placement_gpu.mojo

`test_device_placement.mojo` gates the reset's rules on the CPU leg of the
kernel, in float64. This runs the SAME function as a GPU kernel in float32 —
the path every `so101_tower` SAC reset, `tower_act_eval` round and expert
check takes — so the device-only parts get exercised before a rented box
does: the rest-pose draw (`base_qpos_jitter=`), the `:yaw` draw
(`cube_in_bowl` opts in) with its device `sin`/`cos`, and the Philox streams.

Per task, `BATCH` lanes, one thread each; every lane against the host twin
(`sampler.sample_base_qpos` + `sample_placements` on the table's restated
region frames — `test_device_placement` proves those equal FK):

  1. the six rest words, within float32 of the host draw;
  2. every placed slot's seven pose words (position + the yaw quaternion);
  3. vacuity: pan / roll / jaw drawn on every lane, and on `cube_in_bowl` a
     nonzero yaw on every lane.
"""

from std.math import cos, sin
from std.sys import exit, has_accelerator
from max.gpu import block_idx
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.gpu.constants import (
    METADATA_SIZE, META_IDX_INIT_REGION_0, META_IDX_JINIT_0,
)
from noeira.tasks.spec import (
    load_family, load_task, validate_task_against_family, SLOT_FREE,
)
from noeira.tasks.family import task_path
from noeira.tasks.active import init_region_words
from noeira.tasks.sampler import (
    sample_placements, sample_base_qpos, RegionFrame, SampleReport,
)
from noeira.tasks.placement.table import PlacementTable, reset_task_slots
from noeira.tasks.placement.check import joint_init_words
from noeira.tasks.placement.so101_tower import So101TowerPlacement

comptime T = So101TowerPlacement
comptime F32 = DType.float32
comptime BATCH = 32
comptime NQ = T.NQ
comptime NV = T.NV
comptime SEED = 7
comptime TOL: Float64 = 2.0e-5
"""Float32 on the device against float64 on the host: a few ULP of a unit
quantity, and 1e-5 of a pose that spans 0.4 m."""
comptime L_Q = Layout.row_major(BATCH, NQ)
comptime L_V = Layout.row_major(BATCH, NV)
comptime L_M = Layout.row_major(BATCH, METADATA_SIZE)


def _reset_kernel(
    qpos: LayoutTensor[F32, L_Q, MutAnyOrigin],
    qvel: LayoutTensor[F32, L_V, MutAnyOrigin],
    meta: LayoutTensor[F32, L_M, MutAnyOrigin],
    seed: Int64,
):
    var env = Int(block_idx.x)
    if env < BATCH:
        reset_task_slots[T, F32, BATCH, NQ, NV](qpos, qvel, meta, env, Int(seed))


def main() raises:
    comptime if not has_accelerator():
        print("  SKIPPED: no accelerator")
        print("=== SKIPPED (this is not a pass) ===")
        return
    print("=== the tower reset as a GPU kernel (float32) vs the host ===")
    var ctx = DeviceContext()
    var f = load_family(String("noeira/tasks/families/so101_tower.family"))
    var frames = List[RegionFrame]()
    for r in range(T.N_REGIONS):
        frames.append(RegionFrame(
            Float64(T.region_site_x[DType.float64](r)),
            Float64(T.region_site_y[DType.float64](r)),
            Float64(T.region_site_z[DType.float64](r)),
        ))
    var radii = List[Float64]()
    for si in range(len(f.slots)):
        radii.append(f.slots[si].h_radius if f.slots[si].has_geom else 0.02)

    var names: List[String] = [
        "so101_tower_cube_in_bowl", "so101_tower_lift_brick",
        "so101_tower_reach_clear",
    ]
    var fails = 0
    var worst = 0.0
    var words = 0
    for name in names:
        var t = load_task(task_path(f, name))
        validate_task_against_family(t, f)
        var iw = init_region_words(t, f)
        var jw = joint_init_words[T](t)
        var qs = TensorImpl[F32].alloc(BATCH * NQ)
        var vs = TensorImpl[F32].alloc(BATCH * NV)
        var ms = TensorImpl[F32].alloc(BATCH * METADATA_SIZE)
        for e in range(BATCH):
            for j in range(len(iw)):
                ms.data[e * METADATA_SIZE + META_IDX_INIT_REGION_0 + j] = Scalar[F32](iw[j])
            for j in range(len(jw)):
                ms.data[e * METADATA_SIZE + META_IDX_JINIT_0 + j] = Scalar[F32](jw[j])
        qs.upload(ctx)
        vs.upload(ctx)
        ms.upload(ctx)
        ctx.enqueue_function[_reset_kernel](
            qs.lt["gpu", L_Q](), vs.lt["gpu", L_V](), ms.lt["gpu", L_M](),
            Int64(SEED), grid_dim=(BATCH,), block_dim=(1,),
        )
        ctx.synchronize()
        qs.download(ctx)

        var yawed = 0
        var drawn = 0
        var bad = 0
        var has_yaw = False
        for k in range(len(t.inits)):
            if t.inits[k].yaw:
                has_yaw = True
        for lane in range(BATCH):
            var rest = sample_base_qpos(f, UInt64(SEED), lane)
            for i in range(T.N_BASE_QPOS):
                var got = Float64(qs.data[lane * NQ + i])
                var dd = abs(got - rest[i])
                worst = max(worst, dd)
                words += 1
                if dd > TOL:
                    bad += 1
                if f.base_qpos_jitter[i] > 0.0 and abs(got - f.base_qpos[i]) > 1e-6:
                    drawn += 1
            var rep = SampleReport()
            var placed = sample_placements(t, f, frames, radii, UInt64(SEED), lane, rep)
            for j in range(T.N_FREE):
                var si = T.free_slot(j)
                for p in range(len(placed)):
                    if placed[p].slot != si:
                        continue
                    var want: List[Float64] = [
                        placed[p].x, placed[p].y, placed[p].z,
                        cos(0.5 * placed[p].yaw), 0.0, 0.0,
                        sin(0.5 * placed[p].yaw),
                    ]
                    for w in range(7):
                        var got = Float64(qs.data[lane * NQ + T.free_qadr(j) + w])
                        var dd = abs(got - want[w])
                        worst = max(worst, dd)
                        words += 1
                        if dd > TOL:
                            bad += 1
                            print("      ", name, "lane", lane, "slot", si,
                                  "word", w, ": device", got, "host", want[w])
                    if placed[p].yaw != 0.0:
                        yawed += 1
        var ok = bad == 0 and drawn == BATCH * 3
        if has_yaw:
            ok = ok and yawed == BATCH
        else:
            ok = ok and yawed == 0
        print(("  ok: " if ok else "  FAIL: ") + name + ": " + String(bad)
              + " words off, rest drawn " + String(drawn) + "/" + String(BATCH * 3)
              + ", yawed placements " + String(yawed)
              + (" (task opts in)" if has_yaw else " (no :yaw)"))
        if not ok:
            fails += 1
    print("  compared", words, "words, worst |device - host|", worst)
    if fails == 0:
        print("=== PASS ===")
    else:
        print("=== FAIL:", fails, "task(s) ===")
        exit(1)

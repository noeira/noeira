"""A lidar scan over a `Model`/`Data`: one implementation, CPU and GPU.

`lidar_cast` is the whole sensor for ONE ray of ONE lane: compose the mount's
world pose from its body, rotate the sensor-frame direction into the world,
and call `physics3d.ray.ray_model` (= MuJoCo's `mj_ray`) with the mount body
excluded. `lidar_scan_cpu` loops it on the host; `BatchedLidar` launches it
with one GPU thread per (lane, ray). Neither re-implements any of it, so the
gate against MuJoCo on the CPU and the GPU-vs-CPU gate together cover both.

⚠ THE CALLER OWNS FRESHNESS, as for the camera renderer: this reads
`xpos`/`xquat` as they are. Scan after the step's forward kinematics.

⚠ RAW DISTANCES ARE `mj_ray`'s: statics are included, no geom group is
filtered, and an invisible geom (alpha 0) is skipped — `ray_model`'s rules.
A lidar that should ignore a decoration needs the decoration made invisible.
"""

from std.math import ceildiv
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, TensorLayout

from noeira.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from noeira.physics3d.fields import Data, Model, DYN1, DYN2, rl1, rl2
from noeira.physics3d.fields.dims import DimsLike
from noeira.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE,
    MODEL_BODY_SIZE,
    MODEL_MESH_META_SIZE,
    MAX_GPU_MESHES,
    MESH_ARENA_FLOATS_PER_TRI,
    MODEL_HFIELD_META_SIZE,
    MAX_GPU_HFIELDS,
)
from noeira.physics3d.ray.model import RayHit, ray_model

from .lidar import (
    LidarSpec, LidarMount, LIDAR_SPIRAL, lidar_directions, lidar_return,
)

comptime _TPB = 128

# The per-launch parameter row: mount body, mount pose, range window, noise.
comptime LIDAR_PARAMS: Int = 12
comptime _P_BODY = 0
comptime _P_POS = 1  # 3
comptime _P_QUAT = 4  # 4, (w, x, y, z)
comptime _P_RMIN = 8
comptime _P_RMAX = 9
comptime _P_SIGMA = 10
comptime _P_DROP = 11


@always_inline
def _pos(n: Int) -> Int:
    return n if n > 0 else 1


@always_inline
def lidar_cast[
    DTYPE: DType,
    L_GEOMS: TensorLayout,
    L_BODIES: TensorLayout,
    L_XPOS: TensorLayout,
    L_XQUAT: TensorLayout,
    L_MESH_META: TensorLayout,
    L_TRI: TensorLayout,
    L_HF_META: TensorLayout,
    L_HF: TensorLayout,
](
    geoms: TileTensor[DTYPE, L_GEOMS, MutAnyOrigin],
    ngeom: Int,
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    xpos: TileTensor[DTYPE, L_XPOS, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    env: Int,
    mesh_meta: TileTensor[DTYPE, L_MESH_META, MutAnyOrigin],
    mesh_tris: TileTensor[DTYPE, L_TRI, MutAnyOrigin],
    hfield_meta: TileTensor[DTYPE, L_HF_META, MutAnyOrigin],
    hfield_data: TileTensor[DTYPE, L_HF, MutAnyOrigin],
    hf_stride: Int,
    body: Int,
    mpos: Vec3Generic[DTYPE],
    mquat: QuatGeneric[DTYPE],
    dir: Vec3Generic[DTYPE],
) -> RayHit[DTYPE] where DTYPE.is_floating_point():
    """One sensor-frame direction `dir` (unit), cast from the mount.

    `t` is in metres because the world direction stays unit length.

    ⚠ A MOUNT ON THE WORLD BODY (0) EXCLUDES NOTHING. MuJoCo's rule would
    exclude body 0 — every static geom, the floor and walls included — which
    for a fixed test rig is a sensor that sees only moving things.
    """
    var origin = mpos
    var q = mquat
    var exclude = -1
    if body > 0:
        # `Data.xquat` is packed (x, y, z, w); `Quat` takes (w, x, y, z).
        var bq = QuatGeneric[DTYPE](
            rebind[Scalar[DTYPE]](xquat[env, body * 4 + 3]),
            rebind[Scalar[DTYPE]](xquat[env, body * 4 + 0]),
            rebind[Scalar[DTYPE]](xquat[env, body * 4 + 1]),
            rebind[Scalar[DTYPE]](xquat[env, body * 4 + 2]),
        )
        var bp = Vec3Generic[DTYPE](
            rebind[Scalar[DTYPE]](xpos[env, body * 3 + 0]),
            rebind[Scalar[DTYPE]](xpos[env, body * 3 + 1]),
            rebind[Scalar[DTYPE]](xpos[env, body * 3 + 2]),
        )
        origin = bp + bq.rotate_vec(mpos)
        q = bq * mquat
        exclude = body
    var wdir = q.rotate_vec(dir)
    return ray_model[DTYPE](
        geoms, ngeom, bodies, xpos, xquat, env,
        mesh_meta, mesh_tris, hfield_meta, hfield_data, hf_stride,
        origin, wdir, exclude,
    )


def lidar_mount_world[
    DTYPE: DType, D: DimsLike, BATCH: Int
](
    d: Data[DTYPE, D, BATCH], mount: LidarMount, env: Int = 0
) -> Tuple[Vec3Generic[DType.float64], QuatGeneric[DType.float64]]:
    """The sensor frame in the world, for a consumer turning returns into
    world points (P1's map, with the true pose)."""
    var mp = Vec3Generic[DType.float64](mount.px, mount.py, mount.pz)
    var mq = QuatGeneric[DType.float64](mount.qw, mount.qx, mount.qy, mount.qz)
    if mount.body <= 0:
        return (mp, mq)
    var nb = 4 * d.dims.get_nbody()
    var b = mount.body
    var bq = QuatGeneric[DType.float64](
        Float64(d.xquat.data[env * nb + b * 4 + 3]),
        Float64(d.xquat.data[env * nb + b * 4 + 0]),
        Float64(d.xquat.data[env * nb + b * 4 + 1]),
        Float64(d.xquat.data[env * nb + b * 4 + 2]),
    )
    var n3 = 3 * d.dims.get_nbody()
    var bp = Vec3Generic[DType.float64](
        Float64(d.xpos.data[env * n3 + b * 3 + 0]),
        Float64(d.xpos.data[env * n3 + b * 3 + 1]),
        Float64(d.xpos.data[env * n3 + b * 3 + 2]),
    )
    return (bp + bq.rotate_vec(mp), bq * mq)


def lidar_scan_cpu[
    DTYPE: DType, D: DimsLike, BATCH: Int
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    spec: LidarSpec,
    mount: LidarMount,
    dirs: List[Float64],
    mut ranges: List[Float64],
    mut geom_ids: List[Int],
    env: Int = 0,
    frame: Int = 0,
    seed: Int = 0,
) raises where DTYPE.is_floating_point():
    """One frame of one lane on the host: `ranges[i]` in metres or -1, and the
    geom each beam hit (-1 = none; kept for the segmentation oracle of P3).

    `dirs` is `lidar_directions(spec, frame)` — passed in so a caller scanning
    many lanes builds it once.
    """
    var n = len(dirs) // 3
    if n * 3 != len(dirs):
        raise Error("lidar_scan_cpu: dirs must hold 3 floats per ray")
    if mount.body < 0 or mount.body >= m.dims.get_nbody():
        raise Error("lidar_scan_cpu: mount body " + String(mount.body)
                    + " is not a body of this model")
    var nb = m.dims.get_nbody()
    var ng = m.dims.get_ngeom()
    var hfn = m.dims.get_nhfield_data()
    var geoms_v = m.geoms.tt_dyn["cpu", DYN2](rl2(_pos(ng), MODEL_GEOM_SIZE))
    var bodies_v = m.bodies.tt_dyn["cpu", DYN2](rl2(nb, MODEL_BODY_SIZE))
    var xpos_v = d.xpos.tt_dyn["cpu", DYN2](rl2(BATCH, nb * 3))
    var xquat_v = d.xquat.tt_dyn["cpu", DYN2](rl2(BATCH, nb * 4))
    var mm_v = m.mesh_meta.tt_dyn["cpu", DYN1](
        rl1(MAX_GPU_MESHES * MODEL_MESH_META_SIZE)
    )
    var mt_v = m.mesh_tris.tt_dyn["cpu", DYN1](
        rl1(_pos(m.dims.get_nmesh_tri() * MESH_ARENA_FLOATS_PER_TRI))
    )
    var hm_v = m.hfield_meta.tt_dyn["cpu", DYN1](
        rl1(MAX_GPU_HFIELDS * MODEL_HFIELD_META_SIZE)
    )
    var hd_v = d.hfield_data.tt_dyn["cpu", DYN1](rl1(BATCH * _pos(hfn)))

    var mpos = Vec3Generic[DTYPE](
        Scalar[DTYPE](mount.px), Scalar[DTYPE](mount.py), Scalar[DTYPE](mount.pz)
    )
    var mquat = QuatGeneric[DTYPE](
        Scalar[DTYPE](mount.qw), Scalar[DTYPE](mount.qx),
        Scalar[DTYPE](mount.qy), Scalar[DTYPE](mount.qz),
    )
    var rmin = Scalar[DTYPE](spec.range_min)
    var rmax = Scalar[DTYPE](spec.range_max)
    var sig = Scalar[DTYPE](spec.range_sigma)
    var drop = Scalar[DTYPE](spec.dropout)
    # The low 32 bits, widened exactly as the kernel widens its operand.
    var s64 = UInt64(Int(Int32(seed)))
    ranges = List[Float64](length=n, fill=-1.0)
    geom_ids = List[Int](length=n, fill=-1)
    for i in range(n):
        var dir = Vec3Generic[DTYPE](
            Scalar[DTYPE](dirs[3 * i]),
            Scalar[DTYPE](dirs[3 * i + 1]),
            Scalar[DTYPE](dirs[3 * i + 2]),
        )
        var h = lidar_cast[DTYPE](
            geoms_v, ng, bodies_v, xpos_v, xquat_v, env,
            mm_v, mt_v, hm_v, hd_v, hfn,
            mount.body, mpos, mquat, dir,
        )
        var r = lidar_return[DTYPE](
            h.t, rmin, rmax, sig, drop, s64, frame, env, i
        )
        ranges[i] = Float64(r)
        geom_ids[i] = h.geom if r >= 0 else -1


def _lidar_kernel[
    DTYPE: DType
](
    geoms: TileTensor[DTYPE, DYN2, MutAnyOrigin],
    bodies: TileTensor[DTYPE, DYN2, MutAnyOrigin],
    xpos: TileTensor[DTYPE, DYN2, MutAnyOrigin],
    xquat: TileTensor[DTYPE, DYN2, MutAnyOrigin],
    mesh_meta: TileTensor[DTYPE, DYN1, MutAnyOrigin],
    mesh_tris: TileTensor[DTYPE, DYN1, MutAnyOrigin],
    hfield_meta: TileTensor[DTYPE, DYN1, MutAnyOrigin],
    hfield_data: TileTensor[DTYPE, DYN1, MutAnyOrigin],
    dirs: TileTensor[DTYPE, DYN1, MutAnyOrigin],
    params: TileTensor[DTYPE, DYN1, MutAnyOrigin],
    out_r: TileTensor[DTYPE, DYN1, MutAnyOrigin],
    out_g: TileTensor[DTYPE, DYN1, MutAnyOrigin],
    # ⚠ `Int32`, not `Int`: `Int` is not `DevicePassable`.
    ngeom: Int32,
    nray: Int32,
    nlane: Int32,
    hf_stride: Int32,
    frame: Int32,
    seed: Int32,
):
    """One thread per (lane, ray). Parameterised on DTYPE so it is not also
    instantiated for the host (see `test_ray_model_gpu_vs_cpu`)."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    var n = Int(nray)
    if i >= Int(nlane) * n:
        return
    # ⚠ The positive branch carries the evidence `lidar_cast` needs; an early
    # `comptime if not ...: return` does not (see the camera kernel).
    comptime if DTYPE.is_floating_point():
        var env = i // n
        var ray = i - env * n
        var body = Int(rebind[Scalar[DTYPE]](params[_P_BODY]))
        var mpos = Vec3Generic[DTYPE](
            rebind[Scalar[DTYPE]](params[_P_POS + 0]),
            rebind[Scalar[DTYPE]](params[_P_POS + 1]),
            rebind[Scalar[DTYPE]](params[_P_POS + 2]),
        )
        var mquat = QuatGeneric[DTYPE](
            rebind[Scalar[DTYPE]](params[_P_QUAT + 0]),
            rebind[Scalar[DTYPE]](params[_P_QUAT + 1]),
            rebind[Scalar[DTYPE]](params[_P_QUAT + 2]),
            rebind[Scalar[DTYPE]](params[_P_QUAT + 3]),
        )
        var dir = Vec3Generic[DTYPE](
            rebind[Scalar[DTYPE]](dirs[3 * ray + 0]),
            rebind[Scalar[DTYPE]](dirs[3 * ray + 1]),
            rebind[Scalar[DTYPE]](dirs[3 * ray + 2]),
        )
        var h = lidar_cast[DTYPE](
            geoms, Int(ngeom), bodies, xpos, xquat, env,
            mesh_meta, mesh_tris, hfield_meta, hfield_data, Int(hf_stride),
            body, mpos, mquat, dir,
        )
        var r = lidar_return[DTYPE](
            h.t,
            rebind[Scalar[DTYPE]](params[_P_RMIN]),
            rebind[Scalar[DTYPE]](params[_P_RMAX]),
            rebind[Scalar[DTYPE]](params[_P_SIGMA]),
            rebind[Scalar[DTYPE]](params[_P_DROP]),
            UInt64(Int(seed)),
            Int(frame),
            env,
            ray,
        )
        out_r[i] = r
        out_g[i] = Scalar[DTYPE](h.geom if r >= 0 else -1)


struct BatchedLidar[DTYPE: DType, D: DimsLike, BATCH: Int](Movable):
    """One lidar on every lane, one GPU launch per frame.

    Results stay on the device (`ranges`, `geom_ids`, `[BATCH, n]`) for a
    consumer kernel; `download` copies them out.

    ⚠ `seed` is an `Int32` operand, so only its low 32 bits reach the device.
    The CPU path widens the same 32 bits, so the two draw the same noise.
    """

    var spec: LidarSpec
    var mount: LidarMount
    var n: Int
    var dirs: DeviceBuffer[Self.DTYPE]
    var params: DeviceBuffer[Self.DTYPE]
    var ranges: DeviceBuffer[Self.DTYPE]
    var geom_ids: DeviceBuffer[Self.DTYPE]
    var dirs_frame: Int

    def __init__(
        out self, ctx: DeviceContext, spec: LidarSpec, mount: LidarMount
    ) raises:
        spec.validate()
        self.spec = spec.copy()
        self.mount = mount
        self.n = spec.rays_per_frame()
        self.dirs = ctx.enqueue_create_buffer[Self.DTYPE](3 * self.n)
        self.params = ctx.enqueue_create_buffer[Self.DTYPE](LIDAR_PARAMS)
        self.ranges = ctx.enqueue_create_buffer[Self.DTYPE](Self.BATCH * self.n)
        self.geom_ids = ctx.enqueue_create_buffer[Self.DTYPE](
            Self.BATCH * self.n
        )
        self.dirs_frame = -1
        var p = ctx.enqueue_create_host_buffer[Self.DTYPE](LIDAR_PARAMS)
        ctx.synchronize()
        p[_P_BODY] = Scalar[Self.DTYPE](mount.body)
        p[_P_POS + 0] = Scalar[Self.DTYPE](mount.px)
        p[_P_POS + 1] = Scalar[Self.DTYPE](mount.py)
        p[_P_POS + 2] = Scalar[Self.DTYPE](mount.pz)
        p[_P_QUAT + 0] = Scalar[Self.DTYPE](mount.qw)
        p[_P_QUAT + 1] = Scalar[Self.DTYPE](mount.qx)
        p[_P_QUAT + 2] = Scalar[Self.DTYPE](mount.qy)
        p[_P_QUAT + 3] = Scalar[Self.DTYPE](mount.qz)
        p[_P_RMIN] = Scalar[Self.DTYPE](spec.range_min)
        p[_P_RMAX] = Scalar[Self.DTYPE](spec.range_max)
        p[_P_SIGMA] = Scalar[Self.DTYPE](spec.range_sigma)
        p[_P_DROP] = Scalar[Self.DTYPE](spec.dropout)
        ctx.enqueue_copy(self.params, p)
        self._set_dirs(ctx, 0)

    def _set_dirs(mut self, ctx: DeviceContext, frame: Int) raises:
        var dl = lidar_directions(self.spec, frame)
        var h = ctx.enqueue_create_host_buffer[Self.DTYPE](3 * self.n)
        ctx.synchronize()
        for k in range(3 * self.n):
            h[k] = Scalar[Self.DTYPE](dl[k])
        ctx.enqueue_copy(self.dirs, h)
        ctx.synchronize()
        self.dirs_frame = frame

    def scan(
        mut self,
        ctx: DeviceContext,
        mut d: Data[Self.DTYPE, Self.D, Self.BATCH],
        mut m: Model[Self.DTYPE, Self.D],
        frame: Int = 0,
        seed: Int = 0,
    ) raises:
        """Enqueue one frame for every lane. Does not synchronize.

        Only a `spiral` pattern changes with `frame`; its directions are
        rebuilt on the host and uploaded (`3n` floats, shared by all lanes).
        """
        if self.mount.body < 0 or self.mount.body >= m.dims.get_nbody():
            raise Error("BatchedLidar: mount body " + String(self.mount.body)
                        + " is not a body of this model")
        if self.spec.pattern == LIDAR_SPIRAL and frame != self.dirs_frame:
            self._set_dirs(ctx, frame)
        var nb = m.dims.get_nbody()
        var ng = m.dims.get_ngeom()
        var hfn = m.dims.get_nhfield_data()
        var total = Self.BATCH * self.n
        comptime kern = _lidar_kernel[Self.DTYPE]
        ctx.enqueue_function[kern](
            m.geoms.tt_dyn["gpu", DYN2](rl2(_pos(ng), MODEL_GEOM_SIZE)),
            m.bodies.tt_dyn["gpu", DYN2](rl2(nb, MODEL_BODY_SIZE)),
            d.xpos.tt_dyn["gpu", DYN2](rl2(Self.BATCH, nb * 3)),
            d.xquat.tt_dyn["gpu", DYN2](rl2(Self.BATCH, nb * 4)),
            m.mesh_meta.tt_dyn["gpu", DYN1](
                rl1(MAX_GPU_MESHES * MODEL_MESH_META_SIZE)
            ),
            m.mesh_tris.tt_dyn["gpu", DYN1](
                rl1(_pos(m.dims.get_nmesh_tri() * MESH_ARENA_FLOATS_PER_TRI))
            ),
            m.hfield_meta.tt_dyn["gpu", DYN1](
                rl1(MAX_GPU_HFIELDS * MODEL_HFIELD_META_SIZE)
            ),
            d.hfield_data.tt_dyn["gpu", DYN1](rl1(Self.BATCH * _pos(hfn))),
            TileTensor(self.dirs, rl1(3 * self.n)),
            TileTensor(self.params, rl1(LIDAR_PARAMS)),
            TileTensor(self.ranges, rl1(total)),
            TileTensor(self.geom_ids, rl1(total)),
            Int32(ng),
            Int32(self.n),
            Int32(Self.BATCH),
            Int32(hfn),
            Int32(frame),
            Int32(seed),
            grid_dim=(ceildiv(total, _TPB),),
            block_dim=(_TPB,),
        )

    def download(
        self,
        ctx: DeviceContext,
        mut ranges: List[Float64],
        mut geom_ids: List[Int],
    ) raises:
        """All lanes, `[BATCH * n]` lane-major. Synchronizes."""
        var total = Self.BATCH * self.n
        var hr = ctx.enqueue_create_host_buffer[Self.DTYPE](total)
        var hg = ctx.enqueue_create_host_buffer[Self.DTYPE](total)
        ctx.enqueue_copy(hr, self.ranges)
        ctx.enqueue_copy(hg, self.geom_ids)
        ctx.synchronize()
        ranges = List[Float64](length=total, fill=-1.0)
        geom_ids = List[Int](length=total, fill=-1)
        for k in range(total):
            ranges[k] = Float64(hr[k])
            geom_ids[k] = Int(Float64(hg[k]))

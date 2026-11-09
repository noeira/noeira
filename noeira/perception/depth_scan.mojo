"""A depth image over a `Model`/`Data`: one implementation, CPU and GPU.

`depth_cast` is the whole camera for ONE pixel of ONE lane: the mount's world
pose (`mount_world_pose`, shared with the lidar), the pixel's ray through
`physics3d.ray.ray_model` (= `mj_ray`), planar depth, the incidence angle, and
— for a stereo camera with the shadow on — a second ray from the other imager
to the hit point. `depth_return` (depth.mojo) then applies the error model.
`depth_scan_cpu` loops it; `BatchedDepthCamera` launches one GPU thread per
(lane, pixel).

⚠ THE CALLER OWNS FRESHNESS: scan after the step's forward kinematics.
⚠ The mount body is excluded from its own image, nothing else (MuJoCo's
`bodyexclude`; see `LidarMount`).
"""

from std.math import ceildiv, abs, cos, pi, sqrt
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
    MODEL_CAM_SIZE,
    MAX_GPU_CAMERAS,
    CAM_IDX_BODY,
    CAM_IDX_POS_X,
    CAM_IDX_POS_Y,
    CAM_IDX_POS_Z,
    CAM_IDX_QUAT_X,
    CAM_IDX_QUAT_Y,
    CAM_IDX_QUAT_Z,
    CAM_IDX_QUAT_W,
    CAM_IDX_MODE,
    CAM_IDX_ACTIVE,
    CAM_MODE_FIXED,
)
from noeira.physics3d.ray.model import ray_model

from .lidar import LidarMount
from .lidar_scan import mount_world_pose
from .depth import (
    DepthCameraSpec,
    DEPTH_STEREO,
    depth_directions,
    depth_return,
)

comptime _TPB = 128

# The per-launch parameter row.
comptime DEPTH_PARAMS: Int = 20
comptime _P_BODY = 0
comptime _P_POS = 1  # 3
comptime _P_QUAT = 4  # 4, (w, x, y, z)
comptime _P_RMIN = 8
comptime _P_RMAX = 9
comptime _P_MODEL = 10
comptime _P_KSTEREO = 11
comptime _P_SA = 12
comptime _P_SB = 13
comptime _P_COSMAX = 14
comptime _P_UNIT = 15
comptime _P_DROP = 16
comptime _P_SHADOW = 17
comptime _P_BASE = 18


@always_inline
def _pos(n: Int) -> Int:
    return n if n > 0 else 1


def depth_mount_from_mjcf_camera[
    DTYPE: DType, D: DimsLike
](m: Model[DTYPE, D], cam: Int) raises -> LidarMount:
    """The mount of an MJCF `<camera>` (e.g. the G1's `head_camera`), in the
    sensor frame this package uses.

    MuJoCo's camera looks down its own -Z with +Y up and +X right; ours looks
    along +X with +Y left and +Z up. In the camera frame our forward is -Z,
    our left is -X, our up is +Y — the rotation `q_cs = (0.5, -0.5, 0.5, 0.5)`
    (w, x, y, z), composed after the camera's own orientation.

    ⚠ ONLY A FIXED CAMERA IS A MOUNT. A `track`/`targetbody` camera moves on
    its own rule, which a body-fixed mount cannot express — refused rather
    than mounted at its rest pose. The INTRINSICS are not taken from the
    camera (`fovy` is all the model holds): pass the spec separately, or build
    one with `depth_spec_pinhole_fovy`.
    """
    if cam < 0 or cam >= MAX_GPU_CAMERAS:
        raise Error("depth_mount_from_mjcf_camera: camera " + String(cam) + " out of range")
    var o = cam * MODEL_CAM_SIZE
    if m.cameras.data[o + CAM_IDX_ACTIVE] == 0:
        raise Error("depth_mount_from_mjcf_camera: camera " + String(cam) + " is not declared")
    if Int(m.cameras.data[o + CAM_IDX_MODE]) != CAM_MODE_FIXED:
        raise Error("depth_mount_from_mjcf_camera: camera " + String(cam)
                    + ' is not mode="fixed"; a moving camera is not a body mount')
    var cq = QuatGeneric[DType.float64](
        Float64(m.cameras.data[o + CAM_IDX_QUAT_W]),
        Float64(m.cameras.data[o + CAM_IDX_QUAT_X]),
        Float64(m.cameras.data[o + CAM_IDX_QUAT_Y]),
        Float64(m.cameras.data[o + CAM_IDX_QUAT_Z]),
    )
    var q = cq * QuatGeneric[DType.float64](0.5, -0.5, 0.5, 0.5)
    return LidarMount(
        Int(m.cameras.data[o + CAM_IDX_BODY]),
        Float64(m.cameras.data[o + CAM_IDX_POS_X]),
        Float64(m.cameras.data[o + CAM_IDX_POS_Y]),
        Float64(m.cameras.data[o + CAM_IDX_POS_Z]),
        Float64(q.w), Float64(q.x), Float64(q.y), Float64(q.z),
    )


@fieldwise_init
struct DepthHit[DTYPE: DType](Copyable, ImplicitlyCopyable, Movable):
    var z: Scalar[Self.DTYPE]
    """True planar depth, or -1 for no hit."""
    var geom: Int
    var cos_inc: Scalar[Self.DTYPE]
    var shadowed: Bool


@always_inline
def depth_cast[
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
    shadow: Bool,
    baseline: Scalar[DTYPE],
) -> DepthHit[DTYPE] where DTYPE.is_floating_point():
    """One pixel: `dir` is its UNIT sensor-frame ray (`depth_directions`)."""
    var pose = mount_world_pose[DTYPE](xpos, xquat, env, body, mpos, mquat)
    var origin = pose[0]
    var q = pose[1]
    var exclude = body if body > 0 else -1
    var wdir = q.rotate_vec(dir)
    var h = ray_model[DTYPE](
        geoms, ngeom, bodies, xpos, xquat, env,
        mesh_meta, mesh_tris, hfield_meta, hfield_data, hf_stride,
        origin, wdir, exclude,
    )
    if h.t < 0:
        return DepthHit[DTYPE](Scalar[DTYPE](-1), -1, Scalar[DTYPE](0), False)
    var z = h.t * dir.x
    var c = h.normal.x * wdir.x + h.normal.y * wdir.y + h.normal.z * wdir.z
    var cos_inc = c if c >= 0 else -c
    var shadowed = False
    if shadow:
        # The second imager sits `baseline` to the RIGHT (-y in the sensor
        # frame). The hit is shadowed if anything stands between it and that
        # imager — more than ~1 mm in front of the surface, so the surface
        # itself does not count.
        var o2 = origin + q.rotate_vec(
            Vec3Generic[DTYPE](Scalar[DTYPE](0), -baseline, Scalar[DTYPE](0))
        )
        var p = origin + wdir * h.t
        var v = p - o2
        var L = sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
        if L > 0:
            var cut = L - (Scalar[DTYPE](1e-3) + Scalar[DTYPE](1e-4) * L)
            var h2 = ray_model[DTYPE](
                geoms, ngeom, bodies, xpos, xquat, env,
                mesh_meta, mesh_tris, hfield_meta, hfield_data, hf_stride,
                o2, v * (Scalar[DTYPE](1) / L), exclude,
                tcut=cut,
            )
            shadowed = h2.t >= 0
    return DepthHit[DTYPE](z, h.geom, cos_inc, shadowed)


struct _DepthParams[DTYPE: DType](Copyable, Movable):
    """The spec + mount as the scalars `depth_return` takes."""

    var rmin: Scalar[Self.DTYPE]
    var rmax: Scalar[Self.DTYPE]
    var model: Int
    var k_stereo: Scalar[Self.DTYPE]
    var sa: Scalar[Self.DTYPE]
    var sb: Scalar[Self.DTYPE]
    var cos_max: Scalar[Self.DTYPE]
    var unit: Scalar[Self.DTYPE]
    var drop: Scalar[Self.DTYPE]
    var shadow: Bool
    var baseline: Scalar[Self.DTYPE]

    def __init__(out self, spec: DepthCameraSpec):
        self.rmin = Scalar[Self.DTYPE](spec.range_min)
        self.rmax = Scalar[Self.DTYPE](spec.range_max)
        self.model = spec.model
        var k = 0.0
        if spec.model == DEPTH_STEREO:
            k = spec.subpixel / (spec.fx * spec.baseline)
        self.k_stereo = Scalar[Self.DTYPE](k)
        self.sa = Scalar[Self.DTYPE](spec.sigma_a)
        self.sb = Scalar[Self.DTYPE](spec.sigma_b)
        # A 90 deg limit cuts nothing: cos(90) is ~6e-17, not 0, so it is
        # written as 0 rather than computed.
        var cm = 0.0
        if spec.max_incidence_deg < 90.0:
            cm = cos(spec.max_incidence_deg * pi / 180.0)
        self.cos_max = Scalar[Self.DTYPE](cm)
        self.unit = Scalar[Self.DTYPE](spec.depth_unit)
        self.drop = Scalar[Self.DTYPE](spec.dropout)
        self.shadow = spec.stereo_shadow
        self.baseline = Scalar[Self.DTYPE](spec.baseline)


def depth_scan_cpu[
    DTYPE: DType, D: DimsLike, BATCH: Int
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    spec: DepthCameraSpec,
    mount: LidarMount,
    dirs: List[Float64],
    mut depth: List[Float64],
    mut geom_ids: List[Int],
    env: Int = 0,
    frame: Int = 0,
    seed: Int = 0,
) raises where DTYPE.is_floating_point():
    """One image of one lane: planar depth in metres (0 = none), row-major from
    the top-left, and the geom each pixel saw (-1 = none). `dirs` is
    `depth_directions(spec)`, built once by the caller."""
    var n = spec.npix()
    if len(dirs) != 3 * n:
        raise Error("depth_scan_cpu: dirs must hold 3 floats per pixel")
    if mount.body < 0 or mount.body >= m.dims.get_nbody():
        raise Error("depth_scan_cpu: mount body " + String(mount.body)
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
    var p = _DepthParams[DTYPE](spec)
    var s64 = UInt64(Int(Int32(seed)))
    depth = List[Float64](length=n, fill=0.0)
    geom_ids = List[Int](length=n, fill=-1)
    for i in range(n):
        var dir = Vec3Generic[DTYPE](
            Scalar[DTYPE](dirs[3 * i]),
            Scalar[DTYPE](dirs[3 * i + 1]),
            Scalar[DTYPE](dirs[3 * i + 2]),
        )
        var h = depth_cast[DTYPE](
            geoms_v, ng, bodies_v, xpos_v, xquat_v, env,
            mm_v, mt_v, hm_v, hd_v, hfn,
            mount.body, mpos, mquat, dir, p.shadow, p.baseline,
        )
        var z = depth_return[DTYPE](
            h.z, h.cos_inc, h.shadowed, p.model, p.rmin, p.rmax, p.k_stereo,
            p.sa, p.sb, p.cos_max, p.unit, p.drop, s64, frame, env, i,
        )
        depth[i] = Float64(z)
        geom_ids[i] = h.geom if z > 0 else -1


def _depth_kernel[
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
    out_z: TileTensor[DTYPE, DYN1, MutAnyOrigin],
    out_g: TileTensor[DTYPE, DYN1, MutAnyOrigin],
    ngeom: Int32,
    npix: Int32,
    nlane: Int32,
    hf_stride: Int32,
    frame: Int32,
    seed: Int32,
):
    """One thread per (lane, pixel). Parameterised on DTYPE so it is not also
    instantiated for the host."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    var n = Int(npix)
    if i >= Int(nlane) * n:
        return
    comptime if DTYPE.is_floating_point():
        var env = i // n
        var pix = i - env * n
        var mpos = Vec3Generic[DTYPE](rebind[Scalar[DTYPE]](params[_P_POS]), rebind[Scalar[DTYPE]](params[_P_POS + 1]), rebind[Scalar[DTYPE]](params[_P_POS + 2]))
        var mquat = QuatGeneric[DTYPE](
            rebind[Scalar[DTYPE]](params[_P_QUAT]), rebind[Scalar[DTYPE]](params[_P_QUAT + 1]), rebind[Scalar[DTYPE]](params[_P_QUAT + 2]), rebind[Scalar[DTYPE]](params[_P_QUAT + 3])
        )
        var dir = Vec3Generic[DTYPE](
            rebind[Scalar[DTYPE]](dirs[3 * pix + 0]),
            rebind[Scalar[DTYPE]](dirs[3 * pix + 1]),
            rebind[Scalar[DTYPE]](dirs[3 * pix + 2]),
        )
        var h = depth_cast[DTYPE](
            geoms, Int(ngeom), bodies, xpos, xquat, env,
            mesh_meta, mesh_tris, hfield_meta, hfield_data, Int(hf_stride),
            Int(rebind[Scalar[DTYPE]](params[_P_BODY])), mpos, mquat, dir, rebind[Scalar[DTYPE]](params[_P_SHADOW]) != 0, rebind[Scalar[DTYPE]](params[_P_BASE]),
        )
        var z = depth_return[DTYPE](
            h.z, h.cos_inc, h.shadowed, Int(rebind[Scalar[DTYPE]](params[_P_MODEL])), rebind[Scalar[DTYPE]](params[_P_RMIN]),
            rebind[Scalar[DTYPE]](params[_P_RMAX]), rebind[Scalar[DTYPE]](params[_P_KSTEREO]), rebind[Scalar[DTYPE]](params[_P_SA]), rebind[Scalar[DTYPE]](params[_P_SB]), rebind[Scalar[DTYPE]](params[_P_COSMAX]),
            rebind[Scalar[DTYPE]](params[_P_UNIT]), rebind[Scalar[DTYPE]](params[_P_DROP]), UInt64(Int(seed)), Int(frame), env, pix,
        )
        out_z[i] = z
        out_g[i] = Scalar[DTYPE](h.geom if z > 0 else -1)


struct BatchedDepthCamera[DTYPE: DType, D: DimsLike, BATCH: Int](Movable):
    """One depth camera on every lane, one GPU launch per frame. Results stay
    on the device (`depth`, `geom_ids`, `[BATCH, height*width]`)."""

    var spec: DepthCameraSpec
    var mount: LidarMount
    var n: Int
    var dirs: DeviceBuffer[Self.DTYPE]
    var params: DeviceBuffer[Self.DTYPE]
    var depth: DeviceBuffer[Self.DTYPE]
    var geom_ids: DeviceBuffer[Self.DTYPE]

    def __init__(
        out self, ctx: DeviceContext, spec: DepthCameraSpec, mount: LidarMount
    ) raises:
        spec.validate()
        self.spec = spec.copy()
        self.mount = mount
        self.n = spec.npix()
        self.dirs = ctx.enqueue_create_buffer[Self.DTYPE](3 * self.n)
        self.params = ctx.enqueue_create_buffer[Self.DTYPE](DEPTH_PARAMS)
        self.depth = ctx.enqueue_create_buffer[Self.DTYPE](Self.BATCH * self.n)
        self.geom_ids = ctx.enqueue_create_buffer[Self.DTYPE](Self.BATCH * self.n)
        var p = _DepthParams[Self.DTYPE](spec)
        var hp = ctx.enqueue_create_host_buffer[Self.DTYPE](DEPTH_PARAMS)
        var hd = ctx.enqueue_create_host_buffer[Self.DTYPE](3 * self.n)
        ctx.synchronize()
        for k in range(DEPTH_PARAMS):
            hp[k] = 0
        hp[_P_BODY] = Scalar[Self.DTYPE](mount.body)
        hp[_P_POS + 0] = Scalar[Self.DTYPE](mount.px)
        hp[_P_POS + 1] = Scalar[Self.DTYPE](mount.py)
        hp[_P_POS + 2] = Scalar[Self.DTYPE](mount.pz)
        hp[_P_QUAT + 0] = Scalar[Self.DTYPE](mount.qw)
        hp[_P_QUAT + 1] = Scalar[Self.DTYPE](mount.qx)
        hp[_P_QUAT + 2] = Scalar[Self.DTYPE](mount.qy)
        hp[_P_QUAT + 3] = Scalar[Self.DTYPE](mount.qz)
        hp[_P_RMIN] = p.rmin
        hp[_P_RMAX] = p.rmax
        hp[_P_MODEL] = Scalar[Self.DTYPE](p.model)
        hp[_P_KSTEREO] = p.k_stereo
        hp[_P_SA] = p.sa
        hp[_P_SB] = p.sb
        hp[_P_COSMAX] = p.cos_max
        hp[_P_UNIT] = p.unit
        hp[_P_DROP] = p.drop
        hp[_P_SHADOW] = Scalar[Self.DTYPE](1 if p.shadow else 0)
        hp[_P_BASE] = p.baseline
        var dl = depth_directions(spec)
        for k in range(3 * self.n):
            hd[k] = Scalar[Self.DTYPE](dl[k])
        ctx.enqueue_copy(self.params, hp)
        ctx.enqueue_copy(self.dirs, hd)
        ctx.synchronize()

    def scan(
        mut self,
        ctx: DeviceContext,
        mut d: Data[Self.DTYPE, Self.D, Self.BATCH],
        mut m: Model[Self.DTYPE, Self.D],
        frame: Int = 0,
        seed: Int = 0,
    ) raises:
        """Enqueue one image for every lane. Does not synchronize."""
        if self.mount.body < 0 or self.mount.body >= m.dims.get_nbody():
            raise Error("BatchedDepthCamera: mount body " + String(self.mount.body)
                        + " is not a body of this model")
        var nb = m.dims.get_nbody()
        var ng = m.dims.get_ngeom()
        var hfn = m.dims.get_nhfield_data()
        var total = Self.BATCH * self.n
        comptime kern = _depth_kernel[Self.DTYPE]
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
            TileTensor(self.params, rl1(DEPTH_PARAMS)),
            TileTensor(self.depth, rl1(total)),
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
        self, ctx: DeviceContext, mut depth: List[Float64], mut geom_ids: List[Int]
    ) raises:
        """All lanes, `[BATCH * height * width]` lane-major. Synchronizes."""
        var total = Self.BATCH * self.n
        var hz = ctx.enqueue_create_host_buffer[Self.DTYPE](total)
        var hg = ctx.enqueue_create_host_buffer[Self.DTYPE](total)
        ctx.enqueue_copy(hz, self.depth)
        ctx.enqueue_copy(hg, self.geom_ids)
        ctx.synchronize()
        depth = List[Float64](length=total, fill=0.0)
        geom_ids = List[Int](length=total, fill=-1)
        for k in range(total):
            depth[k] = Float64(hz[k])
            geom_ids[k] = Int(Float64(hg[k]))

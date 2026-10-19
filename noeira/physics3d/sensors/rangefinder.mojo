"""`<rangefinder>` — distance from a site along its own +Z, or -1 for no hit.

`engine_sensor.c:601`. For a SITE-attached rangefinder the whole sensor is four
lines of the reference:

    rvec   = site_xmat column 2          (the site's own +Z, in world)
    origin = site_xpos
    dist   = mj_ray(m, d, origin, rvec, NULL, 1, site_bodyid, &geomid, NULL)

⚠⚠ THE RAY POINTS ALONG +Z, NOT -Z. A CAMERA looks down its own -Z
(`mjCCamera`) and a rangefinder fires along its +Z; the two conventions sit
three hundred lines apart in the same reference and are opposite. Getting it
backwards gives a sensor that reads the scenery BEHIND the robot, which on a
symmetric arena is a plausible-looking signal that never converges.

⚠ `bodyexclude` IS THE SITE'S BODY AND NOTHING MORE. A rangefinder on a
quadruped's torso excludes the torso and still sees its own legs — MuJoCo's
behaviour, reproduced rather than improved, and it is why `escape`'s readings
are not simply "distance to terrain".

⚠ STATICS ARE INCLUDED (`flg_static = 1`) and no group is filtered
(`geomgroup = NULL`), so the floor and the terrain both occlude. The only
exclusions left are the sensor's own body and INVISIBILITY — see
`ray/model.mojo`, where the latter is a precomputed flag precisely because it
is the difference between a rangefinder reading terrain and reading a
decoration.

⚠ **NO HIT IS -1, NOT INFINITY, NOT THE CUTOFF** — and -1 is a SENTINEL, not
a distance. ⚠⚠ AN EARLIER VERSION OF THIS NOTE SAID `tanh(-1)` MAKES IT A
NEGATIVE READING. THAT IS WRONG. dm_control's `Physics.rangefinder`
(`quadruped.py:204`) reads

    np.where(rf_readings == -1.0, 1.0, np.tanh(rf_readings))

so a miss is replaced by **1.0** — the same value a very distant hit
saturates to — BEFORE the `tanh` is applied to the rest. There is no divide by
a scale either. The consequence for a consumer is the opposite of what that
note claimed: a miss reads as MAXIMUM range, and it is the caller's job to
apply that substitution, not this function's. Returning -1 here is what makes
the substitution possible; returning a large number would make it impossible.

COST. `ray_model` is a linear scan over every geom, so N rangefinders on a
model of G geoms cost N*G ray/geom queries per step — and any geom that is a
MESH costs its whole triangle soup on top. ⚠ An earlier form hoisted the geom
world poses into a `List` shared by all N sensors; `ray_model` composes them
inline now, because a GPU thread owns one RAY and cannot hold a scene.

⚠ `mut d` / `mut m` AND THE `mut` ON `Env.get_state` ARE THE SAME REQUIREMENT.
`TensorImpl.lt_dyn` needs a mutable container to hand out a `LayoutTensor`,
Mojo forbids caching one in a struct field (`AnyOrigin`), and there is no
non-mutating constructor. Nothing here writes; see `core/env.mojo` on why the
trait carries the marker anyway.
"""

from layout import TileTensor, TensorLayout

from noeira.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric

from ..fields import Data, Model, DYN1, DYN2, rl1, rl2
from ..fields.dims import DimsLike
from ..gpu.constants import (
    MODEL_SITE_SIZE,
    SITE_IDX_BODY,
    MODEL_GEOM_SIZE,
    MODEL_BODY_SIZE,
    MODEL_MESH_META_SIZE,
    MAX_GPU_MESHES,
    MESH_ARENA_FLOATS_PER_TRI,
    MODEL_HFIELD_META_SIZE,
    MAX_GPU_HFIELDS,
)
from ..kinematics.site_frame import site_world_quat
from ..ray.model import ray_model

@always_inline
def _pos(n: Int) -> Int:
    """`_at_least_one` — every tensor allocates one element even when unused."""
    return n if n > 0 else 1


def rangefinder_ray[
    DTYPE: DType,
    L_SITES: TensorLayout,
    L_S3: TensorLayout,
    L_B3: TensorLayout,
    L_B4: TensorLayout,
    L_GEOMS: TensorLayout,
    L_BODIES: TensorLayout,
    L_MESH_META: TensorLayout,
    L_TRI: TensorLayout,
    L_HF_META: TensorLayout,
    L_HF: TensorLayout,
](
    sites: TileTensor[DTYPE, L_SITES, MutAnyOrigin],
    site_xpos: TileTensor[DTYPE, L_S3, MutAnyOrigin],
    xpos: TileTensor[DTYPE, L_B3, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_B4, MutAnyOrigin],
    geoms: TileTensor[DTYPE, L_GEOMS, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    mesh_meta: TileTensor[DTYPE, L_MESH_META, MutAnyOrigin],
    mesh_tris: TileTensor[DTYPE, L_TRI, MutAnyOrigin],
    hfield_meta: TileTensor[DTYPE, L_HF_META, MutAnyOrigin],
    hfield_data: TileTensor[DTYPE, L_HF, MutAnyOrigin],
    ngeom: Int,
    hf_stride: Int,
    env: Int,
    site: Int,
) -> Scalar[DTYPE] where DTYPE.is_floating_point():
    """One site-attached `<rangefinder>`, over the batched tensors.

    ⚠⚠ THIS IS THE WHOLE SENSOR, AND `rangefinder_site` BELOW IS NOW A
    WRAPPER. It used to bind these ten views itself from `Data`/`Model`, which
    made it unusable from a GPU kernel — a kernel is handed tensors, not
    structs, and cannot manufacture one. Everything the sensor does is here;
    the wrapper exists for the CPU config hooks that hold a `Data`.

    -1.0 means the ray hit nothing. ⚠ A SENTINEL, not a distance:
    `dm_control`'s `Physics.rangefinder` replaces it with 1.0 BEFORE the
    `tanh`, so a miss reads as MAXIMUM range. Applying that is the caller's
    job — returning -1 is what makes it possible.
    """
    var body = Int(rebind[Scalar[DTYPE]](sites[site, SITE_IDX_BODY]))
    var origin = Vec3Generic[DTYPE](
        rebind[Scalar[DTYPE]](site_xpos[env, site * 3 + 0]),
        rebind[Scalar[DTYPE]](site_xpos[env, site * 3 + 1]),
        rebind[Scalar[DTYPE]](site_xpos[env, site * 3 + 2]),
    )
    # `xquat[body] * site_quat` — the site's own world frame. `site_xmat` is
    # not materialised; see `kinematics/site_frame.mojo`.
    var q4 = site_world_quat[DTYPE](env, site, sites, xquat)
    var sq = QuatGeneric[DTYPE](q4[3], q4[0], q4[1], q4[2])
    # ⚠ +Z. A rangefinder fires along the site's own +Z; a CAMERA looks down
    # its -Z. See the module docstring.
    var rvec = sq.rotate_vec(Vec3Generic[DTYPE](0, 0, 1))

    var hit = ray_model[DTYPE](
        geoms, ngeom, bodies, xpos, xquat, env,
        mesh_meta, mesh_tris, hfield_meta, hfield_data, hf_stride,
        origin, rvec, body,
    )
    return hit.t


def rangefinder_site[
    DTYPE: DType, D: DimsLike, BATCH: Int = 1
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    site: Int,
    env: Int = 0,
) raises -> Float64 where DTYPE.is_floating_point():
    """`rangefinder_ray` over a `Data`/`Model` pair, in METRES.

    The views this binds are the ten the sensor needs; the arithmetic all
    lives in `rangefinder_ray`. Kept because the env config hooks that predate
    the sensor framework hold a `Data` and call this directly.
    """
    var nb = m.dims.get_nbody()
    var ns = _pos(m.dims.get_nsite())
    var ng = m.dims.get_ngeom()
    var hfn = _pos(m.dims.get_nhfield_data())
    return Float64(
        rangefinder_ray[DTYPE](
            m.sites.tt_dyn["cpu", DYN2](rl2(ns, MODEL_SITE_SIZE)),
            d.site_xpos.tt_dyn["cpu", DYN2](rl2(BATCH, ns * 3)),
            d.xpos.tt_dyn["cpu", DYN2](rl2(BATCH, nb * 3)),
            d.xquat.tt_dyn["cpu", DYN2](rl2(BATCH, nb * 4)),
            m.geoms.tt_dyn["cpu", DYN2](rl2(ng, MODEL_GEOM_SIZE)),
            m.bodies.tt_dyn["cpu", DYN2](rl2(nb, MODEL_BODY_SIZE)),
            m.mesh_meta.tt_dyn["cpu", DYN1](
                rl1(MAX_GPU_MESHES * MODEL_MESH_META_SIZE)
            ),
            m.mesh_tris.tt_dyn["cpu", DYN1](
                rl1(_pos(m.dims.get_nmesh_tri() * MESH_ARENA_FLOATS_PER_TRI))
            ),
            m.hfield_meta.tt_dyn["cpu", DYN1](
                rl1(MAX_GPU_HFIELDS * MODEL_HFIELD_META_SIZE)
            ),
            d.hfield_data.tt_dyn["cpu", DYN1](rl1(BATCH * hfn)),
            ng,
            m.dims.get_nhfield_data(),
            env,
            site,
        )
    )

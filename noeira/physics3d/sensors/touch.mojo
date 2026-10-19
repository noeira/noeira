"""Touch sensor — MuJoCo `<touch site="..."/>`.

Port of `engine_sensor.c`'s `mjSENS_TOUCH` case. The sensor sums the NORMAL
force of every active contact that (a) involves the site's body and (b) whose
contact point projects into the site's volume along the contact normal:

    for each contact j with efc_address >= 0:
        if site_body not in {body(geom0), body(geom1)}: skip
        f = mj_contactForce(j)[0]           # normal component, contact frame
        if f <= 0: skip
        ray = normalize(frame_normal * f)   # == the unit normal, geom1 -> geom2
        if site_body == body(geom2): ray = -ray
        if rayGeom(site_xpos, site_xmat, site_size, contact_pos, ray, type) >= 0:
            sensordata += f

⚠ OUR CONTACT NORMAL POINTS `body_b` -> `body_a`, the reverse of MuJoCo's
geom1 -> geom2, so MuJoCo's `geom2` is our `body_a` and the flip above is keyed
on `body_a`. See the comment at the flip itself for the measurement.

Note the ray starts at the CONTACT POINT and is cast along the normal, and the
zone being intersected is the SITE. A contact inside the site volume always
registers; one outside registers only if the normal points through the site.

Used by dm_control's hopper, whose OBSERVATION carries
`np.log1p(sensordata[['touch_toe', 'touch_heel']])` — so this feeds the policy
input, not only a reward term.

SCOPE: sphere and BOX zones, plus ellipsoid zones MEASURED AS a sphere of
radius size[0]. Hopper's sites are spheres; finger's `touchtop`/`touchbottom`
are ellipsoids (`size=".025 .03 .025"`) and take the approximation, which is
exact there because the in-plane semi-axes are equal and the model is planar —
`test_finger_vs_dm_control::test_touch_site_sphere_approximation_is_exact`
pins both facts. Any OTHER site type raises rather than being silently
treated as a sphere; a capsule zone needs its own ray test.

Box zones landed 2026-08-01 with manipulator, all five of whose `<touch>`
sensors are boxes. They are the first zone whose answer depends on the site's
ORIENTATION, which is why they had to wait for the site quaternion to reach
the model record (`SITE_IDX_QUAT_*`). Two of manipulator's zones carry
`euler="0 15 0"`, so treating them as axis-aligned is not a small error on the
two pads that decide whether a grasp registers.

PRECONDITION: contact records must be POST-SOLVE, i.e. read after the
integrator has run the constraint solve for this step. `CONTACT_IDX_FORCE_N`
is zero before that, so a hook that reads it too early gets a silent all-zero
sensor rather than an error.
"""

from layout import TileTensor, TensorLayout

from ..fields import Data, Dims, DimsLike
from noeira.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from ..ray import ray_geom
from ..kinematics.quat_math import gpu_quat_mul, gpu_quat_rotate
from ..gpu.constants import (
    CONTACT_SIZE,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_POS_X,
    CONTACT_IDX_POS_Y,
    CONTACT_IDX_POS_Z,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_FORCE_N,
    METADATA_SIZE,
    META_IDX_NUM_CONTACTS,
    MODEL_SITE_SIZE,
    SITE_IDX_BODY,
    SITE_IDX_TYPE,
    SITE_IDX_SIZE_0,
    SITE_IDX_SIZE_1,
    SITE_IDX_SIZE_2,
    SITE_IDX_QUAT_X,
    SITE_IDX_QUAT_Y,
    SITE_IDX_QUAT_Z,
    SITE_IDX_QUAT_W,
)


def touch_sphere_site[DTYPE: DType, D: DimsLike](
    d: Data[DTYPE, D, 1],
    m_sites: List[Scalar[DTYPE]],
    site: Int,
    scale: Float64,
) raises -> Float64:
    """`sensordata` for one `<touch>` sensor, single-env (BATCH=1) CPU path.

    `scale` multiplies every normal force before summing. `CONTACT_IDX_FORCE_N`
    is already in `mj_contactForce`'s units (verified on a settling drop
    against MuJoCo's own `sensordata`), so callers matching MuJoCo pass 1.0.
    The parameter exists for callers wanting impulses or a normalised signal.
    """
    var sbase = site * MODEL_SITE_SIZE
    var stype = Int(m_sites[sbase + SITE_IDX_TYPE])
    # ⚠⚠ ALL SIX ZONE TYPES, THROUGH `ray_geom` (AUD-45). This block used to
    # RAISE on a capsule or cylinder zone and measure an ellipsoid one as a
    # sphere of radius `size[0]`, because the zone test was two private ray
    # routines living in this file — `_ray_hits_box` and `_ray_hits_sphere`.
    #
    # They are gone. MuJoCo's touch sensor tests the zone with
    # `mju_rayGeom(site_xpos, site_xmat, site_size, con->pos, conray,
    # site_type, NULL) >= 0` (engine_sensor.c, `case mjSENS_TOUCH`), and
    # `ray/geom.ray_geom` IS `mju_rayGeom` — swept against it over all six
    # types by `test_ray_geom_vs_mujoco`, which asserts both the residual and
    # the hit/miss SPLIT at zero. So this is not a new implementation to be
    # gated; it is the removal of a second spelling of a rule the tree already
    # states once, which is the defect shape that put the flipped ray below
    # into four domains before anyone saw it.
    #
    # ⚠ THE ELLIPSOID APPROXIMATION IS GONE WITH THEM, and that is a BEHAVIOUR
    # CHANGE on finger, whose `touchtop`/`touchbottom` are
    # `type="ellipsoid" size=".025 .03 .025"`. It should be invisible there:
    # `test_finger_vs_dm_control::test_touch_site_sphere_approximation_is_exact`
    # pins the case where sphere and ellipsoid agree (equal in-plane
    # semi-axes, planar model), so the two answers coincide on that model and
    # the ellipsoid one is right on every other.
    #
    # ⚠ MESH and HFIELD still have no zone test — `ray_geom` returns NO HIT
    # for them rather than raising, so a site declared with one would silently
    # read zero force. MuJoCo does not allow either as a site type, so the
    # model cannot reach here; the parser is what would have to change first.

    var sbody = Int(m_sites[sbase + SITE_IDX_BODY])
    var sx = Float64(d.site_xpos.data[site * 3 + 0])
    var sy = Float64(d.site_xpos.data[site * 3 + 1])
    var sz = Float64(d.site_xpos.data[site * 3 + 2])

    # Box half-extents and the site's WORLD orientation, needed only by the
    # box branch. `site_xmat` has no equivalent in `Data` — see
    # `kinematics/site_frame.mojo` for why it is composed rather than stored.
    var hx = Float64(m_sites[sbase + SITE_IDX_SIZE_0])
    var hy = Float64(m_sites[sbase + SITE_IDX_SIZE_1])
    var hz = Float64(m_sites[sbase + SITE_IDX_SIZE_2])
    var wq = gpu_quat_mul[DType.float64](
        Float64(d.xquat.data[sbody * 4 + 0]),
        Float64(d.xquat.data[sbody * 4 + 1]),
        Float64(d.xquat.data[sbody * 4 + 2]),
        Float64(d.xquat.data[sbody * 4 + 3]),
        Float64(m_sites[sbase + SITE_IDX_QUAT_X]),
        Float64(m_sites[sbase + SITE_IDX_QUAT_Y]),
        Float64(m_sites[sbase + SITE_IDX_QUAT_Z]),
        Float64(m_sites[sbase + SITE_IDX_QUAT_W]),
    )

    var ncon = Int(d.meta.data[META_IDX_NUM_CONTACTS])
    if ncon > D.MAX_CONTACTS:
        ncon = D.MAX_CONTACTS

    var total = 0.0
    for c in range(ncon):
        var base = c * CONTACT_SIZE
        var ba = Int(d.contacts.data[base + CONTACT_IDX_BODY_A])
        var bb = Int(d.contacts.data[base + CONTACT_IDX_BODY_B])
        if sbody != ba and sbody != bb:
            continue

        # NB: `fn` is a removed Mojo keyword — do not rename this back.
        var f_normal = (
            Float64(d.contacts.data[base + CONTACT_IDX_FORCE_N]) * scale
        )
        if f_normal <= 0.0:
            continue

        var nx = Float64(d.contacts.data[base + CONTACT_IDX_NX])
        var ny = Float64(d.contacts.data[base + CONTACT_IDX_NY])
        var nz = Float64(d.contacts.data[base + CONTACT_IDX_NZ])
        # MuJoCo flips the ray when the sensorized body is the one carrying
        # `geom2`, so the ray always leaves the contact on the same side.
        #
        # ⚠ THAT IS OUR `body_a`, NOT OUR `body_b`. MuJoCo's `con->frame` normal
        # points geom1 -> geom2; ours points BODY_B -> BODY_A. So MuJoCo's
        # "geom2" is our "body_a", and flipping on `bb` — which is what this
        # line did until 2026-08-01 — reverses every ray.
        #
        # MEASURED, not argued (stacker's closed hand, 8 contacts, both engines
        # on the same state): `dot(n, xpos[bb] - xpos[ba])` is negative for all
        # eight of ours while MuJoCo's `dot(frame, geom_xpos[g2] - geom_xpos[g1])`
        # is positive for all eight, and the two engines' normals agree up to
        # exactly that sign once the pairs are matched.
        #
        # ⚠ WHY THIS SURVIVED FOUR DOMAINS. The ray only changes the ANSWER for
        # a contact point OUTSIDE the zone: a point inside is hit from either
        # direction. hopper's and finger's zones contain their contacts, and so
        # do the only two manipulator zones that ever carried force. stacker's
        # `thumb_touch` / `finger_touch` are the first zones to see contacts
        # 2 mm outside them, where MuJoCo reports 0 and the flipped ray reported
        # a full 55 N.
        if sbody == ba:
            nx = -nx
            ny = -ny
            nz = -nz

        var px = Float64(d.contacts.data[base + CONTACT_IDX_POS_X])
        var py = Float64(d.contacts.data[base + CONTACT_IDX_POS_Y])
        var pz = Float64(d.contacts.data[base + CONTACT_IDX_POS_Z])

        # `mju_rayGeom(...) >= 0` — a contact point INSIDE the zone always
        # hits, because `ray_quad` returns the smallest NON-NEGATIVE root and
        # an interior origin makes `c < 0`, so the exit root is the answer.
        var zt = ray_geom[DType.float64](
            Vec3Generic[DType.float64](sx, sy, sz),
            QuatGeneric[DType.float64](wq[3], wq[0], wq[1], wq[2]),
            Vec3Generic[DType.float64](hx, hy, hz),
            Vec3Generic[DType.float64](px, py, pz),
            Vec3Generic[DType.float64](nx, ny, nz),
            stype,
        )
        if zt[0] >= 0.0:
            total += f_normal

    return total


# ⚠ NO LONGER REACHABLE FROM A SITE TYPE (AUD-45). Every zone this sensor can
# be handed now goes through `ray_geom`, which covers all six. The constant
# stays because callers test for it — dog's batched obs used to come back a
# constant -1.0 here, and that sentinel is what made the diagnosis one line
# (`test_dog_gpu_vs_cpu`, obs[181], cpu 0.0 vs gpu -1.0). Removing it would
# turn any future unsupported zone back into a plausible 0.0.
comptime TOUCH_UNSUPPORTED_ZONE: Float64 = -1.0


@always_inline
def touch_sphere_site_gpu[
    DTYPE: DType,
    D: DimsLike,
    L_CONTACTS: TensorLayout,
    L_SITE_XPOS: TensorLayout,
    L_SITES: TensorLayout,
    L_META: TensorLayout,
    L_XQUAT: TensorLayout,
](
    dims: D,
    contacts: TileTensor[DTYPE, L_CONTACTS, MutAnyOrigin],
    site_xpos: TileTensor[DTYPE, L_SITE_XPOS, MutAnyOrigin],
    sites: TileTensor[DTYPE, L_SITES, MutAnyOrigin],
    meta: TileTensor[DTYPE, L_META, MutAnyOrigin],
    # ⚠ ADDED 2026-08-10 FOR THE BOX BRANCH. `site_xmat` is not stored — it is
    # composed as `xquat[body] * site_localquat`, so the box path needs the
    # BODY quaternions. The old signature had no way to get them, which is why
    # a box zone used to bail out; the model table already carried the local
    # quat (`SITE_IDX_QUAT_*`), so this parameter was the only missing piece.
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    env: Int,
    site: Int,
    scale: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """`sensordata` for one `<touch>` sensor, one lane of the batched path.

    ALL SIX zone types, through `ray_geom` — the same routine the CPU twin
    calls, which IS `mju_rayGeom` (AUD-45). `TOUCH_UNSUPPORTED_ZONE` is no
    longer reachable from a site type; it is kept only so a caller that still
    tests for it keeps compiling.

    ⚠ THE BOX BRANCH LANDED BECAUSE dog NEEDED IT, not manipulator/stacker.
    This function used to reject box zones with a note saying tranche 4 would
    bring them; dog's four touch sites (palm_L/R, sole_L/R) are `type="box"`,
    so the batched dog reward and the four touch obs dims came back as a
    CONSTANT -1.0 — caught by `test_dog_gpu_vs_cpu`'s per-block diff at
    obs[181], cpu 0.0 vs gpu -1.0. A sentinel rather than a plausible number
    is what made that a one-line diagnosis.

    ELLIPSOID is now a REAL ellipsoid, as on the CPU side — the shared
    `ray_geom` has no sphere approximation in it. The two paths therefore
    still agree, which is what `test_dog_gpu_vs_cpu` checks.

    ⚠⚠ THE RAY FLIP IS ON `body_a`, NOT `body_b`. MuJoCo flips when the
    sensorized body carries `geom2`; our normal points BODY_B -> BODY_A, so
    MuJoCo's geom2 is our `body_a`. Flipping on the wrong one reverses every
    ray and only changes the ANSWER for contacts OUTSIDE the zone — which is
    why it survived four domains on the CPU side before stacker caught it.
    Do not "simplify" this to `bb`.
    """
    var max_contacts = dims.get_max_contacts()
    comptime ZERO = Scalar[DTYPE](0)
    var sbase = site * MODEL_SITE_SIZE
    var stype = Int(rebind[Scalar[DTYPE]](sites[site, SITE_IDX_TYPE]))

    var sbody = Int(rebind[Scalar[DTYPE]](sites[site, SITE_IDX_BODY]))
    var radius = rebind[Scalar[DTYPE]](sites[site, SITE_IDX_SIZE_0])
    var sx = rebind[Scalar[DTYPE]](site_xpos[env, site * 3 + 0])
    var sy = rebind[Scalar[DTYPE]](site_xpos[env, site * 3 + 1])
    var sz = rebind[Scalar[DTYPE]](site_xpos[env, site * 3 + 2])

    # Box half-extents and the site's WORLD orientation — needed only by the
    # box branch, composed exactly as the CPU twin does.
    var hx = rebind[Scalar[DTYPE]](sites[site, SITE_IDX_SIZE_0])
    var hy = rebind[Scalar[DTYPE]](sites[site, SITE_IDX_SIZE_1])
    var hz = rebind[Scalar[DTYPE]](sites[site, SITE_IDX_SIZE_2])
    var wq = gpu_quat_mul[DTYPE](
        rebind[Scalar[DTYPE]](xquat[env, sbody * 4 + 0]),
        rebind[Scalar[DTYPE]](xquat[env, sbody * 4 + 1]),
        rebind[Scalar[DTYPE]](xquat[env, sbody * 4 + 2]),
        rebind[Scalar[DTYPE]](xquat[env, sbody * 4 + 3]),
        rebind[Scalar[DTYPE]](sites[site, SITE_IDX_QUAT_X]),
        rebind[Scalar[DTYPE]](sites[site, SITE_IDX_QUAT_Y]),
        rebind[Scalar[DTYPE]](sites[site, SITE_IDX_QUAT_Z]),
        rebind[Scalar[DTYPE]](sites[site, SITE_IDX_QUAT_W]),
    )

    var ncon = Int(rebind[Scalar[DTYPE]](meta[env, META_IDX_NUM_CONTACTS]))
    if ncon > max_contacts:
        ncon = max_contacts

    var total = ZERO
    for c in range(ncon):
        var base = c * CONTACT_SIZE
        var ba = Int(
            rebind[Scalar[DTYPE]](contacts[env, base + CONTACT_IDX_BODY_A])
        )
        var bb = Int(
            rebind[Scalar[DTYPE]](contacts[env, base + CONTACT_IDX_BODY_B])
        )
        if sbody != ba and sbody != bb:
            continue

        var f_normal = (
            rebind[Scalar[DTYPE]](contacts[env, base + CONTACT_IDX_FORCE_N])
            * scale
        )
        if f_normal <= ZERO:
            continue

        var nx = rebind[Scalar[DTYPE]](contacts[env, base + CONTACT_IDX_NX])
        var ny = rebind[Scalar[DTYPE]](contacts[env, base + CONTACT_IDX_NY])
        var nz = rebind[Scalar[DTYPE]](contacts[env, base + CONTACT_IDX_NZ])
        if sbody == ba:
            nx = -nx
            ny = -ny
            nz = -nz

        var px = rebind[Scalar[DTYPE]](contacts[env, base + CONTACT_IDX_POS_X])
        var py = rebind[Scalar[DTYPE]](contacts[env, base + CONTACT_IDX_POS_Y])
        var pz = rebind[Scalar[DTYPE]](contacts[env, base + CONTACT_IDX_POS_Z])

        # `mju_rayGeom(...) >= 0`, the same call the CPU twin makes.
        #
        # ⚠⚠ SPLIT BY DTYPE AT COMPTIME, AND NOT BY CHOICE. `ray_geom`
        # carries `where DTYPE.is_floating_point()`, and this function cannot:
        # it is reached through the env-config trait's `custom_extract_obs_gpu`,
        # whose signature every environment in the tree implements. Adding the
        # constraint here propagates to that trait method, and the compiler
        # then rejects the WHOLE conformance —
        #   "method 'init_qpos_gpu' has constraints that cannot be proven or
        #    disproven from conformance constraint"
        # — so proving it properly means widening the trait for every env.
        # Naming the two concrete types supplies the evidence locally instead,
        # and keeps ONE spelling of the zone test across CPU and GPU, which is
        # the entire point of routing through `ray_geom` (AUD-45).
        #
        # ⚠ A NON-FLOAT `DTYPE` FALLS THROUGH AS NO HIT rather than silently
        # summing every contact. Physics `DTYPE` is always float32 or float64,
        # so this is unreachable; it is written down because the alternative
        # default would be a plausible wrong number.
        var hit = False
        comptime if DTYPE == DType.float32:
            var t32 = ray_geom[DType.float32](
                Vec3Generic[DType.float32](
                    Float32(sx), Float32(sy), Float32(sz)
                ),
                QuatGeneric[DType.float32](
                    Float32(wq[3]), Float32(wq[0]),
                    Float32(wq[1]), Float32(wq[2]),
                ),
                Vec3Generic[DType.float32](
                    Float32(hx), Float32(hy), Float32(hz)
                ),
                Vec3Generic[DType.float32](
                    Float32(px), Float32(py), Float32(pz)
                ),
                Vec3Generic[DType.float32](
                    Float32(nx), Float32(ny), Float32(nz)
                ),
                stype,
            )
            hit = t32[0] >= Float32(0)
        else:
            comptime if DTYPE == DType.float64:
                var t64 = ray_geom[DType.float64](
                    Vec3Generic[DType.float64](
                        Float64(sx), Float64(sy), Float64(sz)
                    ),
                    QuatGeneric[DType.float64](
                        Float64(wq[3]), Float64(wq[0]),
                        Float64(wq[1]), Float64(wq[2]),
                    ),
                    Vec3Generic[DType.float64](
                        Float64(hx), Float64(hy), Float64(hz)
                    ),
                    Vec3Generic[DType.float64](
                        Float64(px), Float64(py), Float64(pz)
                    ),
                    Vec3Generic[DType.float64](
                        Float64(nx), Float64(ny), Float64(nz)
                    ),
                    stype,
                )
                hit = t64[0] >= Float64(0)
        if hit:
            total += f_normal
    _ = sbase
    return total

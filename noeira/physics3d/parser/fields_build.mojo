"""Spec-direct Model build: FlatModelDef → packed record tensors (G4).

Replaces the legacy two-hop build (`FlatModelDef.setup_model` → CPU `Model` →
`Model.load_from_model`) with ONE pass that writes the packed record
tensors directly. Every derived quantity the legacy path computed is ported
verbatim:

  * body inv mass / inv inertia (`Model.set_body`: unguarded 1/x),
  * body_rootid / body_weldid (parent-chain walks),
  * joint qpos_adr / dof_adr allocation (running per-type counters),
    hinge/slide axis normalization, per-type defaults (tau_limit 1000 for
    hinge/slide, 0 for free), qpos0 = joint `ref` value,
  * per-joint solref/solimp limit fallbacks (parsed value if >= 0, else the
    MuJoCo model defaults) + the model-level limit sync from joint 0,
  * geom rbound per type (planes 1e10, capsule r+hl, cylinder/box norms),
  * mesh convex-hull loading (STL → dedup → hull) with shared-mesh remap,
  * `<compiler inertiafromgeom>` (modes 1=true / 2=auto) via the ported
    `_inertia_from_geoms_staging` + `<compiler settotalmass>` rescale
    (applied only when inertiafromgeom is active — legacy behavior),
  * equality records with the legacy hardcoded solimp[3]=0.5 / solimp[4]=2.0
    (the parsed values were dropped by `add_connect/weld_constraint` too).

INTENTIONAL FIX vs legacy: `mf.sites` records are now populated (body id +
local pos). `load_from_model` never wrote them — sites were silently
all-zero on the fields path.

invweight0 is NOT computed here — `init_fields` runs the fields-native
`compute_invweight0` (G1) after this build.

Only the BODY mass/inertia block is staged in host Lists (inertiafromgeom +
settotalmass mutate it); all other records are written straight into `mf`.
"""

from std.collections import Array
from std.math import sqrt, tan, pi

from noeira.physics3d.joint_types import (
    JNT_FREE,
    JNT_BALL,
    JNT_SLIDE,
    JNT_HINGE,
)
from noeira.physics3d.constants import (
    GEOM_PLANE,
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_BOX,
    GEOM_CYLINDER,
    GEOM_HFIELD,
    GEOM_MESH,
    GEOM_ELLIPSOID,
)
from noeira.physics3d.fields import Model, SpecFields, Dims, DimsLike
from noeira.physics3d.collision.native_multicontact import (
    MC_MAX_POLYVERT,
    MC_MAX_DEG,
)
from noeira.physics3d.collision.convex_hull import (
    load_mesh_hull,
    compute_mesh_rbound_at,
    compute_mesh_half_extents_at,
)
from noeira.physics3d.model.inertia_from_geom import (
    geom_volume,
    geom_inertia,
    globalinertia,
    offcenter,
    eig3_symmetric,
    quat_to_mat,
    MJ_DEFAULT_DENSITY,
    _mulquat as _mulquat_xyzw,
)
from noeira.physics3d.model.mesh_inertia import (
    MeshInertia,
    mesh_inertia_from_file,
)
from noeira.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_EQ_SIZE,
    MODEL_SITE_SIZE,
    MODEL_SENSOR_SIZE,
    SENSOR_IDX_TYPE,
    SENSOR_IDX_OBJTYPE,
    SENSOR_IDX_OBJID,
    SENSOR_IDX_DIM,
    SENSOR_IDX_ADR,
    SENSOR_IDX_DATATYPE,
    SENSOR_IDX_NEEDSTAGE,
    SENSOR_IDX_CUTOFF,
    SENSOR_IDX_BODY,
    SENSOR_IDX_SERVED,
    SENSOR_IDX_REFTYPE,
    SENSOR_IDX_MAG_X,
    SENSOR_IDX_MAG_Y,
    SENSOR_IDX_MAG_Z,
    SENSOR_IDX_REFID,
    MODEL_MESH_META_SIZE,
    MESH_ARENA_FLOATS_PER_TRI,
    MESH_META_IDX_BVHADR,
    MESH_META_IDX_BVHNUM,
    MODEL_MESH_POLY_SIZE,
    MESH_META_IDX_POLYADR,
    MESH_META_IDX_POLYNUM,
    MESH_META_IDX_TRIADR,
    MESH_META_IDX_TRINUM,
    MESH_POLY_IDX_VERTADR,
    MESH_POLY_IDX_VERTNUM,
    MESH_POLY_IDX_NX,
    MESH_POLY_IDX_NY,
    MESH_POLY_IDX_NZ,
    mesh_max_poly,
    mesh_max_polyvert,
    mesh_max_edge,
    MAX_GPU_MESHES,
    MAX_GPU_HFIELDS,
    MODEL_HFIELD_META_SIZE,
    HFIELD_META_IDX_ADR,
    HFIELD_META_IDX_NROW,
    HFIELD_META_IDX_NCOL,
    HFIELD_META_IDX_SIZE_X,
    HFIELD_META_IDX_SIZE_Y,
    HFIELD_META_IDX_SIZE_Z,
    HFIELD_META_IDX_SIZE_BASE,
    GEOM_IDX_HFIELD_ID,
    BODY_IDX_MASS,
    BODY_IDX_INV_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    BODY_IDX_INV_IXX,
    BODY_IDX_INV_IYY,
    BODY_IDX_INV_IZZ,
    BODY_IDX_POS_X,
    BODY_IDX_POS_Y,
    BODY_IDX_POS_Z,
    BODY_IDX_QUAT_X,
    BODY_IDX_QUAT_Y,
    BODY_IDX_QUAT_Z,
    BODY_IDX_QUAT_W,
    BODY_IDX_PARENT,
    BODY_IDX_IPOS_X,
    BODY_IDX_IPOS_Y,
    BODY_IDX_IPOS_Z,
    BODY_IDX_IQUAT_X,
    BODY_IDX_IQUAT_Y,
    BODY_IDX_IQUAT_Z,
    BODY_IDX_IQUAT_W,
    BODY_IDX_ROOTID,
    MODEL_TREE_SIZE,
    TREE_IDX_DOF_ADR,
    TREE_IDX_DOF_NUM,
    TREE_IDX_KIND,
    TREE_KIND_DENSE,
    TREE_KIND_COMPACT,
    MODEL_META_IDX_NTREE,
    BODY_IDX_WELDID,
    BODY_IDX_DOFNUM,
    BODY_IDX_MOCAP,
    BODY_IDX_GRAVCOMP,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_POS_X,
    JOINT_IDX_POS_Y,
    JOINT_IDX_POS_Z,
    JOINT_IDX_AXIS_X,
    JOINT_IDX_AXIS_Y,
    JOINT_IDX_AXIS_Z,
    JOINT_IDX_TAU_LIMIT,
    JOINT_IDX_RANGE_MIN,
    JOINT_RANGE_UNLIMITED,
    JOINT_IDX_RANGE_MAX,
    JOINT_IDX_ARMATURE,
    JOINT_IDX_DAMPING,
    JOINT_IDX_STIFFNESS,
    JOINT_IDX_SPRINGREF,
    JOINT_IDX_FRICTIONLOSS,
    JOINT_IDX_SOLREF_LIMIT_0,
    JOINT_IDX_SOLREF_LIMIT_1,
    JOINT_IDX_SOLIMP_LIMIT_0,
    JOINT_IDX_SOLIMP_LIMIT_1,
    JOINT_IDX_SOLIMP_LIMIT_2,
    JOINT_IDX_SOLIMP_LIMIT_3,
    JOINT_IDX_SOLIMP_LIMIT_4,
    JOINT_IDX_QPOS0,
    JOINT_IDX_MARGIN,
    MODEL_META_IDX_NBODY,
    MODEL_META_IDX_NJOINT,
    MODEL_META_IDX_GRAVITY_X,
    MODEL_META_IDX_GRAVITY_Y,
    MODEL_META_IDX_GRAVITY_Z,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_DENSITY,
    MODEL_META_IDX_VISCOSITY,
    MODEL_META_IDX_WIND_X,
    MODEL_META_IDX_WIND_Y,
    MODEL_META_IDX_WIND_Z,
    MODEL_META_IDX_SOLREF_CONTACT_0,
    MODEL_META_IDX_SOLREF_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_0,
    MODEL_META_IDX_SOLIMP_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_2,
    MODEL_META_IDX_SOLIMP_CONTACT_3,
    MODEL_META_IDX_SOLIMP_CONTACT_4,
    MODEL_META_IDX_SOLREF_LIMIT_0,
    MODEL_META_IDX_SOLREF_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_0,
    MODEL_META_IDX_SOLIMP_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_2,
    MODEL_META_IDX_SOLIMP_LIMIT_3,
    MODEL_META_IDX_SOLIMP_LIMIT_4,
    MODEL_META_IDX_IMPRATIO,
    MODEL_META_IDX_CONE,
    MODEL_META_IDX_SOLVER,
    MODEL_META_IDX_INTEGRATOR,
    MODEL_META_IDX_MAX_CONDIM,
    MODEL_META_IDX_EULERDAMP_DISABLED,
    MODEL_META_IDX_NGRAVCOMP,
    MODEL_META_IDX_NEQUALITY,
    MODEL_META_IDX_NTENDON,
    MODEL_META_IDX_NEXCLUDE,
    MODEL_META_IDX_NPAIR,
    MODEL_META_IDX_NOSLIP_TOLERANCE,
    MODEL_META_IDX_NOSLIP_ITERATIONS,
    MODEL_META_IDX_CCD_TOLERANCE,
    MODEL_META_IDX_CCD_ITERATIONS,
    MODEL_META_IDX_SOLVER_ITERATIONS,
    MODEL_META_IDX_SOLVER_TOLERANCE,
    MODEL_META_IDX_LS_ITERATIONS,
    MODEL_META_IDX_LS_TOLERANCE,
    MODEL_META_IDX_CTRL_MIN,
    MODEL_META_IDX_CTRL_MAX,
    MODEL_META_IDX_MULTICCD_DISABLED,
    MODEL_META_IDX_FILTERPARENT_DISABLED,
    MODEL_META_IDX_WARMSTART_DISABLED,
    MODEL_PAIR_SIZE,
    PAIR_IDX_GEOM1,
    PAIR_IDX_GEOM2,
    PAIR_IDX_CONDIM,
    PAIR_IDX_FRICTION,
    PAIR_IDX_FRICTION_SPIN,
    PAIR_IDX_FRICTION_ROLL,
    PAIR_IDX_SOLREF_0,
    PAIR_IDX_SOLREF_1,
    PAIR_IDX_SOLIMP_0,
    PAIR_IDX_SOLIMP_1,
    PAIR_IDX_SOLIMP_2,
    PAIR_IDX_SOLIMP_3,
    PAIR_IDX_SOLIMP_4,
    PAIR_IDX_MARGIN,
    PAIR_IDX_GAP,
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_W,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_HALF_X,
    GEOM_IDX_HALF_Y,
    GEOM_IDX_HALF_Z,
    GEOM_IDX_FRICTION,
    GEOM_IDX_CONTYPE,
    GEOM_IDX_CONAFFINITY,
    GEOM_IDX_CONDIM,
    GEOM_IDX_FRICTION_SPIN,
    GEOM_IDX_FRICTION_ROLL,
    GEOM_IDX_RBOUND,
    GEOM_IDX_PRIORITY,
    GEOM_IDX_RAY_VISIBLE,
    MODEL_GEOM_RGBA_SIZE,
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
    CAM_IDX_FOVY,
    CAM_IDX_MODE,
    CAM_IDX_TARGET_BODY,
    CAM_IDX_ACTIVE,
    CAM_IDX_TAN_HALF_FOVY,
    CAM_MODE_FIXED,
    CAM_MODE_TRACK,
    CAM_MODE_TRACKCOM,
    CAM_MODE_TARGETBODY,
    CAM_MODE_TARGETBODYCOM,
    GEOM_IDX_GROUP,
    GEOM_IDX_SOLREF_0,
    GEOM_IDX_SOLREF_1,
    GEOM_IDX_SOLIMP_0,
    GEOM_IDX_SOLIMP_1,
    GEOM_IDX_SOLIMP_2,
    GEOM_IDX_SOLIMP_3,
    GEOM_IDX_SOLIMP_4,
    GEOM_IDX_MARGIN,
    GEOM_IDX_GAP,
    GEOM_IDX_MESH_ID,
    EQ_IDX_TYPE,
    EQ_IDX_OBJTYPE,
    EQ_IDX_BODY_A,
    EQ_IDX_BODY_B,
    EQ_IDX_ANCHOR_AX,
    EQ_IDX_ANCHOR_AY,
    EQ_IDX_ANCHOR_AZ,
    EQ_IDX_ANCHOR_BX,
    EQ_IDX_ANCHOR_BY,
    EQ_IDX_ANCHOR_BZ,
    EQ_IDX_RELPOSE_X,
    EQ_IDX_RELPOSE_Y,
    EQ_IDX_RELPOSE_Z,
    EQ_IDX_RELPOSE_W,
    EQ_IDX_SOLREF_0,
    EQ_IDX_SOLREF_1,
    EQ_IDX_SOLIMP_0,
    EQ_IDX_SOLIMP_1,
    EQ_IDX_SOLIMP_2,
    EQ_IDX_SOLIMP_3,
    EQ_IDX_SOLIMP_4,
    EQ_IDX_TORQUESCALE,
    SITE_IDX_BODY,
    SITE_IDX_POS_X,
    MODEL_TENDON_SIZE,
    TENDON_MAX_WRAPS,
    TENDON_MAX_SPATIAL_WRAPS,
    TENDON_IDX_KIND,
    TENDON_IDX_IS_EQUALITY,
    TENDON_IDX_NUM_JOINTS,
    TENDON_IDX_JOINT_0,
    TENDON_IDX_COEF_0,
    TENDON_IDX_LENGTH_REF,
    TENDON_IDX_NUM_WRAPS,
    TENDON_IDX_WOBJ_0,
    TENDON_IDX_WTYPE_0,
    TENDON_IDX_WPRM_0,
    TENDON_IDX_LIMITED,
    TENDON_IDX_RANGE_MIN,
    TENDON_IDX_RANGE_MAX,
    TENDON_IDX_MARGIN,
    TENDON_IDX_SOLREF_0,
    TENDON_IDX_SOLREF_1,
    TENDON_IDX_SOLIMP_0,
    TENDON_IDX_SOLIMP_1,
    TENDON_IDX_SOLIMP_2,
    TENDON_IDX_SOLIMP_3,
    TENDON_IDX_SOLIMP_4,
    TENDON_IDX_SOLREF_LIM_0,
    TENDON_IDX_SOLREF_LIM_1,
    TENDON_IDX_SOLIMP_LIM_0,
    TENDON_IDX_SOLIMP_LIM_1,
    TENDON_IDX_SOLIMP_LIM_2,
    TENDON_IDX_SOLIMP_LIM_3,
    TENDON_IDX_SOLIMP_LIM_4,
    SITE_IDX_POS_Y,
    SITE_IDX_POS_Z,
    SITE_IDX_TYPE,
    SITE_IDX_SIZE_0,
    SITE_IDX_SIZE_1,
    SITE_IDX_SIZE_2,
    SITE_IDX_QUAT_W,
    SITE_IDX_QUAT_X,
    SITE_IDX_QUAT_Y,
    SITE_IDX_QUAT_Z,
    MODEL_ACTUATOR_SIZE,
    ACTDAMP_TRN_SIZE,
    ACTDAMP_IDX_N,
    ACTDAMP_IDX_DOF_0,
    ACTDAMP_IDX_PAIR_0,
    ACT_IDX_KIND,
    ACT_IDX_GEAR,
    ACT_IDX_CTRL_MIN,
    ACT_IDX_CTRL_MAX,
    ACT_IDX_CTRL_LIMITED,
    ACT_IDX_FORCE_MIN,
    ACT_IDX_FORCE_MAX,
    ACT_IDX_FORCE_LIMITED,
    ACT_IDX_KP,
    ACT_IDX_BIAS0,
    ACT_IDX_BIAS1,
    ACT_IDX_KV,
    ACT_IDX_DYN_TAU,
    ACT_IDX_DYN_TYPE,
    ACT_IDX_ACT_LIMITED,
    ACT_IDX_ACT_MIN,
    ACT_IDX_ACT_MAX,
    ACT_IDX_ACT_ADR,
    ACT_IDX_BODY_ID,
    ACT_IDX_PID_KI,
    ACT_IDX_PID_KD,
    ACT_IDX_PID_IMAX,
    ACT_IDX_PID_SLEW,
    ACT_IDX_TRN_N,
    ACT_IDX_DOF_ADR,
    ACT_IDX_TENDON_ID,
    ACT_IDX_SITE_ID,
    ACT_IDX_GEAR_1,
    ACT_IDX_GEAR_2,
    ACT_IDX_GEAR_3,
    ACT_IDX_GEAR_4,
    ACT_IDX_GEAR_5,
    ACT_IDX_JOINT_ID,
    ACT_IDX_TRN_QADR_0,
    ACT_IDX_TRN_DADR_0,
    ACT_IDX_TRN_COEF_0,
    MODEL_ACT_TENDON_SIZE,
    ACTTEN_IDX_STIFFNESS,
    ACTTEN_IDX_DAMPING,
    ACTTEN_IDX_SPRING_LO,
    ACTTEN_IDX_SPRING_HI,
    ACTTEN_IDX_TRN_N,
    ACTTEN_IDX_TRN_QADR_0,
    ACTTEN_IDX_TRN_DADR_0,
    ACTTEN_IDX_TRN_COEF_0,
    POSE_META_SIZE,
    POSE_IDX_QPOS0_NQ,
    POSE_IDX_FREE_JOINT_QPOS_ADR,
    KEY_META_SIZE,
    KEY_IDX_TIME,
    KEY_IDX_NQPOS,
    KEY_IDX_NQVEL,
    KEY_IDX_NCTRL,
    JLIM_SIZE,
    JLIM_IDX_LIMITED,
    JLIM_IDX_QPOS_ADR,
    JLIM_IDX_DOF_ADR,
    JLIM_IDX_ACTFRC_LIMITED,
    JLIM_IDX_ACTFRC_MIN,
    JLIM_IDX_ACTFRC_MAX,
    JLIM_IDX_RANGE_MIN,
    JLIM_IDX_RANGE_MAX,
)
from .mesh_bvh_build import build_mesh_bvh
from .flat_model import (
    FlatModelDef, _EQ_CONNECT, _EQ_WELD, _EQ_JOINT,
    ACT_KIND_POSITION, ACT_KIND_VELOCITY,
)
from .flat_model import (
    CAM_MODE_FIXED as _P_CAM_FIXED,
    CAM_MODE_TRACK as _P_CAM_TRACK,
    CAM_MODE_TRACKCOM as _P_CAM_TRACKCOM,
    CAM_MODE_TARGETBODY as _P_CAM_TARGETBODY,
    CAM_MODE_TARGETBODYCOM as _P_CAM_TARGETBODYCOM,
)

# ⚠⚠ TWO SPELLINGS OF ONE ENCODING, PAIRED HERE BECAUSE THIS IS THE ONLY FILE
# THAT SEES BOTH. The parser writes `CameraData.mode` and this file copies it
# into `CAM_IDX_MODE`; `raytrace/camera.mojo` dispatches on `gpu.constants`'
# copy. If the two ever drifted, a `trackcom` camera would be rendered as a
# `track` camera — still a picture, still plausible, and nothing downstream
# could tell. `raytrace` must not import the parser, so a shared definition is
# not available; an assert is.



def _jnt_qpos_size(jnt_type: Int) -> Int:
    if jnt_type == JNT_FREE:
        return 7
    elif jnt_type == JNT_BALL:
        return 4
    return 1


def _jnt_qvel_size(jnt_type: Int) -> Int:
    if jnt_type == JNT_FREE:
        return 6
    elif jnt_type == JNT_BALL:
        return 3
    return 1


def _geom_mass_and_inertia[
    DTYPE: DType
](
    g: Int,
    geom_type: Int,
    stored_mass: Scalar[DTYPE],
    radius: Scalar[DTYPE],
    half_length: Scalar[DTYPE],
    half_x: Scalar[DTYPE],
    half_y: Scalar[DTYPE],
    half_z: Scalar[DTYPE],
    geom_mesh_vol: List[Scalar[DTYPE]],
    geom_mesh_eig: List[Scalar[DTYPE]],
    stored_density: Scalar[DTYPE],
) -> Array[Scalar[DTYPE], 4]:
    """(mass, Ixx, Iyy, Izz) for one geom, meshes included.

    ⚠ A MESH's moments are `eigval * (mass / volume)`, and both halves matter:
    the eigenvalues are UNITLESS (length^5, volume-weighted), and the volume is
    the one MuJoCo recomputes in its inertia pass, not the one from its
    centre-of-mass pass. `geom_inertia`'s `else` branch returns (0, 0, 0) for a
    mesh — silently — which is what this exists to stop.
    """
    var out = Array[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    if geom_type == GEOM_MESH:
        var vol = geom_mesh_vol[g]
        if vol <= Scalar[DTYPE](1e-30):
            return out^
        # mass= wins; otherwise density * volume, the same rule
        # `geom_effective_mass` applies to every other type.
        # ⚠⚠ THE GEOM'S OWN DENSITY, NOT THE GLOBAL DEFAULT. This used to read
        # `MJ_DEFAULT_DENSITY` (1000) for every mesh, which threw away
        # `<geom density=...>` on exactly the geom type that cannot state a
        # mass any other way: `geom_volume` returns 0 for a mesh, so the parser
        # leaves `mass = -1` and the density is the ONLY thing carrying the
        # answer here.
        #
        # ⚠ `density="0"` IS THE COMMON CASE, NOT AN EXOTIC ONE. Menagerie's
        # `<default class="visual">` blocks set it so the render-only shells
        # weigh nothing, and every one of those was being charged 1000 kg/m^3.
        # trs_so_arm100's base carries the same `Base` mesh twice — once
        # visual, once collision — and came out at 1.168357 kg against
        # MuJoCo's 0.562466, i.e. the visual copy again plus the motor shell.
        # arx_l5's base read 1309.42 kg against 0.128420.
        var m = stored_mass
        if m < Scalar[DTYPE](0):
            m = stored_density * vol
        var dens = m / vol
        out[0] = m
        out[1] = geom_mesh_eig[g * 3 + 0] * dens
        out[2] = geom_mesh_eig[g * 3 + 1] * dens
        out[3] = geom_mesh_eig[g * 3 + 2] * dens
        return out^
    # ⚠ THE GEOM'S OWN DENSITY HERE TOO. `geom_effective_mass` falls back to
    # `MJ_DEFAULT_DENSITY`, which is the right number only for a geom that
    # never stated one — and `<default class="visual"><geom density="0"/>` is
    # the single most common thing a Menagerie model states.
    var m2 = stored_mass
    if m2 < Scalar[DTYPE](0):
        m2 = stored_density * geom_volume[DTYPE](
            geom_type, radius, half_length, half_x, half_y, half_z
        )
    var d = geom_inertia[DTYPE](
        geom_type, m2, radius, half_length, half_x, half_y, half_z
    )
    out[0] = m2
    out[1] = d[0]
    out[2] = d[1]
    out[3] = d[2]
    return out^


def _inertia_from_geoms_staging[
    DTYPE: DType,
](
    # ⚠ `NBODY`/`NGEOM` WERE COMPTIME PARAMETERS (3c-a). They are used for
    # nothing but five loop bounds, and being comptime is what forced every
    # caller to have a comptime provider — the last thing standing between
    # `build_model_fields_from_flat` and a runtime one. Making them runtime
    # also collapses one instantiation per model shape, which the comment
    # below already celebrates for `auto_mode`.
    nbody: Int,
    ngeom: Int,
    # ⚠ RUNTIME, NOT COMPTIME, since phase 1b. These three came off the raw
    # MJCF in the comptime interpreter, which is what pinned every model to a
    # `String` in Mojo source; they ride on `FlatModelDef` now. Making the auto
    # flag runtime also collapses two instantiations of this function into one.
    inertia_group_min: Int,
    inertia_group_max: Int,
    auto_mode: Bool,
    geoms: List[Scalar[DTYPE]],  # packed [NGEOM * MODEL_GEOM_SIZE] records
    bodies: List[Scalar[DTYPE]],  # packed [NBODY * MODEL_BODY_SIZE] records
    geom_mass: List[Scalar[DTYPE]],  # build-only (-1 = use density*volume)
    # ⚠ AND THE DENSITY THAT `-1` REFERS TO. A mesh geom always arrives with
    # `mass = -1` (the parser cannot compute its volume), so without this the
    # only thing left to multiply by is a constant.
    geom_density: List[Scalar[DTYPE]],  # build-only
    geom_group: List[Int],  # build-only (inertiagrouprange filter)
    # Mesh geoms only. `geom_inertia` takes scalar dims and cannot reach a
    # vertex list, so a mesh's volume and UNITLESS principal moments are
    # resolved during the geom pass and read back here. Zero for every
    # non-mesh geom, and for a mesh whose frame was never computed (see the
    # cache filter in `build_model_fields_from_flat`).
    geom_mesh_vol: List[Scalar[DTYPE]],
    geom_mesh_eig: List[Scalar[DTYPE]],  # [NGEOM * 3]
    body_has_explicit_inertia: List[Bool],
    mut body_mass: List[Scalar[DTYPE]],
    mut body_inv_mass: List[Scalar[DTYPE]],
    mut body_inertia: List[Scalar[DTYPE]],
    mut body_inv_inertia: List[Scalar[DTYPE]],
    mut body_ipos: List[Scalar[DTYPE]],
    mut body_iquat: List[Scalar[DTYPE]],
):
    """`compute_inertia_from_geoms` (MuJoCo inertiafromgeom) ported onto the
    build staging arrays + packed geom records — arithmetic verbatim from the
    legacy Model-typed routine (deleted at G4)."""
    for body_id in range(1, nbody):
        if auto_mode:
            if body_has_explicit_inertia[body_id]:
                continue

        var bo = body_id * MODEL_BODY_SIZE
        var total_mass = Scalar[DTYPE](0)
        var num_contributing = 0

        for g in range(ngeom):
            var go = g * MODEL_GEOM_SIZE
            var ggrp = geom_group[g]
            if ggrp < inertia_group_min or ggrp > inertia_group_max:
                continue
            if Int(geoms[go + GEOM_IDX_BODY]) == body_id:
                var gmi0 = _geom_mass_and_inertia[DTYPE](
                    g,
                    Int(geoms[go + GEOM_IDX_TYPE]),
                    geom_mass[g],
                    geoms[go + GEOM_IDX_RADIUS],
                    geoms[go + GEOM_IDX_HALF_LENGTH],
                    geoms[go + GEOM_IDX_HALF_X],
                    geoms[go + GEOM_IDX_HALF_Y],
                    geoms[go + GEOM_IDX_HALF_Z],
                    geom_mesh_vol,
                    geom_mesh_eig,
                    geom_density[g],
                )
                var gm = gmi0[0]
                if gm > Scalar[DTYPE](1e-10):
                    num_contributing += 1
                    total_mass += gm

        if num_contributing == 0:
            # ⚠ A BODY WITH NOTHING TO WEIGH HAS ZERO MASS, NOT THE DEFAULT ONE.
            #
            # This used to `continue`, leaving `BodyData`'s constructor defaults
            # (mass = 1.0, diaginertia = 0.01) standing. MuJoCo assigns 0/0:
            # under `inertiafromgeom="auto"` a body with no explicit `<inertial>`
            # takes its inertia from geoms, and if NONE qualify there is nothing
            # to take, so it stays massless.
            #
            # "None qualify" is not the same as "no geoms". `inertiagrouprange`
            # (default "0 5") filters by geom GROUP, so a body can be covered in
            # geometry and still weigh nothing. MetaWorld's sawyer sets
            # `inertiagrouprange="4 5"`, which excludes the group-1 visual shells:
            # `hand` (the mocap weld's target), `rightpad` and `leftpad` each own
            # a group-1 box and so each carried a PHANTOM KILOGRAM.
            #
            # That is defect 28. Three phantom kg at the wrist, ~1.15 m from the
            # j0 axis, added 3 x 1.15^2 = 3.97 kg m^2 to the base joint —
            # `dof_invweight0[j0]` was 0.4664x MuJoCo's, and the error compounded
            # down the chain to 0.0249x at the hand. body_invweight0 feeds
            # diagApprox, so the weld's R was 4.4x too large and the constraint
            # 4.4x too soft: the commanded hand pose sagged 77.6 mm where MuJoCo
            # tracks to 3.8 mm. It reached the weld through the constraint
            # regularizer, but the phantom mass was in the MASS MATRIX — every
            # force, acceleration and contact on those bodies was wrong too.
            body_mass[body_id] = Scalar[DTYPE](0)
            body_inv_mass[body_id] = Scalar[DTYPE](0)
            for c in range(3):
                body_inertia[body_id * 3 + c] = Scalar[DTYPE](0)
                body_inv_inertia[body_id * 3 + c] = Scalar[DTYPE](0)

            # ...and its INERTIAL FRAME is the body's own frame, not the origin.
            #
            # `mjCBody::Compile` (user_objects.cc:2738): once `InertiaFromGeom`
            # has run and left `ipos` undefined, MuJoCo copies `pos`/`quat` into
            # `ipos`/`iquat`. Note it copies the PARENT-frame pos into a
            # BODY-frame field — quirk faithfully reproduced, because the value
            # is observable.
            #
            # It is observable through `xipos`, which `mj_jacBodyCom` uses as
            # the Jacobian reference point for `body_invweight0`. Zeroing the
            # mass without this left the hand's translational weight at 2.168
            # against MuJoCo's 6.106 — still 2.8x off, in the opposite
            # direction, i.e. a weld 2.8x too STIFF where it had been 4.4x too
            # soft. Verified across every zero-mass body in sawyer: ipos == pos
            # and iquat == quat for all twelve of them.
            body_ipos[body_id * 3 + 0] = bodies[bo + BODY_IDX_POS_X]
            body_ipos[body_id * 3 + 1] = bodies[bo + BODY_IDX_POS_Y]
            body_ipos[body_id * 3 + 2] = bodies[bo + BODY_IDX_POS_Z]
            body_iquat[body_id * 4 + 0] = bodies[bo + BODY_IDX_QUAT_X]
            body_iquat[body_id * 4 + 1] = bodies[bo + BODY_IDX_QUAT_Y]
            body_iquat[body_id * 4 + 2] = bodies[bo + BODY_IDX_QUAT_Z]
            body_iquat[body_id * 4 + 3] = bodies[bo + BODY_IDX_QUAT_W]
            continue

        if num_contributing == 1:
            for g in range(ngeom):
                var go = g * MODEL_GEOM_SIZE
                var ggrp1 = geom_group[g]
                if ggrp1 < inertia_group_min or ggrp1 > inertia_group_max:
                    continue
                if Int(geoms[go + GEOM_IDX_BODY]) == body_id:
                    var gmi = _geom_mass_and_inertia[DTYPE](
                        g,
                        Int(geoms[go + GEOM_IDX_TYPE]),
                        geom_mass[g],
                        geoms[go + GEOM_IDX_RADIUS],
                        geoms[go + GEOM_IDX_HALF_LENGTH],
                        geoms[go + GEOM_IDX_HALF_X],
                        geoms[go + GEOM_IDX_HALF_Y],
                        geoms[go + GEOM_IDX_HALF_Z],
                        geom_mesh_vol,
                        geom_mesh_eig,
                        geom_density[g],
                    )
                    var gm = gmi[0]
                    if gm > Scalar[DTYPE](1e-10):
                        body_ipos[body_id * 3 + 0] = geoms[go + GEOM_IDX_POS_X]
                        body_ipos[body_id * 3 + 1] = geoms[go + GEOM_IDX_POS_Y]
                        body_ipos[body_id * 3 + 2] = geoms[go + GEOM_IDX_POS_Z]
                        body_iquat[body_id * 4 + 0] = geoms[
                            go + GEOM_IDX_QUAT_X
                        ]
                        body_iquat[body_id * 4 + 1] = geoms[
                            go + GEOM_IDX_QUAT_Y
                        ]
                        body_iquat[body_id * 4 + 2] = geoms[
                            go + GEOM_IDX_QUAT_Z
                        ]
                        body_iquat[body_id * 4 + 3] = geoms[
                            go + GEOM_IDX_QUAT_W
                        ]
                        body_mass[body_id] = gm
                        body_inv_mass[body_id] = Scalar[DTYPE](1.0) / gm
                        var inertia = Array[Scalar[DTYPE], 3](
                            fill=Scalar[DTYPE](0)
                        )
                        inertia[0] = gmi[1]
                        inertia[1] = gmi[2]
                        inertia[2] = gmi[3]
                        body_inertia[body_id * 3 + 0] = inertia[0]
                        body_inertia[body_id * 3 + 1] = inertia[1]
                        body_inertia[body_id * 3 + 2] = inertia[2]
                        body_inv_inertia[body_id * 3 + 0] = (
                            Scalar[DTYPE](1.0) / inertia[0]
                        )
                        body_inv_inertia[body_id * 3 + 1] = (
                            Scalar[DTYPE](1.0) / inertia[1]
                        )
                        body_inv_inertia[body_id * 3 + 2] = (
                            Scalar[DTYPE](1.0) / inertia[2]
                        )
                        break
        else:
            var com_x = Scalar[DTYPE](0)
            var com_y = Scalar[DTYPE](0)
            var com_z = Scalar[DTYPE](0)
            for g in range(ngeom):
                var go = g * MODEL_GEOM_SIZE
                var ggrp2 = geom_group[g]
                if ggrp2 < inertia_group_min or ggrp2 > inertia_group_max:
                    continue
                if Int(geoms[go + GEOM_IDX_BODY]) == body_id:
                    var gmi = _geom_mass_and_inertia[DTYPE](
                        g,
                        Int(geoms[go + GEOM_IDX_TYPE]),
                        geom_mass[g],
                        geoms[go + GEOM_IDX_RADIUS],
                        geoms[go + GEOM_IDX_HALF_LENGTH],
                        geoms[go + GEOM_IDX_HALF_X],
                        geoms[go + GEOM_IDX_HALF_Y],
                        geoms[go + GEOM_IDX_HALF_Z],
                        geom_mesh_vol,
                        geom_mesh_eig,
                        geom_density[g],
                    )
                    var gm = gmi[0]
                    if gm > Scalar[DTYPE](1e-10):
                        com_x += gm * geoms[go + GEOM_IDX_POS_X]
                        com_y += gm * geoms[go + GEOM_IDX_POS_Y]
                        com_z += gm * geoms[go + GEOM_IDX_POS_Z]
            com_x /= total_mass
            com_y /= total_mass
            com_z /= total_mass

            body_ipos[body_id * 3 + 0] = com_x
            body_ipos[body_id * 3 + 1] = com_y
            body_ipos[body_id * 3 + 2] = com_z

            body_mass[body_id] = total_mass
            body_inv_mass[body_id] = Scalar[DTYPE](1.0) / total_mass

            var toti = Array[Scalar[DTYPE], 6](fill=Scalar[DTYPE](0))

            for g in range(ngeom):
                var go = g * MODEL_GEOM_SIZE
                var ggrp3 = geom_group[g]
                if ggrp3 < inertia_group_min or ggrp3 > inertia_group_max:
                    continue
                if Int(geoms[go + GEOM_IDX_BODY]) == body_id:
                    var gmi = _geom_mass_and_inertia[DTYPE](
                        g,
                        Int(geoms[go + GEOM_IDX_TYPE]),
                        geom_mass[g],
                        geoms[go + GEOM_IDX_RADIUS],
                        geoms[go + GEOM_IDX_HALF_LENGTH],
                        geoms[go + GEOM_IDX_HALF_X],
                        geoms[go + GEOM_IDX_HALF_Y],
                        geoms[go + GEOM_IDX_HALF_Z],
                        geom_mesh_vol,
                        geom_mesh_eig,
                        geom_density[g],
                    )
                    var gm = gmi[0]
                    if gm > Scalar[DTYPE](1e-10):
                        var diag = Array[Scalar[DTYPE], 3](
                            fill=Scalar[DTYPE](0)
                        )
                        diag[0] = gmi[1]
                        diag[1] = gmi[2]
                        diag[2] = gmi[3]

                        var inert_global = Array[Scalar[DTYPE], 6](
                            fill=Scalar[DTYPE](0)
                        )
                        globalinertia(
                            diag[0],
                            diag[1],
                            diag[2],
                            geoms[go + GEOM_IDX_QUAT_X],
                            geoms[go + GEOM_IDX_QUAT_Y],
                            geoms[go + GEOM_IDX_QUAT_Z],
                            geoms[go + GEOM_IDX_QUAT_W],
                            inert_global,
                        )

                        var dx = geoms[go + GEOM_IDX_POS_X] - com_x
                        var dy = geoms[go + GEOM_IDX_POS_Y] - com_y
                        var dz = geoms[go + GEOM_IDX_POS_Z] - com_z
                        var inert_offset = Array[Scalar[DTYPE], 6](
                            fill=Scalar[DTYPE](0)
                        )
                        offcenter(gm, dx, dy, dz, inert_offset)

                        for j in range(6):
                            toti[j] += inert_global[j] + inert_offset[j]

            var eig = eig3_symmetric(toti)
            body_inertia[body_id * 3 + 0] = eig[0]
            body_inertia[body_id * 3 + 1] = eig[1]
            body_inertia[body_id * 3 + 2] = eig[2]
            body_inv_inertia[body_id * 3 + 0] = Scalar[DTYPE](1.0) / eig[0]
            body_inv_inertia[body_id * 3 + 1] = Scalar[DTYPE](1.0) / eig[1]
            body_inv_inertia[body_id * 3 + 2] = Scalar[DTYPE](1.0) / eig[2]
            body_iquat[body_id * 4 + 0] = eig[3]
            body_iquat[body_id * 4 + 1] = eig[4]
            body_iquat[body_id * 4 + 2] = eig[5]
            body_iquat[body_id * 4 + 3] = eig[6]



def apply_auto_spring_damper[
    DTYPE: DType,
    D: DimsLike,
](fmd: FlatModelDef, mut mf: Model[DTYPE, D]) raises:
    """`<joint springdamper>` -> stiffness/damping, from `dof_invweight0`.

    ⚠ EXTRACTED FROM `ModelDefFromXML.init_fields` (3c-c), NOT COPIED.
    The runtime loader has to run this too — it OVERWRITES the joint
    records that the XML supplied — and a second spelling of the same
    formula is precisely how this tree has lost days before. One body,
    two callers.

    ⚠ ORDER IS LOAD-BEARING and unchanged: this READS `dof_invweight0`,
    so it must run AFTER `compute_invweight0` and before any upload —
    the same order MuJoCo uses (`mj_setConst` then `AutoSpringDamper`).
    """
    # ── AutoSpringDamper (mjCModel::AutoSpringDamper, user_model.cc:2369)
    #
    # ⚠ ORDER IS LOAD-BEARING: this READS `dof_invweight0`, so it must run
    # AFTER `compute_invweight0` and before the upload. MuJoCo does exactly
    # the same thing — `mj_setConst(m, d)` then `AutoSpringDamper(m)`
    # (user_model.cc:5242-5245).
    #
    # `<joint springdamper="timeconst dampratio">` asks MuJoCo to DERIVE
    # the spring from the body's own inertia rather than take a number:
    #
    #     inertia   = ndim / sum(dof_invweight0[adr .. adr+ndim])
    #     stiffness = inertia / (timeconst^2 * dampratio^2)
    #     damping   = 2 * inertia / timeconst
    #
    # and it OVERWRITES whatever `stiffness`/`damping` the XML or class
    # supplied. dm_control's dog declares `springdamper="0.001 50"` once,
    # in a default class, which is why ~20 of its `jnt_stiffness` values
    # (0.0400187, 0.0401469, ...) appear NOWHERE in the XML and why our
    # stiffness was wrong before this. It is not a cosmetic mismatch: the
    # same formula sets `dof_damping`, so getting it wrong is a passive-
    # force error, i.e. a dynamics defect.
    #
    # Both parameters must be strictly positive for MuJoCo to act, which
    # is why (0, 0) is a sufficient "absent" encoding.
    comptime _MJMINVAL = 1e-15
    for j in range(mf.dims.get_njoint()):
        # `mjCJoint::nv` — dofs per joint type. Free is 6 (not 7: qpos and
        # dof sizes differ for a free joint, and this formula wants DOFS).

        var jd = fmd.joints[j]
        var tc = jd.springdamper_0
        var dr = jd.springdamper_1
        if tc <= 0.0 or dr <= 0.0:
            continue
        var jo = j * MODEL_JOINT_SIZE
        var dof_adr = Int(mf.joints.data[jo + JOINT_IDX_DOF_ADR])
        var jt = Int(mf.joints.data[jo + JOINT_IDX_TYPE])
        var ndim = 6 if jt == JNT_FREE else (3 if jt == JNT_BALL else 1)
        var acc = Float64(0)
        for i in range(ndim):
            acc += Float64(mf.dof_invweight0.data[dof_adr + i])
        if acc < _MJMINVAL:
            acc = _MJMINVAL
        var inertia = Float64(ndim) / acc
        var denom = tc * tc * dr * dr
        if denom < _MJMINVAL:
            denom = _MJMINVAL
        var tc_d = tc if tc > _MJMINVAL else _MJMINVAL
        mf.joints.data[jo + JOINT_IDX_STIFFNESS] = Scalar[DTYPE](
            inertia / denom
        )
        mf.joints.data[jo + JOINT_IDX_DAMPING] = Scalar[DTYPE](
            2.0 * inertia / tc_d
        )


@always_inline
def _dofs_are_ancestor_related[
    DTYPE: DType, D: DimsLike
](dof_body: List[Int], mf: Model[DTYPE, D], i: Int, j: Int) -> Bool:
    """Is `(i, j)` a slot MuJoCo's `qDeriv` sparsity actually has?

    ⚠⚠ THIS IS NOT AN OPTIMISATION, IT IS THE SEMANTICS. `addJTBJSparse`
    (`engine_derivative.c:768`) accumulates the actuator's `moment (x) moment`
    into `d->qDeriv` through `mju_addToSclSparseInc`, which writes only the
    columns the `D` pattern lists for that row and **silently drops the rest**.
    `D` is the mass matrix's pattern, and a tree's mass matrix couples two dofs
    only when one's body is an ancestor of the other's.

    So a tendon spanning two SIBLING branches — every 2-dof gripper in this
    tree — contributes its diagonal and nothing else, while a serial chain
    (hello_robot_stretch's four telescoping links) gets the full block.
    Writing the whole outer product regardless took the Menagerie sweep from
    74/85 to 72/85.

    ⚠ Checked rather than assumed: over **74,671** dof pairs across all 85
    scenes, this predicate and membership in `D_colind` agree every time.
    """
    if i < 0 or j < 0 or i >= len(dof_body) or j >= len(dof_body):
        return False
    var bi = dof_body[i]
    var bj = dof_body[j]
    # walk bi's parents looking for bj, then the other way round
    var c = bi
    while c > 0:
        if c == bj:
            return True
        c = Int(mf.bodies.data[c * MODEL_BODY_SIZE + BODY_IDX_PARENT])
    c = bj
    while c > 0:
        if c == bi:
            return True
        c = Int(mf.bodies.data[c * MODEL_BODY_SIZE + BODY_IDX_PARENT])
    return False


def build_actuator_damping[
    DTYPE: DType,
    D: DimsLike,
    D2: DimsLike,
](
    fmd: FlatModelDef,
    mut mf: Model[DTYPE, D],
    mut sf: SpecFields[DTYPE, D2],
) raises:
    """Resolve `<position dampratio>` to a `kv`, then bank `dof_actdamp`.

    ⚠ THE ACTUATOR HAS NO DAMPING UNTIL THIS RUNS. `dampratio` is the ONLY
    way g1 and 30-odd other Menagerie models state their servo damping, and
    an unparsed one leaves `kv = 0` — an undamped 500 N.m/rad spring. Driven
    by anything (a policy, a nudge, a contact) it pumps energy with nothing to
    remove it: g1 under a random policy left the ground and reached 2.9 m,
    while the SAME model with no actions sat still, because an undamped spring
    at its equilibrium is indistinguishable from a damped one.

    THE FORMULA (`engine_setconst.c:998-1035`), which is why it cannot live in
    the parser: the reflected inertia does not exist until M does.

        mass = sum over the transmission dofs of  dof_M0[dof] / trn^2
        kv   = dampratio * 2 * sqrt(kp * mass)

    where `trn` is the actuator's moment entry, `gear * coef`, and the sum
    skips entries whose `trn^2` is below mjMINVAL — a zero-coefficient dof
    contributes nothing and would divide by zero.

    ⚠ ORDER IS LOAD-BEARING, exactly as for `apply_auto_spring_damper`: this
    READS `mf.dof_M0`, so it must run AFTER `compute_invweight0` and before
    any upload. MuJoCo has the same ordering for the same reason — the
    conversion is at the END of `mj_setConst`, after the mass matrix.

    ⚠ POSITION-LIKE ONLY. MuJoCo gates on `gainprm[0] == -biasprm[1]`, which
    is what makes an actuator a position servo; `<velocity>` and `<motor>` do
    not admit `dampratio` at all (`xml_native_reader.cc:346`). The parser only
    sets `dampratio` on `<position>`, so the loop below is already restricted
    — but a non-position actuator carrying one would be a parser bug, not
    something to silently convert.
    """
    # mjMINVAL — the same floor `apply_auto_spring_damper` uses, spelled
    # locally because that one is a function-local comptime.
    comptime _MINVAL = 1e-15
    var nv = mf.dims.get_nv()
    var nact = len(fmd.actuators)
    # ⚠ SAME RULE AS THE MESH AND HEIGHTFIELD BUDGETS: a capacity that is too
    # small is announced HERE, at model build, naming the knob. Without this
    # an under-declared `nact` surfaces as a bounds assert inside
    # `dynamics/actuation.mojo` on the first step — a per-step hot path that
    # cannot afford a check and should never have been the one to notice.
    # `test_model_def_from_xml` spelled a `Dims[...]` by hand, omitted `nact`,
    # and got exactly that: "index 1 is out of bounds" from a file that has
    # nothing to do with actuator COUNTS.
    if nact > mf.dims.get_nact():
        raise Error(
            String("actuator capacity exceeded — nact = ")
            + String(mf.dims.get_nact()) + " but this model declares "
            + String(nact) + " actuators. Set `nact` on this model's dims."
        )
    for i in range(nact):
        var dr = fmd.actuators[i].dampratio
        if dr <= 0.0:
            continue
        var o = i * MODEL_ACTUATOR_SIZE
        var gear = Float64(sf.actuators.data[o + ACT_IDX_GEAR])
        var n = Int(sf.actuators.data[o + ACT_IDX_TRN_N])
        var mass = Float64(0)
        for k in range(n):
            var dadr = Int(sf.actuators.data[o + ACT_IDX_TRN_DADR_0 + k])
            if dadr < 0 or dadr >= nv:
                continue
            var trn = gear * Float64(
                sf.actuators.data[o + ACT_IDX_TRN_COEF_0 + k]
            )
            var trn2 = trn * trn
            if trn2 > _MINVAL:
                mass += Float64(mf.dof_M0.data[dadr]) / trn2
        var kp = Float64(sf.actuators.data[o + ACT_IDX_KP])
        var kpm = kp * mass
        if kpm < 0.0:
            kpm = 0.0
        sf.actuators.data[o + ACT_IDX_KV] = Scalar[DTYPE](
            dr * 2.0 * sqrt(kpm)
        )

    # ── dof_actdamp — the servo damping the IMPLICIT integrators fold in ──
    #
    # ⚠ SECOND LOOP, AND IT MUST STAY SECOND. It reads the FINAL kv of every
    # actuator, which for a `dampratio` model only exists after the loop
    # above. Merging the two would bank the pre-conversion 0 for exactly the
    # actuators this whole pass is about.
    for i in range(nv):
        mf.dof_actdamp.data[i] = Scalar[DTYPE](0)
    # ⚠ AND ITS OFF-DIAGONAL COMPANION, cleared over the WHOLE capacity — the
    # record is read by actuator index and an unwritten one must say `n = 0`.
    for i in range(mf.dims.get_nact() * ACTDAMP_TRN_SIZE):
        mf.actdamp_trn.data[i] = Scalar[DTYPE](0)
    # dof -> owning body, for the ancestor test below. Built from the joint
    # records rather than stored, because it is only wanted here.
    var dof_body = List[Int](length=nv if nv > 0 else 1, fill=0)
    for j in range(mf.dims.get_njoint()):
        var jo = j * MODEL_JOINT_SIZE
        var jb = Int(mf.joints.data[jo + JOINT_IDX_BODY_ID])
        var jda = Int(mf.joints.data[jo + JOINT_IDX_DOF_ADR])
        var jt = Int(mf.joints.data[jo + JOINT_IDX_TYPE])
        var jnd = 1
        if jt == JNT_FREE:
            jnd = 6
        elif jt == JNT_BALL:
            jnd = 3
        for k in range(jnd):
            if jda + k >= 0 and jda + k < nv:
                dof_body[jda + k] = jb
    for i in range(nact):
        var o = i * MODEL_ACTUATOR_SIZE
        var kind = Int(sf.actuators.data[o + ACT_IDX_KIND])
        # ⚠ ONLY THESE TWO CARRY A `-kv*vel` TERM. A `<motor>` has no velocity
        # feedback at all, so banking a kv for it would invent damping the
        # reference does not have — `apply_actions_fields` applies the term
        # under this same test.
        if kind != ACT_KIND_POSITION and kind != ACT_KIND_VELOCITY:
            continue
        var kv = Float64(sf.actuators.data[o + ACT_IDX_KV])
        if kv == 0.0:
            continue
        var gear = Float64(sf.actuators.data[o + ACT_IDX_GEAR])
        var n = Int(sf.actuators.data[o + ACT_IDX_TRN_N])
        # ⚠ `ao` IS INDEXED BY ACTUATOR, `dof_actdamp` BY DOF. The two halves
        # of `mjd_actuator_vel` are banked in the same pass so they cannot
        # disagree about which actuators have velocity feedback.
        var ao = i * ACTDAMP_TRN_SIZE
        var ndof = 0
        var moms = List[Float64]()
        var dofs = List[Int]()
        for k in range(n):
            var dadr = Int(sf.actuators.data[o + ACT_IDX_TRN_DADR_0 + k])
            if dadr < 0 or dadr >= nv:
                continue
            var trn = gear * Float64(
                sf.actuators.data[o + ACT_IDX_TRN_COEF_0 + k]
            )
            mf.dof_actdamp.data[dadr] += Scalar[DTYPE](kv * trn * trn)
            # ⚠ ONLY THE RESOLVED DOFS, PACKED. `k` skips an unresolvable
            # `dadr`, so the record's slot index is `ndof`, not `k` — writing
            # at `k` would leave a hole the integrator reads as dof 0.
            if ndof < TENDON_MAX_WRAPS:
                mf.actdamp_trn.data[ao + ACTDAMP_IDX_DOF_0 + ndof] = (
                    Scalar[DTYPE](dadr)
                )
                dofs.append(dadr)
                moms.append(trn)
            ndof += 1
        # ── the off-diagonal pairs, FILTERED BY THE SPARSITY MuJoCo WRITES
        #    INTO. See `ACTDAMP_TRN_SIZE`: `addJTBJSparse` accumulates into
        #    `qDeriv`'s `D` pattern and drops every column that pattern does
        #    not have, and `D` holds `(i, j)` only for ANCESTOR-RELATED dofs.
        #    Checked exhaustively against `D_colind`: over 74,671 dof pairs in
        #    all 85 Menagerie models the two agree every time.
        for pp in range(len(dofs)):
            for qq in range(len(dofs)):
                if pp == qq:
                    continue  # the diagonal is `dof_actdamp`'s
                if not _dofs_are_ancestor_related(
                    dof_body, mf, dofs[pp], dofs[qq]
                ):
                    continue
                mf.actdamp_trn.data[
                    ao + ACTDAMP_IDX_PAIR_0 + pp * TENDON_MAX_WRAPS + qq
                ] = Scalar[DTYPE](kv * moms[pp] * moms[qq])
        # ⚠ THE CAP IS `TENDON_MAX_WRAPS` AND IT IS THE SAME ONE THE
        # TRANSMISSION ITSELF USES, so a transmission that fits in the
        # actuator record fits here too. Recording a SHORT `n` rather than the
        # full one would silently drop the tail's off-diagonal terms; raising
        # instead, because there is no honest partial answer.
        if ndof > TENDON_MAX_WRAPS:
            raise Error(
                String(
                    "physics3d: actuator ", i, " resolves ", ndof,
                    " dofs but `Model.actdamp_trn` holds TENDON_MAX_WRAPS (",
                    TENDON_MAX_WRAPS, "). The actuator record itself is capped",
                    " at the same number, so this should be unreachable —",
                    " raise both together.",
                )
            )
        # ⚠ `kv` IS NOT STORED — it is already inside every `pair` entry.
        mf.actdamp_trn.data[ao + ACTDAMP_IDX_N] = Scalar[DTYPE](ndof)


def assert_no_pending_dampratio(fmd: FlatModelDef, who: String) raises:
    """Refuse to build actuators whose `dampratio` nothing will resolve.

    ⚠⚠ THE ALTERNATIVE IS A SILENT `kv = 0`, which is an UNDAMPED servo and
    the exact defect this pass exists to fix — so a `SpecFields` builder that
    has no `Model` to read `dof_M0` from must not quietly produce one. The
    comptime builders (`init_spec_fields` / `make_spec_fields`) re-parse the
    XML and hold no model, so they call this.

    No in-tree comptime model uses `dampratio` (so_arm100's XML carries a
    hand-converted `kv` and says so in a comment), which is why this is a
    guard rather than a second implementation. If one ever does, the fix is to
    give that path its model, not to loosen this.
    """
    for i in range(len(fmd.actuators)):
        if fmd.actuators[i].dampratio > 0.0:
            raise Error(
                String(
                    "physics3d: actuator #", i, " uses <position dampratio=",
                    fmd.actuators[i].dampratio,
                    ">, which resolves to a kv only against the mass matrix at"
                    " qpos0 (dof_M0) — and ", who, " has no Model to read it"
                    " from. Building anyway would leave kv = 0: an UNDAMPED"
                    " servo, which looks correct at rest and pumps energy"
                    " under any drive. Use the runtime loader"
                    " (`spec_fields_runtime`), which takes the model.",
                )
            )


def rbound_of(
    geom_type: Int, radius: Float64, half_length: Float64,
    half_x: Float64, half_y: Float64, half_z: Float64,
) -> Float64:
    """The broadphase bounding-sphere radius for one geom. ONE implementation.

    ⚠⚠ SHARED WITH THE STUDIO'S EDIT PATH ON PURPOSE. `rbound` is DERIVED
    from a geom's type and size, so an interactive resize has to recompute it
    — and a second copy of these five formulas is precisely the "two spellings
    of one quantity" failure this tree keeps paying for. A stale or divergent
    `rbound` is invisible: it is too small, the pair is rejected in the
    broadphase, and the contact simply never happens.

    A mesh's bound is measured from its hull and is NOT here; the builder
    overwrites this value for mesh geoms after loading them.
    """
    if geom_type == GEOM_PLANE:
        return 1e10  # planes are infinite
    if geom_type == GEOM_CAPSULE:
        return radius + half_length
    if geom_type == GEOM_CYLINDER:
        return sqrt(half_length * half_length + radius * radius)
    if geom_type == GEOM_BOX:
        return sqrt(half_x * half_x + half_y * half_y + half_z * half_z)
    if geom_type == GEOM_ELLIPSOID:
        # `max(size)` — mjCGeom::GetRBound (user_objects.cc:3345). Without
        # this case `rbound` fell through to `radius`, which the parser sets
        # to size[0] for an ellipsoid; any ellipsoid whose LARGEST semi-axis
        # is not the first would get a bounding sphere smaller than itself and
        # the broad phase would silently drop its contacts. Harmless until the
        # ellipsoid narrow phase landed (2026-07-31), because a colliding
        # ellipsoid was rejected at build time; a real bug from that point on.
        # quadruped's torso (.3 .27 .2) is ordered largest-first and so would
        # NOT have exposed it.
        var mx = half_x
        if half_y > mx:
            mx = half_y
        if half_z > mx:
            mx = half_z
        return mx
    return radius


def build_model_fields_from_flat[

    DTYPE: DType,
    # ⚠ THE FlatModelDef DIMS USED TO BE PARAMETERS HERE — all fourteen of
    # them. They are gone: `FlatModelDef` is non-generic (List-backed) since
    # 2026-08-05, so its counts come from the Lists themselves.
    #
    # What remains below is the MODEL side, and it is deliberate: these size
    # `fields.Model`'s tensors, which are the hot path. Do not confuse the two
    # — de-genericizing the PARSER did not make the ENGINE runtime-dimensioned.
    # Model dims (record capacities)
    # ⚠ THE FOUR `<compiler>` BUILD MODES USED TO BE PARAMETERS HERE. They
    # are gone: phase 1b moved them onto `FlatModelDef`, read from the same
    # parse that produced everything else this function fills. Hand-passing
    # them was its own bug source — a call site could state a mode its own XML
    # contradicts, and one did (`test_xml_full_parser` passed 0 = OFF for an
    # XML whose ABSENT attribute means MuJoCo's AUTO).
    # Appended rather than grouped with NEXCLUDE — see `fields.Model`.

    D: DimsLike,
](
    fmd: FlatModelDef,
    mut mf: Model[DTYPE, D],
) raises:
    """Fill every `mf` record tensor from the parsed `fmd` — see module
    docstring. Does NOT compute invweight0 and does NOT upload."""

    # ── meta ─────────────────────────────────────────────────────────────
    mf.meta.data[MODEL_META_IDX_NBODY] = Scalar[DTYPE](mf.dims.get_nbody())
    mf.meta.data[MODEL_META_IDX_NJOINT] = Scalar[DTYPE](mf.dims.get_njoint())
    mf.meta.data[MODEL_META_IDX_GRAVITY_X] = Scalar[DTYPE](fmd.gravity_x)
    mf.meta.data[MODEL_META_IDX_GRAVITY_Y] = Scalar[DTYPE](fmd.gravity_y)
    mf.meta.data[MODEL_META_IDX_GRAVITY_Z] = Scalar[DTYPE](fmd.gravity_z)
    # The env's advertised scalar action bounds — see the constant.
    mf.meta.data[MODEL_META_IDX_CTRL_MIN] = Scalar[DTYPE](
        fmd.default_motor_ctrl_min
    )
    mf.meta.data[MODEL_META_IDX_CTRL_MAX] = Scalar[DTYPE](
        fmd.default_motor_ctrl_max
    )
    mf.meta.data[MODEL_META_IDX_TIMESTEP] = Scalar[DTYPE](fmd.timestep)
    mf.meta.data[MODEL_META_IDX_DENSITY] = Scalar[DTYPE](fmd.opt_density)
    mf.meta.data[MODEL_META_IDX_VISCOSITY] = Scalar[DTYPE](fmd.opt_viscosity)
    mf.meta.data[MODEL_META_IDX_WIND_X] = Scalar[DTYPE](fmd.opt_wind_x)
    mf.meta.data[MODEL_META_IDX_WIND_Y] = Scalar[DTYPE](fmd.opt_wind_y)
    mf.meta.data[MODEL_META_IDX_WIND_Z] = Scalar[DTYPE](fmd.opt_wind_z)
    # ⚠⚠ WAS HARDCODED `1.0`, AND FIVE SOLVERS READ THIS SLOT. `contact_solve`,
    # `newton_solve` (twice), `cg_solve` and `island_pgs_solve` all take
    # `impratio` from here, so `<option impratio>` was parsed by nothing and
    # every model ran at 1 regardless of what it asked for. Every model in the
    # tree happened to use 1 — `contact_solve.mojo`'s note says so — which is
    # why the write and the reads agreed for as long as they did.
    mf.meta.data[MODEL_META_IDX_IMPRATIO] = Scalar[DTYPE](fmd.impratio)
    # `<option cone/solver/integrator>` — what the MODEL asked for. Nothing in
    # the engine dispatches on these yet (the cone and the solver are comptime
    # parameters of the integrator), which is exactly why they are recorded:
    # a tool can now compare what it built against what the file wanted instead
    # of having no way to know.
    mf.meta.data[MODEL_META_IDX_CONE] = Scalar[DTYPE](fmd.cone)
    mf.meta.data[MODEL_META_IDX_SOLVER] = Scalar[DTYPE](fmd.solver)
    mf.meta.data[MODEL_META_IDX_INTEGRATOR] = Scalar[DTYPE](fmd.integrator)
    mf.meta.data[MODEL_META_IDX_MAX_CONDIM] = Scalar[DTYPE](fmd.max_condim)
    # `<flag eulerdamp="disable"/>`. Written unconditionally, like the
    # `multiccd` flag below: a record slot that is only sometimes written is a
    # slot whose zero cannot be told from an absent attribute.
    mf.meta.data[MODEL_META_IDX_EULERDAMP_DISABLED] = Scalar[DTYPE](
        1.0 if fmd.eulerdamp_disabled else 0.0
    )
    # `mjModel.ngravcomp` — `engine_setconst.c:99-104` counts bodies with
    # `body_gravcomp > 0` STRICTLY, so a negative value (which the compiler
    # does not reject) would not raise the count and `mj_gravcomp` would then
    # skip the whole pass. Transcribed with the same `> 0`, not `!= 0`, so a
    # nonsense model diverges the same way in both engines rather than only in
    # ours.
    # ⚠ COUNTED FROM `fmd.bodies`, NOT FROM `mf.bodies` — this meta block runs
    # ~140 lines BEFORE the body records are written, so reading them back here
    # would count zeros on every model.
    var _ngravcomp = 0
    for _bi in range(len(fmd.bodies)):
        if _bi + 1 >= mf.dims.get_nbody():
            break
        if fmd.bodies[_bi].gravcomp > 0:
            _ngravcomp += 1
    mf.meta.data[MODEL_META_IDX_NGRAVCOMP] = Scalar[DTYPE](_ngravcomp)
    mf.meta.data[MODEL_META_IDX_NEQUALITY] = Scalar[DTYPE](
        len(fmd.equalities) if len(fmd.equalities) < mf.dims.get_nequality()
        else mf.dims.get_nequality()
    )
    # Honest tendon count. This used to be hardcoded 0, which made every
    # tendon record dead. `_tendon_env` treats a record as a BILATERAL
    # EQUALITY, so waking it up is safe only because that pass now also
    # requires TENDON_IDX_IS_EQUALITY — humanoid declares two <fixed> tendons
    # that MuJoCo constrains in no way.
    mf.meta.data[MODEL_META_IDX_NTENDON] = Scalar[DTYPE](
        len(fmd.tendons) if len(fmd.tendons) < mf.dims.get_ntendon()
        else mf.dims.get_ntendon()
    )
    mf.meta.data[MODEL_META_IDX_NEXCLUDE] = Scalar[DTYPE](len(fmd.excludes))
    mf.meta.data[MODEL_META_IDX_NPAIR] = Scalar[DTYPE](len(fmd.pairs))
    # ⚠ COPIED VERBATIM, INCLUDING 0. `<option noslip_tolerance="0">` is
    # dm_control's manipulation setting and means "run every noslip
    # iteration"; clamping it to a positive floor here would silently restore
    # the early exit. `_parse_option` supplies MuJoCo's 1e-6 default when the
    # attribute is absent, so a 0 that arrives here was WRITTEN by the model.
    mf.meta.data[MODEL_META_IDX_NOSLIP_TOLERANCE] = Scalar[DTYPE](
        fmd.noslip_tolerance
    )
    # ⚠⚠ THE COUNT, AND WITHOUT IT THE TOLERANCE ABOVE HAD NOTHING TO TRIM.
    # `mj_solNoSlip` runs `opt.noslip_iterations` friction-only sweeps; that
    # number reached the solver only as `solve_newton`'s comptime
    # `NOSLIP_ITER`, so every model built through this function at RUNTIME —
    # the studio's, and the fidelity harnesses' — stepped with the pass off no
    # matter what its `<option>` said. Written unconditionally for the same
    # reason the two below are: the slot has to differ between a model that
    # sets the attribute and one that does not, and 0 is the value MuJoCo
    # gives the second.
    mf.meta.data[MODEL_META_IDX_NOSLIP_ITERATIONS] = Scalar[DTYPE](
        fmd.noslip_iterations
    )
    # EPA's stopping rule. `Model.__init__` already seeded MuJoCo's defaults,
    # so this only ever OVERWRITES with what `<option>` actually said — but it
    # must run unconditionally, because a model that sets them and a model that
    # does not have to end up with different values in the same slot.
    mf.meta.data[MODEL_META_IDX_CCD_TOLERANCE] = Scalar[DTYPE](
        fmd.ccd_tolerance
    )
    mf.meta.data[MODEL_META_IDX_CCD_ITERATIONS] = Scalar[DTYPE](
        fmd.ccd_iterations
    )
    # The constraint solver's budget — see `MODEL_META_IDX_SOLVER_ITERATIONS`.
    # Written unconditionally beside the CCD pair for the same reason: an
    # absent attribute is MuJoCo's default, which the parser already resolved.
    mf.meta.data[MODEL_META_IDX_SOLVER_ITERATIONS] = Scalar[DTYPE](
        fmd.solver_iterations
    )
    mf.meta.data[MODEL_META_IDX_SOLVER_TOLERANCE] = Scalar[DTYPE](
        fmd.solver_tolerance
    )
    mf.meta.data[MODEL_META_IDX_LS_ITERATIONS] = Scalar[DTYPE](
        fmd.ls_iterations
    )
    mf.meta.data[MODEL_META_IDX_LS_TOLERANCE] = Scalar[DTYPE](
        fmd.ls_tolerance
    )
    # `<flag multiccd="disable"/>`. Written unconditionally for the same reason
    # the two above are: the slot has to differ between a model that sets the
    # flag and one that does not, and 0 (the MuJoCo default, feature ON) is
    # what an unwritten slot already means.
    mf.meta.data[MODEL_META_IDX_MULTICCD_DISABLED] = Scalar[DTYPE](
        1.0 if fmd.multiccd_disabled else 0.0
    )
    mf.meta.data[MODEL_META_IDX_FILTERPARENT_DISABLED] = Scalar[DTYPE](
        1.0 if fmd.filterparent_disabled else 0.0
    )
    # `<flag warmstart="disable"/>` — same disable-sense, same unconditional
    # write.
    mf.meta.data[MODEL_META_IDX_WARMSTART_DISABLED] = Scalar[DTYPE](
        1.0 if fmd.warmstart_disabled else 0.0
    )

    # Contact solref/solimp: MuJoCo model defaults, then geom[0]'s parsed
    # values (floor / first worldbody geom inherits <default><geom>).
    mf.meta.data[MODEL_META_IDX_SOLREF_CONTACT_0] = Scalar[DTYPE](0.02)
    mf.meta.data[MODEL_META_IDX_SOLREF_CONTACT_1] = Scalar[DTYPE](1.0)
    mf.meta.data[MODEL_META_IDX_SOLIMP_CONTACT_0] = Scalar[DTYPE](0.9)
    mf.meta.data[MODEL_META_IDX_SOLIMP_CONTACT_1] = Scalar[DTYPE](0.95)
    mf.meta.data[MODEL_META_IDX_SOLIMP_CONTACT_2] = Scalar[DTYPE](0.001)
    mf.meta.data[MODEL_META_IDX_SOLIMP_CONTACT_3] = Scalar[DTYPE](0.5)
    mf.meta.data[MODEL_META_IDX_SOLIMP_CONTACT_4] = Scalar[DTYPE](2.0)
    # ⚠ RUNTIME, AND GUARDING `fmd.geoms`, NOT `mf.dims.get_ngeom()` (3c-a). The thing
    # being indexed on the next line is the PARSED list, so its own length is
    # the honest guard — and `mf.dims.get_ngeom()` is `DIM_POISON` on a dynamic provider,
    # which skipped this block and left the contact solref/solimp meta at the
    # defaults seeded above.
    if len(fmd.geoms) > 0:
        var g0 = fmd.geoms[0]
        mf.meta.data[MODEL_META_IDX_SOLREF_CONTACT_0] = Scalar[DTYPE](
            g0.solref_0
        )
        mf.meta.data[MODEL_META_IDX_SOLREF_CONTACT_1] = Scalar[DTYPE](
            g0.solref_1
        )
        mf.meta.data[MODEL_META_IDX_SOLIMP_CONTACT_0] = Scalar[DTYPE](
            g0.solimp_0
        )
        mf.meta.data[MODEL_META_IDX_SOLIMP_CONTACT_1] = Scalar[DTYPE](
            g0.solimp_1
        )
        mf.meta.data[MODEL_META_IDX_SOLIMP_CONTACT_2] = Scalar[DTYPE](
            g0.solimp_2
        )
        mf.meta.data[MODEL_META_IDX_SOLIMP_CONTACT_3] = Scalar[DTYPE](
            g0.solimp_3
        )
        mf.meta.data[MODEL_META_IDX_SOLIMP_CONTACT_4] = Scalar[DTYPE](
            g0.solimp_4
        )

    # Limit solref/solimp defaults (overridden from joint 0 after the joint
    # fill, matching the legacy model-level sync).
    var def_solref_limit_0 = Scalar[DTYPE](0.02)
    var def_solref_limit_1 = Scalar[DTYPE](1.0)
    var def_solimp_limit_0 = Scalar[DTYPE](0.9)
    var def_solimp_limit_1 = Scalar[DTYPE](0.95)
    var def_solimp_limit_2 = Scalar[DTYPE](0.001)
    var def_solimp_limit_3 = Scalar[DTYPE](0.5)
    var def_solimp_limit_4 = Scalar[DTYPE](2.0)
    mf.meta.data[MODEL_META_IDX_SOLREF_LIMIT_0] = def_solref_limit_0
    mf.meta.data[MODEL_META_IDX_SOLREF_LIMIT_1] = def_solref_limit_1
    mf.meta.data[MODEL_META_IDX_SOLIMP_LIMIT_0] = def_solimp_limit_0
    mf.meta.data[MODEL_META_IDX_SOLIMP_LIMIT_1] = def_solimp_limit_1
    mf.meta.data[MODEL_META_IDX_SOLIMP_LIMIT_2] = def_solimp_limit_2
    mf.meta.data[MODEL_META_IDX_SOLIMP_LIMIT_3] = def_solimp_limit_3
    mf.meta.data[MODEL_META_IDX_SOLIMP_LIMIT_4] = def_solimp_limit_4

    # ── body staging (mass/inertia block; inertiafromgeom + settotalmass
    #    mutate it before the record write) ─────────────────────────────────
    var body_mass = List[Scalar[DTYPE]](length=mf.dims.get_nbody(), fill=Scalar[DTYPE](0))
    var body_inv_mass = List[Scalar[DTYPE]](
        length=mf.dims.get_nbody(), fill=Scalar[DTYPE](0)
    )
    var body_inertia = List[Scalar[DTYPE]](
        length=mf.dims.get_nbody() * 3, fill=Scalar[DTYPE](0)
    )
    var body_inv_inertia = List[Scalar[DTYPE]](
        length=mf.dims.get_nbody() * 3, fill=Scalar[DTYPE](0)
    )
    var body_ipos = List[Scalar[DTYPE]](
        length=mf.dims.get_nbody() * 3, fill=Scalar[DTYPE](0)
    )
    var body_iquat = List[Scalar[DTYPE]](
        length=mf.dims.get_nbody() * 4, fill=Scalar[DTYPE](0)
    )
    var body_has_explicit_inertia = List[Bool](length=mf.dims.get_nbody(), fill=False)
    var body_parent = List[Int](length=mf.dims.get_nbody(), fill=0)

    # Worldbody (index 0): mass/inertia zero, identity iquat.
    body_iquat[3] = Scalar[DTYPE](1)

    # Bodies 1..NBODY-1 from fmd (legacy `set_body` semantics: unguarded 1/x).
    for i in range(len(fmd.bodies)):
        var b = fmd.bodies[i]
        var bi = i + 1
        body_mass[bi] = Scalar[DTYPE](b.mass)
        body_inv_mass[bi] = Scalar[DTYPE](1.0) / Scalar[DTYPE](b.mass)
        body_inertia[bi * 3 + 0] = Scalar[DTYPE](b.ixx)
        body_inertia[bi * 3 + 1] = Scalar[DTYPE](b.iyy)
        body_inertia[bi * 3 + 2] = Scalar[DTYPE](b.izz)
        body_inv_inertia[bi * 3 + 0] = Scalar[DTYPE](1.0) / Scalar[DTYPE](
            b.ixx
        )
        body_inv_inertia[bi * 3 + 1] = Scalar[DTYPE](1.0) / Scalar[DTYPE](
            b.iyy
        )
        body_inv_inertia[bi * 3 + 2] = Scalar[DTYPE](1.0) / Scalar[DTYPE](
            b.izz
        )
        body_ipos[bi * 3 + 0] = Scalar[DTYPE](b.ipos_x)
        body_ipos[bi * 3 + 1] = Scalar[DTYPE](b.ipos_y)
        body_ipos[bi * 3 + 2] = Scalar[DTYPE](b.ipos_z)
        body_iquat[bi * 4 + 0] = Scalar[DTYPE](b.iquat_x)
        body_iquat[bi * 4 + 1] = Scalar[DTYPE](b.iquat_y)
        body_iquat[bi * 4 + 2] = Scalar[DTYPE](b.iquat_z)
        body_iquat[bi * 4 + 3] = Scalar[DTYPE](b.iquat_w)
        body_has_explicit_inertia[bi] = b.has_explicit_inertia
        body_parent[bi] = b.parent

        # Non-staged body columns straight into the record.
        var o = bi * MODEL_BODY_SIZE
        mf.bodies.data[o + BODY_IDX_POS_X] = Scalar[DTYPE](b.pos_x)
        mf.bodies.data[o + BODY_IDX_POS_Y] = Scalar[DTYPE](b.pos_y)
        mf.bodies.data[o + BODY_IDX_POS_Z] = Scalar[DTYPE](b.pos_z)
        mf.bodies.data[o + BODY_IDX_QUAT_X] = Scalar[DTYPE](b.quat_x)
        mf.bodies.data[o + BODY_IDX_QUAT_Y] = Scalar[DTYPE](b.quat_y)
        mf.bodies.data[o + BODY_IDX_QUAT_Z] = Scalar[DTYPE](b.quat_z)
        mf.bodies.data[o + BODY_IDX_QUAT_W] = Scalar[DTYPE](b.quat_w)
        mf.bodies.data[o + BODY_IDX_PARENT] = Scalar[DTYPE](b.parent)
        mf.bodies.data[o + BODY_IDX_MOCAP] = Scalar[DTYPE](
            1.0 if b.is_mocap else 0.0
        )
        mf.bodies.data[o + BODY_IDX_GRAVCOMP] = Scalar[DTYPE](b.gravcomp)

    # Worldbody record: pos 0, quat identity, parent 0, mocap 0.
    mf.bodies.data[BODY_IDX_QUAT_W] = Scalar[DTYPE](1)

    # body_rootid (root = child of worldbody).
    var body_rootid = List[Int](length=mf.dims.get_nbody(), fill=0)
    for bi in range(1, mf.dims.get_nbody()):
        var p = body_parent[bi]
        if p == 0:
            body_rootid[bi] = bi
        else:
            body_rootid[bi] = body_rootid[p]
        mf.bodies.data[bi * MODEL_BODY_SIZE + BODY_IDX_ROOTID] = Scalar[DTYPE](
            body_rootid[bi]
        )

    # ── joints ───────────────────────────────────────────────────────────
    var qpos_adr = 0
    var dof_adr = 0
    for j in range(len(fmd.joints)):
        var jd = fmd.joints[j]
        var o = j * MODEL_JOINT_SIZE

        mf.joints.data[o + JOINT_IDX_TYPE] = Scalar[DTYPE](jd.jnt_type)
        mf.joints.data[o + JOINT_IDX_BODY_ID] = Scalar[DTYPE](jd.body_id)
        mf.joints.data[o + JOINT_IDX_QPOS_ADR] = Scalar[DTYPE](qpos_adr)
        mf.joints.data[o + JOINT_IDX_DOF_ADR] = Scalar[DTYPE](dof_adr)

        if jd.jnt_type == JNT_HINGE or jd.jnt_type == JNT_SLIDE:
            # Legacy create_hinge/create_slide: normalized axis, parsed
            # pos/range/dynamics, tau_limit default 1000.
            var ax = Scalar[DTYPE](jd.axis_x)
            var ay = Scalar[DTYPE](jd.axis_y)
            var az = Scalar[DTYPE](jd.axis_z)
            var length = sqrt(ax * ax + ay * ay + az * az)
            if length > Scalar[DTYPE](1e-10):
                ax = ax / length
                ay = ay / length
                az = az / length
            mf.joints.data[o + JOINT_IDX_POS_X] = Scalar[DTYPE](jd.pos_x)
            mf.joints.data[o + JOINT_IDX_POS_Y] = Scalar[DTYPE](jd.pos_y)
            mf.joints.data[o + JOINT_IDX_POS_Z] = Scalar[DTYPE](jd.pos_z)
            mf.joints.data[o + JOINT_IDX_AXIS_X] = ax
            mf.joints.data[o + JOINT_IDX_AXIS_Y] = ay
            mf.joints.data[o + JOINT_IDX_AXIS_Z] = az
            mf.joints.data[o + JOINT_IDX_TAU_LIMIT] = Scalar[DTYPE](1000.0)
            mf.joints.data[o + JOINT_IDX_RANGE_MIN] = Scalar[DTYPE](
                jd.range_min
            )
            mf.joints.data[o + JOINT_IDX_RANGE_MAX] = Scalar[DTYPE](
                jd.range_max
            )
            mf.joints.data[o + JOINT_IDX_ARMATURE] = Scalar[DTYPE](jd.armature)
            mf.joints.data[o + JOINT_IDX_DAMPING] = Scalar[DTYPE](jd.damping)
            mf.joints.data[o + JOINT_IDX_STIFFNESS] = Scalar[DTYPE](
                jd.stiffness
            )
            mf.joints.data[o + JOINT_IDX_SPRINGREF] = Scalar[DTYPE](
                jd.springref
            )
            mf.joints.data[o + JOINT_IDX_FRICTIONLOSS] = Scalar[DTYPE](
                jd.frictionloss
            )
        else:
            # JNT_FREE (legacy create_free + armature/damping overrides);
            # JNT_BALL was never wired on the XML path.
            mf.joints.data[o + JOINT_IDX_POS_X] = Scalar[DTYPE](0)
            mf.joints.data[o + JOINT_IDX_POS_Y] = Scalar[DTYPE](0)
            mf.joints.data[o + JOINT_IDX_POS_Z] = Scalar[DTYPE](0)
            mf.joints.data[o + JOINT_IDX_AXIS_X] = Scalar[DTYPE](0)
            mf.joints.data[o + JOINT_IDX_AXIS_Y] = Scalar[DTYPE](0)
            mf.joints.data[o + JOINT_IDX_AXIS_Z] = Scalar[DTYPE](1)
            mf.joints.data[o + JOINT_IDX_TAU_LIMIT] = Scalar[DTYPE](0)
            # The record has no `limited` flag; a wide range IS the encoding.
            # See `JOINT_RANGE_UNLIMITED` — a consumer must test against it,
            # not against `min < max`.
            mf.joints.data[o + JOINT_IDX_RANGE_MIN] = Scalar[DTYPE](
                -JOINT_RANGE_UNLIMITED
            )
            mf.joints.data[o + JOINT_IDX_RANGE_MAX] = Scalar[DTYPE](
                JOINT_RANGE_UNLIMITED
            )
            mf.joints.data[o + JOINT_IDX_ARMATURE] = Scalar[DTYPE](jd.armature)
            mf.joints.data[o + JOINT_IDX_DAMPING] = Scalar[DTYPE](jd.damping)
            # ⚠ AUD-43: a BALL joint's stiffness reaches the record now — the
            # passive routine springs it with `subQuat` against the identity.
            # (This branch used to write 0, so no ball spring could ever act.)
            mf.joints.data[o + JOINT_IDX_STIFFNESS] = Scalar[DTYPE](
                jd.stiffness
            )
            mf.joints.data[o + JOINT_IDX_SPRINGREF] = Scalar[DTYPE](0)
            mf.joints.data[o + JOINT_IDX_FRICTIONLOSS] = Scalar[DTYPE](0)

        # qpos0 = joint ref value (MuJoCo: displacement = qpos - qpos0).
        mf.joints.data[o + JOINT_IDX_QPOS0] = Scalar[DTYPE](jd.ref_val)

        mf.joints.data[o + JOINT_IDX_MARGIN] = Scalar[DTYPE](jd.margin)

        # Per-joint limit solref, AS PARSED. ⚠ AUD-29: this used to keep the
        # value only `if >= 0.0`, which replaced a NEGATIVE (direct
        # stiffness/damping form) solreflimit by the default — the joint
        # limit came out 2.5x stiffer than MuJoCo's. The parser resolves
        # element -> class -> MuJoCo default itself, so there is no sentinel
        # left to test for. solimp keeps its `>= 0` test: a negative solimp
        # has no meaning in MuJoCo.
        mf.joints.data[o + JOINT_IDX_SOLREF_LIMIT_0] = Scalar[DTYPE](
            jd.solref_limit_0
        )
        mf.joints.data[o + JOINT_IDX_SOLREF_LIMIT_1] = Scalar[DTYPE](
            jd.solref_limit_1
        )
        mf.joints.data[o + JOINT_IDX_SOLIMP_LIMIT_0] = (
            Scalar[DTYPE](jd.solimp_limit_0)
            if jd.solimp_limit_0 >= 0.0
            else def_solimp_limit_0
        )
        mf.joints.data[o + JOINT_IDX_SOLIMP_LIMIT_1] = (
            Scalar[DTYPE](jd.solimp_limit_1)
            if jd.solimp_limit_1 >= 0.0
            else def_solimp_limit_1
        )
        mf.joints.data[o + JOINT_IDX_SOLIMP_LIMIT_2] = (
            Scalar[DTYPE](jd.solimp_limit_2)
            if jd.solimp_limit_2 >= 0.0
            else def_solimp_limit_2
        )
        mf.joints.data[o + JOINT_IDX_SOLIMP_LIMIT_3] = (
            Scalar[DTYPE](jd.solimp_limit_3)
            if jd.solimp_limit_3 >= 0.0
            else def_solimp_limit_3
        )
        mf.joints.data[o + JOINT_IDX_SOLIMP_LIMIT_4] = (
            Scalar[DTYPE](jd.solimp_limit_4)
            if jd.solimp_limit_4 >= 0.0
            else def_solimp_limit_4
        )

        qpos_adr += _jnt_qpos_size(jd.jnt_type)
        dof_adr += _jnt_qvel_size(jd.jnt_type)

    # Model-level limit meta sync from joint 0 (legacy CPU/GPU consistency
    # sync — uniform joint solimp across all current models).
    # Same as the geom gate above; guards an index into `mf.joints`.
    if mf.dims.get_njoint() > 0:
        mf.meta.data[MODEL_META_IDX_SOLREF_LIMIT_0] = mf.joints.data[
            JOINT_IDX_SOLREF_LIMIT_0
        ]
        mf.meta.data[MODEL_META_IDX_SOLREF_LIMIT_1] = mf.joints.data[
            JOINT_IDX_SOLREF_LIMIT_1
        ]
        mf.meta.data[MODEL_META_IDX_SOLIMP_LIMIT_0] = mf.joints.data[
            JOINT_IDX_SOLIMP_LIMIT_0
        ]
        mf.meta.data[MODEL_META_IDX_SOLIMP_LIMIT_1] = mf.joints.data[
            JOINT_IDX_SOLIMP_LIMIT_1
        ]
        mf.meta.data[MODEL_META_IDX_SOLIMP_LIMIT_2] = mf.joints.data[
            JOINT_IDX_SOLIMP_LIMIT_2
        ]
        mf.meta.data[MODEL_META_IDX_SOLIMP_LIMIT_3] = mf.joints.data[
            JOINT_IDX_SOLIMP_LIMIT_3
        ]
        mf.meta.data[MODEL_META_IDX_SOLIMP_LIMIT_4] = mf.joints.data[
            JOINT_IDX_SOLIMP_LIMIT_4
        ]

    # body_weldid: a body with joints is its own weld root, and so is a MOCAP
    # body (3.12, commit ed13bf56: `weld_root = joints || mocap`,
    # user_model.cc:4512). A jointless, non-mocap body inherits its parent's.
    # ⚠ AUD-35: before 3.12 a mocap body inherited the world's weld id 0, so
    # its jointed children escaped the parent-child filter (a mocap hand
    # collided with its own fingers) and its geoms were static for rays.
    # body_dofnum (the body's OWN dofs) is written beside it for the
    # dof-less pair rule the same commit introduced.
    var body_has_joint = List[Bool](length=mf.dims.get_nbody(), fill=False)
    var body_dofnum = List[Int](length=mf.dims.get_nbody(), fill=0)
    for j in range(len(fmd.joints)):
        body_has_joint[fmd.joints[j].body_id] = True
        body_dofnum[fmd.joints[j].body_id] += fmd.joints[j].nv
    for bi in range(mf.dims.get_nbody()):
        mf.bodies.data[bi * MODEL_BODY_SIZE + BODY_IDX_DOFNUM] = Scalar[DTYPE](
            body_dofnum[bi]
        )
    var body_weldid = List[Int](length=mf.dims.get_nbody(), fill=0)
    for bi in range(1, mf.dims.get_nbody()):
        if body_has_joint[bi] or fmd.bodies[bi - 1].is_mocap:
            body_weldid[bi] = bi
        else:
            body_weldid[bi] = body_weldid[body_parent[bi]]
        mf.bodies.data[bi * MODEL_BODY_SIZE + BODY_IDX_WELDID] = Scalar[DTYPE](
            body_weldid[bi]
        )

    # ── mesh frames + unitless principal moments (MuJoCo `mjCMesh::Compute`)
    #
    # Computed BEFORE the geom loop because both consumers need it: the geom's
    # pos/quat are composed with the mesh frame right below, and
    # `load_mesh_hull` bakes the same frame into the vertices further down. The
    # two are only equivalent TOGETHER — moving the vertices without composing
    # the frame collides the mesh in the wrong place, and vice versa.
    #
    # ⚠ COMPUTED ONLY WHERE IT IS USED: a mesh geom that is neither collidable
    # nor a source of body inertia (its body carries an explicit `<inertial>`)
    # gets nothing, and keeps its declared frame. That is dog — 162 visual-only
    # meshes whose frames reach nothing but the renderer — and it is what keeps
    # this off dog's build path entirely. The same filter §22 applied to hull
    # construction, for the same reason.
    var mesh_inertia_cache = List[MeshInertia[DTYPE]]()
    var mesh_inertia_valid = List[Bool](
        length=fmd.num_mesh_assets, fill=False
    )
    for _ in range(fmd.num_mesh_assets):
        mesh_inertia_cache.append(MeshInertia[DTYPE]())
    for i in range(len(fmd.geoms)):
        var gd_m = fmd.geoms[i]
        # ⚠ `fit_from_mesh` GEOMS NEED THIS TOO, and their type is NOT
        # `GEOM_MESH` — that is the whole point of them. MuJoCo sizes a
        # fitted primitive from the mesh's INERTIA BOX, so the inertia this
        # loop computes is the input to the fit below, not just to the body's
        # mass properties.
        if gd_m.geom_type != GEOM_MESH and not gd_m.fit_from_mesh:
            continue
        if gd_m.mesh_id < 0 or gd_m.mesh_filename.byte_length() == 0:
            continue
        if mesh_inertia_valid[gd_m.mesh_id]:
            continue
        var collidable = gd_m.contype != 0 or gd_m.conaffinity != 0
        var needs_inertia = not body_has_explicit_inertia[gd_m.body_id]
        if not (collidable or needs_inertia):
            continue
        try:
            # ⚠ THE INERTIA MUST SEE THE SCALED MESH. `mesh_pos`/`mesh_quat`
            # are the CoM and principal axes MuJoCo bakes into every mesh, and
            # they are computed from the COMPILED vertices — i.e. after
            # `<mesh scale>`. Feeding the raw STL here would put a millimetre
            # model's centre of mass a thousand times too far out, and the hull
            # (which IS scaled) would then be built in a different frame from
            # the geom that carries it.
            mesh_inertia_cache[gd_m.mesh_id] = mesh_inertia_from_file[DTYPE](
                gd_m.mesh_filename,
                gd_m.mesh_scale_x,
                gd_m.mesh_scale_y,
                gd_m.mesh_scale_z,
                gd_m.mesh_ref_pos_x,
                gd_m.mesh_ref_pos_y,
                gd_m.mesh_ref_pos_z,
                gd_m.mesh_ref_quat_w,
                gd_m.mesh_ref_quat_x,
                gd_m.mesh_ref_quat_y,
                gd_m.mesh_ref_quat_z,
                gd_m.mesh_inertia_shell,
            )
            mesh_inertia_valid[gd_m.mesh_id] = True
        except:
            print(
                "Warning: failed to compute mesh inertia:",
                gd_m.mesh_filename,
            )

    # ── geoms (+ build-only mass/group staging for inertiafromgeom) ───────
    var geom_mass = List[Scalar[DTYPE]](length=mf.dims.get_ngeom(), fill=Scalar[DTYPE](0))
    var geom_density = List[Scalar[DTYPE]](
        length=mf.dims.get_ngeom(), fill=Scalar[DTYPE](MJ_DEFAULT_DENSITY)
    )
    var geom_group = List[Int](length=mf.dims.get_ngeom(), fill=0)
    # Per-geom mesh volume + UNITLESS principal moments, staged for the inertia
    # assembly. `geom_inertia` cannot compute these — it takes scalar dims and
    # has no way to reach a vertex list — so the mesh case is resolved here and
    # read back as a table.
    var geom_mesh_vol = List[Scalar[DTYPE]](
        length=mf.dims.get_ngeom(), fill=Scalar[DTYPE](0)
    )
    var geom_mesh_eig = List[Scalar[DTYPE]](
        length=mf.dims.get_ngeom() * 3, fill=Scalar[DTYPE](0)
    )
    for i in range(len(fmd.geoms)):
        var gd = fmd.geoms[i]
        var o = i * MODEL_GEOM_SIZE
        mf.geoms.data[o + GEOM_IDX_TYPE] = Scalar[DTYPE](gd.geom_type)
        mf.geoms.data[o + GEOM_IDX_BODY] = Scalar[DTYPE](gd.body_id)
        mf.geoms.data[o + GEOM_IDX_POS_X] = Scalar[DTYPE](gd.pos_x)
        mf.geoms.data[o + GEOM_IDX_POS_Y] = Scalar[DTYPE](gd.pos_y)
        mf.geoms.data[o + GEOM_IDX_POS_Z] = Scalar[DTYPE](gd.pos_z)
        mf.geoms.data[o + GEOM_IDX_QUAT_X] = Scalar[DTYPE](gd.quat_x)
        mf.geoms.data[o + GEOM_IDX_QUAT_Y] = Scalar[DTYPE](gd.quat_y)
        mf.geoms.data[o + GEOM_IDX_QUAT_Z] = Scalar[DTYPE](gd.quat_z)
        mf.geoms.data[o + GEOM_IDX_QUAT_W] = Scalar[DTYPE](gd.quat_w)
        mf.geoms.data[o + GEOM_IDX_RADIUS] = Scalar[DTYPE](gd.radius)
        mf.geoms.data[o + GEOM_IDX_HALF_LENGTH] = Scalar[DTYPE](gd.half_length)
        mf.geoms.data[o + GEOM_IDX_HALF_X] = Scalar[DTYPE](gd.half_x)
        mf.geoms.data[o + GEOM_IDX_HALF_Y] = Scalar[DTYPE](gd.half_y)
        mf.geoms.data[o + GEOM_IDX_HALF_Z] = Scalar[DTYPE](gd.half_z)
        mf.geoms.data[o + GEOM_IDX_FRICTION] = Scalar[DTYPE](gd.friction)
        mf.geoms.data[o + GEOM_IDX_CONTYPE] = Scalar[DTYPE](gd.contype)
        mf.geoms.data[o + GEOM_IDX_CONAFFINITY] = Scalar[DTYPE](
            gd.conaffinity
        )
        mf.geoms.data[o + GEOM_IDX_CONDIM] = Scalar[DTYPE](gd.condim)
        mf.geoms.data[o + GEOM_IDX_FRICTION_SPIN] = Scalar[DTYPE](
            gd.friction_spin
        )
        mf.geoms.data[o + GEOM_IDX_FRICTION_ROLL] = Scalar[DTYPE](
            gd.friction_roll
        )
        mf.geoms.data[o + GEOM_IDX_PRIORITY] = Scalar[DTYPE](gd.priority)
        # `ray_eliminate`'s two STATIC rules, collapsed to one flag. MuJoCo:
        #   matid <  0 and rgba[3] == 0        -> invisible
        #   matid >= 0 and mat_rgba[3] == 0    -> invisible
        # Neither can change after the model is built, so a ray never has to
        # look at a colour. ⚠ THE MATERIAL WINS WHEN THERE IS ONE — a geom
        # with `rgba` opaque and an alpha-0 MATERIAL is invisible, and reading
        # only `rgba` would leave a decoration in front of every rangefinder.
        var _ray_vis = 1.0
        if gd.material_id >= 0 and gd.material_id < len(fmd.materials):
            if fmd.materials[gd.material_id].rgba_a == 0.0:
                _ray_vis = 0.0
        elif gd.rgba_a == 0.0:
            _ray_vis = 0.0
        mf.geoms.data[o + GEOM_IDX_RAY_VISIBLE] = Scalar[DTYPE](_ray_vis)
        mf.geoms.data[o + GEOM_IDX_GROUP] = Scalar[DTYPE](gd.group)
        mf.geoms.data[o + GEOM_IDX_SOLREF_0] = Scalar[DTYPE](gd.solref_0)
        mf.geoms.data[o + GEOM_IDX_SOLREF_1] = Scalar[DTYPE](gd.solref_1)
        mf.geoms.data[o + GEOM_IDX_SOLIMP_0] = Scalar[DTYPE](gd.solimp_0)
        mf.geoms.data[o + GEOM_IDX_SOLIMP_1] = Scalar[DTYPE](gd.solimp_1)
        mf.geoms.data[o + GEOM_IDX_SOLIMP_2] = Scalar[DTYPE](gd.solimp_2)
        mf.geoms.data[o + GEOM_IDX_SOLIMP_3] = Scalar[DTYPE](gd.solimp_3)
        mf.geoms.data[o + GEOM_IDX_SOLIMP_4] = Scalar[DTYPE](gd.solimp_4)
        mf.geoms.data[o + GEOM_IDX_MARGIN] = Scalar[DTYPE](gd.margin)
        mf.geoms.data[o + GEOM_IDX_GAP] = Scalar[DTYPE](gd.gap)
        mf.geoms.data[o + GEOM_IDX_MESH_ID] = Scalar[DTYPE](gd.mesh_id)
        mf.geoms.data[o + GEOM_IDX_HFIELD_ID] = Scalar[DTYPE](gd.hfield_id)
        # ── the ray tracer's colour, a separate tensor ────────────────────
        # ⚠ `gd.rgba_*` IS ALREADY MATERIAL-RESOLVED.
        # `_resolve_geom_materials` applied MuJoCo's rule (the material wins
        # only where the geom's own rgba still equals the internal default)
        # during parsing, so this is a copy and NOT a fourth place the rule
        # gets re-derived. See `Model.geom_rgba`.
        var co = i * MODEL_GEOM_RGBA_SIZE
        mf.geom_rgba.data[co + 0] = Scalar[DTYPE](gd.rgba_r)
        mf.geom_rgba.data[co + 1] = Scalar[DTYPE](gd.rgba_g)
        mf.geom_rgba.data[co + 2] = Scalar[DTYPE](gd.rgba_b)
        mf.geom_rgba.data[co + 3] = Scalar[DTYPE](gd.rgba_a)
        geom_mass[i] = Scalar[DTYPE](gd.mass)
        geom_density[i] = Scalar[DTYPE](gd.density)
        geom_group[i] = gd.group

        # Compose the mesh frame into the geom frame, as MuJoCo's compiler
        # does: `geom_pos = declared_pos + R(declared_quat) * mesh_pos` and
        # `geom_quat = declared_quat (x) mesh_quat`. Measured on Jaco, whose
        # mesh geoms declare no pos/quat: `geom_pos == mesh_pos` and
        # `geom_quat == mesh_quat` for all 13 of them.
        #
        # ⚠⚠ A FITTED PRIMITIVE GETS THE SAME FRAME. `mjuu_frameaccum` sits
        # OUTSIDE MuJoCo's `if (type != mjGEOM_MESH)` fitting branch
        # (`user_objects.cc:4053`), so a sphere fitted to a mesh is offset by
        # that mesh's CoM exactly as a mesh geom is. Gating this on
        # `GEOM_MESH` left every fitted geom at its body's origin, and on
        # rby1 that put two arm spheres CONCENTRIC — our reported penetration
        # was exactly `-(r1 + r2)`, which is what a zero centre-distance
        # gives and is the fingerprint of a dropped offset rather than a
        # wrong radius.
        if (
            (gd.geom_type == GEOM_MESH or gd.fit_from_mesh)
            and gd.mesh_id >= 0
            and mesh_inertia_valid[gd.mesh_id]
        ):
            var mi_g = mesh_inertia_cache[gd.mesh_id]
            var rm = Array[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
            quat_to_mat[DTYPE](
                Scalar[DTYPE](gd.quat_x),
                Scalar[DTYPE](gd.quat_y),
                Scalar[DTYPE](gd.quat_z),
                Scalar[DTYPE](gd.quat_w),
                rm,
            )
            mf.geoms.data[o + GEOM_IDX_POS_X] = Scalar[DTYPE](gd.pos_x) + (
                rm[0] * mi_g.com_x + rm[1] * mi_g.com_y + rm[2] * mi_g.com_z
            )
            mf.geoms.data[o + GEOM_IDX_POS_Y] = Scalar[DTYPE](gd.pos_y) + (
                rm[3] * mi_g.com_x + rm[4] * mi_g.com_y + rm[5] * mi_g.com_z
            )
            mf.geoms.data[o + GEOM_IDX_POS_Z] = Scalar[DTYPE](gd.pos_z) + (
                rm[6] * mi_g.com_x + rm[7] * mi_g.com_y + rm[8] * mi_g.com_z
            )
            var cq = _mulquat_xyzw[DTYPE](
                Scalar[DTYPE](gd.quat_x),
                Scalar[DTYPE](gd.quat_y),
                Scalar[DTYPE](gd.quat_z),
                Scalar[DTYPE](gd.quat_w),
                mi_g.qx,
                mi_g.qy,
                mi_g.qz,
                mi_g.qw,
            )
            mf.geoms.data[o + GEOM_IDX_QUAT_X] = cq[0]
            mf.geoms.data[o + GEOM_IDX_QUAT_Y] = cq[1]
            mf.geoms.data[o + GEOM_IDX_QUAT_Z] = cq[2]
            mf.geoms.data[o + GEOM_IDX_QUAT_W] = cq[3]
            # ⚠ BUT NOT ITS INERTIA. `geom_mesh_vol`/`_eig` feed the body's
            # mass properties when no `<inertial>` is given, and MuJoCo
            # computes a FITTED geom's inertia from the PRIMITIVE it became,
            # not from the mesh it was fitted to — it has cleared the mesh
            # pointer by then. Writing these for a fitted geom would give a
            # sphere the mesh's inertia tensor.
            if not gd.fit_from_mesh:
                geom_mesh_vol[i] = mi_g.volume
                geom_mesh_eig[i * 3 + 0] = mi_g.eig0
                geom_mesh_eig[i * 3 + 1] = mi_g.eig1
                geom_mesh_eig[i * 3 + 2] = mi_g.eig2

        # ── `mjCMesh::FitGeom` — a PRIMITIVE sized from a mesh ────────────
        #
        # A geom that names a mesh while its type is a primitive is fitted to
        # that mesh and then loses its mesh reference (`user_objects.cc:4038`).
        # The default `fitaabb` is FALSE, so the fit uses the mesh's INERTIA
        # BOX, not its axis-aligned bounding box (`user_mesh.cc:944`):
        #
        #     boxsz[k] = 0.5*sqrt(6*(sum of the OTHER two eigvals - eigval[k])
        #                          / volume)
        #     SPHERE            size[0] = (bx + by + bz)/3
        #     CAPSULE           size[0] = (bx + by)/2, size[1] = max(0, bz - size[0]/2)
        #     CYLINDER          size[0] = (bx + by)/2, size[1] = bz
        #     BOX / ELLIPSOID   size    = (bx, by, bz)
        #
        # ⚠ VERIFIED AGAINST THE RUNTIME, NOT TRANSCRIBED AND HOPED FOR:
        # rby1's `EE_FINGER_0` gives boxsz (0.029504, 0.013904, 0.001500) and
        # a fitted sphere radius of 0.0149700, which is MuJoCo's
        # `geom_rbound` for that geom to seven digits.
        #
        # ⚠ THE EIGENVALUES ARE UNITLESS (length^5) ON BOTH SIDES — MuJoCo
        # divides by `GetVolumeRef()` here for the same reason, so no mass or
        # density enters. Scaling them by `mass/volume` first (which is what
        # the INERTIA path does) would make the box depend on the body's mass.
        if gd.fit_from_mesh and gd.mesh_id >= 0 and mesh_inertia_valid[
            gd.mesh_id
        ]:
            var mif = mesh_inertia_cache[gd.mesh_id]
            var volf = Float64(mif.volume)
            if volf > 1e-18:
                var e0 = Float64(mif.eig0)
                var e1 = Float64(mif.eig1)
                var e2 = Float64(mif.eig2)
                var t0 = 6.0 * (e1 + e2 - e0) / volf
                var t1 = 6.0 * (e0 + e2 - e1) / volf
                var t2 = 6.0 * (e0 + e1 - e2) / volf
                var bx = 0.5 * sqrt(t0) if t0 > 0.0 else 0.0
                var by = 0.5 * sqrt(t1) if t1 > 0.0 else 0.0
                var bz = 0.5 * sqrt(t2) if t2 > 0.0 else 0.0
                if gd.geom_type == GEOM_SPHERE:
                    gd.radius = (bx + by + bz) / 3.0
                elif gd.geom_type == GEOM_CAPSULE:
                    gd.radius = (bx + by) / 2.0
                    var hl = bz - gd.radius / 2.0
                    gd.half_length = hl if hl > 0.0 else 0.0
                elif gd.geom_type == GEOM_CYLINDER:
                    gd.radius = (bx + by) / 2.0
                    gd.half_length = bz
                else:
                    gd.half_x = bx
                    gd.half_y = by
                    gd.half_z = bz
                mf.geoms.data[o + GEOM_IDX_RADIUS] = Scalar[DTYPE](gd.radius)
                mf.geoms.data[o + GEOM_IDX_HALF_LENGTH] = Scalar[DTYPE](
                    gd.half_length
                )
                mf.geoms.data[o + GEOM_IDX_HALF_X] = Scalar[DTYPE](gd.half_x)
                mf.geoms.data[o + GEOM_IDX_HALF_Y] = Scalar[DTYPE](gd.half_y)
                mf.geoms.data[o + GEOM_IDX_HALF_Z] = Scalar[DTYPE](gd.half_z)

        # Bounding sphere radius for broad-phase (legacy per-type formulas).
        # ⚠ THE FORMULA IS `rbound_of`, NOT INLINE, because the studio has to
        # recompute it when a size is EDITED. A resize that left `rbound`
        # stale gives a broadphase bound too small for the new shape, so pairs
        # are rejected before the narrow phase ever sees them — MISSED
        # CONTACTS, silently. Caught by `test_studio_edit_roundtrip` on its
        # first run, which is exactly what a byte-identity gate is for.
        var rbound = Scalar[DTYPE](
            rbound_of(
                gd.geom_type, gd.radius, gd.half_length,
                gd.half_x, gd.half_y, gd.half_z,
            )
        )
        # GEOM_MESH: refined from hull vertices below.
        mf.geoms.data[o + GEOM_IDX_RBOUND] = rbound

        # ── HEIGHTFIELD: size and rbound come from the ASSET, not the geom ──
        #
        # `mjCGeom::Compile` (`user_objects.cc:4078`) overwrites a heightfield
        # geom's `size` wholesale:
        #
        #     size = [rx, ry, 0.25*elevation + 0.5*base]
        #
        # and `GetRBound` (:3721) is
        #
        #     sqrt(rx^2 + ry^2 + max(elevation^2, base^2))
        #
        # ⚠ NEITHER IS THE OBVIOUS FORMULA. `size[2]` is not half the total
        # height (that would be `0.5*(elevation + base)`), and `rbound` takes
        # the LARGER of the two z extents squared rather than their sum — a
        # field whose base is deeper than its peaks is bounded by the base.
        # Measured on `google_barkour_vb`, `size = "10 10 .05 0.1"`: MuJoCo
        # reports `geom_size = [10, 10, 0.0625]` and
        # `geom_rbound = 14.142489172702236`.
        if gd.geom_type == GEOM_HFIELD and gd.hfield_id >= 0:
            var hb = gd.hfield_id * 4
            var hrx = fmd.hfield_size[hb + 0]
            var hry = fmd.hfield_size[hb + 1]
            var hez = fmd.hfield_size[hb + 2]
            var hbz = fmd.hfield_size[hb + 3]
            mf.geoms.data[o + GEOM_IDX_HALF_X] = Scalar[DTYPE](hrx)
            mf.geoms.data[o + GEOM_IDX_HALF_Y] = Scalar[DTYPE](hry)
            mf.geoms.data[o + GEOM_IDX_HALF_Z] = Scalar[DTYPE](
                0.25 * hez + 0.5 * hbz
            )
            var zz = hez * hez if hez * hez > hbz * hbz else hbz * hbz
            mf.geoms.data[o + GEOM_IDX_RBOUND] = Scalar[DTYPE](
                sqrt(hrx * hrx + hry * hry + zz)
            )

    # ── mesh convex hulls (STL → dedup → hull; shared meshes remapped) ────
    var mesh_vert = List[Scalar[DTYPE]]()
    var mesh_vertadr = List[Int]()
    var mesh_vertnum = List[Int]()
    var num_meshes = 0
    # Polygon topology, accumulated across meshes exactly as `mesh_polyvert`
    # and `mesh_polymap` are in `mjModel`. See `collision/mesh_polygons.mojo`.
    var mesh_polyadr = List[Int]()
    var mesh_polynum = List[Int]()
    var poly_vert = List[Int]()
    var poly_vertadr = List[Int]()
    var poly_vertnum = List[Int]()
    var poly_normal = List[Scalar[DTYPE]]()
    var polymap = List[Int]()
    var polymap_adr = List[Int]()
    var polymap_num = List[Int]()
    # Hull vertex adjacency (MuJoCo's `mesh_graph`), for the plane-mesh path.
    var edge_adr = List[Int]()
    var edge_list = List[Int]()
    # The mesh's ORIGINAL triangles, 9 floats each — what `ray/mesh.mojo`
    # walks. The hull above cannot answer a ray against a non-convex part.
    var mesh_tri = List[Scalar[DTYPE]]()
    var mesh_triadr = List[Int]()
    var mesh_trinum = List[Int]()
    var loaded_mesh_ids = List[Int](length=fmd.num_mesh_assets, fill=-1)
    for i in range(len(fmd.geoms)):
        var gd = fmd.geoms[i]
        var o = i * MODEL_GEOM_SIZE
        # ⚠ COLLIDABLE MESHES ONLY. This used to build a hull for EVERY mesh
        # geom carrying a filename, ignoring `contype`/`conaffinity`. Dog has
        # 162 mesh geoms and ALL of them are visual (`contype=0
        # conaffinity=0`), so every dog model build ran the exact incremental
        # hull — O(n*h) — over 162 meshes and threw the whole result away, then
        # tripped the capacity message on vertices nothing could ever collide.
        #
        # A geom with neither bit set cannot pair with anything (MuJoCo's
        # filter is `contype1 & conaffinity2 || contype2 & conaffinity1`), so
        # its hull is dead weight in both time and capacity. Skipping it is
        # also what lets the capacity check below be a hard error: otherwise
        # dog would fail a check for a feature it does not use.
        var mesh_collidable = gd.contype != 0 or gd.conaffinity != 0
        # ⚠ A SKIPPED MESH GEOM MUST NOT KEEP ITS ASSET INDEX. The main geom
        # loop writes `GEOM_IDX_MESH_ID = gd.mesh_id`, which is the index into
        # the XML's mesh ASSETS; the loop below overwrites it with the
        # COMPACTED runtime id for meshes it actually loads. Skipping the
        # visual ones leaves that raw asset index in place, pointing past the
        # end of `mesh_meta` — sawyer keeps 10 visual mesh geoms carrying ids
        # up to 13 while only 2 meshes are loaded. Nothing dereferences it
        # today (the collision filter rejects those geoms first), but a stale
        # index that is only safe because of a check somewhere else is the
        # shape of the `mesh_vertadr` unit bug. Make it unusable instead.
        if gd.geom_type == GEOM_MESH and not mesh_collidable:
            mf.geoms.data[o + GEOM_IDX_MESH_ID] = Scalar[DTYPE](-1)
        if (
            gd.geom_type == GEOM_MESH
            and gd.mesh_id >= 0
            and gd.mesh_filename.byte_length() > 0
            and mesh_collidable
        ):
            if loaded_mesh_ids[gd.mesh_id] >= 0:
                var mid = loaded_mesh_ids[gd.mesh_id]
                mf.geoms.data[o + GEOM_IDX_MESH_ID] = Scalar[DTYPE](mid)
                # `mesh_vertadr` is a VERTEX index (MuJoCo's convention, and
                # what every collision consumer expects); this helper walks the
                # FLAT scalar list, so convert. See `load_mesh_hull`.
                mf.geoms.data[o + GEOM_IDX_RBOUND] = (
                    compute_mesh_rbound_at[DTYPE](
                        mesh_vert, mesh_vertadr[mid] * 3, mesh_vertnum[mid]
                    )
                )
                # ⚠ AND `geom_size`, WHICH USED TO BE A 0.5 PLACEHOLDER
                # (AUD-46). `mj_ray`'s mesh path runs `ray_box(geom_size)`
                # BEFORE any triangle (engine_ray.c:894, :963), so a mesh
                # reaching further than 0.5 m from its frame origin lost
                # every hit past that — silently, as a miss.
                var _he = compute_mesh_half_extents_at[DTYPE](
                    mesh_vert, mesh_vertadr[mid] * 3, mesh_vertnum[mid]
                )
                mf.geoms.data[o + GEOM_IDX_HALF_X] = _he[0]
                mf.geoms.data[o + GEOM_IDX_HALF_Y] = _he[1]
                mf.geoms.data[o + GEOM_IDX_HALF_Z] = _he[2]
            else:
                try:
                    var result = load_mesh_hull[DTYPE](
                        gd.mesh_filename,
                        mesh_vert,
                        mesh_vertadr,
                        mesh_vertnum,
                        num_meshes,
                        mesh_polyadr,
                        mesh_polynum,
                        poly_vert,
                        poly_vertadr,
                        poly_vertnum,
                        poly_normal,
                        polymap,
                        polymap_adr,
                        polymap_num,
                        edge_adr,
                        edge_list,
                        mesh_tri,
                        mesh_triadr,
                        mesh_trinum,
                        mesh_inertia_cache[gd.mesh_id],
                        gd.mesh_scale_x,
                        gd.mesh_scale_y,
                        gd.mesh_scale_z,
                        gd.mesh_ref_pos_x,
                        gd.mesh_ref_pos_y,
                        gd.mesh_ref_pos_z,
                        gd.mesh_ref_quat_w,
                        gd.mesh_ref_quat_x,
                        gd.mesh_ref_quat_y,
                        gd.mesh_ref_quat_z,
                        gd.mesh_maxhullvert,
                    )
                    var mesh_id = result[0]
                    mf.geoms.data[o + GEOM_IDX_MESH_ID] = Scalar[DTYPE](
                        mesh_id
                    )
                    mf.geoms.data[o + GEOM_IDX_RBOUND] = result[1]
                    # `geom_size` from the mesh just loaded (AUD-46) — the
                    # same call the already-loaded branch above makes.
                    var _he2 = compute_mesh_half_extents_at[DTYPE](
                        mesh_vert,
                        mesh_vertadr[mesh_id] * 3,
                        mesh_vertnum[mesh_id],
                    )
                    mf.geoms.data[o + GEOM_IDX_HALF_X] = _he2[0]
                    mf.geoms.data[o + GEOM_IDX_HALF_Y] = _he2[1]
                    mf.geoms.data[o + GEOM_IDX_HALF_Z] = _he2[2]
                    loaded_mesh_ids[gd.mesh_id] = mesh_id
                except e:
                    # ⚠⚠ UNUSABLE, NOT WHATEVER WAS THERE. This used to print a
                    # bare warning and carry on, leaving `GEOM_IDX_MESH_ID`
                    # holding a stale value — so the geom collided against a
                    # DIFFERENT mesh's hull. Measured when a hull change made
                    # this path fire on barkour's `lower_leg_1to1`: four legs
                    # silently took `upper_right_1`'s shape, the mesh count
                    # went 18 -> 17, and the board row moved five orders. Same
                    # rule as the `mesh_collidable` branch above — a geom with
                    # NO collision geometry is recoverable, a geom with the
                    # WRONG collision geometry is not.
                    print(
                        "Warning: failed to load mesh:", gd.mesh_filename,
                        "--", String(e),
                        "-- this geom will have NO collision geometry",
                    )
                    mf.geoms.data[o + GEOM_IDX_MESH_ID] = Scalar[DTYPE](-1)

    # ⚠ THE SECOND SILENT TRUNCATION OF THE SAME KIND. A collidable mesh past
    # `MAX_GPU_MESHES` gets a hull built and an id assigned above, then no
    # `mesh_meta` row here — so every consumer reads vertadr/vertnum 0 and
    # collides against an empty mesh. The `<mesh>` asset cap that this comment's
    # sibling in `full_parser.mojo` describes cost SO-ARM100 two collision
    # surfaces exactly this way, and it took a per-geom `rbound` diff against
    # MuJoCo to notice. Say so rather than break quietly.
    # ⚠⚠ A RAISE, NOT A PRINT. This used to announce the overflow and carry on,
    # which leaves every mesh past the cap with a hull and an id but no
    # `mesh_meta` row — so consumers read vertadr/vertnum 0 and collide against
    # an EMPTY mesh. Wrong physics that runs, in a viewer where a printed line
    # scrolls away. The number needed is in the message, and `MAX_GPU_MESHES`
    # is 256 now, so reaching this at all means something unusual.
    if num_meshes > MAX_GPU_MESHES:
        raise Error(
            "physics3d: model needs " + String(num_meshes)
            + " collidable meshes but MAX_GPU_MESHES is "
            + String(MAX_GPU_MESHES)
            + ". Meshes past the cap would carry NO collision geometry"
            + " (mesh_meta has no row for them), so this is fatal rather than"
            + " a truncation. Raise MAX_GPU_MESHES in gpu/constants.mojo — it"
            + " sizes one [N, 4] table and nothing else."
        )
    for m in range(num_meshes):
        if m >= MAX_GPU_MESHES:
            break
        mf.mesh_meta.data[m * MODEL_MESH_META_SIZE + 0] = Scalar[DTYPE](
            mesh_vertadr[m]
        )
        mf.mesh_meta.data[m * MODEL_MESH_META_SIZE + 1] = Scalar[DTYPE](
            mesh_vertnum[m]
        )
        mf.mesh_meta.data[
            m * MODEL_MESH_META_SIZE + MESH_META_IDX_POLYADR
        ] = Scalar[DTYPE](mesh_polyadr[m])
        mf.mesh_meta.data[
            m * MODEL_MESH_META_SIZE + MESH_META_IDX_POLYNUM
        ] = Scalar[DTYPE](mesh_polynum[m])

    # ── HEIGHTFIELDS ────────────────────────────────────────────────────────
    #
    # The grids came out of the PARSE already normalised to [0, 1]
    # (`mjCHField::Compile`'s tail), so this is a straight copy into the two
    # tensors the narrow phase reads.
    if len(fmd.hfield_names) > MAX_GPU_HFIELDS:
        raise Error(
            String("physics3d: model declares ")
            + String(len(fmd.hfield_names))
            + " heightfields but MAX_GPU_HFIELDS is "
            + String(MAX_GPU_HFIELDS)
            + ". Raise it in gpu/constants.mojo — it sizes one small table"
            " and nothing else."
        )
    # ⚠ THE "CPU PATH ONLY" WARNING THAT USED TO BE HERE IS GONE, and its
    # absence is the load-bearing part. It said the GPU batched path reported
    # NO contacts against a heightfield, because `_detect_contacts_env`
    # compiled its heightfield branch out for Metal: the prism is a sixth
    # shape type for GJK, and a second instantiation of EPA's polytope on the
    # per-thread stack exceeded the limit. EPA's polytope now lives in
    # `d.ccd_ws` (`collision/ccd_workspace.mojo`), which is where MuJoCo has
    # always kept it, so both targets collide heightfields and there is
    # nothing left to warn about. Restore the print if that ever regresses —
    # a silent difference between targets is the failure mode this engine has
    # paid for repeatedly.
    # ⚠⚠ ONLY THE HEIGHTFIELDS A GEOM ACTUALLY NAMES ARE MATERIALISED, and that
    # is the same rule the mesh loop below already follows for the same reason.
    # `<asset>` may DECLARE a heightfield no geom uses: dm_control's quadruped
    # carries a 201x201 `terrain` — 40 401 samples — that only the `escape`
    # task ever puts a geom on, so `walk`, `run` and `fetch` were being asked
    # for 40 401 floats of capacity to store a field they never touch. Every
    # test building those models through `ModelDims[...]`, whose
    # `nhfield_data` defaults to 0, died on the unguarded write below —
    # `test_cfrc_ext_batched_vs_cpu`, `test_jac_site_vs_mujoco` and
    # `test_rne_post_sensors_vs_mujoco`, all with an "index 1 is out of bounds"
    # that names neither heightfields nor the dimension to raise.
    #
    # ⚠ REFERENCED FIELDS ARE COMPACTED, so `hfield_adr` is rebuilt here and is
    # NOT `fmd.hfield_adr`. The meta ROW INDEX is what a geom stores in
    # `GEOM_IDX_HFIELD_ID`, so rows keep their positions and only the data
    # offset moves; an unreferenced row is zeroed, which reads as `nrow == 0`
    # and is what `_hfield_contacts` already treats as "no field".
    var hf_used = List[Bool](length=len(fmd.hfield_names), fill=False)
    for i in range(len(fmd.geoms)):
        var hid = fmd.geoms[i].hfield_id
        if hid >= 0 and hid < len(hf_used):
            hf_used[hid] = True

    var hf_adr = List[Int](length=len(fmd.hfield_names), fill=-1)
    var hf_need = 0
    for h in range(len(fmd.hfield_names)):
        if hf_used[h]:
            hf_adr[h] = hf_need
            hf_need += fmd.hfield_nrow[h] * fmd.hfield_ncol[h]

    # ⚠ ANNOUNCED, NOT ASSERTED. The write used to run straight into the
    # tensor, so an under-sized budget surfaced as a bounds assert inside
    # `fields_build` with no mention of the feature or the knob. Say both.
    if hf_need > mf.dims.get_nhfield_data():
        raise Error(
            String("heightfield capacity exceeded — nhfield_data = ")
            + String(mf.dims.get_nhfield_data())
            + " but the heightfields a GEOM references need "
            + String(hf_need)
            + " samples. Raise `nhfield_data` on this model's dims (for"
            " `ModelDims[...]` it is the second parameter and defaults to 0)."
        )
    if len(fmd.hfield_names) > MAX_GPU_HFIELDS:
        raise Error(
            String("this model declares ") + String(len(fmd.hfield_names))
            + " heightfields and MAX_GPU_HFIELDS is " + String(MAX_GPU_HFIELDS)
            + ". The meta rows are comptime-sized; raise it in"
            " `gpu/constants.mojo`."
        )

    for h in range(len(fmd.hfield_names)):
        var ho = h * MODEL_HFIELD_META_SIZE
        if not hf_used[h]:
            for f in range(MODEL_HFIELD_META_SIZE):
                mf.hfield_meta.data[ho + f] = Scalar[DTYPE](0)
            continue
        mf.hfield_meta.data[ho + HFIELD_META_IDX_ADR] = Scalar[DTYPE](
            hf_adr[h]
        )
        mf.hfield_meta.data[ho + HFIELD_META_IDX_NROW] = Scalar[DTYPE](
            fmd.hfield_nrow[h]
        )
        mf.hfield_meta.data[ho + HFIELD_META_IDX_NCOL] = Scalar[DTYPE](
            fmd.hfield_ncol[h]
        )
        mf.hfield_meta.data[ho + HFIELD_META_IDX_SIZE_X] = Scalar[DTYPE](
            fmd.hfield_size[h * 4 + 0]
        )
        mf.hfield_meta.data[ho + HFIELD_META_IDX_SIZE_Y] = Scalar[DTYPE](
            fmd.hfield_size[h * 4 + 1]
        )
        mf.hfield_meta.data[ho + HFIELD_META_IDX_SIZE_Z] = Scalar[DTYPE](
            fmd.hfield_size[h * 4 + 2]
        )
        mf.hfield_meta.data[ho + HFIELD_META_IDX_SIZE_BASE] = Scalar[DTYPE](
            fmd.hfield_size[h * 4 + 3]
        )
        var src = fmd.hfield_adr[h]
        var n = fmd.hfield_nrow[h] * fmd.hfield_ncol[h]
        for k in range(n):
            mf.hfield_data0.data[hf_adr[h] + k] = Scalar[DTYPE](
                fmd.hfield_data[src + k]
            )

    # ── CAMERAS ──────────────────────────────────────────────────────────
    #
    # `mjModel.cam_*` for the batched ray tracer. Straight copies except for
    # the quaternion, which the RECORDS store (x, y, z, w) and `CameraData`
    # stores as four separate fields in the same order.
    #
    # ⚠ ANNOUNCED, NOT TRUNCATED — the `MAX_GPU_MESHES` lesson. A camera past
    # the cap would be a row nothing filled, i.e. `fovy = 0`, a degenerate
    # frustum that renders a single point and looks like a shading bug.
    comptime assert (
        _P_CAM_FIXED == CAM_MODE_FIXED
        and _P_CAM_TRACK == CAM_MODE_TRACK
        and _P_CAM_TRACKCOM == CAM_MODE_TRACKCOM
        and _P_CAM_TARGETBODY == CAM_MODE_TARGETBODY
        and _P_CAM_TARGETBODYCOM == CAM_MODE_TARGETBODYCOM
    ), (
        "parser/flat_model.mojo's CAM_MODE_* and gpu/constants.mojo's CAM_MODE_*"
        " have drifted. `fields_build` writes the first into CAM_IDX_MODE and"
        " `raytrace/camera.mojo` dispatches on the second."
    )
    if len(fmd.cameras) > MAX_GPU_CAMERAS:
        raise Error(
            String("this model declares ") + String(len(fmd.cameras))
            + " cameras and MAX_GPU_CAMERAS is " + String(MAX_GPU_CAMERAS)
            + ". The rows are comptime-sized; raise it in"
            " `gpu/constants.mojo`."
        )
    for c in range(len(fmd.cameras)):
        var cd = fmd.cameras[c]
        var cbase = c * MODEL_CAM_SIZE
        mf.cameras.data[cbase + CAM_IDX_BODY] = Scalar[DTYPE](cd.body_id)
        mf.cameras.data[cbase + CAM_IDX_POS_X] = Scalar[DTYPE](cd.pos_x)
        mf.cameras.data[cbase + CAM_IDX_POS_Y] = Scalar[DTYPE](cd.pos_y)
        mf.cameras.data[cbase + CAM_IDX_POS_Z] = Scalar[DTYPE](cd.pos_z)
        mf.cameras.data[cbase + CAM_IDX_QUAT_X] = Scalar[DTYPE](cd.quat_x)
        mf.cameras.data[cbase + CAM_IDX_QUAT_Y] = Scalar[DTYPE](cd.quat_y)
        mf.cameras.data[cbase + CAM_IDX_QUAT_Z] = Scalar[DTYPE](cd.quat_z)
        mf.cameras.data[cbase + CAM_IDX_QUAT_W] = Scalar[DTYPE](cd.quat_w)
        mf.cameras.data[cbase + CAM_IDX_FOVY] = Scalar[DTYPE](cd.fovy)
        mf.cameras.data[cbase + CAM_IDX_MODE] = Scalar[DTYPE](cd.mode)
        mf.cameras.data[cbase + CAM_IDX_TARGET_BODY] = Scalar[DTYPE](
            cd.target_body
        )
        # ⚠ THE TANGENT IS BAKED HERE BECAUSE `std.math.tan` IS CPU-ONLY.
        # See `CAM_IDX_TAN_HALF_FOVY`. Computed at float64 regardless of the
        # model's DTYPE, so a float32 model still gets a correctly-rounded
        # half-angle rather than the tangent of a rounded angle.
        mf.cameras.data[cbase + CAM_IDX_TAN_HALF_FOVY] = Scalar[DTYPE](
            tan(0.5 * cd.fovy * (pi / 180.0))
        )
        mf.cameras.data[cbase + CAM_IDX_ACTIVE] = Scalar[DTYPE](1)
    # ⚠ CAPACITY IS ANNOUNCED, NOT SILENTLY TRUNCATED. This loop used to just
    # `break` at the cap, which drops hull vertices from the LAST meshes and
    # shrinks their collision shape — an error with one sign, invisible to
    # every gate, and exactly the failure mode that let `mesh_vertadr`'s unit
    # bug hide. Exact hulls need roughly 10x what support sampling did (sawyer:
    # ~5.6k vertices against the 648 the sampler kept), so an under-sized
    # NMESHV is now easy to hit.
    # ⚠⚠ THIS RAISES NOW, AND THE PRINT IT REPLACED IS WHY. Every environment
    # ran with `NMESH_VERTS = 0` — `Phyics3dEnv` hardcoded it — so every mesh
    # geom in every env emitted NO CONTACT (both narrow phases guard the mesh
    # branch with `comptime if NMESH_VERTS > 0`). This message printed on every
    # sawyer construction the whole time and nothing read it. A printed "ERROR"
    # that no assert consumes is a comment, and it hid the entire mesh collider
    # from the env layer while every gate stayed green.
    #
    # Only safe because the loop above now skips non-collidable meshes: dog
    # carries 162 VISUAL meshes and would otherwise fail here for a feature it
    # does not use.
    if len(mesh_vert) > mf.dims.get_nmesh_verts() * 3:
        raise Error(
            String("mesh vertex capacity exceeded — mf.dims.get_nmesh_verts() = ")
            + String(mf.dims.get_nmesh_verts()) + " holds " + String(mf.dims.get_nmesh_verts() * 3)
            + " scalars but the COLLIDABLE hulls need " + String(len(mesh_vert))
            + ". Truncating would shrink those collision shapes silently, so"
            " this is fatal. Raise the config's mf.dims.get_nmesh_verts() to at least "
            + String((len(mesh_vert) + 2) // 3) + " vertices."
        )
    for i in range(len(mesh_vert)):
        if i >= mf.dims.get_nmesh_verts() * 3:
            break
        mf.mesh_verts.data[i] = mesh_vert[i]

    # ── the mesh TRIANGLE SOUP ───────────────────────────────────────────
    #
    # ⚠⚠ SILENT WHEN OFF, BY DESIGN AND SAID OUT LOUD. `nmesh_tri` defaults to
    # 0, so a model whose meshes nothing rays allocates nothing and this loop
    # writes nothing — `ray/mesh.mojo` then reports NO HIT on every mesh. That
    # is a capacity of zero being read as "no geometry", the exact shape of
    # `feedback_a_capacity_answer_read_as_a_feature_gate`, so it is stated
    # here and the ray entry point takes `ntri` explicitly rather than
    # inferring it.
    #
    # ⚠ The capacity check RAISES rather than truncating, like the hull's
    # above: a half-written soup is a mesh with holes in it, and a ray through
    # the missing part reports the geometry behind it.
    if mf.dims.get_nmesh_tri() > 0 and len(mesh_tri) > (
        mf.dims.get_nmesh_tri() * 9
    ):
        raise Error(
            String("physics3d: nmesh_tri holds ")
            + String(mf.dims.get_nmesh_tri())
            + " triangles but the collidable meshes carry "
            + String(len(mesh_tri) // 9)
            + ". Truncating would put holes in a surface a ray reads through,"
            " so this is fatal. Pass nmesh_tri=" + String(len(mesh_tri) // 9)
            + " to dims_from_flat."
        )
    for i in range(len(mesh_tri)):
        if i >= mf.dims.get_nmesh_tri() * 9:
            break
        mf.mesh_tris.data[i] = mesh_tri[i]

    # ── the soup's BVH ───────────────────────────────────────────────────
    #
    # Built HERE rather than lazily, because the alternative is a first frame
    # that pays for it inside a kernel. The nodes go into the SAME tensor,
    # after the triangles — see `MESH_ARENA_FLOATS_PER_TRI` for why one arena
    # and not two.
    #
    # ⚠ NOT AN OPTIONAL EXTRA THE MODEL CAN BE MISSING. A model with a soup
    # and no tree still works (`ray_model` takes the linear leg on
    # `BVHNUM == 0`), which is a fallback and not a mode: nothing sets it, and
    # a silent fall back to a 34x slower path is exactly the kind of thing
    # that goes unnoticed for a month. If this ever becomes conditional, the
    # condition needs a gate.
    var mesh_bvh = List[Scalar[DTYPE]]()
    var mesh_bvhadr = List[Int]()
    var mesh_bvhnum = List[Int]()
    var soup_carried = mf.dims.get_nmesh_tri() > 0 and len(mesh_tri) > 0
    if soup_carried:
        build_mesh_bvh[DTYPE](
            mesh_tri, mesh_triadr, mesh_trinum,
            mesh_bvh, mesh_bvhadr, mesh_bvhnum,
        )
        # The arena is `3 * nmesh_tri` records and the tree is `2n - 1` per
        # mesh, so a soup that fit CANNOT overflow this — which is the point
        # of sizing it that way. Checked anyway: the bound holds for the
        # builder as written, and a builder bug is what would break it.
        var tri_floats = len(mesh_tri)
        var cap = mf.dims.get_nmesh_tri() * MESH_ARENA_FLOATS_PER_TRI
        if tri_floats + len(mesh_bvh) > cap:
            raise Error(
                String("physics3d: the mesh arena holds ")
                + String(cap) + " floats but the soup plus its BVH need "
                + String(tri_floats + len(mesh_bvh))
                + ". A one-triangle-per-leaf tree is 2n-1 nodes, so this"
                " cannot happen for a soup that fit — the builder is wrong."
            )
        for i in range(len(mesh_bvh)):
            mf.mesh_tris.data[tri_floats + i] = mesh_bvh[i]

    # `mesh_meta` gains the soup's per-mesh window, parallel to the hull's.
    #
    # ⚠⚠ ONLY WHEN THE SOUP WAS ACTUALLY CARRIED, WHICH IS A FIX AND NOT A
    # GUARD. `mesh_tri` is collected for every collidable mesh REGARDLESS of
    # `nmesh_tri`; only the copy into the tensor is capacity-gated. So with
    # the default `nmesh_tri = 0` this loop used to publish a TRIADR/TRINUM
    # window into an arena of ONE float — a table saying "this mesh has 1 200
    # triangles" over memory that holds none. `ray/mesh.mojo`'s docstring has
    # claimed the opposite ("`nmesh_tri` defaults to 0, so ... `ntri` here is
    # 0") the whole time; nothing had noticed because the only ray consumer
    # was a heightfield scene. `MESH_META_IDX_BVHNUM` would have made it
    # WORSE — a walk over uninitialised escape indices rather than a scan over
    # zeros — which is how it surfaced.
    for m in range(num_meshes):
        if not soup_carried:
            break
        if m >= MAX_GPU_MESHES or m >= len(mesh_triadr):
            break
        mf.mesh_meta.data[
            m * MODEL_MESH_META_SIZE + MESH_META_IDX_TRIADR
        ] = Scalar[DTYPE](mesh_triadr[m])
        mf.mesh_meta.data[
            m * MODEL_MESH_META_SIZE + MESH_META_IDX_TRINUM
        ] = Scalar[DTYPE](mesh_trinum[m])
        if m < len(mesh_bvhadr):
            mf.mesh_meta.data[
                m * MODEL_MESH_META_SIZE + MESH_META_IDX_BVHADR
            ] = Scalar[DTYPE](mesh_bvhadr[m])
            mf.mesh_meta.data[
                m * MODEL_MESH_META_SIZE + MESH_META_IDX_BVHNUM
            ] = Scalar[DTYPE](mesh_bvhnum[m])

    # ── mesh polygon topology ────────────────────────────────────────────
    #
    # The capacities here are Euler's formula on NMESH_VERTS (see
    # `mesh_max_poly` / `mesh_max_polyvert`), so a convex hull that fits in the
    # vertex budget cannot overflow them. They are still checked, because the
    # bound holds for a CONVEX hull and a builder bug is exactly what would
    # break that assumption — and a silent overrun here would corrupt the
    # tensor allocated next to it.
    var NMESH_POLY = mesh_max_poly(mf.dims.get_nmesh_verts())
    var NMESH_POLYVERT = mesh_max_polyvert(mf.dims.get_nmesh_verts())
    if len(poly_vertadr) > NMESH_POLY or len(poly_vert) > NMESH_POLYVERT:
        raise Error(
            String("mesh POLYGON capacity exceeded: ")
            + String(len(poly_vertadr)) + " polygons (cap " + String(NMESH_POLY)
            + ") and " + String(len(poly_vert)) + " polygon-vertices (cap "
            + String(NMESH_POLYVERT) + ") for mf.dims.get_nmesh_verts() = "
            + String(mf.dims.get_nmesh_verts()) + ". These caps are Euler's formula for a"
            " CONVEX hull, so exceeding them means the hull or the polygon"
            " merge is wrong, not that the budget is too small."
        )
    # ⚠⚠ AND THE NARROW PHASE'S OWN TWO CAPS, SAID OUT LOUD. `MC_MAX_POLYVERT`
    # and `MC_MAX_DEG` size the multi-contact region of the CCD workspace row,
    # and neither is a capacity of THIS tensor — they are comptime because the
    # row's offsets are, where MuJoCo carries `npolygonmax` / `nmeshdegmax` as
    # runtime model fields with no cap at all.
    #
    # Exceeding either is NOT fatal: `_mesh_face` returns 0 and the caller
    # emits the single EPA point, which is exactly the reference's own
    # fallback when no faces line up. It is a LOST MANIFOLD, not a lost geom,
    # and refusing to load a model over it would be worse than the degradation.
    #
    # ⚠ IT USED TO BE SILENT, AND THAT COST A REAL DEFECT. At the old cap of
    # 16, 47 of Menagerie's 59 mesh-bearing scenes had at least one polygon it
    # could not hold and 39 had a vertex with more polygons than `MC_MAX_DEG`
    # — so those pairs quietly fell back to one point. At 56 it was still 21
    # scenes on the width axis; at today's 144 it is none, and this print
    # fires for no model in the tree. kinova_gen3, whose reset pose interpenetrates two 31-vertex-face
    # hulls, was 4.4e-02 from MuJoCo at step one and 5.7e-12 once the faces
    # fit. Print what was dropped; a bound nobody can see is a bound nobody
    # raises.
    #
    # ⚠⚠ AND IT IS THE ONLY REASON THE DEGREE GAP WAS EVER FOUND. `MC_MAX_DEG`
    # sat at 48 — Menagerie's worst plus one — while `envs/robots/assets/
    # so_arm101.xml`, a model this repo SHIPS, needs 50. Nothing pointed at it
    # except this print, on a test that loads so_arm101 for an unrelated
    # reason. When a cap is stated as "the tree's worst", say WHICH tree: the
    # census has to cover the models we ship as well as the reference ones.
    var _mc_worst_pv = 0
    var _mc_worst_deg = 0
    for p in range(len(poly_vertnum)):
        if poly_vertnum[p] > _mc_worst_pv:
            _mc_worst_pv = poly_vertnum[p]
    for v in range(len(polymap_num)):
        if polymap_num[v] > _mc_worst_deg:
            _mc_worst_deg = polymap_num[v]
    if _mc_worst_pv > MC_MAX_POLYVERT or _mc_worst_deg > MC_MAX_DEG:
        print(
            "physics3d: a collision mesh exceeds the multi-contact caps —"
            " widest face polygon", _mc_worst_pv, "vertices (MC_MAX_POLYVERT",
            MC_MAX_POLYVERT, ") and busiest vertex", _mc_worst_deg,
            "polygons (MC_MAX_DEG", MC_MAX_DEG, "). Those pairs fall back to"
            " a SINGLE contact point where MuJoCo would clip a face manifold"
            " of up to four. Matching the reference here needs at least",
            _mc_worst_pv, "and", _mc_worst_deg,
            "— raise them in `collision/ccd_workspace.mojo`, which sizes the"
            " workspace row they live in. Measured per SCENE over collision"
            " meshes only, the worst in Menagerie is 144 and 47 and the worst"
            " in this repo's own models is 82 and 50 (so_arm101's STS3215"
            " servo hulls), so the caps cover both today.",
        )
    for p in range(len(poly_vertadr)):
        var o = p * MODEL_MESH_POLY_SIZE
        mf.mesh_polys.data[o + MESH_POLY_IDX_VERTADR] = Scalar[DTYPE](
            poly_vertadr[p]
        )
        mf.mesh_polys.data[o + MESH_POLY_IDX_VERTNUM] = Scalar[DTYPE](
            poly_vertnum[p]
        )
        mf.mesh_polys.data[o + MESH_POLY_IDX_NX] = poly_normal[p * 3 + 0]
        mf.mesh_polys.data[o + MESH_POLY_IDX_NY] = poly_normal[p * 3 + 1]
        mf.mesh_polys.data[o + MESH_POLY_IDX_NZ] = poly_normal[p * 3 + 2]
    for k in range(len(poly_vert)):
        mf.mesh_polyvert.data[k] = Scalar[DTYPE](poly_vert[k])
    for k in range(len(polymap)):
        mf.mesh_polymap.data[k] = Scalar[DTYPE](polymap[k])
    for v in range(len(polymap_adr)):
        if v >= mf.dims.get_nmesh_verts():
            break
        mf.mesh_vert_polymap.data[v * 2 + 0] = Scalar[DTYPE](polymap_adr[v])
        mf.mesh_vert_polymap.data[v * 2 + 1] = Scalar[DTYPE](polymap_num[v])

    # ── hull edge graph ──────────────────────────────────────────────────
    # `Model` sizes this at 8 * NMESH_VERTS against MuJoCo's exact
    # `numvert + 3*numface` = 7V - 12. Checked rather than assumed, for the
    # same reason as the polygon caps above: the identity holds for a
    # TRIANGULATED CONVEX hull, so overflowing it means the hull builder is
    # wrong. Truncating would leave some vertices with a neighbour list that
    # runs off the end of the block, and `_plane_mesh_contacts` walks to a -1
    # terminator that would no longer be there.
    var NMESH_EDGE = mesh_max_edge(mf.dims.get_nmesh_verts())
    if len(edge_list) > NMESH_EDGE:
        raise Error(
            String("mesh hull EDGE capacity exceeded: ")
            + String(len(edge_list)) + " edge slots (cap "
            + String(NMESH_EDGE) + ") for mf.dims.get_nmesh_verts() = "
            + String(mf.dims.get_nmesh_verts()) + ". A triangulated convex hull needs"
            " 7V - 12, so exceeding 8V (plus the hill climb's seed tables,"
            " see `mesh_max_edge`) means the hull is not simplicial or the"
            " adjacency build is wrong, not that the budget is too small."
        )
    for v in range(len(edge_adr)):
        if v >= mf.dims.get_nmesh_verts():
            break
        mf.mesh_vert_edgeadr.data[v] = Scalar[DTYPE](edge_adr[v])
    for k in range(len(edge_list)):
        mf.mesh_edges.data[k] = Scalar[DTYPE](edge_list[k])

    # ── sites (INTENTIONAL FIX: legacy load_from_model left these zero) ───
    for i in range(mf.dims.get_nsite()):
        var sd = fmd.sites[i]
        var o = i * MODEL_SITE_SIZE
        mf.sites.data[o + SITE_IDX_BODY] = Scalar[DTYPE](sd.body_id)
        mf.sites.data[o + SITE_IDX_POS_X] = Scalar[DTYPE](sd.pos_x)
        mf.sites.data[o + SITE_IDX_POS_Y] = Scalar[DTYPE](sd.pos_y)
        mf.sites.data[o + SITE_IDX_POS_Z] = Scalar[DTYPE](sd.pos_z)
        mf.sites.data[o + SITE_IDX_TYPE] = Scalar[DTYPE](sd.site_type)
        mf.sites.data[o + SITE_IDX_SIZE_0] = Scalar[DTYPE](sd.size_0)
        mf.sites.data[o + SITE_IDX_SIZE_1] = Scalar[DTYPE](sd.size_1)
        mf.sites.data[o + SITE_IDX_SIZE_2] = Scalar[DTYPE](sd.size_2)
        mf.sites.data[o + SITE_IDX_QUAT_X] = Scalar[DTYPE](sd.quat_x)
        mf.sites.data[o + SITE_IDX_QUAT_Y] = Scalar[DTYPE](sd.quat_y)
        mf.sites.data[o + SITE_IDX_QUAT_Z] = Scalar[DTYPE](sd.quat_z)
        mf.sites.data[o + SITE_IDX_QUAT_W] = Scalar[DTYPE](sd.quat_w)

    # ── sensors ──────────────────────────────────────────────────────────
    # ⚠ CAPACITY-GUARDED LIKE TENDONS, NOT EXACT LIKE SITES. `NSENSOR` is a
    # hand-supplied parameter of `ModelDefFromXML` (the MJCF is not readable at
    # compile time in this tree), so a model can declare more `<sensor>`
    # elements than the caller reserved room for. The guard below stops the
    # write; the RAISE that makes that loud lives in `init_fields`, beside the
    # other declared-vs-parsed checks.
    #
    # ⚠ EVERY ROW IS WRITTEN, SERVED OR NOT. An unserved sensor's row carries
    # its real `dim` and `adr` and an `objid` of -1 — it holds its slot so the
    # sensors after it keep the right offset. See `_fill_sensors`.
    for i in range(len(fmd.sensors)):
        if i >= mf.dims.get_nsensor():
            break
        var se = fmd.sensors[i]
        var so = i * MODEL_SENSOR_SIZE
        mf.sensors.data[so + SENSOR_IDX_TYPE] = Scalar[DTYPE](se.sensor_type)
        mf.sensors.data[so + SENSOR_IDX_OBJTYPE] = Scalar[DTYPE](se.objtype)
        mf.sensors.data[so + SENSOR_IDX_OBJID] = Scalar[DTYPE](se.objid)
        mf.sensors.data[so + SENSOR_IDX_DIM] = Scalar[DTYPE](se.dim)
        mf.sensors.data[so + SENSOR_IDX_ADR] = Scalar[DTYPE](se.adr)
        mf.sensors.data[so + SENSOR_IDX_DATATYPE] = Scalar[DTYPE](se.datatype)
        mf.sensors.data[so + SENSOR_IDX_NEEDSTAGE] = Scalar[DTYPE](
            se.needstage
        )
        mf.sensors.data[so + SENSOR_IDX_CUTOFF] = Scalar[DTYPE](se.cutoff)
        mf.sensors.data[so + SENSOR_IDX_BODY] = Scalar[DTYPE](se.body_id)
        mf.sensors.data[so + SENSOR_IDX_SERVED] = Scalar[DTYPE](
            1.0 if se.served else 0.0
        )
        mf.sensors.data[so + SENSOR_IDX_REFTYPE] = Scalar[DTYPE](se.reftype)
        mf.sensors.data[so + SENSOR_IDX_REFID] = Scalar[DTYPE](se.refid)
        # ⚠ ON EVERY ROW, NOT ONLY THE MAGNETOMETERS. `<option magnetic>` is
        # a document global; writing it everywhere costs three floats per row
        # and removes the one way this could go wrong — a row that is a
        # magnetometer by a path the packer did not anticipate, reading three
        # zeros. Only `SENS_MAGNETOMETER` ever looks at them.
        mf.sensors.data[so + SENSOR_IDX_MAG_X] = Scalar[DTYPE](
            fmd.opt_magnetic_x
        )
        mf.sensors.data[so + SENSOR_IDX_MAG_Y] = Scalar[DTYPE](
            fmd.opt_magnetic_y
        )
        mf.sensors.data[so + SENSOR_IDX_MAG_Z] = Scalar[DTYPE](
            fmd.opt_magnetic_z
        )

    # ── tendons ──────────────────────────────────────────────────────────
    #
    # NTENDON_P == MAX_TENDON by construction (ModelDefFromXML passes
    # `max_tendon` for both), so this is a straight copy. INVWEIGHT0 is left
    # zero here and filled by the invweight pass, which needs FK at qpos0.
    for i in range(len(fmd.tendons)):
        if i >= mf.dims.get_ntendon():
            break
        var td = fmd.tendons[i]
        var o = i * MODEL_TENDON_SIZE
        mf.tendons.data[o + TENDON_IDX_KIND] = Scalar[DTYPE](td.kind)
        mf.tendons.data[o + TENDON_IDX_IS_EQUALITY] = Scalar[DTYPE](
            td.is_equality
        )
        mf.tendons.data[o + TENDON_IDX_NUM_JOINTS] = Scalar[DTYPE](
            td.num_joints
        )
        # ⚠ LOOPED, NOT UNROLLED. These were four explicit writes each, which
        # is how the 4-wrap cap got baked into a fourth place: widening the
        # record without touching this would have left wraps 4..15 as
        # uninitialised garbage. Driving both from `TENDON_MAX_WRAPS` means the
        # cap moves in one edit.
        for k in range(TENDON_MAX_WRAPS):
            mf.tendons.data[o + TENDON_IDX_JOINT_0 + k] = Scalar[DTYPE](
                td.joint_ids[k]
            )
            mf.tendons.data[o + TENDON_IDX_COEF_0 + k] = Scalar[DTYPE](
                td.coefs[k]
            )
        mf.tendons.data[o + TENDON_IDX_LENGTH_REF] = Scalar[DTYPE](
            td.length_ref
        )
        mf.tendons.data[o + TENDON_IDX_NUM_WRAPS] = Scalar[DTYPE](td.num_wraps)
        # ⚠ ALL THREE RUNS, ALWAYS. The type run is what tells the dynamics
        # whether entry `k` is a site or a wrap geom; leaving it at the slab's
        # default would make every waypoint read as `WRAP_NONE` and the tendon
        # route through nothing.
        for k in range(TENDON_MAX_SPATIAL_WRAPS):
            mf.tendons.data[o + TENDON_IDX_WOBJ_0 + k] = Scalar[DTYPE](
                td.wrap_objs[k]
            )
            mf.tendons.data[o + TENDON_IDX_WTYPE_0 + k] = Scalar[DTYPE](
                td.wrap_types[k]
            )
            mf.tendons.data[o + TENDON_IDX_WPRM_0 + k] = Scalar[DTYPE](
                td.wrap_sides[k]
            )
        mf.tendons.data[o + TENDON_IDX_LIMITED] = Scalar[DTYPE](td.limited)
        mf.tendons.data[o + TENDON_IDX_RANGE_MIN] = Scalar[DTYPE](td.range_min)
        mf.tendons.data[o + TENDON_IDX_RANGE_MAX] = Scalar[DTYPE](td.range_max)
        mf.tendons.data[o + TENDON_IDX_MARGIN] = Scalar[DTYPE](td.margin)
        mf.tendons.data[o + TENDON_IDX_SOLREF_LIM_0] = Scalar[DTYPE](
            td.solref_lim_0
        )
        mf.tendons.data[o + TENDON_IDX_SOLREF_LIM_1] = Scalar[DTYPE](
            td.solref_lim_1
        )
        mf.tendons.data[o + TENDON_IDX_SOLIMP_LIM_0] = Scalar[DTYPE](
            td.solimp_lim_0
        )
        mf.tendons.data[o + TENDON_IDX_SOLIMP_LIM_1] = Scalar[DTYPE](
            td.solimp_lim_1
        )
        mf.tendons.data[o + TENDON_IDX_SOLIMP_LIM_2] = Scalar[DTYPE](
            td.solimp_lim_2
        )
        mf.tendons.data[o + TENDON_IDX_SOLIMP_LIM_3] = Scalar[DTYPE](
            td.solimp_lim_3
        )
        mf.tendons.data[o + TENDON_IDX_SOLIMP_LIM_4] = Scalar[DTYPE](
            td.solimp_lim_4
        )
        # The EQUALITY pair, slots 10..16. `constraints/equality_tendon.mojo`
        # has always read these; until 2026-07-31 nothing wrote them, so every
        # tendon-equality row would have been built on a zero solref timeconst
        # (had any row ever been built — `is_equality` was never set either).
        mf.tendons.data[o + TENDON_IDX_SOLREF_0] = Scalar[DTYPE](td.solref_eq_0)
        mf.tendons.data[o + TENDON_IDX_SOLREF_1] = Scalar[DTYPE](td.solref_eq_1)
        mf.tendons.data[o + TENDON_IDX_SOLIMP_0] = Scalar[DTYPE](td.solimp_eq_0)
        mf.tendons.data[o + TENDON_IDX_SOLIMP_1] = Scalar[DTYPE](td.solimp_eq_1)
        mf.tendons.data[o + TENDON_IDX_SOLIMP_2] = Scalar[DTYPE](td.solimp_eq_2)
        mf.tendons.data[o + TENDON_IDX_SOLIMP_3] = Scalar[DTYPE](td.solimp_eq_3)
        mf.tendons.data[o + TENDON_IDX_SOLIMP_4] = Scalar[DTYPE](td.solimp_eq_4)

    # ── equality constraints (legacy add_connect/add_weld semantics:
    #    solimp[3]=0.5 / solimp[4]=2.0 hardcoded, parsed values dropped) ────
    var num_eq = 0
    for i in range(len(fmd.equalities)):
        if num_eq >= mf.dims.get_nequality():
            break
        var ed = fmd.equalities[i]
        if (
            ed.eq_type != _EQ_CONNECT
            and ed.eq_type != _EQ_WELD
            and ed.eq_type != _EQ_JOINT
        ):
            continue
        var o = num_eq * MODEL_EQ_SIZE
        mf.equality.data[o + EQ_IDX_TYPE] = Scalar[DTYPE](ed.eq_type)
        mf.equality.data[o + EQ_IDX_OBJTYPE] = Scalar[DTYPE](ed.objtype)
        mf.equality.data[o + EQ_IDX_BODY_A] = Scalar[DTYPE](ed.body_a)
        mf.equality.data[o + EQ_IDX_BODY_B] = Scalar[DTYPE](ed.body_b)
        mf.equality.data[o + EQ_IDX_ANCHOR_AX] = Scalar[DTYPE](ed.anchor_a_x)
        mf.equality.data[o + EQ_IDX_ANCHOR_AY] = Scalar[DTYPE](ed.anchor_a_y)
        mf.equality.data[o + EQ_IDX_ANCHOR_AZ] = Scalar[DTYPE](ed.anchor_a_z)
        mf.equality.data[o + EQ_IDX_ANCHOR_BX] = Scalar[DTYPE](ed.anchor_b_x)
        mf.equality.data[o + EQ_IDX_ANCHOR_BY] = Scalar[DTYPE](ed.anchor_b_y)
        mf.equality.data[o + EQ_IDX_ANCHOR_BZ] = Scalar[DTYPE](ed.anchor_b_z)
        if ed.eq_type == _EQ_WELD:
            mf.equality.data[o + EQ_IDX_RELPOSE_X] = Scalar[DTYPE](
                ed.relpose_x
            )
            mf.equality.data[o + EQ_IDX_RELPOSE_Y] = Scalar[DTYPE](
                ed.relpose_y
            )
            mf.equality.data[o + EQ_IDX_RELPOSE_Z] = Scalar[DTYPE](
                ed.relpose_z
            )
            mf.equality.data[o + EQ_IDX_RELPOSE_W] = Scalar[DTYPE](
                ed.relpose_w
            )
        else:
            mf.equality.data[o + EQ_IDX_RELPOSE_X] = Scalar[DTYPE](0)
            mf.equality.data[o + EQ_IDX_RELPOSE_Y] = Scalar[DTYPE](0)
            mf.equality.data[o + EQ_IDX_RELPOSE_Z] = Scalar[DTYPE](0)
            mf.equality.data[o + EQ_IDX_RELPOSE_W] = Scalar[DTYPE](1)
        mf.equality.data[o + EQ_IDX_SOLREF_0] = Scalar[DTYPE](ed.solref_0)
        mf.equality.data[o + EQ_IDX_SOLREF_1] = Scalar[DTYPE](ed.solref_1)
        mf.equality.data[o + EQ_IDX_SOLIMP_0] = Scalar[DTYPE](ed.solimp_0)
        mf.equality.data[o + EQ_IDX_SOLIMP_1] = Scalar[DTYPE](ed.solimp_1)
        mf.equality.data[o + EQ_IDX_SOLIMP_2] = Scalar[DTYPE](ed.solimp_2)
        # ⚠ THE PARSER READS ALL FIVE; THIS WROTE THE DEFAULTS FOR TWO
        # (AUD-06). `_fill_equality_solparams` fills `solimp_3` (the width of
        # the impedance ramp) and `solimp_4` (its power) from the element,
        # the class and the root default, and both were then thrown away
        # here. The existing equality gates all state `solimp="… 0.5 2"`, so
        # every one of them agreed with the hardcode — the tendon equality
        # writer four blocks up already passes its own two through.
        mf.equality.data[o + EQ_IDX_SOLIMP_3] = Scalar[DTYPE](ed.solimp_3)
        mf.equality.data[o + EQ_IDX_SOLIMP_4] = Scalar[DTYPE](ed.solimp_4)
        # Weld only (MuJoCo's eq_data[10]); connect has no orientation rows,
        # so 1 keeps the multiply a no-op rather than zeroing anything.
        mf.equality.data[o + EQ_IDX_TORQUESCALE] = Scalar[DTYPE](
            ed.torquescale if ed.eq_type == _EQ_WELD else 1.0
        )
        num_eq += 1
    mf.meta.data[MODEL_META_IDX_NEQUALITY] = Scalar[DTYPE](num_eq)

    # ── contact exclusion pairs ────────────────────────────────────────────
    # ⚠ ANNOUNCED, like the pair check below and for the same reason. This
    # loop used to write `len(fmd.excludes)` entries into a tensor sized by
    # NEXCLUDE with nothing in between, so a model def that left `nexclude` at
    # its default of 0 ran off the end of a 1-element allocation. That
    # surfaced as `index 1 is out of bounds` from deep inside `fields_build` —
    # a stack trace that names neither the parameter nor the model, and costs
    # a bisect to trace back to a missing `nexclude=pm.NEXCLUDE`.
    #
    # ⚠ It is a HARD ERROR rather than a clamp because dropping exclusions
    # ADDS collisions: the excluded pairs start colliding, and the model still
    # simulates. Same argument as `<pair>` below.
    if len(fmd.excludes) > mf.dims.get_nexclude():
        raise Error(
            String("physics3d: parsed ")
            + String(len(fmd.excludes))
            + " <contact><exclude> entries but mf.dims.get_nexclude() = "
            + String(mf.dims.get_nexclude())
            + ". Pass `nexclude=parse_xml(xml).mf.dims.get_nexclude()` to the model def."
            " Truncating would let the excluded geom pairs collide, which"
            " leaves a model that still runs and is quietly wrong."
        )
    for i in range(len(fmd.excludes)):
        var ex = fmd.excludes[i]
        mf.excludes.data[i * 2 + 0] = Scalar[DTYPE](ex.body1)
        mf.excludes.data[i * 2 + 1] = Scalar[DTYPE](ex.body2)

    # ── predefined contact pairs ───────────────────────────────────────────
    # ⚠ RAISES rather than clamping. `NPAIR` is the sole knob and is derived
    # from the same `<contact>` text `_fill_pairs` walks, so a mismatch means
    # the two disagree about the model — and a clamp would drop the tail
    # pairs, which is invisible: the model still simulates, just without some
    # of the collisions it declared. Compare the tendon count above, which
    # clamps to `MAX_TENDON`.
    if len(fmd.pairs) > mf.dims.get_npair():
        raise Error(
            "physics3d: parsed "
            + String(len(fmd.pairs))
            + " <contact><pair> records but mf.dims.get_npair() is "
            + String(mf.dims.get_npair())
            + ". Pass npair=<count> to ModelDefFromXML."
        )
    for i in range(len(fmd.pairs)):
        var pr = fmd.pairs[i]
        var o = i * MODEL_PAIR_SIZE
        mf.pairs.data[o + PAIR_IDX_GEOM1] = Scalar[DTYPE](pr.geom1)
        mf.pairs.data[o + PAIR_IDX_GEOM2] = Scalar[DTYPE](pr.geom2)
        mf.pairs.data[o + PAIR_IDX_CONDIM] = Scalar[DTYPE](pr.condim)
        mf.pairs.data[o + PAIR_IDX_FRICTION] = Scalar[DTYPE](pr.friction)
        mf.pairs.data[o + PAIR_IDX_FRICTION_SPIN] = Scalar[DTYPE](
            pr.friction_spin
        )
        mf.pairs.data[o + PAIR_IDX_FRICTION_ROLL] = Scalar[DTYPE](
            pr.friction_roll
        )
        mf.pairs.data[o + PAIR_IDX_SOLREF_0] = Scalar[DTYPE](pr.solref_0)
        mf.pairs.data[o + PAIR_IDX_SOLREF_1] = Scalar[DTYPE](pr.solref_1)
        mf.pairs.data[o + PAIR_IDX_SOLIMP_0] = Scalar[DTYPE](pr.solimp_0)
        mf.pairs.data[o + PAIR_IDX_SOLIMP_1] = Scalar[DTYPE](pr.solimp_1)
        mf.pairs.data[o + PAIR_IDX_SOLIMP_2] = Scalar[DTYPE](pr.solimp_2)
        mf.pairs.data[o + PAIR_IDX_SOLIMP_3] = Scalar[DTYPE](pr.solimp_3)
        mf.pairs.data[o + PAIR_IDX_SOLIMP_4] = Scalar[DTYPE](pr.solimp_4)
        mf.pairs.data[o + PAIR_IDX_MARGIN] = Scalar[DTYPE](pr.margin)
        mf.pairs.data[o + PAIR_IDX_GAP] = Scalar[DTYPE](pr.gap)

    # ── <compiler inertiafromgeom> + settotalmass (staging mutations) ─────
    var ifg_mode = fmd.inertiafromgeom
    if ifg_mode == 1 or ifg_mode == 2:
        _inertia_from_geoms_staging[DTYPE](
            mf.dims.get_nbody(),
            mf.dims.get_ngeom(),
            fmd.inertiagrouprange_min,
            fmd.inertiagrouprange_max,
            ifg_mode == 2,
            mf.geoms.data,
            mf.bodies.data,
            geom_mass,
            geom_density,
            geom_group,
            geom_mesh_vol,
            geom_mesh_eig,
            body_has_explicit_inertia,
            body_mass,
            body_inv_mass,
            body_inertia,
            body_inv_inertia,
            body_ipos,
            body_iquat,
        )
    # ── <compiler boundmass / boundinertia> ──────────────────────────────
    #
    # `mjCBody::Compile`: for EVERY body with id > 0, after the inertial frame
    # is set, `mass = max(mass, boundmass)` and the same per principal moment.
    # ⚠ Applied to every body, not only the ones `inertiafromgeom` touched —
    # a body with an explicit `<inertial>` is bounded too, and so is one with
    # no geoms at all. That last case is the whole point on composer models:
    # `jaco_arm/` and `jaco_arm/jaco_hand/` are pure attachment frames carrying
    # NO geoms, and `b_6`'s only geom has mass 1e-9. All three take their
    # entire mass (1e-05) and inertia (1e-11) from these two numbers.
    #
    # ⚠ Runs AFTER the inertia staging and BEFORE settotalmass, matching
    # MuJoCo's order — settotalmass rescales whatever the bound produced.
    if fmd.boundmass > 0.0 or fmd.boundinertia > 0.0:
        var bm = Scalar[DTYPE](fmd.boundmass)
        var bi = Scalar[DTYPE](fmd.boundinertia)
        for i in range(1, mf.dims.get_nbody()):
            if body_mass[i] < bm:
                body_mass[i] = bm
                body_inv_mass[i] = Scalar[DTYPE](1.0) / bm
            for k in range(3):
                if body_inertia[i * 3 + k] < bi:
                    body_inertia[i * 3 + k] = bi
                    body_inv_inertia[i * 3 + k] = Scalar[DTYPE](1.0) / bi

    # ⚠ `settotalmass` IS UNCONDITIONAL (AUD-11). It used to be gated on
    # `ifg_mode > 0` with a comment calling that "MuJoCo's legacy ordering";
    # there is no such ordering. `mjCModel::Compile`
    # (user_model.cc:5455-5458) runs `if (compiler.settotalmass > 0)
    # mj_setTotalmass(m, ...)` after everything else, and `mj_setTotalmass`
    # (engine_support.c) scales `body_mass` and all three `body_inertia`
    # components for every body but the world. MEASURED on the 3.12 runtime,
    # one two-body model with `settotalmass="14"` and explicit `<inertial>`
    # tags, all four spellings of the compiler flag:
    #
    #     inertiafromgeom="false"  ->  [5.25, 8.75]   sum 14
    #     inertiafromgeom="true"   ->  [7.00, 7.00]   sum 14
    #     inertiafromgeom="auto"   ->  [5.25, 8.75]   sum 14
    #     (absent)                 ->  [5.25, 8.75]   sum 14
    #
    # The flag changes WHAT the masses are, never WHETHER they are rescaled.
    # The old gate made a model that states its inertias by hand and asks for
    # a total mass keep the hand-written total — dm_control's cheetah passed
    # only because it takes the AUTO default and so entered the branch.
    if fmd.settotalmass > 0.0:
        var total_mass = Scalar[DTYPE](0)
        for i in range(1, mf.dims.get_nbody()):
            total_mass += body_mass[i]
        if total_mass > Scalar[DTYPE](0):
            var scale = Scalar[DTYPE](fmd.settotalmass) / total_mass
            for i in range(1, mf.dims.get_nbody()):
                body_mass[i] *= scale
                body_inv_mass[i] = Scalar[DTYPE](1.0) / body_mass[i]
                for k in range(3):
                    body_inertia[i * 3 + k] *= scale
                    body_inv_inertia[i * 3 + k] = (
                        Scalar[DTYPE](1.0) / body_inertia[i * 3 + k]
                    )

    # ── body mass/inertia record write (post ifg/settotalmass) ────────────
    for b in range(len(fmd.bodies) + 1):
        var o = b * MODEL_BODY_SIZE
        mf.bodies.data[o + BODY_IDX_MASS] = body_mass[b]
        mf.bodies.data[o + BODY_IDX_INV_MASS] = body_inv_mass[b]
        mf.bodies.data[o + BODY_IDX_IXX] = body_inertia[b * 3 + 0]
        mf.bodies.data[o + BODY_IDX_IYY] = body_inertia[b * 3 + 1]
        mf.bodies.data[o + BODY_IDX_IZZ] = body_inertia[b * 3 + 2]
        mf.bodies.data[o + BODY_IDX_INV_IXX] = body_inv_inertia[b * 3 + 0]
        mf.bodies.data[o + BODY_IDX_INV_IYY] = body_inv_inertia[b * 3 + 1]
        mf.bodies.data[o + BODY_IDX_INV_IZZ] = body_inv_inertia[b * 3 + 2]
        mf.bodies.data[o + BODY_IDX_IPOS_X] = body_ipos[b * 3 + 0]
        mf.bodies.data[o + BODY_IDX_IPOS_Y] = body_ipos[b * 3 + 1]
        mf.bodies.data[o + BODY_IDX_IPOS_Z] = body_ipos[b * 3 + 2]
        mf.bodies.data[o + BODY_IDX_IQUAT_X] = body_iquat[b * 4 + 0]
        mf.bodies.data[o + BODY_IDX_IQUAT_Y] = body_iquat[b * 4 + 1]
        mf.bodies.data[o + BODY_IDX_IQUAT_Z] = body_iquat[b * 4 + 2]
        mf.bodies.data[o + BODY_IDX_IQUAT_W] = body_iquat[b * 4 + 3]

    # ── kinematic TREES — M's diagonal blocks ────────────────────────────
    #
    # ⚠⚠ THIS IS THE MASS MATRIX'S SPARSITY, WRITTEN DOWN. M couples a dof
    # only with its tree ANCESTORS, so its diagonal blocks are exactly the
    # trees and every entry outside one is STRUCTURALLY zero. See
    # `MODEL_TREE_SIZE` for the argument and the two CRBA sites that
    # guarantee it.
    #
    # ⚠⚠ LAST IN THIS FUNCTION, AND THE POSITION IS LOAD-BEARING. It reads
    # `body_rootid`, the joint records, AND `BODY_IDX_IPOS_*`/`IQUAT_*` — and
    # those last two are written by the loop DIRECTLY ABOVE, at the very end of
    # the build, because they are only final after `inertiafromgeom` and
    # `settotalmass`. Built beside `body_rootid` instead (1,300 lines earlier)
    # it reads them as ZEROS, every body fails `iquat == identity`, and every
    # block is classified DENSE — correct output, silently no faster, and the
    # partition arm of the gate still passes. That is how it was written the
    # first time; `test_tree_blocks_vs_mujoco`'s conservative-block COUNT is
    # what caught it, which is why that count is printed rather than tallied.
    #
    # It must also come BEFORE `compute_invweight0`, which calls `ldl_factor`
    # during model build — that is a later step of `build_model_runtime`, so
    # the end of this function satisfies both.
    #
    # Two dofs share a tree iff their bodies share `BODY_IDX_ROOTID`. Dof
    # ranges are contiguous per body and bodies are in tree order, so a tree
    # is one `(adr, num)` interval and this is a single pass over the dofs.
    var _nv_t = mf.dims.get_nv()
    # dof -> owning body. Same construction as `build_actuator_damping`'s
    # (`fields_build.mojo:1029`); built from the joint records rather than
    # stored, because only these two passes want it.
    var dof_body_t = List[Int](length=_nv_t if _nv_t > 0 else 1, fill=0)
    for j in range(len(fmd.joints)):
        var jo = j * MODEL_JOINT_SIZE
        var jb = Int(mf.joints.data[jo + JOINT_IDX_BODY_ID])
        var jda = Int(mf.joints.data[jo + JOINT_IDX_DOF_ADR])
        var jt = Int(mf.joints.data[jo + JOINT_IDX_TYPE])
        var jnd = 1
        if jt == JNT_FREE:
            jnd = 6
        elif jt == JNT_BALL:
            jnd = 3
        for k in range(jnd):
            if jda + k >= 0 and jda + k < _nv_t:
                dof_body_t[jda + k] = jb

    var ntree = 0
    var d0 = 0
    while d0 < _nv_t:
        var root0 = Int(
            mf.bodies.data[dof_body_t[d0] * MODEL_BODY_SIZE + BODY_IDX_ROOTID]
        )
        var d1 = d0 + 1
        while d1 < _nv_t:
            var r1 = Int(
                mf.bodies.data[
                    dof_body_t[d1] * MODEL_BODY_SIZE + BODY_IDX_ROOTID
                ]
            )
            if r1 != root0:
                break
            d1 += 1
        var num = d1 - d0

        # ── kind ──
        #
        # ⚠⚠ CONSERVATIVE ON PURPOSE, AND THE ASYMMETRY IS THE POINT. Calling
        # a block COMPACT when M is not diagonal is a SILENT WRONG ANSWER;
        # calling it DENSE when M is diagonal is only slower. So this demands
        # a condition strong enough to be airtight rather than one that
        # matches MuJoCo's `body_simple` row for row.
        #
        # ⚠ THE STRENGTHENING IS "EXACTLY ONE BODY". MuJoCo also admits a body
        # whose parent is a FIXED CHILD OF WORLD (`user_model.cc:2814`), but
        # then `body_rootid` is that parent and the free joint's rotational
        # cdofs are anchored at `subtree_com[root]` — the COMBINED com of both
        # bodies — which is not the child's own com unless the parent is
        # massless. That leaves a translation/rotation coupling MuJoCo's rule
        # tolerates and this one will not guess at. With one body,
        # `subtree_com[root] == xipos == xpos` (ipos is zero, below) and the
        # offset is identically zero.
        #
        # The remaining conditions are MuJoCo's, and each one is load-bearing:
        #   ipos == 0, iquat == identity  the inertia is diagonal IN THE BODY
        #                                 AXES the rotational cdofs use;
        #   jnt_pos == 0                  the hinge anchor is the com, so the
        #                                 cdof has no linear part;
        #   axis-aligned slide/hinge      two mis-aligned axes have a nonzero
        #                                 dot product;
        #   at most one rotational joint  a second one couples to the first.
        var kind = TREE_KIND_COMPACT
        var b0 = dof_body_t[d0]
        for k in range(d0 + 1, d1):
            if dof_body_t[k] != b0:
                kind = TREE_KIND_DENSE
        var bo = b0 * MODEL_BODY_SIZE
        # ⚠ A 1x1 BLOCK IS DIAGONAL BY DEFINITION — it has no off-diagonal
        # entry to discard, so none of the conditions below can make it wrong.
        # Checking them anyway is not conservative, it is a MISS: `finger`'s
        # lone hinge has `iquat` at 90 degrees and a `-6.5e-19` `ipos`, both
        # irrelevant to a single dof, and demanding an identity inertial frame
        # cost the one block this classifier used to get wrong against MuJoCo.
        # ⚠ A 1x1 BLOCK IS DIAGONAL BY DEFINITION — it has no off-diagonal
        # entry to discard, so nothing below can make it wrong, and testing it
        # anyway is not conservatism but a MISS: `finger`'s lone hinge has its
        # inertial frame at 90 degrees and an `ipos` of -6.5e-19, both
        # irrelevant to a single dof, and demanding an identity inertial frame
        # was the one block this classifier got wrong against MuJoCo.
        if num > 1:
            if (
                mf.bodies.data[bo + BODY_IDX_IPOS_X] != 0
                or mf.bodies.data[bo + BODY_IDX_IPOS_Y] != 0
                or mf.bodies.data[bo + BODY_IDX_IPOS_Z] != 0
                or mf.bodies.data[bo + BODY_IDX_IQUAT_X] != 0
                or mf.bodies.data[bo + BODY_IDX_IQUAT_Y] != 0
                or mf.bodies.data[bo + BODY_IDX_IQUAT_Z] != 0
                or mf.bodies.data[bo + BODY_IDX_IQUAT_W] != 1
            ):
                kind = TREE_KIND_DENSE
            var rotfound = False
            for j in range(len(fmd.joints)):
                var jo = j * MODEL_JOINT_SIZE
                if Int(mf.joints.data[jo + JOINT_IDX_BODY_ID]) != b0:
                    continue
                var jt = Int(mf.joints.data[jo + JOINT_IDX_TYPE])
                # A joint FOLLOWING a rotational one is demoted whatever it is
                # — MuJoCo's `rotfound` test, `user_model.cc:2851`.
                if rotfound:
                    kind = TREE_KIND_DENSE
                if (
                    mf.joints.data[jo + JOINT_IDX_POS_X] != 0
                    or mf.joints.data[jo + JOINT_IDX_POS_Y] != 0
                    or mf.joints.data[jo + JOINT_IDX_POS_Z] != 0
                ):
                    kind = TREE_KIND_DENSE
                if jt == JNT_HINGE or jt == JNT_SLIDE:
                    var na = 0
                    if abs(mf.joints.data[jo + JOINT_IDX_AXIS_X]) > 1e-14:
                        na += 1
                    if abs(mf.joints.data[jo + JOINT_IDX_AXIS_Y]) > 1e-14:
                        na += 1
                    if abs(mf.joints.data[jo + JOINT_IDX_AXIS_Z]) > 1e-14:
                        na += 1
                    if na != 1:
                        kind = TREE_KIND_DENSE
                # ⚠ `JNT_FREE` IS IN THIS LIST AND IS NOT IN MUJOCO'S
                # (`user_model.cc:2859` marks BALL and HINGE only). MJCF
                # forbids a free joint alongside any other, so the two rules
                # cannot disagree on a loadable model; ours demotes the
                # impossible case instead of calling six free dofs plus a
                # hinge diagonal.
                if jt == JNT_BALL or jt == JNT_HINGE or jt == JNT_FREE:
                    rotfound = True

        var to = ntree * MODEL_TREE_SIZE
        mf.trees.data[to + TREE_IDX_DOF_ADR] = Scalar[DTYPE](d0)
        mf.trees.data[to + TREE_IDX_DOF_NUM] = Scalar[DTYPE](num)
        mf.trees.data[to + TREE_IDX_KIND] = Scalar[DTYPE](kind)
        ntree += 1
        d0 = d1
    mf.meta.data[MODEL_META_IDX_NTREE] = Scalar[DTYPE](ntree)

    # ── `dof_parentid` (MuJoCo `m->dof_parentid`), from the dof->body map
    # above. Dofs of one body chain in order — across that body's joints too —
    # and a body's first dof hangs from the last dof of the nearest ancestor
    # body that has any; a dof with no such ancestor is a root (-1). Gated
    # entry for entry against MuJoCo in `test_dof_parentid_vs_mujoco`.
    var _nbody_t = mf.dims.get_nbody()
    var body_last_dof = List[Int](
        length=_nbody_t if _nbody_t > 0 else 1, fill=-1
    )
    for d in range(_nv_t):
        body_last_dof[dof_body_t[d]] = d
    for d in range(_nv_t):
        var b = dof_body_t[d]
        var par = -1
        if d > 0 and dof_body_t[d - 1] == b:
            par = d - 1
        else:
            var pb = Int(mf.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_PARENT])
            while pb > 0:
                if body_last_dof[pb] >= 0:
                    par = body_last_dof[pb]
                    break
                pb = Int(mf.bodies.data[pb * MODEL_BODY_SIZE + BODY_IDX_PARENT])
        mf.dof_parentid.data[d] = Scalar[DTYPE](par)



# =============================================================================
# Actuation records (phase 1a.2)
# =============================================================================


def build_spec_fields[DTYPE: DType, D: DimsLike](
    fmd: FlatModelDef,
    mut sf: SpecFields[DTYPE, D],
) raises:
    """Fill the spec record tensors from the parsed `fmd`.

    The runtime twin of what `parse_xml_model_data` produces for the comptime
    `_acd`, and gated against it field by field while both exist
    (`tests/physics3d/test_actuator_record_equivalence.mojo`).

    Every value here was already parsed by `full_parser` in phase 1a.1 — this
    is a transcription into the packed layout, not a second reading of the
    XML. The one exception is the fixed-tendon transmission, whose qpos/dof
    ADDRESSES are recovered from the joint list here by the same cumulative
    sum `_fill_actuator_transmission` uses for actuators; `TendonData` stores
    wraps as joint IDS because its other consumers want joints.

    Does NOT upload — the caller does, alongside `Model.upload_all`.
    """
    # ⚠ READ ONCE, FROM THE PROVIDER `SpecFields` NOW CARRIES (3d).
    # These were `D.NACT`/`D.NQ`/… — comptime, hence `DIM_POISON` on a
    # runtime-loaded model, hence a spurious raise from the nact check
    # and an empty reference pose from every loop below.
    var n_act = sf.dims.get_nact()
    var n_ten = sf.dims.get_nten()
    var n_q = sf.dims.get_nq()
    var n_v = sf.dims.get_nv()
    var n_key = sf.dims.get_nkey()
    var n_joint = sf.dims.get_njoint()
    var nact = len(fmd.actuators)
    if nact != n_act:
        raise Error(
            String(
                "physics3d: parser/dimension mismatch on ACTUATORS — declared",
                " nact=", n_act, ", full_parser found ", nact,
                ". Actuation is addressed BY ACTUATOR INDEX, so a mismatch",
                " here drives the wrong dof.",
            )
        )
    # ⚠ `>` NOT `!=`. `NTEN` is `max_tendon`, which every model def passes as a
    # ROW/RECORD budget and routinely over-declares (it is floored at 1 even
    # for a model with no tendons at all). Under-declaring is the failure that
    # matters and `init_fields` already raises on it via
    # `tendon_count_overflow`; this is the storage-side backstop.
    if len(fmd.tendons) > n_ten:
        raise Error(
            String(
                "physics3d: this model declares ", len(fmd.tendons),
                " tendons but max_tendon=", n_ten,
                ". Pass `max_tendon = <parse>.NTENDON`.",
            )
        )

    # Joint qpos/dof addresses: a cumulative sum over `fmd.joints` in XML
    # order — the same walk `_fill_actuator_transmission` does, and the same
    # order `_acd`'s joint table uses.
    var nj = len(fmd.joints)
    var qadr = List[Int](capacity=nj)
    var dadr = List[Int](capacity=nj)
    var q = 0
    var dd = 0
    for i in range(nj):
        qadr.append(q)
        dadr.append(dd)
        q += fmd.joints[i].nq
        dd += fmd.joints[i].nv

    # ── actuators ────────────────────────────────────────────────────────
    for i in range(nact):
        var a = fmd.actuators[i]
        var o = i * MODEL_ACTUATOR_SIZE
        sf.actuators.data[o + ACT_IDX_KIND] = Scalar[DTYPE](a.kind)
        sf.actuators.data[o + ACT_IDX_GEAR] = Scalar[DTYPE](a.gear)
        sf.actuators.data[o + ACT_IDX_CTRL_MIN] = Scalar[DTYPE](a.ctrl_min)
        sf.actuators.data[o + ACT_IDX_CTRL_MAX] = Scalar[DTYPE](a.ctrl_max)
        sf.actuators.data[o + ACT_IDX_CTRL_LIMITED] = Scalar[DTYPE](
            1 if a.is_ctrl_limited else 0
        )
        sf.actuators.data[o + ACT_IDX_FORCE_MIN] = Scalar[DTYPE](a.force_min)
        sf.actuators.data[o + ACT_IDX_FORCE_MAX] = Scalar[DTYPE](a.force_max)
        sf.actuators.data[o + ACT_IDX_FORCE_LIMITED] = Scalar[DTYPE](
            1 if a.force_limited else 0
        )
        sf.actuators.data[o + ACT_IDX_KP] = Scalar[DTYPE](a.kp)
        sf.actuators.data[o + ACT_IDX_KV] = Scalar[DTYPE](a.kv)
        # `biasprm[0]` / `biasprm[1]`. Written unconditionally, including
        # for a MOTOR whose biastype is NONE: the force law gates on `kind`,
        # and a slot only sometimes written is a slot whose zero cannot be
        # told from an absent attribute.
        sf.actuators.data[o + ACT_IDX_BIAS0] = Scalar[DTYPE](a.bias0)
        sf.actuators.data[o + ACT_IDX_BIAS1] = Scalar[DTYPE](a.bias1)
        # `mujoco.pid`. ⚠ `IMAX` AND `SLEW` CARRY -1 FOR "ABSENT", so these
        # must be written for EVERY actuator — a zeroed slot would read as
        # "clamp the integral to zero" on every non-plugin actuator the day
        # something else looked at them.
        # `<adhesion body=>` (`mjTRN_BODY`). Written for every actuator so
        # the -1 cannot be mistaken for an absent column.
        sf.actuators.data[o + ACT_IDX_BODY_ID] = Scalar[DTYPE](a.body_id)
        sf.actuators.data[o + ACT_IDX_PID_KI] = Scalar[DTYPE](a.pid_ki)
        sf.actuators.data[o + ACT_IDX_PID_KD] = Scalar[DTYPE](a.pid_kd)
        sf.actuators.data[o + ACT_IDX_PID_IMAX] = Scalar[DTYPE](a.pid_imax)
        sf.actuators.data[o + ACT_IDX_PID_SLEW] = Scalar[DTYPE](a.pid_slew)
        sf.actuators.data[o + ACT_IDX_DYN_TAU] = Scalar[DTYPE](a.dyn_tau)
        # ⚠ `dyn_type` IS NOT DERIVABLE FROM `dyn_tau` — see
        # `ACT_IDX_DYN_TYPE`. `integrator` has no tau and `filterexact`
        # shares `filter`'s.
        sf.actuators.data[o + ACT_IDX_DYN_TYPE] = Scalar[DTYPE](a.dyn_type)
        sf.actuators.data[o + ACT_IDX_ACT_LIMITED] = Scalar[DTYPE](
            1.0 if a.act_limited else 0.0
        )
        sf.actuators.data[o + ACT_IDX_ACT_MIN] = Scalar[DTYPE](a.act_min)
        sf.actuators.data[o + ACT_IDX_ACT_MAX] = Scalar[DTYPE](a.act_max)
        sf.actuators.data[o + ACT_IDX_ACT_ADR] = Scalar[DTYPE](a.act_adr)
        sf.actuators.data[o + ACT_IDX_TRN_N] = Scalar[DTYPE](a.trn_n)
        sf.actuators.data[o + ACT_IDX_DOF_ADR] = Scalar[DTYPE](a.dof_adr)
        sf.actuators.data[o + ACT_IDX_TENDON_ID] = Scalar[DTYPE](a.tendon_id)
        # `site=` (`mjTRN_SITE`) and the five gear components beyond
        # `gear[0]`. Meaningful only on a site transmission — see
        # `ACT_IDX_SITE_ID` — and written unconditionally so the slot's 0
        # cannot be mistaken for an absent attribute.
        sf.actuators.data[o + ACT_IDX_SITE_ID] = Scalar[DTYPE](a.site_id)
        sf.actuators.data[o + ACT_IDX_GEAR_1] = Scalar[DTYPE](a.gear1)
        sf.actuators.data[o + ACT_IDX_GEAR_2] = Scalar[DTYPE](a.gear2)
        sf.actuators.data[o + ACT_IDX_GEAR_3] = Scalar[DTYPE](a.gear3)
        sf.actuators.data[o + ACT_IDX_GEAR_4] = Scalar[DTYPE](a.gear4)
        sf.actuators.data[o + ACT_IDX_GEAR_5] = Scalar[DTYPE](a.gear5)
        sf.actuators.data[o + ACT_IDX_JOINT_ID] = Scalar[DTYPE](a.joint_id)
        # ⚠ ONLY THE FIRST `trn_n` TRIPLES ARE MEANINGFUL, and the rest keep
        # the -1 the constructor seeded. Copying all `TENDON_MAX_WRAPS` slots
        # would be harmless for qadr/dadr (still -1 in `fmd`) but would make a
        # future non-sentinel fill silently readable past `trn_n`.
        for k in range(a.trn_n):
            var fb = i * TENDON_MAX_WRAPS + k
            sf.actuators.data[o + ACT_IDX_TRN_QADR_0 + k] = Scalar[DTYPE](
                fmd.motor_trn_qadr[fb]
            )
            sf.actuators.data[o + ACT_IDX_TRN_DADR_0 + k] = Scalar[DTYPE](
                fmd.motor_trn_dadr[fb]
            )
            sf.actuators.data[o + ACT_IDX_TRN_COEF_0 + k] = Scalar[DTYPE](
                fmd.motor_trn_coef[fb]
            )

    # ── fixed-tendon springs ─────────────────────────────────────────────
    for t in range(len(fmd.tendons)):
        var td = fmd.tendons[t]
        var o = t * MODEL_ACT_TENDON_SIZE
        sf.act_tendons.data[o + ACTTEN_IDX_STIFFNESS] = Scalar[DTYPE](
            td.stiffness
        )
        sf.act_tendons.data[o + ACTTEN_IDX_DAMPING] = Scalar[DTYPE](
            td.damping
        )
        sf.act_tendons.data[o + ACTTEN_IDX_SPRING_LO] = Scalar[DTYPE](
            td.spring_lo
        )
        sf.act_tendons.data[o + ACTTEN_IDX_SPRING_HI] = Scalar[DTYPE](
            td.spring_hi
        )
        var n = td.num_joints
        if n > TENDON_MAX_WRAPS:
            n = TENDON_MAX_WRAPS
        sf.act_tendons.data[o + ACTTEN_IDX_TRN_N] = Scalar[DTYPE](n)
        for k in range(n):
            var jid = td.joint_ids[k]
            if jid >= 0 and jid < nj:
                sf.act_tendons.data[
                    o + ACTTEN_IDX_TRN_QADR_0 + k
                ] = Scalar[DTYPE](qadr[jid])
                sf.act_tendons.data[
                    o + ACTTEN_IDX_TRN_DADR_0 + k
                ] = Scalar[DTYPE](dadr[jid])
            sf.act_tendons.data[o + ACTTEN_IDX_TRN_COEF_0 + k] = Scalar[DTYPE](
                td.coefs[k]
            )

    # ── reference pose (`mj_resetData`) ──────────────────────────────────
    #
    # ⚠ `qpos0_nq == 0` MEANS "NO POSE WAS PARSED", NOT "the pose is zero",
    # and the two need different resets. `reset_data` zeroes qpos and then
    # sets the free-joint quaternion to identity in that case; writing zeros
    # here and letting it read them would drop the `qw = 1` and leave a
    # degenerate quaternion for FK. The flag rides in `pose_meta`.
    sf.pose_meta.data[POSE_IDX_QPOS0_NQ] = Scalar[DTYPE](fmd.qpos0_nq)
    sf.pose_meta.data[POSE_IDX_FREE_JOINT_QPOS_ADR] = Scalar[DTYPE](
        fmd.free_joint_qpos_adr
    )
    for i in range(fmd.qpos0_nq):
        if i < n_q and i < len(fmd.qpos0):
            sf.qpos0.data[i] = Scalar[DTYPE](fmd.qpos0[i])

    # ── keyframes (`mj_resetDataKeyframe`) ───────────────────────────────
    if fmd.nkey > n_key:
        raise Error(
            String(
                "physics3d: this model declares ", fmd.nkey,
                " <keyframe><key> entries but nkey=", n_key,
                " was passed to the model def. Truncating is not safe: a",
                " caller asking for the dropped key would silently reset to",
                " qpos0 instead.",
            )
        )
    for k in range(fmd.nkey):
        var o = k * KEY_META_SIZE
        sf.key_meta.data[o + KEY_IDX_TIME] = Scalar[DTYPE](fmd.key_time[k])
        # ⚠ THE LENGTHS ARE PRESENCE FLAGS. MuJoCo fills an absent `qpos=`
        # from qpos0 and an absent `qvel=`/`ctrl=` with zero, so a reader has
        # to distinguish "attribute missing" from "attribute of zeros".
        sf.key_meta.data[o + KEY_IDX_NQPOS] = Scalar[DTYPE](fmd.key_nqpos[k])
        sf.key_meta.data[o + KEY_IDX_NQVEL] = Scalar[DTYPE](fmd.key_nqvel[k])
        sf.key_meta.data[o + KEY_IDX_NCTRL] = Scalar[DTYPE](fmd.key_nctrl[k])
        for i in range(n_q):
            if k * n_q + i < len(fmd.key_qpos):
                sf.key_qpos.data[k * n_q + i] = Scalar[DTYPE](
                    fmd.key_qpos[k * n_q + i]
                )
        for i in range(n_v):
            if k * n_q + i < len(fmd.key_qvel):
                # ⚠ STRIDE `NQ`, NOT `NV`. `FlatModelDef.key_qvel` mirrors the
                # comptime twin, whose row stride is `NQ0` for BOTH qpos and
                # qvel (`key_qvel[k * NQ0 + i]`) — one allocation shape for
                # two arrays. The tensor is honestly `[NKEY, NV]`, so the two
                # strides differ and reading them the same way walks into the
                # next key on any model with nq != nv.
                sf.key_qvel.data[k * n_v + i] = Scalar[DTYPE](
                    fmd.key_qvel[k * n_q + i]
                )
        for i in range(n_act):
            if k * n_act + i < len(fmd.key_ctrl):
                sf.key_ctrl.data[k * n_act + i] = Scalar[DTYPE](
                    fmd.key_ctrl[k * n_act + i]
                )

    # ── joint limits (the `enforce_limits` clamp) ────────────────────────
    #
    # ⚠ `qadr` HERE IS THE SAME CUMULATIVE SUM the transmission used, not
    # `JointData` — the record carries per-joint `nq`/`nv`, never an absolute
    # address.
    for j in range(nj):
        if j >= n_joint:
            break
        var o = j * JLIM_SIZE
        sf.joint_limits.data[o + JLIM_IDX_LIMITED] = Scalar[DTYPE](
            1 if fmd.joints[j].is_limited else 0
        )
        sf.joint_limits.data[o + JLIM_IDX_QPOS_ADR] = Scalar[DTYPE](qadr[j])
        sf.joint_limits.data[o + JLIM_IDX_RANGE_MIN] = Scalar[DTYPE](
            fmd.joints[j].range_min
        )
        sf.joint_limits.data[o + JLIM_IDX_RANGE_MAX] = Scalar[DTYPE](
            fmd.joints[j].range_max
        )
        # ── `<joint actuatorfrcrange>` (MuJoCo's `jnt_actfrcrange`) ──────
        #
        # ⚠ THE DOF ADDRESS, NOT THE QPOS ONE. `mj_fwdActuation` indexes the
        # clamp with `m->jnt_dofadr`, and the two differ on every model with
        # a free or ball joint — g1's are 7 and 6, so a qpos-indexed clamp
        # would land one dof past every joint in the tree.
        sf.joint_limits.data[o + JLIM_IDX_DOF_ADR] = Scalar[DTYPE](dadr[j])
        sf.joint_limits.data[o + JLIM_IDX_ACTFRC_LIMITED] = Scalar[DTYPE](
            1 if fmd.joints[j].is_actfrc_limited else 0
        )
        sf.joint_limits.data[o + JLIM_IDX_ACTFRC_MIN] = Scalar[DTYPE](
            fmd.joints[j].actfrc_min
        )
        sf.joint_limits.data[o + JLIM_IDX_ACTFRC_MAX] = Scalar[DTYPE](
            fmd.joints[j].actfrc_max
        )

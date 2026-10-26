"""ModelDefFromXML — ModelDefLike implementation from an embedded MJCF XML string.

Enables zero-boilerplate physics environments from XML:

    comptime pm = parse_xml(my_xml)
    comptime XmlModel = ModelDefFromXML[
        my_xml,
        pm.NBODY, pm.NJOINT, pm.NQ, pm.NV, pm.NGEOM, pm.NACT,
        max_contacts=50,
    ]
    var env = Phyics3dEnv[XmlModel, MyConfig]()

CPU path:  parse_xml_full() → FlatModelDef.setup_model() + FK + invweight0.
GPU path:  CPU Model → HostBuffer → DeviceBuffer → _compute_invweight0_gpu().
GPU kernels: comptimefor loops over comptime scalar helpers.
Rendering: no-op stubs (XML models have no visual configuration yet).

Note: Mojo nightly requires struct parameters to be accessed as 'Self.param'
inside the struct body. All dimension parameters follow this convention.
"""

from std.collections import Array

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from max.gpu import thread_idx, block_idx, block_dim
from std.random.philox import Random as PhiloxRandom
from noeira.render import Color, Renderer3D, Light, LightMode, Camera3D
from noeira.math3d import Vec3 as _Vec3G, Quat as _QuatG

from noeira.physics3d.types import ConeType
from noeira.physics3d.joint_types import (
    JNT_FREE,
    JNT_BALL,
    JNT_HINGE,
    JNT_SLIDE,
)
from noeira.physics3d.gpu.constants import (
    TPB,
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_DAMPING,
    JOINT_IDX_STIFFNESS,
    MODEL_ACTUATOR_SIZE,
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
    ACT_IDX_TRN_N,
    ACT_IDX_TRN_QADR_0,
    ACT_IDX_TRN_DADR_0,
    ACT_IDX_TRN_COEF_0,
    MODEL_ACT_TENDON_SIZE,
    ACTTEN_IDX_STIFFNESS,
    ACTTEN_IDX_SPRING_LO,
    ACTTEN_IDX_SPRING_HI,
    ACTTEN_IDX_TRN_N,
    ACTTEN_IDX_TRN_QADR_0,
    ACTTEN_IDX_TRN_DADR_0,
    ACTTEN_IDX_TRN_COEF_0,
    POSE_IDX_QPOS0_NQ,
    POSE_IDX_FREE_JOINT_QPOS_ADR,
    KEY_META_SIZE,
    KEY_IDX_TIME,
    KEY_IDX_NQPOS,
    KEY_IDX_NQVEL,
    KEY_IDX_NCTRL,
    POSE_META_SIZE,
    JLIM_SIZE,
    JLIM_IDX_LIMITED,
    JLIM_IDX_QPOS_ADR,
    JLIM_IDX_RANGE_MIN,
    JLIM_IDX_RANGE_MAX,
    JLIM_IDX_DOF_ADR,
    JLIM_IDX_ACTFRC_LIMITED,
    JLIM_IDX_ACTFRC_MIN,
    JLIM_IDX_ACTFRC_MAX,
    METADATA_SIZE,
    META_IDX_ACTDAMP_LIVE,
)
from noeira.physics3d.joint_types import JNT_FREE, JNT_BALL
from noeira.physics3d.fields import Model, Data, DynamicsScratch, SpecFields, Dims, DimsLike
from noeira.physics3d.dynamics.actuation import apply_actions_fields
from noeira.physics3d.dynamics.activation import next_activation
from noeira.physics3d.dynamics.invweight import (
    compute_invweight0,
)
from noeira.physics3d.model.model_def import ModelDefLike
from .fields_build import (
    build_model_fields_from_flat,
    build_spec_fields,
    assert_no_pending_dampratio,
    apply_auto_spring_damper,
)
from .flat_model import (
    TEX_CUBE,
    ACT_KIND_MOTOR,
    ACT_KIND_POSITION,
    ACT_KIND_VELOCITY,
    ACT_KIND_GENERAL,
    act_kind_name,
    _GEOM_ELLIPSOID,
    TEX_SKYBOX,
    TEX_BUILTIN_GRADIENT,
    TEX_BUILTIN_CHECKER,
)
from .full_parser import parse_xml_full
from .expander import expand_mjcf
from .render_fields import (
    RenderFields,
    body_geom_visible,
    build_render_fields,
)
from noeira.physics3d.model.model_dims import ModelDims
from .xml_parser import (
    MAX_COMPTIME_TENDONS,
    MAX_COMPTIME_TENDON_WRAPS,
    MAX_COMPTIME_ACTUATORS,
    MAX_COMPTIME_JOINTS,
    MAX_COMPTIME_NQ,
    _xml_nth_motor_gear,
    _xml_nth_motor_dof_adr,
    _xml_nth_joint_qpos_adr,
    _xml_nth_joint_limited,
    _xml_nth_joint_range_min,
    _xml_nth_joint_range_max,
    _xml_fixed_tendon_njoints,
    _xml_fixed_tendon_joint_name,
    _xml_fixed_tendon_coef,
    _xml_find_joint_dof_adr,
    _xml_find_joint_index,
)
from noeira.core.tiles import Tile

# Type aliases matching model_def.mojo module scope (required for trait conformance)
comptime _RVec3 = _Vec3G[DType.float64]
comptime _RQuat = _QuatG[DType.float64]


def _attr_between(src: String, lo: Int, hi: Int, attr: String) -> String:
    """`attr="value"` inside `src[lo:hi]`, or "".

    A local reader so `body_names` does not depend on the comptime parser's
    helpers — this one runs at RUNTIME and has no interpreter constraints to
    respect.
    """
    var needle = attr + '="'
    var nlen = needle.byte_length()
    var p = src.find(needle, lo)
    while p != -1 and p < hi:
        var ok = p == 0
        if not ok:
            var c = src.as_bytes()[p - 1]
            ok = c == 32 or c == 9 or c == 10 or c == 13
        if ok:
            var vs = p + nlen
            var ve = src.find('"', vs)
            if ve != -1 and ve <= hi:
                return String(src[byte=vs:ve])
        p = src.find(needle, p + 1)
    return String("")



# ⚠ MODULE LEVEL, NOT A MEMBER, AND THAT IS THE POINT. It is pure quaternion
# math — nothing about it depends on the model — and `setup_cameras` was the
# last thing keeping the render hooks reachable only through a comptime type.
# See `scripts/audit_render_hooks_are_rf_pure.py`.
def _rcd_rotate_by_quat(
    qx: Float64, qy: Float64, qz: Float64, qw: Float64,
    vx: Float64, vy: Float64, vz: Float64,
) -> List[Float64]:
    """Rotate (vx,vy,vz) by the quaternion, returned as [x, y, z]."""
    var tx = 2.0 * (qy * vz - qz * vy)
    var ty = 2.0 * (qz * vx - qx * vz)
    var tz = 2.0 * (qx * vy - qy * vx)
    var out = List[Float64]()
    out.append(vx + qw * tx + qy * tz - qz * ty)
    out.append(vy + qw * ty + qz * tx - qx * tz)
    out.append(vz + qw * tz + qx * ty - qy * tx)
    return out^


def _resolve_asset(base: String, path: String) -> String:
    """Join a relative asset path onto `base`. Absolute paths escape.

    ⚠ MODULE LEVEL because `render_skin` needs it and `render_skin` must be a
    pure function of `rf` — see `ModelDefFromXML.resolve_asset`, which is now
    this with `base` bound to `Self.asset_base_dir()`. The reason the skin
    chain needs resolution at all is that it is walked by string search over
    the raw MJCF, so it bypasses what `parse_xml_full` does for
    `mesh_asset_files` and `TextureData.file`.
    """
    if path.byte_length() == 0 or path.startswith("/"):
        return path
    if base.byte_length() == 0:
        return path
    return base + "/" + path



def body_names_of(src: String) raises -> List[String]:
    """Body names by index (0 = worldbody), parsed from the model XML AT
    RUNTIME.

    ⚠ RUNTIME ON PURPOSE, and it is not a style choice. A `<skin>`'s bones
    name the bodies they follow and those names live inside the BINARY
    `.skn`, so the match cannot happen at compile time — which leaves the
    names needing to survive into runtime somehow. Writing them into the
    comptime render data does not compile, in every spelling tried (see
    `xml_parser.mojo`, above `MAX_COMPTIME_RENDER_GEOMS`). The XML is
    already a comptime parameter of this struct, so materializing it once
    and scanning it with ORDINARY runtime string code costs nothing extra
    and is subject to none of those restrictions.

    Document order, which is the order the parser assigns body ids: both
    walk `<body` opening tags left to right.

    ⚠ CALL IT ONCE. It rescans the whole model XML;
    `render_skin` does so only on the frame that loads the skin.
    """
    
    var out = List[String]()
    out.append(String(""))  # index 0 = worldbody, which has no <body> tag

    var wb = src.find("<worldbody")
    if wb == -1:
        return out^
    var stop = src.find("</worldbody>", wb)
    if stop == -1:
        stop = src.byte_length()

    var pos = src.find(">", wb)
    if pos == -1:
        return out^
    pos += 1
    while True:
        var bt = src.find("<body", pos)
        if bt == -1 or bt >= stop:
            break
        var bte = src.find(">", bt)
        if bte == -1:
            break
        out.append(_attr_between(src, bt, bte, "name"))
        pos = bte + 1
    return out^


@fieldwise_init
struct ModelDefFromXML[
    # ⚠ THE MJCF USED TO BE HERE, as `xml: String` — ~1.1 MB of it across 48
    # source files. Phase 1b moved every reader to runtime and the model to a
    # file; `xml_path` below is what remains. See §10.2: the comptime
    # interpreter cannot `open()`, so anything reading the MJCF at compile
    # time pinned it to a string literal in Mojo source.
    nbody: Int,
    njoint: Int,
    nq: Int,
    nv: Int,
    ngeom: Int,
    nact: Int,
    ntex: Int = 0,
    nmat: Int = 0,
    nlight: Int = 0,
    ncam: Int = 0,
    max_contacts: Int = 50,
    max_equality: Int = 0,
    cone_type: Int = ConeType.PYRAMIDAL,
    max_tendon: Int = 0,
    nsite: Int = 0,
    neq: Int = 0,
    nexclude: Int = 0,
    npair: Int = 0,
    obs_qpos_skip: Int = 1,
    obs_dim_override: Int = -1,
    action_dim_override: Int = -1,
    timestep: Float64 = 0.01,
    allow_unsupported_actuators: Bool = False,
    max_condim: Int = 3,
    noslip_iter: Int = 0,
    allow_missing_noslip: Bool = False,
    # ⚠⚠ APPENDED, AND THEY MUST STAY LAST. Every parameter here is an `Int`
    # or `Bool`, so inserting one mid-list silently shifts every positional
    # instantiation — the trap `NPAIR` already documents on `fields.Model`.
    #
    # MuJoCo's `m->na` and `m->nkey`. These used to be READ OFF `_acd`
    # (`comptime NA = Self._acd.na`), which made them expressions the compiler
    # could not unify against the trait's symbolic `Self.NA` once they entered
    # a signature — `SpecFields[..., Self.NKEY]` on `ModelDefLike.reset_data`
    # failed to typecheck with `parse_xml_model_data(xml).nkey` on the
    # implementation side. That is what forced them to become PARAMETERS, and
    # it is also what finally lets `_acd` go.
    #
    # `parse_xml` does not compute either one, so they are hand-supplied and
    # `init_fields` asserts them against `FlatModelDef`. The blast radius is
    # small and measured: `na > 0` only for quadruped (12), dog (38) and
    # dog_fetch; `nkey > 0` only for so_arm100 (2).
    na: Int = 0,
    nkey: Int = 0,
    # ⚠ PHASE 1b. The model's MJCF, as a FILE — the whole point of the phase.
    # The runtime parse reads this path, and asset paths inside the model
    # resolve against its DIRECTORY, which is MuJoCo's own rule
    # (§10.5 decision 1). EVERY SHIPPED MODEL USES THIS.
    #
    # Appended for the reason the block above gives: inserting mid-list
    # silently shifts every positional instantiation.
    xml_path: String = "",
    # ⚠ INLINE MJCF — FOR FIXTURES, NOT FOR MODELS. Phase 1b moved the 57
    # shipped models to `.xml` files and deleted ~1.1 MB of embedded MJCF;
    # this is what remains for the ~78 places that build a MODEL OUT OF A
    # DOZEN LINES TO TEST ONE RULE. Those are not models — they are fixtures,
    # they have no assets, and turning each into a file on disk would trade
    # readable tests for filesystem noise.
    #
    # ⚠ Assets inside an inline model resolve against the PROCESS CWD, since
    # there is no model file to be relative to. Fixtures citing
    # `references/...` meshes depend on that.
    #
    # Exactly one of `xml_path` / `xml` must be set; `xml_text()` raises if
    # neither is.
    xml: String = "",
    # ⚠⚠ APPENDED AFTER `xml`, WHICH THE DOCSTRING BELOW CALLS THE LAST TWO
    # PARAMETERS. That sentence is now stale by two, and these still go at the
    # END for the reason the `na`/`nkey` block gives: inserting mid-list
    # silently shifts every positional instantiation. They are keyword-only in
    # practice — nothing spells 30 parameters positionally.
    #
    # MuJoCo's `m->nsensor` and `m->nsensordata`. Like `na` and `nkey`, they
    # are HAND-SUPPLIED and checked against the parse: `init_fields` raises if
    # the model declares more sensors than were declared here, so an
    # under-declared count fails loudly instead of truncating `sensordata`.
    #
    # ⚠ DEFAULT 0 IS THE HONEST DEFAULT, not a placeholder. A model whose
    # sensors this engine does not serve still parses; leaving these at 0 just
    # means no `sensordata` is allocated for it, which is exactly right for
    # the 50-odd fixtures and models with no `<sensor>` block at all.
    nsensor: Int = 0,
    nsensordata: Int = 0,
](ModelDefLike):
    """`ModelDefLike` driven entirely from an MJCF document.

    The document is a FILE (`xml_path`) for every shipped model since phase
    1b; `xml` carries inline MJCF and is for fixtures. Exactly one of the two
    must be set.

    All physics dimensions must be provided; obtain them from `parse_xml()`.
    ⚠ NAME THEM — `xml_path` and `xml` are the LAST two parameters (28 and
    29), so the positional form no longer starts with the document:

        comptime pm = parse_xml(xml)
        comptime XmlModel = ModelDefFromXML[
            xml_path="path/to/model.xml",
            nbody=pm.NBODY, njoint=pm.NJOINT, nq=pm.NQ, nv=pm.NV,
            ngeom=pm.NGEOM, nact=pm.NACT, ntex=pm.NTEX, nmat=pm.NMAT,
            nlight=pm.NLIGHT, ncam=pm.NCAM,
        ]

    Parameters:
        nbody:         Total body count including worldbody.
        njoint:        Total joint count.
        nq:            Total position DOF.
        nv:            Total velocity DOF.
        ngeom:         Total geometry count.
        nact:          Total actuator count.
        ntex:          Texture count from <asset> (default 0).
        nmat:          Material count from <asset> (default 0).
        nlight:        Light count in <worldbody> (default 0).
        ncam:          Camera count in <worldbody> (default 0).
        max_contacts:  Maximum contacts per step (default 50).
        max_equality:  Maximum equality constraints (default 0).
        cone_type:     Friction cone type (default PYRAMIDAL).
        max_tendon:    Maximum fixed tendons (default 0).
        nsite:   Total site count (default 0).
        neq:           Number of equality constraints (default 0).
        nexclude:      Number of contact exclusion pairs (default 0).
        npair:         Number of `<contact><pair>` records (default 0). This is
            the ONLY knob — it both counts and sizes the slab. Contrast
            `neq`/`max_equality`, where passing the count alone leaves storage
            at zero and the constraint silently vanishes.
        obs_qpos_skip: Leading qpos DOF to exclude from obs (default 1).
        obs_dim_override: Override OBS_DIM (default -1 = compute from nq-skip+nv).
            Use when custom_extract_obs_gpu produces different dimensionality than
            the default formula (e.g. InvertedDoublePendulum needs OBS_DIM=9 with
            sin/cos transforms despite nq-skip+nv=6).
        action_dim_override: Override ACTION_DIM (default -1 = use nact).
        timestep:      Simulation timestep (default 0.01).
        allow_unsupported_actuators: Build the model even when the XML declares
            <position>/<velocity>/<general> actuators. Those are servos the
            engine cannot simulate (no gainprm/biasprm — see gap G3 in
            docs/DM_CONTROL_PORT.md), so `init_fields` rejects them by default.
            Set True ONLY when the env's CONFIG bypasses `apply_actions`
            entirely (returns True from `custom_apply_actions_cpu`) and drives
            those DOFs itself — SawyerReach does exactly this for its two
            kp=400 gripper servos.
        max_condim:    Largest condim over the model's geoms; pass
                       `parse_xml(xml).MAX_CONDIM`. Over-estimating only
                       costs unused rows, under-estimating is silent.
        noslip_iter:   `<option noslip_iterations>`; pass
                       `parse_xml(xml).NOSLIP_ITER`. 0 disables the pass,
                       which is MuJoCo's default and correct for every suite
                       model except dog.
        allow_missing_noslip: DEPRECATED, and a no-op since 2026-08-13. It used
            to be the opt-in for `noslip_iter > 0` on an ELLIPTIC-cone model,
            back when `mj_solNoSlip` existed for the pyramidal cone only and
            the elliptic path skipped the pass in silence. Both branches are
            implemented now (`solver/noslip.mojo`) and the solver dispatches on
            the cone, so there is nothing to permit. Kept as an accepted
            parameter so existing model defs still compile; new code must not
            pass it.
        na:            MuJoCo's `m->na`, the actuator-activation count. Hand
            supplied — `parse_xml` does not compute it — and asserted against
            `FlatModelDef` by `init_fields`. Non-zero only for quadruped (12),
            dog (38) and dog_fetch.
        nkey:          MuJoCo's `m->nkey`, the `<keyframe>` count. Hand
            supplied on the same terms as `na`; non-zero only for so_arm100
            (2).
        xml_path:      Path to the model's `.xml`. Asset paths inside it
            resolve against its DIRECTORY, which is MuJoCo's own rule. Every
            shipped model uses this.
        xml:           Inline MJCF, for FIXTURES (must be comptime-known).
            Assets inside one resolve against the process CWD, since there is
            no model file to be relative to.
        nsensor:       MuJoCo's `m->nsensor`. Hand supplied on the same terms
            as `na`; `init_fields` raises if the model declares more (default 0).
        nsensordata:   MuJoCo's `m->nsensordata`, the `sensordata` length
            (default 0 allocates none).
    """

    # === Dimensions required by ModelDefLike ===
    comptime NBODY: Int = Self.nbody
    comptime NJOINT: Int = Self.njoint
    comptime NQ: Int = Self.nq
    comptime NV: Int = Self.nv
    comptime NGEOM: Int = Self.ngeom
    comptime MAX_EQUALITY: Int = Self.max_equality
    comptime CONE_TYPE: Int = Self.cone_type
    # Largest condim in the model — sizes the PYRAMIDAL edge list at
    # 2*(MAX_CONDIM-1) rows per contact and the ELLIPTIC tangential block at
    # MAX_CONDIM-1. Pass `parse_xml(xml).MAX_CONDIM`.
    #
    # ⚠ LEAVING IT AT 3 ON A MODEL WITH condim 4/6 GEOMS SILENTLY DROPS THEIR
    # TORSIONAL AND ROLLING ROWS. `_precompute_contact_friction` clamps each
    # contact's own `condim` to this, so the rows are not merely unread — they
    # are never built, the contact solves as if it were condim 3, and nothing
    # reports it. Measured on a spinning ball
    # (`tests/physics3d/test_elliptic_condim46_vs_mujoco.mojo`): dropping the
    # torsional row moves `qacc` by 1.6e+3 against a `|qacc|` of 3e+3, and the
    # rolling pair by another 1.1e+3.
    comptime MAX_CONDIM: Int = Self.max_condim
    # `<option noslip_iterations>`. Runs MuJoCo's `mj_solNoSlip` after the
    # primal solve: a friction-only Gauss-Seidel sweep with the normal forces
    # held fixed. 0 = off, which is MuJoCo's default. dm_control's dog is the
    # only in-scope model that sets it (to 4), and it is first-order there —
    # 2.9e-2 of qvel on the first contacting step.
    comptime NOSLIP_ITER: Int = Self.noslip_iter
    comptime MAX_CONTACTS: Int = Self.max_contacts
    comptime MAX_TENDON: Int = Self.max_tendon
    comptime NSITE: Int = Self.nsite
    # MuJoCo `m->na` — activation variables, NOT `nu`. Nonzero only for
    # actuators with a `dyntype`; 0 for every <motor>/<position>.
    comptime NA: Int = Self.na
    # Floored at 1 so a model with no activation still has a bindable `act`
    # tensor — a zero-extent operand segfaults.
    comptime NA_F: Int = Self.NA if Self.NA > 0 else 1
    comptime NEXCLUDE: Int = Self.nexclude
    comptime NPAIR: Int = Self.npair
    comptime OBS_DIM: Int = Self.obs_dim_override if Self.obs_dim_override > 0 else (
        Self.nq - Self.obs_qpos_skip + Self.nv
    )
    comptime ACTION_DIM: Int = Self.action_dim_override if Self.action_dim_override > 0 else Self.nact
    comptime TIMESTEP: Float64 = Self.timestep

    # ⚠⚠ `_acd` (`ComptimeActData`) LIVED HERE AND IS GONE (phase 1a.4e).
    # It was the model's XML interpreted at struct-elaboration time into ~20
    # `Array`s — every actuator value the engine used, the reference
    # pose, the keyframes and the joint limit tables. All of that is
    # `SpecFields` now, built at RUNTIME by `build_spec_fields` from
    # `FlatModelDef`. The `_NACT` / `_NJNT` / `_NQ0` / `_NTEN` / `_WRAPS`
    # sizing helpers went with it; `NACT_F` / `NTEN_F` / `NQ_F` below are what
    # size the records.
    #
    # ⚠ `na` and `nkey` are PARAMETERS because of this: they were
    # `Self._acd.na` / `.nkey`, and a member read off an interpreted struct
    # cannot appear in a trait signature (see the `na`/`nkey` parameter note).

    # Actuation record capacities (phase 1a.2), declared on `ModelDefLike` so
    # the trait and this implementation spell `SpecFields` and the kernel
    # operand layouts with the SAME parameter names. A signature that says
    # `Self.nact` where the trait says `Self.NACT` leaves the compiler
    # comparing two unmaterialized expressions it will not unify — the same
    # trap `test_fullinertia_vs_mujoco` documents for `NPAIR`.
    #
    # ⚠ `NACT` IS UNFLOORED AND `NACT_F` IS NOT. `build_spec_fields` checks
    # `len(fmd.actuators) == NACT`, so a model with no actuators must pass 0;
    # the STORAGE is floored at 1 because a zero-extent operand aborts at
    # bind. Both numbers are needed and they are not interchangeable.
    comptime NACT: Int = Self.nact
    comptime NACT_F: Int = Self.nact if Self.nact > 0 else 1
    comptime NTEN_F: Int = (
        Self.max_tendon if Self.max_tendon > 0 else 1
    )
    # ⚠ UNFLOORED, like `NACT`: `build_spec_fields` checks the real count
    # (`fmd.nkey > NKEY` raises), while the STORAGE floors at 1 inside
    # `SpecFields`. In 1a.4 this still reads `_acd`; it becomes a comptime
    # PARAMETER in the same phase, which is what finally lets `_acd` go.
    comptime NKEY: Int = Self.nkey
    """Number of `<keyframe><key>` entries, in XML order (MuJoCo's `nkey`)."""
    comptime NSENSOR: Int = Self.nsensor
    """`<sensor>` element count (MuJoCo's `nsensor`) — served or merely
    addressed; both hold a row."""
    comptime NSENSORDATA: Int = Self.nsensordata
    """`sensordata` length (MuJoCo's `nsensordata`) — `sum(sensor_dim)`, NOT
    `NSENSOR`."""
    comptime NQ_F: Int = Self.nq if Self.nq > 0 else 1

    # Precomputed rendering data — evaluated once at struct level.
    # Replaces 11 separate parse_xml_full calls that crashed the comptime
    # interpreter for large (25+ body) models.

    # =========================================================================
    # CPU: state hooks (fields-native; G2). The legacy CPU model build
    # (setup_model_and_data + _reset_data_legacy) was deleted at G4 — the
    # model build is `init_fields` (spec-direct) below.
    # =========================================================================

    @staticmethod
    def reset_data[DTYPE: DType, D: DimsLike, D2: DimsLike](
        sf: SpecFields[DTYPE, D],
        mut d: Data[DTYPE, D2, 1],
    ):
        """Reset qpos to initial pose, zero qvel/qacc/qfrc.

        `qpos0` is MuJoCo's: joint `ref`s and each free joint's body pose.
        (A `<custom><numeric name="init_qpos">` is NOT applied — MuJoCo does
        not read it either.) If no pose was parsed, qpos is zeroed and the
        free joint quaternion (if any) is set to identity (qw=1) so that FK
        does not degenerate.
        """
        # ⚠ `qpos0_nq == 0` MEANS "NO POSE WAS PARSED", NOT "the pose is
        # zero" — the two want different resets, and the second branch below
        # is what supplies the free joint's `qw = 1`. This used to be a
        # `comptime if` on `_acd.nq`; the branch is now a runtime test on a
        # record, which is the whole point of the phase.
        var nq0 = Int(sf.pose_meta.data[POSE_IDX_QPOS0_NQ])
        if nq0 > 0:
            # Apply qpos0 (joint refs, free-joint body poses).
            for i in range(Self.NQ):
                if i < nq0:
                    d.qpos.data[i] = sf.qpos0.data[i]
                else:
                    d.qpos.data[i] = Scalar[DTYPE](0)
        else:
            # No pose parsed — zero everything, then fix free-joint quaternion.
            for i in range(Self.NQ):
                d.qpos.data[i] = Scalar[DTYPE](0)
            var fj = Int(sf.pose_meta.data[POSE_IDX_FREE_JOINT_QPOS_ADR])
            if fj >= 0 and fj + 3 < Self.NQ:
                # qpos[adr+3] is qw for a free joint (MuJoCo convention:
                # [tx, ty, tz, qw, qx, qy, qz]).  Set qw=1 for identity.
                d.qpos.data[fj + 3] = Scalar[DTYPE](1)
        for i in range(Self.NV):
            d.qvel.data[i] = Scalar[DTYPE](0)
            d.qacc.data[i] = Scalar[DTYPE](0)
            d.qfrc.data[i] = Scalar[DTYPE](0)
            # ⚠ `mj_resetData` ZEROES `qacc_warmstart` TOO. The primal solvers
            # start from the cheaper of it and `qacc_smooth`, so leaving the
            # last episode's acceleration here would price a candidate MuJoCo
            # never sees. The cost comparison would usually discard it —
            # "usually" is not the reference's algorithm.
            d.qacc_warmstart.data[i] = Scalar[DTYPE](0)


    # ⚠ THE ROW STRIDES ARE `Self.NQ` / `Self.NV` / `Self.NACT` — the tensors'
    # own shapes. `FlatModelDef.key_qvel` strides by NQ for BOTH key arrays
    # (one allocation shape), so the two differ on any model with nq != nv;
    # `test_pose_key_stride` is the fixture that can see it, and nothing in
    # the model tree can.
    # ⚠ THE ROW STRIDES ARE `Self.NQ` / `Self.NV` / `Self.NACT` — the tensor's
    # own shapes, NOT `_acd`'s. The comptime side strides `key_qvel` by `NQ0`
    # as well (one allocation shape for both key arrays), so the two differ on
    # any model with nq != nv. `test_pose_key_stride` is the fixture that can
    # see it; nothing in the model tree can.
    @staticmethod
    def key_qpos_at[DTYPE: DType, D: DimsLike](sf: SpecFields[DTYPE, D], k: Int, i: Int) -> Float64:
        """`mjModel.key_qpos[k][i]`, falling back to qpos0 when qpos is absent.
        """
        if k < 0 or k >= Self.nkey or i < 0 or i >= Self.NQ:
            return 0.0
        if sf.key_meta.data[k * KEY_META_SIZE + KEY_IDX_NQPOS] == 0:
            return Float64(sf.qpos0.data[i])
        return Float64(sf.key_qpos.data[k * Self.NQ + i])

    @staticmethod
    def key_qvel_at[DTYPE: DType, D: DimsLike](sf: SpecFields[DTYPE, D], k: Int, i: Int) -> Float64:
        """`mjModel.key_qvel[k][i]` — zero when absent, as MuJoCo fills it."""
        if k < 0 or k >= Self.nkey or i < 0 or i >= Self.NV:
            return 0.0
        if sf.key_meta.data[k * KEY_META_SIZE + KEY_IDX_NQVEL] == 0:
            return 0.0
        return Float64(sf.key_qvel.data[k * Self.NV + i])

    @staticmethod
    def key_ctrl_at[DTYPE: DType, D: DimsLike](sf: SpecFields[DTYPE, D], k: Int, i: Int) -> Float64:
        """`mjModel.key_ctrl[k][i]` — zero when absent, as MuJoCo fills it."""
        if k < 0 or k >= Self.nkey or i < 0 or i >= Self.NACT:
            return 0.0
        if sf.key_meta.data[k * KEY_META_SIZE + KEY_IDX_NCTRL] == 0:
            return 0.0
        return Float64(sf.key_ctrl.data[k * Self.NACT + i])

    @staticmethod
    def key_time_at[DTYPE: DType, D: DimsLike](sf: SpecFields[DTYPE, D], k: Int) -> Float64:
        """`mjModel.key_time[k]`."""
        if k < 0 or k >= Self.nkey:
            return 0.0
        return Float64(sf.key_meta.data[k * KEY_META_SIZE + KEY_IDX_TIME])

    @staticmethod
    def reset_data_keyframe[DTYPE: DType, D: DimsLike, D2: DimsLike](
        sf: SpecFields[DTYPE, D],
        mut d: Data[DTYPE, D2, 1],
        k: Int,
    ):
        """`mj_resetDataKeyframe(m, d, k)` — reset to keyframe `k`.

        ⚠⚠ THIS IS DELIBERATELY SEPARATE FROM `reset_data`, AND CALLING IT IS
        THE CALLER'S CHOICE. Measured on the 3.10.0 runtime: with a keyframe
        present, `mj_resetData` still writes `qpos0` — only an explicit
        `mj_resetDataKeyframe` applies one. Having `reset_data` silently
        "prefer" a keyframe would change the reset pose of every model that
        declares one, away from what MuJoCo does, with nothing to notice it.

        That distinction is the whole point of the feature for ToddlerBot: its
        reference env resets from `keyframe("home").qpos`, whose values differ
        from qpos0 in 26 of 51 slots by up to 1.5708 rad. The env asks for the
        keyframe; the engine does not assume it.

        `qacc`/`qfrc` are zeroed like `mj_resetData` does. `ctrl` lives on the
        caller's action path rather than in `Data`, so read it with
        `key_ctrl_at` — ToddlerBot's `home` sets 18 of its 30 controls, and
        dropping them would leave the actuators commanding zero at t=0.
        """
        for i in range(Self.NQ):
            d.qpos.data[i] = Scalar[DTYPE](Self.key_qpos_at[DTYPE](sf, k, i))
        for i in range(Self.NV):
            d.qvel.data[i] = Scalar[DTYPE](Self.key_qvel_at[DTYPE](sf, k, i))
            d.qacc.data[i] = Scalar[DTYPE](0)
            d.qfrc.data[i] = Scalar[DTYPE](0)
            # ⚠ `mj_resetDataKeyframe` calls `mj_resetData` first, so the
            # carried acceleration goes with them. See `reset_data`.
            d.qacc_warmstart.data[i] = Scalar[DTYPE](0)

    @staticmethod
    def extract_obs[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        mut obs: List[Scalar[DTYPE]],
    ):
        """Extract observation: qpos[obs_qpos_skip:] followed by qvel[:]."""
        for i in range(Self.NQ - Self.obs_qpos_skip):
            obs.append(d.qpos.data[Self.obs_qpos_skip + i])
        for i in range(Self.NV):
            obs.append(d.qvel.data[i])

    @staticmethod
    def enforce_limits[DTYPE: DType, D: DimsLike, D2: DimsLike](
        sf: SpecFields[DTYPE, D],
        mut d: Data[DTYPE, D2, 1],
    ):
        """Clamp qpos to joint range limits (limited joints only)."""
        # ⚠ FROM `sf.joint_limits`, NOT from four materialized comptime
        # arrays. `Model.joints` has the ranges too but no LIMITED flag, and
        # `range_min < range_max` is not a substitute for one — MuJoCo spells
        # an unlimited joint both as `[0, 0]` and as `[-1e10, 1e10]`.
        for j in range(Self.NJOINT):
            var o = j * JLIM_SIZE
            if sf.joint_limits.data[o + JLIM_IDX_LIMITED] == 0:
                continue
            var qp_adr = Int(sf.joint_limits.data[o + JLIM_IDX_QPOS_ADR])
            if qp_adr < 0 or qp_adr >= Self.NQ:
                continue
            var lo = sf.joint_limits.data[o + JLIM_IDX_RANGE_MIN]
            var hi = sf.joint_limits.data[o + JLIM_IDX_RANGE_MAX]
            var v = d.qpos.data[qp_adr]
            if v < lo:
                d.qpos.data[qp_adr] = lo
            elif v > hi:
                d.qpos.data[qp_adr] = hi

    @staticmethod
    def ctrl_min_at[DTYPE: DType, D: DimsLike](sf: SpecFields[DTYPE, D], i: Int) -> Float64:
        """`actuator_ctrlrange[i][0]` — the bound `apply_actions` clamps to.

        ⚠ NOT `CTRL_MIN`. That is a single model-wide pair read from a root
        `<default><motor ctrlrange>` and it falls back to (-1, 1) on any model
        that keeps its ranges per actuator or per default class. This reads
        the array the clamp itself uses.
        """
        if i < 0 or i >= Self.nact:
            return 0.0
        return Float64(
            sf.actuators.data[i * MODEL_ACTUATOR_SIZE + ACT_IDX_CTRL_MIN]
        )

    @staticmethod
    def ctrl_max_at[DTYPE: DType, D: DimsLike](sf: SpecFields[DTYPE, D], i: Int) -> Float64:
        """`actuator_ctrlrange[i][1]`. See `ctrl_min_at`."""
        if i < 0 or i >= Self.nact:
            return 0.0
        return Float64(
            sf.actuators.data[i * MODEL_ACTUATOR_SIZE + ACT_IDX_CTRL_MAX]
        )

    @staticmethod
    def ctrl_limited_at[DTYPE: DType, D: DimsLike](sf: SpecFields[DTYPE, D], i: Int) -> Bool:
        """`actuator_ctrllimited[i]` — whether the range above is APPLIED.

        ⚠ READ THIS BEFORE READING `ctrl_min_at`/`ctrl_max_at`. MuJoCo's
        `ctrllimited` defaults to "auto", so an actuator that declares no
        range is UNLIMITED and its stored range is meaningless — ours holds a
        (-1, 1) fallback there, MuJoCo reports [0, 0], and neither is a bound
        anyone should clamp to. `apply_actions` consults this first, and so
        should any caller deriving an action space: for an unlimited actuator
        the honest bound is unbounded, not (-1, 1).
        """
        if i < 0 or i >= Self.nact:
            return False
        return (
            sf.actuators.data[i * MODEL_ACTUATOR_SIZE + ACT_IDX_CTRL_LIMITED]
            != 0
        )

    @staticmethod
    def init_spec_fields[DTYPE: DType, D: DimsLike](
        ctx: DeviceContext,
        mut sf: SpecFields[DTYPE, D],
    ) raises:
        """Build + upload the actuation records (phase 1a.2/1a.3).

        ⚠ THIS PARSES THE XML A SECOND TIME — `init_fields` does its own
        `parse_xml_full`. Measured 2026-08-15: 0.42 ms on cartpole, 8.96 ms on
        dog, once per env construction. That is cheaper than threading an
        out-parameter through `init_fields`' 550-line body, and 1a.4 removes
        the duplication; merging the two builds is 1b's business.
        """
        var fmd = parse_xml_full(Self.xml_text(), Self.asset_base_dir())
        assert_no_pending_dampratio(fmd, String("init_spec_fields"))
        build_spec_fields[DTYPE](fmd, sf)
        sf.upload_all(ctx)

    @staticmethod
    def make_spec_fields[DTYPE: DType, D: DimsLike = ModelDims[Self]]() raises -> SpecFields[DTYPE, ModelDims[Self]]:
        """Host-only actuation records — no `DeviceContext`, no upload.

        For the CPU `apply_actions` path, which reads `sf.actuators.data`
        directly and has no kernel to feed. It exists mostly so a caller need
        not spell out `SpecFields`' six parameters — they are all
        derivable from the model def and getting them wrong is a type error
        with a page-long message.
        """
        var sf = SpecFields[DTYPE, ModelDims[Self]]()
        var fmd_ms = parse_xml_full(Self.xml_text(), Self.asset_base_dir())
        assert_no_pending_dampratio(fmd_ms, String("make_spec_fields"))
        build_spec_fields[DTYPE](fmd_ms, sf)
        return sf^

    @staticmethod
    def apply_actions[
        DTYPE: DType, D: DimsLike, D2: DimsLike, NORMALIZED: Bool = False
    ](
        sf: SpecFields[DTYPE, D],
        mut d: Data[DTYPE, D2, 1],
        actions: List[Float64],
        mut act: List[Scalar[DTYPE]],
    ):
        """See `dynamics.actuation.apply_actions_fields`, which is the body.

        ⚠ MOVED, NOT REIMPLEMENTED (3d). It read `Self.NV`/`Self.NQ`/
        `Self.nact`/`Self.TIMESTEP`, which is what confined driving a model to
        the comptime path. `Self.TIMESTEP` is passed through because it is the
        one value there that is not a dimension.
        """
        apply_actions_fields[DTYPE, NORMALIZED=NORMALIZED](
            sf, d, actions, act, Self.TIMESTEP
        )


    @staticmethod
    def init_fields[DTYPE: DType, D: DimsLike](
        ctx: DeviceContext,
        mut mf: Model[DTYPE, D],
    ) raises:
        """Spec-direct fields model build (G4): parse the XML into a
        FlatModelDef and write the packed record tensors DIRECTLY
        (`fields_build.build_model_fields_from_flat`) — no CPU `Model`/`Data`
        staging, no `setup_model_and_data`, no `load_from_model`. invweight0
        is computed fields-natively (G1) from the reference pose given by the
        fields `reset_data`. The legacy trait-default (setup_model_and_data →
        load_from_model) was deleted at G4."""
        var fmd = parse_xml_full(Self.xml_text(), Self.asset_base_dir())

        # ⚠ THE DIMENSION CHECK THAT REPLACED SILENT TRUNCATION.
        #
        # `FlatModelDef` used to be `Array`-backed and sized by these
        # very parameters, so the parser wrote `if joint_count < NJOINT:` and
        # incremented the counter REGARDLESS — a model with more elements than
        # its declared dimension dropped the overflow without a word. Same
        # shape as `MAX_COMPTIME_TENDONS` and `MAX_NAMED_DEFAULTS`.
        #
        # With `List` storage the parser cannot truncate, so the failure mode
        # inverts: a disagreement between `parse_xml` (which supplied these
        # comptime dims) and `full_parser` (which produced the Lists) now shows
        # up as a LENGTH MISMATCH, and `fields_build` would index past the end.
        # Catch it here, where both numbers are in hand and the message can say
        # which element type disagreed.
        if len(fmd.bodies) != Self.NBODY - 1:
            raise Error(
                String(
                    "physics3d: parser/dimension mismatch on BODIES — ",
                    "parse_xml declared nbody=", Self.NBODY, " (so ",
                    Self.NBODY - 1, " non-world bodies) but full_parser found ",
                    len(fmd.bodies),
                    ". The two MJCF paths disagree; fix the parser, do not",
                    " widen the dimension.",
                )
            )
        if len(fmd.joints) != Self.NJOINT:
            raise Error(
                String(
                    "physics3d: parser/dimension mismatch on JOINTS — declared",
                    " njoint=", Self.NJOINT, ", full_parser found ",
                    len(fmd.joints), ".",
                )
            )
        if len(fmd.geoms) != Self.NGEOM:
            raise Error(
                String(
                    "physics3d: parser/dimension mismatch on GEOMS — declared",
                    " ngeom=", Self.NGEOM, ", full_parser found ",
                    len(fmd.geoms), ".",
                )
            )
        # ⚠⚠ AND `NQ`/`NV`, WHICH THE JOINT COUNT ABOVE DOES NOT COVER
        # (AUD-19). The two paths can agree on HOW MANY joints there are and
        # disagree on what they weigh: `parse_xml` reads a joint's `type` off
        # the ELEMENT only (`xml_parser.mojo:2956-2961`), so a class-level
        # `type="ball"` or `type="free"` counts as a 1-dof hinge there and as
        # 4/3 or 7/6 here. njoint matches, NQ/NV do not, and every qpos index
        # downstream is shifted — the quietest possible way to get a wrong
        # model, because nothing is out of range.
        var _nq = 0
        var _nv = 0
        for _ji in range(len(fmd.joints)):
            _nq += fmd.joints[_ji].nq
            _nv += fmd.joints[_ji].nv
        if _nq != Self.NQ or _nv != Self.NV:
            raise Error(
                String(
                    "physics3d: parser/dimension mismatch on NQ/NV — declared",
                    " nq=", Self.NQ, " nv=", Self.NV,
                    ", full_parser's joints sum to nq=", _nq, " nv=", _nv,
                    ". The joint COUNT agrees, so this is a joint whose",
                    " `type` the comptime scan could not see — it reads the",
                    " element only, and a `type=\"ball\"` or `type=\"free\"`",
                    " stated in a <default> class is invisible to it.",
                )
            )
        # ⚠⚠ `na`/`nkey` ARE HAND-SUPPLIED AND NOTHING ELSE CHECKS THEM.
        # `parse_xml` does not compute either, so they are the one pair of
        # dimensions with no automatic source — and both fail SILENTLY when
        # wrong. Too small an `na` leaves a `dyntype` actuator's activation
        # outside the `act` slab, so `apply_actions` skips the filter and the
        # servo becomes a direct drive; too small an `nkey` drops keyframes
        # that `reset_data_keyframe` would then read as qpos0. This is the
        # staleness assert that makes the defaults safe.
        if fmd.na != Self.na:
            raise Error(
                String(
                    "physics3d: `na` mismatch — the model def declares na=",
                    Self.na, " but the XML resolves to ", fmd.na,
                    ". `na` counts ACTIVATION VARIABLES (actuators with a",
                    " dyntype), not actuators; pass `na = <that number>` on",
                    " the ModelDefFromXML declaration.",
                )
            )
        # ⚠⚠ AND THE ACTUATOR COUNT, WHICH WAS THE ONE DIMENSION WITH NO
        # CHECK AT ALL (AUD-19 tail). `parse_xml` counted four actuator tags
        # where `full_parser` builds six — `<adhesion>` and a `<plugin>`
        # actuator were invisible to it — so `NACT` came out short, the
        # control vector was shorter than MuJoCo's `nu`, and every actuator
        # past the uncounted one read the previous one's control. Silently:
        # njoint, nbody, ngeom, na and nkey all still agreed.
        #
        # The count is fixed at the source now; this is the guard that makes
        # the NEXT divergence loud, and it is the same shape as the NQ/NV one
        # above — the two parsers are separate code and will drift again.
        if len(fmd.actuators) != Self.NACT:
            raise Error(
                String(
                    "physics3d: parser/dimension mismatch on ACTUATORS —",
                    " declared nact=", Self.NACT, ", full_parser found ",
                    len(fmd.actuators),
                    ". The control vector is sized by the declaration, so a",
                    " short count silently shifts every actuator past the",
                    " one the comptime scan could not see (`<adhesion>` and",
                    " `<plugin>` were the two it missed).",
                )
            )
        if fmd.nkey != Self.nkey:
            raise Error(
                String(
                    "physics3d: `nkey` mismatch — the model def declares nkey=",
                    Self.nkey, " but the XML has ", fmd.nkey,
                    " <keyframe><key> entries. Pass `nkey = <that number>`.",
                )
            )

        if len(fmd.sites) != Self.NSITE:
            raise Error(
                String(
                    "physics3d: parser/dimension mismatch on SITES — declared",
                    " nsite=", Self.NSITE, ", full_parser found ",
                    len(fmd.sites), ". Sensors are addressed BY SITE INDEX, so",
                    " a mismatch here reads the wrong sensor.",
                )
            )

        # ⚠⚠ EXACT, NOT A CEILING, AND FOR A SHARPER REASON THAN SITES. An
        # UNDER-declared `nsensor` makes `fields_build` stop writing rows
        # partway, so the sensors past the cut keep whatever was in the buffer
        # — and an OVER-declared one leaves trailing rows of zeros, which read
        # as `type 0` (touch) on `site 0` with `dim 0`. Both are silent. The
        # same holds for `nsensordata`: it sizes `d.sensordata`, so declaring
        # it short truncates the values of the LAST sensors only, which is the
        # hardest shape of this bug to see.
        #
        # ⚠ BOTH NUMBERS ARE CHECKED, because they are independent — a sensor
        # is 1, 3, 4 or 6 wide, so getting `nsensor` right says nothing about
        # `nsensordata`. `FlatModelDef.nsensordata()` recomputes the sum rather
        # than trusting a stored field, so this compares the parse against the
        # declaration and not against itself.
        if len(fmd.sensors) != Self.NSENSOR:
            raise Error(
                String(
                    "physics3d: parser/dimension mismatch on SENSORS —",
                    " declared nsensor=", Self.NSENSOR,
                    ", full_parser found ", len(fmd.sensors),
                    ". Pass `nsensor = <that number>`; a short count leaves",
                    " later sensor rows unwritten and a long one leaves rows",
                    " of zeros that read as a touch sensor on site 0.",
                )
            )
        if fmd.nsensordata() != Self.NSENSORDATA:
            raise Error(
                String(
                    "physics3d: parser/dimension mismatch on SENSORDATA —",
                    " declared nsensordata=", Self.NSENSORDATA,
                    ", the model's sensors sum to ", fmd.nsensordata(),
                    ". This is sum(sensor_dim), NOT nsensor (a sensor is 1,",
                    " 3, 4 or 6 wide). Pass `nsensordata = <that number>`.",
                )
            )

        # ⚠ A TENDON THAT OUTGREW `MAX_COMPTIME_TENDON_WRAPS` MUST NOT BUILD.
        # The wrap loop used to stop at a bare 4 and write `tendon_trn_n = 4`,
        # so a longer tendon looked complete to every consumer. dm_control's
        # dog wraps 11 and 10 joints on its tail tendons, and drove a third of
        # them. Section 4.3 of the port plan is explicit that widening a cap is
        # the easy half and the assert is the load-bearing one — this is that
        # assert, and it is why the parse counts the overflow instead of
        # discarding it.
        # ⚠ SAME RULE FOR THE RENDER RECORD. `ComptimeRenderData`'s geom and
        # site arrays were bare literals (64 and 16) with a SILENT fill guard
        # (`if geom_count < 64:`), while every consumer loops `range(NGEOM)`.
        # dog has 128 geoms, so the viewer indexed 64 past the end — a
        # `debug_assert` in the viewer build, and an out-of-bounds READ in a
        # release one. quadruped (30 sites), humanoid (25) and manipulator (20)
        # were over the site cap identically.
        #
        # Checked against NGEOM/NSITE, which come from `parse_xml` and are the
        # TRUE counts, so this catches the overflow the render fill silently
        # dropped. No counter is needed on `ComptimeRenderData` itself, which
        # also avoids touching its constructors — adding fields there has
        # tripped the comptime interpreter before ("interpreting memcpy can't
        # get dst memory").
        # ⚠ THE RENDER-GEOM AND RENDER-SITE CAPS ARE GONE (phase 1a.5c).
        # `NGEOM > MAX_COMPTIME_RENDER_GEOMS` and the matching site guard
        # existed to stop a model overflowing `ComptimeRenderData`'s fixed
        # `Array`s, which truncated silently and left the renderer
        # reading past the end. `RenderFields` is `List`-backed and cannot
        # truncate, so there is nothing left to bound.

        # ⚠ A TENDON THAT OUTGREW `TENDON_MAX_WRAPS` MUST NOT BUILD, and
        # this now reads the RUNTIME record's own overflow counter rather than
        # `_acd`'s. Both parsers count the surplus instead of discarding it,
        # for the reason the original note gives: the wrap loop used to stop
        # at a bare 4 and write `trn_n = 4`, so a longer tendon looked
        # complete to every consumer. dm_control's dog wraps 11 and 10 joints
        # on its tail tendons and drove a third of them.
        for _t in range(len(fmd.tendons)):
            if fmd.tendons[_t].wrap_overflow > 0:
                raise Error(
                    String(
                        "physics3d: tendon ", _t, " wraps ",
                        fmd.tendons[_t].wrap_overflow,
                        " joints more than TENDON_MAX_WRAPS. Raise it in",
                        " gpu/constants.mojo; it is NOT safe to truncate —",
                        " the actuator would drive a subset of the joints and",
                        " every gate would still pass.",
                    )
                )

        # The COUNT of tendons, guarded like their WIDTH above.
        #
        # ⚠ `max_tendon` defaults to 0 and is written by hand on each model, so
        # a model with tendons that never passes it asks for a 1-slot array.
        # That was harmless while `_NTEN` was a global cap and became a silent
        # truncation the moment it was not (cc7021d0); fish shipped with one of
        # its two tendons dropped for a day. An unset parameter must fail the
        # build, not quietly resize the model.
        if len(fmd.tendons) > Self.MAX_TENDON:
            raise Error(
                String(
                    "physics3d: this model declares ", len(fmd.tendons),
                    " tendons but max_tendon=", Self.MAX_TENDON,
                    ". Pass `max_tendon = <parse>.NTENDON` on the",
                    " ModelDefFromXML declaration. Truncating is NOT safe: a",
                    " dropped tendon's actuator resolves to trn_n == 0, which",
                    " apply_actions skips, so the env builds and runs with",
                    " that degree of freedom simply inert.",
                )
            )

        # The COUNT of equalities — the last member of this family that still
        # truncated in silence.
        #
        # ⚠⚠ `max_equality` DOES DOUBLE DUTY, which is why this was missed.
        # `fields_build`'s fill loop breaks on `num_eq >= MAX_EQUALITY`, so it
        # is a RECORD cap there; every model in the tree instead sizes it as a
        # ROW budget (sawyer passes 6 for a single weld, "1 weld = 6 rows").
        # Both readings are over-satisfied by a generous number, so nobody
        # noticed that the record cap can also be UNDER-satisfied — and then
        # `MODEL_META_IDX_NEQUALITY` is clamped to the cap and the surplus
        # equalities are gone. Measured before this guard existed: a model with
        # two `<weld>`s and `max_equality=1` BUILT, reported `nequality = 1`,
        # and simply did not enforce the second weld.
        #
        # ⚠ Its siblings all already raised — `nexclude` and `npair` in
        # `fields_build` (a model with one `<exclude>` and `nexclude=0` raises),
        # and `max_tendon` directly above. Equality was the only hole, so the
        # rule is now uniform: an under-declared dimension fails the build.
        #
        # ⚠ `>` not `!=`: over-allocating is legitimate and universal here,
        # because the row reading needs more slots than the record reading.
        if len(fmd.equalities) > Self.MAX_EQUALITY:
            raise Error(
                String(
                    "physics3d: this model declares ", len(fmd.equalities),
                    " <equality> records but max_equality=", Self.MAX_EQUALITY,
                    ". Pass a value at least that large — and note it is ALSO",
                    " read as a ROW budget elsewhere, where a weld needs 6",
                    " rows and a connect 3, so size it for the rows and it",
                    " covers the records. Truncating is NOT safe: the fill",
                    " loop stops at the cap and the model builds, runs, and",
                    " simply does not enforce the constraints past it.",
                )
            )

        # Reject unplumbed dof-friction solver params LOUDLY, for the same
        # reason as the actuator guard below: the parser sees the attribute,
        # the model would build, and `friction_dof.mojo` would quietly use
        # MuJoCo's DEFAULT solref/solimp instead of the ones the XML asked for.
        for j in range(Self.njoint):
            if fmd.joints[j].has_friction_solparams:
                raise Error(
                    String(
                        "physics3d: joint index ",
                        j,
                        " sets solreffriction/solimpfriction, which are not"
                        " plumbed. constraints/friction_dof.mojo assumes"
                        " MuJoCo's defaults (solref 0.02 1, solimp"
                        " 0.9 0.95 0.001 0.5 2); give them their own model"
                        " meta slots beside the limit ones before using these"
                        " attributes.",
                    )
                )

        # Reject a <spatial> tendon that has nowhere to go. `max_tendon` sizes
        # the tendon record array, and a model that leaves it 0 while the XML
        # routes a string through sites would build fine and simulate with the
        # string simply absent — ball_in_cup would become a ball falling past
        # a cup. A <fixed> tendon is exempt: those also ride the comptime
        # transmission/spring path, which needs no records (fish).
        # ⚠ THIS GUARD USED TO BE A COMPTIME `Self.xml.find("<spatial")`,
        # and it was REDUNDANT. `full_parser` puts spatial tendons into
        # `fmd.tendons` like every other kind — verified on a spatial-only
        # fixture — so the runtime `len(fmd.tendons) > MAX_TENDON` check in
        # `init_fields` already rejects exactly this case, with a message that
        # names the count. Removed rather than converted: phase 1b needs the
        # MJCF to have NO comptime readers left, and a redundant one is not
        # worth reproducing.

        # Tendon LIMIT rows are built only on the PYRAMIDAL edge list. The
        # elliptic core keeps its scalar rows in (dof, sign) form to stay
        # under Metal's local-memory ceiling, so a dense tendon row there is
        # a separate change. No model needs it yet — but a model that did
        # would otherwise simulate with the limit simply absent.
        comptime if Self.cone_type == ConeType.ELLIPTIC:
            for t in range(Self.MAX_TENDON):
                if fmd.tendons[t].limited != 0:
                    raise Error(
                        "physics3d: tendon limits are implemented on the"
                        " PYRAMIDAL cone only; this model sets cone=elliptic"
                        " and has a limited tendon. See"
                        " constraints/tendon_limit.mojo."
                    )

        # A `<fixed>` tendon past the comptime cap is dropped SILENTLY by
        # `parse_xml_model_data`, taking every actuator transmitted through it
        # with it (`motor_trn_n == 0`, which `apply_actions` skips). Fail to
        # compile instead.
        comptime assert Self.MAX_TENDON <= MAX_COMPTIME_TENDONS, (
            "physics3d: this model has more <fixed> tendons than"
            " MAX_COMPTIME_TENDONS; raise it in xml_parser.mojo. Leaving it"
            " would silently drop the surplus tendons AND disable every"
            " actuator transmitted through them."
        )

        # ── The same trap, on three more comptime tables (2026-08-03) ────────
        #
        # `parse_xml_model_data` scans actuators and joints with
        # `while count < CAP`, while `ParsedModel` counts the tags
        # INDEPENDENTLY — so before these asserts a model past either cap built
        # and ran, with the right nact/njoint and a silently truncated `_acd`.
        # The two payloads differ and both are invisible at runtime:
        #   actuators — the env exposes the full action space and every
        #               actuator past the cap applies ZERO FORCE;
        #   joints    — the dof survives but its LIMIT ROW is never built, so
        #               the joint quietly loses its stops.
        # `qpos0` is worse still: it is indexed by the joint's own qpos address
        # rather than scanned, so `nq > MAX_COMPTIME_NQ` writes OUT OF BOUNDS.
        comptime assert Self.nact <= MAX_COMPTIME_ACTUATORS, (
            "physics3d: this model has more actuators than"
            " MAX_COMPTIME_ACTUATORS; raise it in xml_parser.mojo. Leaving it"
            " would expose the full action space while every actuator past the"
            " cap applies zero force."
        )
        comptime assert Self.njoint <= MAX_COMPTIME_JOINTS, (
            "physics3d: this model has more joints than MAX_COMPTIME_JOINTS;"
            " raise it in xml_parser.mojo. Leaving it would keep every dof past"
            " the cap while silently dropping its joint-limit row."
        )
        comptime assert Self.nq <= MAX_COMPTIME_NQ, (
            "physics3d: this model's nq exceeds MAX_COMPTIME_NQ; raise it in"
            " xml_parser.mojo. Unlike the caps above this one is not a"
            " truncating scan — `qpos0` is indexed by qpos address, so it"
            " writes out of bounds."
        )
        # And on the RENDER tables, where it had already gone off. `nmat` here
        # is `ParsedModel.NMAT`, an uncapped `_count_tag`; `_rcd`'s material
        # arrays hold MAX_COMPTIME_MATERIALS. `render_body_geoms` guards its
        # material lookup with `mid < Self.nmat`, so while those two disagreed
        # the guard was checking the WRONG BOUND and let an out-of-range index
        # straight through to the array — point_mass, fish and reacher aborted
        # at the first frame. Anything that survived did so by having every
        # material land under the cap.
        # ⚠ THE `<material>` AND `<texture>` CAPS ARE GONE for the same
        # reason as the geom and site ones above: they bounded
        # `ComptimeRenderData`'s arrays. Their notes recorded two real
        # failures — a material index checked against `nmat` while the array
        # was sized `MAX_COMPTIME_MATERIALS` aborted point_mass, fish and
        # reacher at the first frame, and textures past the cap were dropped
        # so the skybox fell back to flat colour. Neither is expressible now:
        # `RenderFields.mat_*` and `tex_*` are sized by what the model has.

        # `<option noslip_iterations>` used to be refused here on an
        # ELLIPTIC-cone model, because `mj_solNoSlip` was implemented for the
        # pyramidal cone only and the elliptic solve path had no call — the
        # pass would have vanished without a word. `solver/noslip.mojo` now
        # carries BOTH branches and `_newton_solve_env` dispatches to the
        # matching one inside each cone's solve body, so the combination is
        # supported and the assert is gone (2026-08-13, task #53).
        #
        # ⚠ `allow_missing_noslip` IS NOW A NO-OP and is kept only so the
        # model defs that pass it still compile. It is deprecated; see the
        # parameter's doc entry. Do not add new uses — there is nothing left
        # for it to permit.

        # ⚠⚠ THESE FOUR WERE `comptime assert`s AND ARE NOW RUNTIME RAISES,
        # AND THAT IS A REAL WEAKENING — a model with an unsupported keyframe
        # or `<general>` shape now BUILDS and fails when the env is
        # constructed, instead of failing to compile. It is not avoidable:
        # the data they test moved off the comptime interpreter, which is the
        # whole point of the phase, and a runtime record cannot be read by a
        # `comptime assert`.
        #
        # What survives is that they still fail LOUDLY and before a single
        # step, which is what the original notes actually argue for — the
        # alternative each one guards against is a model that runs and is
        # quietly wrong. `init_fields` is called from every env constructor
        # and from every test that builds a model, so nothing reaches physics
        # without passing here.
        if fmd.bad_keyframe_code == 2:
            raise Error(
                "physics3d: <key act=/mpos=/mquat=> is not modelled. We carry"
                " no actuator activation state and no mocap pose in a"
                " keyframe, and applying the key while ignoring those would"
                " reset to a DIFFERENT state than MuJoCo does. Zero of"
                " Menagerie's 147 keyframe attributes use any of the three, so"
                " this refuses nothing that exists today."
            )
        # ⚠ A wrong-length qpos/qvel/ctrl is caught here rather than padded.
        # MuJoCo pads a SHORT attribute, but from spec-level default state in
        # RAW units, not from qpos0. 145 of Menagerie's 145 real keyframe
        # attributes are exactly full length, so nothing depends on it.
        for _k in range(fmd.nkey):
            if fmd.key_nqpos[_k] != 0 and fmd.key_nqpos[_k] != Self.nq:
                raise Error(
                    String(
                        "physics3d: <key qpos=...> at key ", _k, " has length ",
                        fmd.key_nqpos[_k], ", not nq=", Self.nq,
                        ". MuJoCo pads a short one from unconverted spec"
                        " defaults; we refuse it instead.",
                    )
                )
            if fmd.key_nctrl[_k] != 0 and fmd.key_nctrl[_k] != Self.nact:
                raise Error(
                    String(
                        "physics3d: <key ctrl=...> at key ", _k,
                        " has length ", fmd.key_nctrl[_k], ", not nu=",
                        Self.nact, ".",
                    )
                )
            if fmd.key_nqvel[_k] != 0 and fmd.key_nqvel[_k] != Self.nv:
                raise Error(
                    String(
                        "physics3d: <key qvel=...> at key ", _k,
                        " has length ", fmd.key_nqvel[_k], ", not nv=",
                        Self.nv, ".",
                    )
                )

        # A `<general>` whose gain/bias/dyn shape we do not implement. The
        # parser records the offender rather than raising mid-parse; this is
        # where it becomes an error. Codes are on `bad_actuator_code`.
        if fmd.bad_actuator >= 0:
            raise Error(
                String(
                    "physics3d: <general> actuator at index ",
                    fmd.bad_actuator,
                    " has an unsupported gain/bias/dyn shape (code ",
                    fmd.bad_actuator_code,
                    "). Supported: gaintype=fixed, biastype none or"
                    " affine with ANY biasprm (the full affine law lives in"
                    " `actuator_scalar_force` since 2026-08-21 — codes 2 and"
                    " 3 are retired), and dyntype none|filter.",
                )
            )

        # Reject unimplemented actuator transmissions LOUDLY. Building the
        # model anyway would simulate a servo as a torque motor with no error
        # at all. `<motor>`, `<position>`, `<velocity>` and the `<general>`
        # shape validated just above are modelled (see `apply_actions`).
        # See docs/DM_CONTROL_PORT.md (gap G3).
        comptime if not Self.allow_unsupported_actuators:
            for a in range(Self.nact):
                var kind = fmd.actuators[a].kind
                if (
                    kind != ACT_KIND_MOTOR
                    and kind != ACT_KIND_POSITION
                    and kind != ACT_KIND_VELOCITY
                    and kind != ACT_KIND_GENERAL
                ):
                    raise Error(
                        String(
                            "physics3d: unimplemented actuator transmission ",
                            act_kind_name(kind),
                            " at actuator index ",
                            a,
                            ". <motor> (force = gear*ctrl), <position>",
                            " (force = kp*(ctrl - length) - kv*velocity),",
                            " <velocity> (force = kv*(ctrl - velocity)) and",
                            " <general> restricted to that same affine shape",
                            " are supported. If this env's CONFIG",
                            " drives those DOFs itself",
                            " (custom_apply_actions_cpu -> True), pass",
                            " allow_unsupported_actuators=True.",
                        )
                    )
                # Same trap one level down: `site=`, `body=` and `cranksite=`
                # are valid MJCF transmissions that carry neither a `joint`
                # nor a `tendon` attribute, so nothing resolves and the
                # actuator would be built against a garbage index instead of
                # failing. `tendon=` IS resolved (fish's `fins_flap`), so the
                # check is on the comptime transmission list rather than on
                # `joint_id`, which a tendon transmission legitimately leaves
                # at its -1 sentinel.
                if fmd.actuators[a].trn_n == 0:
                    raise Error(
                        String(
                            "physics3d: actuator index ",
                            a,
                            " has no resolvable transmission. `joint=` and",
                            " `tendon=` (fixed tendons) are modelled;",
                            " site/body/cranksite are not. Rewrite it as a",
                            " joint actuator if the transmission is",
                            " equivalent, or pass",
                            " allow_unsupported_actuators=True if this env's",
                            " CONFIG drives the DOF itself.",
                        )
                    )

        # ELLIPSOID narrow phase, as of 2026-07-31, covers the PLANE pair only
        # (`ellipsoid_plane`, the closed-form support point that MuJoCo reaches
        # via mjc_PlaneConvex; verified to 1.1e-16 against MuJoCo over 3041
        # contacts). This check used to reject any collidable ellipsoid
        # outright, which is no longer right — but it is not yet right to
        # accept every pair either.
        #
        # ⚠ KNOWN LIMITATION, deliberately not a hard error. An ellipsoid
        # paired with a sphere/capsule/box/mesh matches NO branch of the
        # narrow-phase dispatch and therefore yields NO CONTACT — silently.
        # (Missing a contact is at least safer than the sphere-of-size[0]
        # substitution this check originally guarded against.) It is left
        # permitted because the static test that would catch it — "does a
        # collidable non-plane geom exist that MuJoCo would pair with this
        # ellipsoid" — is true for quadruped (torso vs the leg capsules, which
        # are not parent/child) while the contact is unreachable in practice:
        # measured over 60,000 MuJoCo steps of aggressive random control from
        # 40 random orientations, the torso ellipsoid touched ONLY the floor
        # (27,748 contacts, zero against any capsule). A model that does need
        # ellipsoid-vs-convex must add it; see noeira/envs/ROADMAP.md.
        comptime if not Self.allow_unsupported_actuators:
            for g in range(Self.NGEOM):
                if fmd.geoms[g].geom_type != _GEOM_ELLIPSOID:
                    continue
                if fmd.geoms[g].contype == 0 and fmd.geoms[g].conaffinity == 0:
                    continue
                # Reachable and handled: plane pairs. Nothing to reject.
                break

        # ⚠ THE THREE `<compiler>` BUILD MODES WERE READ HERE AT COMPTIME,
        # off `Self.xml`, and passed down as compile-time parameters. They
        # come off `fmd` now — the parse this function already has in hand.
        # They were three of the last comptime readers of the MJCF string,
        # and every one of those pins the model to Mojo source: the comptime
        # interpreter cannot `open()` a file (§10.2).
        build_model_fields_from_flat[DTYPE](fmd, mf)

        # Reference pose + fields-native invweight0 (G1).
        var d_inv = Data[DTYPE, D, 1]()
        # ⚠ Built from the `fmd` ALREADY IN HAND — `init_spec_fields` would
        # re-parse the XML a third time for a value this function has sitting
        # in a local.
        var sf_inv = SpecFields[DTYPE, ModelDims[Self]]()
        build_spec_fields[DTYPE](fmd, sf_inv)
        Self.reset_data[DTYPE](sf_inv, d_inv)
        var sc_inv = DynamicsScratch[DTYPE, D, 1]()
        compute_invweight0[DTYPE](d_inv, mf, sc_inv)

        # ⚠ MOVED TO `fields_build.apply_auto_spring_damper` (3c-c) so the
        # runtime loader runs the SAME body. Its docstring carries the
        # ordering rule this comment used to.
        apply_auto_spring_damper[DTYPE](fmd, mf)

        mf.upload_all(ctx)

    # =========================================================================
    # GPU: Joints / Actuators kernel delegates
    # =========================================================================

    @staticmethod
    def apply_actions_kernel_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        ACTION_DIM: Int,
        NORMALIZED: Bool = False,
    ](
        ctx: DeviceContext,
        qfrc: Tile[DTYPE, BATCH_SIZE, Self.NV],
        actions: Tile[DTYPE, BATCH_SIZE, ACTION_DIM],
        qpos: Tile[DTYPE, BATCH_SIZE, Self.NQ],
        qvel: Tile[DTYPE, BATCH_SIZE, Self.NV],
        act: Tile[DTYPE, BATCH_SIZE, Self.NA_F],
        acts: Tile[DTYPE, Self.NACT_F * MODEL_ACTUATOR_SIZE],
        act_tendons: Tile[DTYPE, Self.NTEN_F * MODEL_ACT_TENDON_SIZE],
        # `sf.joint_limits` — for `JLIM_IDX_ACTFRC_*` only. The GPU twin has
        # to receive it because `mj_fwdActuation`'s SECOND force clamp is
        # per-JOINT, and a kernel that clamps only the actuator's own
        # `forcerange` computes a different force from the CPU path for the
        # same action — the shape of defect #54, which this file already
        # carries two warnings about.
        joint_limits: Tile[DTYPE, Self.NJOINT * JLIM_SIZE],
        # `d.dof_actdamp` + `d.meta` — THIS STEP's actuator damping diagonal
        # and the flag saying it was filled. MuJoCo's `mjd_actuator_vel` skips
        # an actuator whose force is clamped by `forcerange`, so the value is
        # state-dependent and cannot live on `Model`. Filled here for the same
        # reason the ctrl and force clamps are: the CPU and GPU paths must
        # compute the same forces AND the same derivative from one action.
        dof_actdamp: Tile[DTYPE, BATCH_SIZE, Self.NV],
        # `d.actdamp_act` — the OFF-DIAGONAL's live gate, 1.0 per actuator
        # that contributed. `Model.actdamp_trn` holds the constants; this
        # says which of them count this step. Same reason as `dof_actdamp`:
        # one action must give the CPU and GPU paths the same derivative.
        actdamp_act: Tile[DTYPE, BATCH_SIZE, Self.NACT_F],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
    ) raises:
        """Generalized forces from the model spec — the GPU mirror of
        `apply_actions` above, term for term.

        ⚠⚠ THIS USED TO BE `qfrc[dof] = gear * ctrl` OVER A SINGLE DOF, and
        that was blocker G. It read `motor_dof_adr` — one dof per actuator —
        where the CPU path walks the TRANSMISSION TRIPLES
        (`motor_trn_n/qadr/dadr/coef`, up to MAX_COMPTIME_TENDON_WRAPS of
        them). For a plain joint transmission the two agree exactly, which is
        why every Gym env was fine and nobody noticed. For a FIXED-TENDON
        transmission — point_mass, fish, manipulator, stacker, quadruped — the
        old form applied the whole force to one dof with coefficient 1 instead
        of distributing `gear * coef_k * force` across the tendon's dofs.

        Measured on point_mass before the fix, CPU vs GPU over 12 steps:
            action = 0.0   worst |qvel diff| = 0.0     (bit-identical)
            action = 0.8   worst |qvel diff| = 0.043
        i.e. the integrator was exact and the transmission was not.

        It also ASSIGNED rather than accumulating, and never zeroed the dofs no
        actuator drives — so a tendon transmission and a tendon spring landing
        on the same dof (fish's `fins_flap` + `fins_sym`) could not both apply.

        ACTIVATION (`d->act`) landed with blocker E3: `Phyics3dBatchedEnv`
        owns a `[N_ENVS, NA_F]` activation slab, and the mjDYN_FILTER
        integration at the end of the actuator loop below advances it.
        """
        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        # ⚠ CADENCE — RESOLVED 2026-08-07, READ THIS BEFORE MOVING THE CALL.
        # Every term below is a PER-SUBSTEP quantity, and this used to be
        # invoked once per CONTROL step while `Phyics3dEnv.step` calls the CPU
        # twin once per SUBSTEP. A `<motor>` is immune (its force is
        # `gear * coef * kp * ctrl`, constant across the step), which is why
        # every model gated before quadruped was unaffected — but a position
        # servo, a fixed-tendon spring and a `dyntype` activation all move
        # every substep, and two comptime asserts refused those models rather
        # than integrate a frozen force. `Phyics3dBatchedEnv._step_impl` now
        # calls this at the top of every substep, so the asserts are gone.
        # If this ever moves back outside the frame-skip loop, restore them.

        @always_inline
        def apply_kernel(
            qfrc: Tile[DTYPE, BATCH_SIZE, Self.NV],
            actions: Tile[DTYPE, BATCH_SIZE, ACTION_DIM],
            qpos: Tile[DTYPE, BATCH_SIZE, Self.NQ],
            qvel: Tile[DTYPE, BATCH_SIZE, Self.NV],
            act: Tile[DTYPE, BATCH_SIZE, Self.NA_F],
            acts: Tile[DTYPE, Self.NACT_F * MODEL_ACTUATOR_SIZE],
            act_tendons: Tile[DTYPE, Self.NTEN_F * MODEL_ACT_TENDON_SIZE],
            joint_limits: Tile[DTYPE, Self.NJOINT * JLIM_SIZE],
            dof_actdamp: Tile[DTYPE, BATCH_SIZE, Self.NV],
            actdamp_act: Tile[DTYPE, BATCH_SIZE, Self.NACT_F],
            meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return

            # Zero first: this ACCUMULATES, and dofs no actuator drives must
            # not keep the previous step's force.
            comptime for i in range(Self.NV):
                qfrc[env, i] = Scalar[DTYPE](0)
            comptime for i in range(Self.NV):
                dof_actdamp[env, i] = Scalar[DTYPE](0)
            comptime for i in range(Self.NACT_F):
                actdamp_act[env, i] = Scalar[DTYPE](0)
            meta[env, META_IDX_ACTDAMP_LIVE] = Scalar[DTYPE](1)

            # ⚠⚠ THIS LOOP USED TO BE `comptime for`, WITH EVERY VALUE A
            # BAKED LITERAL. It now reads `acts` / `act_tendons` — the same
            # `SpecFields` records the CPU twin reads, so the two cannot drift
            # apart the way defect #54's two noslip branches did. The wrap
            # loops HAD to become runtime loops (`n` is a load now), and the
            # outer one follows: 38 unrolled bodies each containing a runtime
            # loop is strictly more code than one loop that runs 38 times.
            for act_i in range(Self.nact):
                if act_i >= ACTION_DIM:
                    break
                var o = act_i * MODEL_ACTUATOR_SIZE
                var n = Int(rebind[Scalar[DTYPE]](acts[o + ACT_IDX_TRN_N]))
                if n == 0:
                    continue
                var gear = rebind[Scalar[DTYPE]](acts[o + ACT_IDX_GEAR])

                # ⚠ GATED ON `ctrllimited`, and this is the SECOND site — the
                # CPU `apply_actions` above is the other. Both were once
                # unconditional, so fixing one would have left the two targets
                # computing different forces from the same action, which is the
                # shape of defect #54 all over again.
                var ctrl = rebind[Scalar[DTYPE]](actions[env, act_i])
                if acts[o + ACT_IDX_CTRL_LIMITED] != 0:
                    var c_max = rebind[Scalar[DTYPE]](
                        acts[o + ACT_IDX_CTRL_MAX]
                    )
                    var c_min = rebind[Scalar[DTYPE]](
                        acts[o + ACT_IDX_CTRL_MIN]
                    )
                    # ⚠⚠ BYTE-FOR-BYTE THE CPU HOOK'S MAP. The note above says
                    # it outright: these two clamps were once unconditional and
                    # fixing one alone left the two targets computing different
                    # forces from the same action. The normalized map is the
                    # same hazard, so it is the same three lines here.
                    comptime if NORMALIZED:
                        ctrl = c_min + (ctrl + Scalar[DTYPE](1.0)) * Scalar[
                            DTYPE
                        ](0.5) * (c_max - c_min)
                    if ctrl > c_max:
                        ctrl = c_max
                    elif ctrl < c_min:
                        ctrl = c_min

                # ACTIVATION (MuJoCo `d->act`): `force = gain .* [ctrl/act]`
                # (mj_fwdActuation). An actuator with a `dyntype` feeds its
                # ACTIVATION to the gain; a plain one feeds `ctrl`. `u` is
                # whichever the gain multiplies.
                var adr = Int(rebind[Scalar[DTYPE]](acts[o + ACT_IDX_ACT_ADR]))
                var u = ctrl
                if adr >= 0 and adr < Self.NA_F:
                    u = rebind[Scalar[DTYPE]](act[env, adr])
                var kp = rebind[Scalar[DTYPE]](acts[o + ACT_IDX_KP])
                var force = kp * u

                # The CPU twin's comment explains why POSITION and VELOCITY
                # share this block: MuJoCo gives them the same
                # gaintype/biastype and they differ only in `biasprm[1]`,
                # i.e. in whether `length` is subtracted from the setpoint.
                var kind = Int(rebind[Scalar[DTYPE]](acts[o + ACT_IDX_KIND]))
                if kind == ACT_KIND_POSITION or kind == ACT_KIND_VELOCITY:
                    var kv = rebind[Scalar[DTYPE]](acts[o + ACT_IDX_KV])
                    var b0_g = rebind[Scalar[DTYPE]](
                        acts[o + ACT_IDX_BIAS0]
                    )
                    var b1_g = rebind[Scalar[DTYPE]](
                        acts[o + ACT_IDX_BIAS1]
                    )
                    var length = Scalar[DTYPE](0)
                    var vel = Scalar[DTYPE](0)
                    for k in range(n):
                        var qadr = Int(
                            rebind[Scalar[DTYPE]](
                                acts[o + ACT_IDX_TRN_QADR_0 + k]
                            )
                        )
                        var dadr = Int(
                            rebind[Scalar[DTYPE]](
                                acts[o + ACT_IDX_TRN_DADR_0 + k]
                            )
                        )
                        var coef = rebind[Scalar[DTYPE]](
                            acts[o + ACT_IDX_TRN_COEF_0 + k]
                        )
                        # ⚠ THE GATE IS `biasprm[1] != 0`, NOT `kind ==
                        # POSITION`. It used to be the latter, which is the
                        # same set only while `biasprm[1]` is assumed to be
                        # `-gainprm[0]` or 0 — the assumption `ACT_IDX_BIAS1`
                        # exists to stop making. A VELOCITY actuator still
                        # does not load a position it will not use.
                        if b1_g != 0.0 and qadr >= 0 and qadr < Self.NQ:
                            length += coef * rebind[Scalar[DTYPE]](
                                qpos[env, qadr]
                            )
                        if dadr >= 0 and dadr < Self.NV:
                            vel += coef * rebind[Scalar[DTYPE]](
                                qvel[env, dadr]
                            )
                    length *= gear
                    vel *= gear
                    # `u`, not `ctrl` — for a dyntype actuator the servo
                    # setpoint is the ACTIVATION, which lags the control.
                    # ⚠ ONE LAW, FIVE CALLERS: this is
                    # `dynamics.actuation.actuator_scalar_force` inlined for
                    # the kernel's scalar type. The CPU twin calls the
                    # function; the arithmetic and its GROUPING are the
                    # same, and the helper's docstring is the reference.
                    force = kp * u + (
                        b0_g + b1_g * length + (-kv) * vel
                    )

                # `forcerange` — the CPU twin's comment explains why the
                # clamp sits here, on the scalar force, and not on `qfrc`.
                var saturated = False
                if acts[o + ACT_IDX_FORCE_LIMITED] != 0:
                    var f_hi = rebind[Scalar[DTYPE]](
                        acts[o + ACT_IDX_FORCE_MAX]
                    )
                    var f_lo = rebind[Scalar[DTYPE]](
                        acts[o + ACT_IDX_FORCE_MIN]
                    )
                    if force > f_hi:
                        force = f_hi
                    elif force < f_lo:
                        force = f_lo
                    # `<=` / `>=` on the CLAMPED force, exactly as MuJoCo
                    # tests it — an actuator sitting ON its bound also loses
                    # its derivative.
                    saturated = force <= f_lo or force >= f_hi

                # `-d force / d qvel` onto each transmission dof, skipped for
                # a saturated actuator (`mjd_actuator_vel`).
                if (
                    kind == ACT_KIND_POSITION or kind == ACT_KIND_VELOCITY
                ) and not saturated:
                    var kv_d = rebind[Scalar[DTYPE]](acts[o + ACT_IDX_KV])
                    if kv_d != Scalar[DTYPE](0):
                        for k in range(n):
                            var dadr_d = Int(
                                rebind[Scalar[DTYPE]](
                                    acts[o + ACT_IDX_TRN_DADR_0 + k]
                                )
                            )
                            if dadr_d >= 0 and dadr_d < Self.NV:
                                var gc = gear * rebind[Scalar[DTYPE]](
                                    acts[o + ACT_IDX_TRN_COEF_0 + k]
                                )
                                dof_actdamp[env, dadr_d] = (
                                    dof_actdamp[env, dadr_d] + kv_d * gc * gc
                                )
                        # The off-diagonal's gate — the CPU twin sets it at
                        # exactly this point: AFTER the dof loop, inside
                        # `kv_d != 0`. ⚠ The actuator index here is `act_i`,
                        # not `i` — writing `i` compiled nowhere `drive`
                        # instantiates this kernel and broke eight test files
                        # at once.
                        actdamp_act[env, act_i] = Scalar[DTYPE](1)

                for k in range(n):
                    var dadr = Int(
                        rebind[Scalar[DTYPE]](
                            acts[o + ACT_IDX_TRN_DADR_0 + k]
                        )
                    )
                    if dadr >= 0 and dadr < Self.NV:
                        qfrc[env, dadr] = qfrc[env, dadr] + gear * rebind[
                            Scalar[DTYPE]
                        ](acts[o + ACT_IDX_TRN_COEF_0 + k]) * force

                # mjDYN_FILTER, Euler-integrated exactly as `nextActivation`
                # does (engine_forward.c:341):
                #     act_dot = (ctrl - act) / tau ; act += act_dot * dt
                # ⚠ AFTER the force, matching MuJoCo's order —
                # `mj_fwdActuation` reads the CURRENT act and `mj_advance`
                # advances it at the end of the same step. `ctrl` is already
                # ctrlrange-clamped, as MuJoCo clamps `d->ctrl` before
                # computing act_dot.
                if adr >= 0 and adr < Self.NA_F:
                    act[env, adr] = next_activation[DTYPE](
                        Int(rebind[Scalar[DTYPE]](acts[o + ACT_IDX_DYN_TYPE])),
                        u,
                        ctrl,
                        rebind[Scalar[DTYPE]](acts[o + ACT_IDX_DYN_TAU]),
                        Scalar[DTYPE](Self.TIMESTEP),
                        rebind[Scalar[DTYPE]](acts[o + ACT_IDX_ACT_LIMITED])
                        != Scalar[DTYPE](0),
                        rebind[Scalar[DTYPE]](acts[o + ACT_IDX_ACT_MIN]),
                        rebind[Scalar[DTYPE]](acts[o + ACT_IDX_ACT_MAX]),
                    )

            # ── `jnt_actfrcrange` (`engine_forward.c:477`) ──────────────
            #
            # ⚠ ON THE ACCUMULATED `qfrc`, AFTER EVERY ACTUATOR, AND BEFORE
            # THE SPRINGS. MuJoCo clamps `qfrc_actuator`; a fixed-tendon
            # spring is `qfrc_passive` and is NOT subject to this limit, so
            # putting it after the spring loop would clamp a sum MuJoCo never
            # clamps. The CPU twin sits at the identical position.
            comptime for jj in range(Self.NJOINT):
                var jo = jj * JLIM_SIZE
                if joint_limits[jo + JLIM_IDX_ACTFRC_LIMITED] != 0:
                    var jdof = Int(
                        rebind[Scalar[DTYPE]](joint_limits[jo + JLIM_IDX_DOF_ADR])
                    )
                    if jdof >= 0 and jdof < Self.NV:
                        var a_hi = rebind[Scalar[DTYPE]](
                            joint_limits[jo + JLIM_IDX_ACTFRC_MAX]
                        )
                        var a_lo = rebind[Scalar[DTYPE]](
                            joint_limits[jo + JLIM_IDX_ACTFRC_MIN]
                        )
                        if qfrc[env, jdof] > a_hi:
                            qfrc[env, jdof] = a_hi
                        elif qfrc[env, jdof] < a_lo:
                            qfrc[env, jdof] = a_lo

            # Fixed-tendon springs, deadbanded on `tendon_lengthspring`.
            # ⚠ BOUND BY THE RECORD CAPACITY, not by `_acd.ntendon` — padding
            # rows are zero-filled and `stiffness == 0` skips them, exactly as
            # the CPU twin does.
            for t in range(Self.NTEN_F):
                var to = t * MODEL_ACT_TENDON_SIZE
                var k_spring = rebind[Scalar[DTYPE]](
                    act_tendons[to + ACTTEN_IDX_STIFFNESS]
                )
                if k_spring == Scalar[DTYPE](0):
                    continue
                var nt = Int(
                    rebind[Scalar[DTYPE]](act_tendons[to + ACTTEN_IDX_TRN_N])
                )
                if nt == 0:
                    continue
                var tlen = Scalar[DTYPE](0)
                for k in range(nt):
                    var qadr = Int(
                        rebind[Scalar[DTYPE]](
                            act_tendons[to + ACTTEN_IDX_TRN_QADR_0 + k]
                        )
                    )
                    if qadr >= 0 and qadr < Self.NQ:
                        tlen += rebind[Scalar[DTYPE]](
                            act_tendons[to + ACTTEN_IDX_TRN_COEF_0 + k]
                        ) * rebind[Scalar[DTYPE]](qpos[env, qadr])
                var hi = rebind[Scalar[DTYPE]](
                    act_tendons[to + ACTTEN_IDX_SPRING_HI]
                )
                var lo = rebind[Scalar[DTYPE]](
                    act_tendons[to + ACTTEN_IDX_SPRING_LO]
                )
                var frc = Scalar[DTYPE](0)
                if tlen > hi:
                    frc = k_spring * (hi - tlen)
                elif tlen < lo:
                    frc = k_spring * (lo - tlen)
                if frc != Scalar[DTYPE](0):
                    for k in range(nt):
                        var dadr = Int(
                            rebind[Scalar[DTYPE]](
                                act_tendons[to + ACTTEN_IDX_TRN_DADR_0 + k]
                            )
                        )
                        if dadr >= 0 and dadr < Self.NV:
                            qfrc[env, dadr] = qfrc[env, dadr] + rebind[
                                Scalar[DTYPE]
                            ](act_tendons[to + ACTTEN_IDX_TRN_COEF_0 + k]) * frc

        ctx.enqueue_function[apply_kernel](
            qfrc,
            actions,
            qpos,
            qvel,
            act,
            acts,
            act_tendons,
            joint_limits,
            dof_actdamp,
            actdamp_act,
            meta,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # GPU inline: Per-env methods
    # =========================================================================

    @always_inline
    @staticmethod
    def reset_env_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
    ](
        qpos: Tile[DTYPE, BATCH_SIZE, Self.NQ],
        qvel: Tile[DTYPE, BATCH_SIZE, Self.NV],
        qacc: Tile[DTYPE, BATCH_SIZE, Self.NV],
        qfrc: Tile[DTYPE, BATCH_SIZE, Self.NV],
        qpos0: Tile[DTYPE, Self.NQ_F],
        pose_meta: Tile[DTYPE, POSE_META_SIZE],
        env: Int,
        noise_scale: Scalar[DTYPE],
        seed: Int,
    ):
        """Reset a single env with small random noise around qpos0, qvel=0.

        ⚠ `qpos0`/`pose_meta` added 2026-08-15 with phase 1a.4. The reference
        pose used to be baked into the kernel as comptime literals off `_acd`;
        it is now read from the same `SpecFields` records the CPU
        `reset_data` reads, so the two cannot drift."""
        comptime TOTAL_VALS = Self.NQ + Self.NV
        comptime NUM_BATCHES = (TOTAL_VALS + 3) // 4

        var rng = PhiloxRandom(
            seed=UInt64(seed * 2654435761 + env * 12345), offset=0
        )
        var rand_vals = Array[Scalar[DType.float32], NUM_BATCHES * 4](
            fill=Scalar[DType.float32](0)
        )
        for b in range(NUM_BATCHES):
            var batch = rng.step_uniform()
            rand_vals[b * 4 + 0] = batch[0]
            rand_vals[b * 4 + 1] = batch[1]
            rand_vals[b * 4 + 2] = batch[2]
            rand_vals[b * 4 + 3] = batch[3]

        # ⚠ `qpos0_nq == 0` MEANS "NO POSE WAS PARSED", not "the pose is
        # zero" — the second branch is what supplies the free joint's
        # identity `qw`, and reading a row of zeros instead would leave a
        # degenerate quaternion for FK. Same test as the CPU `reset_data`.
        var nq0 = Int(rebind[Scalar[DTYPE]](pose_meta[POSE_IDX_QPOS0_NQ]))
        var fj = Int(
            rebind[Scalar[DTYPE]](pose_meta[POSE_IDX_FREE_JOINT_QPOS_ADR])
        )
        comptime for i in range(Self.NQ):
            var noise = Scalar[DTYPE](rand_vals[i] * 2.0 - 1.0) * noise_scale
            if nq0 > 0 and i < nq0:
                qpos[env, i] = rebind[Scalar[DTYPE]](qpos0[i]) + noise
            elif fj >= 0 and i == fj + 3:
                # Free-joint qw: start from identity (1.0) + small noise.
                qpos[env, i] = Scalar[DTYPE](1) + noise
            else:
                qpos[env, i] = noise

        comptime for i in range(Self.NV):
            var noise = (
                Scalar[DTYPE](rand_vals[Self.NQ + i] * 2.0 - 1.0) * noise_scale
            )
            qvel[env, i] = noise

        for i in range(Self.NV):
            qacc[env, i] = Scalar[DTYPE](0)
            qfrc[env, i] = Scalar[DTYPE](0)

    @always_inline
    @staticmethod
    def extract_obs_gpu[
        DTYPE: DType,
        BATCH_SIZE: Int,
        OBS_DIM: Int,
    ](
        qpos: Tile[DTYPE, BATCH_SIZE, Self.NQ],
        qvel: Tile[DTYPE, BATCH_SIZE, Self.NV],
        obs: Tile[DTYPE, BATCH_SIZE, OBS_DIM],
        env: Int,
    ):
        """Extract obs = qpos[obs_qpos_skip:] + qvel[:] for a single env."""
        comptime for i in range(Self.NQ - Self.obs_qpos_skip):
            obs[env, i] = qpos[env, Self.obs_qpos_skip + i]

        comptime for i in range(Self.NV):
            obs[env, Self.NQ - Self.obs_qpos_skip + i] = qvel[env, i]

    # =========================================================================
    # Rendering — driven from parsed XML assets, lights, cameras, geoms
    # =========================================================================

    @staticmethod
    def setup_lights(rf: RenderFields) raises -> List[Light]:
        """Return Light objects parsed from <light> elements in <worldbody>."""
        # Mojo 1.0: `Array` is not `ImplicitlyCopyable`, so a comptime array
        # indexed at runtime must be materialized. Hoisted here so each array
        # is copied once per call rather than once per access in the loops.
        ref _m_light_ambient_b = rf.light_ambient_b
        ref _m_light_ambient_g = rf.light_ambient_g
        ref _m_light_ambient_r = rf.light_ambient_r
        ref _m_light_castshadow = rf.light_castshadow
        ref _m_light_diffuse_b = rf.light_diffuse_b
        ref _m_light_diffuse_g = rf.light_diffuse_g
        ref _m_light_diffuse_r = rf.light_diffuse_r
        ref _m_light_dir_x = rf.light_dir_x
        ref _m_light_dir_y = rf.light_dir_y
        ref _m_light_dir_z = rf.light_dir_z
        ref _m_light_directional = rf.light_directional
        ref _m_light_exponent = rf.light_exponent
        ref _m_light_specular_b = rf.light_specular_b
        ref _m_light_specular_g = rf.light_specular_g
        ref _m_light_specular_r = rf.light_specular_r

        var lights = List[Light]()
        for i in range(len(rf.light_dir_x)):
            # ⚠ `directional="false"` (the MJCF default) IS A SPOT — position,
            # 45-degree cone, exponent, attenuation — and MuJoCo lights nothing
            # outside its cone. This line used to read
            # `1 if directional else 0` against a `LightMode` whose 0 is
            # DIRECTIONAL: inverted, and harmless only because the renderer
            # ignored the mode and treated every light as directional. Both
            # are fixed together; see `render/light.mojo`.
            var mode = LightMode.DIRECTIONAL if _m_light_directional[i] else LightMode.SPOT
            var amb = (_m_light_ambient_r[i] + _m_light_ambient_g[i] + _m_light_ambient_b[i]) / 3.0
            var spec_int = (_m_light_specular_r[i] + _m_light_specular_g[i] + _m_light_specular_b[i]) / 3.0
            lights.append(
                Light(
                    mode=mode,
                    dir_x=_m_light_dir_x[i],
                    dir_y=_m_light_dir_y[i],
                    dir_z=_m_light_dir_z[i],
                    color_r=_m_light_diffuse_r[i],
                    color_g=_m_light_diffuse_g[i],
                    color_b=_m_light_diffuse_b[i],
                    ambient=amb,
                    specular_intensity=spec_int,
                    specular_exponent=_m_light_exponent[i],
                    cast_shadow=_m_light_castshadow[i],
                    # ⚠ BODY-LOCAL, like `cam_pos`; `ModelRenderer.render`
                    # re-poses a light on a moving body every frame.
                    pos_x=rf.light_pos_x[i],
                    pos_y=rf.light_pos_y[i],
                    pos_z=rf.light_pos_z[i],
                    cutoff=rf.light_cutoff[i],
                    exponent=_m_light_exponent[i],
                    attenuation_0=rf.light_att_0[i],
                    attenuation_1=rf.light_att_1[i],
                    attenuation_2=rf.light_att_2[i],
                    ambient_r=_m_light_ambient_r[i],
                    ambient_g=_m_light_ambient_g[i],
                    ambient_b=_m_light_ambient_b[i],
                    specular_r=_m_light_specular_r[i],
                    specular_g=_m_light_specular_g[i],
                    specular_b=_m_light_specular_b[i],
                )
            )
        return lights^

    @staticmethod
    def setup_cameras(rf: RenderFields, width: Int, height: Int) raises -> List[Camera3D]:
        """Return Camera3D objects parsed from <camera> elements in <worldbody>.

        ⚠ THE LOOK DIRECTION AND THE UP VECTOR BOTH COME FROM THE CAMERA'S OWN
        ORIENTATION, for every mode. MuJoCo's camera frame looks along its -Z
        with +Y up (`mjCCamera`), and `mode` governs only whether the POSITION
        follows a body — `track`/`trackcom` keep the declared orientation.

        Until 2026-08-03 this was inside out: the quaternion was used only for
        `targetbody`, and fixed/track/trackcom got the invented target
        `(cam_pos_x, 0, 0)` with `up` hardcoded to +Z. For a camera at
        (-3, 0, 1) — humanoid's default `back` camera — that aims straight DOWN
        at (-3, 0, 0), which makes the view direction PARALLEL to the up
        vector. `look_at` then has a zero cross product and the whole view
        matrix degenerates, so the model rendered as nothing at all. That is
        the "humanoid: I can't see anything" report, and point_mass's top-down
        camera hit the same wall.

        Cameras that already looked right are unaffected: cheetah's
        `xyaxes="1 0 0 0 0 1"` at (0,-3,0) resolves to the same +Y look with a
        +Z up it was getting by accident.
        """
        # Mojo 1.0: `Array` is not `ImplicitlyCopyable`, so a comptime array
        # indexed at runtime must be materialized. Hoisted here so each array
        # is copied once per call rather than once per access in the loops.
        ref _m_cam_fovy = rf.cam_fovy
        ref _m_cam_pos_x = rf.cam_pos_x
        ref _m_cam_pos_y = rf.cam_pos_y
        ref _m_cam_pos_z = rf.cam_pos_z
        ref _m_cam_quat_w = rf.cam_quat_w
        ref _m_cam_quat_x = rf.cam_quat_x
        ref _m_cam_quat_y = rf.cam_quat_y
        ref _m_cam_quat_z = rf.cam_quat_z

        var cameras = List[Camera3D]()
        for i in range(len(rf.cam_fovy)):
            var eye = _RVec3(_m_cam_pos_x[i], _m_cam_pos_y[i], _m_cam_pos_z[i])
            var qx = _m_cam_quat_x[i]
            var qy = _m_cam_quat_y[i]
            var qz = _m_cam_quat_z[i]
            var qw = _m_cam_quat_w[i]
            var look = _rcd_rotate_by_quat(qx, qy, qz, qw, 0.0, 0.0, -1.0)
            var up_v = _rcd_rotate_by_quat(qx, qy, qz, qw, 0.0, 1.0, 0.0)
            var target = _RVec3(
                eye.x + look[0], eye.y + look[1], eye.z + look[2]
            )
            cameras.append(
                Camera3D(
                    eye=eye,
                    target=target,
                    up=_RVec3(up_v[0], up_v[1], up_v[2]),
                    fov=_m_cam_fovy[i],
                    aspect=Float64(width) / Float64(height),
                    near=Float64(0.1),
                    far=Float64(100.0),
                    screen_width=width,
                    screen_height=height,
                )
            )
        return cameras^

    @staticmethod
    def setup_camera_modes(rf: RenderFields) raises -> List[Int]:
        """MJCF `mode` -> the renderer's own encoding.

        Renderer: 0 = TRACKCOM (translate to follow the torso), 1 = FIXED,
        2 = TARGETBODY (re-aim at a body every frame). MJCF: 0 fixed, 1 track,
        2 trackcom, 3 targetbody, 4 targetbodycom.

        ⚠ targetbody USED TO COLLAPSE INTO TRACKCOM, which is a different
        behaviour entirely: trackcom moves the camera and keeps its
        orientation, targetbody holds the camera still and turns it. cartpole's
        `lookatcart` is the one model that asks for it.
        """
        # Mojo 1.0: `Array` is not `ImplicitlyCopyable`, so a comptime array
        # indexed at runtime must be materialized. Hoisted here so each array
        # is copied once per call rather than once per access in the loops.
        ref _m_cam_mode = rf.cam_mode

        var modes = List[Int]()
        for i in range(len(rf.cam_fovy)):
            var xml_mode = _m_cam_mode[i]
            if xml_mode == 0:
                modes.append(1)  # fixed
            elif xml_mode == 3 or xml_mode == 4:
                modes.append(2)  # targetbody / targetbodycom
            else:
                modes.append(0)  # track / trackcom
        return modes^

    @staticmethod
    def get_camera_target_bodies(rf: RenderFields) -> List[Int]:
        """Body index each camera aims at, or -1. Parallel to `setup_cameras`."""
        # Mojo 1.0: `Array` is not `ImplicitlyCopyable`, so a comptime array
        # indexed at runtime must be materialized. Hoisted here so each array
        # is copied once per call rather than once per access in the loops.
        ref _m_cam_target_body = rf.cam_target_body

        var out = List[Int]()
        for i in range(len(rf.cam_fovy)):
            out.append(_m_cam_target_body[i])
        return out^

    @staticmethod
    def get_skybox_colors(rf: RenderFields) -> List[Float64]:
        """Return [top_r, top_g, top_b, bottom_r, bottom_g, bottom_b] from the
        first skybox/gradient texture, or an empty list if none exists."""
        # Mojo 1.0: `Array` is not `ImplicitlyCopyable`, so a comptime array
        # indexed at runtime must be materialized. Hoisted here so each array
        # is copied once per call rather than once per access in the loops.
        ref _m_tex_builtin = rf.tex_builtin
        ref _m_tex_rgb1_b = rf.tex_rgb1_b
        ref _m_tex_rgb1_g = rf.tex_rgb1_g
        ref _m_tex_rgb1_r = rf.tex_rgb1_r
        ref _m_tex_rgb2_b = rf.tex_rgb2_b
        ref _m_tex_rgb2_g = rf.tex_rgb2_g
        ref _m_tex_rgb2_r = rf.tex_rgb2_r
        ref _m_tex_type = rf.tex_type

        # TEX_SKYBOX=1, TEX_BUILTIN_GRADIENT=1
        for i in range(rf.ntex):
            # ⚠⚠ `== 1` MEANT SKYBOX HERE AND MEANS 2D IN THE RUNTIME
            # RECORD. The two parsers numbered `tex_type` differently and
            # NEITHER matched MuJoCo's mjtTexture (2d=0/cube=1/skybox=2):
            # comptime was 2d=0/skybox=1/cube=3, `flat_model` is
            # skybox=0/2d=1/cube=2. `RenderFields` carries `flat_model`'s, so
            # this had to be renamed rather than repointed — a literal 1 here
            # would now select every 2D texture as the skybox.
            if (
                _m_tex_type[i] == TEX_SKYBOX
                or _m_tex_builtin[i] == TEX_BUILTIN_GRADIENT
            ):
                var result = List[Float64]()
                result.append(_m_tex_rgb1_r[i])
                result.append(_m_tex_rgb1_g[i])
                result.append(_m_tex_rgb1_b[i])
                result.append(_m_tex_rgb2_r[i])
                result.append(_m_tex_rgb2_g[i])
                result.append(_m_tex_rgb2_b[i])
                return result^
        return List[Float64]()

    @staticmethod
    def get_skybox_mark(rf: RenderFields) -> List[Float64]:
        """Return [kind, r, g, b, density] for the skybox's `mark`, else empty.

        Only `mark="random"` (kind 3) means anything to the renderer: MuJoCo
        bakes random dots into the generated skybox texture, which over a dark
        gradient is a starfield. dm_control's `common/skybox.xml` asks for
        exactly that (`mark="random" markrgb="1 1 1"`), and a plain two-colour
        gradient has no way to show it — hence "the stars aren't showing".
        `edge` and `cross` mark 2D textures, not the sky, so they are ignored
        here rather than approximated.
        """
        # Mojo 1.0: `Array` is not `ImplicitlyCopyable`, so a comptime array
        # indexed at runtime must be materialized. Hoisted here so each array
        # is copied once per call rather than once per access in the loops.
        ref _m_tex_builtin = rf.tex_builtin
        ref _m_tex_mark = rf.tex_mark
        ref _m_tex_markrgb_b = rf.tex_markrgb_b
        ref _m_tex_markrgb_g = rf.tex_markrgb_g
        ref _m_tex_markrgb_r = rf.tex_markrgb_r
        ref _m_tex_random = rf.tex_random
        ref _m_tex_type = rf.tex_type

        for i in range(rf.ntex):
            # ⚠⚠ `== 1` MEANT SKYBOX HERE AND MEANS 2D IN THE RUNTIME
            # RECORD. The two parsers numbered `tex_type` differently and
            # NEITHER matched MuJoCo's mjtTexture (2d=0/cube=1/skybox=2):
            # comptime was 2d=0/skybox=1/cube=3, `flat_model` is
            # skybox=0/2d=1/cube=2. `RenderFields` carries `flat_model`'s, so
            # this had to be renamed rather than repointed — a literal 1 here
            # would now select every 2D texture as the skybox.
            if (
                _m_tex_type[i] == TEX_SKYBOX
                or _m_tex_builtin[i] == TEX_BUILTIN_GRADIENT
            ):
                var result = List[Float64]()
                result.append(Float64(_m_tex_mark[i]))
                result.append(_m_tex_markrgb_r[i])
                result.append(_m_tex_markrgb_g[i])
                result.append(_m_tex_markrgb_b[i])
                result.append(_m_tex_random[i])
                return result^
        return List[Float64]()

    @staticmethod
    def get_checker_colors(rf: RenderFields) -> List[Float64]:
        """Return [r, g, b] of the checker texture's secondary (light square) colour,
        or an empty list if no checker texture is found."""
        # Mojo 1.0: `Array` is not `ImplicitlyCopyable`, so a comptime array
        # indexed at runtime must be materialized. Hoisted here so each array
        # is copied once per call rather than once per access in the loops.
        ref _m_tex_builtin = rf.tex_builtin
        ref _m_tex_rgb2_b = rf.tex_rgb2_b
        ref _m_tex_rgb2_g = rf.tex_rgb2_g
        ref _m_tex_rgb2_r = rf.tex_rgb2_r

        # TEX_BUILTIN_CHECKER=2
        for i in range(rf.ntex):
            if _m_tex_builtin[i] == TEX_BUILTIN_CHECKER:
                var result = List[Float64]()
                result.append(_m_tex_rgb2_r[i])
                result.append(_m_tex_rgb2_g[i])
                result.append(_m_tex_rgb2_b[i])
                return result^
        return List[Float64]()

    @staticmethod
    def get_ground_rgba(rf: RenderFields) -> List[Float64]:
        """Return [r, g, b] of the first plane geom's rgba color,
        or empty list if no plane geom exists."""
        # Mojo 1.0: `Array` is not `ImplicitlyCopyable`, so a comptime array
        # indexed at runtime must be materialized. Hoisted here so each array
        # is copied once per call rather than once per access in the loops.
        ref _m_geom_rgba_b = rf.geom_rgba_b
        ref _m_geom_rgba_g = rf.geom_rgba_g
        ref _m_geom_rgba_r = rf.geom_rgba_r
        ref _m_geom_type = rf.geom_type

        for i in range(len(rf.geom_type)):
            if _m_geom_type[i] == 0:  # GEOM_PLANE
                var result = List[Float64]()
                result.append(_m_geom_rgba_r[i])
                result.append(_m_geom_rgba_g[i])
                result.append(_m_geom_rgba_b[i])
                return result^
        return List[Float64]()

    @staticmethod
    def get_visual_settings(rf: RenderFields) -> List[Float64]:
        """Return [znear, fogstart, fogend, shadowsize, hl_r, hl_g, hl_b, has_headlight]."""
        var result = List[Float64]()
        result.append(rf.vis_znear)
        result.append(rf.vis_fogstart)
        result.append(rf.vis_fogend)
        result.append(Float64(rf.vis_shadowsize))
        result.append(rf.vis_headlight_ambient_r)
        result.append(rf.vis_headlight_ambient_g)
        result.append(rf.vis_headlight_ambient_b)
        result.append(Float64(1.0) if rf.vis_has_headlight else Float64(0.0))
        return result^

    @staticmethod
    def render_spatial_tendons(
        rf: RenderFields,
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
    ) raises:
        """Draw `<spatial>` tendons as a chain of thin capsules.

        This is the only thing that makes ball_in_cup's string visible: the
        tendon is load-bearing in the physics (an inextensible 30 cm link
        between the ball's site and the cup's) but the renderer had no notion
        of tendons at all, so the ball appeared to fly free.

        Sites are stored in the body frame, so each endpoint is the body's
        world pose composed with the site's local offset — the same transform
        `render_body_geoms` applies to a geom.

        ⚠ STRAIGHT SEGMENTS ONLY. MuJoCo routes a spatial tendon around
        `<geom>` wrapping objects; those children are not recorded, so a
        wrapping tendon would be drawn as the chord it wraps around rather
        than the path it takes. ball_in_cup has none, which is why this is
        enough for it and would not be for, say, a pulley.
        """
        # Mojo 1.0: `Array` is not `ImplicitlyCopyable`, so a comptime array
        # indexed at runtime must be materialized. Hoisted here so each array
        # is copied once per call rather than once per access in the loops.
        ref _m_site_body_id = rf.site_body_id
        ref _m_site_pos_x = rf.site_pos_x
        ref _m_site_pos_y = rf.site_pos_y
        ref _m_site_pos_z = rf.site_pos_z
        ref _m_sten_nsite = rf.sten_nsite
        ref _m_sten_rgba_b = rf.sten_rgba_b
        ref _m_sten_rgba_g = rf.sten_rgba_g
        ref _m_sten_rgba_r = rf.sten_rgba_r
        ref _m_sten_sites = rf.sten_sites
        ref _m_sten_width = rf.sten_width

        var base = 0
        for t in range(rf.nsten):
            var n = _m_sten_nsite[t]
            var radius = _m_sten_width[t]
            var col = Color(
                UInt8(_m_sten_rgba_r[t] * 255),
                UInt8(_m_sten_rgba_g[t] * 255),
                UInt8(_m_sten_rgba_b[t] * 255),
                255,
            )
            for k in range(n - 1):
                var ia = _m_sten_sites[base + k]
                var ib = _m_sten_sites[base + k + 1]
                if ia < 0 or ib < 0:
                    continue
                var ba = _m_site_body_id[ia]
                var bb = _m_site_body_id[ib]
                if ba >= len(positions) or bb >= len(positions):
                    continue
                var pa = positions[ba] + quaternions[ba].rotate_vec(
                    _RVec3(
                        _m_site_pos_x[ia],
                        _m_site_pos_y[ia],
                        _m_site_pos_z[ia],
                    )
                )
                var pb = positions[bb] + quaternions[bb].rotate_vec(
                    _RVec3(
                        _m_site_pos_x[ib],
                        _m_site_pos_y[ib],
                        _m_site_pos_z[ib],
                    )
                )
                var seg = pb - pa
                var length = seg.length()
                if length < 1e-9:
                    continue
                # `draw_capsule` takes a centre, an orientation and a half
                # height about its local +Z, so the segment becomes a rotation
                # of +Z onto its direction.
                var orient = _RQuat.from_two_vectors(
                    _RVec3(0.0, 0.0, 1.0), seg
                )
                renderer.draw_capsule(
                    center=_RVec3(
                        (pa.x + pb.x) * 0.5,
                        (pa.y + pb.y) * 0.5,
                        (pa.z + pb.z) * 0.5,
                    ),
                    orientation=orient,
                    radius=radius,
                    half_height=length * 0.5,
                    axis=2,
                    color=col,
                )
            base += n

    @staticmethod
    def raw_xml_text() raises -> String:
        """The model's MJCF **as authored** — from `xml_path`, else `xml`.

        ⚠⚠ NOT WHAT THE PARSER SEES, AND ALMOST CERTAINLY NOT WHAT YOU WANT.
        `full_parser` knows neither `<include>` nor `<attach>` nor `<frame>`
        and ASSUMES they are already gone (`full_parser.mojo`'s meshdir note
        says so outright: "`expand_mjcf` has already rebased every `file=` it
        spliced in"). `xml_text()` below is the composed document, and is the
        only thing that should ever reach a parser or a renderer.

        It exists so that the composition rule is written ONCE — `xml_text()`
        is literally `expand_mjcf(raw_xml_text(), asset_base_dir())`, rather
        than the expansion being repeated in each branch of the `comptime if`
        below. Its one caller is the P1a gate
        (`tests/physics3d/test_expand_identity.mojo`), which compares these two
        methods against each other: unchanged for a model that composes
        nothing, changed for one that does.

        ⚠ BOTH CARRY DEFAULTS ONLY BECAUSE THEY MUST STAY LAST in the
        parameter list — inserting mid-list silently shifts every positional
        instantiation, the trap `NPAIR` already documents. Supplying NEITHER
        is an error, and raises rather than letting `open("")` produce a
        mystery.
        """
        comptime if Self.xml_path != "":
            with open(Self.xml_path, "r") as f:
                return f.read()
        elif Self.xml != "":
            return String(Self.xml)
        else:
            raise Error(
                "physics3d: this ModelDefFromXML has neither `xml_path` nor"
                " `xml`. A shipped model wants `xml_path` (the MJCF is a file"
                " on disk since phase 1b); an inline test fixture wants `xml`."
            )

    @staticmethod
    def xml_text() raises -> String:
        """The model's MJCF, **composed** — `<include>`, `<attach>`, `<frame>`
        expanded. What every parser and renderer on this path is handed.

        ## ⚠⚠ WHY THE EXPANSION IS HERE — P1a, 2026-09-02

        `expand_mjcf` had exactly ONE caller, `runtime_load.read_model_source`,
        so the two ways into this parser disagreed about what a model IS:

            parse_model_runtime(path)      → expanded   (CPU only)
            ModelDefFromXML[xml_path=path] → raw        (the only leg with a GPU)

        A composed scene therefore loaded CPU-only, which is not a property of
        the GPU decision (kernels stay comptime, and the dims here are
        hand-supplied comptime parameters — a composed model can perfectly well
        be one). It was an accident of where the expander got called. That is
        what blocked `noeira/tasks/` from ever putting a composed family scene
        on the batched path; see `docs/TASK_LAYER_IMPLEMENTATION.md` Gap A.

        ⚠ THE FAILURE MODE THIS PREVENTS IS SILENT AND ENORMOUS, and the tree
        has already paid for it once, one layer over. `runtime_load.mojo`'s own
        header records what "`resolve_includes` alone, no `<attach>`" measured
        as: `humanoid100` 5 bodies where MuJoCo says 117, `100_humanoids` ZERO
        where MuJoCo says 1601. A model missing most of itself, with nothing
        said about it.

        ⚠ EXPANSION IS THE IDENTITY ON A FLAT MODEL, and that is asserted, not
        assumed — `tests/physics3d/test_expand_identity.mojo` gates every
        shipped model byte-for-byte. Verified 2026-09-02: NO shipped `.xml` and
        NO inline fixture in this tree contains a real `<frame>`, `<attach>` or
        `<include>` (a boundary-aware search; the 14 assets that look like they
        do are `<framepos>`/`<framequat>` SENSORS, which `_find_tag` correctly
        declines to match). So this commit changes no shipped model — it
        changes what the NEXT one is allowed to be.

        ⚠ COSTS THREE EARLY-RETURN SCANS on a flat model: `resolve_includes`
        returns the input unchanged when it finds no `<include`, `expand_attach`
        likewise, and `expand_frames` likewise. Against `parse_xml_full`'s
        measured 0.42 ms (cartpole) / 8.96 ms (dog) this is noise.

        ⚠ READS THE FILE ON EVERY CALL, deliberately not cached — unchanged
        from before, and for the reason phase 1b gave: a cache would need
        invalidating the moment "edit a model without a rebuild" is exercised.
        """
        return expand_mjcf(Self.raw_xml_text(), Self.asset_base_dir())

    @staticmethod
    def asset_base_dir() -> String:
        """The directory relative asset paths resolve against.

        MuJoCo's rule is "the directory of the model file", so this is
        `dirname(xml_path)`. An INLINE fixture has no model file and returns
        "", which `parse_xml_full` reads as "resolve against the process CWD".
        See §10.5 decision 1.
        """
        comptime if Self.xml_path == "":
            return String("")
        var p = String(Self.xml_path)
        var cut = p.rfind("/")
        if cut <= 0:
            return String("")
        return String(p[byte=0:cut])

    @staticmethod
    def resolve_asset(path: String) -> String:
        """Join a relative asset path onto `asset_base_dir()`.

        ⚠ `render_skin` NEEDS THIS AND `parse_xml_full` CANNOT DO IT FOR IT.
        The skin chain — `<skin file= material=>` -> `<material texture=>` ->
        `<texture file=>` — is walked by string search over the raw MJCF, not
        through the parsed record, so those two paths bypass the resolution
        `parse_xml_full` applies to `mesh_asset_files` and `TextureData.file`.
        Without this they would come out model-file-relative and un-based, and
        dog's skin would silently fail to load — the one model with a `<skin>`,
        and nothing gates rendering visually.

        ⚠ Absolute paths escape, as everywhere else.
        """
        return _resolve_asset(Self.asset_base_dir(), path)

    @staticmethod
    def default_ctrl_range() raises -> Tuple[Float64, Float64]:
        """The ROOT `<default>`'s motor ctrlrange — the model-wide SUMMARY.

        ⚠ NOT THE CLAMP, and knowingly wrong on models that set their ranges
        per actuator or per default class. `ctrl_min_at` / `ctrl_max_at` are
        the per-actuator answer and are what `apply_actions` uses.

        ⚠ RE-PARSES. `Phyics3dEnv.action_low/action_high` do NOT call this —
        they read `MODEL_META_IDX_CTRL_MIN/_CTRL_MAX` off the model they
        already hold. This exists for callers that have a model DEF and no
        built model, which in practice means tests.

        Was `comptime CTRL_MIN/CTRL_MAX` until phase 1b, computed by a
        comptime scan of the MJCF — the last such scan in this struct, and a
        comptime reader of the XML is what pins a model to a `String` in Mojo
        source.
        """
        var fmd = parse_xml_full(Self.xml_text(), Self.asset_base_dir())
        return (fmd.default_motor_ctrl_min, fmd.default_motor_ctrl_max)

    @staticmethod
    def make_render_fields() raises -> RenderFields:
        """`parse_xml_full` → `build_render_fields`, once per renderer.

        The runtime replacement for `comptime _rcd = parse_xml_render_data(
        Self.xml)`. That ran in the comptime interpreter and cost build time;
        this runs when a window opens and costs a parse.
        """
        return build_render_fields(
            parse_xml_full(Self.xml_text(), Self.asset_base_dir()),
            Self.xml_text(),
            Self.asset_base_dir(),
        )

    @staticmethod
    def render_ground_geoms(
        rf: RenderFields,
        mut renderer: Renderer3D,
        torso_x: Float64,
        follow: Bool,
        visual_radius_scale: Float64,
    ) raises:
        """Draw plane geoms (body_id=0): the ground as a grid, walls as slabs.

        ⚠ NOT EVERY PLANE IS A FLOOR. This used to send every plane geom to
        `draw_ground_grid` at its own `pos_z`, dropping its x/y position and
        its orientation entirely — so manipulator's four planes (floor at z=0,
        two 45° walls at z=.283, and a `background` at z=.5) came out as four
        stacked horizontal grids, the topmost ABOVE the arm's base. Read as a
        picture that is the arm hanging under a ceiling, which is exactly how
        it was reported: "things are below the ground".

        A plane is treated as the ground when it is UNROTATED; anything tilted
        or vertical is drawn as a thin oriented box of its declared
        half-extents. The ground keeps the grid path because that is what
        carries the infinite extent, the texture repeat and the reflection
        pass — a slab cannot stand in for it.

        ⚠ ORIENTATION DECIDES, POSITION MUST NOT. This test also required the
        plane to sit at x=y=0, and that was a REGRESSION (introduced with the
        oriented-plane fix, `8f248290`): the running tracks of cheetah
        (`pos="98 0 0"`), hopper (`pos="48 0 0"`) and walker (`pos="248 0 0"`)
        are unrotated floors that are merely OFFSET, so they fell to the slab
        path and rendered as flat white rectangles — `draw_box` takes a single
        colour and cannot carry the `grid` material's texture. The position
        test bought nothing even for the case it was added for: every wall and
        backdrop in manipulator, stacker, point_mass and reacher is rotated, so
        `upright` alone already excludes all of them. Verified by enumerating
        every plane geom in the suite — exactly those three change class.

        ⚠ This is the RENDER half of a defect whose PHYSICS half is still open:
        the plane narrow phase also assumes every plane is a horizontal floor
        at its origin's z with normal (0,0,1). Fixing the picture does not make
        an inclined plane collide correctly.
        """
        # Mojo 1.0: `Array` is not `ImplicitlyCopyable`, so a comptime array
        # indexed at runtime must be materialized. Hoisted here so each array
        # is copied once per call rather than once per access in the loops.
        ref _m_geom_body_id = rf.geom_body_id
        ref _m_geom_half_x = rf.geom_half_x
        ref _m_geom_half_y = rf.geom_half_y
        ref _m_geom_material_id = rf.geom_material_id
        ref _m_geom_pos_x = rf.geom_pos_x
        ref _m_geom_pos_y = rf.geom_pos_y
        ref _m_geom_pos_z = rf.geom_pos_z
        ref _m_geom_quat_w = rf.geom_quat_w
        ref _m_geom_quat_x = rf.geom_quat_x
        ref _m_geom_quat_y = rf.geom_quat_y
        ref _m_geom_quat_z = rf.geom_quat_z
        ref _m_geom_radius = rf.geom_radius
        ref _m_geom_rgba_a = rf.geom_rgba_a
        ref _m_geom_rgba_b = rf.geom_rgba_b
        ref _m_geom_rgba_g = rf.geom_rgba_g
        ref _m_geom_rgba_r = rf.geom_rgba_r
        ref _m_geom_type = rf.geom_type
        ref _m_mat_rgba_a = rf.mat_rgba_a
        ref _m_mat_rgba_b = rf.mat_rgba_b
        ref _m_mat_rgba_g = rf.mat_rgba_g
        ref _m_mat_rgba_r = rf.mat_rgba_r
        ref _m_mat_tex_id = rf.mat_tex_id
        ref _m_mat_texrepeat_u = rf.mat_texrepeat_u
        ref _m_mat_texrepeat_v = rf.mat_texrepeat_v

        # GEOM_PLANE=0
        var has_plane = False
        var max_body_radius = Float64(0.0)
        for j in range(len(rf.geom_type)):
            if _m_geom_body_id[j] > 0 and _m_geom_radius[j] > max_body_radius:
                max_body_radius = _m_geom_radius[j]
        for i in range(len(rf.geom_type)):
            if _m_geom_type[i] == 0:  # PLANE
                var pqx = _m_geom_quat_x[i]
                var pqy = _m_geom_quat_y[i]
                var pqz = _m_geom_quat_z[i]
                var pqw = _m_geom_quat_w[i]
                var upright = (
                    pqx == 0.0 and pqy == 0.0 and pqz == 0.0 and pqw == 1.0
                )
                var hx = _m_geom_half_x[i]
                var hy = _m_geom_half_y[i]
                if not upright and hx > 0.0 and hy > 0.0:
                    # A wall, a ramp or a backdrop. Half-extents come straight
                    # from MJCF `size="x y spacing"`; a zero there means the
                    # plane is INFINITE along that axis and there is no slab to
                    # draw, so it falls through to the grid path instead.
                    var wall_r = Float32(_m_geom_rgba_r[i])
                    var wall_g = Float32(_m_geom_rgba_g[i])
                    var wall_b = Float32(_m_geom_rgba_b[i])
                    var wall_a = Float32(_m_geom_rgba_a[i])
                    var wmid = _m_geom_material_id[i]
                    if wmid >= 0 and wmid < len(rf.mat_rgba_r):
                        wall_r = Float32(_m_mat_rgba_r[wmid])
                        wall_g = Float32(_m_mat_rgba_g[wmid])
                        wall_b = Float32(_m_mat_rgba_b[wmid])
                        wall_a = Float32(_m_mat_rgba_a[wmid])
                    if wall_a >= 0.99:
                        renderer.draw_box(
                            center=_RVec3(
                                _m_geom_pos_x[i],
                                _m_geom_pos_y[i],
                                _m_geom_pos_z[i],
                            ),
                            orientation=_RQuat(pqw, pqx, pqy, pqz),
                            half_extents=_RVec3(hx, hy, 0.002),
                            color=Color(
                                UInt8(wall_r * 255), UInt8(wall_g * 255),
                                UInt8(wall_b * 255), UInt8(wall_a * 255),
                            ),
                        )
                    continue
                has_plane = True
                var ground_offset = _m_geom_pos_z[i] - max_body_radius * (visual_radius_scale - 1.0)
                var grid_cx = torso_x if follow else Float64(0.0)
                # Resolve material → texture for this plane geom
                var tex_name = String("")
                var tex_file = String("")
                var texrep_u = Float64(1.0)
                var texrep_v = Float64(1.0)
                var texuni = False
                var g_cs = 0
                var g_shin = Float32(0.5)
                var g_spec = Float32(0.5)
                # MuJoCo's default material reflectance is 0: no material, no mirror.
                var g_refl = Float32(0.0)
                var mid = _m_geom_material_id[i]
                if mid >= 0 and mid < len(rf.mat_rgba_r):
                    texuni = rf.mat_texuniform[mid]
                    g_shin = Float32(rf.mat_shininess[mid])
                    g_spec = Float32(rf.mat_specular[mid])
                    g_refl = Float32(rf.mat_reflectance[mid])
                    var tex_id = _m_mat_tex_id[mid]
                    # ⚠ THIS WAS A `comptime for` OVER EVERY TEXTURE.
                    # Pulling a String out of `_rcd` needed
                    # `comptime _tn: String = ...` inside a comptime loop
                    # comparing `tex_id` to each index, because a comptime
                    # String STORE does not compile
                    # (see `has_skin`'s note). `rf.tex_names` is a runtime
                    # `List[String]`, so the whole scan is one index.
                    if tex_id >= 0 and tex_id < rf.ntex:
                        tex_name = rf.tex_names[tex_id]
                        tex_file = rf.tex_files[tex_id]
                        g_cs = rf.tex_colorspace[tex_id]
                    texrep_u = _m_mat_texrepeat_u[mid]
                    texrep_v = _m_mat_texrepeat_v[mid]
                renderer.draw_ground_grid(
                    grid_cx, height=ground_offset,
                    texture_name=tex_name, texture_path=tex_file,
                    texrepeat_u=texrep_u, texrepeat_v=texrep_v,
                    texuniform=texuni, plane_half_x=hx, plane_half_y=hy,
                    shininess=g_shin, specular=g_spec,
                    texture_colorspace=g_cs, reflectance=g_refl,
                )
        if not has_plane:
            # No ground plane defined in XML — skip ground rendering.
            # Models like InvertedPendulum intentionally omit the ground.
            pass

    @staticmethod
    def render_body_geoms(
        rf: RenderFields,
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
        visual_radius_scale: Float64,
    ) raises:
        """Draw body-attached geoms using parsed geometry + colour."""
        # Mojo 1.0: `Array` is not `ImplicitlyCopyable`, so a comptime array
        # indexed at runtime must be materialized. Hoisted here so each array
        # is copied once per call rather than once per access in the loops.
        ref _m_geom_body_id = rf.geom_body_id
        ref _m_geom_half_length = rf.geom_half_length
        ref _m_geom_half_x = rf.geom_half_x
        ref _m_geom_half_y = rf.geom_half_y
        ref _m_geom_half_z = rf.geom_half_z
        ref _m_geom_material_id = rf.geom_material_id
        ref _m_geom_mesh_id = rf.geom_mesh_id
        ref _m_geom_pos_x = rf.geom_pos_x
        ref _m_geom_pos_y = rf.geom_pos_y
        ref _m_geom_pos_z = rf.geom_pos_z
        ref _m_geom_quat_w = rf.geom_quat_w
        ref _m_geom_quat_x = rf.geom_quat_x
        ref _m_geom_quat_y = rf.geom_quat_y
        ref _m_geom_quat_z = rf.geom_quat_z
        ref _m_geom_radius = rf.geom_radius
        ref _m_geom_rgba_a = rf.geom_rgba_a
        ref _m_geom_rgba_b = rf.geom_rgba_b
        ref _m_geom_rgba_g = rf.geom_rgba_g
        ref _m_geom_rgba_r = rf.geom_rgba_r
        ref _m_geom_type = rf.geom_type
        ref _m_mat_reflectance = rf.mat_reflectance
        ref _m_mat_rgba_a = rf.mat_rgba_a
        ref _m_mat_rgba_b = rf.mat_rgba_b
        ref _m_mat_rgba_g = rf.mat_rgba_g
        ref _m_mat_rgba_r = rf.mat_rgba_r
        ref _m_mat_shininess = rf.mat_shininess
        ref _m_mat_specular = rf.mat_specular
        ref _m_mat_tex_id = rf.mat_tex_id

        # SPHERE=1, CAPSULE=2, BOX=3, CYLINDER=4, MESH=5
        for i in range(len(rf.geom_type)):
            var bid = _m_geom_body_id[i]
            if bid < 0:
                continue
            if bid >= len(positions):
                continue
            # Plane / group / alpha, ONE predicate shared with the picker and
            # the selection outline — see `body_geom_visible` for all three
            # rules and for the 490 geoms the old inline `alpha < 0.99` hid.
            if not body_geom_visible(rf, i):
                continue
            var body_pos = positions[bid]
            var body_quat = quaternions[bid]
            var gx = _m_geom_pos_x[i]
            var gy = _m_geom_pos_y[i]
            var gz = _m_geom_pos_z[i]
            var geom_pos: _RVec3
            if gx == 0.0 and gy == 0.0 and gz == 0.0:
                geom_pos = body_pos
            else:
                geom_pos = body_pos + body_quat.rotate_vec(_RVec3(gx, gy, gz))
            var gqx = _m_geom_quat_x[i]
            var gqy = _m_geom_quat_y[i]
            var gqz = _m_geom_quat_z[i]
            var gqw = _m_geom_quat_w[i]
            var geom_quat: _RQuat
            if gqx == 0.0 and gqy == 0.0 and gqz == 0.0 and gqw == 1.0:
                geom_quat = body_quat
            else:
                geom_quat = body_quat * _RQuat(gqw, gqx, gqy, gqz)
            var r = Float32(_m_geom_rgba_r[i])
            var g = Float32(_m_geom_rgba_g[i])
            var b = Float32(_m_geom_rgba_b[i])
            var a = Float32(_m_geom_rgba_a[i])
            var mid = _m_geom_material_id[i]
            if mid >= 0 and mid < len(rf.mat_rgba_r):
                r = Float32(_m_mat_rgba_r[mid])
                g = Float32(_m_mat_rgba_g[mid])
                b = Float32(_m_mat_rgba_b[mid])
                a = Float32(_m_mat_rgba_a[mid])
            var geom_color = Color(UInt8(r * 255), UInt8(g * 255), UInt8(b * 255), UInt8(a * 255))
            var shininess = Float32(0.5)
            var specular = Float32(0.5)
            var reflectance = Float32(0.0)
            if mid >= 0 and mid < len(rf.mat_rgba_r):
                shininess = Float32(_m_mat_shininess[mid])
                specular = Float32(_m_mat_specular[mid])
                reflectance = Float32(_m_mat_reflectance[mid])
            # Resolve material → texture chain for this geom
            var tex_name_str = String("")
            var tex_file_str = String("")
            # ⚠ THE MAPPING, TRANSCRIBED FROM `render_gl3.c:settexture`:
            #   * a 2D texture on a builtin shape uses the shape's UVs scaled
            #     by `texrepeat`, times the geom's `size[0]`/`size[1]` when
            #     `texuniform` — so the LIBERO walls (`texrepeat="3 3"
            #     texuniform="true"`, size 1.75 x 1.5) tile 5.25 x 4.5 times,
            #     not once;
            #   * a CUBE texture is looked up by object-space POSITION
            #     (cube-map texgen), scaled by `size` when `texuniform`;
            #   * a mesh with its own texcoords (LIBERO's `.msh`) uses them,
            #     scaled by `texrepeat`.
            # Both were "UVs, once" before, which drew the 3x3 plaster as one
            # stretched tile and the wood table as a single texel row.
            var rep_u = Float32(1.0)
            var rep_v = Float32(1.0)
            var tex_cube = False
            var tex_cs = 0
            var tex_scale = _RVec3(1.0, 1.0, 1.0)
            if mid >= 0 and mid < len(rf.mat_rgba_r):
                var tex_id = _m_mat_tex_id[mid]
                # Same collapse as in `render_ground_geoms` — one index
                # where a `comptime for` over every texture used to be.
                if tex_id >= 0 and tex_id < rf.ntex:
                    tex_name_str = rf.tex_names[tex_id]
                    tex_file_str = rf.tex_files[tex_id]
                    tex_cube = rf.tex_type[tex_id] == TEX_CUBE
                    tex_cs = rf.tex_colorspace[tex_id]
                var tr_u = rf.mat_texrepeat_u[mid]
                var tr_v = rf.mat_texrepeat_v[mid]
                rep_u = Float32(tr_u if tr_u > 0.0 else 1.0)
                rep_v = Float32(tr_v if tr_v > 0.0 else 1.0)
                if rf.mat_texuniform[mid]:
                    # MuJoCo's `geom->size` per type: box half-extents,
                    # sphere radius, capsule/cylinder (radius, half-length).
                    var s0 = Float64(0.0)
                    var s1 = Float64(0.0)
                    var s2 = Float64(0.0)
                    var gt0 = _m_geom_type[i]
                    if gt0 == 3 or gt0 == 6:
                        s0 = _m_geom_half_x[i]
                        s1 = _m_geom_half_y[i]
                        s2 = _m_geom_half_z[i]
                    elif gt0 == 1:
                        s0 = _m_geom_radius[i]
                        s1 = _m_geom_radius[i]
                        s2 = _m_geom_radius[i]
                    elif gt0 == 2 or gt0 == 4:
                        s0 = _m_geom_radius[i]
                        s1 = _m_geom_half_length[i]
                        s2 = _m_geom_half_length[i]
                    if s0 > 0.0:
                        rep_u *= Float32(s0)
                    if s1 > 0.0:
                        rep_v *= Float32(s1)
                    if s0 > 0.0 and s1 > 0.0 and s2 > 0.0:
                        tex_scale = _RVec3(s0, s1, s2)

            var gt = _m_geom_type[i]
            if gt == 2:  # CAPSULE
                renderer.draw_capsule(center=geom_pos, orientation=geom_quat,
                    radius=_m_geom_radius[i] * visual_radius_scale,
                    half_height=_m_geom_half_length[i], axis=2,
                    color=geom_color, shininess=shininess, specular=specular, reflectance=reflectance,
                    texture_name=tex_name_str, texture_path=tex_file_str,
                    texrepeat_u=rep_u, texrepeat_v=rep_v, tex_cube=tex_cube, texture_colorspace=tex_cs,
                    tex_scale=tex_scale)
            elif gt == 1:  # SPHERE
                renderer.draw_sphere(center=geom_pos,
                    radius=_m_geom_radius[i] * visual_radius_scale,
                    color=geom_color, shininess=shininess, specular=specular, reflectance=reflectance,
                    texture_name=tex_name_str, texture_path=tex_file_str,
                    texrepeat_u=rep_u, texrepeat_v=rep_v, tex_cube=tex_cube, texture_colorspace=tex_cs,
                    tex_scale=tex_scale)
            elif gt == 3:  # BOX
                renderer.draw_box(center=geom_pos, orientation=geom_quat,
                    half_extents=_RVec3(_m_geom_half_x[i], _m_geom_half_y[i], _m_geom_half_z[i]),
                    color=geom_color, shininess=shininess, specular=specular, reflectance=reflectance,
                    texture_name=tex_name_str, texture_path=tex_file_str,
                    texrepeat_u=rep_u, texrepeat_v=rep_v, tex_cube=tex_cube, texture_colorspace=tex_cs,
                    tex_scale=tex_scale)
            elif gt == 4:  # CYLINDER
                renderer.draw_cylinder(center=geom_pos, orientation=geom_quat,
                    radius=_m_geom_radius[i] * visual_radius_scale,
                    half_height=_m_geom_half_length[i], axis=2,
                    color=geom_color, shininess=shininess, specular=specular, reflectance=reflectance,
                    texture_name=tex_name_str, texture_path=tex_file_str,
                    texrepeat_u=rep_u, texrepeat_v=rep_v, tex_cube=tex_cube, texture_colorspace=tex_cs,
                    tex_scale=tex_scale)
            elif gt == 6:  # ELLIPSOID
                # ⚠ THIS BRANCH DID NOT EXIST until 2026-08-03, and there is no
                # trailing `else`, so every ellipsoid geom was silently skipped
                # — invisible in the renderer while colliding normally. It hit
                # quadruped (the torso), swimmer, fish and finger. Semi-axes
                # live in half_x/y/z, the same slots a box uses.
                renderer.draw_ellipsoid(center=geom_pos, orientation=geom_quat,
                    radii=_RVec3(
                        _m_geom_half_x[i] * visual_radius_scale,
                        _m_geom_half_y[i] * visual_radius_scale,
                        _m_geom_half_z[i] * visual_radius_scale,
                    ),
                    color=geom_color, shininess=shininess, specular=specular,
                    reflectance=reflectance)
            elif gt == 5:  # MESH
                var mid2 = _m_geom_mesh_id[i]
                # Draw mesh with optional texture
                # Same collapse again. ⚠ AND THE BOUNDS CHECK IS NEW: the
                # `comptime for` could only match an index it enumerated, so
                # a `mesh_id` outside the asset table silently drew NOTHING.
                # A runtime index would trap instead, which is the better
                # failure but only if it cannot happen — `build_render_fields`
                # resolves `geom_mesh_id` by filename against the same table,
                # so -1 (no mesh) is the only out-of-range value it can hold.
                if mid2 >= 0 and mid2 < rf.nmesh:
                    renderer.draw_mesh(
                        name=rf.mesh_names[mid2], file_path=rf.mesh_files[mid2],
                        center=geom_pos, orientation=geom_quat,
                        # ⚠ `<mesh scale>`, APPLIED AT DRAW TIME. The
                        # renderer caches one GPU upload per mesh NAME, so
                        # baking the scale into the vertices would give a
                        # mirrored pair a single shared (wrong) copy. As a
                        # draw transform each geom gets its own, and op3 —
                        # whose STLs are in millimetres — stops rendering
                        # 1000x oversized.
                        scale=_RVec3(
                            rf.geom_mesh_scale[i * 3 + 0],
                            rf.geom_mesh_scale[i * 3 + 1],
                            rf.geom_mesh_scale[i * 3 + 2],
                        ),
                        color=geom_color, shininess=shininess,
                        specular=specular, reflectance=reflectance,
                        texture_name=tex_name_str,
                        texture_path=tex_file_str,
                        texrepeat_u=rep_u, texrepeat_v=rep_v,
                        tex_cube=tex_cube, texture_colorspace=tex_cs,
                        # A cube lookup on a mesh reads its unscaled
                        # vertices; the `<mesh scale>` ratios are what
                        # MuJoCo's compiled (pre-scaled) mesh would carry.
                        tex_scale=_RVec3(
                            rf.geom_mesh_scale[i * 3 + 0],
                            rf.geom_mesh_scale[i * 3 + 1],
                            rf.geom_mesh_scale[i * 3 + 2],
                        ),
                    )

    @staticmethod
    def has_skin(rf: RenderFields) raises -> Bool:
        """Whether the model declares a `<skin>`.

        ⚠ A `find` ON THE XML, NOT A PARSED FLAG. Recording anything about the
        skin in the comptime render data does not compile (see the note above
        `MAX_COMPTIME_RENDER_GEOMS` in `xml_parser.mojo`) — but `find` never
        slices, so asking the question is safe even though storing the answer
        is not. Comptime-resolvable, so a model without a skin still compiles
        `render_skin` away to nothing.
        """
        return rf.xml_text.find("<skin") != -1

    @staticmethod
    def geom_group_at(rf: RenderFields, i: Int) -> Int:
        """MuJoCo's geom `group` for geom `i` — visibility, not a tag.

        Exposed so a test can count what `render_body_geoms` will skip; see the
        group note there.

        ⚠ TAKES `rf` LIKE EVERY OTHER RENDER HOOK, and did not until now. The
        1a.5 repoint rewrote this body from `materialize[Self._rcd.geom_group]`
        to `rf.geom_group` without widening the SIGNATURE, so it referenced an
        `rf` that was never in scope. It did not compile from `84d61724` until
        this commit, and nothing said so: the only caller is
        `tests/dm_control/test_dog_skin.mojo`, which was not in the set swept
        after that repoint. `feedback_confirm_the_code_under_test_actually_runs`
        — an uncalled generic is uncompiled code.
        """
        return rf.geom_group[i]

    @staticmethod
    def render_skin(
        rf: RenderFields,
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
    ) raises:
        """Deform and draw the model's `<skin>`, if it has one.

        The skin is the ENVELOPE MuJoCo actually shows for a model like dog —
        the geoms it draws alongside are only those in groups 0-2. See
        `render_body_geoms` for the group rule.
        """
        # ⚠ THE COMPTIME `Self.xml.find("<skin") == -1` GUARD IS GONE. It
        # elided this whole body at compile time for the models without a
        # skin, which is every model but dog — but the runtime `st == -1`
        # check three lines down already returns for exactly those, so the
        # two were never doing different work. What the comptime version DID
        # do was read the MJCF at compile time, and phase 1b's whole point is
        # that nothing may.
        # ⚠ THE WHOLE ASSET CHAIN IS WALKED AT RUNTIME, from the XML this
        # `rf` now carries (`RenderFields.xml_text`). `<skin file= material=>` ->
        # `<material texture=>` -> `<texture file=>` is three attribute
        # reads, and doing any of them in the comptime interpreter is a
        # compile failure the moment it hits. See `body_names`.
        var src = rf.xml_text
        var st = src.find("<skin")
        if st == -1:
            return
        var se = src.find(">", st)
        if se == -1:
            return

        var skin_file = _resolve_asset(rf.asset_base_dir,
            _attr_between(src, st, se, "file")
        )
        if skin_file.byte_length() == 0:
            return
        var skin_mat = _attr_between(src, st, se, "material")

        var tex_name = String("")
        var tex_file = String("")
        if skin_mat.byte_length() > 0:
            var want_tex = String("")
            var mp = 0
            while True:
                var mt = src.find("<material", mp)
                if mt == -1:
                    break
                var me = src.find(">", mt)
                if me == -1:
                    break
                if _attr_between(src, mt, me, "name") == skin_mat:
                    want_tex = _attr_between(src, mt, me, "texture")
                    break
                mp = me + 1
            if want_tex.byte_length() > 0:
                var tp = 0
                while True:
                    var tt = src.find("<texture", tp)
                    if tt == -1:
                        break
                    var te = src.find(">", tt)
                    if te == -1:
                        break
                    if _attr_between(src, tt, te, "name") == want_tex:
                        tex_name = want_tex
                        tex_file = _resolve_asset(rf.asset_base_dir,
                            _attr_between(src, tt, te, "file")
                        )
                        break
                    tp = te + 1

        # Flatten the poses `skin_pose` wants: plain float arrays with
        # (w, x, y, z) quats, which is MuJoCo's order and `_RQuat`'s.
        var xpos = List[Float32]()
        var xquat = List[Float32]()
        for b in range(len(positions)):
            xpos.append(Float32(positions[b].x))
            xpos.append(Float32(positions[b].y))
            xpos.append(Float32(positions[b].z))
            xquat.append(Float32(quaternions[b].w))
            xquat.append(Float32(quaternions[b].x))
            xquat.append(Float32(quaternions[b].y))
            xquat.append(Float32(quaternions[b].z))

        renderer.draw_skin(
            name=skin_file,
            skn_path=skin_file,
            body_names=body_names_of(rf.xml_text),
            xpos=xpos,
            xquat=xquat,
            texture_name=tex_name,
            texture_path=tex_file,
        )

    @staticmethod
    def render_sites(
        rf: RenderFields,
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
    ) raises:
        """Draw all sites as small bright-green spheres (visual markers)."""
        # Mojo 1.0: `Array` is not `ImplicitlyCopyable`, so a comptime array
        # indexed at runtime must be materialized. Hoisted here so each array
        # is copied once per call rather than once per access in the loops.
        ref _m_site_body_id = rf.site_body_id
        ref _m_site_pos_x = rf.site_pos_x
        ref _m_site_pos_y = rf.site_pos_y
        ref _m_site_pos_z = rf.site_pos_z
        ref _m_site_size_0 = rf.site_size_0

        for i in range(len(rf.site_pos_x)):
            var sbid = _m_site_body_id[i]
            if sbid <= 0 or sbid >= len(positions):
                continue
            var body_pos = positions[sbid]
            var body_quat = quaternions[sbid]
            var sx = _m_site_pos_x[i]
            var sy = _m_site_pos_y[i]
            var sz = _m_site_pos_z[i]
            var site_world_pos: _RVec3
            if sx == 0.0 and sy == 0.0 and sz == 0.0:
                site_world_pos = body_pos
            else:
                site_world_pos = body_pos + body_quat.rotate_vec(_RVec3(sx, sy, sz))
            var radius = _m_site_size_0[i] if _m_site_size_0[i] > 0.0 else 0.005
            renderer.draw_sphere(
                center=site_world_pos,
                radius=radius,
                color=Color(0, 255, 0, 255),
                shininess=Float32(0.9),
                specular=Float32(0.9),
                reflectance=Float32(0.0),
            )


comptime RfOnlyModelDef = ModelDefFromXML[
    xml_path="", nbody=1, njoint=0, nq=0, nv=0, ngeom=0, nact=0,
]
"""THE RENDER HOOKS, ADDRESSABLE WITHOUT A MODEL — how one renderer draws any file.

All seventeen render hooks above are pure functions of `rf: RenderFields`:
none of them reads a `Self.` member any more (the counts became
`len(rf.…)`, the source text became `rf.xml_text`/`rf.asset_base_dir`, and
the two helpers that were members are now module-level). So which
instantiation you call them ON does not matter, and this one — a model with
no bodies, no geoms and no XML — exists so the studio can call them with no
model type at all.

⚠⚠ THE PURITY IS A PROPERTY, NOT A GUARANTEE, AND IT IS LINTED. Reintroduce
one `Self.NGEOM` into a hook and this alias silently draws ZERO geoms for
every runtime-loaded model — the exact "two model paths" failure §10 of
`docs/PHYSICS3D_STUDIO_PLAN.md` names as the top risk, and one that no
compile catches. `scripts/audit_render_hooks_are_rf_pure.py` fails the moment
a hook body mentions `Self.`; run it in CI, not by hand.

⚠ WHY AN ALIAS AND NOT A SEPARATE `RuntimeModelDef` STRUCT. A second struct
would be a second copy of ~900 lines of hook, which is the same risk with
worse odds. There is exactly one implementation, and this names it.

The physics side needs nothing equivalent: `Model[DTYPE, DynDims]` and the
integrators already take their dimensions as a runtime value.
"""

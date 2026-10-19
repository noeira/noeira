"""`ModelDefLike` — the trait every physics3d model definition implements.

Declares the compile-time dimensions, the CPU state hooks (fields-native:
`reset_data` / `extract_obs` / `enforce_limits` / `apply_actions` over
`Data`), the spec-direct fields model build (`init_fields`), the GPU
kernel delegates, and the render hooks. Sole implementer: `ModelDefFromXML`
(the spec-based `ModelDef` compositor was deleted at the G2 fields sunset;
the legacy CPU `Model`/`Data` build at G4).
"""

from noeira.render import Renderer3D, Light, Camera3D
from noeira.math3d import Vec3 as _Vec3G, Quat as _QuatG

from ..fields import Model, Data, SpecFields, Dims, DimsLike
from ..parser.render_fields import RenderFields
from ..gpu.constants import (
    MODEL_ACTUATOR_SIZE,
    MODEL_ACT_TENDON_SIZE,
    JLIM_SIZE,
    METADATA_SIZE,
    POSE_META_SIZE,
)

# GPU imports
from max.gpu.host import DeviceContext, DeviceBuffer
from noeira.core.tiles import Tile


comptime _RVec3 = _Vec3G[DType.float64]
comptime _RQuat = _QuatG[DType.float64]


trait ModelDefLike:
    """Trait for model definition types."""

    # === Dimensions ===
    comptime NQ: Int
    comptime NV: Int
    comptime NBODY: Int
    comptime NJOINT: Int
    comptime NGEOM: Int
    comptime MAX_EQUALITY: Int
    comptime CONE_TYPE: Int
    comptime MAX_CONTACTS: Int
    comptime MAX_TENDON: Int
    comptime NSITE: Int
    # `<sensor>` count and `sensordata` length — MuJoCo's `m->nsensor` and
    # `m->nsensordata`. ⚠ TWO NUMBERS, NOT ONE: a sensor is 1, 3, 4 or 6 wide
    # by type, so they differ on almost every model that declares any.
    #
    # ⚠ ON THE TRAIT FOR THE REASON `NPAIR` GIVES ABOVE. The env forwards
    # these into the integrator, and a dimension the trait does not expose is
    # one the env can quietly forget — the model would still build and
    # `sensordata` would simply stay empty. Declaring them here makes omitting
    # them a compile error.
    comptime NSENSOR: Int
    comptime NSENSORDATA: Int
    # MuJoCo `m->na`: ACTIVATION variables, not `nu`.
    comptime NA: Int
    comptime NA_F: Int
    comptime NEXCLUDE: Int
    # `<contact><pair>` count — sizes the predefined-pair table.
    #
    # ⚠ ON THE TRAIT FOR THE SAME REASON `MAX_CONDIM` IS (see below). The env
    # forwards these dims into the integrator, and a dimension the trait does
    # not expose is one the env can quietly forget: the model would still
    # build, the pair table would be empty, and the declared collisions would
    # simply not happen. Declaring it here makes omitting it a compile error.
    comptime NPAIR: Int
    # Largest `condim` in the model — sizes the PYRAMIDAL edge list at
    # `2*(MAX_CONDIM-1)` rows per contact.
    #
    # ⚠ ON THE TRAIT ON PURPOSE, not just on `ModelDefFromXML`. `Phyics3dEnv`
    # reads it through this trait to forward into the integrator, and until
    # 2026-08-03 it did not forward it at all — so every env silently ran the
    # pyramidal builder at condim 3 regardless of its model, and quadruped
    # `fetch` and dog both lost their condim-6 friction rows. Declaring it here
    # is what makes omitting it a compile error rather than a silent downgrade.
    comptime MAX_CONDIM: Int
    # `<option noslip_iterations>`; 0 disables `mj_solNoSlip`. Same reasoning —
    # the env has to forward it, so the trait has to expose it.
    comptime NOSLIP_ITER: Int
    comptime OBS_DIM: Int
    comptime ACTION_DIM: Int
    comptime TIMESTEP: Float64
    # ⚠⚠ THESE ARE A SUMMARY, NOT THE CLAMP, AND THEY LIE ON SOME MODELS.
    # A single pair for the whole model, read by
    # `_xml_default_motor_ctrlrange` from a ROOT `<default><motor ctrlrange>`
    # — only that, and only a `<motor>` tag. A model that sets its ranges per
    # actuator or per default CLASS falls back to (-1, 1) here while
    # `apply_actions` correctly clamps each actuator to its own range.
    #
    # MEASURED against dm_control's `action_spec`:
    #   reach_site_features  advertised (-1, 1); real +/-0.6283 x3,
    #                        +/-0.8378 x3, +/-5.0 x3  -> every one wrong
    #   quadruped walk       advertised (-1, 1); real lo in [-1, -0.8],
    #                        hi in [0.8, 1.1]        -> already live
    #   dog / humanoid / walker / cheetah / finger    -> uniform +/-1, fine
    #
    # Use `ctrl_min_at` / `ctrl_max_at` for anything that matters. These two
    # stay because `BoxContinuousActionEnv.action_low/high` are scalars by
    # contract, and changing THAT changes the action scaling of every shipped
    # env — a behaviour change that needs its own before/after, not a
    # side-effect of a bug fix.
    # ⚠ NO LONGER HERE. They were `comptime CTRL_MIN/CTRL_MAX: Float64`,
    # computed by `_xml_default_motor_ctrlrange[Self.xml]()` — the LAST
    # comptime reader of the MJCF inside `ModelDefFromXML`, and a comptime
    # reader of the XML is precisely what pins a model to a `String` in Mojo
    # source (§10.2: the interpreter cannot `open()` a file).
    #
    # The values are unchanged, wrong models included. They live in
    # `Model.meta` at `MODEL_META_IDX_CTRL_MIN` / `_CTRL_MAX` now, written by
    # `build_model_fields_from_flat` from `FlatModelDef`, and read by
    # `Phyics3dEnv.action_low/action_high` off the model the env already
    # holds — no re-parse. Verified identical on all 56 shipped models
    # (`test_ctrl_range_source`).

        # === Actuation records (phases 1a.2 / 1a.4) ===
    #
    # `NACT` is MuJoCo's `m->nu`; `NACT_F`/`NTEN_F` are the same numbers
    # floored at 1, which is the STORAGE capacity. Both are needed and they
    # are not interchangeable: `build_spec_fields` checks the real count, and
    # a zero-extent tensor aborts at bind.
    #
    # ⚠⚠ DECLARED WITH THE OTHER DIMENSIONS, NOT NEXT TO THE METHODS THAT USE
    # THEM. These lived beside `apply_actions` and `reset_data` — declared
    # EARLIER in the trait — could not see them: `Self.NACT` stayed an
    # unresolved reference on the trait side while the implementation expanded
    # it to `nact`, and conformance failed with a two-page "no candidates have
    # type" diff whose real content was one unbound parameter. A trait member
    # must be declared before the first signature that mentions it.
    comptime NACT: Int
    comptime NACT_F: Int
    comptime NTEN_F: Int
    comptime NKEY: Int
    # `NQ` floored at 1 — the `qpos0` operand's length. A zero-extent tensor
    # aborts at bind, so a model with no qpos still gets a 1-element buffer.
    comptime NQ_F: Int

# === Components ===
    # comptime BODIES: BodiesLike
    # comptime JOINTS: JointsLike
    # comptime GEOMS: GeomsLike
    # comptime ACTUATORS: ActuatorsLike
    # comptime DEFAULTS: ModelDefaultsLike
    # comptime LIGHTS: LightsLike
    # comptime TEXTURES: TexturesLike
    # comptime MATERIALS: MaterialsLike
    # comptime CAMERAS: CamerasLike

    # === CPU: state hooks (fields-native; G2) ===
    @staticmethod
    def reset_data[DTYPE: DType, D: DimsLike, D2: DimsLike](
        sf: SpecFields[DTYPE, D],
        mut d: Data[DTYPE, D2, 1],
    ):
        ...

    @staticmethod
    def extract_obs[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        mut obs: List[Scalar[DTYPE]],
    ):
        ...

    @staticmethod
    def enforce_limits[DTYPE: DType, D: DimsLike, D2: DimsLike](
        sf: SpecFields[DTYPE, D],
        mut d: Data[DTYPE, D2, 1],
    ):
        ...

    @staticmethod
    def init_spec_fields[DTYPE: DType, D: DimsLike](
        ctx: DeviceContext,
        mut sf: SpecFields[DTYPE, D],
    ) raises:
        """Build + upload the actuation record tensors (`SpecFields`), the
        runtime replacement for the comptime `_acd` actuator arrays."""
        ...

    @staticmethod
    def apply_actions[
        DTYPE: DType, D: DimsLike, D2: DimsLike, NORMALIZED: Bool = False
    ](
        sf: SpecFields[DTYPE, D],
        mut d: Data[DTYPE, D2, 1],
        actions: List[Float64],
        mut act: List[Scalar[DTYPE]],
    ):
        ...

    @staticmethod
    def ctrl_min_at[DTYPE: DType, D: DimsLike](sf: SpecFields[DTYPE, D], i: Int) -> Float64:
        """Lower `ctrlrange` bound of actuator `i` — MuJoCo's
        `actuator_ctrlrange[i][0]`, and what `apply_actions` actually clamps
        against.

        The per-actuator answer that `CTRL_MIN` cannot give. Resolved through
        the element attribute, then `class=`, then the root default, so a
        model that keeps its ranges in a default class (quadruped does, and so
        does the Jaco manipulation task) reports them correctly here.
        """
        ...

    @staticmethod
    def ctrl_max_at[DTYPE: DType, D: DimsLike](sf: SpecFields[DTYPE, D], i: Int) -> Float64:
        """Upper `ctrlrange` bound of actuator `i`. See `ctrl_min_at`."""
        ...

    # === CPU: Float getters (can't use Float64 as comptime in traits) ===

    # === Fields-native model build (spec-direct; G4) ===
    @staticmethod
    def init_fields[DTYPE: DType, D: DimsLike](
        ctx: DeviceContext,
        mut mf: Model[DTYPE, D],
    ) raises:
        """Build the Model record tensors + fields-native invweight0
        and upload. Implemented spec-direct by `ModelDefFromXML`
        (parse_xml_full -> fields_build.build_model_fields_from_flat)."""
        ...

    # === GPU: kernel delegates (per-field tensors; G5) ===
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
        joint_limits: Tile[DTYPE, Self.NJOINT * JLIM_SIZE],
        dof_actdamp: Tile[DTYPE, BATCH_SIZE, Self.NV],
        actdamp_act: Tile[DTYPE, BATCH_SIZE, Self.NACT_F],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
    ) raises:
        """⚠ `joint_limits` added for `jnt_actfrcrange` — MuJoCo's SECOND
        actuator force clamp (`engine_forward.c:477`), which is per-JOINT on
        the ACCUMULATED `qfrc_actuator` and is NOT the actuator's own
        `forcerange`. It is in the TRAIT and not just the implementation for
        the usual reason: the CPU and GPU paths must compute the same force
        from the same action, and a trait that does not carry the argument
        cannot make that a requirement.

        ⚠ `acts`/`act_tendons` added 2026-08-15 with phase 1a.3. Every
        actuator value used to be a comptime literal baked into a fully
        unrolled loop; they are now loads from the SAME `SpecFields` records
        the CPU `apply_actions` reads, so the two targets cannot drift.

        ⚠ `qpos`/`qvel`/`act` added 2026-08-07 with the blocker-G fix. The GPU
        actuator path used to apply `gear * ctrl` to ONE dof; it now mirrors
        `apply_actions` term for term, which means walking the transmission
        triples and — for position servos and fixed-tendon springs — reading
        the joint state. See `ModelDefFromXML.apply_actions_kernel_gpu`."""
        ...

    # === GPU inline: Per-env methods (called from inside GPU kernels) ===
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
        ...

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
        ...

    @staticmethod
    def make_render_fields() raises -> RenderFields:
        """Build this model's render records. Called ONCE per renderer.

        A factory on the trait rather than `xml` as a trait member: the
        renderer needs the RECORDS, not the source text, and `ModelDefLike`
        deliberately exposes dimensions and behaviour rather than the MJCF it
        came from. It is also the seam phase 1b needs — when the XML moves to
        a file on disk, this is the one place that changes.
        """
        ...

    @staticmethod
    def render_ground_geoms(
        rf: RenderFields,
        mut renderer: Renderer3D,
        torso_x: Float64,
        follow: Bool,
        visual_radius_scale: Float64,
    ) raises:
        ...

    @staticmethod
    def render_body_geoms(
        rf: RenderFields,
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
        visual_radius_scale: Float64,
    ) raises:
        ...

    @staticmethod
    def render_spatial_tendons(
        rf: RenderFields,
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
    ) raises:
        """Draw `<spatial>` tendons. Defaults to nothing, since most models
        have none and a model definition predating this should still build."""
        pass

    @staticmethod
    def render_skin(
        rf: RenderFields,
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
    ) raises:
        """Deform and draw a MuJoCo `<skin>`. Defaults to nothing.

        A DEFAULT for the same reason the tendons have one: exactly one ported
        model (dog) declares a skin, and a model definition written before this
        existed must still conform. `ModelDefFromXML` overrides it and compiles
        the body away when its XML has no `<skin>`, so the cost to everyone
        else is zero.
        """
        pass

    @staticmethod
    def setup_lights(rf: RenderFields) raises -> List[Light]:
        ...

    @staticmethod
    def setup_cameras(rf: RenderFields, width: Int, height: Int) raises -> List[Camera3D]:
        ...

    @staticmethod
    def setup_camera_modes(rf: RenderFields) raises -> List[Int]:
        ...

    @staticmethod
    def get_camera_target_bodies(rf: RenderFields) -> List[Int]:
        """Body each camera aims at (mode="targetbody"), or -1 for none.

        Defaults to empty: a model definition with no cameras, or one written
        before targetbody was honoured, simply gets no re-aiming."""
        return List[Int]()

    @staticmethod
    def get_skybox_colors(rf: RenderFields) -> List[Float64]:
        ...

    @staticmethod
    def get_skybox_mark(rf: RenderFields) -> List[Float64]:
        """[kind, r, g, b, density] for the skybox `mark`, or empty for none.

        Defaults to empty so a model definition that predates the starfield
        keeps compiling — the renderer treats an empty list, and any kind other
        than 3 (`random`), as "no stars".
        """
        return List[Float64]()

    @staticmethod
    def get_checker_colors(rf: RenderFields) -> List[Float64]:
        ...

    @staticmethod
    def get_ground_rgba(rf: RenderFields) -> List[Float64]:
        ...

    @staticmethod
    def get_visual_settings(rf: RenderFields) -> List[Float64]:
        """Return visual settings: [znear, fogstart, fogend, shadowsize,
        headlight_r, headlight_g, headlight_b, has_headlight].
        Empty list = use defaults."""
        ...

    @staticmethod
    def render_sites(
        rf: RenderFields,
        mut renderer: Renderer3D,
        positions: List[_RVec3],
        quaternions: List[_RQuat],
    ) raises:
        ...


# The spec-based `ModelDef` compositor struct (Bodies/Joints/Geoms/Actuators
# cascade) was deleted at the G2 fields sunset: it had ZERO instantiation
# sites — every env model is a `ModelDefFromXML`, the trait's only
# implementer. The `ModelDefLike` trait above is the surviving surface.

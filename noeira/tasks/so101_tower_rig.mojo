"""The so101_tower rig as a vision policy sees it — pixels and units, written once.

    from noeira.tasks.so101_tower_rig import (
        TowerRenderer, make_tower_renderer, tower_cameras, pack_camera_u8,
        So101TowerUnits,
    )

Two tools produce or consume the student's observation and must agree to the
byte: `examples/so101/tower_demo_rerender.mojo` renders the TRAINING frames
from recorded states, and `examples/so101/tower_act_eval.mojo` renders the
frames the student acts on in the closed loop. A picture or a unit spelled
twice is `_a_rule_written_inline_twice_drifts`, and its failure here would be
silent: a student that scores well on the store and worse in the loop, with
nothing raising. So both import these.

## The pixels

The env's own model shape (`TOWER_MD`), 320x240 (`act/config`'s
`SO101_IMG_W` / `SO101_IMG_H`, the real import's working resolution), 4x
MSAA, groups 0+2 (props + the arm's and stand's visual meshes; group 3 is the
collision set), the preview's background, overhead in SLOT 0 and wrist in
SLOT 1 (the real dataset's sorted video-key order). A frame is packed uint8
CHW, top row first, `round(x * 255)` clamped — `pack_camera_u8`.

⚠ THE POSE IS HOST FK OF `qpos`, NOT THE ENV'S DEVICE `xpos`. The tower
config leaves `SYNC_FK_AFTER_STEP` off, so after a step the env's derived
poses describe the state one substep BEFORE the integrated `qpos`. The store
was rendered from `forward_kinematics` of the recorded `qpos`; the loop does
the same into its own `Data`, so frame t is the picture of state t in both.

## The units (`So101TowerUnits`)

The store's `qpos` / `action` are LeRobot's: the five body joints in DEGREES,
the gripper 0..100 by FRACTION of its range (`robot/so101/sim_map.mojo`, sign
+1). The env's action is each joint TARGET normalised onto the actuator's
`ctrlrange` to [-1, 1]. Both directions of both maps live here.

⚠⚠ THE JOINT ZERO IS A CHOICE THE STORE AND THE EVAL MUST SHARE
(`RIG_JOINT_ZERO_*`). `none`: body degrees = radians x 180/pi, the lerobot
reference's map, what every store before 2026-09-23 was written with.
`follower`: the so101-tower follower's MEASURED zero
(`sim_map.tower_follower_zero_deg`: pan -10.7, lift -3.6, elbow -7.3, roll
+5.0 deg — the roll since 2026-09-25 — and the gripper on its measured line,
-9.25 + 130.46 deg over LeRobot 0..100, not the fraction of the model's 110;
a `follower` store written before then is `follower-v1`), so
a store's degrees are the ones the REAL arm reports at that pose and a
student trained on it commands the real arm where the sim arm went. Either
is self-consistent in the sim; a store written under one and evaluated under
the other puts the arm ~10 degrees off in pan, and nothing raises. The
images do not depend on it (they are rendered from the sim state).
"""

from std.math import pi

from max.gpu.host import DeviceContext
from noeira.math3d import Vec3 as Vec3Generic

from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.physics3d.fields import Model, actuator_column
from noeira.physics3d.gpu.constants import ACT_IDX_CTRL_MAX, ACT_IDX_CTRL_MIN
from noeira.physics3d.parser.flat_model import FlatModelDef
from noeira.physics3d.raytrace import BatchedCameraRenderer
from noeira.physics3d.raytrace.visual import build_visual_model, VisualModel
from noeira.physics3d.raytrace.visual_records import (
    VIS_GEOM_APPEARANCE, VIS_LIGHT_WORDS, APP_IDX_R, APP_IDX_G, APP_IDX_B,
    APP_IDX_REFLECT,
    LIGHT_IDX_AMBIENT_R, LIGHT_IDX_DIFFUSE_R,
)
from noeira.physics3d.raytrace.randomize import geom_labels, VisualRandomizer
from noeira.tasks.family_config import So101TowerConfig
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.robot.so101.sim_map import (
    tower_follower_zero_deg, TOWER_FOLLOWER_GRIPPER_ZERO_RAD,
    tower_follower_gripper_span_rad,
)
from noeira.utils.fmt import fixed


comptime RIG_DT = DType.float32
"""The device's type — the batched env's `Data` is float32."""
comptime TOWER_MD = Phyics3dEnv[So101TowerModel, So101TowerConfig, RIG_DT, False].MD
"""⚠ THE ENV'S OWN MODEL SHAPE, so the renderer is the kernel a
render-in-the-loop eval over the batched env instantiates."""
comptime RIG_CAM_W = 320
comptime RIG_CAM_H = 240
comptime RIG_N_CAMS = 2
comptime RIG_NPIX = RIG_CAM_W * RIG_CAM_H
comptime RIG_CAM_ELEMS = 3 * RIG_NPIX
comptime RIG_IMG_ELEMS = RIG_N_CAMS * RIG_CAM_ELEMS
comptime RIG_SAMPLES = 4
"""MuJoCo's `offsamples`: 4x MSAA. One ray per pixel aliases the jaw and the
brick's edges, the two things the student must localise."""
comptime RIG_VISUAL_GROUP_MASK: Int = (1 << 0) | (1 << 2)
comptime RIG_BACKGROUND_R: Float64 = 0.82
comptime RIG_BACKGROUND_G: Float64 = 0.86
comptime RIG_BACKGROUND_B: Float64 = 0.90
comptime RIG_ACT = 6
comptime RIG_GRIPPER = 5
comptime RIG_DR_TARGET = (0.32, 0.0, 0.0)
"""What `--dr`'s extra spot lights aim at: the desk mat's centre
(`scenes/so101_tower.xml`, the `desk_mat` frame)."""

comptime TowerRendererSized[
    LANES: Int, W: Int, H: Int, SAMPLES: Int, REFLECT: Bool = False
] = BatchedCameraRenderer[
    RIG_DT, TOWER_MD, LANES, W, H, False, REFLECT, SAMPLES
]
"""The rig's renderer at another resolution / sample count — what a pixel-RL
observation would be (e.g. 128x128, one sample). The store and the eval use
`TowerRenderer`, the rig's own pixels.

⚠ `REFLECT=False` BY DEFAULT: the tower scene has no reflective geom, so the
reflection pass never runs, but as `True` its code — a second full trace per
pixel — is still compiled into the kernel and holds registers. The pictures
are the same bytes either way; `make_tower_renderer` REFUSES a visual set with
a reflective geom when it is compiled out (the DR never writes reflectance,
`randomize.mojo`'s NEVER list), so a mirror added to the scene later fails at
construction instead of silently rendering as matte."""

comptime TowerRenderer[LANES: Int] = TowerRendererSized[
    LANES, RIG_CAM_W, RIG_CAM_H, RIG_SAMPLES
]


def _camera(names: List[String], suffix: String) raises -> Int:
    for i in range(len(names)):
        if String(names[i]).endswith(suffix):
            return i
    raise Error("so101 tower rig: no camera named *" + suffix + " in the scene")


def tower_cameras(fmd: FlatModelDef) raises -> List[Int]:
    """`[overhead, wrist]` — slot 0, slot 1."""
    var cams = List[Int]()
    cams.append(_camera(fmd.camera_names, String("overhead_cam")))
    cams.append(_camera(fmd.camera_names, String("wrist_cam")))
    return cams^


comptime RIG_BACKDROP_GROUP: Int = 4
"""The desk asset's BACKDROP geoms (visual only: the floor past the real
desk's +y edge, the desk's -y extension, the wall — `desk_mat.xml`). Drawn by
the calibrated look only."""
comptime RIG_MARKER_GROUP: Int = 5
"""The ArUco marker taped on the back of the wrist camera's plate (DICT_4X4_50
id 7, 30 mm — `bake_so_arm101_tower.py` step 8): the extrinsics target, which
stays on the rig and is in the overhead view whenever the jaw points down.
Visual only. Drawn by the calibrated look only."""


def rig_visual_group_mask(look: String) -> Int:
    """Groups 0+2 (props, the arm's and stand's visual meshes), plus the
    backdrop (group 4) and the gripper's ArUco marker (group 5) under the
    calibrated look. `legacy` keeps the pre-calibration set, so its stores
    stay byte-identical."""
    if look == RIG_LOOK_LEGACY:
        return RIG_VISUAL_GROUP_MASK
    return RIG_VISUAL_GROUP_MASK | (1 << RIG_BACKDROP_GROUP) | (1 << RIG_MARKER_GROUP)


def rig_background(look: String = RIG_LOOK_CALIBRATED) -> Vec3Generic[RIG_DT]:
    """What a ray that hits nothing shows. Legacy: the preview's blue-grey
    (.82,.86,.90). Calibrated: the room the wrist camera sees past the wall's
    top (median 151,155,144 on the real wrist frames, 2026-09-24)."""
    if look == RIG_LOOK_LEGACY:
        return Vec3Generic[RIG_DT](
            Scalar[RIG_DT](RIG_BACKGROUND_R), Scalar[RIG_DT](RIG_BACKGROUND_G),
            Scalar[RIG_DT](RIG_BACKGROUND_B),
        )
    return Vec3Generic[RIG_DT](
        Scalar[RIG_DT](0.59), Scalar[RIG_DT](0.61), Scalar[RIG_DT](0.56),
    )


comptime RIG_DR_OVERHEAD_CAM_SCALE = (0.4, 0.375, 0.25)
"""The overhead camera's (position, rotation, fovy) multipliers on the DR
preset's camera ranges: under `full`, +-4 mm / +-0.75 deg / +-0.75 deg. It is
CALIBRATED (`so101_tower_stand.xml`, 62 marker captures, 2026-09-25):
leave-one-out sd 2-3 mm, the fit's variants within 0.3-0.6 deg; the margin
is for the tower being bumped between sessions."""
comptime RIG_DR_WRIST_CAM_SCALE = (0.4, 1.75, 0.25)
"""The wrist camera's: +-4 mm / +-3.5 deg / +-0.75 deg under `full`. Its
position is the mount's STL plus the measured lens depth (19.0 +-2 mm, bake
`PUPIL_ALONG_NORMAL_MM`); its ROTATION is WIDER than the preset's +-2 deg
because the real arm SAGS under load and the camera rides it: two camera
instruments (the marker captures; noeira-72's wrist-vs-overhead fit, 826
slow frames of 70 episodes) see the wrist camera pitched ~3-4 deg low in
working poses, while the joint zeros are right (a straight-edge across the
wrist reads flex +0.4 deg, and the real meshes at the folded rest allow no
extra fold of any pitch joint). Best fit: a lift-joint droop proportional to
reach, ~9 mm at the gripper at typical reach — beyond the servo encoders,
so FK cannot see it; DR covers it (2026-09-25).

FOVY, both: the real frames are undistorted to the sim's exact pinhole
through a 0.18 px rms fisheye calibration, so the preset's +-3 deg is not a
real uncertainty; +-0.75 deg is margin."""


def tower_camera_dr_describe() -> String:
    """The provenance words for the scales (a store drawn before 2026-09-25
    has none: its cameras took the preset's ranges)."""
    var o = RIG_DR_OVERHEAD_CAM_SCALE
    var w = RIG_DR_WRIST_CAM_SCALE
    return (
        String("camera range scales (pos rot fovy) overhead ") + String(o[0])
        + " " + String(o[1]) + " " + String(o[2]) + ", wrist " + String(w[0])
        + " " + String(w[1]) + " " + String(w[2])
    )


def scale_tower_camera_dr[
    DT: DType
](mut dr: VisualRandomizer[DT], fmd: FlatModelDef) raises:
    """Set the rig cameras' DR scales on a randomizer built over them — by
    NAME, whatever order the caller listed them in. Every listed camera
    must be one of the two."""
    for k in range(len(dr.cams)):
        var name = String(fmd.camera_names[dr.cams[k]])
        if name.endswith("overhead_cam"):
            var s = RIG_DR_OVERHEAD_CAM_SCALE
            dr.scale_camera(k, s[0], s[1], s[2])
        elif name.endswith("wrist_cam"):
            var s = RIG_DR_WRIST_CAM_SCALE
            dr.scale_camera(k, s[0], s[1], s[2])
        else:
            raise Error("scale_tower_camera_dr: '" + name + "' is not a rig camera")


def make_tower_model(ctx: DeviceContext) raises -> Model[RIG_DT, TOWER_MD]:
    """The rig's model, on the device — what the renderer reads."""
    var m = Model[RIG_DT, TOWER_MD]()
    So101TowerModel.init_fields[RIG_DT](ctx, m)
    m.upload_all(ctx)
    return m^


comptime RIG_LOOK_CALIBRATED = "calibrated"
comptime RIG_LOOK_LEGACY = "legacy"


def apply_tower_look[
    DTYPE: DType
](mut vis: VisualModel[DTYPE], fmd: FlatModelDef, look: String) raises:
    """The rig's LOOK — the lights and the surface colours the tracer draws.

    `calibrated` (the default) is the scene as composed: lights and albedos
    fitted to the rig's recorded frames (`so101_tower.family`, 2026-09-24).
    `legacy` puts back what every store before that was rendered with — the
    headlight at .45 ambient / .3 diffuse, the floor light's .7 sun, the desk
    at .93, the white arm parts at .92 (the camera mount's grey .92), the
    props' filament colours, and no backdrop or marker (the group mask) — so a
    student can be compared on the SAME demos under both looks, the arms
    differing in the look alone. It rewrites the tracer's tables only (no
    physics), and must run BEFORE a `VisualRandomizer` takes its base copy, so
    `--dr` jitters around whichever look was chosen.

    ⚠ THE STORE AND ITS EVAL MUST SHARE IT, like the joint zero: the rerender
    records it in the store's provenance line and `tower_act_eval` takes the
    same flag."""
    if look == RIG_LOOK_CALIBRATED:
        return
    if look != RIG_LOOK_LEGACY:
        raise Error(
            "so101 tower rig: look '" + look + "' — expected '"
            + RIG_LOOK_CALIBRATED + "' or '" + RIG_LOOK_LEGACY + "'"
        )
    if vis.nlight != 2:
        raise Error(
            "so101 tower rig: the legacy look expects the headlight and the"
            " floor's sun (2 light rows), the scene has " + String(vis.nlight)
        )
    # the lights: row 0 the headlight, row 1 the floor's directional sun
    for c in range(3):
        vis.lights.data[LIGHT_IDX_AMBIENT_R + c] = Scalar[DTYPE](0.45)
        vis.lights.data[LIGHT_IDX_DIFFUSE_R + c] = Scalar[DTYPE](0.3)
        vis.lights.data[VIS_LIGHT_WORDS + LIGHT_IDX_DIFFUSE_R + c] = Scalar[DTYPE](0.7)
    # the colours, by the geom's `<body>/<geom>` label, on its APPEARANCE
    # row: the shader colours a hit from `APP_IDX_R..B` (the parser resolves a
    # material's rgba into `geom_rgba`), the material row only carries its
    # specular / shininess / texture — so the material rows are not touched
    var labels = geom_labels(fmd)
    var desk = 0
    var white = 0
    var mount = 0
    var bowl = 0
    var brick = 0
    for k in range(vis.ngeom):
        var lab = labels[vis.src_geom[k]]
        var o = k * VIS_GEOM_APPEARANCE
        var r = Float64(vis.appearance.data[o + APP_IDX_R])
        var g = Float64(vis.appearance.data[o + APP_IDX_G])
        var b = Float64(vis.appearance.data[o + APP_IDX_B])
        var rgb = List[Float64]()
        if lab.startswith("desk_"):
            rgb = [0.93, 0.93, 0.91]
            desk += 1
        elif lab.startswith("robot_") and _near(r, 0.78) and _near(g, 0.78) and _near(b, 0.76):
            rgb = [0.92, 0.92, 0.90]
            white += 1
        elif lab.startswith("robot_") and _near(r, 0.78) and _near(g, 0.78) and _near(b, 0.78):
            # the wrist camera mount (bake step 6b): neutral 0.78 calibrated,
            # 0.92 grey before
            rgb = [0.92, 0.92, 0.92]
            mount += 1
        elif lab.startswith("bowl_") and _near(r, 1.0) and _near(g, 0.66) and _near(b, 0.09):
            rgb = [0.996, 0.776, 0.0]
            bowl += 1
        elif lab.startswith("brick_") and _near(r, 0.17) and _near(g, 0.474) and _near(b, 0.662):
            rgb = [0.0, 0.471, 0.749]
            brick += 1
        if len(rgb) == 3:
            vis.appearance.data[o + APP_IDX_R] = Scalar[DTYPE](rgb[0])
            vis.appearance.data[o + APP_IDX_G] = Scalar[DTYPE](rgb[1])
            vis.appearance.data[o + APP_IDX_B] = Scalar[DTYPE](rgb[2])
    if desk == 0 or white == 0 or mount != 1 or bowl == 0 or brick == 0:
        raise Error(
            "so101 tower rig: the legacy look found desk " + String(desk)
            + ", white arm " + String(white) + ", camera mount " + String(mount)
            + ", bowl " + String(bowl)
            + ", brick " + String(brick) + " geoms — the calibrated scene"
            " changed under it"
        )


@always_inline
def _near(x: Float64, y: Float64) -> Bool:
    return abs(x - y) < 1e-4


def make_tower_renderer[
    LANES: Int,
    W: Int = RIG_CAM_W,
    H: Int = RIG_CAM_H,
    SAMPLES: Int = RIG_SAMPLES,
    REFLECT: Bool = False,
](
    ctx: DeviceContext, fmd: FlatModelDef, mut m: Model[RIG_DT, TOWER_MD],
    look: String = RIG_LOOK_CALIBRATED,
    bvh_sah: Bool = True,
) raises -> TowerRendererSized[LANES, W, H, SAMPLES, REFLECT]:
    """The renderer with the rig's visual set, look and background, camera
    slot 0. The defaults are the rig's pixels (`TowerRenderer`).

    `bvh_sah=False` builds the mesh trees with the reference's median split
    instead of SAH — the same pictures, slower; for a benchmark's A/B."""
    var cams = tower_cameras(fmd)
    var r = TowerRendererSized[LANES, W, H, SAMPLES, REFLECT](ctx, m, cams[0])
    var vis = build_visual_model[RIG_DT, TOWER_MD](
        fmd, m, group_mask=rig_visual_group_mask(look), bvh_sah=bvh_sah
    )
    apply_tower_look(vis, fmd, look)
    comptime if not REFLECT:
        for g in range(vis.ngeom + vis.ncond):
            if vis.appearance.data[g * VIS_GEOM_APPEARANCE + APP_IDX_REFLECT] != 0:
                raise Error(
                    "make_tower_renderer: visual geom " + String(g)
                    + " is reflective and this renderer has the reflection"
                    " pass compiled out (REFLECT=False) — it would render"
                    " matte. Build it with REFLECT=True."
                )
    r.set_visual(ctx, vis^)
    r.background = rig_background(look)
    return r^


@always_inline
def rig_byte(x: Float64) -> UInt8:
    """A tracer float in [0, 1] to the byte the store holds."""
    var v = Int(x * 255.0 + 0.5)
    if v < 0:
        v = 0
    if v > 255:
        v = 255
    return UInt8(v)


def pack_camera_u8(
    src: Pointer[Scalar[RIG_DT], MutAnyOrigin],
    lane: Int,
    dst: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    dst_off: Int,
) -> Bool:
    """Lane `lane` of the renderer's interleaved `rgb` (copied to the host,
    `mptr(host_buffer.unsafe_ptr())`) ->
    uint8 CHW at `dst + dst_off`. Returns True when every byte is the same
    (a flat picture: a camera inside a mesh, a pose that never arrived)."""
    var base = lane * RIG_NPIX * 3
    var first = rig_byte(Float64(src[unsafe_offset=base]))
    var all_same = True
    for q in range(RIG_NPIX):
        for c in range(3):
            var b = rig_byte(Float64(src[unsafe_offset = base + q * 3 + c]))
            dst[unsafe_offset = dst_off + c * RIG_NPIX + q] = b
            if b != first:
                all_same = False
    return all_same


comptime RIG_JOINT_ZERO_NONE = "none"
comptime RIG_JOINT_ZERO_FOLLOWER = "follower"
comptime RIG_JOINT_ZERO_FOLLOWER_V1 = "follower-v1"
"""`follower` as it was until 2026-09-25: the same body zero with the roll
at 0 (before the marker captures measured +5.0), and the gripper by FRACTION
of the model's range (before the measured line, `sim_map.
tower_follower_gripper_span_rad`). Every `follower` store and checkpoint
written before then (the 5cc24a9b / ce84f398 series among them) reads in
these units — name it to reproduce their numbers."""


struct So101TowerUnits(Copyable, Movable):
    """LeRobot units <-> the model's joint values <-> the env's action word.

    `lo` / `hi` are the actuators' `ctrlrange`, read from the MODEL (for a
    `<position>` servo it is the joint's range) — never a copy of the numbers.
    `zero_rad` is the joint-zero choice (the module header):
    `model_rad = deg2rad(lerobot_deg) + zero_rad`.
    """

    var lo: List[Float64]
    var hi: List[Float64]
    var zero_rad: List[Float64]
    var joint_zero: String
    var grip_lo: Float64
    """The gripper's model angle at LeRobot 0; `grip_span` over 0..100.
    `none` / `follower-v1`: the ctrlrange (the fraction map); `follower`: the
    measured line."""
    var grip_span: Float64

    def __init__(out self, joint_zero: String = RIG_JOINT_ZERO_NONE) raises:
        if (
            joint_zero != RIG_JOINT_ZERO_NONE
            and joint_zero != RIG_JOINT_ZERO_FOLLOWER
            and joint_zero != RIG_JOINT_ZERO_FOLLOWER_V1
        ):
            raise Error(
                "So101TowerUnits: joint zero '" + joint_zero + "' — expected '"
                + RIG_JOINT_ZERO_NONE + "', '" + RIG_JOINT_ZERO_FOLLOWER
                + "' or '" + RIG_JOINT_ZERO_FOLLOWER_V1 + "'"
            )
        var sf = So101TowerModel.make_spec_fields[DType.float64]()
        var lo_col = actuator_column(sf, ACT_IDX_CTRL_MIN, RIG_ACT)
        var hi_col = actuator_column(sf, ACT_IDX_CTRL_MAX, RIG_ACT)
        self.lo = List[Float64]()
        self.hi = List[Float64]()
        self.zero_rad = List[Float64]()
        self.joint_zero = joint_zero
        self.grip_lo = Float64(lo_col[RIG_GRIPPER])
        self.grip_span = Float64(hi_col[RIG_GRIPPER]) - Float64(lo_col[RIG_GRIPPER])
        if joint_zero == RIG_JOINT_ZERO_FOLLOWER:
            self.grip_lo = TOWER_FOLLOWER_GRIPPER_ZERO_RAD
            self.grip_span = tower_follower_gripper_span_rad()
        for k in range(RIG_ACT):
            self.lo.append(Float64(lo_col[k]))
            self.hi.append(Float64(hi_col[k]))
            var z = 0.0
            if joint_zero != RIG_JOINT_ZERO_NONE and k != RIG_GRIPPER:
                z = tower_follower_zero_deg(k) * pi / 180.0
                if joint_zero == RIG_JOINT_ZERO_FOLLOWER_V1 and k == 4:
                    z = 0.0  # the roll, before 2026-09-25
            self.zero_rad.append(z)

    def describe(self) -> String:
        """The provenance words: the choice and, when not `none`, its degrees."""
        if self.joint_zero == RIG_JOINT_ZERO_NONE:
            return String("joint zero none")
        var out = String("joint zero ") + self.joint_zero + " ("
        for k in range(RIG_ACT):
            if k == RIG_GRIPPER:
                continue
            if k > 0:
                out += " "
            out += fixed(self.zero_rad[k] * 180.0 / pi, 2)
        out += " deg"
        if self.joint_zero == RIG_JOINT_ZERO_FOLLOWER:
            out += "; gripper " + fixed(self.grip_lo * 180.0 / pi, 2) + " + "
            out += fixed(self.grip_span * 180.0 / pi, 2) + " deg over 0..100"
        return out + ")"

    def joint_to_lerobot(self, k: Int, q: Float64) -> Float64:
        """Radians (body) / the gripper hinge -> degrees / 0..100."""
        if k == RIG_GRIPPER:
            return 100.0 * (q - self.grip_lo) / self.grip_span
        return (q - self.zero_rad[k]) * 180.0 / pi

    def lerobot_to_joint(self, k: Int, v: Float64) -> Float64:
        if k == RIG_GRIPPER:
            return self.grip_lo + v / 100.0 * self.grip_span
        return v * pi / 180.0 + self.zero_rad[k]

    def action_to_joint(self, k: Int, a: Float64) -> Float64:
        """The env's normalised action word -> the joint target it commands."""
        return self.lo[k] + (a + 1.0) * 0.5 * (self.hi[k] - self.lo[k])

    def joint_to_action(self, k: Int, q: Float64) -> Float64:
        """A joint target -> the env's action word, NOT clamped (the caller
        counts saturations before clamping)."""
        return 2.0 * (q - self.lo[k]) / (self.hi[k] - self.lo[k]) - 1.0

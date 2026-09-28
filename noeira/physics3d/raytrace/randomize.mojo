"""Render-time domain randomization — the tracer's appearance, re-drawn per launch.

`noeira-docs/DOMAIN_RANDOMIZATION_PLAN.md` level L2. The tracer's appearance
tables (`VisualModel`) and the camera rows of `Model` are model constants,
shared by every lane of a launch (`visual.mojo`'s header). They are also plain
host lists with a device copy, so between two launches they can be rewritten
and re-uploaded. A store rendered OFFLINE from recorded states needs nothing
more: a row rendered under draw `k` is a training sample under look `k`.

    var dr = VisualRandomizer[DT](cfg, groups, r.vis, m, geom_labels(fmd),
                                  cams, background, target)
    for launch ...:
        var bg = dr.apply(draw, r.vis, m)      # host tables <- base + draw
        dr.upload(ctx, r.vis, m)               # the four tables, in place
        r.background = bg
        r.render(...)

## What a draw touches, and what it never touches

- LIGHTS: the scene's own lights (direction tilted about the base, colour
  scaled and tinted, an ambient term), the headlight (row 0: scaled, or off),
  and up to `max_extra_lights` SPOT lights added above a target point and
  aimed at it (`directional="false"` is a spot here, never a point light).
- COLOURS: per GROUP of geoms named by prefix — one hue / saturation / value /
  tint draw per group, applied to every material AND per-geom rgba the group
  uses, so a group keeps its internal contrast (the arm's white parts stay
  lighter than its black servos). Specular and shininess are scaled per group.
- CAMERAS: position, a small rotation and `fovy` (+ its baked tangent) around
  the BASE row of each listed camera.
- BACKGROUND: returned, for the render argument.

NEVER: geometry, groups, `src_geom`, the reflectance column (the one-mirror
rule's output) or anything the solver reads. So `seg` is identical across
draws whenever the camera knobs are zero — the gate
`tests/physics3d/test_domain_randomization_gpu.mojo` holds that.

## Draws are a function of (seed, draw index), and jitter is never accumulated

`apply` first restores the base copies taken at construction, then writes
`base + jitter(seed, draw)`. So `apply(k)` gives the same tables whatever was
applied before, and a store row can be re-rendered from its recorded draw
index. The stream is a host splitmix64 keyed by `seed ^ DR_SALT` and the draw
index — separate from the placement sampler's Philox, so changing the
appearance seed never moves a brick.

A config with `enabled = False` makes `apply` a restore and nothing else: the
tables are the build's, byte for byte (gate G2a).
"""

from std.math import cos, sin, sqrt, tan, pi

from max.gpu.host import DeviceContext
from noeira.math3d import Vec3 as Vec3Generic

from ..fields import Model
from ..fields.dims import DimsLike
from ..gpu.constants import (
    MODEL_CAM_SIZE,
    MAX_GPU_CAMERAS,
    CAM_IDX_POS_X,
    CAM_IDX_POS_Y,
    CAM_IDX_POS_Z,
    CAM_IDX_QUAT_X,
    CAM_IDX_QUAT_Y,
    CAM_IDX_QUAT_Z,
    CAM_IDX_QUAT_W,
    CAM_IDX_FOVY,
    CAM_IDX_TAN_HALF_FOVY,
)
from ..parser.flat_model import FlatModelDef
from .visual import VisualModel
from .visual_records import *


comptime DR_SALT: UInt64 = 0xD0_3A1D_5EED_0002


# ── config ───────────────────────────────────────────────────────────────────


@fieldwise_init
struct ColourJitter(Copyable, ImplicitlyCopyable, Movable):
    """Ranges for one group. Hue in degrees, the rest as fractions."""

    var hue_deg: Float64
    """Hue rotation `U(-h, h)`. Meaningless on a grey surface — see `tint`."""
    var sat: Float64
    """Saturation scale `U(1-s, 1+s)`."""
    var val: Float64
    """Value (brightness) scale `U(1-v, 1+v)`."""
    var tint: Float64
    """Per-channel multiplicative `U(1-t, 1+t)` — what gives a WHITE surface a
    colour cast, which hue rotation cannot."""
    var spec: Float64
    """Specular and shininess scale `U(1-p, 1+p)`, clamped to [0, 1]."""

    @staticmethod
    def none() -> Self:
        return Self(0.0, 0.0, 0.0, 0.0, 0.0)

    def scaled(self, k: Float64) -> Self:
        return Self(
            self.hue_deg * k, self.sat * k, self.val * k, self.tint * k,
            self.spec * k,
        )


@fieldwise_init
struct SurfaceGroup(Copyable, Movable):
    """Every visual geom whose LABEL (`geom_labels`: `"<body>/<geom>"`)
    starts with one of `prefixes`. `""` matches nothing.

    ⚠ BY BODY, NOT BY GEOM NAME. On the tower scene 21 of the 25 visible
    geoms are UNNAMED (every arm and stand mesh), so a geom-name prefix
    matched nothing; `<attach prefix=...>` names the BODIES."""

    var name: String
    var prefixes: List[String]
    var jitter: ColourJitter


@fieldwise_init
struct DomainRandConfig(Copyable, ImplicitlyCopyable, Movable, Writable):
    """Every range of the L2 randomizer. `strength` scales the colour groups'
    own ranges, so a scene declares its groups once and a preset scales them.
    """

    var enabled: Bool
    var seed: UInt64
    var strength: Float64
    """Multiplies every `SurfaceGroup.jitter`."""
    var light_tilt_deg: Float64
    """The scene lights' direction, tilted by up to this about the base."""
    var light_scale: Float64
    """Diffuse/specular scale `U(1-r, 1+r)` on each scene light."""
    var light_tint: Float64
    var light_ambient: Float64
    """Ambient added to each scene light, `U(0, a)` grey."""
    var headlight_scale: Float64
    var headlight_off_prob: Float64
    var max_extra_lights: Int
    var extra_light_max: Float64
    """Diffuse of an extra spot, `U(0.1, e)`."""
    var cam_pos_m: Float64
    var cam_rot_deg: Float64
    var cam_fovy_deg: Float64
    var background: Float64
    """Per-channel `U(-b, b)` around the base background."""

    @staticmethod
    def off() -> Self:
        return Self(
            False, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0,
            0.0, 0.0,
        )

    @staticmethod
    def light(seed: UInt64) -> Self:
        return Self(
            True, seed, 0.5, 20.0, 0.25, 0.05, 0.05, 0.25, 0.0, 1, 0.3,
            0.005, 1.0, 1.5, 0.05,
        )

    @staticmethod
    def full(seed: UInt64) -> Self:
        """The plan's §2 ranges. Camera ±10 mm / ±2° / ±3° assumes CALIBRATED
        extrinsics; before the desk session, widen `cam_*` (plan §6). A scene
        whose cameras are known better narrows them per camera
        (`VisualRandomizer.scale_camera`; the tower rig does,
        `so101_tower_rig.scale_tower_camera_dr`).

        ⚠ RE-CENTRED ON THE CALIBRATED LOOK (2026-09-24). The ranges only ADD
        light twice — the ambient `U(0, a)` and the extra spots — so around
        the tower's calibrated scene the frame mean's MEDIAN sat 15 / 28 grey
        levels above the base (overhead / wrist) and 34 / 39 above the real
        frames. Added ambient .15 -> .05 and the spots' diffuse .5 -> .3 bring
        the median back to the base. The headlight-off draw (was 10%) is 0:
        the calibrated headlight carries the scene's ambient (.6), so "off"
        rendered a nearly black frame no room in the recordings looks like."""
        return Self(
            True, seed, 1.0, 40.0, 0.5, 0.10, 0.05, 0.6, 0.0, 2, 0.3,
            0.010, 2.0, 3.0, 0.10,
        )

    @staticmethod
    def parse(name: String, seed: UInt64) raises -> Self:
        if name == "" or name == "off":
            return Self.off()
        if name == "light":
            return Self.light(seed)
        if name == "full":
            return Self.full(seed)
        raise Error(
            "DomainRandConfig: unknown preset '" + name
            + "' (expected off | light | full)"
        )

    def write_to(self, mut writer: Some[Writer]):
        if not self.enabled:
            writer.write("off")
            return
        writer.write(
            "seed ", self.seed, ", colours x", self.strength, ", light tilt ",
            self.light_tilt_deg, "deg scale ±", self.light_scale,
            " tint ±", self.light_tint, " ambient<=", self.light_ambient,
            ", headlight ±", self.headlight_scale, " off p=",
            self.headlight_off_prob, ", extra spots<=", self.max_extra_lights,
            ", camera ±", self.cam_pos_m * 1000.0, "mm ±", self.cam_rot_deg,
            "deg fovy ±", self.cam_fovy_deg, "deg, background ±",
            self.background,
        )


# ── the stream ───────────────────────────────────────────────────────────────


struct _Stream:
    var s: UInt64

    def __init__(out self, seed: UInt64, draw: Int):
        self.s = (seed ^ DR_SALT) + UInt64(draw) * UInt64(0xA24BAED4963EE407)
        _ = self.next()

    def next(mut self) -> UInt64:
        self.s += UInt64(0x9E3779B97F4A7C15)
        var z = self.s
        z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
        return z ^ (z >> 31)

    def u(mut self) -> Float64:
        """U[0, 1) with 53 bits."""
        return Float64(Int(self.next() >> 11)) * (1.0 / 9007199254740992.0)

    def sym(mut self, r: Float64) -> Float64:
        return (2.0 * self.u() - 1.0) * r

    def unit_vec(mut self) -> Tuple[Float64, Float64, Float64]:
        var z = 2.0 * self.u() - 1.0
        var a = 2.0 * pi * self.u()
        var rr = sqrt(max(0.0, 1.0 - z * z))
        return (rr * cos(a), rr * sin(a), z)


# ── colour helpers ───────────────────────────────────────────────────────────


def _clamp01(x: Float64) -> Float64:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _rgb_to_hsv(r: Float64, g: Float64, b: Float64) -> Tuple[Float64, Float64, Float64]:
    var mx = max(r, max(g, b))
    var mn = min(r, min(g, b))
    var d = mx - mn
    var h = 0.0
    if d > 0.0:
        if mx == r:
            h = (g - b) / d
            if h < 0.0:
                h += 6.0
        elif mx == g:
            h = (b - r) / d + 2.0
        else:
            h = (r - g) / d + 4.0
        h *= 60.0
    var s = 0.0 if mx <= 0.0 else d / mx
    return (h, s, mx)


def _hsv_to_rgb(h: Float64, s: Float64, v: Float64) -> Tuple[Float64, Float64, Float64]:
    var hh = h
    while hh < 0.0:
        hh += 360.0
    while hh >= 360.0:
        hh -= 360.0
    var c = v * s
    var hp = hh / 60.0
    var k = hp - 2.0 * Float64(Int(hp / 2.0))
    var x = c * (1.0 - abs(k - 1.0))
    var r = 0.0
    var g = 0.0
    var b = 0.0
    if hp < 1.0:
        r = c; g = x
    elif hp < 2.0:
        r = x; g = c
    elif hp < 3.0:
        g = c; b = x
    elif hp < 4.0:
        g = x; b = c
    elif hp < 5.0:
        r = x; b = c
    else:
        r = c; b = x
    var mm = v - c
    return (r + mm, g + mm, b + mm)


@fieldwise_init
struct _GroupDraw(Copyable, ImplicitlyCopyable, Movable):
    var dh: Float64
    var ks: Float64
    var kv: Float64
    var tr: Float64
    var tg: Float64
    var tb: Float64
    var kp: Float64

    def colour(self, r: Float64, g: Float64, b: Float64) -> Tuple[Float64, Float64, Float64]:
        var hsv = _rgb_to_hsv(r, g, b)
        var rgb = _hsv_to_rgb(
            hsv[0] + self.dh, _clamp01(hsv[1] * self.ks), _clamp01(hsv[2] * self.kv)
        )
        return (
            _clamp01(rgb[0] * self.tr),
            _clamp01(rgb[1] * self.tg),
            _clamp01(rgb[2] * self.tb),
        )


# ── the randomizer ───────────────────────────────────────────────────────────


struct VisualRandomizer[DTYPE: DType](Movable):
    """Base copies of the tables it rewrites, the group map, and the config.

    Build one per (VisualModel, Model) PAIR: the host check's runtime-built
    pair gets its own randomizer with the same config and groups, and the
    same draw index then writes the same numbers into it."""

    var cfg: DomainRandConfig
    var groups: List[SurfaceGroup]
    var base_lights: List[Scalar[Self.DTYPE]]
    var base_nlight: Int
    var base_materials: List[Scalar[Self.DTYPE]]
    var base_appearance: List[Scalar[Self.DTYPE]]
    var base_cameras: List[Scalar[Self.DTYPE]]
    var cams: List[Int]
    var cam_scale: List[Float64]
    """Per listed camera (3 words: position, rotation, fovy): a multiplier on
    the preset's camera ranges, 1 by default (`scale_camera`). A calibrated
    fixed camera and an arm-mounted one are known to different accuracies;
    the preset stays the one knob, the scene says how each camera sits in
    it."""
    var geom_group: List[Int]
    """Per visual row: its group, or -1."""
    var mat_group: List[Int]
    """Per material: the group of the FIRST matched geom using it, or -1."""
    var base_background: Vec3Generic[Self.DTYPE]
    var target: Tuple[Float64, Float64, Float64]
    """What the extra spots aim at (the workspace centre)."""

    def __init__[
        D: DimsLike
    ](
        out self,
        cfg: DomainRandConfig,
        var groups: List[SurfaceGroup],
        ref vis: VisualModel[Self.DTYPE],
        ref m: Model[Self.DTYPE, D],
        labels: List[String],
        var cams: List[Int],
        background: Vec3Generic[Self.DTYPE],
        target: Tuple[Float64, Float64, Float64],
    ) raises:
        self.cfg = cfg
        self.groups = groups^
        self.cams = cams^
        self.cam_scale = List[Float64](length=3 * len(self.cams), fill=1.0)
        self.base_background = background
        self.target = target
        self.base_lights = vis.lights.data.copy()
        self.base_nlight = vis.nlight
        self.base_materials = vis.materials.data.copy()
        self.base_appearance = vis.appearance.data.copy()
        self.base_cameras = m.cameras.data.copy()
        if len(self.base_lights) < MAX_VIS_LIGHTS * VIS_LIGHT_WORDS:
            raise Error("VisualRandomizer: the light table is not MAX_VIS_LIGHTS rows")
        if len(vis.src_geom) != vis.ngeom:
            raise Error("VisualRandomizer: the VisualModel carries no src_geom map")
        for c in self.cams:
            if c < 0 or c >= MAX_GPU_CAMERAS:
                raise Error("VisualRandomizer: camera " + String(c) + " out of range")

        self.geom_group = List[Int](length=vis.ngeom, fill=-1)
        self.mat_group = List[Int](length=vis.nmat, fill=-1)
        var matched = List[Int](length=len(self.groups), fill=0)
        for k in range(vis.ngeom):
            var gname = labels[vis.src_geom[k]]
            for gi in range(len(self.groups)):
                var hit = False
                for p in self.groups[gi].prefixes:
                    if p.byte_length() > 0 and gname.startswith(p):
                        hit = True
                if hit:
                    self.geom_group[k] = gi
                    matched[gi] += 1
                    var mid = Int(vis.appearance.data[k * VIS_GEOM_APPEARANCE + APP_IDX_MATID])
                    if mid >= 0 and mid < vis.nmat and self.mat_group[mid] < 0:
                        self.mat_group[mid] = gi
                    break
        for gi in range(len(self.groups)):
            if matched[gi] == 0:
                raise Error(
                    "VisualRandomizer: group '" + self.groups[gi].name
                    + "' matches no visual geom — a prefix typo would leave a"
                    " surface silently un-randomized"
                )

    def group_count(self, gi: Int) -> Int:
        var n = 0
        for g in self.geom_group:
            if g == gi:
                n += 1
        return n

    def apply[
        D: DimsLike
    ](
        self, draw: Int, mut vis: VisualModel[Self.DTYPE], mut m: Model[Self.DTYPE, D]
    ) raises -> Vec3Generic[Self.DTYPE]:
        """Restore the base, then write draw `draw`. Returns the background."""
        # ── restore ─────────────────────────────────────────────────────
        for i in range(len(self.base_lights)):
            vis.lights.data[i] = self.base_lights[i]
        vis.nlight = self.base_nlight
        for i in range(len(self.base_materials)):
            vis.materials.data[i] = self.base_materials[i]
        for i in range(len(self.base_appearance)):
            vis.appearance.data[i] = self.base_appearance[i]
        for c in self.cams:
            var cb = c * MODEL_CAM_SIZE
            for w in range(MODEL_CAM_SIZE):
                m.cameras.data[cb + w] = self.base_cameras[cb + w]
        if not self.cfg.enabled:
            return self.base_background

        # ⚠ THE ORDER OF DRAWS BELOW IS THE STREAM'S LAYOUT. Every knob draws
        # its uniforms whether or not its range is zero, so turning one knob
        # off does not shift what every later knob draws.
        var st = _Stream(self.cfg.seed, draw)
        var c = self.cfg

        # ── colours, per group ──────────────────────────────────────────
        var gd = List[_GroupDraw]()
        for gi in range(len(self.groups)):
            var j = self.groups[gi].jitter.scaled(c.strength)
            gd.append(
                _GroupDraw(
                    st.sym(j.hue_deg), 1.0 + st.sym(j.sat), 1.0 + st.sym(j.val),
                    1.0 + st.sym(j.tint), 1.0 + st.sym(j.tint),
                    1.0 + st.sym(j.tint), 1.0 + st.sym(j.spec),
                )
            )
        for mi in range(vis.nmat):
            var gi = self.mat_group[mi]
            if gi < 0:
                continue
            var o = mi * VIS_MAT_WORDS
            var rgb = gd[gi].colour(
                Float64(vis.materials.data[o + MAT_IDX_R]),
                Float64(vis.materials.data[o + MAT_IDX_G]),
                Float64(vis.materials.data[o + MAT_IDX_B]),
            )
            vis.materials.data[o + MAT_IDX_R] = Scalar[Self.DTYPE](rgb[0])
            vis.materials.data[o + MAT_IDX_G] = Scalar[Self.DTYPE](rgb[1])
            vis.materials.data[o + MAT_IDX_B] = Scalar[Self.DTYPE](rgb[2])
            vis.materials.data[o + MAT_IDX_SPECULAR] = Scalar[Self.DTYPE](
                _clamp01(Float64(vis.materials.data[o + MAT_IDX_SPECULAR]) * gd[gi].kp)
            )
            vis.materials.data[o + MAT_IDX_SHININESS] = Scalar[Self.DTYPE](
                _clamp01(Float64(vis.materials.data[o + MAT_IDX_SHININESS]) * gd[gi].kp)
            )
        for k in range(vis.ngeom):
            var gi = self.geom_group[k]
            if gi < 0:
                continue
            var o = k * VIS_GEOM_APPEARANCE
            var rgb = gd[gi].colour(
                Float64(vis.appearance.data[o + APP_IDX_R]),
                Float64(vis.appearance.data[o + APP_IDX_G]),
                Float64(vis.appearance.data[o + APP_IDX_B]),
            )
            vis.appearance.data[o + APP_IDX_R] = Scalar[Self.DTYPE](rgb[0])
            vis.appearance.data[o + APP_IDX_G] = Scalar[Self.DTYPE](rgb[1])
            vis.appearance.data[o + APP_IDX_B] = Scalar[Self.DTYPE](rgb[2])

        # ── the headlight (row 0) ───────────────────────────────────────
        var hk = 1.0 + st.sym(c.headlight_scale)
        var hoff = st.u() < c.headlight_off_prob
        for w in [LIGHT_IDX_AMBIENT_R, LIGHT_IDX_AMBIENT_G, LIGHT_IDX_AMBIENT_B,
                  LIGHT_IDX_DIFFUSE_R, LIGHT_IDX_DIFFUSE_G, LIGHT_IDX_DIFFUSE_B,
                  LIGHT_IDX_SPECULAR_R, LIGHT_IDX_SPECULAR_G, LIGHT_IDX_SPECULAR_B]:
            vis.lights.data[w] = Scalar[Self.DTYPE](
                max(0.0, Float64(vis.lights.data[w]) * hk)
            )
        if hoff:
            vis.lights.data[LIGHT_IDX_ACTIVE] = Scalar[Self.DTYPE](0)

        # ── the scene's own lights (rows 1..base_nlight-1) ──────────────
        for li in range(1, self.base_nlight):
            var o = li * VIS_LIGHT_WORDS
            var ks = 1.0 + st.sym(c.light_scale)
            var t0 = 1.0 + st.sym(c.light_tint)
            var t1 = 1.0 + st.sym(c.light_tint)
            var t2 = 1.0 + st.sym(c.light_tint)
            var amb = st.u() * c.light_ambient
            var ax = st.unit_vec()
            var ang = st.sym(c.light_tilt_deg) * pi / 180.0
            var tint = SIMD[DType.float64, 4](t0, t1, t2, 1.0)
            comptime for ch in range(3):
                vis.lights.data[o + LIGHT_IDX_DIFFUSE_R + ch] = Scalar[Self.DTYPE](
                    max(0.0, Float64(vis.lights.data[o + LIGHT_IDX_DIFFUSE_R + ch]) * ks * tint[ch])
                )
                vis.lights.data[o + LIGHT_IDX_SPECULAR_R + ch] = Scalar[Self.DTYPE](
                    max(0.0, Float64(vis.lights.data[o + LIGHT_IDX_SPECULAR_R + ch]) * ks * tint[ch])
                )
                vis.lights.data[o + LIGHT_IDX_AMBIENT_R + ch] = Scalar[Self.DTYPE](
                    Float64(vis.lights.data[o + LIGHT_IDX_AMBIENT_R + ch]) + amb
                )
            var d = _rotate(
                (
                    Float64(vis.lights.data[o + LIGHT_IDX_DIR_X]),
                    Float64(vis.lights.data[o + LIGHT_IDX_DIR_Y]),
                    Float64(vis.lights.data[o + LIGHT_IDX_DIR_Z]),
                ),
                ax, ang,
            )
            vis.lights.data[o + LIGHT_IDX_DIR_X] = Scalar[Self.DTYPE](d[0])
            vis.lights.data[o + LIGHT_IDX_DIR_Y] = Scalar[Self.DTYPE](d[1])
            vis.lights.data[o + LIGHT_IDX_DIR_Z] = Scalar[Self.DTYPE](d[2])

        # ── extra spots, aimed at the target ────────────────────────────
        var n_extra = Int(st.u() * Float64(c.max_extra_lights + 1))
        if n_extra > c.max_extra_lights:
            n_extra = c.max_extra_lights
        for e in range(c.max_extra_lights):
            var px = self.target[0] + st.sym(0.6)
            var py = self.target[1] + st.sym(0.6)
            var pz = self.target[2] + 0.5 + st.u() * 0.8
            var aim_x = self.target[0] + st.sym(0.15)
            var aim_y = self.target[1] + st.sym(0.15)
            var cutoff = 30.0 + st.u() * 40.0
            var dif = 0.1 + st.u() * max(0.0, c.extra_light_max - 0.1)
            var t0 = 1.0 + st.sym(c.light_tint)
            var t1 = 1.0 + st.sym(c.light_tint)
            var t2 = 1.0 + st.sym(c.light_tint)
            if e >= n_extra:
                continue
            var li = vis.nlight
            if li >= MAX_VIS_LIGHTS:
                break
            var o = li * VIS_LIGHT_WORDS
            for w in range(VIS_LIGHT_WORDS):
                vis.lights.data[o + w] = Scalar[Self.DTYPE](0)
            var dx = aim_x - px
            var dy = aim_y - py
            var dz = self.target[2] - pz
            var dn = sqrt(dx * dx + dy * dy + dz * dz)
            vis.lights.data[o + LIGHT_IDX_BODY] = Scalar[Self.DTYPE](0)
            vis.lights.data[o + LIGHT_IDX_POS_X] = Scalar[Self.DTYPE](px)
            vis.lights.data[o + LIGHT_IDX_POS_Y] = Scalar[Self.DTYPE](py)
            vis.lights.data[o + LIGHT_IDX_POS_Z] = Scalar[Self.DTYPE](pz)
            vis.lights.data[o + LIGHT_IDX_DIR_X] = Scalar[Self.DTYPE](dx / dn)
            vis.lights.data[o + LIGHT_IDX_DIR_Y] = Scalar[Self.DTYPE](dy / dn)
            vis.lights.data[o + LIGHT_IDX_DIR_Z] = Scalar[Self.DTYPE](dz / dn)
            var tint = SIMD[DType.float64, 4](t0, t1, t2, 1.0)
            comptime for ch in range(3):
                vis.lights.data[o + LIGHT_IDX_DIFFUSE_R + ch] = Scalar[Self.DTYPE](dif * tint[ch])
                vis.lights.data[o + LIGHT_IDX_SPECULAR_R + ch] = Scalar[Self.DTYPE](0.3 * dif * tint[ch])
            vis.lights.data[o + LIGHT_IDX_DIRECTIONAL] = Scalar[Self.DTYPE](0)
            vis.lights.data[o + LIGHT_IDX_CASTSHADOW] = Scalar[Self.DTYPE](0)
            vis.lights.data[o + LIGHT_IDX_CUTOFF] = Scalar[Self.DTYPE](cutoff)
            vis.lights.data[o + LIGHT_IDX_ACTIVE] = Scalar[Self.DTYPE](1)
            vis.nlight += 1

        # ── cameras, around their base rows ─────────────────────────────
        # (every draw is taken whatever the scale, so a scale moves no
        # other draw of the stream)
        for ki in range(len(self.cams)):
            var cam = self.cams[ki]
            var sp = self.cam_scale[3 * ki]
            var cb = cam * MODEL_CAM_SIZE
            var dpx = st.sym(c.cam_pos_m * sp)
            var dpy = st.sym(c.cam_pos_m * sp)
            var dpz = st.sym(c.cam_pos_m * sp)
            var ax = st.unit_vec()
            var ang = st.sym(c.cam_rot_deg * self.cam_scale[3 * ki + 1]) * pi / 180.0
            var dfov = st.sym(c.cam_fovy_deg * self.cam_scale[3 * ki + 2])
            m.cameras.data[cb + CAM_IDX_POS_X] += Scalar[Self.DTYPE](dpx)
            m.cameras.data[cb + CAM_IDX_POS_Y] += Scalar[Self.DTYPE](dpy)
            m.cameras.data[cb + CAM_IDX_POS_Z] += Scalar[Self.DTYPE](dpz)
            # q' = q (x) dq: the rotation is about an axis in the CAMERA's
            # own frame, so it jitters the aim, not the mount.
            var q = _quat_mul(
                (
                    Float64(m.cameras.data[cb + CAM_IDX_QUAT_X]),
                    Float64(m.cameras.data[cb + CAM_IDX_QUAT_Y]),
                    Float64(m.cameras.data[cb + CAM_IDX_QUAT_Z]),
                    Float64(m.cameras.data[cb + CAM_IDX_QUAT_W]),
                ),
                _quat_axis_angle(ax, ang),
            )
            m.cameras.data[cb + CAM_IDX_QUAT_X] = Scalar[Self.DTYPE](q[0])
            m.cameras.data[cb + CAM_IDX_QUAT_Y] = Scalar[Self.DTYPE](q[1])
            m.cameras.data[cb + CAM_IDX_QUAT_Z] = Scalar[Self.DTYPE](q[2])
            m.cameras.data[cb + CAM_IDX_QUAT_W] = Scalar[Self.DTYPE](q[3])
            var fovy = Float64(m.cameras.data[cb + CAM_IDX_FOVY]) + dfov
            m.cameras.data[cb + CAM_IDX_FOVY] = Scalar[Self.DTYPE](fovy)
            # ⚠ The kernel reads the BAKED tangent, not `fovy` — writing only
            # the angle would change nothing but the printout.
            m.cameras.data[cb + CAM_IDX_TAN_HALF_FOVY] = Scalar[Self.DTYPE](
                tan(0.5 * fovy * pi / 180.0)
            )

        # ── background ──────────────────────────────────────────────────
        var b0 = st.sym(c.background)
        var b1 = st.sym(c.background)
        var b2 = st.sym(c.background)
        return Vec3Generic[Self.DTYPE](
            Scalar[Self.DTYPE](_clamp01(Float64(self.base_background.x) + b0)),
            Scalar[Self.DTYPE](_clamp01(Float64(self.base_background.y) + b1)),
            Scalar[Self.DTYPE](_clamp01(Float64(self.base_background.z) + b2)),
        )

    def scale_camera(
        mut self, k: Int, pos: Float64, rot: Float64, fovy: Float64
    ) raises:
        """Multiply the preset's camera ranges for `cams[k]` (position,
        rotation, fovy). Set BEFORE the first `apply`; 1 is the preset."""
        if k < 0 or k >= len(self.cams):
            raise Error("VisualRandomizer.scale_camera: no listed camera " + String(k))
        if pos < 0.0 or rot < 0.0 or fovy < 0.0:
            raise Error("VisualRandomizer.scale_camera: a scale is >= 0")
        self.cam_scale[3 * k] = pos
        self.cam_scale[3 * k + 1] = rot
        self.cam_scale[3 * k + 2] = fovy

    def upload[
        D: DimsLike
    ](
        self, ctx: DeviceContext, mut vis: VisualModel[Self.DTYPE], mut m: Model[Self.DTYPE, D]
    ) raises:
        """The four tables a draw writes, into their EXISTING device buffers.
        Stream-ordered after the previous launch, so no synchronize."""
        vis.lights.upload_resident(ctx)
        vis.materials.upload_resident(ctx)
        vis.appearance.upload_resident(ctx)
        m.cameras.upload_resident(ctx)


# ── small rotation helpers (quaternions are (x, y, z, w), as the records) ────


def _quat_axis_angle(
    ax: Tuple[Float64, Float64, Float64], ang: Float64
) -> Tuple[Float64, Float64, Float64, Float64]:
    var s = sin(0.5 * ang)
    return (ax[0] * s, ax[1] * s, ax[2] * s, cos(0.5 * ang))


def _quat_mul(
    a: Tuple[Float64, Float64, Float64, Float64],
    b: Tuple[Float64, Float64, Float64, Float64],
) -> Tuple[Float64, Float64, Float64, Float64]:
    var ax = a[0]; var ay = a[1]; var az = a[2]; var aw = a[3]
    var bx = b[0]; var by = b[1]; var bz = b[2]; var bw = b[3]
    var x = aw * bx + ax * bw + ay * bz - az * by
    var y = aw * by - ax * bz + ay * bw + az * bx
    var z = aw * bz + ax * by - ay * bx + az * bw
    var w = aw * bw - ax * bx - ay * by - az * bz
    var n = sqrt(x * x + y * y + z * z + w * w)
    return (x / n, y / n, z / n, w / n)


def _rotate(
    v: Tuple[Float64, Float64, Float64],
    ax: Tuple[Float64, Float64, Float64],
    ang: Float64,
) -> Tuple[Float64, Float64, Float64]:
    """Rodrigues: `v` rotated by `ang` about the unit axis `ax`."""
    var c = cos(ang)
    var s = sin(ang)
    var kx = ax[0]; var ky = ax[1]; var kz = ax[2]
    var dot = kx * v[0] + ky * v[1] + kz * v[2]
    var cx = ky * v[2] - kz * v[1]
    var cy = kz * v[0] - kx * v[2]
    var cz = kx * v[1] - ky * v[0]
    return (
        v[0] * c + cx * s + kx * dot * (1.0 - c),
        v[1] * c + cy * s + ky * dot * (1.0 - c),
        v[2] * c + cz * s + kz * dot * (1.0 - c),
    )


def geom_labels(fmd: FlatModelDef) -> List[String]:
    """`"<body name>/<geom name>"` per MODEL geom — what `SurfaceGroup`
    prefixes match. The worldbody is `world`."""
    var out = List[String]()
    for g in range(len(fmd.geoms)):
        var b = fmd.geoms[g].body_id
        var bn = String(fmd.body_names[b]) if b >= 0 and b < len(
            fmd.body_names
        ) else String("?")
        out.append(bn + "/" + String(fmd.geom_names[g]))
    return out^


def so101_tower_surface_groups() -> List[SurfaceGroup]:
    """The tower rig's surfaces, at `strength = 1` (plan §2.2).

    Task objects (bowl, brick): hue ±15°, value ±20% — enough to survive a
    lighting temperature, not so much the student cannot find them. White
    surfaces (desk, floor, stand): a colour cast by tint. The arm: small, and
    ONE draw for all 17 of its materials, so it stays one arm."""
    var g = List[SurfaceGroup]()
    g.append(SurfaceGroup("desk", ["desk_"], ColourJitter(0.0, 0.3, 0.15, 0.10, 0.5)))
    g.append(SurfaceGroup("floor", ["world/floor"], ColourJitter(0.0, 0.5, 0.30, 0.15, 0.5)))
    g.append(SurfaceGroup("stand", ["tower_"], ColourJitter(10.0, 0.2, 0.15, 0.08, 0.5)))
    g.append(SurfaceGroup("arm", ["robot_"], ColourJitter(0.0, 0.1, 0.10, 0.05, 0.5)))
    g.append(SurfaceGroup("bowl", ["bowl_"], ColourJitter(15.0, 0.2, 0.20, 0.05, 0.5)))
    g.append(SurfaceGroup("brick", ["brick_"], ColourJitter(15.0, 0.2, 0.20, 0.05, 0.5)))
    return g^

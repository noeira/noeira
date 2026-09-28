"""Generic model renderer that draws geoms from GeomSpec definitions.

Parameterized by a variadic list of GeomSpec types, iterates at compile time
to draw all visible geoms automatically. Eliminates per-environment renderer boilerplate.

Supports all geom types: capsule, sphere, box, and plane (ground).
"""

from std.collections import Array
from noeira.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from noeira.render import Renderer3D, RendererHandoff, Camera3D, Color
from noeira.render.renderer3d import PipView
from noeira.render.ui import UIRect, UIText
from noeira.render.light import Light
from noeira.core import EnvRenderer3D

from . import ModelDefLike
from ..parser.render_fields import RenderFields
from ..gpu.constants import (
    MODEL_HFIELD_META_SIZE,
    HFIELD_META_IDX_ADR,
    HFIELD_META_IDX_NROW,
    HFIELD_META_IDX_NCOL,
    HFIELD_META_IDX_SIZE_X,
    HFIELD_META_IDX_SIZE_Y,
    HFIELD_META_IDX_SIZE_Z,
    HFIELD_META_IDX_SIZE_BASE,
)

comptime _RGEOM_HFIELD: Int = 7
"""`flat_model._GEOM_HFIELD`. ⚠ NOT MuJoCo's `mjGEOM_HFIELD` (1) — this engine
numbers its geom types differently, and the two are easy to confuse because
both enums start at PLANE = 0."""

from ..kinematics.camera_frame import (
    camera_world_pos,
    camera_world_quat,
    camera_look_dir,
    camera_up_dir,
)

@fieldwise_init
struct OverlayLine(Copyable, Movable):
    """One world-space segment drawn over the scene. See `overlay_lines`."""

    var a: Vec3Generic[DType.float64]
    var b: Vec3Generic[DType.float64]
    var color: Color


comptime Vec3 = Vec3Generic[DType.float64]
comptime Quat = QuatGeneric[DType.float64]


@fieldwise_init
struct ModelRenderer[MODEL_DEF: ModelDefLike](EnvRenderer3D, Movable):
    """Generic renderer for models defined by GeomSpec types.

    Draws all geom types (capsule, sphere, box, plane) using compile-time
    geometry from GeomSpec. Supports multiple cameras (switch with 1-9 keys)
    and multiple lights.

    Parameters:
        MODEL_DEF: ModelDefLike type defining the model's definition.
    """

    var rf: RenderFields
    """The model's render records, built ONCE here.

    ⚠ ONCE, NOT PER FRAME. This is what `ModelDefFromXML._rcd` was — data the
    comptime interpreter produced at build time, so reading it cost nothing at
    runtime. Now that it comes from `parse_xml_full`, rebuilding it inside
    `render_frame` would re-parse the whole MJCF every frame. The hooks take
    it as an argument for exactly this reason."""

    var renderer: Renderer3D
    var initialized: Bool
    var follow: Bool
    var show_velocity: Bool
    var visual_radius_scale: Float64

    # Multi-camera support
    var cameras: List[Camera3D]
    var camera_modes: List[Int]  # CAM_TRACKCOM=0, CAM_FIXED=1, CAM_TARGETBODY=2
    var camera_targets: List[Int]
    """Body each camera aims at for CAM_TARGETBODY, -1 otherwise."""

    var hfield_grid: List[Float64]
    """The live elevation grid, NORMALISED TO [0, 1], or empty.

    ⚠ RUNTIME STATE IN A COMPTIME-BUILT RENDERER, and it has to be. `RenderFields`
    is produced by the comptime interpreter and knows the model's SHAPE; a
    `<hfield nrow ncol>` with no `file` is a grid of zeros the TASK rewrites on
    every reset, so its contents are `Data`, not model. `set_heightfield` is
    how the env pushes them across.
    """
    var hfield_meta: List[Float64]
    """Per-field `(adr, nrow, ncol, size_x, size_y, size_z, size_base)`, in
    `MODEL_HFIELD_META_SIZE`-sized blocks — the same packing `Model.hfield_meta`
    uses, copied rather than re-derived so the surface drawn and the surface
    the rays hit come from ONE description."""
    var hfield_rev: Int
    """Bumped by the env when it rewrites the grid. ⚠ THE REASON THE RENDERER
    DOES NOT DIFF THE GRID ITSELF: escape's is 40,401 samples and comparing them
    every frame costs more than the draw does."""

    var light_bodies: List[Int]
    """Body each light is ATTACHED to (`mjModel.light_bodyid`); 0 = world.
    With `light_local_pos` / `light_local_dir` this is what re-poses a light
    riding a body every frame, exactly as the three camera lists below do
    for a wrist camera. Parallel to the renderer's `lights`, up to the
    fallback light `__init__` appends when a model declares none."""
    var light_local_pos: List[Vec3]
    var light_local_dir: List[Vec3]
    var camera_bodies: List[Int]
    """Body each camera is ATTACHED to (`mjModel.cam_bodyid`); 0 = worldbody."""
    var camera_local_pos: List[Vec3]
    var camera_local_quat: List[Quat]
    """The camera's pose IN ITS BODY'S FRAME, kept because `Camera3D` stores
    only the derived eye/target/up and those have to be rebuilt every frame
    once the body moves. All three lists are parallel to `cameras`, including
    the synthesised default orbit camera (body 0, identity), so
    `active_camera` indexes all four.
    """

    var active_camera: Int

    var axes_offset: Float64
    var vel_arrow_height: Float64
    var vel_arrow_scale: Float64

    # Velocity arrow color
    var vel_color: Color

    # Site marker visibility
    # Extra HUD lines the APPLICATION owns — task name, drive mode, whatever
    # the tool wants on screen. `_draw_hud` renders the fixed engine controls;
    # these are appended under them so a viewer can label itself without the
    # renderer knowing anything about viewers.
    var hud_extra: List[String]
    # Deferred UI command list. Widgets record into these off-frame and they
    # are painted at HUD time, because an application cannot draw inside
    # `render_frame`'s begin/end span.
    var ui_rects: List[UIRect]
    var ui_texts: List[UIText]

    var overlay_lines: List[OverlayLine]
    """World-space line segments painted ON TOP of the scene, replaced each
    frame by the application.

    ⚠ DEFERRED FOR THE SAME REASON `ui_rects` IS: an application cannot draw
    inside `render_frame`'s begin/end span. A studio that called
    `draw_line_3d` from its own loop would either be outside the span (the
    call is dropped) or, worse, inside somebody else's — so the selection
    outline is RECORDED here and painted below, between the tendons and the
    HUD.

    ⚠ REPLACED, NOT APPENDED. `set_overlay_lines` overwrites, so a frame that
    forgets to set them clears them — which is the behaviour a selection
    highlight wants (deselect = pass none) and the opposite of what an
    append-only list would do (the outline of every geom ever selected)."""

    var show_sites: Bool

    var track_prev: Vec3
    """Torso position the trackcom camera was last moved to, or unset.

    A trackcom camera follows the body by translating eye AND target by the
    torso's displacement since the previous frame — see the trackcom branch of
    `render` for why re-anchoring the target on the torso was wrong."""

    var track_prev_set: Bool

    var free_cam_reframe: Bool
    """One-shot: reposition the free camera to a 3/4 view on the next frame.

    Deferred rather than done in `request_free_camera` because framing needs
    the torso position, which only `render_frame` has."""
    # Model camera indices drawn as insets every frame — `set_pip_cameras`.
    var pip_cameras: List[Int]

    var show_hud: Bool
    """Draw the built-in keybind/camera/step overlay.

    ⚠ TURN THIS OFF WHEN AN ImGui SIDEBAR IS UP. The two report the same
    facts — camera, step, pause, recording — and the HUD is drawn over the
    SCENE, so leaving both on costs a strip of the robot to tell the user
    something the panel already says."""

    # HUD state
    var step_count: Int

    def __init__(
        out self,
        width: Int = 1280,
        height: Int = 720,
        visual_radius_scale: Float64 = 2.0,
        axes_offset: Float64 = 1.5,
        vel_arrow_height: Float64 = 0.15,
        vel_arrow_scale: Float64 = 0.1,
        vel_color: Color = Color(0, 255, 255, 255),
        follow: Bool = True,
        show_velocity: Bool = True,
        show_sites: Bool = False,
        show_fog: Bool = False,
        title: String = String("Model Environment"),
        adopt_rf: Optional[RenderFields] = None,
    ) raises:
        # ⚠ BEFORE ANY HOOK — every one of them reads it.
        #
        # ⚠⚠ `adopt_rf` IS WHAT LETS ONE RENDERER DRAW A FILE CHOSEN AT
        # RUNTIME, and it is the whole renderer half of the physics3d studio.
        # `make_render_fields()` is the ONE hook that names a model — it reads
        # `Self.xml_text()`, which only a comptime `ModelDefFromXML` has. Every
        # OTHER hook is a pure function of `rf` (linted:
        # `scripts/audit_render_hooks_are_rf_pure.py`), so handing the records
        # in here is the entire difference between "this renderer draws
        # walker2d" and "this renderer draws whatever you opened".
        #
        # A runtime caller builds them with
        # `build_render_fields(fmd, xml_text, base_dir)` from the same
        # `FlatModelDef` its `Model` was built from, instantiates
        # `ModelRenderer[RfOnlyModelDef]`, and passes them here.
        var rf: RenderFields
        if adopt_rf:
            rf = adopt_rf.value().copy()
        else:
            rf = Self.MODEL_DEF.make_render_fields()

        # Setup all cameras from spec (fallback to default if none defined)
        var cam_list = Self.MODEL_DEF.setup_cameras(rf, width, height)
        var mode_list = Self.MODEL_DEF.setup_camera_modes(rf)
        self.cameras = List[Camera3D]()
        self.camera_modes = List[Int]()
        self.hfield_grid = List[Float64]()
        self.hfield_meta = List[Float64]()
        self.hfield_rev = -1
        self.camera_bodies = List[Int]()
        self.camera_local_pos = List[Vec3]()
        self.camera_local_quat = List[Quat]()
        if len(cam_list) == 0:
            # No cameras in XML — add a default orbit camera
            self.cameras.append(Camera3D(
                eye=Vec3(0.0, -2.5, 1.5),
                target=Vec3(0.0, 0.5, 0.3),
                up=Vec3(0.0, 0.0, 1.0),
                fov=45.0,
                aspect=Float64(width) / Float64(height),
                near=0.01,
                far=100.0,
                screen_width=width,
                screen_height=height,
            ))
            self.camera_modes.append(1)  # CAM_FIXED
            # Synthesised, so it belongs to no body and has no local pose.
            self.camera_bodies.append(0)
            self.camera_local_pos.append(Vec3(0.0, 0.0, 0.0))
            self.camera_local_quat.append(Quat(1.0, 0.0, 0.0, 0.0))
        else:
            for i in range(len(cam_list)):
                self.cameras.append(cam_list[i].copy())
                if i < len(mode_list):
                    self.camera_modes.append(mode_list[i])
                else:
                    self.camera_modes.append(0)  # CAM_TRACKCOM fallback
                # ⚠ Read straight off `rf` rather than through a `MODEL_DEF`
                # static like `get_camera_target_bodies`: these are a verbatim
                # copy of a column, and the trait has two implementers, so a
                # new member there is a second place to keep in sync for no
                # gain.
                if i < len(rf.cam_body):
                    self.camera_bodies.append(rf.cam_body[i])
                    self.camera_local_pos.append(
                        Vec3(rf.cam_pos_x[i], rf.cam_pos_y[i], rf.cam_pos_z[i])
                    )
                    self.camera_local_quat.append(
                        Quat(
                            rf.cam_quat_w[i],
                            rf.cam_quat_x[i],
                            rf.cam_quat_y[i],
                            rf.cam_quat_z[i],
                        )
                    )
                else:
                    self.camera_bodies.append(0)
                    self.camera_local_pos.append(Vec3(0.0, 0.0, 0.0))
                    self.camera_local_quat.append(Quat(1.0, 0.0, 0.0, 0.0))
        self.active_camera = 0
        self.camera_targets = Self.MODEL_DEF.get_camera_target_bodies(rf)

        # Setup all lights from spec (fallback to default if none defined)
        var lights = Self.MODEL_DEF.setup_lights(rf)
        self.light_bodies = List[Int]()
        self.light_local_pos = List[Vec3]()
        self.light_local_dir = List[Vec3]()
        for li in range(len(lights)):
            var lb = rf.light_body[li] if li < len(rf.light_body) else 0
            self.light_bodies.append(lb)
            self.light_local_pos.append(
                Vec3(lights[li].pos_x, lights[li].pos_y, lights[li].pos_z)
            )
            self.light_local_dir.append(
                Vec3(lights[li].dir_x, lights[li].dir_y, lights[li].dir_z)
            )
        if len(lights) == 0:
            # ⚠ NO `<light>` MEANS HEADLIGHT ONLY, as in MuJoCo — not an
            # invented sun. This slot used to hold a 0.7-diffuse,
            # 0.3-ambient directional light, which with the headlight now
            # lit (below) would light such a model twice. It stays as a DARK
            # placeholder because `Renderer3D` reads an empty list as "make
            # your own default light". Its ambient carries
            # `render_gl3.c:initLights`'s global 0.3, which MuJoCo adds only
            # when there is no light at all — headlight included.
            var global_amb = 0.0 if rf.vis_headlight_active else 0.3
            lights.append(Light(
                mode=0,  # directional
                dir_x=0.5, dir_y=0.5, dir_z=-1.0,
                color_r=0.0, color_g=0.0, color_b=0.0,
                ambient=global_amb,
                specular_intensity=0.0,
                specular_exponent=10.0,
                cast_shadow=False,
            ))

        # Read visual settings from model (znear, fog, shadow, headlight)
        var vis = Self.MODEL_DEF.get_visual_settings(rf)
        var shadow_size = Int(4096)
        var fog_start = Float32(0.0)
        var fog_end = Float32(0.0)
        if len(vis) >= 8:
            # Apply znear to all cameras
            var znear = vis[0]
            for ci in range(len(self.cameras)):
                # Camera stores fov in radians already, near is Float64
                self.cameras[ci].near = znear
            # MuJoCo defines fog map params (<map fogstart fogend/>) but does
            # NOT render fog unless the mjVIS_FOG flag is enabled (off by
            # default). Mirror that: keep fog disabled unless explicitly asked.
            if show_fog:
                fog_start = Float32(vis[1])
                fog_end = Float32(vis[2])
            shadow_size = Int(vis[3])

        self.visual_radius_scale = visual_radius_scale
        self.axes_offset = axes_offset
        self.vel_arrow_height = vel_arrow_height
        self.vel_arrow_scale = vel_arrow_scale
        self.vel_color = vel_color
        self.follow = follow
        self.show_velocity = show_velocity
        self.show_sites = show_sites
        self.show_hud = True
        self.free_cam_reframe = False
        self.track_prev = Vec3(0.0, 0.0, 0.0)
        self.track_prev_set = False
        self.pip_cameras = List[Int]()
        self.hud_extra = List[String]()
        self.ui_rects = List[UIRect]()
        self.ui_texts = List[UIText]()
        self.overlay_lines = List[OverlayLine]()

        var camera = self.cameras[0].copy()
        self.renderer = Renderer3D(
            width=width,
            height=height,
            camera=camera,
            draw_grid=True,
            draw_axes=False,
            lights=lights,
            shadow_size=shadow_size,
            fog_start=fog_start,
            fog_end=fog_end,
        )

        # ⚠ THE HEADLIGHT, WHETHER OR NOT THE MODEL WROTE ONE. MuJoCo's is on
        # by default (`mjv_defaultVisual`; `mjv_makeLights` adds it whenever
        # `vis.headlight.active`), and `rf` carries the defaults when the
        # element is absent. Until 2026-09-15 only a DECLARED headlight's
        # ambient was honoured, folded into every light's ambient as a scalar
        # — which left a LIBERO arena (two spots, no `<visual>`) with black
        # walls, since nothing but the headlight reaches them.
        self.renderer.set_headlight(
            rf.vis_headlight_active,
            Float32(rf.vis_headlight_ambient_r),
            Float32(rf.vis_headlight_ambient_g),
            Float32(rf.vis_headlight_ambient_b),
            Float32(rf.vis_headlight_diffuse_r),
            Float32(rf.vis_headlight_diffuse_g),
            Float32(rf.vis_headlight_diffuse_b),
            Float32(rf.vis_headlight_specular_r),
            Float32(rf.vis_headlight_specular_g),
            Float32(rf.vis_headlight_specular_b),
        )

        # Configure skybox from GradientTexture (if model defines one)
        var skybox = Self.MODEL_DEF.get_skybox_colors(rf)
        if len(skybox) == 6:
            self.renderer.set_skybox(
                top_r=Float32(skybox[0]),
                top_g=Float32(skybox[1]),
                top_b=Float32(skybox[2]),
                bottom_r=Float32(skybox[3]),
                bottom_g=Float32(skybox[4]),
                bottom_b=Float32(skybox[5]),
            )
            # `mark="random"` on the same texture — dm_control's night sky.
            var mark = Self.MODEL_DEF.get_skybox_mark(rf)
            if len(mark) == 5 and Int(mark[0]) == 3:
                self.renderer.set_skybox_stars(
                    r=Float32(mark[1]),
                    g=Float32(mark[2]),
                    b=Float32(mark[3]),
                    density=Float32(mark[4]),
                )

        # Configure ground appearance from model textures/geom colors
        var checker = Self.MODEL_DEF.get_checker_colors(rf)
        if len(checker) == 3:
            # Model has a checker texture — use it
            self.renderer.set_ground_checker_colors(
                r=Float32(checker[0]),
                g=Float32(checker[1]),
                b=Float32(checker[2]),
            )
        else:
            # No checker texture — use plane geom's rgba as solid color
            var ground_rgba = Self.MODEL_DEF.get_ground_rgba(rf)
            if len(ground_rgba) == 3:
                self.renderer.set_ground_solid_color(
                    r=Float32(ground_rgba[0]),
                    g=Float32(ground_rgba[1]),
                    b=Float32(ground_rgba[2]),
                )

        self.rf = rf^
        self.step_count = 0
        self.initialized = False

    def __init__(out self, *, deinit move: Self):
        self.rf = move.rf^
        self.renderer = move.renderer^
        self.initialized = move.initialized
        self.follow = move.follow
        self.show_velocity = move.show_velocity
        self.show_sites = move.show_sites
        self.show_hud = move.show_hud
        self.free_cam_reframe = move.free_cam_reframe
        self.track_prev = move.track_prev
        self.track_prev_set = move.track_prev_set
        self.pip_cameras = move.pip_cameras^
        self.hud_extra = move.hud_extra^
        self.ui_rects = move.ui_rects^
        self.ui_texts = move.ui_texts^
        self.overlay_lines = move.overlay_lines^
        self.visual_radius_scale = move.visual_radius_scale
        self.cameras = move.cameras^
        self.camera_modes = move.camera_modes^
        self.camera_targets = move.camera_targets^
        self.hfield_grid = move.hfield_grid^
        self.hfield_meta = move.hfield_meta^
        self.hfield_rev = move.hfield_rev
        self.light_bodies = move.light_bodies^
        self.light_local_pos = move.light_local_pos^
        self.light_local_dir = move.light_local_dir^
        self.camera_bodies = move.camera_bodies^
        self.camera_local_pos = move.camera_local_pos^
        self.camera_local_quat = move.camera_local_quat^
        self.active_camera = move.active_camera
        self.axes_offset = move.axes_offset
        self.vel_arrow_height = move.vel_arrow_height
        self.vel_arrow_scale = move.vel_arrow_scale
        self.vel_color = move.vel_color
        self.step_count = move.step_count

    def init(mut self) raises -> None:
        self.init(None)

    def init(mut self, adopt: Optional[RendererHandoff]) raises -> None:
        """Open a window, or ADOPT one a previous model's renderer detached.

        Adopting is what lets a model-swapping tool keep the same window across
        the swap — same monitor, same position, same ImGui state. See
        `RendererHandoff`.
        """
        var title = String("Model Environment")
        self.renderer.init(title, adopt)
        self.initialized = True

    def close(mut self) raises -> None:
        if self.initialized:
            self.renderer.close()
            self.initialized = False

    def detach(mut self) raises -> RendererHandoff:
        """Give up this model's GPU caches and hand the window on.

        ⚠ THE CALLER OWNS THE RESULT. Nothing frees it implicitly: pass it to
        the next `init`, or end it with `Renderer3D.close_handoff`.
        """
        var h = self.renderer.detach()
        self.initialized = False
        return h^

    def check_quit(mut self) -> Bool:
        return self.renderer.check_quit()

    def is_open(self) -> Bool:
        return self.initialized

    def delay(self, ms: Int) -> None:
        try:
            self.renderer.delay_ms(ms)
        except:
            pass

    def orbit_camera(
        mut self, delta_theta: Float64, delta_phi: Float64
    ) -> None:
        self.renderer.orbit_camera(delta_theta, delta_phi)

    def zoom_camera(mut self, delta: Float64) -> None:
        self.renderer.zoom_camera(delta)

    def pan_camera(mut self, delta_x: Float64, delta_y: Float64) -> None:
        self.renderer.pan_camera(delta_x, delta_y)

    def reset_camera(mut self) -> None:
        self.renderer.reset_camera()

    def render_from_body_state[
        DTYPE: DType, SIZE_POS: Int, SIZE_QUAT: Int
    ](
        mut self,
        xpos: Array[Scalar[DTYPE], SIZE_POS],
        xquat: Array[Scalar[DTYPE], SIZE_QUAT],
        num_bodies: Int,
        vel_x: Float64 = 0.0,
    ):
        """Render directly from physics Data body arrays.

        Extracts body positions and quaternions from raw xpos/xquat arrays
        and delegates to render(). Eliminates per-environment boilerplate.

        Args:
            xpos: Flat array of body positions [x0,y0,z0, x1,y1,z1, ...].
            xquat: Flat array of body quaternions [x0,y0,z0,w0, ...].
            num_bodies: Number of bodies in the arrays.
            vel_x: Forward velocity for indicator arrow.
        """
        var positions = List[Vec3](capacity=num_bodies)
        var quaternions = List[Quat](capacity=num_bodies)

        for i in range(num_bodies):
            positions.append(
                Vec3(
                    Float64(xpos[i * 3 + 0]),
                    Float64(xpos[i * 3 + 1]),
                    Float64(xpos[i * 3 + 2]),
                )
            )
            # xquat stored as [x, y, z, w], Quat constructor is (w, x, y, z)
            quaternions.append(
                Quat(
                    Float64(xquat[i * 4 + 3]),
                    Float64(xquat[i * 4 + 0]),
                    Float64(xquat[i * 4 + 1]),
                    Float64(xquat[i * 4 + 2]),
                )
            )

        self.render(positions, quaternions, vel_x)

    def render(
        mut self,
        positions: List[Vec3],
        quaternions: List[Quat],
        vel_x: Float64 = 0.0,
    ):
        """Render all visible geoms.

        Args:
            positions: World positions for each body (indexed by BODY_IDX).
            quaternions: World orientations for each body (indexed by BODY_IDX).
            vel_x: Forward velocity for indicator arrow.
        """
        if not self.initialized:
            return

        # Handle camera switch request from noeira.renderer3D (number keys 1-9)
        var cam_req = self.renderer.camera_switch_request
        if cam_req >= 0 and cam_req < len(self.cameras):
            self.active_camera = cam_req
            # Load the new camera settings into the renderer
            var new_cam = self.cameras[self.active_camera].copy()
            self.renderer.camera.eye = new_cam.eye
            self.renderer.camera.target = new_cam.target
            self.renderer.camera.up = new_cam.up
            self.renderer.camera.fov = new_cam.fov
            self.renderer.camera.near = new_cam.near
            self.renderer.camera.far = new_cam.far
            self.track_prev_set = False

        var torso_pos = positions[1]  # Body 1 = torso (body 0 = worldbody)

        # ── lights riding a body: world pose from this frame's FK ─────────
        # `mj_camlight` does the same for `light_xpos` / `light_xdir`. A
        # world light (body 0) keeps its declared pose, which the local
        # lists already hold.
        for li in range(len(self.light_bodies)):
            var lb = self.light_bodies[li]
            if lb <= 0 or lb >= len(positions) or li >= len(self.renderer.lights):
                continue
            var wp = positions[lb] + quaternions[lb].rotate_vec(self.light_local_pos[li])
            var wd = quaternions[lb].rotate_vec(self.light_local_dir[li])
            self.renderer.lights[li].pos_x = wp.x
            self.renderer.lights[li].pos_y = wp.y
            self.renderer.lights[li].pos_z = wp.z
            self.renderer.lights[li].dir_x = wd.x
            self.renderer.lights[li].dir_y = wd.y
            self.renderer.lights[li].dir_z = wd.z

        # ── free camera ──────────────────────────────────────────────────
        # `active_camera == -1` is dm_control's free camera (its viewer starts
        # there: `_camera_idx = -1`, model cameras only via cycling). It is not
        # an entry in `cameras`; it is the ABSENCE of a model camera, so every
        # per-frame re-aim below is skipped and the pose belongs entirely to
        # mouse orbit/pan/zoom.
        #
        # ⚠ THIS IS WHY A MODEL CAMERA CAN FEEL "STUCK". A trackcom or
        # targetbody camera is re-aimed by the branches below on EVERY frame,
        # so dragging fights the model and loses. Only the free camera is
        # actually free.
        if self.free_cam_reframe:
            self.free_cam_reframe = False
            # Keep the DISTANCE that was on screen — it is the only
            # model-scale information available here, and it is already correct
            # for this model — but replace the direction with a 3/4 view. That
            # fixes the complaint (fish's camera 0 looks straight down) without
            # needing to know how big the robot is.
            var cur_off = self.renderer.camera.eye - self.renderer.camera.target
            var dist = cur_off.length()
            if dist < 1e-6:
                dist = 3.0
            var dir = Vec3(0.6, -1.0, 0.5).normalized()
            self.renderer.camera.target = torso_pos
            self.renderer.camera.eye = torso_pos + dir * dist
            self.renderer.camera.up = Vec3(0.0, 0.0, 1.0)

        # Camera follow torso (only for trackcom mode cameras)
        var cam_mode = -1
        if self.active_camera >= 0 and self.active_camera < len(
            self.camera_modes
        ):
            cam_mode = self.camera_modes[self.active_camera]

        # ── mj_camlight's default: take the camera into the world ─────────
        # `mj_camlight` calls `mj_local2Global(cam_pos, cam_quat, cam_bodyid)`
        # for EVERY mode before it dispatches, because `cam_pos`/`cam_quat` are
        # stored in the parent BODY's frame. We never did, so a camera declared
        # inside a body was drawn at its LOCAL pose read as a world pose: a
        # wrist camera stayed welded to the origin while the wrist moved. The
        # arithmetic is `xpos[b] + xmat[b]*cam_pos` / `xmat[b]*cam_quat`, here
        # in quaternion form since `render` is handed `quaternions`, not mats.
        #
        # ⚠ GATED ON `body > 0`, WHICH IS A DELIBERATE DEVIATION and a
        # deliberately EMPTY one. For the worldbody the transform is the
        # identity, so the pose this would produce is bit-identical to the one
        # `setup_cameras` already baked; skipping it only preserves today's
        # behaviour that a world-fixed camera can still be dragged with the
        # mouse. MuJoCo's own fixed cameras cannot be dragged — see the "STUCK"
        # note above, which now applies to body-attached cameras too, on
        # purpose.
        if (
            self.active_camera >= 0
            and self.active_camera < len(self.camera_bodies)
            and self.camera_bodies[self.active_camera] > 0
            and self.camera_bodies[self.active_camera] < len(positions)
            and (cam_mode == 1 or cam_mode == 2)
        ):
            var cb = self.camera_bodies[self.active_camera]
            var bq = quaternions[cb]
            var cam_xpos = camera_world_pos(
                positions[cb], bq, self.camera_local_pos[self.active_camera]
            )
            var cam_xquat = camera_world_quat(
                bq, self.camera_local_quat[self.active_camera]
            )
            self.renderer.camera.eye = cam_xpos
            if cam_mode == 1:
                # MuJoCo's camera frame looks down its own -Z with +Y up
                # (`mjCCamera`) — the same convention `setup_cameras` resolves
                # at load, applied here to the composed world orientation.
                self.renderer.camera.target = cam_xpos + camera_look_dir(
                    cam_xquat
                )
                self.renderer.camera.up = camera_up_dir(cam_xquat)
            # cam_mode == 2 (targetbody) keeps only the POSITION from here and
            # is re-aimed by the branch below, which is what `mj_camlight`
            # does: the targetbody case overwrites the orientation and leaves
            # `cam_xpos` from the local2Global default standing.
        if self.follow and cam_mode == 0:  # CAM_TRACKCOM
            # MuJoCo's trackcom keeps the camera's declared offset from the
            # body and its declared orientation: the camera TRANSLATES with the
            # body. So both eye and target move by the torso's displacement
            # since the last frame, which also leaves a mouse orbit standing.
            #
            # ⚠ THIS USED TO RE-ANCHOR THE TARGET ON THE TORSO and keep the
            # eye-to-target offset. `setup_cameras` puts a model camera's target
            # ONE UNIT in front of its eye (only the direction is known), so
            # every trackcom camera ended up 1 m from the torso whatever its
            # declared distance: humanoid's 3 m `side` and quadruped's 4 m `y`
            # became close-ups, fish's 0.3 m `tracking_y` a distant speck.
            #
            # On the first frame after a switch the camera is PLACED: a camera
            # declared inside a body (walker, cheetah, humanoid, dog all put
            # theirs in the torso) has a body-local pose, and MuJoCo's
            # `cam_pos0` is that pose taken into the world. `setup_cameras`
            # read it as a world pose, which put cheetah's `side` camera at
            # floor height.
            #
            # ⚠ ONLY THE BODY'S POSITION IS APPLIED, NOT ITS ORIENTATION.
            # `cam_pos0` / `cam_mat0` are taken at qpos0, and dm_control's
            # resets randomise the torso's orientation (walker, humanoid,
            # quadruped, dog): composing with this frame's body quaternion
            # rolled the whole view. At qpos0 these torsos are unrotated, so
            # the declared local pose read as an offset from the body's current
            # position is MuJoCo's placement. A torso declared rotated would
            # need its qpos0 quaternion here, which the renderer does not have.
            var ac = self.active_camera
            if self.track_prev_set:
                var delta = torso_pos - self.track_prev
                self.renderer.camera.target = self.renderer.camera.target + delta
                self.renderer.camera.eye = self.renderer.camera.eye + delta
            elif (
                ac < len(self.camera_bodies)
                and self.camera_bodies[ac] > 0
                and self.camera_bodies[ac] < len(positions)
            ):
                var cb = self.camera_bodies[ac]
                var eye = positions[cb] + self.camera_local_pos[ac]
                var q = self.camera_local_quat[ac]
                self.renderer.camera.eye = eye
                self.renderer.camera.target = eye + camera_look_dir(q)
                self.renderer.camera.up = camera_up_dir(q)
            self.track_prev = torso_pos
            self.track_prev_set = True
        elif cam_mode == 2:  # CAM_TARGETBODY
            # MuJoCo `mj_camlight`, mjCAMLIGHT_TARGETBODY: the camera does NOT
            # move — it TURNS to face the body every frame. That is the whole
            # difference from trackcom, which moves it and leaves its
            # orientation alone, and it is why collapsing the two was wrong.
            #
            #   zaxis = normalize(cam_pos - target_pos)   (points AT the camera)
            #   xaxis = normalize(cross(+Z, zaxis))
            #   yaxis = cross(zaxis, xaxis)               (the camera's up)
            #
            # ⚠ targetbodycom is aimed at the body's ORIGIN here, not its
            # subtree centre of mass, which the renderer does not have. The two
            # coincide for a single-body target — cartpole's `cart`, the only
            # user in the suite — and diverge for a multi-body subtree.
            var tgt = -1
            if self.active_camera < len(self.camera_targets):
                tgt = self.camera_targets[self.active_camera]
            if tgt >= 0 and tgt < len(positions):
                var eye = self.renderer.camera.eye
                var zax = (eye - positions[tgt]).normalized()
                if zax.length_squared() > 1e-12:
                    var world_up = Vec3(0.0, 0.0, 1.0)
                    var xax = world_up.cross(zax)
                    if xax.length_squared() < 1e-12:
                        # Camera directly above or below the target: +Z gives no
                        # usable xaxis, so fall back to +X the way a look-at has
                        # to when the view direction is the world vertical.
                        xax = Vec3(1.0, 0.0, 0.0).cross(zax)
                    xax = xax.normalized()
                    self.renderer.camera.target = eye - zax
                    self.renderer.camera.up = zax.cross(xax).normalized()

        # Prevent camera from going below ground
        if self.renderer.has_ground:
            self.renderer.camera.clamp_above_ground(self.renderer.ground_z)

        # ── the insets: this frame's pose of each picture-in-picture camera ──
        if len(self.pip_cameras) > 0:
            var views = List[PipView]()
            var scene_w = self.renderer.scene_width()
            var tile_w = scene_w * 3 // 10
            var x0 = self.renderer.ui_sidebar_width + scene_w - tile_w - 8
            var y0 = 8
            for k in range(len(self.pip_cameras)):
                var ci = self.pip_cameras[k]
                if ci < 0 or ci >= len(self.cameras):
                    continue
                var cam = self._camera_pose_now(ci, positions, quaternions)
                var asp = cam.aspect if cam.aspect > 0.1 else 4.0 / 3.0
                var tile_h = Int(Float64(tile_w) / asp)
                if y0 + tile_h > self.renderer.height:
                    break
                views.append(PipView(camera=cam^, x=x0, y=y0, w=tile_w, h=tile_h))
                y0 += tile_h + 8
            self.renderer.set_pip_views(views^)
        else:
            self.renderer.set_pip_views(List[PipView]())

        self.renderer.begin_frame()

        # Render ground geoms (planes or fallback grid)
        try:
            Self.MODEL_DEF.render_ground_geoms(
                self.rf,
                self.renderer,
                torso_pos.x,
                self.follow,
                self.visual_radius_scale,
            )
        except:
            pass

        # Render body-attached geoms
        try:
            Self.MODEL_DEF.render_body_geoms(
                self.rf,
                self.renderer,
                positions,
                quaternions,
                self.visual_radius_scale,
            )
        except:
            pass

        # ⚠ AFTER THE GEOMS AND BEFORE THE SKIN, for the depth reason the skin
        # note below gives: the terrain is opaque ground, and the robot standing
        # on it must be drawn against it, not through it.
        self._draw_heightfields(positions, quaternions)

        # Deformable skin (dog's envelope). Unconditional for the same reason
        # as the tendons — `render_skin` compiles to nothing on a model with no
        # `<skin>`, since `has_skin` is comptime.
        #
        # ⚠ AFTER THE GEOMS, BEFORE THE TENDONS. The skin is opaque and encloses
        # the group 0-2 geoms it shares a body with, so drawing it first would
        # have it depth-fight whatever pokes through; the tendons are lines and
        # want to sit on top of everything solid.
        try:
            Self.MODEL_DEF.render_skin(
                self.rf,
                self.renderer, positions, quaternions
            )
        except e:
            print("render_skin failed:", e)

        # Spatial tendons (ball_in_cup's string). Unconditional: models
        # without any record zero of them and the call costs a loop bound.
        try:
            Self.MODEL_DEF.render_spatial_tendons(
                self.rf,
                self.renderer, positions, quaternions
            )
        except:
            pass

        # ⚠ APPLICATION OVERLAY, AFTER EVERYTHING SOLID. The selection outline
        # is the reason this exists, and it has to be drawn last among the 3D
        # passes or the geom it outlines occludes its own highlight — a
        # highlight that disappears exactly when you select something is worse
        # than none.
        for ln in self.overlay_lines:
            self.renderer.draw_line_3d(ln.a, ln.b, ln.color)

        # Render site markers (bright green spheres, optional)
        if self.show_sites:
            try:
                Self.MODEL_DEF.render_sites(
                self.rf,
                    self.renderer, positions, quaternions
                )
            except:
                pass

        # Velocity indicator
        if self.show_velocity:
            self._draw_velocity_indicator(torso_pos, vel_x)

        # HUD overlay
        if self.show_hud:
            self._draw_hud()

        # Increment step counter AFTER drawing (so first frame shows 0)
        # Only increment when not paused (paused display should freeze the count)
        if not self.renderer.is_paused:
            self.step_count += 1

        # End frame
        try:
            self.renderer.end_frame()
        except:
            pass

    def set_heightfield(
        mut self,
        grid: List[Float64],
        meta: List[Float64],
        revision: Int,
    ):
        """Hand the renderer the live elevation grid.

        Args:
            grid: Elevations, NORMALISED TO [0, 1], all fields concatenated.
            meta: `MODEL_HFIELD_META_SIZE` floats per field — see `hfield_meta`.
            revision: Bump when `grid` changes; the surface is rebuilt only then.

        ⚠ COPIES, AND DELIBERATELY. The alternative is holding a borrow of
        `Data.hfield_data` across frames, which Mojo will not let a struct field
        do (`AnyOrigin`) — the same constraint that put the `mut` on
        `Env.get_state`. The copy costs 40,401 doubles ONCE PER RESET, not per
        frame, because the caller only calls this when `revision` moves.
        """
        if revision == self.hfield_rev:
            return
        self.hfield_rev = revision
        self.hfield_grid = grid.copy()
        self.hfield_meta = meta.copy()

    def _draw_heightfields(mut self, positions: List[Vec3], quats: List[Quat]):
        """Draw every `type="hfield"` geom, surface and base.

        ⚠ NOT IN `render_body_geoms`, WHICH IS WHERE EVERY OTHER GEOM TYPE IS
        DRAWN. That routine is a static method on `MODEL_DEF` taking only
        `RenderFields` — comptime model shape — and a heightfield's elevations
        are runtime `Data`. Widening its signature would push a runtime grid
        through the comptime path for all fifty models to carry; this reads the
        geom's pose and colour from `rf` exactly as that routine does, and the
        heights from the state the env pushed.
        """
        if len(self.hfield_grid) == 0 or len(self.hfield_meta) == 0:
            return
        for g in range(len(self.rf.geom_type)):
            if self.rf.geom_type[g] != _RGEOM_HFIELD:
                continue
            if g >= len(self.rf.geom_hfield_id):
                continue
            var hid = self.rf.geom_hfield_id[g]
            if hid < 0:
                continue
            var b = hid * MODEL_HFIELD_META_SIZE
            if b + MODEL_HFIELD_META_SIZE > len(self.hfield_meta):
                continue
            var adr = Int(self.hfield_meta[b + HFIELD_META_IDX_ADR])
            var nrow = Int(self.hfield_meta[b + HFIELD_META_IDX_NROW])
            var ncol = Int(self.hfield_meta[b + HFIELD_META_IDX_NCOL])
            var sx = self.hfield_meta[b + HFIELD_META_IDX_SIZE_X]
            var sy = self.hfield_meta[b + HFIELD_META_IDX_SIZE_Y]
            var sz = self.hfield_meta[b + HFIELD_META_IDX_SIZE_Z]
            var sb = self.hfield_meta[b + HFIELD_META_IDX_SIZE_BASE]

            # The geom's world pose, composed the way `render_body_geoms`
            # composes every other geom's.
            var bid = self.rf.geom_body_id[g]
            if bid >= len(positions):
                continue
            var lpos = Vec3(
                self.rf.geom_pos_x[g],
                self.rf.geom_pos_y[g],
                self.rf.geom_pos_z[g],
            )
            var lq = Quat(
                self.rf.geom_quat_w[g],
                self.rf.geom_quat_x[g],
                self.rf.geom_quat_y[g],
                self.rf.geom_quat_z[g],
            )
            var wq = quats[bid] * lq
            var wpos = positions[bid] + quats[bid].rotate_vec(lpos)

            var col = Color(
                UInt8(Int(self.rf.geom_rgba_r[g] * 255.0)),
                UInt8(Int(self.rf.geom_rgba_g[g] * 255.0)),
                UInt8(Int(self.rf.geom_rgba_b[g] * 255.0)),
                UInt8(Int(self.rf.geom_rgba_a[g] * 255.0)),
            )

            try:
                self.renderer.draw_heightfield(
                    name=String("hfield_") + String(hid),
                    center=wpos,
                    orientation=wq,
                    nrow=nrow,
                    ncol=ncol,
                    size_x=sx,
                    size_y=sy,
                    size_z=sz,
                    grid=self.hfield_grid,
                    adr=adr,
                    revision=self.hfield_rev,
                    color=col,
                )
                # ⚠ THE BASE, WITHOUT WHICH THE TERRAIN IS A SHEET. A
                # `<hfield>` sits on a solid box `size[3]` deep — the same box
                # `mj_rayHfield` intersects first — and drawing only the
                # surface makes the rim read as infinitely thin from below.
                # Skipped when it is degenerate rather than drawn zero-thick.
                if sb > 1e-9:
                    self.renderer.draw_box(
                        center=wpos
                        + wq.rotate_vec(Vec3(0.0, 0.0, -0.5 * sb)),
                        orientation=wq,
                        half_extents=Vec3(sx, sy, 0.5 * sb),
                        color=col,
                    )
            except:
                pass

    def set_hud_extra(mut self, lines: List[String]):
        """Replace the application-owned HUD lines."""
        self.hud_extra = lines.copy()

    def n_cameras(self) -> Int:
        return len(self.cameras)

    def current_camera(self) -> Int:
        return self.active_camera

    def request_camera(mut self, index: Int):
        self.renderer.request_camera(index)

    def set_pip_cameras(mut self, var cams: List[Int]):
        """Draw these model cameras as insets at the right edge of the 3D
        viewport, stacked from the top (each 30% of the scene width, the
        camera's own aspect) — a teleoperator's overhead and wrist views
        beside the free camera. Empty clears them. An index out of range is
        skipped, not an error."""
        self.pip_cameras = cams^

    def _camera_pose_now(
        self, i: Int, positions: List[Vec3], quaternions: List[Quat]
    ) -> Camera3D:
        """Model camera `i` at THIS frame's FK — `mj_camlight`'s local2Global
        for a fixed camera on a body; a world camera and the trackcom /
        targetbody modes return the stored camera (the tower's two are both
        fixed). The same arithmetic `render` applies to the ACTIVE camera."""
        var cam = self.cameras[i].copy()
        var mode = self.camera_modes[i] if i < len(self.camera_modes) else 1
        if (
            mode == 1 and i < len(self.camera_bodies)
            and self.camera_bodies[i] > 0
            and self.camera_bodies[i] < len(positions)
        ):
            var cb = self.camera_bodies[i]
            var bq = quaternions[cb]
            var cam_xpos = camera_world_pos(
                positions[cb], bq, self.camera_local_pos[i]
            )
            var cam_xquat = camera_world_quat(bq, self.camera_local_quat[i])
            cam.eye = cam_xpos
            cam.target = cam_xpos + camera_look_dir(cam_xquat)
            cam.up = camera_up_dir(cam_xquat)
        return cam^

    def request_free_camera(mut self):
        """Detach from model cameras — dm_control's free camera.

        Set directly rather than through `renderer.request_camera`, whose
        `camera_switch_request` already uses -1 to mean "no request"; routing
        free through it would be indistinguishable from silence.
        """
        self.active_camera = -1
        self.free_cam_reframe = True

    def is_free_camera(self) -> Bool:
        return self.active_camera < 0

    def set_free_camera(mut self, eye: Vec3, target: Vec3):
        """The free camera at an EXACT pose — no 3/4-view reframe.

        For putting this renderer's eye where another renderer's was: a
        `mujoco.Renderer` free camera (`lookat`, `distance`, `azimuth`,
        `elevation`) converts to eye/target in `mjv_cameraInModel`'s
        convention, and a same-pose pair of images is what a rendering
        comparison needs.
        """
        self.active_camera = -1
        self.free_cam_reframe = False
        self.renderer.camera.eye = eye
        self.renderer.camera.target = target
        self.renderer.camera.up = Vec3(0.0, 0.0, 1.0)

    def request_screenshot(mut self):
        self.renderer.request_screenshot()

    def is_recording(self) -> Bool:
        return self.renderer.is_recording()

    def recording_frames(self) -> Int:
        return self.renderer.recording_frames()

    def toggle_recording(mut self) raises:
        self.renderer.toggle_recording()

    def paused(self) -> Bool:
        return self.renderer.paused()

    def toggle_pause(mut self):
        self.renderer.toggle_pause()

    def set_text_input_mode(mut self, on: Bool):
        self.renderer.set_text_input_mode(on)

    def set_ui_sidebar_width(mut self, w: Int):
        """Reserve `w` px on the left for UI; the scene renders to the rest."""
        self.renderer.set_ui_sidebar_width(w)

    def set_pointer_claimed(mut self, on: Bool):
        """Declare that a viewport overlay owns the pointer this frame.

        For overlays ImGui cannot report on — a transform gizmo's window is
        created with `NoInputs`, so `ig_want_mouse()` returns False while it
        is being dragged and the same drag would also orbit the camera."""
        self.renderer.set_pointer_claimed(on)

    def imgui_init(mut self) raises -> Bool:
        """Attach a Dear ImGui overlay; False if the shim is not built."""
        return self.renderer.imgui_init()

    def imgui_new_frame(mut self) raises:
        """Open an ImGui frame. Call before building widgets and before
        `render_frame`."""
        self.renderer.imgui_new_frame()

    def imgui_active(self) -> Bool:
        return self.renderer.imgui_active()

    def set_capture_scene_only(mut self, on: Bool):
        """Crop screenshots/recordings to the 3D viewport (default on)."""
        self.renderer.set_capture_scene_only(on)

    def set_show_sites(mut self, on: Bool):
        """Show or hide the site markers."""
        self.show_sites = on

    def set_group_shown(mut self, group: Int, on: Bool):
        """`mjvOption.geomgroup[group]` — draw (or not) geoms of that group.

        MuJoCo's default shows 0-2. robosuite and LIBERO hide group 0
        (`render_collision_mesh=False`), where the Panda's collision meshes
        coincide with its visual ones and z-fight into a speckle over every
        link. Read by `body_geom_visible`, so the picker and the selection
        outline follow.
        """
        if group >= 0 and group < len(self.rf.group_shown):
            self.rf.group_shown[group] = on

    def set_show_hud(mut self, on: Bool):
        """Show or hide the built-in text HUD."""
        self.show_hud = on

    def set_render_fields(mut self, rf: RenderFields):
        """Swap the model's render records — what an EDIT needs.

        ⚠ THE GEOM MESH CACHE IS KEYED BY NAME AND SURVIVES, which is the
        point: re-parsing is not needed for a colour or a size change, so an
        edit stays interactive. A structural change (a different model) goes
        through `detach`/`init` instead, which drops the caches."""
        self.rf = rf.copy()

    def set_overlay_lines(mut self, lines: List[OverlayLine]):
        """Replace the world-space overlay for the next frame."""
        self.overlay_lines = lines.copy()

    def set_ui(mut self, rects: List[UIRect], texts: List[UIText]):
        """Replace the deferred UI command list for the next frame."""
        self.ui_rects = rects.copy()
        self.ui_texts = texts.copy()

    def take_click(mut self) -> Bool:
        return self.renderer.take_click()

    def mouse_x(self) -> Float32:
        return self.renderer.mouse_x

    def mouse_y(self) -> Float32:
        return self.renderer.mouse_y

    def take_key(mut self) -> Int:
        """Consume a keycode the renderer's own bindings did not claim."""
        return self.renderer.take_key()

    def _draw_hud(mut self):
        """Draw MuJoCo-style HUD: controls help, camera name, step counter, pause indicator.
        """
        # ⚠ START AFTER THE RESERVED STRIP. The engine HUD belongs over the
        # SCENE, not under the application's sidebar: panels are drawn
        # semi-transparent, so a HUD at x=12 ghosts through a sidebar rather
        # than being hidden by it.
        var x0 = Float32(self.renderer.ui_sidebar_width + 12)
        var y = Float32(12)
        var s = 2  # 2× scale → 16×16 px per char

        # Controls (dim white) — shadow then bright
        var dim = Color(180, 180, 180, 200)
        self.renderer.draw_text(
            x0 + 1, y + 1, "[Spc] Pause", Color(0, 0, 0, 160), s
        )
        self.renderer.draw_text(x0, y, "[Spc] Pause", dim, s)
        y += 20
        self.renderer.draw_text(
            x0 + 1, y + 1, "[->]  Step", Color(0, 0, 0, 160), s
        )
        self.renderer.draw_text(x0, y, "[->]  Step", dim, s)
        y += 20
        self.renderer.draw_text(
            x0 + 1, y + 1, "[1-9] Camera", Color(0, 0, 0, 160), s
        )
        self.renderer.draw_text(x0, y, "[1-9] Camera", dim, s)
        y += 20
        self.renderer.draw_text(
            x0 + 1, y + 1, "[R]   Reset cam", Color(0, 0, 0, 160), s
        )
        self.renderer.draw_text(x0, y, "[R]   Reset cam", dim, s)
        y += 20
        self.renderer.draw_text(
            x0 + 1, y + 1, "[S]   Screenshot", Color(0, 0, 0, 160), s
        )
        self.renderer.draw_text(x0, y, "[S]   Screenshot", dim, s)
        y += 20
        self.renderer.draw_text(
            x0 + 1, y + 1, "[V]   Record", Color(0, 0, 0, 160), s
        )
        self.renderer.draw_text(x0, y, "[V]   Record", dim, s)
        y += 28  # gap

        # Camera name (bright white)
        # `active_camera + 1` would read "Cam 0" for the free camera, which is
        # a real camera index in every other line of this HUD.
        var cam_name = (
            String("Cam free") if self.active_camera < 0
            else String("Cam ") + String(self.active_camera + 1)
        )
        self.renderer.draw_text(x0 + 1, y + 1, cam_name, Color(0, 0, 0, 160), s)
        self.renderer.draw_text(x0, y, cam_name, Color(255, 255, 255, 255), s)
        y += 20

        # Step counter (yellow-white)
        var step_str = String("Step: ") + String(self.step_count)
        self.renderer.draw_text(x0 + 1, y + 1, step_str, Color(0, 0, 0, 160), s)
        self.renderer.draw_text(x0, y, step_str, Color(255, 255, 200, 255), s)
        y += 20

        # Pause indicator (yellow, only when paused)
        if self.renderer.is_paused:
            self.renderer.draw_text(
                x0 + 1, y + 1, "[PAUSED]", Color(0, 0, 0, 160), s
            )
            self.renderer.draw_text(
                x0, y, "[PAUSED]", Color(255, 220, 0, 255), s
            )
            y += 20

        # Recording indicator (red, only when recording)
        if self.renderer.recorder.is_recording:
            var rec_str = (
                "* REC  " + String(self.renderer.recorder.frame_count) + "f"
            )
            self.renderer.draw_text(
                x0 + 1, y + 1, rec_str, Color(0, 0, 0, 160), s
            )
            self.renderer.draw_text(x0, y, rec_str, Color(220, 40, 40, 255), s)

        # Widget command list first, so HUD text stays legible over panels.
        for rc in self.ui_rects:
            self.renderer.draw_rect(rc.x, rc.y, rc.w, rc.h, rc.color)
        for tx in self.ui_texts:
            self.renderer.draw_text(tx.x, tx.y, tx.text, tx.color, tx.scale)

        # Application-owned lines last, in cyan so they read as "not engine".
        for line in self.hud_extra:
            self.renderer.draw_text(x0 + 1, y + 1, line, Color(0, 0, 0, 160), s)
            self.renderer.draw_text(x0, y, line, Color(120, 230, 255, 255), s)
            y += 20

    def _draw_velocity_indicator(mut self, torso_pos: Vec3, vel_x: Float64):
        """Draw a velocity indicator arrow above the torso."""
        var arrow_start = Vec3(
            torso_pos.x, torso_pos.y, torso_pos.z + self.vel_arrow_height
        )
        var arrow_length = vel_x * self.vel_arrow_scale
        var arrow_end = Vec3(
            arrow_start.x + arrow_length, arrow_start.y, arrow_start.z
        )

        self.renderer.draw_line_3d(arrow_start, arrow_end, self.vel_color)

        if abs(arrow_length) > 0.03:
            var head_size = 0.04
            var direction = 1.0 if arrow_length > 0 else -1.0
            self.renderer.draw_line_3d(
                arrow_end,
                Vec3(
                    arrow_end.x - head_size * direction,
                    arrow_end.y,
                    arrow_end.z + head_size,
                ),
                self.vel_color,
            )
            self.renderer.draw_line_3d(
                arrow_end,
                Vec3(
                    arrow_end.x - head_size * direction,
                    arrow_end.y,
                    arrow_end.z - head_size,
                ),
                self.vel_color,
            )

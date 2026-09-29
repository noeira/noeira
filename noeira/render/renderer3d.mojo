"""GPU-Accelerated 3D Renderer.

Uses SDL3's GPU API with cross-platform shaders (MSL on Metal, SPIR-V on Vulkan)
for true 3D rendering with Blinn-Phong lighting, depth buffering, and procedural
checkerboard ground.
"""

from std.memory import Pointer, unsafe_memcpy, alloc
from std.math import sqrt, sin, cos, tan
from noeira.math3d import (
    Vec3 as Vec3Generic,
    Quat as QuatGeneric,
    Mat4 as Mat4Generic,
)
# ⚠ THE ONE PLACE THE HFIELD SAMPLE POSITIONS ARE SPELLED. `mj_rayHfield`
# builds the same surface for the physics; see that module's header for why a
# second copy here would be a silent divergence.
from noeira.physics3d.model.hfield_surface import (
    hfield_node_x, hfield_node_y, hfield_node_z,
)
from std.ffi import _get_dylib_function
from std.sys import CompilationTarget
from std.sys.info import size_of
from .sdl import (
    _null_ptr,
    untracked,
    Ptr,
    lib,
    c_char,
    c_float,
    c_int,
    # GPU types
    GPUDevice,
    GPUBuffer,
    GPUTransferBuffer,
    GPUTexture,
    GPUSampler,
    GPUShader,
    GPUGraphicsPipeline,
    GPUCommandBuffer,
    GPURenderPass,
    GPUCopyPass,
    # Create info structs
    GPUShaderCreateInfo,
    GPUGraphicsPipelineCreateInfo,
    GPUGraphicsPipelineTargetInfo,
    GPUBufferCreateInfo,
    GPUTransferBufferCreateInfo,
    GPUTextureCreateInfo,
    GPUSamplerCreateInfo,
    GPUTextureSamplerBinding,
    GPUColorTargetDescription,
    GPUColorTargetBlendState,
    GPUColorTargetInfo,
    GPUDepthStencilTargetInfo,
    GPUDepthStencilState,
    GPURasterizerState,
    GPUMultisampleState,
    GPUVertexInputState,
    GPUVertexBufferDescription,
    GPUVertexAttribute,
    GPUBufferBinding,
    GPUBufferRegion,
    GPUTransferBufferLocation,
    GPUTextureTransferInfo,
    GPUTextureRegion,
    GPUViewport,
    # Enums
    GPUPrimitiveType,
    GPULoadOp,
    GPUStoreOp,
    GPUShaderStage,
    GPUShaderFormat,
    GPUTextureFormat,
    GPUTextureType,
    GPUTextureUsageFlags,
    GPUBufferUsageFlags,
    GPUTransferBufferUsage,
    GPUVertexElementFormat,
    GPUVertexInputRate,
    GPUFillMode,
    GPUCullMode,
    GPUFrontFace,
    GPUCompareOp,
    GPUStencilOp,
    GPUStencilOpState,
    GPUSampleCount,
    GPUIndexElementSize,
    GPUBlendFactor,
    GPUBlendOp,
    GPUColorComponentFlags,
    GPUFilter,
    GPUSamplerMipmapMode,
    GPUSamplerAddressMode,
    PropertiesID,
    # SDL core
    Window,
    WindowFlags,
    Event,
    EventType,
    KeyboardEvent,
    MouseButtonEvent,
    MouseMotionEvent,
    MouseWheelEvent,
    Keycode,
    FColor,
    InitFlags,
    # Functions
    init,
    quit,
    create_window,
    destroy_window,
    poll_event,
    delay,
    destroy_gpu_device,
    claim_window_for_gpu_device,
    release_window_from_gpu_device,
    get_gpu_swapchain_texture_format,
    create_gpu_shader,
    create_gpu_graphics_pipeline,
    create_gpu_buffer,
    create_gpu_transfer_buffer,
    create_gpu_texture,
    create_gpu_sampler,
    acquire_gpu_command_buffer,
    wait_and_acquire_gpu_swapchain_texture,
    begin_gpu_render_pass,
    end_gpu_render_pass,
    begin_gpu_copy_pass,
    end_gpu_copy_pass,
    submit_gpu_command_buffer,
    bind_gpu_graphics_pipeline,
    set_gpu_viewport,
    set_gpu_scissor,
    bind_gpu_vertex_buffers,
    bind_gpu_index_buffer,
    draw_gpu_indexed_primitives,
    draw_gpu_primitives,
    push_gpu_vertex_uniform_data,
    push_gpu_fragment_uniform_data,
    bind_gpu_fragment_samplers,
    upload_to_gpu_buffer,
    upload_to_gpu_texture,
    generate_mipmaps_for_gpu_texture,
    map_gpu_transfer_buffer,
    unmap_gpu_transfer_buffer,
    release_gpu_buffer,
    release_gpu_transfer_buffer,
    release_gpu_texture,
    release_gpu_sampler,
    release_gpu_shader,
    release_gpu_graphics_pipeline,
    # Screenshot: GPU idle wait + texture download
    wait_for_gpu_idle,
    download_from_gpu_texture,
)
from .camera3d import Camera3D
from .types import Color
from .imgui import (
    ig_init,
    ig_shutdown,
    ig_new_frame,
    ig_prepare,
    ig_render,
    ig_process_event,
    ig_want_mouse,
    ig_want_keyboard,
    imgui_shim_available,
)
from .light import Light, LightMode
from .video_recorder import VideoRecorder
from .gpu_types import (
    SCENE_UNIFORMS_BYTES,
    OBJECT_UNIFORMS_BYTES,
    MAX_SCENE_LIGHTS,
    GPUVertex,
    SceneUniforms,
    ObjectUniforms,
    LineUniforms,
    ShadowUniforms,
    SkyboxUniforms,
    TextVertex,
    TextUniforms,
    MeshData,
    MeshHandle,
    CapsuleCacheEntry,
    MeshCacheEntry,
    TextureCacheEntry,
    SolidDrawCommand,
    mat4_to_gpu_f32,
    perspective_projection,
    ortho_projection,
    color_to_vec4,
    make_identity_f32,
)
from .sdl.sdl_keyboard import get_mod_state
from .sdl.sdl_rect import Rect
from .sdl.sdl_keycode import Keymod
from .stl_loader import load_stl
from .skn_loader import load_skn, SkinData
from .skinning import resolve_skin_bones, skin_pose
from .gpu_mesh import (
    generate_sphere,
    generate_box,
    generate_capsule,
    generate_cylinder,
    generate_ground,
)
from .gpu_shaders import (
    SOLID_VERTEX_MSL,
    SOLID_FRAGMENT_MSL,
    GROUND_VERTEX_MSL,
    GROUND_FRAGMENT_MSL,
    LINE_VERTEX_MSL,
    LINE_FRAGMENT_MSL,
    SHADOW_VERTEX_MSL,
    SHADOW_FRAGMENT_MSL,
    REFLECTION_FRAGMENT_MSL,
    SKYBOX_VERTEX_MSL,
    SKYBOX_FRAGMENT_MSL,
    TEXT_VERTEX_MSL,
    TEXT_FRAGMENT_MSL,
)
from .font_atlas import build_font_atlas_r8, glyph_uv, solid_uv
from .png_loader import load_png, load_texture_png, TextureData
from .gpu_shaders_spirv import load_spirv_shaders, SPIRVShaders

comptime Vec3 = Vec3Generic[DType.float64]
comptime Quat = QuatGeneric[DType.float64]
comptime Mat4 = Mat4Generic[DType.float64]


# Maximum line vertices per frame
comptime MAX_LINE_VERTICES = 512

# Maximum text characters per frame for HUD overlay.
#
# ⚠ THIS IS ALSO THE RECTANGLE BUDGET. `draw_rect` writes quads into the same
# buffer, so panels and buttons spend from the same pot as glyphs.
#
# Raised 512 → 2048 on 2026-08-03 for the viewer's 39-row task list. The old
# value left ~55 quads of headroom (the engine HUD plus four application lines
# already cost ~457), and one list blows through that several times over. Cost
# is a static 2048*4*32 = 256 KiB vertex buffer plus a 24 KiB index buffer,
# allocated once at init.
comptime MAX_TEXT_CHARS = 2048


# --- Line color entry for list storage ---


struct LineColorEntry(Copyable, Movable):
    """Stores RGBA color for a line segment."""

    var r: Float32
    var g: Float32
    var b: Float32
    var a: Float32

    def __init__(out self, color: Array[Float32, 4]):
        self.r = color[0]
        self.g = color[1]
        self.b = color[2]
        self.a = color[3]

    def __init__(out self, *, copy: Self):
        self.r = copy.r
        self.g = copy.g
        self.b = copy.b
        self.a = copy.a

    def __init__(out self, *, deinit move: Self):
        self.r = move.r
        self.g = move.g
        self.b = move.b
        self.a = move.a

    def to_inline_array(self) -> Array[Float32, 4]:
        var out = Array[Float32, 4](fill=Float32(0))
        out[0] = self.r
        out[1] = self.g
        out[2] = self.b
        out[3] = self.a
        return out^


struct HfieldCacheEntry(Movable):
    """A `<hfield>` surface: its GPU home and the revision it was built from.

    ⚠ THE GPU SIDE LIVES IN `mesh_cache`, exactly as `SkinCacheEntry`'s does —
    a heightfield is an ordinary indexed triangle mesh once its vertices are
    computed, so it is drawn by the existing mesh path and released by the
    existing `_release_model_caches`. What this entry adds is the state that
    tells a re-upload from a redraw.

    ⚠ THE TOPOLOGY NEVER CHANGES, ONLY THE HEIGHTS. `nrow`/`ncol` are fixed by
    the asset, so the INDEX buffer is uploaded once and only the vertex buffer
    is rewritten. That is what makes a per-episode terrain affordable: escape's
    grid is 80,000 triangles, and re-deriving its 240,000 indices every reset
    would cost far more than the elevations do.

    ⚠ `revision` IS THE WHOLE REASON THIS IS NOT `draw_skin`. A skin deforms
    every frame and re-uploads unconditionally; a heightfield changes only when
    the task rewrites it — once per episode for `quadruped escape`, never for a
    terrain loaded from a PNG. Uploading 1.3 MB at 60 Hz for a surface that is
    not moving is the shape this field exists to avoid.
    """

    var name: String
    var mesh_idx: Int
    var revision: Int
    var nrow: Int
    var ncol: Int

    var transfer: Ptr[GPUTransferBuffer, MutUntrackedOrigin]
    """Persistent upload staging. ⚠ ALLOCATED ONCE — see `SkinCacheEntry`."""
    var vbuf_bytes: Int

    var verts: List[GPUVertex]
    """Scratch for the rebuild. A field rather than a local for the same reason
    the skin's is, even though this runs per RESET and not per frame: 40,401
    vertices is not an allocation to make casually."""

    def __init__[
        tb_o: MutOrigin, //
    ](
        out self,
        var name: String,
        mesh_idx: Int,
        nrow: Int,
        ncol: Int,
        transfer: Ptr[GPUTransferBuffer, tb_o],
        vbuf_bytes: Int,
    ):
        self.name = name^
        self.mesh_idx = mesh_idx
        # ⚠ NOT 0. `revision` 0 is a legitimate value a caller can pass on the
        # very first frame, and starting equal to it would skip the initial
        # upload and draw a FLAT sheet until the first reset.
        self.revision = -1
        self.nrow = nrow
        self.ncol = ncol
        self.transfer = untracked(transfer)
        self.vbuf_bytes = vbuf_bytes
        self.verts = List[GPUVertex]()

    def __init__(out self, *, deinit move: Self):
        self.name = move.name^
        self.mesh_idx = move.mesh_idx
        self.revision = move.revision
        self.nrow = move.nrow
        self.ncol = move.ncol
        self.transfer = move.transfer
        self.vbuf_bytes = move.vbuf_bytes
        self.verts = move.verts^


struct SkinCacheEntry(Movable):
    """A loaded skin: its rest data, its bone->body map, and its GPU home.

    ⚠ THE GPU SIDE LIVES IN `mesh_cache`, not here. The posed skin is an
    ordinary indexed triangle mesh once it has been deformed, so it is drawn by
    the existing mesh path (`SolidDrawCommand.is_mesh`) and its buffers are
    released by the existing `_release_model_caches`. What this entry adds is
    the CPU state a mesh geom never needs: the rest vertices, the bones, and
    the scratch the per-frame deformation writes into.

    ⚠ THE SCRATCH IS FIELDS, NOT LOCALS, and that is the point. `draw_skin`
    runs every frame; allocating 24k vertices' worth of `List` per frame would
    cost more than the skinning does.
    """

    var name: String
    var skin: SkinData
    var bone_body: List[Int]
    var n_unbound: Int
    """Bones whose body name matched nothing. Reported once at load — the
    symptom is a collapsed region of the mesh, never an error."""

    var mesh_idx: Int
    """Index into `mesh_cache`. Its vertex buffer is rewritten every frame; its
    index buffer never changes, since deformation moves vertices and not
    topology."""

    var transfer: Ptr[GPUTransferBuffer, MutUntrackedOrigin]
    """Persistent upload staging. ⚠ ALLOCATED ONCE. Creating a transfer buffer
    per frame is the shape that has burned this codebase before."""
    var vbuf_bytes: Int

    var posed: List[Float32]
    var normals: List[Float32]
    var verts: List[GPUVertex]

    def __init__[
        tb_o: MutOrigin, //
    ](
        out self,
        var name: String,
        var skin: SkinData,
        var bone_body: List[Int],
        n_unbound: Int,
        mesh_idx: Int,
        transfer: Ptr[GPUTransferBuffer, tb_o],
        vbuf_bytes: Int,
    ):
        self.name = name^
        self.skin = skin^
        self.bone_body = bone_body^
        self.n_unbound = n_unbound
        self.mesh_idx = mesh_idx
        self.transfer = untracked(transfer)
        self.vbuf_bytes = vbuf_bytes
        self.posed = List[Float32]()
        self.normals = List[Float32]()
        self.verts = List[GPUVertex]()

    def __init__(out self, *, deinit move: Self):
        self.name = move.name^
        self.skin = move.skin^
        self.bone_body = move.bone_body^
        self.n_unbound = move.n_unbound
        self.mesh_idx = move.mesh_idx
        self.transfer = move.transfer
        self.vbuf_bytes = move.vbuf_bytes
        self.posed = move.posed^
        self.normals = move.normals^
        self.verts = move.verts^


@fieldwise_init
struct RendererHandoff(Copyable, Movable):
    """The window, device and every MODEL-INDEPENDENT GPU resource, detached
    from one `Renderer3D` so the next one can adopt them.

    WHY THIS EXISTS. A tool that swaps models — the dm_control viewer — used to
    destroy the whole renderer per switch, and with it the SDL window. A
    destroyed window is re-created wherever the OS decides, which on a
    multi-monitor desktop means it JUMPS BACK to the primary display: the user
    drags the viewer to an external screen, picks another robot, and the window
    reappears on the laptop. It also blinks, drops the ImGui context (and with
    it the sidebar's expanded/scrolled state), and pays for shader compilation,
    the font atlas and the static meshes all over again.

    ⚠ MODEL-INDEPENDENT IS THE WHOLE CRITERION. Everything here is built from
    the swapchain format and the window size, never from the model: pipelines,
    depth/shadow targets, the sphere/box/ground meshes, the line and text
    buffers, the 1x1 default texture. What is NOT here — the capsule, cylinder,
    STL and PNG caches — is keyed by the model's geoms and is released by
    `detach`, because carrying it over would hand the next model the previous
    one's limb meshes.

    ⚠ `shadow_size` COMES FROM THE MODEL (`<visual shadowsize=>`), so the
    shadow map is carried but re-created when the sizes disagree. It rides here
    rather than in the caches because most models agree and re-allocating a
    4096² depth texture per switch is the one adoption that would visibly cost
    something.
    """

    var window: Ptr[Window, MutUntrackedOrigin]
    var device: Ptr[GPUDevice, MutUntrackedOrigin]
    var swapchain_format: GPUTextureFormat

    var width: Int
    var height: Int
    """The window's size AT DETACH, not the size the renderer was built with.
    The user may have resized it, and the adopting renderer has to agree with
    the swapchain or it re-creates the depth texture on its first frame."""

    var solid_pipeline: Ptr[GPUGraphicsPipeline, MutUntrackedOrigin]
    var ground_pipeline: Ptr[GPUGraphicsPipeline, MutUntrackedOrigin]
    var line_pipeline: Ptr[GPUGraphicsPipeline, MutUntrackedOrigin]
    var shadow_pipeline: Ptr[GPUGraphicsPipeline, MutUntrackedOrigin]
    var reflection_pipeline: Ptr[GPUGraphicsPipeline, MutUntrackedOrigin]
    var skybox_pipeline: Ptr[GPUGraphicsPipeline, MutUntrackedOrigin]
    var text_pipeline: Ptr[GPUGraphicsPipeline, MutUntrackedOrigin]

    var depth_texture: Ptr[GPUTexture, MutUntrackedOrigin]
    var shadow_map: Ptr[GPUTexture, MutUntrackedOrigin]
    var shadow_sampler: Ptr[GPUSampler, MutUntrackedOrigin]
    var shadow_size: Int

    var sphere_mesh: MeshHandle
    var box_mesh: MeshHandle
    var ground_mesh: MeshHandle

    var line_vertex_buffer: Ptr[GPUBuffer, MutUntrackedOrigin]
    var line_transfer_buffer: Ptr[GPUTransferBuffer, MutUntrackedOrigin]

    var font_atlas_tex: Ptr[GPUTexture, MutUntrackedOrigin]
    var font_sampler: Ptr[GPUSampler, MutUntrackedOrigin]
    var text_vertex_buffer: Ptr[GPUBuffer, MutUntrackedOrigin]
    var text_index_buffer: Ptr[GPUBuffer, MutUntrackedOrigin]
    var text_transfer_buffer: Ptr[GPUTransferBuffer, MutUntrackedOrigin]

    var default_texture: Ptr[GPUTexture, MutUntrackedOrigin]
    var default_tex_sampler: Ptr[GPUSampler, MutUntrackedOrigin]

    var imgui_on: Bool
    """Whether the ImGui backend is still attached to this window+device.

    It survives a handoff untouched — the context, its font texture and its
    pipeline all live on the device being carried over — so the adopting
    renderer must NOT call `ig_init` again. `imgui_init` returns early on
    `imgui_on`, which is exactly that."""


@fieldwise_init
struct PipView(Copyable, Movable):
    """A picture-in-picture view: the scene drawn again from `camera` into
    the window rectangle `(x, y, w, h)` (pixels, origin top-left), on top
    of the main view — a teleoperator's second and third eye.

    Set per frame through `Renderer3D.set_pip_views` (the model renderer
    computes body-attached camera poses from that frame's FK). Each view is
    its own render pass: colour LOADED (the main view stays underneath),
    depth CLEARED (the inset must not depth-test against the main scene),
    viewport AND scissor set to the rectangle so the skybox and ground do
    not paint outside it. The shadow map is light-space and shared. No
    reflections, lines or HUD text in an inset.
    """

    var camera: Camera3D
    var x: Int
    var y: Int
    var w: Int
    var h: Int


struct Renderer3D(Movable):
    """GPU-accelerated 3D renderer using SDL3 GPU API.

    Uses Metal (MSL) or Vulkan (SPIR-V) shaders for Blinn-Phong lit solid
    rendering with procedural checkerboard ground and flat-color line drawing.
    """

    # SDL3 handles. Mojo nightly removed nullable Pointer; these
    # GPU resources are populated by init() and need to be wrapped in
    # Optional[] so the field can start out empty.
    var window: Optional[Ptr[Window, MutUntrackedOrigin]]
    var device: Optional[Ptr[GPUDevice, MutUntrackedOrigin]]

    # Pipelines
    var solid_pipeline: Optional[Ptr[GPUGraphicsPipeline, MutUntrackedOrigin]]
    var ground_pipeline: Optional[Ptr[GPUGraphicsPipeline, MutUntrackedOrigin]]
    var line_pipeline: Optional[Ptr[GPUGraphicsPipeline, MutUntrackedOrigin]]
    var shadow_pipeline: Optional[Ptr[GPUGraphicsPipeline, MutUntrackedOrigin]]
    var reflection_pipeline: Optional[Ptr[GPUGraphicsPipeline, MutUntrackedOrigin]]
    var skybox_pipeline: Optional[Ptr[GPUGraphicsPipeline, MutUntrackedOrigin]]

    # Depth buffer
    var depth_texture: Optional[Ptr[GPUTexture, MutUntrackedOrigin]]

    # Shadow mapping resources
    var shadow_map: Optional[Ptr[GPUTexture, MutUntrackedOrigin]]
    var shadow_sampler: Optional[Ptr[GPUSampler, MutUntrackedOrigin]]
    var shadow_uniforms: ShadowUniforms
    var ground_z: Float64
    var ground_reflectance: Float32
    """The ground plane material's `reflectance`: the mirror pass's blend
    weight, and 0 skips the pass. MuJoCo draws a reflection only when the
    plane's material has one (`render_gl3.c`, `reflectance > 0`); the pass ran
    at a hard-coded 0.2 on every ground, which put a ghost of the LIBERO Panda
    under a floor whose material says 0.0."""

    # Cached static meshes
    var sphere_mesh: Optional[MeshHandle]
    var box_mesh: Optional[MeshHandle]
    var ground_mesh: Optional[MeshHandle]
    var capsule_cache: List[CapsuleCacheEntry]
    var cylinder_cache: List[
        CapsuleCacheEntry
    ]  # Same cache type (radius, half_height)
    var mesh_cache: List[MeshCacheEntry]
    var mesh_failed: List[String]
    """Meshes whose load or upload raised, so the diagnostic prints ONCE.

    ⚠⚠ `draw_mesh` USED TO `return` SILENTLY ON A FAILURE, and a mesh geom
    that draws nothing is indistinguishable from one the model never
    declared — the viewer just shows a robot with parts missing. That is the
    most expensive kind of bug this renderer can have, because the model, the
    parse, the asset table and the kinematics can all be verified correct
    while the picture stays wrong. Printing per frame would be 60 lines a
    second, so the name is recorded here and reported once."""
    var hfield_cache: List[HfieldCacheEntry]
    var skin_cache: List[SkinCacheEntry]
    """Deformable skins. Separate from `mesh_cache` because a skin carries CPU
    state a rigid mesh never needs; its GPU buffers still live in `mesh_cache`.
    See `SkinCacheEntry`."""

    # Texture cache for PNG textures
    var texture_cache: List[TextureCacheEntry]
    var default_texture: Optional[Ptr[GPUTexture, MutUntrackedOrigin]]  # 1x1 white
    var default_tex_sampler: Optional[Ptr[GPUSampler, MutUntrackedOrigin]]

    # Dynamic line buffer
    var line_vertex_data: List[Float32]  # x,y,z per vertex
    var line_colors: List[LineColorEntry]  # color per segment (2 verts)
    var line_vertex_buffer: Optional[Ptr[GPUBuffer, MutUntrackedOrigin]]
    var line_transfer_buffer: Optional[Ptr[GPUTransferBuffer, MutUntrackedOrigin]]

    # Text (HUD overlay) pipeline and resources
    var text_pipeline: Optional[Ptr[GPUGraphicsPipeline, MutUntrackedOrigin]]
    var font_atlas_tex: Optional[Ptr[GPUTexture, MutUntrackedOrigin]]
    var font_sampler: Optional[Ptr[GPUSampler, MutUntrackedOrigin]]
    var text_vertex_buffer: Optional[Ptr[GPUBuffer, MutUntrackedOrigin]]
    var text_index_buffer: Optional[Ptr[GPUBuffer, MutUntrackedOrigin]]
    var text_transfer_buffer: Optional[Ptr[GPUTransferBuffer, MutUntrackedOrigin]]
    var text_vertex_data: List[
        Float32
    ]  # packed TextVertex fields (8 floats/vertex)
    var text_uniforms: TextUniforms

    # Deferred draw commands
    var solid_draws: List[SolidDrawCommand]
    # Extra views drawn after the main pass — see `PipView`.
    var pip_views: List[PipView]
    var ground_uniforms: ObjectUniforms
    var has_ground: Bool
    var ground_texture_idx: Int  # -1 = no texture (use checker/solid)

    # Camera and scene
    var camera: Camera3D
    var width: Int
    var height: Int
    var background_color: Color
    var scene_uniforms: SceneUniforms

    # Skybox
    var skybox_uniforms: SkyboxUniforms
    var draw_skybox: Bool

    # Configurable light parameters (up to 4 lights)
    var lights: List[Light]
    var headlight_active: Bool
    """MuJoCo's headlight: a directional light at the eye, aimed along the
    view (`mjv_defaultVisual`: ambient .1, diffuse .4, specular .5). OFF on a
    bare `Renderer3D`; `ModelRenderer` calls `set_headlight` from the model,
    where it is ON unless `<headlight active="0">` — a model with no
    `<visual><headlight>` still has one. Without it every surface the
    model's own lights do not reach is black — the LIBERO walls were."""
    var headlight_ambient: Array[Float32, 3]
    var headlight_diffuse: Array[Float32, 3]
    var headlight_specular: Array[Float32, 3]

    # Swapchain format
    var swapchain_format: GPUTextureFormat

    # State
    var initialized: Bool
    var should_quit: Bool
    # Most recent key the renderer's own bindings did NOT claim, 0 when none.
    # The event switch below decodes a fixed set (ESC, 1-9, SPACE, RIGHT, R,
    # S, V) and swallows everything else; without this an application has no
    # way to bind a key of its own. Read it with `take_key`, which clears.
    var last_key: Int
    # Pointer state for screen-space UI. The pump already receives these
    # coordinates and used to keep only a button bool, so nothing could
    # hit-test a widget.
    var mouse_x: Float32
    var mouse_y: Float32
    var mouse_clicked: Bool
    var text_budget_warned: Bool
    """One-shot latch so an exhausted quad budget is reported, not swallowed."""
    var text_input_mode: Bool
    """When set, the renderer claims NO keyboard shortcut and forwards every
    keycode to the application via `take_key`.

    ⚠ WITHOUT THIS A TEXT FIELD IS UNUSABLE, not merely awkward. The bindings
    below swallow R, S, V, SPACE, 1-9 and ESC before the app ever sees them, so
    typing "reacher" would reset the camera, save a screenshot and start a
    video recording. ESC is included deliberately: quitting the window mid-word
    is worse than making the app responsible for unfocusing."""
    var pointer_claimed: Bool
    """An OVERLAY that owns the pointer without ImGui knowing it does.

    ⚠⚠ `ig_want_mouse()` IS NOT A COMPLETE ANSWER, and the gap is not
    theoretical. ImGuizmo draws into a window created with
    `ImGuiWindowFlags_NoInputs`, so ImGui truthfully reports that it does NOT
    want the mouse while a gizmo handle is being dragged — and the drag then
    orbits the camera at the same time as it moves the part. Anything
    hit-testing the viewport for itself sets this each frame and the press
    latch below reads it beside `ig_mouse`.

    ⚠ WRITTEN BY THE APPLICATION, ONE FRAME AHEAD, which is the same staleness
    `ig_want_mouse` already has and for the same reason: the overlay can only
    answer after its own layout has run."""
    var ui_sidebar_width: Int
    """Pixels reserved on the LEFT for screen-space UI. 0 = full-window scene.

    ⚠ THIS RESERVES SPACE, it does not merely draw over it. The 3D phases get a
    viewport inset by this much and the camera's aspect is corrected to match,
    so the scene is squeezed rather than stretched or cropped. The HUD/text
    phase then runs at FULL window size, which is what lets widgets live in the
    reserved strip while the scene keeps the remainder to itself."""
    var draw_grid: Bool
    var draw_axes: Bool

    # Visual settings from MuJoCo <visual> section
    var shadow_size: Int  # shadow map resolution (default 4096)
    var fog_start: Float32  # fog start distance
    var fog_end: Float32  # fog end distance

    # Camera switching (set by check_quit, read by ModelRenderer)
    var camera_switch_request: Int  # -1 = none, 0-8 = switch to camera N

    # Mouse state for camera interaction (wired in check_quit)
    var mouse_left_down: Bool
    var mouse_right_down: Bool

    # Pause / step state (read by simulation loop)
    var is_paused: Bool
    var step_once: Bool  # True for exactly one frame after → pressed while paused

    # Screenshot (S key)
    var screenshot_requested: Bool
    var screenshot_counter: Int

    # Video recording (V key / programmatic API)
    var recorder: VideoRecorder
    # Reusable GPU download buffer for recording (allocated at start, freed at stop)
    var recording_tb: Optional[Ptr[GPUTransferBuffer, MutUntrackedOrigin]]
    var recording_tb_size: Int  # current allocation size in bytes

    # Default camera position for R-key reset
    var default_eye: Vec3
    var default_target: Vec3

    # Dear ImGui overlay (see noeira/render/imgui/). OPT-IN: nothing here
    # touches the ImGui shim unless `imgui_init()` succeeded, so the FFI
    # dependency stays confined to applications that ask for it.
    var capture_scene_only: Bool
    """Crop screenshots and recordings to the 3D viewport.

    ⚠ CROPPED ON THE CPU, NOT ON THE GPU. Narrowing the download region
    instead would be the obvious move and is a trap: SDL_GPU texture downloads
    carry row-pitch alignment rules, so an arbitrary `ui_sidebar_width` can
    produce a skewed image rather than an error. The full swapchain is
    downloaded as before and the columns are dropped in numpy, which has no
    alignment to satisfy.

    No effect when `ui_sidebar_width` is 0 — with no reserved strip there is
    nothing to crop away."""

    var imgui_on: Bool
    var imgui_frame_open: Bool
    """True between `imgui_new_frame()` and the `ImGui::Render()` inside
    `end_frame`.

    ⚠ ONCE A FRAME IS OPEN IT MUST BE CLOSED, on every path. ImGui asserts on
    the *next* `NewFrame` if the previous one was never rendered, so the close
    happens before `end_frame`'s early return on a missing swapchain texture
    rather than beside the draw call."""

    def __init__(
        out self,
        width: Int = 800,
        height: Int = 450,
        camera: Camera3D = Camera3D(),
        draw_grid: Bool = True,
        draw_axes: Bool = True,
        lights: List[Light] = List[Light](),
        light_dir_x: Float32 = 0.3,
        light_dir_y: Float32 = -0.4,
        light_dir_z: Float32 = -0.8,
        light_color_r: Float32 = 1.0,
        light_color_g: Float32 = 0.98,
        light_color_b: Float32 = 0.95,
        light_ambient: Float32 = 0.25,
        shadow_size: Int = 4096,
        fog_start: Float32 = 0.0,
        fog_end: Float32 = 0.0,
    ) raises:
        self.width = width
        self.height = height
        self.background_color = Color(32, 32, 48, 255)
        self.draw_grid = draw_grid
        self.draw_axes = draw_axes
        self.should_quit = False
        self.last_key = 0
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_clicked = False
        self.text_budget_warned = False
        self.ui_sidebar_width = 0
        self.pointer_claimed = False
        self.text_input_mode = False
        self.pip_views = List[PipView]()
        self.initialized = False

        # Copy camera
        self.camera = Camera3D(
            eye=camera.eye,
            target=camera.target,
            up=camera.up,
            fov=camera.fov * 180.0 / 3.14159265358979,
            aspect=Float64(width) / Float64(height),
            near=camera.near,
            far=camera.far,
            screen_width=width,
            screen_height=height,
        )

        # Null handles (populated lazily in init())
        self.window = None
        self.device = None
        self.solid_pipeline = None
        self.ground_pipeline = None
        self.line_pipeline = None
        self.shadow_pipeline = None
        self.reflection_pipeline = None
        self.skybox_pipeline = None
        self.depth_texture = None
        self.shadow_map = None
        self.shadow_sampler = None
        self.shadow_uniforms = ShadowUniforms()
        self.ground_z = 0.0
        self.ground_reflectance = 0.0
        self.line_vertex_buffer = None
        self.line_transfer_buffer = None
        self.text_pipeline = None
        self.font_atlas_tex = None
        self.font_sampler = None
        self.text_vertex_buffer = None
        self.text_index_buffer = None
        self.text_transfer_buffer = None
        self.text_vertex_data = List[Float32]()
        self.text_uniforms = TextUniforms()
        self.swapchain_format = (
            GPUTextureFormat.GPU_TEXTUREFORMAT_B8G8R8A8_UNORM
        )

        # Meshes (uploaded lazily in init())
        self.sphere_mesh = None
        self.box_mesh = None
        self.ground_mesh = None
        self.capsule_cache = List[CapsuleCacheEntry]()
        self.cylinder_cache = List[CapsuleCacheEntry]()
        self.mesh_cache = List[MeshCacheEntry]()
        self.mesh_failed = List[String]()
        self.hfield_cache = List[HfieldCacheEntry]()
        self.skin_cache = List[SkinCacheEntry]()

        # Texture cache
        self.texture_cache = List[TextureCacheEntry]()
        self.default_texture = None
        self.default_tex_sampler = None

        # Line data
        self.line_vertex_data = List[Float32]()
        self.line_colors = List[LineColorEntry]()

        # Draw commands
        self.solid_draws = List[SolidDrawCommand]()
        self.ground_uniforms = ObjectUniforms()
        self.has_ground = False
        self.ground_texture_idx = -1

        self.scene_uniforms = SceneUniforms()
        self.skybox_uniforms = SkyboxUniforms()
        self.draw_skybox = False

        # Visual settings
        self.shadow_size = shadow_size
        self.fog_start = fog_start
        self.fog_end = fog_end

        # Store configurable light parameters (up to 4 lights)
        self.camera_switch_request = -1
        self.mouse_left_down = False
        self.mouse_right_down = False
        self.is_paused = False
        self.step_once = False
        self.screenshot_requested = False
        self.screenshot_counter = 0
        self.recorder = VideoRecorder()
        self.recording_tb = None
        self.recording_tb_size = 0
        self.default_eye = camera.eye
        self.default_target = camera.target
        self.capture_scene_only = True
        self.imgui_on = False
        self.imgui_frame_open = False
        # OFF for a bare renderer: a hand-built scene passes its own `lights`
        # and was tuned without one. `ModelRenderer` turns it on from the model.
        self.headlight_active = False
        self.headlight_ambient = Array[Float32, 3](fill=Float32(0.1))
        self.headlight_diffuse = Array[Float32, 3](fill=Float32(0.4))
        self.headlight_specular = Array[Float32, 3](fill=Float32(0.5))
        if len(lights) > 0:
            self.lights = lights.copy()
        else:
            # Create default light from individual params (backward compatibility)
            self.lights = List[Light]()
            self.lights.append(
                Light(
                    mode=0,  # LIGHT_DIRECTIONAL
                    dir_x=Float64(light_dir_x),
                    dir_y=Float64(light_dir_y),
                    dir_z=Float64(light_dir_z),
                    color_r=Float64(light_color_r),
                    color_g=Float64(light_color_g),
                    color_b=Float64(light_color_b),
                    ambient=Float64(light_ambient),
                    specular_intensity=0.3,
                    specular_exponent=32.0,
                    cast_shadow=True,
                )
            )

    def __init__(out self, *, deinit move: Self):
        self.window = move.window^
        self.device = move.device^
        self.solid_pipeline = move.solid_pipeline^
        self.ground_pipeline = move.ground_pipeline^
        self.line_pipeline = move.line_pipeline^
        self.shadow_pipeline = move.shadow_pipeline^
        self.reflection_pipeline = move.reflection_pipeline^
        self.skybox_pipeline = move.skybox_pipeline^
        self.depth_texture = move.depth_texture^
        self.shadow_map = move.shadow_map^
        self.shadow_sampler = move.shadow_sampler^
        self.shadow_uniforms = move.shadow_uniforms
        self.ground_z = move.ground_z
        self.ground_reflectance = move.ground_reflectance
        self.sphere_mesh = move.sphere_mesh^
        self.box_mesh = move.box_mesh^
        self.ground_mesh = move.ground_mesh^
        self.capsule_cache = move.capsule_cache^
        self.cylinder_cache = move.cylinder_cache^
        self.mesh_cache = move.mesh_cache^
        self.mesh_failed = move.mesh_failed^
        self.hfield_cache = move.hfield_cache^
        self.skin_cache = move.skin_cache^
        self.texture_cache = move.texture_cache^
        self.default_texture = move.default_texture^
        self.default_tex_sampler = move.default_tex_sampler^
        self.line_vertex_data = move.line_vertex_data^
        self.line_colors = move.line_colors^
        self.line_vertex_buffer = move.line_vertex_buffer^
        self.line_transfer_buffer = move.line_transfer_buffer^
        self.text_pipeline = move.text_pipeline^
        self.font_atlas_tex = move.font_atlas_tex^
        self.font_sampler = move.font_sampler^
        self.text_vertex_buffer = move.text_vertex_buffer^
        self.text_index_buffer = move.text_index_buffer^
        self.text_transfer_buffer = move.text_transfer_buffer^
        self.text_vertex_data = move.text_vertex_data^
        self.text_uniforms = move.text_uniforms
        self.solid_draws = move.solid_draws^
        self.ground_uniforms = move.ground_uniforms
        self.has_ground = move.has_ground
        self.ground_texture_idx = move.ground_texture_idx
        self.camera = move.camera^
        self.width = move.width
        self.height = move.height
        self.background_color = move.background_color
        self.scene_uniforms = move.scene_uniforms
        self.skybox_uniforms = move.skybox_uniforms
        self.draw_skybox = move.draw_skybox
        self.swapchain_format = move.swapchain_format
        self.lights = move.lights^
        self.headlight_active = move.headlight_active
        self.headlight_ambient = move.headlight_ambient^
        self.headlight_diffuse = move.headlight_diffuse^
        self.headlight_specular = move.headlight_specular^
        self.shadow_size = move.shadow_size
        self.fog_start = move.fog_start
        self.fog_end = move.fog_end
        self.camera_switch_request = move.camera_switch_request
        self.mouse_left_down = move.mouse_left_down
        self.mouse_right_down = move.mouse_right_down
        self.is_paused = move.is_paused
        self.step_once = move.step_once
        self.screenshot_requested = move.screenshot_requested
        self.screenshot_counter = move.screenshot_counter
        self.recorder = move.recorder^
        self.recording_tb = move.recording_tb^
        self.recording_tb_size = move.recording_tb_size
        self.default_eye = move.default_eye
        self.default_target = move.default_target
        self.capture_scene_only = move.capture_scene_only
        self.imgui_on = move.imgui_on
        self.imgui_frame_open = move.imgui_frame_open
        self.initialized = move.initialized
        self.should_quit = move.should_quit
        self.last_key = move.last_key
        self.mouse_x = move.mouse_x
        self.mouse_y = move.mouse_y
        self.mouse_clicked = move.mouse_clicked
        self.text_budget_warned = move.text_budget_warned
        self.ui_sidebar_width = move.ui_sidebar_width
        self.pointer_claimed = move.pointer_claimed
        self.pip_views = move.pip_views^
        self.text_input_mode = move.text_input_mode
        self.draw_grid = move.draw_grid
        self.draw_axes = move.draw_axes

    @staticmethod
    def _shader_format() -> GPUShaderFormat:
        """Return the shader format for the current platform."""
        comptime if CompilationTarget.is_macos():
            return GPUShaderFormat.GPU_SHADERFORMAT_MSL
        elif CompilationTarget.is_linux():
            return GPUShaderFormat.GPU_SHADERFORMAT_SPIRV
        else:
            comptime assert False, "Unsupported platform for Renderer3D"

    def init(mut self, mut title: String) raises:
        """Initialize SDL3, GPU device, pipelines, and static meshes."""
        self.init(title, None)

    def init(
        mut self, mut title: String, adopt: Optional[RendererHandoff]
    ) raises:
        """Initialize, or ADOPT a `RendererHandoff` from a previous renderer.

        Adopting skips every step below: the window, the device and all the
        model-independent GPU resources already exist and are simply re-pointed
        at. That is what lets a model swap keep the same window — same monitor,
        same position, same size, no blink — instead of destroying it and
        letting the OS re-place the replacement. See `RendererHandoff`.
        """
        if adopt:
            self._adopt(adopt.value())
            return

        # 1. Init SDL3
        init(InitFlags.INIT_VIDEO)

        # 2. Create window
        self.window = untracked(create_window(
            title, c_int(self.width), c_int(self.height), WindowFlags(0)
        ))

        # 3. Create GPU device (MSL on macOS, SPIR-V on Linux).
        # SDL3 docs: name=NULL means "auto-select driver". Mojo nightly
        # rejects the comptime `unsafe_from_address=0` literal; _null_ptr
        # uses the runtime-Int overload to produce a real NULL pointer
        # at the C ABI boundary.
        self.device = untracked(
            _get_dylib_function[
                lib,
                "SDL_CreateGPUDevice",
                def(
                    GPUShaderFormat, Bool, Ptr[c_char, ImmutAnyOrigin]
                ) thin -> Ptr[GPUDevice, MutAnyOrigin],
            ]()(
                Self._shader_format(),
                True,
                _null_ptr[c_char, ImmutAnyOrigin](),
            )
        )

        # 4. Claim window
        claim_window_for_gpu_device(self.device.value(), self.window.value())

        # 5. Get swapchain format
        self.swapchain_format = get_gpu_swapchain_texture_format(
            self.device.value(), self.window.value()
        )

        # 6. Create shaders and pipelines
        self._create_pipelines()

        # 7. Create depth texture
        self._create_depth_texture()

        # 8. Create shadow map resources
        self._create_shadow_resources()

        # 9. Generate and upload static meshes
        self._upload_static_meshes()

        # 10. Allocate line buffers
        self._create_line_buffers()

        # 11. Create text rendering resources (font atlas, pipeline, buffers)
        self._create_text_resources()

        # 12. Create default 1x1 white texture for untextured objects
        self._create_default_texture()

        self.initialized = True

    def _adopt(mut self, h: RendererHandoff) raises:
        """Take over a previous renderer's window, device and shared GPU
        resources. The counterpart of `detach`."""
        self.window = h.window
        self.device = h.device
        self.swapchain_format = h.swapchain_format

        self.solid_pipeline = h.solid_pipeline
        self.ground_pipeline = h.ground_pipeline
        self.line_pipeline = h.line_pipeline
        self.shadow_pipeline = h.shadow_pipeline
        self.reflection_pipeline = h.reflection_pipeline
        self.skybox_pipeline = h.skybox_pipeline
        self.text_pipeline = h.text_pipeline

        self.sphere_mesh = h.sphere_mesh.copy()
        self.box_mesh = h.box_mesh.copy()
        self.ground_mesh = h.ground_mesh.copy()

        self.line_vertex_buffer = h.line_vertex_buffer
        self.line_transfer_buffer = h.line_transfer_buffer

        self.font_atlas_tex = h.font_atlas_tex
        self.font_sampler = h.font_sampler
        self.text_vertex_buffer = h.text_vertex_buffer
        self.text_index_buffer = h.text_index_buffer
        self.text_transfer_buffer = h.text_transfer_buffer

        self.default_texture = h.default_texture
        self.default_tex_sampler = h.default_tex_sampler

        self.imgui_on = h.imgui_on
        self.imgui_frame_open = False

        # ⚠ THE WINDOW'S SIZE WINS, not the size this renderer asked for. The
        # user may have resized it since the last model was built; disagreeing
        # with the live swapchain would make `render_frame`'s resize branch fire
        # on the first frame and throw away the depth texture we just adopted.
        self.width = h.width
        self.height = h.height
        self.camera.set_screen_size(self.scene_width(), self.height)
        self.depth_texture = h.depth_texture

        # Shadow map resolution is the one adopted resource the MODEL chooses
        # (`<visual shadowsize=>`), so a disagreement means re-creating it.
        if self.shadow_size == h.shadow_size:
            self.shadow_map = h.shadow_map
            self.shadow_sampler = h.shadow_sampler
        else:
            release_gpu_texture(self.device.value(), h.shadow_map)
            release_gpu_sampler(self.device.value(), h.shadow_sampler)
            self._create_shadow_resources()

        self.initialized = True

    def detach(mut self) raises -> RendererHandoff:
        """Release only the MODEL-SPECIFIC GPU state and hand the rest over.

        Leaves `self` uninitialized, so the usual `close()` on the way out is a
        no-op and the window survives the renderer that opened it. The caller
        now owns the handoff and MUST either pass it to another renderer's
        `init` or end it with `close_handoff` — nothing else will free the
        device.

        ⚠ NOT A CHEAPER `close()`. `close()` still exists and still tears
        everything down; this is the switch path only.
        """
        # Recording is per-renderer bookkeeping (frame counter, output file),
        # and the next model is a different video. Stop rather than carry.
        if self.recorder.is_recording:
            self.recorder.stop()
        if self.recording_tb:
            release_gpu_transfer_buffer(
                self.device.value(), self.recording_tb.value()
            )
            self.recording_tb = None
            self.recording_tb_size = 0

        self._release_model_caches()

        var h = RendererHandoff(
            window=self.window.value(),
            device=self.device.value(),
            swapchain_format=self.swapchain_format,
            width=self.width,
            height=self.height,
            solid_pipeline=self.solid_pipeline.value(),
            ground_pipeline=self.ground_pipeline.value(),
            line_pipeline=self.line_pipeline.value(),
            shadow_pipeline=self.shadow_pipeline.value(),
            reflection_pipeline=self.reflection_pipeline.value(),
            skybox_pipeline=self.skybox_pipeline.value(),
            text_pipeline=self.text_pipeline.value(),
            depth_texture=self.depth_texture.value(),
            shadow_map=self.shadow_map.value(),
            shadow_sampler=self.shadow_sampler.value(),
            shadow_size=self.shadow_size,
            sphere_mesh=self.sphere_mesh.value().copy(),
            box_mesh=self.box_mesh.value().copy(),
            ground_mesh=self.ground_mesh.value().copy(),
            line_vertex_buffer=self.line_vertex_buffer.value(),
            line_transfer_buffer=self.line_transfer_buffer.value(),
            font_atlas_tex=self.font_atlas_tex.value(),
            font_sampler=self.font_sampler.value(),
            text_vertex_buffer=self.text_vertex_buffer.value(),
            text_index_buffer=self.text_index_buffer.value(),
            text_transfer_buffer=self.text_transfer_buffer.value(),
            default_texture=self.default_texture.value(),
            default_tex_sampler=self.default_tex_sampler.value(),
            imgui_on=self.imgui_on,
        )

        # ⚠ DROP EVERY HANDLE, don't merely lower the flag. `close()` guards on
        # `initialized`, but a stray `imgui_close()` or a future teardown path
        # reading a still-populated field would release an object the NEXT
        # renderer is drawing with — a use-after-free that only shows up as a
        # corrupted frame.
        self.initialized = False
        self.imgui_on = False
        self.imgui_frame_open = False
        self.window = None
        self.device = None
        self.solid_pipeline = None
        self.ground_pipeline = None
        self.line_pipeline = None
        self.shadow_pipeline = None
        self.reflection_pipeline = None
        self.skybox_pipeline = None
        self.text_pipeline = None
        self.depth_texture = None
        self.shadow_map = None
        self.shadow_sampler = None
        self.sphere_mesh = None
        self.box_mesh = None
        self.ground_mesh = None
        self.line_vertex_buffer = None
        self.line_transfer_buffer = None
        self.font_atlas_tex = None
        self.font_sampler = None
        self.text_vertex_buffer = None
        self.text_index_buffer = None
        self.text_transfer_buffer = None
        self.default_texture = None
        self.default_tex_sampler = None
        return h^

    @staticmethod
    def close_handoff(var h: RendererHandoff) raises:
        """Tear down a handoff nobody adopted — the end of a switch chain.

        The viewer calls this when the LAST model's window is closed for real,
        because by then the window and device belong to the handoff rather than
        to any live renderer.
        """
        if h.imgui_on:
            ig_shutdown()

        release_gpu_buffer(h.device, h.sphere_mesh.vertex_buffer)
        release_gpu_buffer(h.device, h.sphere_mesh.index_buffer)
        release_gpu_buffer(h.device, h.box_mesh.vertex_buffer)
        release_gpu_buffer(h.device, h.box_mesh.index_buffer)
        release_gpu_buffer(h.device, h.ground_mesh.vertex_buffer)
        release_gpu_buffer(h.device, h.ground_mesh.index_buffer)

        release_gpu_texture(h.device, h.default_texture)
        release_gpu_sampler(h.device, h.default_tex_sampler)

        release_gpu_buffer(h.device, h.line_vertex_buffer)
        release_gpu_transfer_buffer(h.device, h.line_transfer_buffer)

        release_gpu_texture(h.device, h.depth_texture)
        release_gpu_texture(h.device, h.shadow_map)
        release_gpu_sampler(h.device, h.shadow_sampler)

        release_gpu_buffer(h.device, h.text_vertex_buffer)
        release_gpu_buffer(h.device, h.text_index_buffer)
        release_gpu_transfer_buffer(h.device, h.text_transfer_buffer)
        release_gpu_texture(h.device, h.font_atlas_tex)
        release_gpu_sampler(h.device, h.font_sampler)

        release_gpu_graphics_pipeline(h.device, h.solid_pipeline)
        release_gpu_graphics_pipeline(h.device, h.ground_pipeline)
        release_gpu_graphics_pipeline(h.device, h.line_pipeline)
        release_gpu_graphics_pipeline(h.device, h.shadow_pipeline)
        release_gpu_graphics_pipeline(h.device, h.reflection_pipeline)
        release_gpu_graphics_pipeline(h.device, h.skybox_pipeline)
        release_gpu_graphics_pipeline(h.device, h.text_pipeline)

        release_window_from_gpu_device(h.device, h.window)
        destroy_window(h.window)
        destroy_gpu_device(h.device)
        quit()

    def _release_model_caches(mut self) raises:
        """Free the geom-keyed GPU caches — capsules, cylinders, STL meshes and
        PNG textures.

        ⚠ THE CYLINDER CACHE IS IN HERE FOR A REASON. `close()` released the
        capsule, mesh and texture caches and silently skipped the cylinders;
        that was a one-shot leak at exit, but on the switch path it would be a
        leak PER SWITCH, which is how a viewer someone leaves open for an hour
        ends up out of GPU memory.
        """
        for i in range(len(self.capsule_cache)):
            release_gpu_buffer(
                self.device.value(), self.capsule_cache[i].mesh.vertex_buffer
            )
            release_gpu_buffer(
                self.device.value(), self.capsule_cache[i].mesh.index_buffer
            )
        self.capsule_cache.clear()

        for i in range(len(self.cylinder_cache)):
            release_gpu_buffer(
                self.device.value(), self.cylinder_cache[i].mesh.vertex_buffer
            )
            release_gpu_buffer(
                self.device.value(), self.cylinder_cache[i].mesh.index_buffer
            )
        self.cylinder_cache.clear()

        for i in range(len(self.mesh_cache)):
            release_gpu_buffer(
                self.device.value(), self.mesh_cache[i].mesh.vertex_buffer
            )
            release_gpu_buffer(
                self.device.value(), self.mesh_cache[i].mesh.index_buffer
            )
        self.mesh_cache.clear()

        for i in range(len(self.texture_cache)):
            release_gpu_texture(
                self.device.value(), self.texture_cache[i].texture
            )
            release_gpu_sampler(
                self.device.value(), self.texture_cache[i].sampler
            )
        self.texture_cache.clear()

        # ⚠ THE SKIN'S VERTEX/INDEX BUFFERS ARE ALREADY GONE — they live in
        # `mesh_cache`, released above. What is left here is the persistent
        # upload staging, which nothing else owns. Clearing the list too keeps
        # `mesh_idx` from outliving the `mesh_cache` it points into: on a task
        # switch that index would otherwise address another model's mesh.
        for i in range(len(self.skin_cache)):
            release_gpu_transfer_buffer(
                self.device.value(), self.skin_cache[i].transfer
            )
        self.skin_cache.clear()
        # Same argument as the skins above: the mesh buffers are already gone,
        # what is left is staging this list alone owns, and `mesh_idx` must not
        # outlive the `mesh_cache` it points into.
        for i in range(len(self.hfield_cache)):
            release_gpu_transfer_buffer(
                self.device.value(), self.hfield_cache[i].transfer
            )
        self.hfield_cache.clear()

    def _create_shader_msl(
        self,
        source: String,
        stage: GPUShaderStage,
        num_uniform_buffers: UInt32,
        entrypoint: String,
        num_samplers: UInt32 = 0,
    ) raises -> Ptr[GPUShader, MutAnyOrigin]:
        """Compile an MSL shader from source string."""
        var code_bytes = source.as_bytes()
        var ep = entrypoint

        var info = GPUShaderCreateInfo(
            code_size=UInt(len(code_bytes)),
            code=code_bytes.unsafe_ptr(),
            entrypoint=ep.as_c_string_span().ptr(),
            format=GPUShaderFormat.GPU_SHADERFORMAT_MSL,
            stage=stage,
            num_samplers=num_samplers,
            num_storage_textures=0,
            num_storage_buffers=0,
            num_uniform_buffers=num_uniform_buffers,
            props=PropertiesID(0),
        )

        return create_gpu_shader(self.device.value(), Ptr(to=info))

    def _create_shader_spirv(
        self,
        spirv_data: List[UInt8],
        stage: GPUShaderStage,
        num_uniform_buffers: UInt32,
        num_samplers: UInt32 = 0,
    ) raises -> Ptr[GPUShader, MutAnyOrigin]:
        """Create a shader from pre-compiled SPIR-V bytecode."""
        var ep = String("main")

        var info = GPUShaderCreateInfo(
            code_size=UInt(len(spirv_data)),
            code=spirv_data.unsafe_ptr(),
            entrypoint=ep.as_c_string_span().ptr(),
            format=GPUShaderFormat.GPU_SHADERFORMAT_SPIRV,
            stage=stage,
            num_samplers=num_samplers,
            num_storage_textures=0,
            num_storage_buffers=0,
            num_uniform_buffers=num_uniform_buffers,
            props=PropertiesID(0),
        )

        return create_gpu_shader(self.device.value(), Ptr(to=info))

    def _create_shader(
        self,
        msl_source: String,
        msl_entrypoint: String,
        spirv_data: List[UInt8],
        stage: GPUShaderStage,
        num_uniform_buffers: UInt32,
        num_samplers: UInt32 = 0,
    ) raises -> Ptr[GPUShader, MutAnyOrigin]:
        """Create shader using MSL on macOS or SPIR-V on Linux."""
        comptime if CompilationTarget.is_macos():
            return self._create_shader_msl(
                msl_source,
                stage,
                num_uniform_buffers,
                msl_entrypoint,
                num_samplers,
            )
        elif CompilationTarget.is_linux():
            return self._create_shader_spirv(
                spirv_data,
                stage,
                num_uniform_buffers,
                num_samplers,
            )
        else:
            comptime assert False, "Unsupported platform for Renderer3D"

    def _no_stencil_op(self) -> GPUStencilOpState:
        """Return a zeroed-out stencil op state."""
        return GPUStencilOpState(
            fail_op=GPUStencilOp.GPU_STENCILOP_KEEP,
            pass_op=GPUStencilOp.GPU_STENCILOP_KEEP,
            depth_fail_op=GPUStencilOp.GPU_STENCILOP_KEEP,
            compare_op=GPUCompareOp.GPU_COMPAREOP_ALWAYS,
        )

    @staticmethod
    def _load_spirv() raises -> SPIRVShaders:
        """Load SPIR-V shaders on Linux, return empty on macOS."""
        comptime if CompilationTarget.is_linux():
            return load_spirv_shaders()
        else:
            return SPIRVShaders(
                List[UInt8](),
                List[UInt8](),
                List[UInt8](),
                List[UInt8](),
                List[UInt8](),
                List[UInt8](),
                List[UInt8](),
                List[UInt8](),
                List[UInt8](),
                List[UInt8](),
                List[UInt8](),
                List[UInt8](),
                List[UInt8](),
            )

    def _create_pipelines(mut self) raises:
        """Create solid, ground, line, shadow, and reflection GPU pipelines."""
        var spv = Self._load_spirv()

        # --- Solid pipeline ---
        var solid_vs = self._create_shader(
            SOLID_VERTEX_MSL,
            String("solid_vertex"),
            spv.solid_vert,
            GPUShaderStage.GPU_SHADERSTAGE_VERTEX,
            num_uniform_buffers=2,
        )
        var solid_fs = self._create_shader(
            SOLID_FRAGMENT_MSL,
            String("solid_fragment"),
            spv.solid_frag,
            GPUShaderStage.GPU_SHADERSTAGE_FRAGMENT,
            num_uniform_buffers=2,
            num_samplers=2,
        )

        # Vertex input - allocate attributes contiguously on heap
        var solid_buf_desc = GPUVertexBufferDescription(
            slot=0,
            pitch=32,
            input_rate=GPUVertexInputRate.GPU_VERTEXINPUTRATE_VERTEX,
            instance_step_rate=0,
        )
        # `List`, not `alloc`: this pipeline setup raises in many places between
        # here and the frees at the end, so every one of those paths leaked the
        # attribute array. `GPUVertexInputState.vertex_attributes` is
        # origin-parameterised, so the list is kept alive by the borrow.
        var solid_attrs = List[GPUVertexAttribute]()
        solid_attrs.append(GPUVertexAttribute(
            location=0,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT3,
            offset=0,
        ))
        solid_attrs.append(GPUVertexAttribute(
            location=1,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT3,
            offset=12,
        ))
        solid_attrs.append(GPUVertexAttribute(
            location=2,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT2,
            offset=24,
        ))
        var solid_vi = GPUVertexInputState(
            vertex_buffer_descriptions=Ptr(to=solid_buf_desc),
            num_vertex_buffers=1,
            vertex_attributes=solid_attrs.unsafe_ptr(),
            num_vertex_attributes=3,
        )

        # Color target - no blending
        var solid_ct = GPUColorTargetDescription(
            format=self.swapchain_format,
            blend_state=GPUColorTargetBlendState(
                src_color_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ONE,
                dst_color_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ZERO,
                color_blend_op=GPUBlendOp.GPU_BLENDOP_ADD,
                src_alpha_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ONE,
                dst_alpha_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ZERO,
                alpha_blend_op=GPUBlendOp.GPU_BLENDOP_ADD,
                color_write_mask=GPUColorComponentFlags(0x0F),
                enable_blend=False,
                enable_color_write_mask=False,
                padding1=0,
                padding2=0,
            ),
        )

        var solid_pi = GPUGraphicsPipelineCreateInfo(
            vertex_shader=untracked(solid_vs),
            fragment_shader=untracked(solid_fs),
            vertex_input_state=solid_vi,
            primitive_type=GPUPrimitiveType.GPU_PRIMITIVETYPE_TRIANGLELIST,
            rasterizer_state=GPURasterizerState(
                fill_mode=GPUFillMode.GPU_FILLMODE_FILL,
                cull_mode=GPUCullMode.GPU_CULLMODE_BACK,
                front_face=GPUFrontFace.GPU_FRONTFACE_COUNTER_CLOCKWISE,
                depth_bias_constant_factor=0.0,
                depth_bias_clamp=0.0,
                depth_bias_slope_factor=0.0,
                enable_depth_bias=False,
                enable_depth_clip=True,
                padding1=0,
                padding2=0,
            ),
            multisample_state=GPUMultisampleState(
                sample_count=GPUSampleCount.GPU_SAMPLECOUNT_1,
                sample_mask=0,
                enable_mask=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            depth_stencil_state=GPUDepthStencilState(
                compare_op=GPUCompareOp.GPU_COMPAREOP_LESS,
                back_stencil_state=self._no_stencil_op(),
                front_stencil_state=self._no_stencil_op(),
                compare_mask=0,
                write_mask=0,
                enable_depth_test=True,
                enable_depth_write=True,
                enable_stencil_test=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            target_info=GPUGraphicsPipelineTargetInfo(
                color_target_descriptions=Ptr(to=solid_ct),
                num_color_targets=1,
                depth_stencil_format=GPUTextureFormat.GPU_TEXTUREFORMAT_D32_FLOAT,
                has_depth_stencil_target=True,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            props=PropertiesID(0),
        )

        self.solid_pipeline = untracked(create_gpu_graphics_pipeline(
            self.device.value(), Ptr(to=solid_pi)
        ))

        # --- Ground pipeline (alpha blend for distance fade) ---
        var ground_vs = self._create_shader(
            GROUND_VERTEX_MSL,
            String("ground_vertex"),
            spv.ground_vert,
            GPUShaderStage.GPU_SHADERSTAGE_VERTEX,
            num_uniform_buffers=2,
        )
        var ground_fs = self._create_shader(
            GROUND_FRAGMENT_MSL,
            String("ground_fragment"),
            spv.ground_frag,
            GPUShaderStage.GPU_SHADERSTAGE_FRAGMENT,
            num_uniform_buffers=2,
            num_samplers=2,
        )

        var ground_ct = GPUColorTargetDescription(
            format=self.swapchain_format,
            blend_state=GPUColorTargetBlendState(
                src_color_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_SRC_ALPHA,
                dst_color_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
                color_blend_op=GPUBlendOp.GPU_BLENDOP_ADD,
                src_alpha_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ONE,
                dst_alpha_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ZERO,
                alpha_blend_op=GPUBlendOp.GPU_BLENDOP_ADD,
                color_write_mask=GPUColorComponentFlags(0x0F),
                enable_blend=True,
                enable_color_write_mask=False,
                padding1=0,
                padding2=0,
            ),
        )

        # Ground uses same vertex layout as solid - allocate contiguously
        var ground_buf_desc = GPUVertexBufferDescription(
            slot=0,
            pitch=32,
            input_rate=GPUVertexInputRate.GPU_VERTEXINPUTRATE_VERTEX,
            instance_step_rate=0,
        )
        # `List`, not `alloc`: this pipeline setup raises in many places between
        # here and the frees at the end, so every one of those paths leaked the
        # attribute array. `GPUVertexInputState.vertex_attributes` is
        # origin-parameterised, so the list is kept alive by the borrow.
        var ground_attrs = List[GPUVertexAttribute]()
        ground_attrs.append(GPUVertexAttribute(
            location=0,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT3,
            offset=0,
        ))
        ground_attrs.append(GPUVertexAttribute(
            location=1,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT3,
            offset=12,
        ))
        ground_attrs.append(GPUVertexAttribute(
            location=2,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT2,
            offset=24,
        ))
        var ground_vi = GPUVertexInputState(
            vertex_buffer_descriptions=Ptr(to=ground_buf_desc),
            num_vertex_buffers=1,
            vertex_attributes=ground_attrs.unsafe_ptr(),
            num_vertex_attributes=3,
        )

        var ground_pi = GPUGraphicsPipelineCreateInfo(
            vertex_shader=untracked(ground_vs),
            fragment_shader=untracked(ground_fs),
            vertex_input_state=ground_vi,
            primitive_type=GPUPrimitiveType.GPU_PRIMITIVETYPE_TRIANGLELIST,
            rasterizer_state=GPURasterizerState(
                fill_mode=GPUFillMode.GPU_FILLMODE_FILL,
                cull_mode=GPUCullMode.GPU_CULLMODE_NONE,
                front_face=GPUFrontFace.GPU_FRONTFACE_COUNTER_CLOCKWISE,
                depth_bias_constant_factor=0.0,
                depth_bias_clamp=0.0,
                depth_bias_slope_factor=0.0,
                enable_depth_bias=False,
                enable_depth_clip=True,
                padding1=0,
                padding2=0,
            ),
            multisample_state=GPUMultisampleState(
                sample_count=GPUSampleCount.GPU_SAMPLECOUNT_1,
                sample_mask=0,
                enable_mask=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            depth_stencil_state=GPUDepthStencilState(
                compare_op=GPUCompareOp.GPU_COMPAREOP_LESS_OR_EQUAL,
                back_stencil_state=self._no_stencil_op(),
                front_stencil_state=self._no_stencil_op(),
                compare_mask=0,
                write_mask=0,
                enable_depth_test=True,
                enable_depth_write=True,
                enable_stencil_test=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            target_info=GPUGraphicsPipelineTargetInfo(
                color_target_descriptions=Ptr(to=ground_ct),
                num_color_targets=1,
                depth_stencil_format=GPUTextureFormat.GPU_TEXTUREFORMAT_D32_FLOAT,
                has_depth_stencil_target=True,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            props=PropertiesID(0),
        )

        self.ground_pipeline = untracked(create_gpu_graphics_pipeline(
            self.device.value(), Ptr(to=ground_pi)
        ))

        # --- Line pipeline ---
        var line_vs = self._create_shader(
            LINE_VERTEX_MSL,
            String("line_vertex"),
            spv.line_vert,
            GPUShaderStage.GPU_SHADERSTAGE_VERTEX,
            num_uniform_buffers=1,
        )
        var line_fs = self._create_shader(
            LINE_FRAGMENT_MSL,
            String("line_fragment"),
            spv.line_frag,
            GPUShaderStage.GPU_SHADERSTAGE_FRAGMENT,
            num_uniform_buffers=1,
        )

        var line_ct = GPUColorTargetDescription(
            format=self.swapchain_format,
            blend_state=GPUColorTargetBlendState(
                src_color_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ONE,
                dst_color_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ZERO,
                color_blend_op=GPUBlendOp.GPU_BLENDOP_ADD,
                src_alpha_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ONE,
                dst_alpha_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ZERO,
                alpha_blend_op=GPUBlendOp.GPU_BLENDOP_ADD,
                color_write_mask=GPUColorComponentFlags(0x0F),
                enable_blend=False,
                enable_color_write_mask=False,
                padding1=0,
                padding2=0,
            ),
        )

        # Line vertex input - single attribute (position only)
        var line_buf_desc = GPUVertexBufferDescription(
            slot=0,
            pitch=12,
            input_rate=GPUVertexInputRate.GPU_VERTEXINPUTRATE_VERTEX,
            instance_step_rate=0,
        )
        var line_attr = GPUVertexAttribute(
            location=0,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT3,
            offset=0,
        )
        var line_vi = GPUVertexInputState(
            vertex_buffer_descriptions=Ptr(to=line_buf_desc),
            num_vertex_buffers=1,
            vertex_attributes=Ptr(to=line_attr),
            num_vertex_attributes=1,
        )

        var line_pi = GPUGraphicsPipelineCreateInfo(
            vertex_shader=untracked(line_vs),
            fragment_shader=untracked(line_fs),
            vertex_input_state=line_vi,
            primitive_type=GPUPrimitiveType.GPU_PRIMITIVETYPE_LINELIST,
            rasterizer_state=GPURasterizerState(
                fill_mode=GPUFillMode.GPU_FILLMODE_FILL,
                cull_mode=GPUCullMode.GPU_CULLMODE_NONE,
                front_face=GPUFrontFace.GPU_FRONTFACE_COUNTER_CLOCKWISE,
                depth_bias_constant_factor=0.0,
                depth_bias_clamp=0.0,
                depth_bias_slope_factor=0.0,
                enable_depth_bias=False,
                enable_depth_clip=True,
                padding1=0,
                padding2=0,
            ),
            multisample_state=GPUMultisampleState(
                sample_count=GPUSampleCount.GPU_SAMPLECOUNT_1,
                sample_mask=0,
                enable_mask=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            depth_stencil_state=GPUDepthStencilState(
                compare_op=GPUCompareOp.GPU_COMPAREOP_LESS_OR_EQUAL,
                back_stencil_state=self._no_stencil_op(),
                front_stencil_state=self._no_stencil_op(),
                compare_mask=0,
                write_mask=0,
                enable_depth_test=True,
                enable_depth_write=False,
                enable_stencil_test=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            target_info=GPUGraphicsPipelineTargetInfo(
                color_target_descriptions=Ptr(to=line_ct),
                num_color_targets=1,
                depth_stencil_format=GPUTextureFormat.GPU_TEXTUREFORMAT_D32_FLOAT,
                has_depth_stencil_target=True,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            props=PropertiesID(0),
        )

        self.line_pipeline = untracked(create_gpu_graphics_pipeline(
            self.device.value(), Ptr(to=line_pi)
        ))

        # --- Shadow pipeline (depth-only, from light POV) ---
        var shadow_vs = self._create_shader(
            SHADOW_VERTEX_MSL,
            String("shadow_vertex"),
            spv.shadow_vert,
            GPUShaderStage.GPU_SHADERSTAGE_VERTEX,
            num_uniform_buffers=2,
        )
        var shadow_fs = self._create_shader(
            SHADOW_FRAGMENT_MSL,
            String("shadow_fragment"),
            spv.shadow_frag,
            GPUShaderStage.GPU_SHADERSTAGE_FRAGMENT,
            num_uniform_buffers=0,
        )

        # Shadow uses same vertex layout as solid
        var shadow_buf_desc = GPUVertexBufferDescription(
            slot=0,
            pitch=32,
            input_rate=GPUVertexInputRate.GPU_VERTEXINPUTRATE_VERTEX,
            instance_step_rate=0,
        )
        # `List`, not `alloc`: this pipeline setup raises in many places between
        # here and the frees at the end, so every one of those paths leaked the
        # attribute array. `GPUVertexInputState.vertex_attributes` is
        # origin-parameterised, so the list is kept alive by the borrow.
        var shadow_attrs = List[GPUVertexAttribute]()
        shadow_attrs.append(GPUVertexAttribute(
            location=0,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT3,
            offset=0,
        ))
        shadow_attrs.append(GPUVertexAttribute(
            location=1,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT3,
            offset=12,
        ))
        shadow_attrs.append(GPUVertexAttribute(
            location=2,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT2,
            offset=24,
        ))
        var shadow_vi = GPUVertexInputState(
            vertex_buffer_descriptions=Ptr(to=shadow_buf_desc),
            num_vertex_buffers=1,
            vertex_attributes=shadow_attrs.unsafe_ptr(),
            num_vertex_attributes=3,
        )

        var shadow_pi = GPUGraphicsPipelineCreateInfo(
            vertex_shader=untracked(shadow_vs),
            fragment_shader=untracked(shadow_fs),
            vertex_input_state=shadow_vi,
            primitive_type=GPUPrimitiveType.GPU_PRIMITIVETYPE_TRIANGLELIST,
            rasterizer_state=GPURasterizerState(
                fill_mode=GPUFillMode.GPU_FILLMODE_FILL,
                cull_mode=GPUCullMode.GPU_CULLMODE_BACK,
                front_face=GPUFrontFace.GPU_FRONTFACE_COUNTER_CLOCKWISE,
                depth_bias_constant_factor=1.5,
                depth_bias_clamp=0.0,
                depth_bias_slope_factor=2.0,
                enable_depth_bias=True,
                enable_depth_clip=True,
                padding1=0,
                padding2=0,
            ),
            multisample_state=GPUMultisampleState(
                sample_count=GPUSampleCount.GPU_SAMPLECOUNT_1,
                sample_mask=0,
                enable_mask=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            depth_stencil_state=GPUDepthStencilState(
                compare_op=GPUCompareOp.GPU_COMPAREOP_LESS,
                back_stencil_state=self._no_stencil_op(),
                front_stencil_state=self._no_stencil_op(),
                compare_mask=0,
                write_mask=0,
                enable_depth_test=True,
                enable_depth_write=True,
                enable_stencil_test=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            target_info=GPUGraphicsPipelineTargetInfo(
                color_target_descriptions=_null_ptr[
                    GPUColorTargetDescription, ImmutAnyOrigin
                ](),
                num_color_targets=0,
                depth_stencil_format=GPUTextureFormat.GPU_TEXTUREFORMAT_D32_FLOAT,
                has_depth_stencil_target=True,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            props=PropertiesID(0),
        )

        self.shadow_pipeline = untracked(create_gpu_graphics_pipeline(
            self.device.value(), Ptr(to=shadow_pi)
        ))

        # --- Reflection pipeline (alpha-blended, front-cull, no depth write) ---
        var refl_fs = self._create_shader(
            REFLECTION_FRAGMENT_MSL,
            String("reflection_fragment"),
            spv.reflection_frag,
            GPUShaderStage.GPU_SHADERSTAGE_FRAGMENT,
            num_uniform_buffers=1,
        )
        # Reuse solid vertex shader for reflection (same vertex output struct)
        var refl_vs = self._create_shader(
            SOLID_VERTEX_MSL,
            String("solid_vertex"),
            spv.solid_vert,
            GPUShaderStage.GPU_SHADERSTAGE_VERTEX,
            num_uniform_buffers=2,
        )

        var refl_ct = GPUColorTargetDescription(
            format=self.swapchain_format,
            blend_state=GPUColorTargetBlendState(
                src_color_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_SRC_ALPHA,
                dst_color_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
                color_blend_op=GPUBlendOp.GPU_BLENDOP_ADD,
                src_alpha_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ONE,
                dst_alpha_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ZERO,
                alpha_blend_op=GPUBlendOp.GPU_BLENDOP_ADD,
                color_write_mask=GPUColorComponentFlags(0x0F),
                enable_blend=True,
                enable_color_write_mask=False,
                padding1=0,
                padding2=0,
            ),
        )

        # Reflection uses same vertex layout as solid
        var refl_buf_desc = GPUVertexBufferDescription(
            slot=0,
            pitch=32,
            input_rate=GPUVertexInputRate.GPU_VERTEXINPUTRATE_VERTEX,
            instance_step_rate=0,
        )
        # `List`, not `alloc`: this pipeline setup raises in many places between
        # here and the frees at the end, so every one of those paths leaked the
        # attribute array. `GPUVertexInputState.vertex_attributes` is
        # origin-parameterised, so the list is kept alive by the borrow.
        var refl_attrs = List[GPUVertexAttribute]()
        refl_attrs.append(GPUVertexAttribute(
            location=0,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT3,
            offset=0,
        ))
        refl_attrs.append(GPUVertexAttribute(
            location=1,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT3,
            offset=12,
        ))
        refl_attrs.append(GPUVertexAttribute(
            location=2,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT2,
            offset=24,
        ))
        var refl_vi = GPUVertexInputState(
            vertex_buffer_descriptions=Ptr(to=refl_buf_desc),
            num_vertex_buffers=1,
            vertex_attributes=refl_attrs.unsafe_ptr(),
            num_vertex_attributes=3,
        )

        var refl_pi = GPUGraphicsPipelineCreateInfo(
            vertex_shader=untracked(refl_vs),
            fragment_shader=untracked(refl_fs),
            vertex_input_state=refl_vi,
            primitive_type=GPUPrimitiveType.GPU_PRIMITIVETYPE_TRIANGLELIST,
            rasterizer_state=GPURasterizerState(
                fill_mode=GPUFillMode.GPU_FILLMODE_FILL,
                cull_mode=GPUCullMode.GPU_CULLMODE_FRONT,  # Front-cull (Z-flip reverses winding)
                front_face=GPUFrontFace.GPU_FRONTFACE_COUNTER_CLOCKWISE,
                depth_bias_constant_factor=0.0,
                depth_bias_clamp=0.0,
                depth_bias_slope_factor=0.0,
                enable_depth_bias=False,
                enable_depth_clip=True,
                padding1=0,
                padding2=0,
            ),
            multisample_state=GPUMultisampleState(
                sample_count=GPUSampleCount.GPU_SAMPLECOUNT_1,
                sample_mask=0,
                enable_mask=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            depth_stencil_state=GPUDepthStencilState(
                # ⚠ ALWAYS, NOT LESS — the reflection now draws AFTER the
                # ground (Phase B2), and the ground has written depth by then.
                # Under LESS every reflection fragment would be rejected by the
                # very floor it belongs on, and the pass would silently render
                # nothing. It was LESS while this ran BEFORE the ground, where
                # the depth buffer was still cleared to far and the test passed
                # everything — i.e. it was already a no-op, so nothing is lost.
                #
                # With no depth test, what keeps reflections from painting over
                # the sky past the ground's rim is the fragment shader's own
                # distance fade, which is tighter than the ground's. Still no
                # depth WRITE: reflections must not occlude the solids drawn
                # after them.
                compare_op=GPUCompareOp.GPU_COMPAREOP_ALWAYS,
                back_stencil_state=self._no_stencil_op(),
                front_stencil_state=self._no_stencil_op(),
                compare_mask=0,
                write_mask=0,
                enable_depth_test=True,
                enable_depth_write=False,
                enable_stencil_test=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            target_info=GPUGraphicsPipelineTargetInfo(
                color_target_descriptions=Ptr(to=refl_ct),
                num_color_targets=1,
                depth_stencil_format=GPUTextureFormat.GPU_TEXTUREFORMAT_D32_FLOAT,
                has_depth_stencil_target=True,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            props=PropertiesID(0),
        )

        self.reflection_pipeline = untracked(create_gpu_graphics_pipeline(
            self.device.value(), Ptr(to=refl_pi)
        ))

        # --- Skybox pipeline (fullscreen gradient, no depth write, no vertex input) ---
        var skybox_vs = self._create_shader(
            SKYBOX_VERTEX_MSL,
            String("skybox_vertex"),
            spv.skybox_vert,
            GPUShaderStage.GPU_SHADERSTAGE_VERTEX,
            num_uniform_buffers=0,
        )
        var skybox_fs = self._create_shader(
            SKYBOX_FRAGMENT_MSL,
            String("skybox_fragment"),
            spv.skybox_frag,
            GPUShaderStage.GPU_SHADERSTAGE_FRAGMENT,
            num_uniform_buffers=1,
        )

        var skybox_ct = GPUColorTargetDescription(
            format=self.swapchain_format,
            blend_state=GPUColorTargetBlendState(
                src_color_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ONE,
                dst_color_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ZERO,
                color_blend_op=GPUBlendOp.GPU_BLENDOP_ADD,
                src_alpha_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ONE,
                dst_alpha_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ZERO,
                alpha_blend_op=GPUBlendOp.GPU_BLENDOP_ADD,
                color_write_mask=GPUColorComponentFlags(0x0F),
                enable_blend=False,
                enable_color_write_mask=False,
                padding1=0,
                padding2=0,
            ),
        )

        # Skybox has NO vertex input (uses vertex_id to generate fullscreen triangle).
        var skybox_vi = GPUVertexInputState(
            vertex_buffer_descriptions=_null_ptr[
                GPUVertexBufferDescription, ImmutAnyOrigin
            ](),
            num_vertex_buffers=0,
            vertex_attributes=_null_ptr[GPUVertexAttribute, ImmutAnyOrigin](),
            num_vertex_attributes=0,
        )

        var skybox_pi = GPUGraphicsPipelineCreateInfo(
            vertex_shader=untracked(skybox_vs),
            fragment_shader=untracked(skybox_fs),
            vertex_input_state=skybox_vi,
            primitive_type=GPUPrimitiveType.GPU_PRIMITIVETYPE_TRIANGLELIST,
            rasterizer_state=GPURasterizerState(
                fill_mode=GPUFillMode.GPU_FILLMODE_FILL,
                cull_mode=GPUCullMode.GPU_CULLMODE_NONE,
                front_face=GPUFrontFace.GPU_FRONTFACE_COUNTER_CLOCKWISE,
                depth_bias_constant_factor=0.0,
                depth_bias_clamp=0.0,
                depth_bias_slope_factor=0.0,
                enable_depth_bias=False,
                enable_depth_clip=False,
                padding1=0,
                padding2=0,
            ),
            multisample_state=GPUMultisampleState(
                sample_count=GPUSampleCount.GPU_SAMPLECOUNT_1,
                sample_mask=0,
                enable_mask=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            depth_stencil_state=GPUDepthStencilState(
                compare_op=GPUCompareOp.GPU_COMPAREOP_LESS_OR_EQUAL,
                back_stencil_state=self._no_stencil_op(),
                front_stencil_state=self._no_stencil_op(),
                compare_mask=0,
                write_mask=0,
                enable_depth_test=False,
                enable_depth_write=False,
                enable_stencil_test=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            target_info=GPUGraphicsPipelineTargetInfo(
                color_target_descriptions=Ptr(to=skybox_ct),
                num_color_targets=1,
                depth_stencil_format=GPUTextureFormat.GPU_TEXTUREFORMAT_D32_FLOAT,
                has_depth_stencil_target=True,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            props=PropertiesID(0),
        )

        self.skybox_pipeline = untracked(create_gpu_graphics_pipeline(
            self.device.value(), Ptr(to=skybox_pi)
        ))


        # Keep the attribute lists alive until every pipeline is built. The
        # borrow through `GPUVertexInputState` should already guarantee this,
        # but the pointers cross an FFI boundary and a premature free here is
        # silent — the same shape that made the HDF5 buffers read freed memory.
        _ = solid_attrs
        _ = ground_attrs
        _ = shadow_attrs
        _ = refl_attrs

        # Release shader objects (pipelines retain them)
        release_gpu_shader(self.device.value(), solid_vs)
        release_gpu_shader(self.device.value(), solid_fs)
        release_gpu_shader(self.device.value(), ground_vs)
        release_gpu_shader(self.device.value(), ground_fs)
        release_gpu_shader(self.device.value(), line_vs)
        release_gpu_shader(self.device.value(), line_fs)
        release_gpu_shader(self.device.value(), shadow_vs)
        release_gpu_shader(self.device.value(), shadow_fs)
        release_gpu_shader(self.device.value(), refl_vs)
        release_gpu_shader(self.device.value(), refl_fs)
        release_gpu_shader(self.device.value(), skybox_vs)
        release_gpu_shader(self.device.value(), skybox_fs)

    def _create_depth_texture(mut self) raises:
        """Create the depth buffer texture."""
        var info = GPUTextureCreateInfo(
            type=GPUTextureType.GPU_TEXTURETYPE_2D,
            format=GPUTextureFormat.GPU_TEXTUREFORMAT_D32_FLOAT,
            usage=GPUTextureUsageFlags.GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET,
            width=UInt32(self.width),
            height=UInt32(self.height),
            layer_count_or_depth=1,
            num_levels=1,
            sample_count=GPUSampleCount.GPU_SAMPLECOUNT_1,
            props=PropertiesID(0),
        )
        self.depth_texture = untracked(create_gpu_texture(self.device.value(), Ptr(to=info)))

    def _create_shadow_resources(mut self) raises:
        """Create shadow map texture and comparison sampler.

        ⚠⚠ `shadow_size` COMES FROM THE MODEL (`<visual><quality
        shadowsize=>`) AND MuJoCo LETS IT BE ZERO — that is how a model turns
        shadows OFF. Menagerie's umi_gripper writes `shadowsize="0"`, and
        passing it straight through aborted SDL: "width, height, and
        layer_count_or_depth must be >= 1", inside an assert that kills the
        process rather than raising. The model had already parsed and built;
        the only visible outcome was an empty error message.

        Clamped rather than branched: the shadow PASS still runs and samples a
        1x1 map, which is a shadow nobody can see — the same end state the
        model asked for, without a second code path through the renderer that
        only a handful of models would ever exercise.
        """
        if self.shadow_size < 1:
            self.shadow_size = 1
        # Shadow map: D32_FLOAT, usable as both depth target and sampler source
        var sm_info = GPUTextureCreateInfo(
            type=GPUTextureType.GPU_TEXTURETYPE_2D,
            format=GPUTextureFormat.GPU_TEXTUREFORMAT_D32_FLOAT,
            usage=GPUTextureUsageFlags.GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET
            | GPUTextureUsageFlags.GPU_TEXTUREUSAGE_SAMPLER,
            width=UInt32(self.shadow_size),
            height=UInt32(self.shadow_size),
            layer_count_or_depth=1,
            num_levels=1,
            sample_count=GPUSampleCount.GPU_SAMPLECOUNT_1,
            props=PropertiesID(0),
        )
        self.shadow_map = untracked(create_gpu_texture(self.device.value(), Ptr(to=sm_info)))

        # Comparison sampler for shadow mapping
        var sampler_info = GPUSamplerCreateInfo(
            min_filter=GPUFilter.GPU_FILTER_LINEAR,
            mag_filter=GPUFilter.GPU_FILTER_LINEAR,
            mipmap_mode=GPUSamplerMipmapMode.GPU_SAMPLERMIPMAPMODE_NEAREST,
            address_mode_u=GPUSamplerAddressMode.GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            address_mode_v=GPUSamplerAddressMode.GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            address_mode_w=GPUSamplerAddressMode.GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            mip_lod_bias=0.0,
            max_anisotropy=1.0,
            compare_op=GPUCompareOp.GPU_COMPAREOP_LESS,
            min_lod=0.0,
            max_lod=0.0,
            enable_anisotropy=False,
            enable_compare=True,
            padding1=0,
            padding2=0,
            props=PropertiesID(0),
        )
        self.shadow_sampler = untracked(create_gpu_sampler(
            self.device.value(), Ptr(to=sampler_info)
        ))

    def _upload_mesh(self, mesh_data: MeshData) raises -> MeshHandle:
        """Upload mesh data to GPU buffers via transfer buffer."""
        var vb_size = UInt32(mesh_data.vertex_byte_size())
        var ib_size = UInt32(mesh_data.index_byte_size())
        var total_size = vb_size + ib_size

        # Create transfer buffer
        var tb_info = GPUTransferBufferCreateInfo(
            usage=GPUTransferBufferUsage.GPU_TRANSFERBUFFERUSAGE_UPLOAD,
            size=total_size,
            props=PropertiesID(0),
        )
        var transfer_buf = untracked(create_gpu_transfer_buffer(
            self.device.value(), Ptr(to=tb_info)
        ))

        # Map and copy data
        var mapped = map_gpu_transfer_buffer(self.device.value(), transfer_buf, False)
        var mapped_ptr = mapped.unsafe_bitcast[UInt8]()

        # Copy vertices
        unsafe_memcpy(
            dest=mapped_ptr,
            src=Pointer(to=mesh_data.vertices[0]).unsafe_bitcast[UInt8](),
            count=Int(vb_size),
        )
        # Copy indices after vertices
        unsafe_memcpy(
            dest=mapped_ptr.unsafe_offset(Int(vb_size)),
            src=Pointer(to=mesh_data.indices[0]).unsafe_bitcast[UInt8](),
            count=Int(ib_size),
        )

        unmap_gpu_transfer_buffer(self.device.value(), transfer_buf)

        # Create GPU buffers
        var vb_info = GPUBufferCreateInfo(
            usage=GPUBufferUsageFlags.GPU_BUFFERUSAGE_VERTEX,
            size=vb_size,
            props=PropertiesID(0),
        )
        var vertex_buffer = untracked(create_gpu_buffer(self.device.value(), Ptr(to=vb_info)))

        var ib_info = GPUBufferCreateInfo(
            usage=GPUBufferUsageFlags.GPU_BUFFERUSAGE_INDEX,
            size=ib_size,
            props=PropertiesID(0),
        )
        var index_buffer = untracked(create_gpu_buffer(self.device.value(), Ptr(to=ib_info)))

        # Upload via copy pass
        var cmd_buf = acquire_gpu_command_buffer(self.device.value())
        var copy_pass = begin_gpu_copy_pass(cmd_buf)

        var vb_src = GPUTransferBufferLocation(
            transfer_buffer=untracked(transfer_buf), offset=0
        )
        var vb_dst = GPUBufferRegion(
            buffer=untracked(vertex_buffer), offset=0, size=vb_size
        )
        upload_to_gpu_buffer(copy_pass, Ptr(to=vb_src), Ptr(to=vb_dst), False)

        var ib_src = GPUTransferBufferLocation(
            transfer_buffer=untracked(transfer_buf), offset=vb_size
        )
        var ib_dst = GPUBufferRegion(
            buffer=untracked(index_buffer), offset=0, size=ib_size
        )
        upload_to_gpu_buffer(copy_pass, Ptr(to=ib_src), Ptr(to=ib_dst), False)

        end_gpu_copy_pass(copy_pass)
        submit_gpu_command_buffer(cmd_buf)

        # Release transfer buffer
        release_gpu_transfer_buffer(self.device.value(), transfer_buf)

        return MeshHandle(
            vertex_buffer,
            index_buffer,
            UInt32(len(mesh_data.indices)),
            UInt32(len(mesh_data.vertices)),
        )

    def _upload_static_meshes(mut self) raises:
        """Generate and upload sphere, box, and ground meshes."""
        var sphere_data = generate_sphere(16, 12)
        self.sphere_mesh = self._upload_mesh(sphere_data)

        var box_data = generate_box()
        self.box_mesh = self._upload_mesh(box_data)

        var ground_data = generate_ground(12.0)
        self.ground_mesh = self._upload_mesh(ground_data)

    def _create_line_buffers(mut self) raises:
        """Allocate GPU and transfer buffers for dynamic line rendering."""
        var line_buf_size = UInt32(
            MAX_LINE_VERTICES * 12
        )  # 12 bytes per vertex (float3)

        var vb_info = GPUBufferCreateInfo(
            usage=GPUBufferUsageFlags.GPU_BUFFERUSAGE_VERTEX,
            size=line_buf_size,
            props=PropertiesID(0),
        )
        self.line_vertex_buffer = untracked(create_gpu_buffer(
            self.device.value(), Ptr(to=vb_info)
        ))

        var tb_info = GPUTransferBufferCreateInfo(
            usage=GPUTransferBufferUsage.GPU_TRANSFERBUFFERUSAGE_UPLOAD,
            size=line_buf_size,
            props=PropertiesID(0),
        )
        self.line_transfer_buffer = untracked(create_gpu_transfer_buffer(
            self.device.value(), Ptr(to=tb_info)
        ))

    def _create_text_resources(mut self) raises:
        """Create font atlas texture, sampler, text pipeline and buffers."""
        # --- 1. Build and upload R8_UNORM font atlas (128×64) ---
        var atlas = build_font_atlas_r8()  # List[UInt8], 8192 bytes
        var atlas_size = UInt32(8192)  # 128 * 64 * 1 byte

        var atlas_tex_info = GPUTextureCreateInfo(
            type=GPUTextureType.GPU_TEXTURETYPE_2D,
            format=GPUTextureFormat.GPU_TEXTUREFORMAT_R8_UNORM,
            usage=GPUTextureUsageFlags.GPU_TEXTUREUSAGE_SAMPLER,
            width=128,
            height=64,
            layer_count_or_depth=1,
            num_levels=1,
            sample_count=GPUSampleCount.GPU_SAMPLECOUNT_1,
            props=PropertiesID(0),
        )
        self.font_atlas_tex = untracked(create_gpu_texture(
            self.device.value(), Ptr(to=atlas_tex_info)
        ))

        # Upload atlas via transfer buffer
        var atlas_tb_info = GPUTransferBufferCreateInfo(
            usage=GPUTransferBufferUsage.GPU_TRANSFERBUFFERUSAGE_UPLOAD,
            size=atlas_size,
            props=PropertiesID(0),
        )
        var atlas_tb = untracked(create_gpu_transfer_buffer(
            self.device.value(), Ptr(to=atlas_tb_info)
        ))
        var mapped = map_gpu_transfer_buffer(self.device.value(), atlas_tb, False)
        var mapped_u8 = mapped.unsafe_bitcast[UInt8]()
        for i in range(8192):
            mapped_u8[unsafe_offset=i] = atlas[i]
        unmap_gpu_transfer_buffer(self.device.value(), atlas_tb)

        var atlas_cmd = acquire_gpu_command_buffer(self.device.value())
        var atlas_cp = begin_gpu_copy_pass(atlas_cmd)
        var atlas_src = GPUTextureTransferInfo(
            transfer_buffer=untracked(atlas_tb),
            offset=0,
            pixels_per_row=128,
            rows_per_layer=64,
        )
        var atlas_dst = GPUTextureRegion(
            texture=untracked(self.font_atlas_tex.value()),
            mip_level=0,
            layer=0,
            x=0,
            y=0,
            z=0,
            w=128,
            h=64,
            d=1,
        )
        upload_to_gpu_texture(
            atlas_cp, Ptr(to=atlas_src), Ptr(to=atlas_dst), False
        )
        end_gpu_copy_pass(atlas_cp)
        submit_gpu_command_buffer(atlas_cmd)
        release_gpu_transfer_buffer(self.device.value(), atlas_tb)

        # --- 2. Create NEAREST sampler ---
        var samp_info = GPUSamplerCreateInfo(
            min_filter=GPUFilter.GPU_FILTER_NEAREST,
            mag_filter=GPUFilter.GPU_FILTER_NEAREST,
            mipmap_mode=GPUSamplerMipmapMode.GPU_SAMPLERMIPMAPMODE_NEAREST,
            address_mode_u=GPUSamplerAddressMode.GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            address_mode_v=GPUSamplerAddressMode.GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            address_mode_w=GPUSamplerAddressMode.GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            mip_lod_bias=0.0,
            max_anisotropy=1.0,
            compare_op=GPUCompareOp.GPU_COMPAREOP_ALWAYS,
            min_lod=0.0,
            max_lod=0.0,
            enable_anisotropy=False,
            enable_compare=False,
            padding1=0,
            padding2=0,
            props=PropertiesID(0),
        )
        self.font_sampler = untracked(create_gpu_sampler(self.device.value(), Ptr(to=samp_info)))

        # --- 3. Create text pipeline ---
        var spv = Self._load_spirv()
        var text_vs = self._create_shader(
            TEXT_VERTEX_MSL,
            String("text_vertex"),
            spv.text_vert,
            GPUShaderStage.GPU_SHADERSTAGE_VERTEX,
            num_uniform_buffers=1,
        )
        var text_fs = self._create_shader(
            TEXT_FRAGMENT_MSL,
            String("text_fragment"),
            spv.text_frag,
            GPUShaderStage.GPU_SHADERSTAGE_FRAGMENT,
            num_uniform_buffers=0,
            num_samplers=1,
        )

        var text_buf_desc = GPUVertexBufferDescription(
            slot=0,
            pitch=32,  # 8 floats × 4 bytes = 32 bytes per vertex
            input_rate=GPUVertexInputRate.GPU_VERTEXINPUTRATE_VERTEX,
            instance_step_rate=0,
        )
        # `List`, not `alloc`: this pipeline setup raises in many places between
        # here and the frees at the end, so every one of those paths leaked the
        # attribute array. `GPUVertexInputState.vertex_attributes` is
        # origin-parameterised, so the list is kept alive by the borrow.
        var text_attrs = List[GPUVertexAttribute]()
        text_attrs.append(GPUVertexAttribute(
            location=0,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT2,
            offset=0,
        ))
        text_attrs.append(GPUVertexAttribute(
            location=1,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT2,
            offset=8,
        ))
        text_attrs.append(GPUVertexAttribute(
            location=2,
            buffer_slot=0,
            format=GPUVertexElementFormat.GPU_VERTEXELEMENTFORMAT_FLOAT4,
            offset=16,
        ))
        var text_vi = GPUVertexInputState(
            vertex_buffer_descriptions=Ptr(to=text_buf_desc),
            num_vertex_buffers=1,
            vertex_attributes=text_attrs.unsafe_ptr(),
            num_vertex_attributes=3,
        )

        var text_ct = GPUColorTargetDescription(
            format=self.swapchain_format,
            blend_state=GPUColorTargetBlendState(
                src_color_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_SRC_ALPHA,
                dst_color_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
                color_blend_op=GPUBlendOp.GPU_BLENDOP_ADD,
                src_alpha_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ONE,
                dst_alpha_blendfactor=GPUBlendFactor.GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
                alpha_blend_op=GPUBlendOp.GPU_BLENDOP_ADD,
                color_write_mask=GPUColorComponentFlags(0x0F),
                enable_blend=True,
                enable_color_write_mask=False,
                padding1=0,
                padding2=0,
            ),
        )

        var text_pi = GPUGraphicsPipelineCreateInfo(
            vertex_shader=untracked(text_vs),
            fragment_shader=untracked(text_fs),
            vertex_input_state=text_vi,
            primitive_type=GPUPrimitiveType.GPU_PRIMITIVETYPE_TRIANGLELIST,
            rasterizer_state=GPURasterizerState(
                fill_mode=GPUFillMode.GPU_FILLMODE_FILL,
                cull_mode=GPUCullMode.GPU_CULLMODE_NONE,
                front_face=GPUFrontFace.GPU_FRONTFACE_COUNTER_CLOCKWISE,
                depth_bias_constant_factor=0.0,
                depth_bias_clamp=0.0,
                depth_bias_slope_factor=0.0,
                enable_depth_bias=False,
                enable_depth_clip=False,
                padding1=0,
                padding2=0,
            ),
            multisample_state=GPUMultisampleState(
                sample_count=GPUSampleCount.GPU_SAMPLECOUNT_1,
                sample_mask=0,
                enable_mask=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            depth_stencil_state=GPUDepthStencilState(
                compare_op=GPUCompareOp.GPU_COMPAREOP_ALWAYS,
                back_stencil_state=self._no_stencil_op(),
                front_stencil_state=self._no_stencil_op(),
                compare_mask=0,
                write_mask=0,
                enable_depth_test=False,
                enable_depth_write=False,
                enable_stencil_test=False,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            target_info=GPUGraphicsPipelineTargetInfo(
                color_target_descriptions=Ptr(to=text_ct),
                num_color_targets=1,
                depth_stencil_format=GPUTextureFormat.GPU_TEXTUREFORMAT_D32_FLOAT,
                has_depth_stencil_target=True,
                padding1=0,
                padding2=0,
                padding3=0,
            ),
            props=PropertiesID(0),
        )
        self.text_pipeline = untracked(create_gpu_graphics_pipeline(
            self.device.value(), Ptr(to=text_pi)
        ))
        _ = text_attrs  # keep alive across pipeline creation
        release_gpu_shader(self.device.value(), text_vs)
        release_gpu_shader(self.device.value(), text_fs)

        # --- 4. Allocate text vertex buffer (MAX_TEXT_CHARS quads × 4 verts × 32 bytes) ---
        var tvb_size = UInt32(MAX_TEXT_CHARS * 4 * 32)
        var tvb_info = GPUBufferCreateInfo(
            usage=GPUBufferUsageFlags.GPU_BUFFERUSAGE_VERTEX,
            size=tvb_size,
            props=PropertiesID(0),
        )
        self.text_vertex_buffer = untracked(create_gpu_buffer(
            self.device.value(), Ptr(to=tvb_info)
        ))

        var ttb_info = GPUTransferBufferCreateInfo(
            usage=GPUTransferBufferUsage.GPU_TRANSFERBUFFERUSAGE_UPLOAD,
            size=tvb_size,
            props=PropertiesID(0),
        )
        self.text_transfer_buffer = untracked(create_gpu_transfer_buffer(
            self.device.value(), Ptr(to=ttb_info)
        ))

        # --- 5. Allocate and upload static index buffer (MAX_TEXT_CHARS quads × 6 indices × 2 bytes) ---
        var tib_size = UInt32(MAX_TEXT_CHARS * 6 * 2)
        var tib_info = GPUBufferCreateInfo(
            usage=GPUBufferUsageFlags.GPU_BUFFERUSAGE_INDEX,
            size=tib_size,
            props=PropertiesID(0),
        )
        self.text_index_buffer = untracked(create_gpu_buffer(
            self.device.value(), Ptr(to=tib_info)
        ))

        # Build and upload index data: quads [0,1,2, 2,3,0, 4,5,6, 6,7,4, ...]
        var idx_tb_info = GPUTransferBufferCreateInfo(
            usage=GPUTransferBufferUsage.GPU_TRANSFERBUFFERUSAGE_UPLOAD,
            size=tib_size,
            props=PropertiesID(0),
        )
        var idx_tb = untracked(create_gpu_transfer_buffer(
            self.device.value(), Ptr(to=idx_tb_info)
        ))
        var idx_mapped = map_gpu_transfer_buffer(self.device.value(), idx_tb, False)
        var idx_ptr = idx_mapped.unsafe_bitcast[UInt16]()
        for q in range(MAX_TEXT_CHARS):
            var base = UInt16(q * 4)
            idx_ptr[unsafe_offset=q * 6 + 0] = base + 0
            idx_ptr[unsafe_offset=q * 6 + 1] = base + 1
            idx_ptr[unsafe_offset=q * 6 + 2] = base + 2
            idx_ptr[unsafe_offset=q * 6 + 3] = base + 2
            idx_ptr[unsafe_offset=q * 6 + 4] = base + 3
            idx_ptr[unsafe_offset=q * 6 + 5] = base + 0
        unmap_gpu_transfer_buffer(self.device.value(), idx_tb)

        var idx_cmd = acquire_gpu_command_buffer(self.device.value())
        var idx_cp = begin_gpu_copy_pass(idx_cmd)
        var idx_src = GPUTransferBufferLocation(
            transfer_buffer=untracked(idx_tb), offset=0
        )
        var idx_dst = GPUBufferRegion(
            buffer=untracked(self.text_index_buffer.value()), offset=0, size=tib_size
        )
        upload_to_gpu_buffer(idx_cp, Ptr(to=idx_src), Ptr(to=idx_dst), False)
        end_gpu_copy_pass(idx_cp)
        submit_gpu_command_buffer(idx_cmd)
        release_gpu_transfer_buffer(self.device.value(), idx_tb)

    def _create_default_texture(mut self) raises:
        """Create a 1x1 white RGBA8 texture as default for untextured objects.
        """
        # Create 1x1 RGBA8 texture
        var tex_info = GPUTextureCreateInfo(
            type=GPUTextureType.GPU_TEXTURETYPE_2D,
            format=GPUTextureFormat.GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
            usage=GPUTextureUsageFlags.GPU_TEXTUREUSAGE_SAMPLER,
            width=1,
            height=1,
            layer_count_or_depth=1,
            num_levels=1,
            sample_count=GPUSampleCount.GPU_SAMPLECOUNT_1,
            props=PropertiesID(0),
        )
        self.default_texture = untracked(create_gpu_texture(self.device.value(), Ptr(to=tex_info)))

        # Upload 1x1 white pixel via transfer buffer
        var tb_info = GPUTransferBufferCreateInfo(
            usage=GPUTransferBufferUsage.GPU_TRANSFERBUFFERUSAGE_UPLOAD,
            size=4,  # 4 bytes: RGBA
            props=PropertiesID(0),
        )
        var tb = untracked(create_gpu_transfer_buffer(self.device.value(), Ptr(to=tb_info)))
        var mapped = map_gpu_transfer_buffer(self.device.value(), tb, False)
        var mapped_u8 = mapped.unsafe_bitcast[UInt8]()
        mapped_u8[unsafe_offset=0] = UInt8(255)  # R
        mapped_u8[unsafe_offset=1] = UInt8(255)  # G
        mapped_u8[unsafe_offset=2] = UInt8(255)  # B
        mapped_u8[unsafe_offset=3] = UInt8(255)  # A
        unmap_gpu_transfer_buffer(self.device.value(), tb)

        var cmd = acquire_gpu_command_buffer(self.device.value())
        var cp = begin_gpu_copy_pass(cmd)
        var src = GPUTextureTransferInfo(
            transfer_buffer=untracked(tb),
            offset=0,
            pixels_per_row=1,
            rows_per_layer=1,
        )
        var dst = GPUTextureRegion(
            texture=untracked(self.default_texture.value()),
            mip_level=0,
            layer=0,
            x=0,
            y=0,
            z=0,
            w=1,
            h=1,
            d=1,
        )
        upload_to_gpu_texture(cp, Ptr(to=src), Ptr(to=dst), False)
        end_gpu_copy_pass(cp)
        submit_gpu_command_buffer(cmd)
        release_gpu_transfer_buffer(self.device.value(), tb)

        # Create LINEAR sampler with REPEAT address mode (1x1: no mip chain)
        var samp_info = GPUSamplerCreateInfo(
            min_filter=GPUFilter.GPU_FILTER_LINEAR,
            mag_filter=GPUFilter.GPU_FILTER_LINEAR,
            mipmap_mode=GPUSamplerMipmapMode.GPU_SAMPLERMIPMAPMODE_NEAREST,
            address_mode_u=GPUSamplerAddressMode.GPU_SAMPLERADDRESSMODE_REPEAT,
            address_mode_v=GPUSamplerAddressMode.GPU_SAMPLERADDRESSMODE_REPEAT,
            address_mode_w=GPUSamplerAddressMode.GPU_SAMPLERADDRESSMODE_REPEAT,
            mip_lod_bias=0.0,
            max_anisotropy=1.0,
            compare_op=GPUCompareOp.GPU_COMPAREOP_ALWAYS,
            min_lod=0.0,
            max_lod=0.0,
            enable_anisotropy=False,
            enable_compare=False,
            padding1=0,
            padding2=0,
            props=PropertiesID(0),
        )
        self.default_tex_sampler = untracked(create_gpu_sampler(
            self.device.value(), Ptr(to=samp_info)
        ))

    def upload_texture(
        mut self, name: String, texture_data: TextureData
    ) raises -> Int:
        """Upload a texture to GPU and cache it. Returns the cache index.

        If a texture with the same name already exists, returns its index
        without re-uploading.

        Args:
            name: Cache key for the texture.
            texture_data: CPU-side RGBA8 pixel data.

        Returns:
            Index into self.texture_cache.
        """
        # ⚠⚠ A ZERO-SIZED TEXTURE ABORTS SDL, NOT JUST THIS UPLOAD.
        # `SDL_CreateGPUTexture` asserts "width, height, and
        # layer_count_or_depth must be >= 1", and the raise that follows
        # carried an EMPTY message — so the whole model failed to open with no
        # reason given. Menagerie's umi_gripper is the trigger: a
        # `<texture type="2d" file="...png"/>` declares no width/height,
        # because for a file texture the PNG carries them, and anything that
        # reads the XML's (absent) values gets 0.
        #
        # Refusing ONE texture is the right blast radius: the model still
        # loads and the geom draws untextured, which is visible and
        # recoverable, unlike an abort.
        if texture_data.width < 1 or texture_data.height < 1:
            raise Error(
                "texture '" + name + "' has no pixels ("
                + String(texture_data.width) + "x"
                + String(texture_data.height)
                + ") — a file texture takes its size from the image, so a 0"
                " here means the image did not load. Skipping it."
            )

        # Check cache
        for i in range(len(self.texture_cache)):
            if self.texture_cache[i].matches(name):
                return i

        var w = UInt32(texture_data.width)
        var h = UInt32(texture_data.height)
        var byte_size = UInt32(texture_data.byte_size())

        # ⚠ A FULL MIP CHAIN. LIBERO's scanned objects carry 4096x4096
        # textures on a 10 cm bowl; sampled at level 0 only, a screen pixel
        # lands on one of ~400 texels at random and the surface sparkles.
        # `render_context.c` calls `glGenerateMipmap` on every texture it
        # uploads, so the reference picture is the filtered one. The chain
        # is built on the GPU below (`generate_mipmaps_for_gpu_texture`),
        # which is why the usage also carries COLOR_TARGET: SDL's Metal and
        # Vulkan backends generate mips through blits/render, and both refuse
        # a sampler-only texture.
        var levels = UInt32(1)
        var dim = Int(w if w > h else h)
        while dim > 1:
            dim //= 2
            levels += 1
        # ⚠ sRGB IS A FORMAT, NOT A CONVERSION. `render_context.c` uploads an
        # `mjCOLORSPACE_SRGB` texture as `GL_SRGB8`: the GPU linearizes each
        # texel on sampling (and filters in linear), and the framebuffer is
        # NOT sRGB so nothing re-encodes it. `*_UNORM_SRGB` is that exactly;
        # decoding on the CPU into 8-bit linear would band the darks.
        var tex_format = GPUTextureFormat.GPU_TEXTUREFORMAT_R8G8B8A8_UNORM
        if texture_data.srgb:
            tex_format = GPUTextureFormat.GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB
        var tex_info = GPUTextureCreateInfo(
            type=GPUTextureType.GPU_TEXTURETYPE_2D,
            format=tex_format,
            usage=GPUTextureUsageFlags.GPU_TEXTUREUSAGE_SAMPLER
            | GPUTextureUsageFlags.GPU_TEXTUREUSAGE_COLOR_TARGET,
            width=w,
            height=h,
            layer_count_or_depth=1,
            num_levels=levels,
            sample_count=GPUSampleCount.GPU_SAMPLECOUNT_1,
            props=PropertiesID(0),
        )
        var gpu_tex = untracked(create_gpu_texture(self.device.value(), Ptr(to=tex_info)))

        # Upload via transfer buffer
        var tb_info = GPUTransferBufferCreateInfo(
            usage=GPUTransferBufferUsage.GPU_TRANSFERBUFFERUSAGE_UPLOAD,
            size=byte_size,
            props=PropertiesID(0),
        )
        var tb = untracked(create_gpu_transfer_buffer(self.device.value(), Ptr(to=tb_info)))
        var mapped = map_gpu_transfer_buffer(self.device.value(), tb, False)
        var mapped_u8 = mapped.unsafe_bitcast[UInt8]()
        for i in range(Int(byte_size)):
            mapped_u8[unsafe_offset=i] = texture_data.pixels[i]
        unmap_gpu_transfer_buffer(self.device.value(), tb)

        var cmd = acquire_gpu_command_buffer(self.device.value())
        var cp = begin_gpu_copy_pass(cmd)
        var src = GPUTextureTransferInfo(
            transfer_buffer=untracked(tb),
            offset=0,
            pixels_per_row=w,
            rows_per_layer=h,
        )
        var dst = GPUTextureRegion(
            texture=untracked(gpu_tex),
            mip_level=0,
            layer=0,
            x=0,
            y=0,
            z=0,
            w=w,
            h=h,
            d=1,
        )
        upload_to_gpu_texture(cp, Ptr(to=src), Ptr(to=dst), False)
        end_gpu_copy_pass(cp)
        # Outside any pass, as SDL requires; same command buffer, so the
        # chain is complete before the first draw that samples it.
        if levels > 1:
            generate_mipmaps_for_gpu_texture(cmd, gpu_tex)
        submit_gpu_command_buffer(cmd)
        release_gpu_transfer_buffer(self.device.value(), tb)

        # Create a trilinear sampler with REPEAT address mode
        var samp_info = GPUSamplerCreateInfo(
            min_filter=GPUFilter.GPU_FILTER_LINEAR,
            mag_filter=GPUFilter.GPU_FILTER_LINEAR,
            mipmap_mode=GPUSamplerMipmapMode.GPU_SAMPLERMIPMAPMODE_LINEAR,
            address_mode_u=GPUSamplerAddressMode.GPU_SAMPLERADDRESSMODE_REPEAT,
            address_mode_v=GPUSamplerAddressMode.GPU_SAMPLERADDRESSMODE_REPEAT,
            address_mode_w=GPUSamplerAddressMode.GPU_SAMPLERADDRESSMODE_REPEAT,
            mip_lod_bias=0.0,
            max_anisotropy=1.0,
            compare_op=GPUCompareOp.GPU_COMPAREOP_ALWAYS,
            min_lod=0.0,
            max_lod=Float32(levels),
            enable_anisotropy=False,
            enable_compare=False,
            padding1=0,
            padding2=0,
            props=PropertiesID(0),
        )
        var tex_sampler = untracked(create_gpu_sampler(self.device.value(), Ptr(to=samp_info)))

        # Add to cache
        self.texture_cache.append(
            TextureCacheEntry(name, gpu_tex, tex_sampler, w, h)
        )
        return len(self.texture_cache) - 1

    # --- Public Drawing API ---

    def begin_frame(mut self):
        """Begin a new frame: clear draw command lists."""
        self.solid_draws.clear()
        self.line_vertex_data.clear()
        self.line_colors.clear()
        self.text_vertex_data.clear()
        self.has_ground = False
        self.ground_texture_idx = -1

    def draw_text(
        mut self,
        x: Float32,
        y: Float32,
        text: String,
        color: Color = Color(255, 255, 255, 255),
        scale: Int = 2,
    ):
        """Draw a string of ASCII text as a HUD overlay (screen-space pixels).

        Each character is rendered as a textured quad using the font atlas.
        Text renders top-left at (x, y). Characters advance by 8*scale pixels.

        Args:
            x: Left edge of text in screen pixels.
            y: Top edge of text in screen pixels.
            text: ASCII string to render.
            color: RGBA text color.
            scale: Pixel scale factor (default 2 = 16×16 per char).
        """
        var cr = Float32(color.r) / 255.0
        var cg = Float32(color.g) / 255.0
        var cb = Float32(color.b) / 255.0
        var ca = Float32(color.a) / 255.0
        var glyph_w = Float32(8 * scale)
        var glyph_h = Float32(8 * scale)
        var cx = x
        for i in range(text.byte_length()):
            # ⚠ PER CHARACTER, not once per call. The check used to sit above
            # the loop, which let a single long string append past the end of
            # a buffer sized for MAX_TEXT_CHARS quads — the upload path copies
            # `len(text_vertex_data)` floats with no clamp of its own, so that
            # was a write past the mapped transfer buffer, not a dropped glyph.
            # It never fired only because every caller was short.
            if not self._text_budget_ok():
                break
            var c = text.as_bytes()[i]
            var uv = glyph_uv(c)
            var u0 = uv[0]
            var v0 = uv[1]
            var u1 = uv[2]
            var v1 = uv[3]
            # 4 vertices: top-left, top-right, bottom-right, bottom-left
            # Vertex 0: top-left
            self.text_vertex_data.append(cx)
            self.text_vertex_data.append(y)
            self.text_vertex_data.append(u0)
            self.text_vertex_data.append(v0)
            self.text_vertex_data.append(cr)
            self.text_vertex_data.append(cg)
            self.text_vertex_data.append(cb)
            self.text_vertex_data.append(ca)
            # Vertex 1: top-right
            self.text_vertex_data.append(cx + glyph_w)
            self.text_vertex_data.append(y)
            self.text_vertex_data.append(u1)
            self.text_vertex_data.append(v0)
            self.text_vertex_data.append(cr)
            self.text_vertex_data.append(cg)
            self.text_vertex_data.append(cb)
            self.text_vertex_data.append(ca)
            # Vertex 2: bottom-right
            self.text_vertex_data.append(cx + glyph_w)
            self.text_vertex_data.append(y + glyph_h)
            self.text_vertex_data.append(u1)
            self.text_vertex_data.append(v1)
            self.text_vertex_data.append(cr)
            self.text_vertex_data.append(cg)
            self.text_vertex_data.append(cb)
            self.text_vertex_data.append(ca)
            # Vertex 3: bottom-left
            self.text_vertex_data.append(cx)
            self.text_vertex_data.append(y + glyph_h)
            self.text_vertex_data.append(u0)
            self.text_vertex_data.append(v1)
            self.text_vertex_data.append(cr)
            self.text_vertex_data.append(cg)
            self.text_vertex_data.append(cb)
            self.text_vertex_data.append(ca)
            cx += glyph_w

    def draw_sphere(
        mut self,
        center: Vec3,
        radius: Float64,
        color: Color = Color(255, 255, 255, 255),
        shininess: Float32 = 0.5,
        specular: Float32 = 0.5,
        reflectance: Float32 = 0.0,
        emission: Float32 = 0.0,
        texture_name: String = String(""),
        texture_path: String = String(""),
        texture_colorspace: Int = 0,
        texrepeat_u: Float32 = 1.0,
        texrepeat_v: Float32 = 1.0,
        tex_cube: Bool = False,
        tex_scale: Vec3 = Vec3(1.0, 1.0, 1.0),
    ) raises:
        """Draw a solid sphere, optionally textured.

        Args:
            center: Sphere center in world space.
            radius: Sphere radius.
            color: Surface color.
            shininess: Specular exponent scaling (0-1).
            specular: Specular intensity (0-1).
            reflectance: Reflectance coefficient (0-1).
            emission: Emissive intensity (0-1).
            texture_name: Cache key for the texture (empty = no texture).
            texture_path: Path to the PNG texture file (empty = no texture).
            texture_colorspace: `<texture colorspace>` (`COLORSPACE_AUTO` reads the PNG).
            texrepeat_u: Texture repeat in U direction.
            texrepeat_v: Texture repeat in V direction.
            tex_cube: Sample as a cube/box texture from the object-space position.
            tex_scale: Object-space position scale for the cube lookup.
        """
        # Load and cache texture if provided
        var tex_idx = -1
        if texture_name.byte_length() > 0 and texture_path.byte_length() > 0:
            for ti in range(len(self.texture_cache)):
                if self.texture_cache[ti].matches(texture_name):
                    tex_idx = ti
                    break
            if tex_idx < 0:
                try:
                    var tex_data = load_texture_png(texture_path, texture_colorspace)
                    tex_idx = self.upload_texture(texture_name, tex_data)
                    print(
                        "Loaded texture '",
                        texture_name,
                        "':",
                        tex_data.width,
                        "x",
                        tex_data.height,
                    )
                except e:
                    print("Warning: texture load failed:", String(e))
                    pass

        var model = Mat4.compose(
            center, Quat.identity(), Vec3(radius, radius, radius)
        )
        var uniforms = ObjectUniforms()
        uniforms.model = mat4_to_gpu_f32(model)
        uniforms.color = color_to_vec4(color)
        uniforms.material[0] = shininess
        uniforms.material[1] = specular
        uniforms.material[2] = Float32(1.0) if tex_idx >= 0 else reflectance
        uniforms.material[3] = emission
        uniforms.tex_params[0] = texrepeat_u
        uniforms.tex_params[1] = texrepeat_v
        uniforms.tex_params[2] = Float32(1.0) if tex_cube else Float32(0.0)
        uniforms.tex_scale[0] = Float32(tex_scale.x)
        uniforms.tex_scale[1] = Float32(tex_scale.y)
        uniforms.tex_scale[2] = Float32(tex_scale.z)

        self.solid_draws.append(
            SolidDrawCommand(0, uniforms, texture_cache_idx=tex_idx)
        )

    def take_click(mut self) -> Bool:
        """True once per mouse press; clears on read like `take_key`."""
        var c = self.mouse_clicked
        self.mouse_clicked = False
        return c

    def set_text_input_mode(mut self, on: Bool):
        """Suspend the renderer's own key bindings while a field has focus."""
        self.text_input_mode = on

    def request_camera(mut self, index: Int):
        """Same channel the 1-9 keys use; read once by the next render."""
        self.camera_switch_request = index

    def request_screenshot(mut self):
        self.screenshot_requested = True

    def is_recording(self) -> Bool:
        return self.recorder.is_recording

    def recording_frames(self) -> Int:
        return self.recorder.frame_count

    def toggle_recording(mut self) raises:
        """Start/stop video capture — the V key's behaviour, as a method."""
        if self.recorder.is_recording:
            self.stop_recording()
        else:
            self.start_recording(
                "recording_" + String(self.screenshot_counter) + ".mp4"
            )

    def paused(self) -> Bool:
        return self.is_paused

    def toggle_pause(mut self):
        self.is_paused = not self.is_paused

    def scene_width(self) -> Int:
        """Window width minus the reserved UI strip; at least 1."""
        var w = self.width - self.ui_sidebar_width
        return w if w > 1 else 1

    def set_pip_views(mut self, var views: List[PipView]):
        """Replace the picture-in-picture views drawn this frame."""
        self.pip_views = views^

    def set_ui_sidebar_width(mut self, w: Int):
        """Reserve `w` pixels on the left for UI and re-fit the camera.

        Correcting the aspect here is the whole point: without it the scene
        would keep a full-window aspect while being drawn into a narrower
        viewport, which reads as everything being horizontally stretched — a
        subtle wrongness that is easy to blame on the model.
        """
        self.ui_sidebar_width = w if w > 0 else 0
        self.camera.set_screen_size(self.scene_width(), self.height)

    def set_pointer_claimed(mut self, on: Bool):
        """Declare that an overlay ImGui cannot speak for owns the pointer.

        See `pointer_claimed`. Set it EVERY frame — it is a level, not an
        event, and a latched True would leave the camera permanently frozen.
        """
        self.pointer_claimed = on

    def set_capture_scene_only(mut self, on: Bool):
        """Whether screenshots/recordings exclude the reserved UI strip."""
        self.capture_scene_only = on

    def _capture_x(self) -> Int:
        """First column of the capture region."""
        return self.ui_sidebar_width if self.capture_scene_only else 0

    def _capture_w(self) -> Int:
        """Columns to capture; 0 means the whole frame (the recorder's
        "no crop" sentinel), which is what a full-window capture wants."""
        if not self.capture_scene_only or self.ui_sidebar_width <= 0:
            return 0
        return self.scene_width()

    # --- Dear ImGui overlay ---

    def imgui_init(mut self) raises -> Bool:
        """Attach an ImGui context to THIS window and device. Idempotent.

        Returns False (having printed why) when the shim is missing or ImGui
        declines the device, so a viewer can fall back instead of dying: the
        3D scene is perfectly usable without a UI, and an FFI dependency that
        takes the whole tool down with it would be a poor trade.
        """
        if self.imgui_on:
            return True
        if not self.initialized:
            print("imgui_init: renderer not initialized")
            return False
        if not imgui_shim_available():
            print(
                "imgui_init: Dear ImGui shim not built — run"
                " `pixi run build-imgui`"
            )
            return False
        self.imgui_on = ig_init(
            self.window.value(),
            self.device.value(),
            UInt32(Int(self.swapchain_format)),
        )
        if not self.imgui_on:
            print("imgui_init: ImGui declined this GPU device")
        return self.imgui_on

    def imgui_new_frame(mut self) raises:
        """Open an ImGui frame. Call BEFORE building any widgets, and before
        `render_frame`; `end_frame` closes it."""
        if not self.imgui_on:
            return
        ig_new_frame()
        self.imgui_frame_open = True

    def imgui_active(self) -> Bool:
        return self.imgui_on

    def imgui_close(mut self) raises:
        """Detach ImGui. Must run BEFORE the device and window are destroyed,
        since the backend releases GPU objects it created on them."""
        if not self.imgui_on:
            return
        # An open frame here means the app quit mid-frame. ImGui's shutdown
        # tolerates that; the flag is cleared so a later re-init starts clean.
        self.imgui_frame_open = False
        self.imgui_on = False
        ig_shutdown()

    def _text_budget_ok(mut self) -> Bool:
        """Room for one more quad? Complains ONCE if not.

        Overflow used to be a silent `return`, which is the worst shape for
        this failure: the frame still renders, just missing whatever came
        last, so a half-drawn list reads as a layout or logic bug rather than
        as a budget that ran out. One line on stderr costs nothing and names
        the actual cause.
        """
        if len(self.text_vertex_data) < MAX_TEXT_CHARS * 4 * 8:
            return True
        if not self.text_budget_warned:
            self.text_budget_warned = True
            print(
                "[renderer3d] HUD/UI quad budget exhausted at",
                MAX_TEXT_CHARS,
                "quads — text and rectangles beyond this are DROPPED this"
                " frame and every frame after. Raise MAX_TEXT_CHARS in"
                " renderer3d.mojo.",
            )
        return False

    def draw_rect(
        mut self,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        color: Color = Color(30, 30, 40, 220),
    ):
        """Filled screen-space rectangle, in pixels, top-left origin.

        Rides the FONT pipeline: the same textured-quad buffer `draw_text`
        writes into, with UVs pinned to the atlas's solid cell. That keeps the
        UI to one draw pass and needs no second shader — but it also means
        rectangles share `MAX_TEXT_CHARS` budget with text, so a panel is
        worth a few characters.
        """
        if not self._text_budget_ok():
            return
        var cr = Float32(color.r) / 255.0
        var cg = Float32(color.g) / 255.0
        var cb = Float32(color.b) / 255.0
        var ca = Float32(color.a) / 255.0
        var uv = solid_uv()
        var u = uv[0]
        var v = uv[1]

        # ⚠ Array has no positional-variadic ctor; fill then assign.
        var xs = Array[Float32, 4](fill=Float32(0))
        var ys = Array[Float32, 4](fill=Float32(0))
        xs[0] = x
        ys[0] = y
        xs[1] = x + w
        ys[1] = y
        xs[2] = x + w
        ys[2] = y + h
        xs[3] = x
        ys[3] = y + h
        for i in range(4):
            self.text_vertex_data.append(xs[i])
            self.text_vertex_data.append(ys[i])
            self.text_vertex_data.append(u)
            self.text_vertex_data.append(v)
            self.text_vertex_data.append(cr)
            self.text_vertex_data.append(cg)
            self.text_vertex_data.append(cb)
            self.text_vertex_data.append(ca)

    def take_key(mut self) -> Int:
        """Consume the last unclaimed keycode (0 if none since the last call).

        Clearing on read is what makes this usable as an EVENT rather than a
        state: a viewer polling once per frame gets each press exactly once.
        """
        var k = self.last_key
        self.last_key = 0
        return k

    def draw_ellipsoid(
        mut self,
        center: Vec3,
        orientation: Quat,
        radii: Vec3,
        color: Color = Color(255, 255, 255, 255),
        shininess: Float32 = 0.5,
        specular: Float32 = 0.5,
        reflectance: Float32 = 0.0,
        emission: Float32 = 0.0,
    ) raises:
        """Draw a solid ellipsoid — the sphere mesh under a non-uniform scale.

        `draw_sphere` already composes its model matrix with a scale vector
        `(r, r, r)`; an ellipsoid is the same call with three different radii
        and a real orientation, so this costs a mesh reuse and nothing else.

        Added 2026-08-03: `mjGEOM_ELLIPSOID` had no branch in
        `ModelDefFromXML.render_body_geoms` and no fallback either, so every
        ellipsoid geom was silently INVISIBLE — quadruped's torso, and geoms in
        swimmer, fish and finger. A viewer that hides a robot's torso is worse
        than no viewer, since it reads as a broken model rather than a missing
        draw call.

        Args:
            center: Ellipsoid center in world space.
            orientation: World orientation.
            radii: Semi-axes (x, y, z) in the ellipsoid's local frame.
            color: Surface color.
            shininess: Specular exponent scaling (0-1).
            specular: Specular intensity (0-1).
            reflectance: Reflectance coefficient (0-1).
            emission: Emissive intensity (0-1).
        """
        var model = Mat4.compose(center, orientation, radii)
        var uniforms = ObjectUniforms()
        uniforms.model = mat4_to_gpu_f32(model)
        uniforms.color = color_to_vec4(color)
        uniforms.material[0] = shininess
        uniforms.material[1] = specular
        uniforms.material[2] = reflectance
        uniforms.material[3] = emission

        self.solid_draws.append(
            SolidDrawCommand(0, uniforms, texture_cache_idx=-1)
        )

    def draw_capsule(
        mut self,
        center: Vec3,
        orientation: Quat,
        radius: Float64,
        half_height: Float64,
        axis: Int = 2,
        color: Color = Color(255, 255, 255, 255),
        shininess: Float32 = 0.5,
        specular: Float32 = 0.5,
        reflectance: Float32 = 0.0,
        emission: Float32 = 0.0,
        texture_name: String = String(""),
        texture_path: String = String(""),
        texture_colorspace: Int = 0,
        texrepeat_u: Float32 = 1.0,
        texrepeat_v: Float32 = 1.0,
        tex_cube: Bool = False,
        tex_scale: Vec3 = Vec3(1.0, 1.0, 1.0),
    ) raises:
        """Draw a solid capsule, optionally textured.

        Args:
            center: Capsule center in world space.
            orientation: Capsule orientation.
            radius: Capsule radius.
            half_height: Half-height of cylindrical section.
            axis: Local axis (0=X, 1=Y, 2=Z).
            color: Surface color.
            shininess: Specular exponent scaling (0-1).
            specular: Specular intensity (0-1).
            reflectance: Reflectance coefficient (0-1).
            emission: Emissive intensity (0-1).
            texture_name: Cache key for the texture (empty = no texture).
            texture_path: Path to the PNG texture file (empty = no texture).
            texture_colorspace: `<texture colorspace>` (`COLORSPACE_AUTO` reads the PNG).
            texrepeat_u: Texture repeat in U direction.
            texrepeat_v: Texture repeat in V direction.
            tex_cube: Sample as a cube/box texture from the object-space position.
            tex_scale: Object-space position scale for the cube lookup.
        """
        # Load and cache texture if provided
        var tex_idx = -1
        if texture_name.byte_length() > 0 and texture_path.byte_length() > 0:
            for ti in range(len(self.texture_cache)):
                if self.texture_cache[ti].matches(texture_name):
                    tex_idx = ti
                    break
            if tex_idx < 0:
                try:
                    var tex_data = load_texture_png(texture_path, texture_colorspace)
                    tex_idx = self.upload_texture(texture_name, tex_data)
                    print(
                        "Loaded texture '",
                        texture_name,
                        "':",
                        tex_data.width,
                        "x",
                        tex_data.height,
                    )
                except e:
                    print("Warning: texture load failed:", String(e))
                    pass

        var f_radius = Float32(radius)
        var f_half = Float32(half_height)

        # Look up or create capsule mesh
        var cache_idx = -1
        for i in range(len(self.capsule_cache)):
            if self.capsule_cache[i].matches(f_radius, f_half):
                cache_idx = i
                break

        if cache_idx < 0:
            # Generate new capsule mesh
            var mesh_data = generate_capsule(f_radius, f_half)
            var handle = self._upload_mesh(mesh_data)
            self.capsule_cache.append(
                CapsuleCacheEntry(f_radius, f_half, handle^)
            )
            cache_idx = len(self.capsule_cache) - 1

        # Build model matrix
        # Capsule geometry is along Z-axis. Apply pre-rotation if axis != 2.
        var pre_rot = Quat.identity()
        if axis == 0:
            # Rotate Z -> X: 90 degrees around Y
            pre_rot = Quat.from_axis_angle(Vec3.unit_y(), 1.5707963267949)
        elif axis == 1:
            # Rotate Z -> Y: -90 degrees around X
            pre_rot = Quat.from_axis_angle(Vec3.unit_x(), -1.5707963267949)

        var final_quat = orientation
        # Apply pre-rotation: rotate local capsule geometry, then apply orientation
        var model = Mat4.from_quat(final_quat, center) @ Mat4.from_quat(
            pre_rot, Vec3.zero()
        )

        var uniforms = ObjectUniforms()
        uniforms.model = mat4_to_gpu_f32(model)
        uniforms.color = color_to_vec4(color)
        uniforms.material[0] = shininess
        uniforms.material[1] = specular
        uniforms.material[2] = Float32(1.0) if tex_idx >= 0 else reflectance
        uniforms.material[3] = emission
        uniforms.tex_params[0] = texrepeat_u
        uniforms.tex_params[1] = texrepeat_v
        uniforms.tex_params[2] = Float32(1.0) if tex_cube else Float32(0.0)
        uniforms.tex_scale[0] = Float32(tex_scale.x)
        uniforms.tex_scale[1] = Float32(tex_scale.y)
        uniforms.tex_scale[2] = Float32(tex_scale.z)

        self.solid_draws.append(
            SolidDrawCommand(
                0,
                uniforms,
                is_capsule=True,
                capsule_cache_idx=cache_idx,
                texture_cache_idx=tex_idx,
            )
        )

    def draw_cylinder(
        mut self,
        center: Vec3,
        orientation: Quat,
        radius: Float64,
        half_height: Float64,
        axis: Int = 2,
        color: Color = Color(255, 255, 255, 255),
        shininess: Float32 = 0.5,
        specular: Float32 = 0.5,
        reflectance: Float32 = 0.0,
        emission: Float32 = 0.0,
        texture_name: String = String(""),
        texture_path: String = String(""),
        texture_colorspace: Int = 0,
        texrepeat_u: Float32 = 1.0,
        texrepeat_v: Float32 = 1.0,
        tex_cube: Bool = False,
        tex_scale: Vec3 = Vec3(1.0, 1.0, 1.0),
    ) raises:
        """Draw a solid cylinder with flat disc caps, optionally textured.

        Args:
            center: Cylinder center in world space.
            orientation: Cylinder orientation.
            radius: Cylinder radius.
            half_height: Half-height of the cylinder.
            axis: Local axis (0=X, 1=Y, 2=Z).
            color: Surface color.
            shininess: Specular exponent scaling (0-1).
            specular: Specular intensity (0-1).
            reflectance: Reflectance coefficient (0-1).
            emission: Emissive intensity (0-1).
            texture_name: Cache key for the texture (empty = no texture).
            texture_path: Path to the PNG texture file (empty = no texture).
            texture_colorspace: `<texture colorspace>` (`COLORSPACE_AUTO` reads the PNG).
            texrepeat_u: Texture repeat in U direction.
            texrepeat_v: Texture repeat in V direction.
            tex_cube: Sample as a cube/box texture from the object-space position.
            tex_scale: Object-space position scale for the cube lookup.
        """
        # Load and cache texture if provided
        var tex_idx = -1
        if texture_name.byte_length() > 0 and texture_path.byte_length() > 0:
            for ti in range(len(self.texture_cache)):
                if self.texture_cache[ti].matches(texture_name):
                    tex_idx = ti
                    break
            if tex_idx < 0:
                try:
                    var tex_data = load_texture_png(texture_path, texture_colorspace)
                    tex_idx = self.upload_texture(texture_name, tex_data)
                    print(
                        "Loaded texture '",
                        texture_name,
                        "':",
                        tex_data.width,
                        "x",
                        tex_data.height,
                    )
                except e:
                    print("Warning: texture load failed:", String(e))
                    pass

        var f_radius = Float32(radius)
        var f_half = Float32(half_height)

        # Look up or create cylinder mesh
        var cache_idx = -1
        for i in range(len(self.cylinder_cache)):
            if self.cylinder_cache[i].matches(f_radius, f_half):
                cache_idx = i
                break

        if cache_idx < 0:
            var mesh_data = generate_cylinder(f_radius, f_half)
            var handle = self._upload_mesh(mesh_data)
            self.cylinder_cache.append(
                CapsuleCacheEntry(f_radius, f_half, handle^)
            )
            cache_idx = len(self.cylinder_cache) - 1

        # Build model matrix (same axis pre-rotation as capsule)
        var pre_rot = Quat.identity()
        if axis == 0:
            pre_rot = Quat.from_axis_angle(Vec3.unit_y(), 1.5707963267949)
        elif axis == 1:
            pre_rot = Quat.from_axis_angle(Vec3.unit_x(), -1.5707963267949)

        var final_quat = orientation
        var model = Mat4.from_quat(final_quat, center) @ Mat4.from_quat(
            pre_rot, Vec3.zero()
        )

        var uniforms = ObjectUniforms()
        uniforms.model = mat4_to_gpu_f32(model)
        uniforms.color = color_to_vec4(color)
        uniforms.material[0] = shininess
        uniforms.material[1] = specular
        # material.z > 0 tells the shader to sample the texture
        uniforms.material[2] = Float32(1.0) if tex_idx >= 0 else reflectance
        uniforms.material[3] = emission
        uniforms.tex_params[0] = texrepeat_u
        uniforms.tex_params[1] = texrepeat_v
        uniforms.tex_params[2] = Float32(1.0) if tex_cube else Float32(0.0)
        uniforms.tex_scale[0] = Float32(tex_scale.x)
        uniforms.tex_scale[1] = Float32(tex_scale.y)
        uniforms.tex_scale[2] = Float32(tex_scale.z)

        self.solid_draws.append(
            SolidDrawCommand(
                0,
                uniforms,
                is_cylinder=True,
                cylinder_cache_idx=cache_idx,
                texture_cache_idx=tex_idx,
            )
        )

    def draw_mesh(
        mut self,
        name: String,
        file_path: String,
        center: Vec3,
        orientation: Quat,
        scale: Vec3 = Vec3(1.0, 1.0, 1.0),
        color: Color = Color(200, 200, 200, 255),
        shininess: Float32 = 0.5,
        specular: Float32 = 0.5,
        reflectance: Float32 = 0.0,
        emission: Float32 = 0.0,
        texture_name: String = String(""),
        texture_path: String = String(""),
        texture_colorspace: Int = 0,
        texrepeat_u: Float32 = 1.0,
        texrepeat_v: Float32 = 1.0,
        tex_cube: Bool = False,
        tex_scale: Vec3 = Vec3(1.0, 1.0, 1.0),
    ) raises:
        """Draw a solid mesh loaded from an STL file, optionally textured.

        Meshes are cached by name — the STL file is only loaded and uploaded
        on the first call for each unique name. Textures are also cached by name.

        Args:
            name: Cache key for the mesh (e.g. "gripper_link").
            file_path: Path to the binary STL file.
            center: Mesh center in world space.
            orientation: Mesh orientation quaternion.
            scale: Per-axis scale factors.
            color: Surface color (modulates texture if present).
            shininess: Specular exponent scaling (0-1).
            specular: Specular intensity (0-1).
            reflectance: Reflectance coefficient (0-1).
            emission: Emissive intensity (0-1).
            texture_name: Cache key for the texture (empty = no texture).
            texture_path: Path to the PNG texture file (empty = no texture).
            texture_colorspace: `<texture colorspace>` (`COLORSPACE_AUTO` reads the PNG).
            texrepeat_u: Texture repeat in U direction.
            texrepeat_v: Texture repeat in V direction.
            tex_cube: Sample as a cube/box texture from the object-space position.
            tex_scale: Object-space position scale for the cube lookup.
        """
        # Look up or create mesh in cache
        var cache_idx = -1
        for i in range(len(self.mesh_cache)):
            if self.mesh_cache[i].matches(name):
                cache_idx = i
                break

        if cache_idx < 0:
            # Load STL and upload to GPU
            try:
                var mesh_data = load_stl(file_path)
                var handle = self._upload_mesh(mesh_data)
                self.mesh_cache.append(MeshCacheEntry(name, handle^))
                cache_idx = len(self.mesh_cache) - 1
            except e:
                # ⚠ ONCE, NOT PER FRAME, and never silently. See `mesh_failed`.
                var seen = False
                for k in range(len(self.mesh_failed)):
                    if self.mesh_failed[k] == name:
                        seen = True
                if not seen:
                    self.mesh_failed.append(name)
                    print(
                        "Warning: mesh '", name, "' did not load — DRAWN AS"
                        " NOTHING. path='", file_path, "' :", String(e),
                    )
                return

        # Load and cache texture if provided
        var tex_idx = -1
        if texture_name.byte_length() > 0 and texture_path.byte_length() > 0:
            # Check cache first (avoid reloading PNG every frame)
            for ti in range(len(self.texture_cache)):
                if self.texture_cache[ti].matches(texture_name):
                    tex_idx = ti
                    break
            if tex_idx < 0:
                try:
                    var tex_data = load_texture_png(texture_path, texture_colorspace)
                    tex_idx = self.upload_texture(texture_name, tex_data)
                    print(
                        "Loaded texture '",
                        texture_name,
                        "':",
                        tex_data.width,
                        "x",
                        tex_data.height,
                    )
                except e:
                    print("Warning: texture load failed:", String(e))
                    pass

        # Build model matrix: rotation + translation, then apply scale
        var rot_mat = Mat4.from_quat(orientation, center)
        # Scale columns of the 3x3 rotation submatrix
        rot_mat.m00 *= scale.x
        rot_mat.m01 *= scale.x
        rot_mat.m02 *= scale.x
        rot_mat.m10 *= scale.y
        rot_mat.m11 *= scale.y
        rot_mat.m12 *= scale.y
        rot_mat.m20 *= scale.z
        rot_mat.m21 *= scale.z
        rot_mat.m22 *= scale.z

        var uniforms = ObjectUniforms()
        uniforms.model = mat4_to_gpu_f32(rot_mat)
        uniforms.color = color_to_vec4(color)
        uniforms.material[0] = shininess
        uniforms.material[1] = specular
        # material.z > 0 tells the shader to sample the texture
        uniforms.material[2] = Float32(1.0) if tex_idx >= 0 else reflectance
        uniforms.material[3] = emission
        uniforms.tex_params[0] = texrepeat_u
        uniforms.tex_params[1] = texrepeat_v
        uniforms.tex_params[2] = Float32(1.0) if tex_cube else Float32(0.0)
        uniforms.tex_scale[0] = Float32(tex_scale.x)
        uniforms.tex_scale[1] = Float32(tex_scale.y)
        uniforms.tex_scale[2] = Float32(tex_scale.z)

        self.solid_draws.append(
            SolidDrawCommand(
                0,
                uniforms,
                is_mesh=True,
                mesh_cache_idx=cache_idx,
                texture_cache_idx=tex_idx,
            )
        )

    def draw_heightfield(
        mut self,
        name: String,
        center: Vec3,
        orientation: Quat,
        nrow: Int,
        ncol: Int,
        size_x: Float64,
        size_y: Float64,
        size_z: Float64,
        grid: List[Float64],
        adr: Int,
        revision: Int,
        color: Color = Color(80, 110, 140, 255),
        shininess: Float32 = 0.2,
        specular: Float32 = 0.15,
        reflectance: Float32 = 0.0,
        emission: Float32 = 0.0,
    ) raises:
        """Draw a MuJoCo `<hfield>` elevation surface.

        Args:
            name: Cache key.
            center: The geom's world position.
            orientation: The geom's world orientation.
            nrow: Grid rows (the y axis).
            ncol: Grid columns (the x axis).
            size_x: Half-extent along local x, `hfield_size[0]`.
            size_y: Half-extent along local y, `hfield_size[1]`.
            size_z: Elevation scale, `hfield_size[2]`.
            grid: The elevation grid, NORMALISED TO [0, 1].
            adr: Offset of this field's first sample within `grid`.
            revision: Bump to force a rebuild; see `HfieldCacheEntry`.
            color: Surface colour.
            shininess: Specular exponent scaling (0-1).
            specular: Specular intensity (0-1).
            reflectance: Reflectance coefficient (0-1).
            emission: Emissive intensity (0-1).

        ⚠ THE VERTEX CONVENTION IS `mj_rayHfield`'s, COPIED RATHER THAN
        RE-DERIVED. Sample (r, c) sits at
        `x = 2*size_x*c/(ncol-1) - size_x`, `y = 2*size_y*r/(nrow-1) - size_y`,
        `z = grid[adr + r*ncol + c] * size_z`, and each cell is split
        `(r,c)-(r,c+1)-(r+1,c+1)` then `(r,c)-(r+1,c+1)-(r+1,c)`. Getting the
        split or the winding wrong draws a surface that looks plausible and does
        not match the one the rangefinders hit — which is the single thing this
        view exists to check.

        ⚠ THE VERTICES ARE LOCAL, NOT WORLD, unlike `draw_skin`'s. The geom's
        pose goes in the model matrix, so a terrain on a moving body follows it
        for free and the cached buffer stays valid when it does.

        ⚠ THE BASE IS NOT DRAWN HERE. A `<hfield>` is a surface on a box that
        extends `size[3]` below z=0; this draws the surface only, and the caller
        adds the base with `draw_box` if it wants one. Keeping them apart means
        a caller can suppress a base that is 0.1 m thick under a 60 m field
        without losing the terrain.
        """
        if nrow < 2 or ncol < 2:
            return
        var need = adr + nrow * ncol
        if need > len(grid):
            # ⚠ REPORTED ONCE, THEN DRAWN AS NOTHING — `draw_mesh`'s rule. A
            # short grid is a wiring fault upstream, and a per-frame print
            # would bury it under itself.
            var seen = False
            for k in range(len(self.mesh_failed)):
                if self.mesh_failed[k] == name:
                    seen = True
            if not seen:
                self.mesh_failed.append(name)
                print(
                    "Warning: heightfield '", name, "' needs", need,
                    "samples and the grid holds", len(grid),
                    "— DRAWN AS NOTHING.",
                )
            return

        var idx = -1
        for i in range(len(self.hfield_cache)):
            if self.hfield_cache[i].name == name:
                idx = i
                break

        var nv = nrow * ncol
        if idx < 0:
            # ── first sight: topology, buffers, staging ──────────────────
            var seed = MeshData()
            seed.vertices.reserve(nv)
            seed.indices.reserve(6 * (nrow - 1) * (ncol - 1))
            for _ in range(nv):
                seed.vertices.append(GPUVertex(px=0, py=0, pz=0, nz=Float32(1)))
            for r in range(nrow - 1):
                for c in range(ncol - 1):
                    var i00 = UInt32(r * ncol + c)
                    var i01 = UInt32(r * ncol + c + 1)
                    var i11 = UInt32((r + 1) * ncol + c + 1)
                    var i10 = UInt32((r + 1) * ncol + c)
                    seed.indices.append(i00)
                    seed.indices.append(i01)
                    seed.indices.append(i11)
                    seed.indices.append(i00)
                    seed.indices.append(i11)
                    seed.indices.append(i10)

            var vb_bytes = seed.vertex_byte_size()
            var handle = self._upload_mesh(seed)
            self.mesh_cache.append(MeshCacheEntry(name, handle^))
            var mesh_idx = len(self.mesh_cache) - 1

            var tb_info = GPUTransferBufferCreateInfo(
                usage=GPUTransferBufferUsage.GPU_TRANSFERBUFFERUSAGE_UPLOAD,
                size=UInt32(vb_bytes),
                props=PropertiesID(0),
            )
            var transfer = untracked(
                create_gpu_transfer_buffer(self.device.value(), Ptr(to=tb_info))
            )

            var entry = HfieldCacheEntry(
                name, mesh_idx, nrow, ncol, transfer, vb_bytes
            )
            for _ in range(nv):
                entry.verts.append(GPUVertex(px=0, py=0, pz=0))
            self.hfield_cache.append(entry^)
            idx = len(self.hfield_cache) - 1
            print(
                "Built heightfield '", name, "':", nrow, "x", ncol, "=",
                2 * (nrow - 1) * (ncol - 1), "triangles",
            )

        # ── rebuild the elevations, only when they moved ──────────────────
        if self.hfield_cache[idx].revision != revision:
            self.hfield_cache[idx].revision = revision
            var dx = 2.0 * size_x / Float64(ncol - 1)
            var dy = 2.0 * size_y / Float64(nrow - 1)
            for r in range(nrow):
                for c in range(ncol):
                    var v = r * ncol + c
                    var z = hfield_node_z(grid, adr, ncol, r, c, size_z)
                    # ⚠ NORMALS BY CENTRAL DIFFERENCE ON THE GRID, not by
                    # averaging the two triangles a vertex belongs to. The
                    # difference is visible: per-face normals on a bump field
                    # give the faceted look that makes a smooth bowl read as a
                    # tessellation artefact, which is exactly the wrong thing
                    # for a view whose job is judging the terrain's SHAPE.
                    var cm = c - 1 if c > 0 else c
                    var cp = c + 1 if c < ncol - 1 else c
                    var rm = r - 1 if r > 0 else r
                    var rp = r + 1 if r < nrow - 1 else r
                    var zdx = (
                        grid[adr + r * ncol + cp] - grid[adr + r * ncol + cm]
                    ) * size_z
                    var zdy = (
                        grid[adr + rp * ncol + c] - grid[adr + rm * ncol + c]
                    ) * size_z
                    var wx = dx * Float64(cp - cm)
                    var wy = dy * Float64(rp - rm)
                    # The surface is z = f(x, y), so the (unnormalised) normal
                    # is (-df/dx, -df/dy, 1). `wx`/`wy` are never 0 because
                    # nrow and ncol are both at least 2.
                    var n = Vec3(-zdx / wx, -zdy / wy, 1.0).normalized()
                    self.hfield_cache[idx].verts[v] = GPUVertex(
                        px=Float32(hfield_node_x(c, ncol, size_x)),
                        py=Float32(hfield_node_y(r, nrow, size_y)),
                        pz=Float32(z),
                        nx=Float32(n.x),
                        ny=Float32(n.y),
                        nz=Float32(n.z),
                        u=Float32(Float64(c) / Float64(ncol - 1)),
                        v=Float32(Float64(r) / Float64(nrow - 1)),
                    )

            var vb_size = UInt32(self.hfield_cache[idx].vbuf_bytes)
            var mapped = map_gpu_transfer_buffer(
                self.device.value(), self.hfield_cache[idx].transfer, True
            )
            unsafe_memcpy(
                dest=mapped.unsafe_bitcast[UInt8](),
                src=Pointer(
                    to=self.hfield_cache[idx].verts[0]
                ).unsafe_bitcast[UInt8](),
                count=Int(vb_size),
            )
            unmap_gpu_transfer_buffer(
                self.device.value(), self.hfield_cache[idx].transfer
            )

            var mi = self.hfield_cache[idx].mesh_idx
            var cmd_buf = acquire_gpu_command_buffer(self.device.value())
            var copy_pass = begin_gpu_copy_pass(cmd_buf)
            var src = GPUTransferBufferLocation(
                transfer_buffer=untracked(self.hfield_cache[idx].transfer),
                offset=0,
            )
            var dst = GPUBufferRegion(
                buffer=untracked(self.mesh_cache[mi].mesh.vertex_buffer),
                offset=0,
                size=vb_size,
            )
            upload_to_gpu_buffer(copy_pass, Ptr(to=src), Ptr(to=dst), True)
            end_gpu_copy_pass(copy_pass)
            submit_gpu_command_buffer(cmd_buf)

        var uniforms = ObjectUniforms()
        uniforms.model = mat4_to_gpu_f32(Mat4.from_quat(orientation, center))
        uniforms.color = color_to_vec4(color)
        uniforms.material[0] = shininess
        uniforms.material[1] = specular
        uniforms.material[2] = reflectance
        uniforms.material[3] = emission

        self.solid_draws.append(
            SolidDrawCommand(
                0,
                uniforms,
                is_mesh=True,
                mesh_cache_idx=self.hfield_cache[idx].mesh_idx,
                texture_cache_idx=-1,
            )
        )

    def draw_skin(
        mut self,
        name: String,
        skn_path: String,
        body_names: List[String],
        xpos: List[Float32],
        xquat: List[Float32],
        color: Color = Color(255, 255, 255, 255),
        shininess: Float32 = 0.4,
        specular: Float32 = 0.3,
        texture_name: String = String(""),
        texture_path: String = String(""),
    ) raises:
        """Deform a MuJoCo `.skn` by the current body poses and queue it.

        Args:
            name: Cache key.
            skn_path: Path to the binary `.skn`.
            body_names: Model body names, index-aligned with `xpos`/`xquat`.
            xpos: World body positions, 3 per body.
            xquat: World body orientations, 4 per body, (w, x, y, z).
            color: Modulates the texture.
            shininess: Specular exponent scaling (0-1).
            specular: Specular intensity (0-1).
            texture_name: Cache key for the texture (empty = untextured).
            texture_path: Path to the PNG (empty = untextured).

        ⚠ THE VERTICES GO UP IN WORLD SPACE, so the draw carries an IDENTITY
        model matrix. Skinning has already applied each bone's transform; a
        second one on the GPU would apply the torso's motion twice.

        The first call loads, resolves bones and allocates; every call after it
        deforms and re-uploads. Failure to LOAD is reported and then swallowed,
        matching `draw_mesh` — a missing asset should cost you the skin, not
        the window.
        """
        var idx = -1
        for i in range(len(self.skin_cache)):
            if self.skin_cache[i].name == name:
                idx = i
                break

        if idx < 0:
            idx = self._load_skin(name, skn_path, body_names)
            if idx < 0:
                return

        # ── deform ───────────────────────────────────────────────────────
        skin_pose(
            self.skin_cache[idx].skin,
            self.skin_cache[idx].bone_body,
            xpos,
            xquat,
            self.skin_cache[idx].posed,
            self.skin_cache[idx].normals,
        )

        var nv = self.skin_cache[idx].skin.nvert
        var textured = self.skin_cache[idx].skin.has_texcoords()
        for v in range(nv):
            var u = Float32(0)
            var tv = Float32(0)
            if textured:
                u = self.skin_cache[idx].skin.texcoord[2 * v]
                tv = self.skin_cache[idx].skin.texcoord[2 * v + 1]
            self.skin_cache[idx].verts[v] = GPUVertex(
                px=self.skin_cache[idx].posed[3 * v],
                py=self.skin_cache[idx].posed[3 * v + 1],
                pz=self.skin_cache[idx].posed[3 * v + 2],
                nx=self.skin_cache[idx].normals[3 * v],
                ny=self.skin_cache[idx].normals[3 * v + 1],
                nz=self.skin_cache[idx].normals[3 * v + 2],
                u=u,
                v=tv,
            )

        # ── upload into the mesh's existing vertex buffer ─────────────────
        var vb_size = UInt32(self.skin_cache[idx].vbuf_bytes)
        var mapped = map_gpu_transfer_buffer(
            self.device.value(), self.skin_cache[idx].transfer, True
        )
        unsafe_memcpy(
            dest=mapped.unsafe_bitcast[UInt8](),
            src=Pointer(to=self.skin_cache[idx].verts[0]).unsafe_bitcast[
                UInt8
            ](),
            count=Int(vb_size),
        )
        unmap_gpu_transfer_buffer(
            self.device.value(), self.skin_cache[idx].transfer
        )

        var mi = self.skin_cache[idx].mesh_idx
        var cmd_buf = acquire_gpu_command_buffer(self.device.value())
        var copy_pass = begin_gpu_copy_pass(cmd_buf)
        var src = GPUTransferBufferLocation(
            transfer_buffer=untracked(self.skin_cache[idx].transfer), offset=0
        )
        var dst = GPUBufferRegion(
            buffer=untracked(self.mesh_cache[mi].mesh.vertex_buffer),
            offset=0,
            size=vb_size,
        )
        # `cycle=True`: the GPU may still be reading last frame's copy of this
        # buffer, and this is a per-frame overwrite of the whole thing.
        upload_to_gpu_buffer(copy_pass, Ptr(to=src), Ptr(to=dst), True)
        end_gpu_copy_pass(copy_pass)
        submit_gpu_command_buffer(cmd_buf)

        # ── texture ──────────────────────────────────────────────────────
        var tex_idx = -1
        if texture_name.byte_length() > 0 and texture_path.byte_length() > 0:
            for ti in range(len(self.texture_cache)):
                if self.texture_cache[ti].matches(texture_name):
                    tex_idx = ti
                    break
            if tex_idx < 0:
                try:
                    var tex_data = load_png(texture_path)
                    tex_idx = self.upload_texture(texture_name, tex_data)
                    print(
                        "Loaded skin texture '", texture_name, "':",
                        tex_data.width, "x", tex_data.height,
                    )
                except e:
                    print("Warning: skin texture load failed:", String(e))

        var uniforms = ObjectUniforms()
        uniforms.model = make_identity_f32()
        uniforms.color = color_to_vec4(color)
        uniforms.material[0] = shininess
        uniforms.material[1] = specular
        uniforms.material[2] = Float32(1.0) if tex_idx >= 0 else Float32(0.0)
        uniforms.material[3] = Float32(0.0)

        self.solid_draws.append(
            SolidDrawCommand(
                0,
                uniforms,
                is_mesh=True,
                mesh_cache_idx=mi,
                texture_cache_idx=tex_idx,
            )
        )

    def _load_skin(
        mut self, name: String, skn_path: String, body_names: List[String]
    ) raises -> Int:
        """Load a `.skn`, bind its bones and allocate its GPU home. -1 on
        failure (already reported)."""
        var skin: SkinData
        try:
            skin = load_skn(skn_path)
        except e:
            print("Warning: skin load failed:", String(e))
            return -1

        # ⚠ 16-BIT INDICES. `MeshData` and the mesh draw path are UInt16
        # throughout, so a skin past 65535 vertices would WRAP silently and
        # scramble the topology. dog is 24065; refuse anything that would not
        # fit rather than render a knot.
        if skin.nvert > 65535:
            print(
                "Warning: skin '", name, "' has", skin.nvert,
                "vertices; the mesh path is 16-bit indexed (max 65535)",
            )
            return -1

        var bone_body = resolve_skin_bones(skin, body_names)
        var n_unbound = 0
        for b in range(len(bone_body)):
            if bone_body[b] < 0:
                n_unbound += 1
                print(
                    "Warning: skin bone '", skin.bones[b].body_name,
                    "' matches no body — that region will collapse",
                )

        # Seed the buffers from the REST mesh. The first frame overwrites the
        # vertices anyway; this exists so the index buffer and the allocation
        # come from the one code path that already works.
        var seed = MeshData()
        seed.vertices.reserve(skin.nvert)
        seed.indices.reserve(3 * skin.nface)
        var textured = skin.has_texcoords()
        for v in range(skin.nvert):
            seed.vertices.append(
                GPUVertex(
                    px=skin.vert[3 * v],
                    py=skin.vert[3 * v + 1],
                    pz=skin.vert[3 * v + 2],
                    nz=Float32(1.0),
                    u=skin.texcoord[2 * v] if textured else Float32(0),
                    v=skin.texcoord[2 * v + 1] if textured else Float32(0),
                )
            )
        for i in range(3 * skin.nface):
            seed.indices.append(UInt32(Int(skin.face[i])))

        var vb_bytes = seed.vertex_byte_size()
        var handle = self._upload_mesh(seed)
        self.mesh_cache.append(MeshCacheEntry(name, handle^))
        var mesh_idx = len(self.mesh_cache) - 1

        var tb_info = GPUTransferBufferCreateInfo(
            usage=GPUTransferBufferUsage.GPU_TRANSFERBUFFERUSAGE_UPLOAD,
            size=UInt32(vb_bytes),
            props=PropertiesID(0),
        )
        var transfer = untracked(
            create_gpu_transfer_buffer(self.device.value(), Ptr(to=tb_info))
        )

        var nv = skin.nvert
        var entry = SkinCacheEntry(
            name, skin^, bone_body^, n_unbound, mesh_idx, transfer, vb_bytes
        )
        for _ in range(nv):
            entry.verts.append(GPUVertex(px=0, py=0, pz=0))
        self.skin_cache.append(entry^)

        print(
            "Loaded skin '", name, "':", nv, "verts,",
            len(self.skin_cache[len(self.skin_cache) - 1].skin.bones),
            "bones,", n_unbound, "unbound",
        )
        return len(self.skin_cache) - 1


    def draw_box(
        mut self,
        center: Vec3,
        orientation: Quat,
        half_extents: Vec3,
        color: Color = Color(255, 255, 255, 255),
        shininess: Float32 = 0.5,
        specular: Float32 = 0.5,
        reflectance: Float32 = 0.0,
        emission: Float32 = 0.0,
        texture_name: String = String(""),
        texture_path: String = String(""),
        texture_colorspace: Int = 0,
        texrepeat_u: Float32 = 1.0,
        texrepeat_v: Float32 = 1.0,
        tex_cube: Bool = False,
        tex_scale: Vec3 = Vec3(1.0, 1.0, 1.0),
    ) raises:
        """Draw a solid box, optionally textured.

        Args:
            center: Box center in world space.
            orientation: Box orientation.
            half_extents: Half-extents along local X, Y, Z.
            color: Surface color.
            shininess: Specular exponent scaling (0-1).
            specular: Specular intensity (0-1).
            reflectance: Reflectance coefficient (0-1).
            emission: Emissive intensity (0-1).
            texture_name: Cache key for the texture (empty = no texture).
            texture_path: Path to the PNG texture file (empty = no texture).
            texture_colorspace: `<texture colorspace>` (`COLORSPACE_AUTO` reads the PNG).
            texrepeat_u: Texture repeat in U direction.
            texrepeat_v: Texture repeat in V direction.
            tex_cube: Sample as a cube/box texture from the object-space position.
            tex_scale: Object-space position scale for the cube lookup.
        """
        # Load and cache texture if provided
        var tex_idx = -1
        if texture_name.byte_length() > 0 and texture_path.byte_length() > 0:
            for ti in range(len(self.texture_cache)):
                if self.texture_cache[ti].matches(texture_name):
                    tex_idx = ti
                    break
            if tex_idx < 0:
                try:
                    var tex_data = load_texture_png(texture_path, texture_colorspace)
                    tex_idx = self.upload_texture(texture_name, tex_data)
                    print(
                        "Loaded texture '",
                        texture_name,
                        "':",
                        tex_data.width,
                        "x",
                        tex_data.height,
                    )
                except e:
                    print("Warning: texture load failed:", String(e))
                    pass

        # Unit box is [-0.5, 0.5], so scale by 2 * half_extents
        var scale = Vec3(
            half_extents.x * 2.0,
            half_extents.y * 2.0,
            half_extents.z * 2.0,
        )
        var model = Mat4.compose(center, orientation, scale)

        var uniforms = ObjectUniforms()
        uniforms.model = mat4_to_gpu_f32(model)
        uniforms.color = color_to_vec4(color)
        uniforms.material[0] = shininess
        uniforms.material[1] = specular
        # material.z > 0 tells the shader to sample the texture
        uniforms.material[2] = Float32(1.0) if tex_idx >= 0 else reflectance
        uniforms.material[3] = emission
        uniforms.tex_params[0] = texrepeat_u
        uniforms.tex_params[1] = texrepeat_v
        uniforms.tex_params[2] = Float32(1.0) if tex_cube else Float32(0.0)
        uniforms.tex_scale[0] = Float32(tex_scale.x)
        uniforms.tex_scale[1] = Float32(tex_scale.y)
        uniforms.tex_scale[2] = Float32(tex_scale.z)

        self.solid_draws.append(
            SolidDrawCommand(1, uniforms, texture_cache_idx=tex_idx)
        )

    def set_headlight(
        mut self,
        active: Bool,
        ambient_r: Float32, ambient_g: Float32, ambient_b: Float32,
        diffuse_r: Float32, diffuse_g: Float32, diffuse_b: Float32,
        specular_r: Float32, specular_g: Float32, specular_b: Float32,
    ):
        """MuJoCo's `<visual><headlight>`: a directional light at the eye,
        along the view. Defaults are MuJoCo's (`mjv_defaultVisual`)."""
        self.headlight_active = active
        self.headlight_ambient[0] = ambient_r
        self.headlight_ambient[1] = ambient_g
        self.headlight_ambient[2] = ambient_b
        self.headlight_diffuse[0] = diffuse_r
        self.headlight_diffuse[1] = diffuse_g
        self.headlight_diffuse[2] = diffuse_b
        self.headlight_specular[0] = specular_r
        self.headlight_specular[1] = specular_g
        self.headlight_specular[2] = specular_b

    def set_skybox(
        mut self,
        top_r: Float32 = 0.8,
        top_g: Float32 = 0.85,
        top_b: Float32 = 0.95,
        bottom_r: Float32 = 0.3,
        bottom_g: Float32 = 0.35,
        bottom_b: Float32 = 0.5,
    ):
        """Enable skybox gradient background.

        Args:
            top_r: Gradient top color red (0-1).
            top_g: Gradient top color green (0-1).
            top_b: Gradient top color blue (0-1).
            bottom_r: Gradient bottom color red (0-1).
            bottom_g: Gradient bottom color green (0-1).
            bottom_b: Gradient bottom color blue (0-1).
        """
        self.draw_skybox = True
        self.skybox_uniforms.top_color[0] = top_r
        self.skybox_uniforms.top_color[1] = top_g
        self.skybox_uniforms.top_color[2] = top_b
        self.skybox_uniforms.top_color[3] = 1.0
        self.skybox_uniforms.bottom_color[0] = bottom_r
        self.skybox_uniforms.bottom_color[1] = bottom_g
        self.skybox_uniforms.bottom_color[2] = bottom_b
        self.skybox_uniforms.bottom_color[3] = 1.0

    def set_skybox_stars(
        mut self,
        r: Float32,
        g: Float32,
        b: Float32,
        density: Float32,
    ):
        """Turn on the procedural starfield (MuJoCo `mark="random"`).

        `density` is MuJoCo's `random` attribute — the fraction of texture
        pixels it would have marked, default .01 — reused here as the fraction
        of direction cells holding a star. Passing 0 disables it, which is what
        a model without `mark="random"` gets.
        """
        self.skybox_uniforms.mark_color[0] = r
        self.skybox_uniforms.mark_color[1] = g
        self.skybox_uniforms.mark_color[2] = b
        self.skybox_uniforms.mark_color[3] = density

    def set_ground_checker_colors(
        mut self,
        r: Float32 = 0.22,
        g: Float32 = 0.22,
        b: Float32 = 0.25,
    ):
        """Set ground checker light tile color (stored in scene ground_params.xyz).

        Dark tile is always black (0, 0, 0), matching MuJoCo's rgb1=black convention.

        Args:
            r: Checker light tile color red (0-1). Dark tile is black.
            g: Checker light tile color green (0-1). Dark tile is black.
            b: Checker light tile color blue (0-1). Dark tile is black.
        """
        self.scene_uniforms.ground_params[0] = r
        self.scene_uniforms.ground_params[1] = g
        self.scene_uniforms.ground_params[2] = b

    def set_ground_solid_color(
        mut self,
        r: Float32 = 0.5,
        g: Float32 = 0.5,
        b: Float32 = 0.5,
    ):
        """Set ground to solid (non-checker) color.

        Used when the plane geom has no associated checker texture.
        Encoded as negative values in ground_params.xyz (shader checks sign).

        Args:
            r: Solid color red (0-1).
            g: Solid color green (0-1).
            b: Solid color blue (0-1).
        """
        self.scene_uniforms.ground_params[0] = -r
        self.scene_uniforms.ground_params[1] = -g
        self.scene_uniforms.ground_params[2] = -b

    def draw_ground_grid(
        mut self,
        center_x: Float64 = 0.0,
        size: Float64 = 10.0,
        height: Float64 = 0.0,
        texture_name: String = String(""),
        texture_path: String = String(""),
        texture_colorspace: Int = 0,
        texrepeat_u: Float64 = 1.0,
        texrepeat_v: Float64 = 1.0,
        texuniform: Bool = False,
        plane_half_x: Float64 = 0.0,
        plane_half_y: Float64 = 0.0,
        shininess: Float32 = 0.5,
        specular: Float32 = 0.5,
        reflectance: Float32 = 0.2,
    ) raises:
        """Draw the ground plane with procedural checkerboard or texture.

        Args:
            center_x: X-coordinate to center the ground on (for scrolling envs).
            size: Unused (ground mesh is pre-sized).
            height: Z-coordinate of the ground plane.
            texture_name: Cache key for the ground texture (empty = checker/solid).
            texture_path: Path to the PNG texture file (empty = checker/solid).
            texture_colorspace: `<texture colorspace>` (`COLORSPACE_AUTO` reads the PNG).
            texrepeat_u: Texture repeat in U direction.
            texrepeat_v: Texture repeat in V direction.
            texuniform: `<material texuniform>` — repeats per spatial unit.
            plane_half_x: The plane geom's `size[0]` (0 = infinite).
            plane_half_y: The plane geom's `size[1]` (0 = infinite).
            shininess: The plane material's shininess (0-1).
            specular: The plane material's specular (0-1).
            reflectance: The plane material's reflectance; 0 = no mirror.
                The 0.2 default is the value the pass was hard-coded to.

        ⚠ THE REPEAT IS CONVERTED TO TILES PER METRE HERE, and the shader
        tiles in WORLD space. `render_gl3.c:settexture` scales a plane's
        [0,1] texture coordinates by `texrepeat`, times `size` when
        `texuniform`, over a plane `2*size` across — so a tile is
        `2*size/texrepeat` m, or `2/texrepeat` m uniform. It used to scale the
        24 m ground mesh's own UVs by `texrepeat`, which for LIBERO's
        `texrepeat="3 3" texuniform="true"` floor made each tile 8 m instead
        of 0.67 m: one stretched plank across the whole room. An INFINITE
        plane (size 0) keeps the old mesh-relative reading, which is the only
        extent it has.
        """
        # Load and cache ground texture if provided
        self.ground_texture_idx = -1
        if texture_name.byte_length() > 0 and texture_path.byte_length() > 0:
            for ti in range(len(self.texture_cache)):
                if self.texture_cache[ti].matches(texture_name):
                    self.ground_texture_idx = ti
                    break
            if self.ground_texture_idx < 0:
                try:
                    var tex_data = load_texture_png(texture_path, texture_colorspace)
                    self.ground_texture_idx = self.upload_texture(
                        texture_name, tex_data
                    )
                    print(
                        "Loaded ground texture '",
                        texture_name,
                        "':",
                        tex_data.width,
                        "x",
                        tex_data.height,
                    )
                except e:
                    print("Warning: ground texture load failed:", String(e))
                    pass

        if self.ground_texture_idx >= 0:
            # Signal texture mode: ground_params.z > 1.5 (colors are always 0-1)
            # xy = tiles PER METRE. Note: ground_params.w is reserved for ground_z
            var rep_u = texrepeat_u if texrepeat_u > 0.0 else 1.0
            var rep_v = texrepeat_v if texrepeat_v > 0.0 else 1.0
            var per_m_u = rep_u / 24.0
            var per_m_v = rep_v / 24.0
            if texuniform:
                per_m_u = rep_u / 2.0
                per_m_v = rep_v / 2.0
            else:
                if plane_half_x > 0.0:
                    per_m_u = rep_u / (2.0 * plane_half_x)
                if plane_half_y > 0.0:
                    per_m_v = rep_v / (2.0 * plane_half_y)
            self.scene_uniforms.ground_params[0] = Float32(per_m_u)
            self.scene_uniforms.ground_params[1] = Float32(per_m_v)
            self.scene_uniforms.ground_params[2] = Float32(2.0)

        var model = Mat4.from_translation(Vec3(center_x, 0.0, height))
        self.ground_uniforms = ObjectUniforms()
        self.ground_uniforms.model = mat4_to_gpu_f32(model)
        self.ground_uniforms.color = color_to_vec4(255, 255, 255)
        self.ground_uniforms.material[0] = shininess
        self.ground_uniforms.material[1] = specular
        self.has_ground = True
        self.ground_z = height
        self.ground_reflectance = reflectance

    def draw_coordinate_axes(
        mut self,
        origin: Vec3 = Vec3.zero(),
        length: Float64 = 1.0,
    ):
        """Draw coordinate axes: X=red, Y=green, Z=blue.

        Args:
            origin: Origin point.
            length: Length of each axis.
        """
        # X axis - red
        self._add_line(
            origin,
            origin + Vec3(length, 0.0, 0.0),
            color_to_vec4(255, 50, 50),
        )
        # Y axis - green
        self._add_line(
            origin,
            origin + Vec3(0.0, length, 0.0),
            color_to_vec4(50, 255, 50),
        )
        # Z axis - blue
        self._add_line(
            origin,
            origin + Vec3(0.0, 0.0, length),
            color_to_vec4(80, 80, 255),
        )

    def draw_line_3d(
        mut self,
        start: Vec3,
        end: Vec3,
        color: Color,
    ):
        """Draw a 3D line segment.

        Args:
            start: Start point in world space.
            end: End point in world space.
            color: Line color.
        """
        self._add_line(
            start,
            end,
            color_to_vec4(color),
        )

    def _add_line(
        mut self,
        start: Vec3,
        end: Vec3,
        color: Array[Float32, 4],
    ):
        """Add a line segment to the line accumulator."""
        if len(self.line_vertex_data) + 6 > MAX_LINE_VERTICES * 3:
            return  # Buffer full

        self.line_vertex_data.append(Float32(start.x))
        self.line_vertex_data.append(Float32(start.y))
        self.line_vertex_data.append(Float32(start.z))

        self.line_vertex_data.append(Float32(end.x))
        self.line_vertex_data.append(Float32(end.y))
        self.line_vertex_data.append(Float32(end.z))

        self.line_colors.append(LineColorEntry(color))

    def render_scene(mut self) raises:
        """Render default scene elements (grid and axes)."""
        if self.draw_grid:
            self.draw_ground_grid()

        if self.draw_axes:
            self.draw_coordinate_axes()

    def _select_and_draw(
        self,
        render_pass: Ptr[GPURenderPass, MutAnyOrigin],
        draw: SolidDrawCommand,
    ) raises:
        """Select mesh buffers for a draw command, bind, and draw."""
        var vb: Ptr[GPUBuffer, MutUntrackedOrigin]
        var ib: Ptr[GPUBuffer, MutUntrackedOrigin]
        var n_idx: UInt32

        if draw.is_capsule:
            var ci = draw.capsule_cache_idx
            vb = self.capsule_cache[ci].mesh.vertex_buffer
            ib = self.capsule_cache[ci].mesh.index_buffer
            n_idx = self.capsule_cache[ci].mesh.num_indices
        elif draw.is_cylinder:
            var ci = draw.cylinder_cache_idx
            vb = self.cylinder_cache[ci].mesh.vertex_buffer
            ib = self.cylinder_cache[ci].mesh.index_buffer
            n_idx = self.cylinder_cache[ci].mesh.num_indices
        elif draw.is_mesh:
            var mi = draw.mesh_cache_idx
            vb = self.mesh_cache[mi].mesh.vertex_buffer
            ib = self.mesh_cache[mi].mesh.index_buffer
            n_idx = self.mesh_cache[mi].mesh.num_indices
        elif draw.mesh_idx == 0:
            vb = self.sphere_mesh.value().vertex_buffer
            ib = self.sphere_mesh.value().index_buffer
            n_idx = self.sphere_mesh.value().num_indices
        else:
            vb = self.box_mesh.value().vertex_buffer
            ib = self.box_mesh.value().index_buffer
            n_idx = self.box_mesh.value().num_indices

        var vb_binding = GPUBufferBinding(buffer=untracked(vb), offset=0)
        bind_gpu_vertex_buffers(render_pass, 0, Ptr(to=vb_binding), 1)
        var ib_binding = GPUBufferBinding(buffer=untracked(ib), offset=0)
        bind_gpu_index_buffer(
            render_pass,
            Ptr(to=ib_binding),
            GPUIndexElementSize.GPU_INDEXELEMENTSIZE_32BIT,
        )
        draw_gpu_indexed_primitives(render_pass, n_idx, 1, 0, 0, 0)


    # ── the scene phases, once for the main view and once per inset ──────
    #
    # ⚠ ONE BODY EACH, called from the main pass AND from `_draw_pip_views`.
    # The picture-in-picture views replay exactly these draws from another
    # camera; a phase written twice would let an inset drift from the main
    # view (a material, a texture slot) with nothing to catch it.

    def _draw_skybox_phase(
        mut self,
        render_pass: Ptr[GPURenderPass, MutAnyOrigin],
        cmd_buf: Ptr[GPUCommandBuffer, MutAnyOrigin],
    ) raises:
        bind_gpu_graphics_pipeline(render_pass, self.skybox_pipeline.value())
        # ⚠ SIZED FROM THE STRUCT, NOT BY HAND. This was a literal 32 —
        # correct for two float4s — and adding the starfield's mark colour
        # and camera basis made it 96. Nothing warns about the mismatch:
        # the shader simply reads whatever follows the 32 bytes it was
        # given, so the stars were computed from uninitialised memory and
        # never appeared. The other push sites in this file are still
        # hand-sized literals and carry the same hazard.
        push_gpu_fragment_uniform_data(
            cmd_buf,
            0,
            Ptr(to=self.skybox_uniforms).unsafe_bitcast[NoneType](),
            UInt32(size_of[SkyboxUniforms]()),
        )
        # Draw fullscreen triangle (3 vertices, no vertex buffer)
        draw_gpu_primitives(render_pass, 3, 1, 0, 0)

    def _draw_ground_phase(
        mut self,
        render_pass: Ptr[GPURenderPass, MutAnyOrigin],
        cmd_buf: Ptr[GPUCommandBuffer, MutAnyOrigin],
        shadow_binding: GPUTextureSamplerBinding,
    ) raises:
        bind_gpu_graphics_pipeline(render_pass, self.ground_pipeline.value())

        push_gpu_vertex_uniform_data(
            cmd_buf,
            0,
            Ptr(to=self.scene_uniforms).unsafe_bitcast[NoneType](),
            SCENE_UNIFORMS_BYTES,
        )
        push_gpu_fragment_uniform_data(
            cmd_buf,
            0,
            Ptr(to=self.scene_uniforms).unsafe_bitcast[NoneType](),
            SCENE_UNIFORMS_BYTES,
        )
        # Push shadow uniforms to fragment slot 1
        push_gpu_fragment_uniform_data(
            cmd_buf,
            1,
            Ptr(to=self.shadow_uniforms).unsafe_bitcast[NoneType](),
            80,
        )
        push_gpu_vertex_uniform_data(
            cmd_buf,
            1,
            Ptr(to=self.ground_uniforms).unsafe_bitcast[NoneType](),
            OBJECT_UNIFORMS_BYTES,
        )

        # Bind shadow map + sampler to fragment sampler slot 0
        bind_gpu_fragment_samplers(
            render_pass, 0, Ptr(to=shadow_binding), 1
        )

        # Bind ground texture at fragment sampler slot 1
        if self.ground_texture_idx >= 0:
            var gti = self.ground_texture_idx
            var gt_binding = GPUTextureSamplerBinding(
                texture=untracked(self.texture_cache[gti].texture),
                sampler=self.texture_cache[gti].sampler,
            )
            bind_gpu_fragment_samplers(
                render_pass, 1, Ptr(to=gt_binding), 1
            )
        else:
            var gt_def_binding = GPUTextureSamplerBinding(
                texture=untracked(self.default_texture.value()),
                sampler=self.default_tex_sampler.value(),
            )
            bind_gpu_fragment_samplers(
                render_pass, 1, Ptr(to=gt_def_binding), 1
            )

        var gvb = GPUBufferBinding(
            buffer=untracked(self.ground_mesh.value().vertex_buffer), offset=0
        )
        bind_gpu_vertex_buffers(render_pass, 0, Ptr(to=gvb), 1)

        var gib = GPUBufferBinding(
            buffer=untracked(self.ground_mesh.value().index_buffer), offset=0
        )
        bind_gpu_index_buffer(
            render_pass,
            Ptr(to=gib),
            GPUIndexElementSize.GPU_INDEXELEMENTSIZE_32BIT,
        )

        draw_gpu_indexed_primitives(
            render_pass,
            self.ground_mesh.value().num_indices,
            1,
            0,
            0,
            0,
        )

    def _draw_solids_phase(
        mut self,
        render_pass: Ptr[GPURenderPass, MutAnyOrigin],
        cmd_buf: Ptr[GPUCommandBuffer, MutAnyOrigin],
        shadow_binding: GPUTextureSamplerBinding,
    ) raises:
        bind_gpu_graphics_pipeline(render_pass, self.solid_pipeline.value())

        push_gpu_vertex_uniform_data(
            cmd_buf,
            0,
            Ptr(to=self.scene_uniforms).unsafe_bitcast[NoneType](),
            SCENE_UNIFORMS_BYTES,
        )
        push_gpu_fragment_uniform_data(
            cmd_buf,
            0,
            Ptr(to=self.scene_uniforms).unsafe_bitcast[NoneType](),
            SCENE_UNIFORMS_BYTES,
        )
        # Push shadow uniforms to fragment slot 1
        push_gpu_fragment_uniform_data(
            cmd_buf,
            1,
            Ptr(to=self.shadow_uniforms).unsafe_bitcast[NoneType](),
            80,
        )

        # Bind shadow map + sampler
        bind_gpu_fragment_samplers(
            render_pass, 0, Ptr(to=shadow_binding), 1
        )

        for i in range(len(self.solid_draws)):
            push_gpu_vertex_uniform_data(
                cmd_buf,
                1,
                Ptr(to=self.solid_draws[i].uniforms).unsafe_bitcast[NoneType](),
                OBJECT_UNIFORMS_BYTES,
            )
            # Bind texture at fragment sampler slot 1
            var ti = self.solid_draws[i].texture_cache_idx
            if ti >= 0:
                var tex_binding = GPUTextureSamplerBinding(
                    texture=untracked(self.texture_cache[ti].texture),
                    sampler=self.texture_cache[ti].sampler,
                )
                bind_gpu_fragment_samplers(
                    render_pass, 1, Ptr(to=tex_binding), 1
                )
            else:
                var def_binding = GPUTextureSamplerBinding(
                    texture=untracked(self.default_texture.value()),
                    sampler=self.default_tex_sampler.value(),
                )
                bind_gpu_fragment_samplers(
                    render_pass, 1, Ptr(to=def_binding), 1
                )
            self._select_and_draw(render_pass, self.solid_draws[i])

    def _draw_pip_views(
        mut self,
        cmd_buf: Ptr[GPUCommandBuffer, MutAnyOrigin],
        swapchain_tex: Ptr[GPUTexture, MutAnyOrigin],
        sc_w: UInt32,
        sc_h: UInt32,
        shadow_binding: GPUTextureSamplerBinding,
    ) raises:
        """The picture-in-picture insets — see `PipView`. Runs AFTER the main
        pass has ended and before the ImGui pass; restores the main camera's
        scene uniforms when done."""
        if len(self.pip_views) == 0:
            return
        var saved = self.camera.copy()
        for v in range(len(self.pip_views)):
            ref pv = self.pip_views[v]
            if pv.w < 2 or pv.h < 2:
                continue
            var color_info = GPUColorTargetInfo(
                texture=untracked(swapchain_tex),
                mip_level=0,
                layer_or_depth_plane=0,
                clear_color=FColor(0.0, 0.0, 0.0, 1.0),
                load_op=GPULoadOp.GPU_LOADOP_LOAD,
                store_op=GPUStoreOp.GPU_STOREOP_STORE,
                resolve_texture=_null_ptr[GPUTexture, MutUntrackedOrigin](),
                resolve_mip_level=0,
                resolve_layer=0,
                cycle=False,
                cycle_resolve_texture=False,
                padding1=0,
                padding2=0,
            )
            var depth_info = GPUDepthStencilTargetInfo(
                texture=untracked(self.depth_texture.value()),
                clear_depth=1.0,
                load_op=GPULoadOp.GPU_LOADOP_CLEAR,
                store_op=GPUStoreOp.GPU_STOREOP_DONT_CARE,
                stencil_load_op=GPULoadOp.GPU_LOADOP_DONT_CARE,
                stencil_store_op=GPUStoreOp.GPU_STOREOP_DONT_CARE,
                cycle=True,
                clear_stencil=0,
                padding1=0,
                padding2=0,
            )
            var pass_ = begin_gpu_render_pass(
                cmd_buf, Ptr(to=color_info), 1, Ptr(to=depth_info)
            )
            var vp = GPUViewport(
                x=c_float(pv.x), y=c_float(pv.y),
                w=c_float(pv.w), h=c_float(pv.h),
                min_depth=0.0, max_depth=1.0,
            )
            set_gpu_viewport(pass_, Ptr(to=vp))
            var sc = Rect(x=c_int(pv.x), y=c_int(pv.y), w=c_int(pv.w), h=c_int(pv.h))
            set_gpu_scissor(pass_, Ptr(to=sc))
            # this view's camera: aspect from ITS rectangle
            self.camera = pv.camera.copy()
            self.camera.aspect = Float64(pv.w) / Float64(pv.h)
            self._build_scene_uniforms()
            self.scene_uniforms.ground_params[3] = Float32(self.ground_z)
            # ⚠ THE SKY IS DRAWN UNCONDITIONALLY HERE. The main pass CLEARS to
            # the background colour and draws the skybox only when the model
            # has one; an inset LOADS the colour target, so without its own
            # sky the main view would show through above the horizon — the
            # arm from one camera floating in another's sky. The gradient's
            # uniforms are whatever `set_skybox` (or its defaults) left.
            self._draw_skybox_phase(pass_, cmd_buf)
            if self.has_ground:
                self._draw_ground_phase(pass_, cmd_buf, shadow_binding)
            if len(self.solid_draws) > 0:
                self._draw_solids_phase(pass_, cmd_buf, shadow_binding)
            end_gpu_render_pass(pass_)
        # back to the main camera, so anything reading the uniforms after
        # this (the screenshot crop, next frame's picking) sees the main view
        self.camera = saved.copy()
        self._build_scene_uniforms()
        self.scene_uniforms.ground_params[3] = Float32(self.ground_z)
        _ = sc_w
        _ = sc_h

    def end_frame(mut self) raises:
        """End frame: shadow pass, then main pass with reflections, ground, solids, lines, text.
        """
        # Update ortho projection for current window size
        var ortho = ortho_projection(
            0.0,
            Float64(self.width),
            Float64(self.height),
            0.0,
            0.0,
            1.0,
        )
        self.text_uniforms.ortho_proj = mat4_to_gpu_f32(ortho)

        # Acquire command buffer
        var cmd_buf = acquire_gpu_command_buffer(self.device.value())

        # Upload line data if any
        var num_line_verts = len(self.line_vertex_data) // 3
        if num_line_verts > 0:
            var mapped = map_gpu_transfer_buffer(
                self.device.value(), self.line_transfer_buffer.value(), True
            )
            var mapped_f32 = mapped.unsafe_bitcast[Float32]()
            for i in range(len(self.line_vertex_data)):
                mapped_f32[unsafe_offset=i] = self.line_vertex_data[i]
            unmap_gpu_transfer_buffer(self.device.value(), self.line_transfer_buffer.value())

            var copy_pass = begin_gpu_copy_pass(cmd_buf)
            var src = GPUTransferBufferLocation(
                transfer_buffer=untracked(self.line_transfer_buffer.value()), offset=0
            )
            var dst = GPUBufferRegion(
                buffer=untracked(self.line_vertex_buffer.value()),
                offset=0,
                size=UInt32(len(self.line_vertex_data) * 4),
            )
            # `cycle=True`. SDL_gpu.h: "You must also take care not to
            # overwrite a section of data that has been referenced in a command
            # without cycling first ... overwriting a section of data that has
            # already been referenced will produce unexpected results." This is
            # a WHOLE-BUFFER overwrite, every frame, of a buffer last frame's
            # draw call referenced and the GPU may still be reading. The mesh
            # path at `_upload_deformed_mesh` already learned this; these two
            # per-frame buffers were left behind. Torn vertex data puts a quad
            # at garbage coordinates with a garbage colour — which is what a
            # full- or half-screen colour flash IS.
            upload_to_gpu_buffer(copy_pass, Ptr(to=src), Ptr(to=dst), True)
            end_gpu_copy_pass(copy_pass)

        # Upload text vertex data if any (must be before render pass)
        var num_text_chars = (
            len(self.text_vertex_data) // 32
        )  # 32 floats per quad (4 verts × 8 floats)
        # ⚠ CLAMP, do not trust the producers. The transfer buffer holds
        # exactly MAX_TEXT_CHARS quads; the loop below writes through a raw
        # mapped pointer, so one caller appending past the cap would corrupt
        # memory rather than lose a glyph. `draw_text`/`draw_rect` check their
        # own budget, but this is the write that has to be safe.
        if num_text_chars > MAX_TEXT_CHARS:
            num_text_chars = MAX_TEXT_CHARS
        var n_text_floats = num_text_chars * 32
        if num_text_chars > 0:
            var text_mapped = map_gpu_transfer_buffer(
                self.device.value(), self.text_transfer_buffer.value(), True
            )
            var text_mapped_f32 = text_mapped.unsafe_bitcast[Float32]()
            for i in range(n_text_floats):
                text_mapped_f32[unsafe_offset=i] = self.text_vertex_data[i]
            unmap_gpu_transfer_buffer(self.device.value(), self.text_transfer_buffer.value())

            var text_copy_pass = begin_gpu_copy_pass(cmd_buf)
            var text_src = GPUTransferBufferLocation(
                transfer_buffer=untracked(self.text_transfer_buffer.value()), offset=0
            )
            var text_dst = GPUBufferRegion(
                buffer=untracked(self.text_vertex_buffer.value()),
                offset=0,
                size=UInt32(n_text_floats * 4),
            )
            # `cycle=True` — same hazard as the line buffer above, and worse:
            # the HUD's text CHANGES LENGTH between frames, so a short frame
            # overwrites only part of what a long frame referenced, which is
            # the exact case SDL names as producing unexpected results.
            upload_to_gpu_buffer(
                text_copy_pass, Ptr(to=text_src), Ptr(to=text_dst), True
            )
            end_gpu_copy_pass(text_copy_pass)

        # ====================================================================
        # ImGui geometry upload
        # ====================================================================
        # ⚠ HERE, NOT NEXT TO THE DRAW CALL. `PrepareDrawData` records a COPY
        # pass, and SDL_GPU forbids one while a render pass is open — so it has
        # to precede the shadow pass, not merely the main pass. Placing it
        # before the swapchain acquire has a second benefit: it runs on the
        # early-return path too, so `ImGui::Render()` always closes the frame
        # that `imgui_new_frame` opened and the next frame cannot assert.
        var imgui_pending = self.imgui_frame_open
        if imgui_pending:
            ig_prepare(cmd_buf)
            self.imgui_frame_open = False

        # Build scene uniforms and shadow uniforms
        self._build_scene_uniforms()
        self._build_light_view_proj()

        # Store ground_z in scene_uniforms ground_params.w for reflection clipping
        self.scene_uniforms.ground_params[3] = Float32(self.ground_z)

        # ====================================================================
        # SHADOW PASS (depth-only, from light POV)
        # ====================================================================
        if len(self.solid_draws) > 0:
            var shadow_depth_info = GPUDepthStencilTargetInfo(
                texture=untracked(self.shadow_map.value()),
                clear_depth=1.0,
                load_op=GPULoadOp.GPU_LOADOP_CLEAR,
                store_op=GPUStoreOp.GPU_STOREOP_STORE,
                stencil_load_op=GPULoadOp.GPU_LOADOP_DONT_CARE,
                stencil_store_op=GPUStoreOp.GPU_STOREOP_DONT_CARE,
                cycle=True,
                clear_stencil=0,
                padding1=0,
                padding2=0,
            )

            var shadow_pass = begin_gpu_render_pass(
                cmd_buf,
                _null_ptr[GPUColorTargetInfo, ImmutAnyOrigin](),  # No color targets
                0,
                Ptr(to=shadow_depth_info),
            )

            var shadow_viewport = GPUViewport(
                x=0.0,
                y=0.0,
                w=1024.0,
                h=1024.0,
                min_depth=0.0,
                max_depth=1.0,
            )
            set_gpu_viewport(shadow_pass, Ptr(to=shadow_viewport))

            bind_gpu_graphics_pipeline(shadow_pass, self.shadow_pipeline.value())

            # Push light VP as SceneUniforms (reuse struct layout)
            var light_scene = SceneUniforms()
            light_scene.view_proj = self.shadow_uniforms.light_view_proj.copy()
            push_gpu_vertex_uniform_data(
                cmd_buf,
                0,
                Ptr(to=light_scene).unsafe_bitcast[NoneType](),
                SCENE_UNIFORMS_BYTES,
            )

            for i in range(len(self.solid_draws)):
                push_gpu_vertex_uniform_data(
                    cmd_buf,
                    1,
                    Ptr(to=self.solid_draws[i].uniforms).unsafe_bitcast[NoneType](),
                    OBJECT_UNIFORMS_BYTES,
                )
                self._select_and_draw(shadow_pass, self.solid_draws[i])

            end_gpu_render_pass(shadow_pass)

        # ====================================================================
        # Acquire swapchain texture
        # ====================================================================
        # Mojo nightly: Pointer is non-nullable. _null_ptr returns a
        # zero-address Ptr via the runtime-Int overload of unsafe_from_address.
        # The SDL3 FFI call populates it; we check the raw address afterward.
        var swapchain_tex = _null_ptr[GPUTexture, MutAnyOrigin]()
        var sc_w = UInt32(0)
        var sc_h = UInt32(0)
        wait_and_acquire_gpu_swapchain_texture(
            cmd_buf,
            self.window.value(),
            Ptr(to=swapchain_tex),
            Ptr(to=sc_w),
            Ptr(to=sc_h),
        )

        if Int(swapchain_tex) == 0:
            submit_gpu_command_buffer(cmd_buf)
            return

        # Handle resize
        if Int(sc_w) != self.width or Int(sc_h) != self.height:
            self.width = Int(sc_w)
            self.height = Int(sc_h)
            self.camera.set_screen_size(self.scene_width(), self.height)
            release_gpu_texture(self.device.value(), self.depth_texture.value())
            self._create_depth_texture()

        # ====================================================================
        # MAIN RENDER PASS
        # ====================================================================
        var bg_r = Float32(self.background_color.r) / 255.0
        var bg_g = Float32(self.background_color.g) / 255.0
        var bg_b = Float32(self.background_color.b) / 255.0

        var color_info = GPUColorTargetInfo(
            texture=untracked(swapchain_tex),
            mip_level=0,
            layer_or_depth_plane=0,
            clear_color=FColor(bg_r, bg_g, bg_b, 1.0),
            load_op=GPULoadOp.GPU_LOADOP_CLEAR,
            store_op=GPUStoreOp.GPU_STOREOP_STORE,
            resolve_texture=_null_ptr[GPUTexture, MutUntrackedOrigin](),
            resolve_mip_level=0,
            resolve_layer=0,
            cycle=True,
            cycle_resolve_texture=False,
            padding1=0,
            padding2=0,
        )

        var depth_info = GPUDepthStencilTargetInfo(
            texture=untracked(self.depth_texture.value()),
            clear_depth=1.0,
            load_op=GPULoadOp.GPU_LOADOP_CLEAR,
            store_op=GPUStoreOp.GPU_STOREOP_DONT_CARE,
            stencil_load_op=GPULoadOp.GPU_LOADOP_DONT_CARE,
            stencil_store_op=GPUStoreOp.GPU_STOREOP_DONT_CARE,
            cycle=True,
            clear_stencil=0,
            padding1=0,
            padding2=0,
        )

        var render_pass = begin_gpu_render_pass(
            cmd_buf, Ptr(to=color_info), 1, Ptr(to=depth_info)
        )

        # The 3D phases render into the window MINUS the reserved UI strip.
        # Phase E restores the full window so the HUD and widgets can use it.
        var scene_x = Float32(self.ui_sidebar_width)
        var scene_w = Float32(sc_w) - scene_x
        if scene_w < 1.0:
            scene_w = 1.0
        var viewport = GPUViewport(
            x=c_float(scene_x),
            y=0.0,
            w=c_float(scene_w),
            h=c_float(sc_h),
            min_depth=0.0,
            max_depth=1.0,
        )
        set_gpu_viewport(render_pass, Ptr(to=viewport))

        # Shadow map texture+sampler binding (reused by solid and ground passes)
        var shadow_binding = GPUTextureSamplerBinding(
            texture=untracked(self.shadow_map.value()),
            sampler=self.shadow_sampler.value(),
        )

        # ------------------------------------------------------------------
        # Phase 0: SKYBOX (fullscreen gradient, drawn first)
        # ------------------------------------------------------------------
        if self.draw_skybox:
            self._draw_skybox_phase(render_pass, cmd_buf)

        # ------------------------------------------------------------------
        # Phase B1: GROUND (opaque checkerboard, shadow-mapped)
        # ------------------------------------------------------------------
        if self.has_ground:
            self._draw_ground_phase(render_pass, cmd_buf, shadow_binding)

        # ------------------------------------------------------------------
        # Phase B2: REFLECTIONS (Z-flipped solids, blended ON TOP of the floor)
        #
        # ⚠ AFTER THE GROUND, NOT BEFORE. This ran first and the ground was
        # then drawn semi-transparent over it, which is how the SKYBOX ended up
        # visible through the floor: the floor's missing 45% showed reflections
        # where reflected geometry existed and STARS everywhere else. MuJoCo
        # keeps the floor opaque and blends the mirror term on top at the
        # material's `reflectance`, which is what this order does.
        #
        # Needs the pipeline's depth compare to be ALWAYS (see
        # `_create_pipelines`) — the ground has written depth by now and would
        # otherwise reject every fragment of its own reflection.
        # ------------------------------------------------------------------
        if self.has_ground and self.ground_reflectance > 0.0 and len(self.solid_draws) > 0:
            bind_gpu_graphics_pipeline(render_pass, self.reflection_pipeline.value())

            # Push scene uniforms (fragment slot 0 for reflection clipping)
            push_gpu_vertex_uniform_data(
                cmd_buf,
                0,
                Ptr(to=self.scene_uniforms).unsafe_bitcast[NoneType](),
                SCENE_UNIFORMS_BYTES,
            )
            push_gpu_fragment_uniform_data(
                cmd_buf,
                0,
                Ptr(to=self.scene_uniforms).unsafe_bitcast[NoneType](),
                SCENE_UNIFORMS_BYTES,
            )

            for i in range(len(self.solid_draws)):
                # Build mirrored model matrix: flip Z around ground_z
                var mirrored_uniforms = self.solid_draws[i].uniforms
                # M_reflected = T(0,0,2*gz) * S(1,1,-1) * M
                # This negates row 2 of the model matrix (m20,m21,m22,m23)
                # In column-major storage: m20=[2], m21=[6], m22=[10], m23=[14]
                var gz = Float32(self.ground_z)
                mirrored_uniforms.model[2] = -mirrored_uniforms.model[2]  # m20
                mirrored_uniforms.model[6] = -mirrored_uniforms.model[6]  # m21
                mirrored_uniforms.model[10] = -mirrored_uniforms.model[
                    10
                ]  # m22
                mirrored_uniforms.model[14] = (
                    -mirrored_uniforms.model[14] + 2.0 * gz
                )  # m23

                push_gpu_vertex_uniform_data(
                    cmd_buf,
                    1,
                    Ptr(to=mirrored_uniforms).unsafe_bitcast[NoneType](),
                    OBJECT_UNIFORMS_BYTES,
                )
                self._select_and_draw(render_pass, self.solid_draws[i])

        # ------------------------------------------------------------------
        # Phase C: SOLID OBJECTS (with shadow map sampling)
        # ------------------------------------------------------------------
        if len(self.solid_draws) > 0:
            self._draw_solids_phase(render_pass, cmd_buf, shadow_binding)

        # ------------------------------------------------------------------
        # Phase D: LINES (unchanged)
        # ------------------------------------------------------------------
        if num_line_verts >= 2:
            bind_gpu_graphics_pipeline(render_pass, self.line_pipeline.value())

            var line_offset = 0
            for seg_idx in range(len(self.line_colors)):
                var lu = LineUniforms()
                lu.view_proj = self.scene_uniforms.view_proj.copy()
                lu.color = self.line_colors[seg_idx].to_inline_array()

                push_gpu_vertex_uniform_data(
                    cmd_buf,
                    0,
                    Ptr(to=lu).unsafe_bitcast[NoneType](),
                    80,
                )
                push_gpu_fragment_uniform_data(
                    cmd_buf,
                    0,
                    Ptr(to=lu).unsafe_bitcast[NoneType](),
                    80,
                )

                var lb = GPUBufferBinding(
                    buffer=untracked(self.line_vertex_buffer.value()),
                    offset=UInt32(line_offset * 12),
                )
                bind_gpu_vertex_buffers(render_pass, 0, Ptr(to=lb), 1)

                draw_gpu_primitives(render_pass, 2, 1, 0, 0)

                line_offset += 2

        # ------------------------------------------------------------------
        # Phase E: TEXT HUD OVERLAY (screen-space, alpha-blended, no depth)
        # ------------------------------------------------------------------
        if num_text_chars > 0:
            # ⚠ FULL WINDOW AGAIN. The text ortho projection is built for the
            # whole window (see `ortho_projection` above), so leaving the
            # scene's inset viewport in place would both clip the HUD and shift
            # every glyph right by the sidebar width.
            var full_viewport = GPUViewport(
                x=0.0,
                y=0.0,
                w=c_float(sc_w),
                h=c_float(sc_h),
                min_depth=0.0,
                max_depth=1.0,
            )
            set_gpu_viewport(render_pass, Ptr(to=full_viewport))
            bind_gpu_graphics_pipeline(render_pass, self.text_pipeline.value())
            push_gpu_vertex_uniform_data(
                cmd_buf,
                0,
                Ptr(to=self.text_uniforms).unsafe_bitcast[NoneType](),
                64,
            )
            var atlas_binding = GPUTextureSamplerBinding(
                texture=untracked(self.font_atlas_tex.value()),
                sampler=self.font_sampler.value(),
            )
            bind_gpu_fragment_samplers(render_pass, 0, Ptr(to=atlas_binding), 1)
            var text_vb_binding = GPUBufferBinding(
                buffer=untracked(self.text_vertex_buffer.value()), offset=0
            )
            bind_gpu_vertex_buffers(render_pass, 0, Ptr(to=text_vb_binding), 1)
            var text_ib_binding = GPUBufferBinding(
                buffer=untracked(self.text_index_buffer.value()), offset=0
            )
            bind_gpu_index_buffer(
                render_pass,
                Ptr(to=text_ib_binding),
                GPUIndexElementSize.GPU_INDEXELEMENTSIZE_32BIT,
            )
            draw_gpu_indexed_primitives(
                render_pass, UInt32(num_text_chars * 6), 1, 0, 0, 0
            )

        # End render pass
        end_gpu_render_pass(render_pass)

        # Picture-in-picture insets, each its own pass — see `PipView`.
        self._draw_pip_views(cmd_buf, swapchain_tex, sc_w, sc_h, shadow_binding)

        # ====================================================================
        # IMGUI PASS (color-only, loads the scene, no depth)
        # ====================================================================
        # ⚠ ITS OWN PASS, DELIBERATELY. The obvious placement — alongside the
        # text HUD inside the main pass — makes ImGui's pipeline and that pass
        # disagree about attachments: `ImGui_ImplSDLGPU3_InitInfo` has no
        # depth-stencil field, so the backend builds a color-only pipeline,
        # while the main pass carries a depth attachment. Metal requires the
        # two to match. A second pass costs one begin/end and makes the
        # overlay independent of however the scene pass is configured.
        #
        # LOAD + cycle=False, so the scene underneath survives; CLEAR or
        # cycling would leave the UI floating on an empty background. It runs
        # BEFORE the screenshot/recording downloads below so captures include
        # the UI.
        if imgui_pending:
            var ui_color_info = GPUColorTargetInfo(
                texture=untracked(swapchain_tex),
                mip_level=0,
                layer_or_depth_plane=0,
                clear_color=FColor(0.0, 0.0, 0.0, 1.0),
                load_op=GPULoadOp.GPU_LOADOP_LOAD,
                store_op=GPUStoreOp.GPU_STOREOP_STORE,
                resolve_texture=_null_ptr[GPUTexture, MutUntrackedOrigin](),
                resolve_mip_level=0,
                resolve_layer=0,
                cycle=False,
                cycle_resolve_texture=False,
                padding1=0,
                padding2=0,
            )
            var ui_pass = begin_gpu_render_pass(
                cmd_buf,
                Ptr(to=ui_color_info),
                1,
                _null_ptr[GPUDepthStencilTargetInfo, MutAnyOrigin](),
            )
            ig_render(cmd_buf, ui_pass)
            end_gpu_render_pass(ui_pass)

        # Screenshot capture: append a download copy pass before submitting.
        # The transfer buffer pointer is non-null only if setup succeeded.
        var screenshot_tb: Optional[
            Ptr[GPUTransferBuffer, MutUntrackedOrigin]
        ] = None
        if self.screenshot_requested:
            self.screenshot_requested = False
            try:
                var buf_size = UInt32(self.width * self.height * 4)
                var dl_tb_info = GPUTransferBufferCreateInfo(
                    usage=GPUTransferBufferUsage.GPU_TRANSFERBUFFERUSAGE_DOWNLOAD,
                    size=buf_size,
                    props=PropertiesID(0),
                )
                screenshot_tb = untracked(create_gpu_transfer_buffer(
                    self.device.value(), Ptr(to=dl_tb_info)
                ))
                # Record download into a copy pass in the same command buffer.
                # begin/end_gpu_copy_pass and download_from_gpu_texture only
                # record commands; they will not raise in practice.
                var dl_copy_pass = begin_gpu_copy_pass(cmd_buf)
                var src_region = GPUTextureRegion(
                    texture=untracked(swapchain_tex),
                    mip_level=0,
                    layer=0,
                    x=0,
                    y=0,
                    z=0,
                    w=UInt32(self.width),
                    h=UInt32(self.height),
                    d=1,
                )
                var dst_info = GPUTextureTransferInfo(
                    transfer_buffer=untracked(screenshot_tb.value()),
                    offset=0,
                    pixels_per_row=UInt32(self.width),
                    rows_per_layer=UInt32(self.height),
                )
                download_from_gpu_texture(
                    dl_copy_pass,
                    Ptr(to=src_region),
                    Ptr(to=dst_info),
                )
                end_gpu_copy_pass(dl_copy_pass)
            except e:
                print("Screenshot setup failed: " + String(e))
                if screenshot_tb:
                    release_gpu_transfer_buffer(self.device.value(), screenshot_tb.value())
                    screenshot_tb = None

        # Always submit the command buffer (screenshot download included if set up).
        submit_gpu_command_buffer(cmd_buf)

        # If a download was queued, wait for the GPU to finish, then encode it.
        if screenshot_tb:
            try:
                wait_for_gpu_idle(self.device.value())
                var pixels = map_gpu_transfer_buffer(
                    self.device.value(), screenshot_tb.value(), False
                )
                var filename = (
                    "screenshot_" + String(self.screenshot_counter) + ".jpg"
                )
                self.recorder.save_frame_bgra(
                    Int(pixels), self.width, self.height, filename,
                    self._capture_x(), self._capture_w(),
                )
                unmap_gpu_transfer_buffer(self.device.value(), screenshot_tb.value())
                self.screenshot_counter += 1
            except e:
                print("Screenshot readback failed: " + String(e))
            release_gpu_transfer_buffer(self.device.value(), screenshot_tb.value())

        # Video recording: download this frame and stream it to the encoder
        if self.recorder.is_recording:
            var needed = self.width * self.height * 4
            # Reallocate the persistent transfer buffer if size changed
            if needed != self.recording_tb_size:
                if self.recording_tb:
                    release_gpu_transfer_buffer(self.device.value(), self.recording_tb.value())
                    self.recording_tb = None
                    self.recording_tb_size = 0
                try:
                    var tb_info = GPUTransferBufferCreateInfo(
                        usage=GPUTransferBufferUsage.GPU_TRANSFERBUFFERUSAGE_DOWNLOAD,
                        size=UInt32(needed),
                        props=PropertiesID(0),
                    )
                    self.recording_tb = untracked(create_gpu_transfer_buffer(
                        self.device.value(), Ptr(to=tb_info)
                    ))
                    self.recording_tb_size = needed
                except e:
                    print(
                        "Recording: failed to allocate transfer buffer: "
                        + String(e)
                    )
            # If we have a valid buffer, queue download in a new command buffer
            if self.recording_tb:
                try:
                    var rec_cmd = acquire_gpu_command_buffer(self.device.value())
                    var rec_copy = begin_gpu_copy_pass(rec_cmd)
                    var rec_src = GPUTextureRegion(
                        texture=untracked(swapchain_tex),
                        mip_level=0,
                        layer=0,
                        x=0,
                        y=0,
                        z=0,
                        w=UInt32(self.width),
                        h=UInt32(self.height),
                        d=1,
                    )
                    var rec_dst = GPUTextureTransferInfo(
                        transfer_buffer=untracked(self.recording_tb.value()),
                        offset=0,
                        pixels_per_row=UInt32(self.width),
                        rows_per_layer=UInt32(self.height),
                    )
                    download_from_gpu_texture(
                        rec_copy, Ptr(to=rec_src), Ptr(to=rec_dst)
                    )
                    end_gpu_copy_pass(rec_copy)
                    submit_gpu_command_buffer(rec_cmd)
                    wait_for_gpu_idle(self.device.value())
                    var rec_pixels = map_gpu_transfer_buffer(
                        self.device.value(), self.recording_tb.value(), False
                    )
                    self.recorder.add_frame_bgra(
                        Int(rec_pixels), self.width, self.height,
                        self._capture_x(), self._capture_w(),
                    )
                    unmap_gpu_transfer_buffer(self.device.value(), self.recording_tb.value())
                except e:
                    print("Recording: frame capture failed: " + String(e))

    # --- Recording API ---

    def start_recording(
        mut self, filename: String, fps: Int = 30, skip: Int = 1
    ) raises:
        """Start video recording to a file.

        Captures every rendered frame and encodes it by piping ``ffmpeg``
        (see ``render/video_recorder.mojo``). Requires ``ffmpeg`` on PATH.

        Args:
            filename: Output path, e.g. ``recording_0.mp4`` or ``recording_0.gif``.
            fps: Frames per second written into the video container.
            skip: Only record every Nth frame (1 = all, 2 = half, etc.).
        """
        self.recorder.start(filename, fps, skip)

    def stop_recording(mut self) raises:
        """Stop video recording and flush the file."""
        self.recorder.stop()
        if self.recording_tb:
            release_gpu_transfer_buffer(self.device.value(), self.recording_tb.value())
            self.recording_tb = None
            self.recording_tb_size = 0

    def _build_scene_uniforms(mut self):
        """Build scene uniforms from current camera state."""
        var view = self.camera.get_view_matrix()
        var proj = perspective_projection(
            self.camera.fov,
            self.camera.aspect,
            self.camera.near,
            self.camera.far,
        )
        var view_proj = proj @ view

        self.scene_uniforms.view_proj = mat4_to_gpu_f32(view_proj)

        # Camera basis for the skybox's starfield. Rebuilt every frame from the
        # SAME camera state the view matrix comes from, so the stars cannot
        # drift out of step with the scene; the fragment reconstructs its view
        # ray from these and hashes world directions, which is what keeps a
        # star nailed to a point in the sky rather than to the screen.
        var fwd = (self.camera.target - self.camera.eye).normalized()
        var right = fwd.cross(self.camera.up).normalized()
        var up = right.cross(fwd).normalized()
        self.skybox_uniforms.cam_right[0] = Float32(right.x)
        self.skybox_uniforms.cam_right[1] = Float32(right.y)
        self.skybox_uniforms.cam_right[2] = Float32(right.z)
        # `Camera3D.fov` is already in RADIANS (see ModelRenderer.__init__).
        self.skybox_uniforms.cam_right[3] = Float32(tan(self.camera.fov * 0.5))
        self.skybox_uniforms.cam_up[0] = Float32(up.x)
        self.skybox_uniforms.cam_up[1] = Float32(up.y)
        self.skybox_uniforms.cam_up[2] = Float32(up.z)
        self.skybox_uniforms.cam_up[3] = Float32(self.camera.aspect)
        self.skybox_uniforms.cam_fwd[0] = Float32(fwd.x)
        self.skybox_uniforms.cam_fwd[1] = Float32(fwd.y)
        self.skybox_uniforms.cam_fwd[2] = Float32(fwd.z)
        self.skybox_uniforms.cam_fwd[3] = 0.0

        # ── lights, MuJoCo's way (see `gpu_shaders._MJ_SHADE_MSL`) ──────────
        var num_lights = len(self.lights)
        if num_lights > MAX_SCENE_LIGHTS:
            num_lights = MAX_SCENE_LIGHTS
        self.scene_uniforms.camera_pos[0] = Float32(self.camera.eye.x)
        self.scene_uniforms.camera_pos[1] = Float32(self.camera.eye.y)
        self.scene_uniforms.camera_pos[2] = Float32(self.camera.eye.z)
        self.scene_uniforms.camera_pos[3] = Float32(num_lights)
        self.scene_uniforms.camera_fwd[0] = Float32(fwd.x)
        self.scene_uniforms.camera_fwd[1] = Float32(fwd.y)
        self.scene_uniforms.camera_fwd[2] = Float32(fwd.z)
        self.scene_uniforms.camera_fwd[3] = 0.0

        for li in range(num_lights):
            ref light = self.lights[li]
            var lx = Float32(light.dir_x)
            var ly = Float32(light.dir_y)
            var lz = Float32(light.dir_z)
            var ll = sqrt(lx * lx + ly * ly + lz * lz)
            if ll < 1e-6:
                ll = 1.0
            var o = li * 4
            var is_spot = light.mode == LightMode.SPOT
            self.scene_uniforms.light_pos[o + 0] = Float32(light.pos_x)
            self.scene_uniforms.light_pos[o + 1] = Float32(light.pos_y)
            self.scene_uniforms.light_pos[o + 2] = Float32(light.pos_z)
            self.scene_uniforms.light_pos[o + 3] = 1.0 if is_spot else 0.0
            self.scene_uniforms.light_dir[o + 0] = lx / ll
            self.scene_uniforms.light_dir[o + 1] = ly / ll
            self.scene_uniforms.light_dir[o + 2] = lz / ll
            # ⚠ cos(cutoff) IS THE CONE TEST, and 180 degrees means NO CONE
            # (OpenGL's special value, which MuJoCo gives directional lights).
            # -2 can never be beaten by a dot product, which is how the shader
            # reads "none".
            var cutoff = light.cutoff
            if cutoff >= 180.0 or not is_spot:
                self.scene_uniforms.light_dir[o + 3] = -2.0
            else:
                self.scene_uniforms.light_dir[o + 3] = Float32(
                    cos(cutoff * 3.14159265358979 / 180.0)
                )
            self.scene_uniforms.light_diffuse[o + 0] = Float32(light.color_r)
            self.scene_uniforms.light_diffuse[o + 1] = Float32(light.color_g)
            self.scene_uniforms.light_diffuse[o + 2] = Float32(light.color_b)
            self.scene_uniforms.light_diffuse[o + 3] = (
                1.0 if light.cast_shadow else 0.0
            )
            self.scene_uniforms.light_specular[o + 0] = Float32(light.specular_r)
            self.scene_uniforms.light_specular[o + 1] = Float32(light.specular_g)
            self.scene_uniforms.light_specular[o + 2] = Float32(light.specular_b)
            self.scene_uniforms.light_specular[o + 3] = Float32(light.exponent)
            self.scene_uniforms.light_ambient[o + 0] = Float32(light.ambient_r)
            self.scene_uniforms.light_ambient[o + 1] = Float32(light.ambient_g)
            self.scene_uniforms.light_ambient[o + 2] = Float32(light.ambient_b)
            self.scene_uniforms.light_ambient[o + 3] = 0.0
            self.scene_uniforms.light_atten[o + 0] = Float32(light.attenuation_0)
            self.scene_uniforms.light_atten[o + 1] = Float32(light.attenuation_1)
            self.scene_uniforms.light_atten[o + 2] = Float32(light.attenuation_2)
            self.scene_uniforms.light_atten[o + 3] = 0.0

        for k in range(3):
            self.scene_uniforms.headlight_ambient[k] = self.headlight_ambient[k]
            self.scene_uniforms.headlight_diffuse[k] = self.headlight_diffuse[k]
            self.scene_uniforms.headlight_specular[k] = self.headlight_specular[k]
        self.scene_uniforms.headlight_ambient[3] = (
            1.0 if self.headlight_active else 0.0
        )
        # `render_gl3.c:initLights`: "create some ambient light if no
        # supported lights are present" — 0.3, and only then.
        self.scene_uniforms.headlight_diffuse[3] = (
            0.3 if (num_lights == 0 and not self.headlight_active) else 0.0
        )
        self.scene_uniforms.headlight_specular[3] = 0.0

        # Fog params
        self.scene_uniforms.fog_params[0] = self.fog_start
        self.scene_uniforms.fog_params[1] = self.fog_end
        # z = the ground's reflectance, the reflection pass's blend weight.
        self.scene_uniforms.fog_params[2] = self.ground_reflectance

    def _build_light_view_proj(mut self):
        """Build light's orthographic view-projection matrix for shadow mapping.

        Uses the first shadow-casting light, or light 0 as fallback.
        """
        # Find first shadow-casting light direction
        var light_dir = Vec3(
            Float64(self.scene_uniforms.light_dir[0]),
            Float64(self.scene_uniforms.light_dir[1]),
            Float64(self.scene_uniforms.light_dir[2]),
        )
        if light_dir.length() < 1e-6:
            light_dir = Vec3(0.3, -0.4, -0.8).normalized()
        for li in range(len(self.lights)):
            if self.lights[li].cast_shadow:
                var lx = self.lights[li].dir_x
                var ly = self.lights[li].dir_y
                var lz = self.lights[li].dir_z
                var ll = sqrt(lx * lx + ly * ly + lz * lz)
                if ll < 1e-6:
                    ll = 1.0
                light_dir = Vec3(lx / ll, ly / ll, lz / ll)
                break

        # ⚠ THE FRUSTUM IS FITTED TO THE VIEW, NOT HARDCODED. It used to be a
        # fixed 16 m box 15 m from the target with a 0.1..30 depth range —
        # sized for a 2 m robot on a flat plane, which is every suite model
        # except one. `quadruped escape`'s terrain is 60 m across with 5 m of
        # relief, so that box covered a quarter of its width: shadows existed
        # in a patch around the camera target and nowhere else, with a visible
        # straight edge where the box ended.
        #
        # The camera distance is the scale signal available here — it is set
        # per model by `setup_cameras` and by the user's zoom, so it tracks
        # what is actually on screen. ⚠ FLOORED AT THE OLD VALUES so every
        # model that looked right before still gets exactly the old frustum;
        # this only ever loosens.
        var target = self.camera.target
        var cam_dist = (self.camera.eye - target).length()
        var ortho_size = 8.0
        if 0.75 * cam_dist > ortho_size:
            ortho_size = 0.75 * cam_dist
        var light_distance = 15.0
        if 2.5 * ortho_size > light_distance:
            light_distance = 2.5 * ortho_size
        var light_pos = target - light_dir * light_distance

        # Build look-at view matrix for the light using Mat4.look_at
        # which correctly handles the sign conventions (Z-row = -forward)
        var up = Vec3(0.0, 0.0, 1.0)
        # If light is nearly vertical, use a different up vector
        var abs_dot = abs(
            light_dir.x * up.x + light_dir.y * up.y + light_dir.z * up.z
        )
        if abs_dot > 0.99:
            up = Vec3(0.0, 1.0, 0.0)

        var light_view = Mat4.look_at(light_pos, target, up)

        # ⚠ FAR REACHES PAST THE LIGHT, BY THE BOX'S OWN SIZE. An occluder can
        # sit anywhere in the box, including well below the target; a far plane
        # that only just reaches the target clips the ground out of the map.
        var light_far = light_distance + 2.0 * ortho_size
        var light_proj = ortho_projection(
            -ortho_size,
            ortho_size,
            -ortho_size,
            ortho_size,
            0.1,
            light_far,
        )

        var light_vp = light_proj @ light_view
        self.shadow_uniforms.light_view_proj = mat4_to_gpu_f32(light_vp)

        # ⚠ PUBLISHED HERE, NOT AT RESOURCE CREATION. `shadow_size` comes from
        # `<visual quality shadowsize=>` and the map is re-created on a task
        # switch when the two models disagree; setting it once at init leaves
        # the shader dividing by the PREVIOUS model's resolution. This runs
        # every frame, so it cannot go stale.
        self.shadow_uniforms.params[2] = Float32(self.shadow_size)

    # --- Event Handling ---

    def check_quit(mut self) -> Bool:
        """Poll SDL events: quit, camera switch, mouse orbit/pan/zoom, pause, step.

        Keyboard:
          Escape / window close  → quit
          1-9                    → switch to camera N (read by ModelRenderer)
          Space                  → toggle pause
          → (Right arrow)        → step one frame while paused
          R                      → reset camera to default position
          S                      → save screenshot (screenshotNNNN.jpg)
          V                      → toggle video recording (recordingNNNN.mp4)

        Mouse (any button drag = orbit, Shift+drag = pan, wheel = zoom):
          Button drag            → orbit camera around target
          Shift + button drag    → pan camera (target + eye translate together)
          Scroll wheel           → zoom in/out

        Returns:
            True if quit event detected.
        """
        self.camera_switch_request = -1
        self.step_once = False
        var event = Event()
        var has_events = True

        # Who owns the pointer and the keyboard this frame.
        #
        # ⚠ THESE ARE LAST FRAME'S ANSWERS, and that is correct. ImGui can only
        # decide what it wants once its widgets have been laid out, which
        # happens after `NewFrame` — i.e. after this pump has already run. Every
        # ImGui integration reads them one frame stale; the flags change on
        # hover, so a one-frame lag is invisible.
        var ig_mouse = False
        var ig_kbd = False
        if self.imgui_on:
            try:
                ig_mouse = ig_want_mouse()
                ig_kbd = ig_want_keyboard()
            except:
                pass
        # ⚠ AN OVERLAY ImGui CANNOT SPEAK FOR. See `pointer_claimed` — a
        # gizmo's window carries `NoInputs`, so `ig_want_mouse()` says False
        # while it is being dragged.
        if self.pointer_claimed:
            ig_mouse = True

        while has_events:
            try:
                has_events = poll_event(Ptr(to=event))
            except:
                has_events = False

            if not has_events:
                break

            # ImGui sees EVERY event, including ones claimed below. It tracks
            # key-up, focus and modifier state, so filtering here would leave
            # it believing a key is still held.
            if self.imgui_on:
                try:
                    ig_process_event(Ptr(to=event))
                except:
                    pass

            var event_type = event[UInt32]

            if EventType(event_type) == EventType.EVENT_QUIT:
                self.should_quit = True
                return True

            elif EventType(event_type) == EventType.EVENT_KEY_DOWN:
                var key_event = event[KeyboardEvent]
                var key_val = Int(key_event.key)
                if ig_kbd:
                    # ImGui is taking typed input: "s" is a letter, not the
                    # screenshot shortcut, and ESC closes a popup rather than
                    # the window. This is what makes `text_input_mode`
                    # unnecessary — the UI reports its own focus instead of the
                    # application having to declare it.
                    pass
                elif self.text_input_mode:
                    # Everything goes to the application while it is typing.
                    self.last_key = key_val
                elif key_val == Int(Keycode.SDLK_ESCAPE):
                    self.should_quit = True
                    return True
                elif key_val >= 0x31 and key_val <= 0x39:
                    # Number keys 1-9: SDLK_1=0x31 … SDLK_9=0x39
                    self.camera_switch_request = key_val - 0x31
                elif key_val == Int(Keycode.SDLK_SPACE):
                    self.is_paused = not self.is_paused
                elif key_val == Int(Keycode.SDLK_RIGHT):
                    if self.is_paused:
                        self.step_once = True
                elif key_val == Int(Keycode.SDLK_R):
                    self.camera.eye = self.default_eye
                    self.camera.target = self.default_target
                elif key_val == Int(Keycode.SDLK_S):
                    self.screenshot_requested = True
                elif key_val == Int(Keycode.SDLK_V):
                    try:
                        if self.recorder.is_recording:
                            self.stop_recording()
                        else:
                            var fname = (
                                "recording_"
                                + String(self.screenshot_counter)
                                + ".mp4"
                            )
                            self.start_recording(fname)
                    except:
                        pass
                else:
                    # Anything the bindings above do not claim is handed to
                    # the application rather than dropped.
                    self.last_key = key_val
            elif EventType(event_type) == EventType.EVENT_MOUSE_BUTTON_DOWN:
                # Track any button press (macOS trackpad intermittently
                # misidentifies left as right, so treat all buttons same)
                var mb = event[MouseButtonEvent]
                self.mouse_x = Float32(mb.x)
                self.mouse_y = Float32(mb.y)
                self.mouse_clicked = not ig_mouse
                # A drag that STARTS over the UI must not orbit the camera —
                # otherwise every slider drag spins the scene behind it. Where
                # the gesture began is what matters, not where the pointer
                # currently is, so this is latched on press.
                #
                # Two sources, because both UI layers exist: `ui_sidebar_width`
                # is the reserved strip the hand-rolled widgets live in, and
                # `ig_mouse` covers ImGui windows, which can float anywhere.
                self.mouse_left_down = (
                    not ig_mouse
                    and Float32(mb.x) >= Float32(self.ui_sidebar_width)
                )

            elif EventType(event_type) == EventType.EVENT_MOUSE_BUTTON_UP:
                self.mouse_left_down = False

            elif EventType(event_type) == EventType.EVENT_MOUSE_MOTION:
                var motion = event[MouseMotionEvent]
                self.mouse_x = Float32(motion.x)
                self.mouse_y = Float32(motion.y)
                var dx = Float64(motion.xrel)
                var dy = Float64(motion.yrel)
                if self.mouse_left_down:
                    # Shift+drag = pan, plain drag = orbit
                    var is_shift = False
                    try:
                        var mod = get_mod_state()
                        is_shift = Int(mod) & Int(Keymod.KMOD_SHIFT) != 0
                    except:
                        pass
                    if is_shift:
                        # Pan: scale by distance so speed feels constant
                        var dist = (
                            self.camera.eye - self.camera.target
                        ).length()
                        var scale = dist * 0.002
                        self.camera.pan(-dx * scale, -dy * scale)
                    else:
                        # Orbit: ~0.005 rad/px gives smooth rotation
                        self.camera.orbit(dx * 0.005, dy * 0.005)
                    if self.has_ground:
                        self.camera.clamp_above_ground(self.ground_z)

            elif EventType(event_type) == EventType.EVENT_MOUSE_WHEEL:
                if ig_mouse:
                    # Scrolling a task list must not also dolly the camera.
                    continue
                var wheel = event[MouseWheelEvent]
                # Scroll up (positive y) = zoom in (move eye closer)
                # ⚠ FRACTIONAL, not a fixed 0.5 m step. A fixed step is
                # unusable at both ends: invisible when far out, and a jump
                # straight through a 30 cm robot when close in. 12 % per
                # notch is roughly one comfortable "step" at any scale.
                self.camera.zoom_fraction(-Float64(wheel.y) * 0.12)
                if self.has_ground:
                    self.camera.clamp_above_ground(self.ground_z)

        return self.should_quit

    # --- Camera Controls ---

    def set_camera_position(mut self, eye: Vec3, target: Vec3):
        """Set camera position and target.

        Args:
            eye: Camera position.
            target: Look-at target.
        """
        self.camera.eye = eye
        self.camera.target = target

    def orbit_camera(mut self, delta_theta: Float64, delta_phi: Float64):
        """Orbit camera around target.

        Args:
            delta_theta: Horizontal rotation (radians).
            delta_phi: Vertical rotation (radians).
        """
        self.camera.orbit(delta_theta, delta_phi)

    def zoom_camera(mut self, delta: Float64):
        """Zoom camera in/out.

        Args:
            delta: Zoom amount.
        """
        self.camera.zoom(delta)

    def pan_camera(mut self, delta_x: Float64, delta_y: Float64):
        """Pan camera (translate target and eye together).

        Args:
            delta_x: Horizontal pan in camera-right direction.
            delta_y: Vertical pan in camera-up direction.
        """
        self.camera.pan(delta_x, delta_y)

    def reset_camera(mut self):
        """Reset camera eye and target to the values set at construction."""
        self.camera.eye = self.default_eye
        self.camera.target = self.default_target

    def delay_ms(self, ms: Int) raises:
        """Delay for given milliseconds.

        Args:
            ms: Milliseconds to delay.
        """
        delay(UInt32(ms))

    # --- Cleanup ---

    def close(mut self) raises:
        """Release all GPU resources and shutdown SDL3."""
        if not self.initialized:
            return

        # ⚠ FIRST. ImGui's backend owns GPU buffers, a font texture and a
        # pipeline created on THIS device; tearing the device down first would
        # leave it releasing freed handles. The viewer destroys and rebuilds
        # the window on every task switch, so this runs often, not once.
        self.imgui_close()

        # Stop any active recording and release the persistent transfer buffer
        if self.recorder.is_recording:
            self.recorder.stop()
        if self.recording_tb:
            release_gpu_transfer_buffer(self.device.value(), self.recording_tb.value())
            self.recording_tb = None
            self.recording_tb_size = 0

        # Release the geom-keyed caches (capsules, cylinders, STL, textures).
        self._release_model_caches()

        # Release default texture resources
        if self.default_texture:
            release_gpu_texture(self.device.value(), self.default_texture.value())
            release_gpu_sampler(self.device.value(), self.default_tex_sampler.value())

        # Release static mesh buffers
        release_gpu_buffer(self.device.value(), self.sphere_mesh.value().vertex_buffer)
        release_gpu_buffer(self.device.value(), self.sphere_mesh.value().index_buffer)
        release_gpu_buffer(self.device.value(), self.box_mesh.value().vertex_buffer)
        release_gpu_buffer(self.device.value(), self.box_mesh.value().index_buffer)
        release_gpu_buffer(self.device.value(), self.ground_mesh.value().vertex_buffer)
        release_gpu_buffer(self.device.value(), self.ground_mesh.value().index_buffer)

        # Release line buffers
        release_gpu_buffer(self.device.value(), self.line_vertex_buffer.value())
        release_gpu_transfer_buffer(self.device.value(), self.line_transfer_buffer.value())

        # Release depth texture
        release_gpu_texture(self.device.value(), self.depth_texture.value())

        # Release shadow resources
        release_gpu_texture(self.device.value(), self.shadow_map.value())
        release_gpu_sampler(self.device.value(), self.shadow_sampler.value())

        # Release text resources
        release_gpu_buffer(self.device.value(), self.text_vertex_buffer.value())
        release_gpu_buffer(self.device.value(), self.text_index_buffer.value())
        release_gpu_transfer_buffer(self.device.value(), self.text_transfer_buffer.value())
        release_gpu_texture(self.device.value(), self.font_atlas_tex.value())
        release_gpu_sampler(self.device.value(), self.font_sampler.value())

        # Release pipelines
        release_gpu_graphics_pipeline(self.device.value(), self.solid_pipeline.value())
        release_gpu_graphics_pipeline(self.device.value(), self.ground_pipeline.value())
        release_gpu_graphics_pipeline(self.device.value(), self.line_pipeline.value())
        release_gpu_graphics_pipeline(self.device.value(), self.shadow_pipeline.value())
        release_gpu_graphics_pipeline(self.device.value(), self.reflection_pipeline.value())
        release_gpu_graphics_pipeline(self.device.value(), self.skybox_pipeline.value())
        release_gpu_graphics_pipeline(self.device.value(), self.text_pipeline.value())

        # Release window and device
        release_window_from_gpu_device(self.device.value(), self.window.value())
        destroy_window(self.window.value())
        destroy_gpu_device(self.device.value())
        quit()

        self.initialized = False

"""A mesh index buffer must address EVERY vertex — the 16-bit wrap.

WHY THIS EXISTS
===============
`MeshData.indices` was `List[UInt16]`, so a mesh with more than 65,535
vertices had every index past that WRAP rather than fail. The triangles beyond
the wrap are still drawn — connecting the wrong points — so the mesh renders
PARTIALLY, with no error anywhere.

Menagerie's ToddlerBot head is 62,912 triangles = **188,736 vertices**, and
about a third of it drew: the robot appeared with no face and a single eye.

⚠⚠ WHAT MAKES THIS WORTH A GATE IS HOW WELL IT HID. Every check upstream
passed while the picture was wrong — the `<mesh>` asset resolved, the STL
loaded and uploaded without raising, `geom_mesh_id` was valid, the body sat at
MuJoCo's exact position, and the geom counts matched MuJoCo's visible set
element for element. The loss happens INSIDE one mesh, downstream of
everything any parse-level test can see. The only earlier symptom was a
person looking at a screen and saying "the head is wrong".

⚠ OUR OWN ASSETS ARE ALL UNDER THE LIMIT, which is why it survived: it takes a
~3 MB STL to reach 65,535 vertices, and every mesh shipped in this tree is
smaller. Menagerie is full of them. So the fixture is deliberately an
EXTERNAL model — a gate built only from our own assets could not fail.

Run: pixi run mojo run -I . tests/render/test_mesh_index_width.mojo
"""

from noeira.render.stl_loader import load_stl
from noeira.render.gpu_mesh import generate_sphere
from noeira.render.renderer3d import (
    MAX_TEXT_CHARS, TEXT_INDEX_BYTES, TEXT_INDEX_ELEMENT_SIZE,
)
from noeira.render.sdl.sdl_gpu import GPUIndexElementSize

comptime BIG = String(
    "references/mujoco_menagerie-main/toddlerbot_2xc/assets/head_visual.stl"
)
comptime U16_MAX: Int = 65535


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def main() raises:
    var t = Tally()
    print("=== mesh index width ===")

    var m = load_stl(BIG)
    var nverts = len(m.vertices)
    var nidx = len(m.indices)
    print("   head_visual:", nverts, "vertices,", nidx, "indices")

    # ⚠ NON-VACUITY FIRST, AND IT IS THE WHOLE POINT. A fixture under 65,536
    # vertices cannot distinguish a 16-bit index buffer from a 32-bit one, so
    # every arm below would pass on the broken code. Assert the fixture is big
    # enough BEFORE trusting anything it proves.
    t.truth(
        nverts > U16_MAX,
        String("the fixture EXCEEDS the 16-bit range (", nverts, " > ",
               U16_MAX, ") — the arms below are live"),
    )

    t.truth(nidx == nverts,
            String("one index per vertex (", nidx, " vs ", nverts, ")"))

    # The sharp arm. With `UInt16` the maximum index is exactly 65,535 no
    # matter how large the mesh is; with `UInt32` it is nverts - 1.
    var maxi = 0
    var wrapped = 0
    for i in range(nidx):
        var v = Int(m.indices[i])
        if v > maxi:
            maxi = v
        # An STL has no shared vertices, so index i MUST be i. Any other value
        # is a wrap, and this catches it at the first one rather than only at
        # the maximum.
        if v != i:
            wrapped += 1
    t.truth(maxi == nverts - 1,
            String("the largest index addresses the LAST vertex (", maxi,
                   ", not ", U16_MAX, ")"))
    t.truth(wrapped == 0,
            String("no index wrapped (", wrapped, " of ", nidx, " wrong)"))

    # ⚠ THE SMALL-MESH PATH MUST STILL WORK. Widening the element size is a
    # change to EVERY draw call, not just the big ones, and a generated
    # primitive is what most of the renderer actually draws.
    var s = generate_sphere(16, 12)
    var s_max = 0
    for i in range(len(s.indices)):
        if Int(s.indices[i]) > s_max:
            s_max = Int(s.indices[i])
    t.truth(len(s.indices) > 0 and s_max < len(s.vertices),
            String("a generated sphere still indexes in range (max ", s_max,
                   " < ", len(s.vertices), ")"))

    # ------------------------------------------------------------------
    # ⚠ THE TEXT ATLAS, WHICH THIS GATE'S OWN COMMIT BROKE.
    #
    # `b147dcde8` widened the MESH indices to 32-bit and swept all three
    # `bind_gpu_index_buffer` sites — including the HUD's, whose buffer is
    # still allocated and written as UInt16. The mesh legs above all passed
    # while the HUD spent every frame decoding PAIRS of its indices as single
    # UInt32s: [0,1,2,2,3,0] reads back as 65536, 131074, 3, so the vertex
    # fetch ran far past an 8192-vertex buffer and drew quads from whatever
    # was next in memory — coloured flashes over the scene.
    #
    # The mesh legs could not see it: they test `MeshData`, and the HUD does
    # not go through `MeshData`. This leg ties the ALLOCATION to the BIND so
    # the next blanket edit fails here instead of on screen.
    # ------------------------------------------------------------------
    var text_bind_is_16 = (
        TEXT_INDEX_ELEMENT_SIZE
        == GPUIndexElementSize.GPU_INDEXELEMENTSIZE_16BIT
    )
    t.truth(
        (TEXT_INDEX_BYTES == 2) == text_bind_is_16,
        String("the HUD's index STRIDE and BIND agree (", TEXT_INDEX_BYTES,
               " bytes, 16-bit bind ", text_bind_is_16, ")"),
    )
    var text_max_idx = MAX_TEXT_CHARS * 4 - 1
    t.truth(
        (not text_bind_is_16) or text_max_idx <= U16_MAX,
        String("the HUD's largest index fits its element size (", text_max_idx,
               " <= ", U16_MAX, ")"),
    )

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_mesh_index_width: " + String(t.fails) + " failed")

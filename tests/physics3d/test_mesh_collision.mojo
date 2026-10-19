"""Test: Mesh collision accuracy via GJK/EPA (fields path).

Tests the GJK/EPA collision between a mesh geom and a box geom to verify
correct distance computation. Isolates the bug where GJK reports deep
penetration for clearly separated shapes.

G4: ported from the legacy `gjk.gjk_epa` (List-based mesh verts) to
`gjk.gjk_epa` (shared `[NMESH_VERTS, 3]` LayoutTensor + vert-adr
offsets — the production signature used by contact_detection /
broadphase_sap).
"""

from std.math import sqrt
from noeira.core.tiles import RowMajor
from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.collision.ccd_workspace import L_CCD_WS1
from noeira.physics3d.collision.ccd_workspace_host import ccd_ws_alloc
from noeira.physics3d.collision.gjk import gjk_epa
from noeira.physics3d.collision.gjk_support import (
    support_sphere,
    support_box,
    support_mesh,
)
from noeira.physics3d.constants import GEOM_SPHERE, GEOM_BOX, GEOM_MESH
from noeira.physics3d.gpu.constants import mesh_max_edge

comptime NMV = 8
comptime L_MV = RowMajor[NMV, 3]
# ⚠ NO EDGE GRAPH HERE, ON PURPOSE. `_support_mesh` hill-climbs the hull
# adjacency when it exists and falls back to the exhaustive scan when
# `edgeadr < 0` — which is how `fields/model.mojo` marks an unbuilt graph.
# These fixtures are 8-vertex cubes, below `_HILLCLIMB_MIN`, so the scan is the
# path under test either way; the tensors just have to be present and marked
# absent. `test_gjk_hillclimb_support.mojo` is where the two paths are compared.
comptime L_VEADR = RowMajor[NMV]
comptime L_EDGES = RowMajor[mesh_max_edge(NMV)]


def _no_graph() raises -> TensorImpl[DType.float64]:
    """An all -1 `mesh_vert_edgeadr` — i.e. "this mesh has no hull graph"."""
    var t = TensorImpl[DType.float64].alloc(NMV)
    for i in range(NMV):
        t.data[i] = -1.0
    return t^


def _no_edges() raises -> TensorImpl[DType.float64]:
    var t = TensorImpl[DType.float64].alloc(mesh_max_edge(NMV))
    for i in range(mesh_max_edge(NMV)):
        t.data[i] = -1.0
    return t^


def _mv_tensor(verts: List[Float64]) raises -> TensorImpl[DType.float64]:
    """Pack a flat xyz vert List into the shared [NMV, 3] mesh-verts tensor."""
    var t = TensorImpl[DType.float64].alloc(NMV * 3)
    for i in range(min(len(verts), NMV * 3)):
        t.data[i] = verts[i]
    return t^


def test_box_box_separated() raises:
    """Two boxes clearly separated — should report positive distance."""
    print("=== Test: box-box separated ===")
    var mv = _mv_tensor(List[Float64]())
    var _ng = _no_graph()
    var _ne = _no_edges()
    var ws = ccd_ws_alloc[DType.float64]()
    var result = gjk_epa[DType.float64](
        GEOM_BOX,
        0.0, 0.0, 2.0,  # box1 at z=2
        0.0, 0.0, 0.0, 1.0,  # identity quat
        0.0, 0.0,  # radius, half_length (unused for box)
        0.5, 0.5, 0.5,  # half-extents
        mv.tt["cpu", NMV, 3](), _ng.tt["cpu", NMV](), _ne.tt["cpu", mesh_max_edge(NMV)](), 0, 0,
        GEOM_BOX,
        0.0, 0.0, 0.0,  # box2 at origin
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.0,
        0.5, 0.5, 0.5,
        0, 0,
        ws.tt_dyn["cpu", L_CCD_WS1](L_CCD_WS1()), 0,
    )
    print("  dist=", Float64(result[0]), "(expected ~1.0)")
    if result[0] < 0:
        raise Error("FAIL: reported penetration for separated boxes!")
    else:
        print("  PASS")


def test_box_box_overlapping() raises:
    """Two overlapping boxes — should report negative distance."""
    print("=== Test: box-box overlapping ===")
    var mv = _mv_tensor(List[Float64]())
    var _ng = _no_graph()
    var _ne = _no_edges()
    var ws = ccd_ws_alloc[DType.float64]()
    var result = gjk_epa[DType.float64](
        GEOM_BOX,
        0.0, 0.0, 0.3,  # box1 at z=0.3
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.0,
        0.5, 0.5, 0.5,
        mv.tt["cpu", NMV, 3](), _ng.tt["cpu", NMV](), _ne.tt["cpu", mesh_max_edge(NMV)](), 0, 0,
        GEOM_BOX,
        0.0, 0.0, 0.0,  # box2 at origin
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.0,
        0.5, 0.5, 0.5,
        0, 0,
        ws.tt_dyn["cpu", L_CCD_WS1](L_CCD_WS1()), 0,
    )
    print("  dist=", Float64(result[0]), "(expected ~-0.7)")
    if result[0] >= 0:
        raise Error("FAIL: reported separation for overlapping boxes!")
    else:
        print("  PASS")


def test_sphere_mesh_separated() raises:
    """Sphere far from a mesh cube — should report positive distance."""
    print("=== Test: sphere-mesh separated ===")

    # Create a simple cube mesh (8 vertices)
    var cube_verts = List[Float64]()
    for sx in range(2):
        for sy in range(2):
            for sz in range(2):
                cube_verts.append(Float64(sx) - 0.5)  # x: -0.5 to 0.5
                cube_verts.append(Float64(sy) - 0.5)  # y: -0.5 to 0.5
                cube_verts.append(Float64(sz) - 0.5)  # z: -0.5 to 0.5

    var mv = _mv_tensor(cube_verts)

    var _ng = _no_graph()

    var _ne = _no_edges()
    var ws = ccd_ws_alloc[DType.float64]()
    var result = gjk_epa[DType.float64](
        GEOM_SPHERE,
        0.0, 0.0, 3.0,  # sphere at z=3
        0.0, 0.0, 0.0, 1.0,
        0.1, 0.0,  # radius=0.1
        0.0, 0.0, 0.0,
        mv.tt["cpu", NMV, 3](), _ng.tt["cpu", NMV](), _ne.tt["cpu", mesh_max_edge(NMV)](), 0, 0,
        GEOM_MESH,
        0.0, 0.0, 0.0,  # mesh at origin
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.0,
        0.0, 0.0, 0.0,
        0, 8,
        ws.tt_dyn["cpu", L_CCD_WS1](L_CCD_WS1()), 0,
    )
    print("  dist=", Float64(result[0]), "(expected ~2.4)")
    if result[0] < 0:
        raise Error("FAIL: reported penetration for separated shapes!")
    else:
        print("  PASS")


def test_mesh_box_sawyer_case() raises:
    """Reproduce the Sawyer case: small mesh at z=0.2, large box from z=-0.92 to z=0."""
    print("=== Test: mesh-box Sawyer case (separated) ===")

    # Small cube mesh (simulating eGripperBase hull) centered at local origin
    # with extent ~0.06
    var mesh_verts = List[Float64]()
    for sx in range(2):
        for sy in range(2):
            for sz in range(2):
                mesh_verts.append((Float64(sx) - 0.5) * 0.06)
                mesh_verts.append((Float64(sy) - 0.5) * 0.06)
                mesh_verts.append((Float64(sz) - 0.5) * 0.06)

    var mv = _mv_tensor(mesh_verts)

    var _ng = _no_graph()

    var _ne = _no_edges()
    var ws = ccd_ws_alloc[DType.float64]()
    # Mesh geom: body_pos = (0, 0.6, 0.2), geom_local_offset = (0, 0, 0.03)
    # World pos ≈ (0, 0.6, 0.23)
    var result = gjk_epa[DType.float64](
        GEOM_MESH,
        0.0, 0.6, 0.23,  # mesh world pos
        0.0, 0.0, 0.0, 1.0,  # identity quat
        0.0, 0.0,
        0.0, 0.0, 0.0,
        mv.tt["cpu", NMV, 3](), _ng.tt["cpu", NMV](), _ne.tt["cpu", mesh_max_edge(NMV)](), 0, 8,
        GEOM_BOX,
        0.0, 0.6, -0.46,  # table box center
        0.0, 0.0, 0.0, 1.0,  # identity quat
        0.0, 0.0,
        0.7, 0.4, 0.46,  # half-extents
        0, 0,
        ws.tt_dyn["cpu", L_CCD_WS1](L_CCD_WS1()), 0,
    )
    print("  dist=", Float64(result[0]),
          "contact_z=", Float64(result[3]),
          "normal_z=", Float64(result[6]))
    print("  (expected dist > 0, mesh bottom at z=0.20, box top at z=0.0)")
    if result[0] < 0:
        raise Error("FAIL: reported penetration for separated shapes!")
        print("  Mesh at z=0.23 ± 0.03 → range [0.20, 0.26]")
        print("  Box from z=-0.92 to z=0.0")
        print("  Gap should be ~0.20")
    else:
        print("  PASS: dist=", Float64(result[0]))


def test_mesh_box_touching() raises:
    """Mesh cube sitting exactly on box surface — should report ~0 or slight penetration."""
    print("=== Test: mesh-box touching ===")

    var mesh_verts = List[Float64]()
    for sx in range(2):
        for sy in range(2):
            for sz in range(2):
                mesh_verts.append((Float64(sx) - 0.5) * 0.04)
                mesh_verts.append((Float64(sy) - 0.5) * 0.04)
                mesh_verts.append((Float64(sz) - 0.5) * 0.04)

    var mv = _mv_tensor(mesh_verts)

    var _ng = _no_graph()

    var _ne = _no_edges()
    var ws = ccd_ws_alloc[DType.float64]()
    # Mesh at z=0.02 (bottom at z=0.0 = box top)
    var result = gjk_epa[DType.float64](
        GEOM_MESH,
        0.0, 0.0, 0.02,  # mesh at z=0.02
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.0,
        0.0, 0.0, 0.0,
        mv.tt["cpu", NMV, 3](), _ng.tt["cpu", NMV](), _ne.tt["cpu", mesh_max_edge(NMV)](), 0, 8,
        GEOM_BOX,
        0.0, 0.0, -0.5,  # box center at z=-0.5, half_z=0.5 → top at z=0.0
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.0,
        1.0, 1.0, 0.5,
        0, 0,
        ws.tt_dyn["cpu", L_CCD_WS1](L_CCD_WS1()), 0,
    )
    print("  dist=", Float64(result[0]), "(expected ~0.0)")
    if result[0] < -0.01:
        raise Error("FAIL: too deep penetration!")
    elif result[0] > 0.01:
        raise Error("FAIL: should be touching!")
    else:
        print("  PASS")


def test_mesh_box_rotated() raises:
    """Mesh with non-identity quaternion — simulates Sawyer hand rotation."""
    print("=== Test: mesh-box rotated (Sawyer-like) ===")

    var mesh_verts = List[Float64]()
    for sx in range(2):
        for sy in range(2):
            for sz in range(2):
                mesh_verts.append((Float64(sx) - 0.5) * 0.06)
                mesh_verts.append((Float64(sy) - 0.5) * 0.06)
                mesh_verts.append((Float64(sz) - 0.5) * 0.06)

    var mv = _mv_tensor(mesh_verts)

    var _ng = _no_graph()

    var _ne = _no_edges()
    var ws = ccd_ws_alloc[DType.float64]()
    # Mesh at z=0.23 with 90° rotation (quat = [0.707, 0, 0, 0.707])
    var sq2 = 0.7071067811865476
    var result = gjk_epa[DType.float64](
        GEOM_MESH,
        0.0, 0.6, 0.23,
        sq2, 0.0, 0.0, sq2,  # 90° rotation around X
        0.0, 0.0,
        0.0, 0.0, 0.0,
        mv.tt["cpu", NMV, 3](), _ng.tt["cpu", NMV](), _ne.tt["cpu", mesh_max_edge(NMV)](), 0, 8,
        GEOM_BOX,
        0.0, 0.6, -0.46,
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.0,
        0.7, 0.4, 0.46,
        0, 0,
        ws.tt_dyn["cpu", L_CCD_WS1](L_CCD_WS1()), 0,
    )
    print("  dist=", Float64(result[0]),
          "(expected ~0.2, mesh at z=0.23±0.03, box top at z=0.0)")
    if result[0] < 0:
        raise Error("FAIL: false penetration!")
    else:
        print("  PASS")


def test_mesh_box_asymmetric() raises:
    """Large box vs small mesh — asymmetric sizes like Sawyer."""
    print("=== Test: mesh-box asymmetric sizes ===")

    # Very small mesh (like eGripperBase, ~6cm)
    var mesh_verts = List[Float64]()
    for sx in range(2):
        for sy in range(2):
            for sz in range(2):
                mesh_verts.append((Float64(sx) - 0.5) * 0.04)
                mesh_verts.append((Float64(sy) - 0.5) * 0.03)
                mesh_verts.append((Float64(sz) - 0.5) * 0.05)

    var mv = _mv_tensor(mesh_verts)

    var _ng = _no_graph()

    var _ne = _no_edges()
    var ws = ccd_ws_alloc[DType.float64]()
    # Very large box (table: 1.4m × 0.8m × 0.92m)
    var result = gjk_epa[DType.float64](
        GEOM_MESH,
        0.0, 0.6, 0.23,
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.0,
        0.0, 0.0, 0.0,
        mv.tt["cpu", NMV, 3](), _ng.tt["cpu", NMV](), _ne.tt["cpu", mesh_max_edge(NMV)](), 0, 8,
        GEOM_BOX,
        0.0, 0.6, -0.46,
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.0,
        0.7, 0.4, 0.46,
        0, 0,
        ws.tt_dyn["cpu", L_CCD_WS1](L_CCD_WS1()), 0,
    )
    print("  dist=", Float64(result[0]),
          "(expected ~0.205)")
    if result[0] < 0:
        raise Error("FAIL: false penetration!")
    else:
        print("  PASS")


def test_mesh_box_actual_sawyer() raises:
    """Reproduce the exact Sawyer case: eGripperBase hull bounds + real rotation."""
    print("=== Test: actual eGripperBase vs table box ===")

    # eGripperBase hull: x[-0.024, 0.042], y[-0.053, 0.053], z[-0.003, 0.051]
    var mesh_verts = List[Float64]()
    var xs = [-0.024, 0.042]
    var ys = [-0.053, 0.053]
    var zs = [-0.003, 0.051]
    for xi in range(2):
        for yi in range(2):
            for zi in range(2):
                mesh_verts.append(xs[xi])
                mesh_verts.append(ys[yi])
                mesh_verts.append(zs[zi])

    var mv = _mv_tensor(mesh_verts)

    var _ng = _no_graph()

    var _ne = _no_edges()
    var ws = ccd_ws_alloc[DType.float64]()
    # Body 23 at (0, 0.6, 0.2), quat (0.707, 0, 0, 0.707)
    # Geom local offset (0, 0, 0.03) → world pos via quat_rotate + body_pos
    # With 90° X rotation: local (0,0,0.03) → world (0, -0.03, 0) + body → (0, 0.57, 0.2)
    # But _geom_world_pos also composes quaternions...
    # Let me use the approximate world position
    var sq2 = 0.7071067811865476
    var result = gjk_epa[DType.float64](
        GEOM_MESH,
        0.0, 0.57, 0.2,  # approximate world pos after rotation
        sq2, 0.0, 0.0, sq2,  # body 23 quat: 90° around X
        0.0, 0.0,
        0.0, 0.0, 0.0,
        mv.tt["cpu", NMV, 3](), _ng.tt["cpu", NMV](), _ne.tt["cpu", mesh_max_edge(NMV)](), 0, 8,
        GEOM_BOX,
        0.0, 0.6, -0.46,  # table collision box
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.0,
        0.7, 0.4, 0.46,  # table half-extents
        0, 0,
        ws.tt_dyn["cpu", L_CCD_WS1](L_CCD_WS1()), 0,
    )
    # After 90° X rotation: mesh local z becomes world y, local y becomes world -z
    # Hull z-range [-0.003, 0.051] → world y offset
    # Hull y-range [-0.053, 0.053] → world z-range, centered at z=0.2
    # So world z: [0.2-0.053, 0.2+0.053] = [0.147, 0.253]
    # Table top at z=0.0, gap ≈ 0.147
    print("  dist=", Float64(result[0]),
          "(expected ~0.15)")
    if result[0] < 0:
        raise Error("FAIL: false penetration!")
    else:
        print("  PASS")

    # Also test with Y overlap (mesh y ≈ 0.57, table y range [0.2, 1.0])
    # The mesh y-center is 0.57, hull x-range after rotation: [-0.024, 0.042] → x unchanged
    # Table y-range: 0.6 ± 0.4 = [0.2, 1.0]. Mesh y at 0.57 → inside table y-range. OK.
    # Table x-range: 0 ± 0.7 = [-0.7, 0.7]. Mesh x near 0 → inside. OK.
    # So the only separating axis is Z. Should work.


def test_support_with_180_quat() raises:
    """Test mesh support with near-180° quaternion."""
    print("=== Test: mesh support with 180° quat ===")
    var mesh_verts = List[Float64]()
    var xs = [-0.024, 0.042]
    var ys = [-0.053, 0.053]
    var zs = [-0.003, 0.051]
    for xi in range(2):
        for yi in range(2):
            for zi in range(2):
                mesh_verts.append(xs[xi])
                mesh_verts.append(ys[yi])
                mesh_verts.append(zs[zi])

    var bqx = 0.02798404808475026
    var bqy = -0.9996283797925901
    var bqz = -0.00045769893422206113
    var bqw = 0.0031081505208131114
    var wx = 0.005
    var wy = 0.601
    var wz = 0.285

    # Support along -Z: should return the lowest mesh vertex in world frame
    var s = support_mesh[DType.float64](
        0.0, 0.0, -1.0,  # dir = -Z
        wx, wy, wz,
        bqx, bqy, bqz, bqw,
        mesh_verts, 0, 8)
    print("  support(-Z) =", Float64(s[0]), Float64(s[1]), Float64(s[2]))
    print("  (mesh center at z=0.285, expected support z > 0.2)")

    # Also test +Z
    var s2 = support_mesh[DType.float64](
        0.0, 0.0, 1.0,
        wx, wy, wz,
        bqx, bqy, bqz, bqw,
        mesh_verts, 0, 8)
    print("  support(+Z) =", Float64(s2[0]), Float64(s2[1]), Float64(s2[2]))

    # Test box support for table along +Z (should be table top at z=0.0)
    # ⚠ `corner` IS MuJoCo'S `obj->vertindex`, in and out. The reference uses
    # that ONE field for both a mesh hull vertex and a box CORNER CODE, and
    # EPA's discrete repeated-support-point break compares it, so `support_box`
    # writes it. A caller that does not care still has to own the slot.
    var box_corner = 0
    var b = support_box[DType.float64](
        0.0, 0.0, 1.0,
        0.0, 0.6, -0.46,
        0.0, 0.0, 0.0, 1.0,
        0.7, 0.4, 0.46,
        box_corner)
    print("  box support(+Z) =", Float64(b[0]), Float64(b[1]), Float64(b[2]),
          "(expected z=0.0)")


def test_exact_sawyer_runtime() raises:
    """Use exact runtime values from Sawyer debug output."""
    print("=== Test: exact Sawyer runtime values ===")

    # eGripperBase hull: 8 corners from bounds
    var mesh_verts = List[Float64]()
    var xs = [-0.024, 0.042]
    var ys = [-0.053, 0.053]
    var zs = [-0.003, 0.051]
    for xi in range(2):
        for yi in range(2):
            for zi in range(2):
                mesh_verts.append(xs[xi])
                mesh_verts.append(ys[yi])
                mesh_verts.append(zs[zi])

    var mv = _mv_tensor(mesh_verts)

    var _ng = _no_graph()

    var _ne = _no_edges()
    var ws = ccd_ws_alloc[DType.float64]()

    # Body 23 runtime values:
    # xpos: (0.0053, 0.6013, 0.3151)
    # xquat: (0.028, -0.9996, -0.0005, 0.003)  (x,y,z,w)
    # Geom local pos: (0, 0, 0.03), local quat: identity
    # World pos after rotation: body_pos + quat_rotate(body_quat, (0,0,0.03))
    from noeira.physics3d.kinematics.quat_math import quat_rotate, quat_mul
    var bqx = 0.02798404808475026
    var bqy = -0.9996283797925901
    var bqz = -0.00045769893422206113
    var bqw = 0.0031081505208131114
    var rotated = quat_rotate[DType.float64](bqx, bqy, bqz, bqw, 0.0, 0.0, 0.03)
    var wx = 0.005334793735032231 + rotated[0]
    var wy = 0.6013011256902497 + rotated[1]
    var wz = 0.3151333362383024 + rotated[2]
    print("  mesh world pos:", wx, wy, wz)
    print("  mesh world quat:", bqx, bqy, bqz, bqw)

    var result = gjk_epa[DType.float64](
        GEOM_MESH,
        wx, wy, wz,
        bqx, bqy, bqz, bqw,
        0.0, 0.0,
        0.0, 0.0, 0.0,
        mv.tt["cpu", NMV, 3](), _ng.tt["cpu", NMV](), _ne.tt["cpu", mesh_max_edge(NMV)](), 0, 8,
        GEOM_BOX,
        0.0, 0.6, -0.46,  # table collision box
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.0,
        0.7, 0.4, 0.46,  # table half-extents
        0, 0,
        ws.tt_dyn["cpu", L_CCD_WS1](L_CCD_WS1()), 0,
    )
    print("  dist=", Float64(result[0]),
          "normal=", Float64(result[4]), Float64(result[5]), Float64(result[6]))
    if result[0] < 0:
        raise Error("FAIL: see the print above")
    else:
        print("  PASS: gap=", Float64(result[0]))


def main() raises:
    test_support_with_180_quat()
    test_exact_sawyer_runtime()
    test_box_box_separated()
    test_sphere_mesh_separated()
    test_mesh_box_sawyer_case()
    test_mesh_box_rotated()
    test_mesh_box_asymmetric()
    test_mesh_box_actual_sawyer()

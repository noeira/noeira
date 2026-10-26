"""The surface the VIEWER draws is the surface the RAYS hit.

`mj_rayHfield` builds a heightfield's triangles for the physics; `draw_heightfield`
builds them for the screen. The two spell the same convention, and if they ever
stop agreeing the failure is silent and bad in a specific way: the viewer exists
to answer "is the model built and posed the way I think", so a drawn surface
half a cell off, or transposed in row/column, looks entirely plausible while
quietly contradicting the terrain the robot stands on.

WHAT THIS GATES. `hfield_surface.mojo` is where the sample positions are
spelled ONCE, and `draw_heightfield` calls it. This file drops a vertical ray
onto nodes computed from THOSE functions and asserts `mj_rayHfield` comes back
at that node's height. Renderer formula in, ray answer out — so a divergence
between the two is a test failure rather than a rendering curiosity.

⚠ VERTICAL RAYS ONLY, AND FROM ABOVE. A node is a shared corner of up to four
triangles, so a grazing ray there is a genuine edge case the reference itself
resolves arbitrarily; straight down hits the interior of a triangle whose
corner is the node, which is unambiguous. The oblique cases are already gated
against MuJoCo by `test_quadruped_escape_vs_dm_control`.

⚠ NON-VACUITY IS THE POINT OF THE SPREAD CHECK. Escape's terrain is a bowl and
its centre is nearly flat; a sweep confined to the middle would compare a
constant against a constant and pass with the row and column swapped. The nodes
below deliberately straddle the rim, and the test ASSERTS the heights it
sampled actually vary before it believes the agreement.

Run with:
    pixi run mojo run -I . tests/physics3d/test_hfield_surface_matches_ray.mojo
"""

from std.math import abs, min, max
from std.testing import assert_true, TestSuite


from noeira.math3d import Vec3 as Vec3Generic
from noeira.envs.dm_control.quadruped import DMQuadrupedEscape
from noeira.envs.dm_control.quadruped.quadruped_escape_config import (
    ESCAPE_TERRAIN_GEOM,
)
from noeira.physics3d.model.hfield_surface import (
    hfield_node_x, hfield_node_y, hfield_node_z,
)
from noeira.physics3d.ray.model import ray_model
from noeira.physics3d.fields import DYN1, DYN2, rl1, rl2
from noeira.physics3d.gpu.constants import (
    MODEL_HFIELD_META_SIZE,
    MAX_GPU_HFIELDS,
    MAX_GPU_MESHES,
    MESH_ARENA_FLOATS_PER_TRI,
    MODEL_MESH_META_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_BODY_SIZE,
    GEOM_IDX_POS_Z,
    HFIELD_META_IDX_ADR,
    HFIELD_META_IDX_NROW,
    HFIELD_META_IDX_NCOL,
    HFIELD_META_IDX_SIZE_X,
    HFIELD_META_IDX_SIZE_Y,
    HFIELD_META_IDX_SIZE_Z,
)

comptime DT = DType.float64

# ⚠ A COARSE STRIDE ACROSS THE WHOLE FIELD, not a dense patch. 201/17 walks the
# centre, both rims and the corners; a dense block would sample one slope.
comptime STRIDE = 17
# Well above `size[2]` = 5 m, so every ray starts outside the top box.
comptime DROP_FROM: Float64 = 40.0
# The elevations are float64 here on both sides and the arithmetic is the same
# handful of operations, so this is a "did the formula change" bound, not a
# numerical one.
comptime TOL: Float64 = 1e-9


def test_the_drawn_nodes_are_where_the_rays_land() raises:
    var env = DMQuadrupedEscape[DT]()
    _ = env.reset()

    var adr = Int(env.mf.hfield_meta.data[HFIELD_META_IDX_ADR])
    var nrow = Int(env.mf.hfield_meta.data[HFIELD_META_IDX_NROW])
    var ncol = Int(env.mf.hfield_meta.data[HFIELD_META_IDX_NCOL])
    var sx = Float64(env.mf.hfield_meta.data[HFIELD_META_IDX_SIZE_X])
    var sy = Float64(env.mf.hfield_meta.data[HFIELD_META_IDX_SIZE_Y])
    var sz = Float64(env.mf.hfield_meta.data[HFIELD_META_IDX_SIZE_Z])
    assert_true(nrow >= 2 and ncol >= 2, "degenerate grid")

    # The renderer takes the grid as a `List[Float64]`; build the same one.
    var grid = List[Float64](capacity=len(env.d.hfield_data.data))
    for i in range(len(env.d.hfield_data.data)):
        grid.append(Float64(env.d.hfield_data.data[i]))

    var ng = env.mf.dims.get_ngeom()
    var nb = env.mf.dims.get_nbody()
    var hfn = env.mf.dims.get_nhfield_data()
    var nmt = env.mf.dims.get_nmesh_tri() * MESH_ARENA_FLOATS_PER_TRI
    if nmt < 1:
        nmt = 1

    # ⚠ THE GEOM'S OWN OFFSET. `hfield_surface` answers in the GEOM's LOCAL
    # frame; the ray answers in the WORLD. escape's terrain sits at
    # `pos="0 0 -.01"`, and comparing the two frames directly reports a clean
    # 1 cm "divergence" at every single node — which is what this gate did on
    # its first run, and is exactly the plausible-looking wrong answer it was
    # written to catch. The terrain is on the worldbody, so its world pose is
    # its local one.
    var terrain_z = Float64(
        env.mf.geoms.data[ESCAPE_TERRAIN_GEOM * MODEL_GEOM_SIZE
                          + GEOM_IDX_POS_Z]
    )

    var n_checked = 0
    var n_missed = 0
    var n_floor = 0
    var worst = 0.0
    var z_lo = 1e30
    var z_hi = -1e30
    var worst_rc = String("")

    for r in range(0, nrow, STRIDE):
        for c in range(0, ncol, STRIDE):
            var px = hfield_node_x(c, ncol, sx)
            var py = hfield_node_y(r, nrow, sy)
            var pz = hfield_node_z(grid, adr, ncol, r, c, sz)

            var hit = ray_model[DT](
                env.mf.geoms.tt_dyn["cpu", DYN2](rl2(ng, MODEL_GEOM_SIZE)),
                ng,
                env.mf.bodies.tt_dyn["cpu", DYN2](rl2(nb, MODEL_BODY_SIZE)),
                env.d.xpos.tt_dyn["cpu", DYN2](rl2(1, nb * 3)),
                env.d.xquat.tt_dyn["cpu", DYN2](rl2(1, nb * 4)),
                0,
                env.mf.mesh_meta.tt_dyn["cpu", DYN1](
                    rl1(MAX_GPU_MESHES * MODEL_MESH_META_SIZE)
                ),
                env.mf.mesh_tris.tt_dyn["cpu", DYN1](rl1(nmt)),
                env.mf.hfield_meta.tt_dyn["cpu", DYN1](
                    rl1(MAX_GPU_HFIELDS * MODEL_HFIELD_META_SIZE)
                ),
                env.d.hfield_data.tt_dyn["cpu", DYN1](rl1(hfn)),
                hfn,
                Vec3Generic[DT](px, py, DROP_FROM),
                Vec3Generic[DT](0.0, 0.0, -1.0),
                # ⚠ EXCLUDE THE QUADRUPED'S ROOT. It is reset standing at the
                # origin, so a ray dropped on a node near the centre would
                # otherwise report a TOE and the terrain would look wrong at
                # exactly the nodes the robot is on.
                1,
            )
            if hit.t < 0:
                n_missed += 1
                continue

            # `vec` is a unit -z, so `t` is the drop in metres.
            # `vec` is a unit -z, so `t` is the drop in metres.
            var z_hit = DROP_FROM - Float64(hit.t)
            var want = pz + terrain_z

            # ⚠ THE FLOOR LEGITIMATELY WINS WHERE THE BOWL DIPS BELOW IT.
            # escape keeps an infinite `<geom type="plane">` at z = 0 AND
            # lowers the terrain 1 cm, so any node whose surface lands at or
            # under zero is occluded by the floor and reports it. Counted and
            # asserted-on below rather than quietly skipped — a sweep that
            # started returning the floor EVERYWHERE would otherwise pass by
            # comparing nothing.
            if hit.geom != ESCAPE_TERRAIN_GEOM:
                n_floor += 1
                continue

            # ⚠ THE SPREAD IS MEASURED OVER COMPARED NODES, NOT SAMPLED ONES.
            # A node the floor occludes contributes no comparison, so counting
            # its elevation toward the "did we sweep real terrain" check would
            # let a run that compared nothing but flat ground still look varied.
            z_lo = min(z_lo, pz)
            z_hi = max(z_hi, pz)

            var d = abs(z_hit - want)
            if d > worst:
                worst = d
                worst_rc = String("(") + String(r) + "," + String(c) + ")"
            n_checked += 1

    print(
        "  nodes compared   ", n_checked, " missed", n_missed,
        " occluded by floor", n_floor,
    )
    print("  terrain geom z   ", terrain_z)
    print("  node z range     [", z_lo, ",", z_hi, "]")
    print("  worst |d z|      ", worst, " at", worst_rc)

    # ── non-vacuity, before the verdict ──────────────────────────────────
    assert_true(
        n_checked > 50,
        "only " + String(n_checked) + " nodes were compared; a handful of"
        " agreements does not gate a 201x201 convention.",
    )
    assert_true(
        n_missed == 0,
        String(n_missed) + " vertical rays hit NOTHING. A ray dropped from"
        " above a heightfield node must strike it — a miss means the drawn"
        " node is outside the surface the ray builds, which is the exact"
        " divergence this file exists to catch.",
    )
    # ⚠ THE BOWL IS LOW AT BOTH ENDS, which is not obvious and cost this gate
    # a false failure. `bowl = .5 - cos(2*pi*r)/2` is ZERO at r = 0 AND at
    # r = 1, peaking at the rim in between — so the flat centre and the whole
    # outer ring beyond the unit disc both sit under the 1 cm the terrain is
    # lowered by, and report the floor. Measured: 43 of 144 nodes, ~30%. The
    # bound is "most nodes still reach the terrain", not "almost all".
    assert_true(
        n_checked * 2 > n_checked + n_floor,
        String(n_floor) + " of " + String(n_checked + n_floor) + " sampled"
        " nodes reported a geom other than the terrain. Under half reaching"
        " the heightfield means the sweep is measuring the floor.",
    )
    assert_true(
        z_hi - z_lo > 0.5,
        "the sampled nodes span only " + String(z_hi - z_lo)
        + " m of elevation. A flat sweep agrees with a TRANSPOSED surface"
        " just as well as with the right one — re-aim the stride.",
    )

    assert_true(
        worst <= TOL,
        "the drawn surface and the ray surface disagree by " + String(worst)
        + " m at node " + worst_rc + ". `hfield_surface.mojo` and"
        " `mj_rayHfield` have drifted.",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()

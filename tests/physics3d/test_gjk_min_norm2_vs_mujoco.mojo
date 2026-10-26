"""GJK's convergence floor is a threshold on |v| SQUARED, so a "tiny" constant
is a LARGE distance — and a separated pair under it is reported as a contact.

    pixi run mojo run -I . tests/physics3d/test_gjk_min_norm2_vs_mujoco.mojo

WHAT WENT WRONG. The loop that ends GJK read

    if v_dot_v < GJK_TOLERANCE:      # GJK_TOLERANCE = 1e-10
        break

and the separated-vs-penetrating classification after it compared `dist_sq`
against the same constant. `v_dot_v` is |v| SQUARED, so 1e-10 is a distance
floor of `sqrt(1e-10)` = **1e-5 m = 10 microns**. Every convex pair separated
by less than 10 microns converged "to the origin", was classified PENETRATING,
and was handed to EPA — which returns `-0.0`. A real gap became an invented
contact, and an invented contact is a constraint row the solver cannot tell
from a true one.

⚠⚠ 1e-10 LOOKS LIKE A TIGHT TOLERANCE. It is not a tolerance on a distance, it
is a tolerance on a distance squared, and nothing at the call site says so.

MUJOCO DOES NOT USE A CONSTANT (`engine_collision_gjk.c:212-225`):

    mjtNum tol2      = status->tolerance * status->tolerance;
    mjtNum min_norm2 = discreteGeoms(obj1, obj2) ? mjMINVAL2 : tol2;
    ...
    if ((x_norm = dot3(x_k, x_k)) < min_norm2) break;

`tolerance` is `<option ccd_tolerance>`, default 1e-6, so the floor is 1e-6 m
for a SMOOTH pair and `mjMINVAL` = 1e-15 m for a POLYTOPE pair — 10x and 1e10x
tighter than the constant we had. It is the SAME `discreteGeoms` switch that
`_epa_tolerance` in `gjk.mojo` already implements; that one crossed to EPA and
never to GJK, and `_gjk_min_norm2` is the other half.

⚠ A NON-ZERO MARGIN MAKES A DISCRETE PAIR SMOOTH — MuJoCo's first line, and
load-bearing here: the duplo stud declares `margin=1e-4`, so stud-vs-flange is
a SMOOTH pair and its floor is `ccd_tolerance`, not `mjMINVAL`.

THE FIXTURE IS THE REAL ONE, AND ITS ANSWER IS EXACT. These are the `stud` and
`flange` geoms of `dm_control`'s `duplo2x4.xml` verbatim, in the configuration
`manipulation/reassemble_5_bricks_random_order` starts in. The cylinder's axis
is z and the box's nearest feature is its +x FACE, so the signed gap is

    (box_face_x - cyl_axis_x) - r

with NO approximation and no reference implementation in the loop. In the model
that is `0.00465 - 0.004647` = **3.0 microns** — a fifth of the old floor.
MuJoCo reports +3.02e-06 there; we reported -6.33e-06 on all 48 such pairs,
penetration where there is a gap.

⚠ THE CONTROLS ARE IN THE TABLE, AND THERE ARE TWO OF THEM.

(a) **The reference's own answer, per row.** `mjc_penetration`
(`engine_collision_convex.c`) is what the pipeline runs, and it does NOT ask
for a distance:

    config.dist_cutoff = 0;  // no geom distances needed
    if ((dist = mjc_ccd(&config, &status, obj1, obj2)) < 0) {
      for (int i = 0; i < nwitness; i++) {
        con[i].dist = margin + dist;

Both geoms are inflated by half the margin in `mjc_initCCDObj`, EPA returns a
penetration, and `margin + dist` is the number reported. Every witness of the
manifold gets the SAME one. That is the question `gjk_epa` implements here, so
that is the number to compare against — measured on MuJoCo 3.10.0 with
`/tmp/gjkref.py`, reproduced in the table below.

(b) **The tolerance is the residual, and tightening it converges.** At
`ccd_tolerance = 1e-14` this fixture lands within 5e-15 of the ANALYTIC gap on
every row. That is the control which says the offset at 1e-6 is EPA's stopping
rule and not an algorithmic error — the same rule, and the same offset, that
the reference is subject to.

⚠⚠ THE OLD `2.0e-05 / tol 1e-9` ROW WAS A FROZEN BASELINE AND `d67673df`
INVALIDATED IT. That row asserted "a change confined to the sub-floor band does
not move a row above the band", against the value the engine produced when this
file was written: measured at `d67673df^`, 2.027e-13. `d67673df` — `mjc_Convex`
INFLATES both geoms by the margin and asks PENETRATION, it never runs a
distance query — deliberately changed the arithmetic for EVERY row, and the row
went to 2.445e-09 (and to 4.104e-09 at `fee37034`, the 1:1 GJK/EPA port). The
constant outlived the thing it was pinning. It is replaced above by (a) and
(b), both of which name a reference instead of a remembered number.

⚠⚠ AND THIS CLOSES THE SECOND HALF OF TASK #81, WHICH THE DOC STILL LISTS AS
OPEN. It was filed as "insensitive to `ccd_tolerance` (1e-6..1e-14) ... it
lives in the distance subalgorithm / `_closest_point_on_simplex`". Today it is
ENTIRELY sensitive to `ccd_tolerance` — the widest row goes 3.482e-07 (1e-6) ->
8.217e-10 (1e-8) -> 2.878e-12 (1e-10) -> 2.584e-15 (1e-14) — and
`_closest_point_on_simplex` no longer exists; `fee37034` replaced it with the
reference's `subdistance`. The measurement that filed it predates both.

⚠ EVERY OTHER COLLISION GATE PASSED WITH THE BUG IN. `test_gjk_simplex`,
`test_gjk_float32_no_phantom_contacts`, `test_within_margin_convex_contacts`,
`test_mesh_collision`, `test_mesh_manifold_vs_mujoco`, `test_narrow_phase_pairs`,
`test_epa_optimality_cylinder_mesh`, `test_box_box_sweep`,
`test_capsule_box_sweep`, `test_sawyer_mesh_rest_vs_mujoco` and
`test_mesh_polygons_vs_mujoco` are all GREEN both before and after. None of
them poses a convex pair separated by less than 10 microns.
"""
from std.math import abs
from layout import TileTensor
from noeira.core.tiles import RowMajor, tile

from noeira.physics3d.collision.ccd_workspace import L_CCD_WS1
from noeira.physics3d.collision.ccd_workspace_host import ccd_ws_alloc
from noeira.physics3d.collision.gjk import gjk_epa
from noeira.physics3d.constants import GEOM_BOX, GEOM_CYLINDER

comptime DT = DType.float64
comptime LV = RowMajor[1, 3]
comptime LA = RowMajor[1]

# `duplo2x4.xml`, class `stud`: cylinder, size ".0047 .0023", margin 1e-4. The
# radius is 0.004647 and NOT the 0.0047 in the file: dm_control's `Duplo`
# overwrites it in `initialize_episode_mjcf` from `_STUD_SIZE_PARAMS`, and with
# `variation=0` that is the lower quartile for (easy_align=False, flanges=True).
comptime R_STUD = 0.004647
comptime HL_STUD = 0.0023
# class `flange`: box, size ".0008 .00055 .0087".
comptime HX_FLANGE = 0.0008
comptime HY_FLANGE = 0.00055
comptime HZ_FLANGE = 0.0087
# z offset between the two in the assembled tower: the flange belongs to the
# brick above, whose origin sits 0.0192 higher.
comptime DZ = 0.00655
comptime MARGIN = 1e-4
comptime CCD_TOL = 1e-6
comptime CCD_ITER = 35


def probe(
    gap: Float64,
    ccd_tol: Float64,
    mv: TileTensor[DT, LV, MutAnyOrigin],
    ma: TileTensor[DT, LA, MutAnyOrigin],
    me: TileTensor[DT, LA, MutAnyOrigin],
    ws: TileTensor[DT, L_CCD_WS1, MutAnyOrigin],
) -> Float64:
    """Signed distance our narrow phase reports for a stud/flange pair whose
    true face-to-surface gap is `gap`."""
    var dx = Scalar[DT](gap + HX_FLANGE + R_STUD)
    var res = gjk_epa[DT](
        GEOM_CYLINDER,
        Scalar[DT](0), Scalar[DT](0), Scalar[DT](0),
        Scalar[DT](0), Scalar[DT](0), Scalar[DT](0), Scalar[DT](1),
        Scalar[DT](R_STUD), Scalar[DT](HL_STUD),
        Scalar[DT](0), Scalar[DT](0), Scalar[DT](0),
        mv, ma, me, 0, 0,
        GEOM_BOX,
        dx, Scalar[DT](0), Scalar[DT](DZ),
        Scalar[DT](0), Scalar[DT](0), Scalar[DT](0), Scalar[DT](1),
        Scalar[DT](0), Scalar[DT](0),
        Scalar[DT](HX_FLANGE), Scalar[DT](HY_FLANGE), Scalar[DT](HZ_FLANGE),
        0, 0,
        ws, 0,
        Scalar[DT](ccd_tol), CCD_ITER, Scalar[DT](MARGIN),
    )
    return Float64(res[0])


def main() raises:
    var mbuf = List[Scalar[DT]](length=3, fill=Scalar[DT](0))
    var mv = tile[1, 3](mbuf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]())
    var abuf = List[Scalar[DT]](length=1, fill=Scalar[DT](-1))
    var ma = tile[1](abuf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]())
    var ebuf = List[Scalar[DT]](length=1, fill=Scalar[DT](-1))
    var me = tile[1](ebuf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]())

    # (true gap, MuJoCo 3.10.0's `contact.dist` for the same pair). Every gap
    # here is UNDER the old 1e-5 floor except the last two.
    #
    # ⚠ THE REFERENCE COLUMN IS NOT THE TRUE GAP AND IS NOT MEANT TO BE. It is
    # `margin + mjc_ccd(inflated pair)` at `ccd_tolerance = 1e-6`, which sits
    # 2.5e-07 to 7.9e-07 ABOVE the analytic gap — the reference's own stopping
    # rule. Reproduce with `mj_forward` on two bodies holding this cylinder and
    # this box at `dx = gap + HX_FLANGE + R_STUD`, `margin = 1e-4`; all five
    # witnesses of the manifold carry the same number.
    var gaps = List[Float64]()
    var mjref = List[Float64]()
    gaps.append(2.0e-06); mjref.append(2.2505889437604055e-06)
    gaps.append(3.0e-06); mjref.append(3.2541808316072902e-06)  # the model's own
    gaps.append(4.0e-06); mjref.append(4.2577998470886235e-06)
    gaps.append(5.0e-06); mjref.append(5.7863508672476656e-06)
    gaps.append(7.0e-06); mjref.append(7.7735065488386090e-06)
    gaps.append(1.0e-05); mjref.append(1.0755194628279301e-05)
    gaps.append(2.0e-05); mjref.append(2.0702430974776909e-05)
    gaps.append(5.0e-05); mjref.append(5.0619944637577617e-05)

    var wst = ccd_ws_alloc[DT]()
    var failures = 0

    # ---- (1) no phantom contacts, and (2) inside the reference's tolerance --
    print("  true gap          ours              MuJoCo 3.10.0     |ours-mj|")
    for i in range(len(gaps)):
        var g = gaps[i]
        var got = probe(g, CCD_TOL, mv, ma, me, wst.tt_dyn["cpu", L_CCD_WS1](L_CCD_WS1()))
        var dref = abs(got - mjref[i])
        # A gap is a gap: anything <= 0 here is the defect this file exists for.
        var ok = got > 0.0 and dref <= CCD_TOL
        print("  ", g, " ", got, " ", mjref[i], " ", dref, " ",
              "ok" if ok else "FAIL")
        if not ok:
            failures += 1
            if got <= 0.0:
                print(
                    "     ^ a PHANTOM CONTACT: a pair", g,
                    "m apart reported as touching/penetrating",
                )
            else:
                print(
                    "     ^ further from `mjc_penetration` than the",
                    "`ccd_tolerance` both engines are run at",
                )

    # ---- (3) the residual IS the tolerance: tighten it and it converges -----
    print("")
    print("  control: ccd_tolerance 1e-14, error against the ANALYTIC gap")
    for i in range(len(gaps)):
        var g = gaps[i]
        var tight = probe(g, 1e-14, mv, ma, me, wst.tt_dyn["cpu", L_CCD_WS1](L_CCD_WS1()))
        var err = abs(tight - g)
        var ok = err <= 1e-12
        print("  ", g, " ", tight, " ", err, " ", "ok" if ok else "FAIL")
        if not ok:
            failures += 1
            print(
                "     ^ tightening `ccd_tolerance` did NOT converge, so this",
                "residual is NOT the stopping rule — see task #81",
            )

    _ = mbuf^
    _ = abuf^
    _ = ebuf^

    if failures != 0:
        raise Error(
            String(failures)
            + " of "
            + String(2 * len(gaps))
            + " rows wrong — GJK's floor is a threshold on |v| SQUARED;"
            + " see `_gjk_min_norm2` in physics3d/collision/gjk.mojo"
        )
    print("PASS: all", len(gaps), "separations match MuJoCo, all converge")

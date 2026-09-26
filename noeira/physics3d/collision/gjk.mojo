"""GJK/EPA for mesh collision over per-field tensors (migration P4).

Per-field port of gjk_gpu.mojo — arithmetic verbatim. The only change is
the mesh-vertex operand: instead of the flat model slab
(`model[0, mesh_vert_buf_off + i*3 + k]`) the functions take the
`mesh_verts` record tensor (`[NMESH_VERTS, 3]`, see Model) and a
vertex START index (`vert_adr`, MuJoCo `mesh_vertadr`) so reads become
`mesh_verts[vert_adr + i, k]`. Same floats, same iteration order.
"""

from std.time import perf_counter_ns
from std.ffi import _Global
from std.os import abort
from std.math import sqrt, abs
from layout import Layout, LayoutTensor
from .ccd_workspace import (
    HW_WS_OFF,
    EPA_ITER_CAP,
    EPA_V_CAP,
    EPA_F_CAP,
    CCD_WS_SIZE,
    CCD_WS_SPX,
    CCD_WS_SPX2,
    SPX_STRIDE,
    EPA_V_STRIDE,
)
from .epa import (
    ev,
    set_ev,
    sv,
    set_sv,
    ef,
    eadj,
    set_eadj,
    efv,
    efd,
    efi,
    set_efi,
    emap,
    set_emap,
    ehor,
    set_center,
    attach_face,
    horizon,
    epa_witness,
    tri_point_intersect,
    test_tetra,
    rotmat120,
    ray_triangle,
    epa_minval,
    epa_mindist,
    epa_mindist4,
)
from .gjk_support import _subdistance
from .gjk_support import (
    quat2mat,
    mat_t_vec3,
    local_to_global,
    support_sphere,
    support_capsule,
    support_box,
    support_cylinder,
    support_ellipsoid,
    support_prism,
)
from ..constants import (
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_BOX,
    GEOM_CYLINDER,
    GEOM_ELLIPSOID,
    GEOM_HFIELD,
    GEOM_MESH,
)
from ..kinematics.quat_math import quat_rotate, quat_rotate_inverse
from ..gpu.constants import (
    MJ_CCD_TOLERANCE,
    MJ_CCD_ITERATIONS,
    mesh_max_edge,
)

# Reuse CPU GJK parameters (verbatim from gjk_gpu.mojo)
comptime GJK_MAX_ITERATIONS: Int = 100
comptime GJK_TOLERANCE: Float64 = 1e-10

# ⚠⚠ `EPA_MAX_ITERATIONS` AND `EPA_TOLERANCE` ARE GONE. They were 64 and 1e-8
# — MuJoCo's are `opt.ccd_iterations` = 35 and `opt.ccd_tolerance` = 1e-6, and
# a model that set either was ignored. They now arrive as runtime arguments
# (`ccd_iter`, `ccd_tol`) carried in model META, defaulting to MuJoCo's, which
# is why the constants are DELETED rather than retuned: anything still reading
# them fails to compile instead of quietly keeping the old stopping rule.
#
# ⚠ TIGHTER WAS NOT SAFER. EPA's stopping rule picks WHICH boundary face it
# settles on and the contact NORMAL is that face's, so iterating past the
# reference walks away from it rather than toward it.

# `mjMINVAL` (`mjtnum.h`).
comptime MJ_MINVAL: Float64 = 1e-15


# ⚠ HOW TO SEE INSIDE THE METAL KERNEL. With this on, `gjk_epa_witness`
# returns EPA's own counters — `(nverts, nfaces, face)` where the position goes
# and `(i0, i1, i2)` where the normal goes — so a GPU run can be read through
# the DOWNLOADABLE contact record. `hfield_convex.mojo` reads the same flag and
# writes its sub-grid walk into the record's force slots.
#
# ⚠⚠ IT LIES ABOUT THE CONTACT WHILE IT IS ON. Off in every committed state.
#
# It exists because `print` does not, and because overwriting expendable
# columns of a record the kernel already writes is the ONLY technique that has
# ever worked on this target — see
# `feedback_metal_wide_per_thread_inlinearray_miscompute`, which was found that
# way twice, and `HF_DEBUG` next door, which exists for the same reason.
#
# ⚠ AND TURNING IT ON IS ITSELF A MEASUREMENT. If the contact COUNT moves when
# this flag moves, the kernel is miscompiled: the count is decided by `dist`,
# which does not depend on the witness, so dead-coding `epa_witness` cannot
# change it. That is exactly how the heightfield GPU defect was caught — 11
# contacts with the flag off, 16 with it on, 15 and 15 after the simplex moved
# off the per-thread stack.
comptime EPA_DBG: Bool = False


# ⚠ A PHASE PROBE FOR `gjk_epa_witness`, off and free by default (the fourth of
# the family: `_CPU_PROBE`, `_EULER_PROBE`, `_COLL_PROBE`). On, every call prints
# one `[gprobe]` line: `sep` 1 for a separated exit, then nanoseconds and
# iteration counts for the GJK loop, the polytope construction and the EPA
# loop. Every timer block holds only timer lines (PERFORMANCE.md §13.16).
comptime _GJK_PROBE: Bool = False


# ⚠ A WALK-LENGTH PROBE FOR THE MESH HILL CLIMB, off and free by default (the
# fifth of the family). On, `hillclimb_support_index` counts its calls and
# the neighbourhood scans each walk takes (one scan = one dependent chain of
# global loads on the GPU), split cold (no usable warm vertex) / warm, and —
# when the driver has filled `ext` with a mesh's 27 grid extrema (MuJoCo
# 3.12's `mesh_extrema`, commit 83e621d7) — replays the SAME walk from
# 3.12's seed (the better of the warm vertex and the extremum of the query
# direction quantised onto the (-1,0,1)^3 grid) and counts THAT walk's
# scans and whether it landed on the same vertex. CPU only (`_Global`).
comptime _HILL_PROBE: Bool = False


struct _HillProbe(Movable):
    var calls: Int
    var scans: Int
    var cold_calls: Int
    var cold_scans: Int
    var linear: Int
    """Calls that fell back to the linear scan (no graph / tiny mesh)."""
    var seeded_scans: Int
    var seeded_cold_scans: Int
    var seed_beat_warm: Int
    var seed_mismatch: Int
    """Seeded walk landed on a different vertex (a tie, or a defect)."""
    var seed_mismatch_nontie: Int
    """...and the two landings' dots differ by more than 1e-6 relative."""
    var ordinal: Int
    """Call ordinal within the step; the driver zeroes it each step."""
    var prev_land: List[Int]
    """Landing vertex of the same ordinal's call in the previous step."""
    var xstep_scans: Int
    """Scans a walk from `prev_land` (cross-step warm start) would take."""
    var xstep_calls: Int
    var plane_calls: Int
    """Calls from the plane-mesh path (`_plane_mesh_contacts`)."""
    var ext_keys: List[Int]
    """`vert_adr` of each mesh with extrema in `ext` (27 entries each)."""
    var ext: List[Int]

    def __init__(out self):
        self.calls = 0
        self.scans = 0
        self.cold_calls = 0
        self.cold_scans = 0
        self.linear = 0
        self.seeded_scans = 0
        self.seeded_cold_scans = 0
        self.seed_beat_warm = 0
        self.seed_mismatch = 0
        self.seed_mismatch_nontie = 0
        self.ordinal = 0
        self.prev_land = List[Int]()
        for _ in range(256):
            self.prev_land.append(-1)
        self.xstep_scans = 0
        self.xstep_calls = 0
        self.plane_calls = 0
        self.ext_keys = List[Int]()
        self.ext = List[Int]()

    def reset_counts(mut self):
        self.calls = 0
        self.scans = 0
        self.cold_calls = 0
        self.cold_scans = 0
        self.linear = 0
        self.seeded_scans = 0
        self.seeded_cold_scans = 0
        self.seed_beat_warm = 0
        self.seed_mismatch = 0
        self.seed_mismatch_nontie = 0
        self.xstep_scans = 0
        self.xstep_calls = 0
        self.plane_calls = 0

    def ext_offset(self, vert_adr: Int) -> Int:
        for i in range(len(self.ext_keys)):
            if self.ext_keys[i] == vert_adr:
                return i * 27
        return -1


def _init_hill_probe() -> _HillProbe:
    return _HillProbe()


comptime _HILL_PROBE_SLOT = _Global[
    "noeira.physics3d.collision.hill_probe", _init_hill_probe
]


def hill_probe() -> Pointer[_HillProbe, MutUntrackedOrigin]:
    try:
        return _HILL_PROBE_SLOT.get_or_create_ptr()
    except:
        abort("hill_probe: the compiler runtime refused the global slot")


@always_inline
def _gjk_min_norm2[
    DTYPE: DType
](
    type1: Int,
    type2: Int,
    margin: Scalar[DTYPE],
    ccd_tol: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """MuJoCo's `min_norm2` — the floor on |v|^2 that ends the GJK loop.

        mjtNum tol2 = status->tolerance * status->tolerance;
        mjtNum min_norm2 = discreteGeoms(obj1, obj2) ? mjMINVAL2 : tol2;
        ...
        if ((x_norm = dot3(x_k, x_k)) < min_norm2) break;

    (`engine_collision_gjk.c:212-225`.) It is the SAME `discreteGeoms` switch
    `_epa_tolerance` already implements — that one crossed to EPA and never to
    GJK, and this is the other half.

    ⚠⚠ IT IS A THRESHOLD ON |v| SQUARED, WHICH IS WHY THE OLD CONSTANT LOOKED
    HARMLESS. `GJK_TOLERANCE = 1e-10` reads as a tight tolerance; against a
    SQUARED norm it is a distance floor of `sqrt(1e-10)` = **1e-5 m**. Every
    convex pair separated by less than 10 microns was classified PENETRATING
    and handed to EPA, which returned `-0.0`: an invented contact at a real
    gap. MuJoCo's floor is `ccd_tolerance` = 1e-6 m for a smooth pair and
    `mjMINVAL` = 1e-15 m for a polytope pair — 10x and 1e10x tighter.

    ⚠ MEASURED, dm_control `manipulation/reassemble_5`. The Duplo stud sits
    3.0 microns from the brick flange above it (exact: the stud axis is
    0.00465 from the flange face and its radius is 0.004647). MuJoCo reports
    +3.02e-06 on all 48 such pairs; we reported -6.33e-06 — penetration where
    there is a gap — on all 48, because 3e-06 is under the 1e-5 floor.

    ⚠ THE CLASSIFICATION AFTER THE LOOP MUST USE THE SAME VALUE. `dist_sq >
    min_norm2` is what decides separated-vs-penetrating; if it disagrees with
    the loop's break test, a run that converged to the origin is then reported
    separated at a hair's-breadth distance (see the note at that branch).
    """
    var tol2 = ccd_tol * ccd_tol
    if margin != 0:
        return tol2
    # ⚠⚠ HFIELD IS IN `discreteGeoms` AND WAS MISSING FROM ALL THREE COPIES:
    #   (g1 == mjGEOM_MESH || g1 == mjGEOM_BOX || g1 == mjGEOM_HFIELD) && ...
    # A heightfield prism IS a polytope, so it belongs on the discrete side of
    # every one of these switches. Fourth instance of
    # `feedback_a_geom_type_absent_from_three_fallbacks`.
    var d1 = (
        type1 == GEOM_BOX or type1 == GEOM_MESH or type1 == GEOM_HFIELD
    )
    var d2 = (
        type2 == GEOM_BOX or type2 == GEOM_MESH or type2 == GEOM_HFIELD
    )
    if d1 and d2:
        return Scalar[DTYPE](MJ_MINVAL) * Scalar[DTYPE](MJ_MINVAL)
    return tol2


@always_inline
def _gjk_epsilon[
    DTYPE: DType
](
    type1: Int,
    type2: Int,
    margin: Scalar[DTYPE],
    ccd_tol: Scalar[DTYPE],
    v_dot_v: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """MuJoCo's `epsilon` — the Frank-Wolfe duality-gap floor that ends GJK.

        mjtNum tol2 = status->tolerance * status->tolerance;
        mjtNum epsilon = discreteGeoms(obj1, obj2) ? 0 : 0.5 * tol2;
        ...
        sub3(diff, x_k, s_k);
        if (dot3(x_k, diff) < epsilon) { if (!k) n = 1; break; }

    ⚠⚠ THE QUANTITY IS `x . (x - s)`, IN METRES SQUARED, AND OURS WAS A
    THRESHOLD ON THE SAME GAP DIVIDED BY `|x|`. The old test was
    `w_dot - v_dot < GJK_TOLERANCE` with `GJK_TOLERANCE = 1e-10` — a hardcoded
    constant in METRES, where the reference's is `0.5 * ccd_tolerance^2` and
    is ZERO for a polytope pair. The two agree only where `|x|` happens to be
    5 mm; at 0.5 mm ours was ten times tighter and at 5 cm ten times looser.

    ⚠ AND THE CONSEQUENCE IS NOT A ROUNDING DIFFERENCE. This test fires BEFORE
    `gjkIntersect` gets a chance to run, so stopping one iteration early hands
    EPA a TRIANGLE where the reference hands it a TETRAHEDRON — `polytope3`
    instead of `polytope4`, a different seed, a different final face and a
    different contact normal. Measured on the `box/cylinder` group of
    `test_narrow_phase_pairs`: the normal sat 1.7e-05 off MuJoCo's with the
    old rule and lands within `TOL_DIR_APPROX` with this one.

    ⚠ THE FLOAT32 TERM IS KEPT AND IS RELATIVE. `x . (x - s)` is a difference
    of dot products of magnitude `|x|^2`, so at float32 its rounding floor is
    about `1e-7 * |x|^2` — far above `0.5 * tol^2`. Without the relative term
    the comparison never succeeds, GJK runs to its cap and returns whatever it
    is holding, which produced PHANTOM CONTACTS at 17 cm
    (`test_gjk_float32_no_phantom_contacts`). It is exactly the old relative
    term re-expressed in the new units: `1e-6 * |x|` on the divided gap is
    `1e-6 * |x|^2` on this one.
    """
    var eps = Scalar[DTYPE](0)
    var discrete = margin == 0 and (
        (type1 == GEOM_BOX or type1 == GEOM_MESH or type1 == GEOM_HFIELD)
        and (type2 == GEOM_BOX or type2 == GEOM_MESH or type2 == GEOM_HFIELD)
    )
    if not discrete:
        eps = Scalar[DTYPE](0.5) * ccd_tol * ccd_tol
    comptime if DTYPE != DType.float64:
        eps += Scalar[DTYPE](1e-6) * v_dot_v
    return eps


@always_inline
def _epa_tolerance[
    DTYPE: DType
](
    type1: Int,
    type2: Int,
    margin: Scalar[DTYPE],
    ccd_tol: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """MuJoCo's `discreteGeoms` switch: `ccd_tolerance` is for SMOOTH pairs.

    `epa()` opens with

        int discrete = discreteGeoms(obj1, obj2);
        if (discrete && sizeof(mjtNum) == sizeof(double)) tolerance = mjMINVAL;

    and `discreteGeoms` is "both geoms are MESH/BOX/HFIELD **and** both margins
    are zero" (`engine_collision_gjk.c:159`). A polytope pair has finitely many
    faces, so EPA lands on the exact one and there is nothing for a tolerance to
    trade off; a curved surface has no exact face, so `ccd_tolerance` decides
    where to stop.

    ⚠⚠ MISSING THIS MAKES `ccd_tolerance` A REGRESSION RATHER THAN A FIX, and
    that is how it was found. Wiring `opt.ccd_tolerance` through and applying it
    to EVERYTHING moved mesh-vs-box depth from 1e-16 to 4.98e-7 against MuJoCo
    — inside `ccd_tolerance` and therefore invisible to any check phrased in
    those terms, but a 5e-7 regression on pairs that were EXACT.
    `test_mesh_manifold_vs_mujoco`'s `TOL_DIST = 1e-9` caught it, and the
    tempting response — relax the gate, the number is under 1e-6 — would have
    locked the regression in.

    ⚠ A NON-ZERO MARGIN MAKES A DISCRETE PAIR SMOOTH. That is MuJoCo's first
    line, not an afterthought: margin inflates each geom by a rounded offset
    surface, so the Minkowski boundary stops being piecewise planar.

    ⚠ DOUBLE PRECISION ONLY, matching the `sizeof(mjtNum)` guard. At float32,
    1e-15 is far below epsilon: the convergence test could never fire and EPA
    would run to its iteration cap on every mesh pair.
    """
    if margin != 0:
        return ccd_tol
    # ⚠⚠ HFIELD IS IN `discreteGeoms` AND WAS MISSING FROM ALL THREE COPIES:
    #   (g1 == mjGEOM_MESH || g1 == mjGEOM_BOX || g1 == mjGEOM_HFIELD) && ...
    # A heightfield prism IS a polytope, so it belongs on the discrete side of
    # every one of these switches. Fourth instance of
    # `feedback_a_geom_type_absent_from_three_fallbacks`.
    var d1 = (
        type1 == GEOM_BOX or type1 == GEOM_MESH or type1 == GEOM_HFIELD
    )
    var d2 = (
        type2 == GEOM_BOX or type2 == GEOM_MESH or type2 == GEOM_HFIELD
    )
    if not (d1 and d2):
        return ccd_tol
    # `mjMINEPATOL` — 1e-15 at double, 1e-7 at single. The precision split is
    # the reference's own (`engine_collision_gjk.c`'s `mjUSESINGLE` block) and
    # NOT a guard against the switch itself: at float32, 1e-15 is far below the
    # gap between `upper` and `lower` can ever close to, so the discrete
    # convergence test would never fire and EPA would run to its cap on every
    # mesh pair. This used to return `ccd_tol` at float32, which is the OTHER
    # error — it applied the smooth tolerance to a polytope pair.
    comptime if DTYPE == DType.float64:
        return Scalar[DTYPE](MJ_MINVAL)
    else:
        return Scalar[DTYPE](1e-7)


@always_inline
def _dot3[
    DTYPE: DType
](
    ax: Scalar[DTYPE],
    ay: Scalar[DTYPE],
    az: Scalar[DTYPE],
    bx: Scalar[DTYPE],
    by: Scalar[DTYPE],
    bz: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    return ax * bx + ay * by + az * bz


# =============================================================================
# EPA polytope helpers
#
# Fixed-size, non-recursive, no dynamic allocation — all three are hard
# requirements for the GPU path, and MuJoCo's own EPA satisfies none of them
# (`horizonRec` at `engine_collision_gjk.c` recurses, and the polytope is heap
# allocated). The horizon is therefore computed by an UNDIRECTED edge count
# instead of a recursive walk: an edge belonging to exactly one visible face is
# on the horizon. Same horizon, no stack. See the note at the loop for why the
# directed formulation — which needs consistently wound faces — is wrong here.
#
# WHAT THIS IS AND IS NOT. EPA is exact wherever it can seed: on a cylinder
# against a box face, and on the same box expressed as an 8-vertex MESH, it
# reproduces the analytic depth BIT-FOR-BIT at every penetration from 5e-4 to
# 0.03, where the placeholder it replaced returned ~-1.1 throughout. It does
# `polytope2` / `polytope3` / `polytope4` are all ported now, so a degenerate
# GJK simplex is handled the way the reference handles it rather than by an
# invented fallback — see the EPA phase below.
#
# ⚠ THE CAPS ARE MuJoCo'S OWN ALLOCATION FORMULA and live in `ccd_workspace`
# beside the row layout they size: `EPA_V_CAP = 5 + N`, `EPA_F_CAP = 6 * N`
# for `N = EPA_ITER_CAP`, exactly as `mjc_ccdSize` carves the buffer. Overflow
# is REPORTED (the routine keeps its best face so far, which is a valid lower
# bound) rather than silently truncating the polytope.
#
# ⚠ THE POLYTOPE ITSELF IS NOT A LOCAL. It lives in a `[BATCH, CCD_WS_SIZE]`
# tensor row, one row per env, handed in as `(ws, wrow)` — MuJoCo's
# `config->buffer`. See `ccd_workspace.mojo` for why that is the reference's
# storage class and not a workaround.



@always_inline
def _sel4i(i: Int, a: Int, b: Int, c: Int, d: Int) -> Int:
    """Pick one of four `Int`s by a RUNTIME index, as an if-chain.

    ⚠ THIS EXISTS INSTEAD OF AN `Array[Int, 4]`. Indexing a per-thread
    array by a runtime value reads back the wrong value on Metal with no crash
    — see `feedback_metal_wide_per_thread_inlinearray_miscompute`, whose second
    instance was a THREE-element array.
    """
    if i == 0:
        return a
    if i == 1:
        return b
    if i == 2:
        return c
    return d


@always_inline
def _gjk_signed_distance[
    DTYPE: DType, L_WS: Layout
](
    ws: LayoutTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    base: Int,
    i1: Int, i2: Int, i3: Int,
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Signed distance from the origin to the plane (v1, v2, v3), and its normal.

    Port of `signedDistance` (`engine_collision_gjk.c:375`). The normal is
    `cross(v3 - v1, v2 - v1)` normalised, and the distance is `n . v1`, so the
    SIGN carries the tetrahedron's orientation — which is the whole mechanism
    `_gjk_intersect` relies on. A degenerate face returns a huge distance so it
    loses every minimum comparison, exactly as MuJoCo's `mjMAX_LIMIT` does.
    """
    var ax = sv(ws, wrow, base, i1, 0)
    var ay = sv(ws, wrow, base, i1, 1)
    var az = sv(ws, wrow, base, i1, 2)
    var d1x = sv(ws, wrow, base, i3, 0) - ax
    var d1y = sv(ws, wrow, base, i3, 1) - ay
    var d1z = sv(ws, wrow, base, i3, 2) - az
    var d2x = sv(ws, wrow, base, i2, 0) - ax
    var d2y = sv(ws, wrow, base, i2, 1) - ay
    var d2z = sv(ws, wrow, base, i2, 2) - az
    var nx = d1y * d2z - d1z * d2y
    var ny = d1z * d2x - d1x * d2z
    var nz = d1x * d2y - d1y * d2x
    var n2 = nx * nx + ny * ny + nz * nz
    if n2 > Scalar[DTYPE](1e-30) and n2 < Scalar[DTYPE](1e30):
        var inv = Scalar[DTYPE](1) / sqrt(n2)
        nx *= inv
        ny *= inv
        nz *= inv
        return (nx * ax + ny * ay + nz * az, nx, ny, nz)
    return (Scalar[DTYPE](1e30), Scalar[DTYPE](0), Scalar[DTYPE](0),
            Scalar[DTYPE](0))




# MuJoCo's `mjMESH_HILLCLIMB_MIN` (`engine_collision_convex.h`): below this many
# vertices the linear scan beats walking the graph, and MuJoCo keeps the scan.
comptime _HILLCLIMB_MIN: Int = 10


@always_inline
def hillclimb_support_index[
    DTYPE: DType,
    L_MESH_VERTS: Layout,
    L_MESH_VERT_EDGEADR: Layout,
    L_MESH_EDGES: Layout,
](
    ld_x: Scalar[DTYPE],
    ld_y: Scalar[DTYPE],
    ld_z: Scalar[DTYPE],
    mesh_verts: LayoutTensor[
        DTYPE, L_MESH_VERTS, MutAnyOrigin
    ],
    mesh_vert_edgeadr: LayoutTensor[
        DTYPE, L_MESH_VERT_EDGEADR, MutAnyOrigin
    ],
    mesh_edges: LayoutTensor[
        DTYPE, L_MESH_EDGES, MutAnyOrigin
    ],
    vert_adr: Int,
    num_verts: Int,
    warm: Int,
) -> Int:
    """LOCAL index of the hull vertex maximising `dot(vertex, ld)`, or -1.

    `ld` is the direction in the MESH'S OWN frame — callers holding a world or
    plane-frame direction must rotate it in first (`quat_rotate_inverse`).

    -1 means "this mesh has no usable graph": either it is below
    `_HILLCLIMB_MIN` vertices or its adjacency is absent, and the caller must
    fall back to a linear scan. Returning a sentinel rather than silently
    scanning here keeps the fallback visible at each call site, which matters
    because the two call sites want DIFFERENT things from it — `_support_mesh`
    needs the point, `_plane_mesh_contacts` needs the index AND the heights it
    computes along the way.

    ⚠⚠ EXTRACTED SO THERE IS ONE HILL CLIMB, NOT TWO. `_plane_mesh_contacts`
    reproduced this function's job as a full linear argmin over every hull
    vertex — the very scan this replaced inside GJK — while its NEIGHBOUR walk
    already used the graph. Measured with per-stage timers, that one scan was
    8.1 µs/step on SO-ARM100 and 11.1 µs/step on SO-ARM101, i.e. 23-25% of the
    whole physics step, and on SO-101 it was a SINGLE call per step. MuJoCo
    calls `mjccd_support` there for exactly this reason
    (`mjc_PlaneConvex`, `engine_collision_convex.c:1010`).
    """
    var graph_head = Int(rebind[Scalar[DTYPE]](mesh_vert_edgeadr[vert_adr]))
    if num_verts < _HILLCLIMB_MIN or graph_head < 0:
        comptime if _HILL_PROBE:
            hill_probe()[].linear += 1
        return -1
    comptime if _HILL_PROBE:
        var hp0 = hill_probe()
        hp0[].calls += 1
        if not (warm >= 0 and warm < num_verts):
            hp0[].cold_calls += 1

    # Greedy walk. `imax` is a LOCAL vertex index; `mesh_edges` holds
    # GLOBAL ones, so neighbours are converted on the way in.
    # ⚠ THE GUARD IS NOT PARANOIA ABOUT CALLERS, IT IS THE THING THAT MAKES
    # A CROSSED INDEX HARMLESS. `warm` arrives as -1 on the first call of a
    # pair, and a caller that threaded object 1's index into object 2 would
    # otherwise walk off the end of a smaller mesh into whatever vertices
    # follow it in the model-wide slab. Clamping to 0 turns that into lost
    # speed rather than a support point belonging to another geom.
    var imax = warm if (warm >= 0 and warm < num_verts) else 0
    var best_dot = (
        ld_x * rebind[Scalar[DTYPE]](mesh_verts[vert_adr + imax, 0])
        + ld_y * rebind[Scalar[DTYPE]](mesh_verts[vert_adr + imax, 1])
        + ld_z * rebind[Scalar[DTYPE]](mesh_verts[vert_adr + imax, 2])
    )
    var prev = -1
    # ⚠ THE STEP BUDGET IS A HANG GUARD, NOT AN ALGORITHMIC BOUND. The
    # walk is monotone in `best_dot` and so cannot cycle on a well-formed
    # graph; it terminates in far fewer than `num_verts` steps. A malformed
    # adjacency would otherwise spin forever inside the physics step, and a
    # frozen viewer is the one failure mode here that costs a debugging
    # session rather than a test run.
    var budget = num_verts
    while imax != prev and budget > 0:
        budget -= 1
        prev = imax
        comptime if _HILL_PROBE:
            hill_probe()[].scans += 1
            if not (warm >= 0 and warm < num_verts):
                hill_probe()[].cold_scans += 1
        var e = Int(
            rebind[Scalar[DTYPE]](mesh_vert_edgeadr[vert_adr + imax])
        )
        if e < 0:
            break
        while True:
            var nb = Int(rebind[Scalar[DTYPE]](mesh_edges[e]))
            if nb < 0:
                break
            var d = (
                ld_x * rebind[Scalar[DTYPE]](mesh_verts[nb, 0])
                + ld_y * rebind[Scalar[DTYPE]](mesh_verts[nb, 1])
                + ld_z * rebind[Scalar[DTYPE]](mesh_verts[nb, 2])
            )
            if d > best_dot:
                best_dot = d
                imax = nb - vert_adr
            e += 1
    comptime if _HILL_PROBE:
        # Replay from MuJoCo 3.12's seed, counting scans; the landing vertex
        # must be the one above (a linear objective on a convex hull's
        # vertex graph has one maximum; a difference is a tie).
        var hp = hill_probe()
        var off = hp[].ext_offset(vert_adr)
        if off >= 0:
            var cx = 1
            if ld_x > Scalar[DTYPE](0.4):
                cx = 2
            elif ld_x < Scalar[DTYPE](-0.4):
                cx = 0
            var cy = 1
            if ld_y > Scalar[DTYPE](0.4):
                cy = 2
            elif ld_y < Scalar[DTYPE](-0.4):
                cy = 0
            var cz = 1
            if ld_z > Scalar[DTYPE](0.4):
                cz = 2
            elif ld_z < Scalar[DTYPE](-0.4):
                cz = 0
            var seed = hp[].ext[off + cx * 9 + cy * 3 + cz]
            var cold = not (warm >= 0 and warm < num_verts)
            var s_imax = seed
            var s_best = (
                ld_x * rebind[Scalar[DTYPE]](mesh_verts[vert_adr + seed, 0])
                + ld_y * rebind[Scalar[DTYPE]](mesh_verts[vert_adr + seed, 1])
                + ld_z * rebind[Scalar[DTYPE]](mesh_verts[vert_adr + seed, 2])
            )
            if not cold:
                var w_dot = (
                    ld_x * rebind[Scalar[DTYPE]](mesh_verts[vert_adr + warm, 0])
                    + ld_y * rebind[Scalar[DTYPE]](mesh_verts[vert_adr + warm, 1])
                    + ld_z * rebind[Scalar[DTYPE]](mesh_verts[vert_adr + warm, 2])
                )
                if s_best > w_dot:
                    hp[].seed_beat_warm += 1
                else:
                    s_imax = warm
                    s_best = w_dot
            var s_prev = -1
            var s_budget = num_verts
            while s_imax != s_prev and s_budget > 0:
                s_budget -= 1
                s_prev = s_imax
                hp[].seeded_scans += 1
                if cold:
                    hp[].seeded_cold_scans += 1
                var se = Int(
                    rebind[Scalar[DTYPE]](mesh_vert_edgeadr[vert_adr + s_imax])
                )
                if se < 0:
                    break
                while True:
                    var snb = Int(rebind[Scalar[DTYPE]](mesh_edges[se]))
                    if snb < 0:
                        break
                    var sd = (
                        ld_x * rebind[Scalar[DTYPE]](mesh_verts[snb, 0])
                        + ld_y * rebind[Scalar[DTYPE]](mesh_verts[snb, 1])
                        + ld_z * rebind[Scalar[DTYPE]](mesh_verts[snb, 2])
                    )
                    if sd > s_best:
                        s_best = sd
                        s_imax = snb - vert_adr
                    se += 1
            if s_imax != imax:
                hp[].seed_mismatch += 1
                var dd = abs(Float64(s_best) - Float64(best_dot))
                if dd > 1e-6 * (abs(Float64(best_dot)) + 1e-30):
                    hp[].seed_mismatch_nontie += 1
        # Cross-step warm start: replay from where this ordinal's call landed
        # in the previous step.
        var o = hp[].ordinal
        hp[].ordinal += 1
        if o < 256:
            var pl = hp[].prev_land[o]
            if pl >= 0 and pl < num_verts:
                hp[].xstep_calls += 1
                var x_imax = pl
                var x_best = (
                    ld_x * rebind[Scalar[DTYPE]](mesh_verts[vert_adr + pl, 0])
                    + ld_y * rebind[Scalar[DTYPE]](mesh_verts[vert_adr + pl, 1])
                    + ld_z * rebind[Scalar[DTYPE]](mesh_verts[vert_adr + pl, 2])
                )
                var x_prev = -1
                var x_budget = num_verts
                while x_imax != x_prev and x_budget > 0:
                    x_budget -= 1
                    x_prev = x_imax
                    hp[].xstep_scans += 1
                    var xe = Int(
                        rebind[Scalar[DTYPE]](mesh_vert_edgeadr[vert_adr + x_imax])
                    )
                    if xe < 0:
                        break
                    while True:
                        var xnb = Int(rebind[Scalar[DTYPE]](mesh_edges[xe]))
                        if xnb < 0:
                            break
                        var xd = (
                            ld_x * rebind[Scalar[DTYPE]](mesh_verts[xnb, 0])
                            + ld_y * rebind[Scalar[DTYPE]](mesh_verts[xnb, 1])
                            + ld_z * rebind[Scalar[DTYPE]](mesh_verts[xnb, 2])
                        )
                        if xd > x_best:
                            x_best = xd
                            x_imax = xnb - vert_adr
                        xe += 1
            hp[].prev_land[o] = imax
    return imax


@always_inline
def _support_mesh[
    DTYPE: DType,
    L_MESH_VERTS: Layout,
    L_MESH_VERT_EDGEADR: Layout,
    L_MESH_EDGES: Layout,
](
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    pos_x: Scalar[DTYPE],
    pos_y: Scalar[DTYPE],
    pos_z: Scalar[DTYPE],
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    mesh_verts: LayoutTensor[
        DTYPE, L_MESH_VERTS, MutAnyOrigin
    ],
    mesh_vert_edgeadr: LayoutTensor[
        DTYPE, L_MESH_VERT_EDGEADR, MutAnyOrigin
    ],
    mesh_edges: LayoutTensor[
        DTYPE, L_MESH_EDGES, MutAnyOrigin
    ],
    vert_adr: Int,
    num_verts: Int,
    mut warm: Int,
) -> Array[Scalar[DTYPE], 3]:
    """Support point on the mesh — HILL CLIMB over the hull edge graph.

    This is the hottest function in mesh collision: GJK and EPA call it 10-30
    times per geom pair per step, and it used to be a LINEAR SCAN over every
    hull vertex. That made narrow-phase cost O(hull size), and the bill arrives
    as soon as a model carries CAD-resolution collision meshes. Two changes,
    each measured on its own (ms per env step, which is `FRAME_SKIP=10` physics
    steps; MuJoCo on the same two XMLs is 0.078 and 0.121 for reference):

                   hull verts    scan     + climb   + warm start
        SO-ARM100      2 551    11.95 ms   5.35 ms    2.82 ms   ->  354 Hz
        SO-ARM101     33 076    76.03 ms  12.79 ms    4.74 ms   ->  211 Hz

    (Both arms have since dropped further, to 1.09 and 1.84 ms, by giving the
    SAP broadphase the bounding-sphere reject it was missing — see
    `broadphase_sap.mojo`. That change removes CALLS to this function; it does
    not make a call cheaper, so the columns above still measure what they say.

    Every column re-measured INTERLEAVED against a pristine worktree of the
    commit before it, reported as the MIN of two rounds — a baseline taken
    earlier in a session has drifted by 1.4-1.7x here before, which would have
    inflated this. The MuJoCo figures are a scale reference, not a like-for-like
    gate: both models sit at `ncon = 0` there, so what is being compared is the
    cost of proving pairs APART, which is where GJK spends its time anyway.)

    ⚠ THE 6.6x IS NOT THE MODEL'S FAULT, AND THAT IS THE MEASUREMENT THAT
    MATTERS. The two arms have IDENTICAL dynamics — `nq = nv = nu = 6`,
    `nbody = 8` — so nothing but collision can differ. MuJoCo steps the very
    same two XMLs at 0.092 ms and 0.136 ms, a ratio of 1.5x against our 6.6x.
    A reference that barely notices 13x the vertices is telling you the scaling
    is the implementation's, not the input's.

    MuJoCo's answer is `mjc_hillclimbSupport` (`engine_collision_convex.c`):
    walk the hull's vertex adjacency, greedily stepping to whichever neighbour
    scores higher on `dir`, until no neighbour improves. On a CONVEX hull the
    dot product has no local maximum other than the global one, so the walk is
    exact — it is not an approximation. It keeps `mjc_meshSupport`'s linear
    scan only below `mjMESH_HILLCLIMB_MIN = 10` vertices, where the graph
    overhead outweighs the scan; `_HILLCLIMB_MIN` below is that constant.

    ⚠ THE ADJACENCY WAS ALREADY THERE. `mesh_vert_edgeadr` / `mesh_edges` are
    built by `build_hull_edge_graph` for the plane-mesh path. Ours needs no
    `vert_globalid` indirection because `mesh_verts` holds hull vertices only —
    `mesh_edges` already stores global vertex ids, `-1` terminated.

    `warm` IS THE WARM START, AND IT IS WHAT MAKES THE WALK CHEAP IN PRACTICE.
    It is a LOCAL vertex index — MuJoCo's `mjCCDObj.meshindex` — carrying the
    vertex the previous call landed on, in and out. GJK and EPA ask for support
    points in directions that move a little at a time, so the previous answer is
    usually a neighbour of this one: from vertex 0 the walk costs O(graph
    diameter), from the previous answer it costs a handful of steps. Callers own
    the state; `gjk_epa_witness` keeps one per object for the whole GJK+EPA run,
    which is exactly the lifetime of MuJoCo's `mjCCDObj`.

    ⚠ THE SEED CANNOT MAKE THE ANSWER WRONG, ONLY SLOW. On a convex hull the
    walk is monotone and has no local maximum to be trapped by, so it converges
    to the extreme vertex from ANY starting vertex. That is why the guard below
    can silently clamp a nonsensical seed instead of trusting the caller: a
    crossed or stale index costs steps, never correctness.

    ⚠ THE SCAN ARM DELIBERATELY IGNORES `warm`. MuJoCo's `mjc_meshSupport` seeds
    its running max from the cache, which only shifts tie-breaking (it still
    scans everything). Leaving it out keeps the scan an INDEPENDENT exhaustive
    reference, which is the only reason the hill climb has anything to be
    checked against in `test_gjk_hillclimb_support.mojo`.

    ⚠ FALLS BACK TO THE SCAN when the graph is absent (`edgeadr < 0`, which is
    how `fields/model.mojo` marks an unbuilt graph) or the mesh is tiny. Both
    paths must return the same point — `test_gjk_hillclimb_support.mojo` pins
    that on real hulls, because a support function that is merely CLOSE yields
    plausible contacts in the wrong place rather than an obvious failure.
    """
    # ⚠ `mulMatTVec3(local_dir, obj->mat, dir)` — the reference reaches the
    # local frame through `geom_xmat`, not by rotating with the quaternion. The
    # SEARCH below is still the hill climb (`test_gjk_hillclimb_support` pins
    # it against the exhaustive scan); only the frame arithmetic changed.
    var _mm = quat2mat[DTYPE](qx, qy, qz, qw)
    var local_dir = mat_t_vec3[DTYPE](
        _mm[0], _mm[1], _mm[2], _mm[3], _mm[4], _mm[5], _mm[6], _mm[7], _mm[8],
        dir_x, dir_y, dir_z,
    )
    var ld_x = local_dir[0]
    var ld_y = local_dir[1]
    var ld_z = local_dir[2]

    var best_x: Scalar[DTYPE] = 0
    var best_y: Scalar[DTYPE] = 0
    var best_z: Scalar[DTYPE] = 0

    var imax = hillclimb_support_index[DTYPE](
        ld_x, ld_y, ld_z,
        mesh_verts, mesh_vert_edgeadr, mesh_edges,
        vert_adr, num_verts, warm,
    )
    if imax >= 0:
        # Hand the landing vertex back so the next call starts here. Dropping
        # this line does not break a single result — it silently reverts the
        # walk to starting from vertex 0 every time, which is why
        # `test_gjk_hillclimb_support.mojo` asserts on `warm` itself.
        warm = imax
        best_x = rebind[Scalar[DTYPE]](mesh_verts[vert_adr + imax, 0])
        best_y = rebind[Scalar[DTYPE]](mesh_verts[vert_adr + imax, 1])
        best_z = rebind[Scalar[DTYPE]](mesh_verts[vert_adr + imax, 2])
    else:
        var best_dot: Scalar[DTYPE] = -1e30
        var best_i = -1
        for i in range(num_verts):
            var vx = rebind[Scalar[DTYPE]](mesh_verts[vert_adr + i, 0])
            var vy = rebind[Scalar[DTYPE]](mesh_verts[vert_adr + i, 1])
            var vz = rebind[Scalar[DTYPE]](mesh_verts[vert_adr + i, 2])
            var d = ld_x * vx + ld_y * vy + ld_z * vz
            if d > best_dot:
                best_dot = d
                best_x = vx
                best_y = vy
                best_z = vz
                best_i = i
        # ⚠ THE SCAN IGNORES `warm` AS A SEED BUT STILL REPORTS WHERE IT LANDED.
        # EPA stores `warm` as the vertex's support index and multicontact
        # reads it back instead of re-scanning the hull (`gjk_epa_witness`'s
        # `wfi`); left stale here, a small hull's index would name the
        # PREVIOUS support vertex. Nothing else reads it on this arm: a mesh
        # takes the same arm on every call.
        warm = best_i

    var world_pt = local_to_global[DTYPE](
        _mm[0], _mm[1], _mm[2], _mm[3], _mm[4], _mm[5], _mm[6], _mm[7], _mm[8],
        best_x, best_y, best_z, pos_x, pos_y, pos_z,
    )
    var result = Array[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    result[0] = world_pt[0]
    result[1] = world_pt[1]
    result[2] = world_pt[2]
    return result^


@always_inline
def _support[
    DTYPE: DType,
    L_MESH_VERTS: Layout,
    L_MESH_VERT_EDGEADR: Layout,
    L_MESH_EDGES: Layout,
    # ⚠⚠ THE PRISM'S LENGTH IS A PARAMETER SO IT COSTS NOTHING WHEN UNUSED.
    # It was a plain `Array[..., 18]` with a DEFAULT value, and a
    # defaulted aggregate materialises a fresh temporary at EVERY call site
    # that omits it — twelve of them, several inside the Metal collision
    # kernel. That pushed the kernel over the per-thread stack limit and
    # `test_plane_mesh_fields` died with "Compute function exceeds available
    # stack space", the same ceiling `MC_MAX_POLYVERT` sits on. At `NPRISM=1`
    # the argument is eight bytes and the heightfield branch is not compiled.
    NPRISM: Int = 1,
](
    geom_type: Int,
    pos_x: Scalar[DTYPE],
    pos_y: Scalar[DTYPE],
    pos_z: Scalar[DTYPE],
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    radius: Scalar[DTYPE],
    half_length: Scalar[DTYPE],
    half_x: Scalar[DTYPE],
    half_y: Scalar[DTYPE],
    half_z: Scalar[DTYPE],
    mesh_verts: LayoutTensor[
        DTYPE, L_MESH_VERTS, MutAnyOrigin
    ],
    mesh_vert_edgeadr: LayoutTensor[
        DTYPE, L_MESH_VERT_EDGEADR, MutAnyOrigin
    ],
    mesh_edges: LayoutTensor[
        DTYPE, L_MESH_EDGES, MutAnyOrigin
    ],
    vert_adr: Int,
    mesh_num_verts: Int,
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    mut warm: Int,
    # ⚠ THE HEIGHTFIELD PRISM, and it rides on the STACK rather than in a
    # tensor on purpose. `mjc_ConvexHField` rebuilds these six vertices for
    # every grid cell it walks, so they are per-CALL data — they change many
    # times within one `hfield_convex_contacts`, which nothing outside that
    # loop ever reads. Eighteen floats in an `Array` cost nothing.
    # (EPA's polytope went the other way, into `ws`; it is per-CALL too but
    # 7.5 KB of it. The dividing line is size, not lifetime — and `ws` is
    # race-free because it is indexed per ENV, not shared. See
    # `ccd_workspace`.) Ignored unless `geom_type == GEOM_HFIELD`.
    prism: Array[Scalar[DTYPE], NPRISM],
    # ⚠⚠ `mjc_ccd` SWAPS THE SUPPORT FUNCTION OUT for a sphere or a capsule
    # before it runs GJK: `obj->support = mjc_pointSupport` / `mjc_lineSupport`
    # and `obj->margin = 0`, with the radius folded into a `full_margin` that
    # `inflate()` adds back afterwards. That is not an optimisation — it is how
    # every sphere and capsule pair is SOLVED in the reference, because a
    # point-or-line versus anything is a distance query GJK gets exactly, where
    # the rounded surface is a smooth Minkowski boundary EPA can only
    # approximate. See the two-phase structure in `gjk_epa_witness`.
    shrink: Bool = False,
) -> Array[Scalar[DTYPE], 3]:
    """Unified support function — reads mesh verts from the record tensor.

    `warm` is the mesh hill-climb's start vertex, in and out; every other geom
    type leaves it untouched, so one variable per object is enough regardless
    of what that object turns out to be. See `_support_mesh`.
    """
    comptime if NPRISM >= 18:
        if geom_type == GEOM_HFIELD:
            # A heightfield never enters GJK as a heightfield — the caller has
            # already reduced it to ONE triangular prism. See `support_prism`.
            return support_prism[DTYPE](
                dir_x, dir_y, dir_z, rebind[Array[Scalar[DTYPE], 18]](prism)
            )
    if geom_type == GEOM_SPHERE:
        if shrink:
            # `mjc_pointSupport` — the geom's own position, whatever `dir` is.
            var pt = Array[Scalar[DTYPE], 3](uninitialized=True)
            pt[0] = pos_x
            pt[1] = pos_y
            pt[2] = pos_z
            return pt^
        return support_sphere[DTYPE](
            dir_x, dir_y, dir_z, pos_x, pos_y, pos_z, radius
        )
    elif geom_type == GEOM_CAPSULE and shrink:
        # `mjc_lineSupport` — the segment endpoint along the geom's local z.
        #   dot = mat[2]*d0 + mat[5]*d1 + mat[8]*d2
        #   scl = dot >= 0 ? length : -length     (length = size[1])
        # ⚠ `mjc_lineSupport` reads the axis straight out of the matrix —
        # `mat[2], mat[5], mat[8]`, its THIRD COLUMN — rather than rotating
        # (0, 0, 1) by the quaternion.
        var lm = quat2mat[DTYPE](qx, qy, qz, qw)
        var a0 = lm[2]
        var a1 = lm[5]
        var a2 = lm[8]
        var dt = a0 * dir_x + a1 * dir_y + a2 * dir_z
        var scl = half_length if dt >= Scalar[DTYPE](0) else -half_length
        var ln = Array[Scalar[DTYPE], 3](uninitialized=True)
        ln[0] = a0 * scl + pos_x
        ln[1] = a1 * scl + pos_y
        ln[2] = a2 * scl + pos_z
        return ln^
    elif geom_type == GEOM_CAPSULE:
        return support_capsule[DTYPE](
            dir_x,
            dir_y,
            dir_z,
            pos_x,
            pos_y,
            pos_z,
            qx,
            qy,
            qz,
            qw,
            radius,
            half_length,
        )
    elif geom_type == GEOM_BOX:
        # ⚠ `warm` IS MuJoCo's `obj->vertindex`, and the reference uses that
        # ONE field for both a mesh hull vertex and a box CORNER CODE. EPA's
        # discrete repeated-support break compares it, so a box has to write it.
        return support_box[DTYPE](
            dir_x,
            dir_y,
            dir_z,
            pos_x,
            pos_y,
            pos_z,
            qx,
            qy,
            qz,
            qw,
            half_x,
            half_y,
            half_z,
            warm,
        )
    elif geom_type == GEOM_ELLIPSOID:
        return support_ellipsoid[DTYPE](
            dir_x,
            dir_y,
            dir_z,
            pos_x,
            pos_y,
            pos_z,
            qx,
            qy,
            qz,
            qw,
            half_x,
            half_y,
            half_z,
        )
    elif geom_type == GEOM_CYLINDER:
        return support_cylinder[DTYPE](
            dir_x,
            dir_y,
            dir_z,
            pos_x,
            pos_y,
            pos_z,
            qx,
            qy,
            qz,
            qw,
            radius,
            half_length,
        )
    elif geom_type == GEOM_MESH:
        return _support_mesh[DTYPE](
            dir_x,
            dir_y,
            dir_z,
            pos_x,
            pos_y,
            pos_z,
            qx,
            qy,
            qz,
            qw,
            mesh_verts,
            mesh_vert_edgeadr,
            mesh_edges,
            vert_adr,
            mesh_num_verts,
            warm,
        )
    var result = Array[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    result[0] = pos_x
    result[1] = pos_y
    result[2] = pos_z
    return result^


@always_inline
def _minkowski_support[
    DTYPE: DType,
    L_MESH_VERTS: Layout,
    L_MESH_VERT_EDGEADR: Layout,
    L_MESH_EDGES: Layout,
    NPRISM: Int = 1,
](
    type1: Int,
    p1x: Scalar[DTYPE],
    p1y: Scalar[DTYPE],
    p1z: Scalar[DTYPE],
    q1x: Scalar[DTYPE],
    q1y: Scalar[DTYPE],
    q1z: Scalar[DTYPE],
    q1w: Scalar[DTYPE],
    r1: Scalar[DTYPE],
    hl1: Scalar[DTYPE],
    hx1: Scalar[DTYPE],
    hy1: Scalar[DTYPE],
    hz1: Scalar[DTYPE],
    mesh_verts: LayoutTensor[
        DTYPE, L_MESH_VERTS, MutAnyOrigin
    ],
    mesh_vert_edgeadr: LayoutTensor[
        DTYPE, L_MESH_VERT_EDGEADR, MutAnyOrigin
    ],
    mesh_edges: LayoutTensor[
        DTYPE, L_MESH_EDGES, MutAnyOrigin
    ],
    va1: Int,
    mnv1: Int,
    type2: Int,
    p2x: Scalar[DTYPE],
    p2y: Scalar[DTYPE],
    p2z: Scalar[DTYPE],
    q2x: Scalar[DTYPE],
    q2y: Scalar[DTYPE],
    q2z: Scalar[DTYPE],
    q2w: Scalar[DTYPE],
    r2: Scalar[DTYPE],
    hl2: Scalar[DTYPE],
    hx2: Scalar[DTYPE],
    hy2: Scalar[DTYPE],
    hz2: Scalar[DTYPE],
    va2: Int,
    mnv2: Int,
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    mut warm1: Int,
    mut warm2: Int,
    # The heightfield prism, for geom 1 only — see `_support`.
    prism: Array[Scalar[DTYPE], NPRISM],
    # ⚠⚠ EACH GEOM IS INFLATED BY HALF THE PAIR'S MARGIN, AND THAT IS THE WHOLE
    # OF `mjc_Convex`. `support()` (engine_collision_gjk.c:332) adds
    # `0.5 * obj->margin * dir` to each object's support point, so the
    # Minkowski support gains `margin * dir` — and `mjc_Convex` then asks for
    # the PENETRATION of the inflated pair and reports `margin + dist`
    # (engine_collision_convex.c:104/115). It never runs a distance query at
    # all: `config.dist_cutoff = 0`.
    #
    # ⚠ DEFAULTED TO 0 so the nine other call sites and every margin-free
    # model are bit-identical; `discreteGeoms` already treats a non-zero
    # margin as SMOOTH on both tolerance switches (`_gjk_min_norm2`,
    # `_epa_tolerance`), which is the other half of the same rule.
    #
    # ⚠ `dir` MUST BE A UNIT VECTOR HERE. The GJK loop passes `-v/|v|` and EPA
    # passes a normalised face normal, which is what makes the offset a true
    # Minkowski sum with a ball of radius `margin/2`.
    ccd_margin: Scalar[DTYPE] = Scalar[DTYPE](0),
    # ⚠ A SHRUNKEN OBJECT CARRIES NO MARGIN. `mjc_ccd` writes `obj->margin = 0`
    # alongside the support swap and folds the half-margin into `full_margin`
    # instead, so in the shrunken phase only the OTHER object still offsets its
    # support point. One flag per object, because a capsule against a mesh
    # shrinks one side and not the other.
    shrink1: Bool = False,
    shrink2: Bool = False,
    # ⚠ AUD-32: `inflate1=False` leaves object 1 UNINFLATED while object 2
    # still grows by `ccd_margin/2` — MuJoCo's heightfield collider sets
    # `obj2.margin = margin` and leaves the prism's at 0
    # (engine_collision_convex.c:1246). Everything else (tolerances, the
    # `+ ccd_margin` add-back) is unchanged; the hfield caller subtracts it.
    inflate1: Bool = True,
) -> Tuple[
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
]:
    var s1 = _support[DTYPE, NPRISM=NPRISM](
        type1,
        p1x,
        p1y,
        p1z,
        q1x,
        q1y,
        q1z,
        q1w,
        r1,
        hl1,
        hx1,
        hy1,
        hz1,
        mesh_verts,
        mesh_vert_edgeadr,
        mesh_edges,
        va1,
        mnv1,
        dir_x,
        dir_y,
        dir_z,
        warm1,
        prism,
        shrink1,
    )
    var s2 = _support[DTYPE, NPRISM=NPRISM](
        type2,
        p2x,
        p2y,
        p2z,
        q2x,
        q2y,
        q2z,
        q2w,
        r2,
        hl2,
        hx2,
        hy2,
        hz2,
        mesh_verts,
        mesh_vert_edgeadr,
        mesh_edges,
        va2,
        mnv2,
        -dir_x,
        -dir_y,
        -dir_z,
        warm2,
        prism,
        shrink2,
    )
    var hm1 = Scalar[DTYPE](0)
    if not shrink1 and inflate1:
        hm1 = Scalar[DTYPE](0.5) * ccd_margin
    var hm2 = Scalar[DTYPE](0)
    if not shrink2:
        hm2 = Scalar[DTYPE](0.5) * ccd_margin
    var w1x = s1[0] + dir_x * hm1
    var w1y = s1[1] + dir_y * hm1
    var w1z = s1[2] + dir_z * hm1
    var w2x = s2[0] - dir_x * hm2
    var w2y = s2[1] - dir_y * hm2
    var w2z = s2[2] - dir_z * hm2
    return (
        w1x - w2x,
        w1y - w2y,
        w1z - w2z,
        w1x,
        w1y,
        w1z,
        w2x,
        w2y,
        w2z,
    )


def _gjk_intersect[
    DTYPE: DType,
    L_MESH_VERTS: Layout,
    L_MESH_VERT_EDGEADR: Layout,
    L_MESH_EDGES: Layout,
    L_WS: Layout,
    NPRISM: Int = 1,
](
    ws: LayoutTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    base: Int,
    type1: Int,
    p1x: Scalar[DTYPE], p1y: Scalar[DTYPE], p1z: Scalar[DTYPE],
    q1x: Scalar[DTYPE], q1y: Scalar[DTYPE], q1z: Scalar[DTYPE],
    q1w: Scalar[DTYPE],
    r1: Scalar[DTYPE], hl1: Scalar[DTYPE],
    hx1: Scalar[DTYPE], hy1: Scalar[DTYPE], hz1: Scalar[DTYPE],
    mesh_verts: LayoutTensor[
        DTYPE, L_MESH_VERTS, MutAnyOrigin
    ],
    mesh_vert_edgeadr: LayoutTensor[
        DTYPE, L_MESH_VERT_EDGEADR, MutAnyOrigin
    ],
    mesh_edges: LayoutTensor[
        DTYPE, L_MESH_EDGES, MutAnyOrigin
    ],
    va1: Int, mnv1: Int,
    type2: Int,
    p2x: Scalar[DTYPE], p2y: Scalar[DTYPE], p2z: Scalar[DTYPE],
    q2x: Scalar[DTYPE], q2y: Scalar[DTYPE], q2z: Scalar[DTYPE],
    q2w: Scalar[DTYPE],
    r2: Scalar[DTYPE], hl2: Scalar[DTYPE],
    hx2: Scalar[DTYPE], hy2: Scalar[DTYPE], hz2: Scalar[DTYPE],
    va2: Int, mnv2: Int,
    mut warm1: Int, mut warm2: Int,
    # The heightfield prism, for geom 1 only — see `_support`.
    prism: Array[Scalar[DTYPE], NPRISM],
    # ⚠ FORWARDED, NOT RECOMPUTED — see `_minkowski_support`. This backup
    # path builds its own tetrahedron from support points and must inflate
    # them the same way, or it certifies separation on the UNINFLATED pair
    # while the loop around it is working on the inflated one.
    ccd_margin: Scalar[DTYPE] = Scalar[DTYPE](0),
    # ⚠ AND THE SHRINK FLAGS FOR THE SAME REASON. `gjkIntersect` runs inside
    # `gjk()`, so it sees whichever support functions `mjc_ccd` has installed;
    # running it on the ROUNDED shapes while the loop around it is on the
    # shrunken ones would certify a tetrahedron the caller cannot use.
    shrink1: Bool = False,
    shrink2: Bool = False,
    # ⚠ AUD-32: `inflate1=False` leaves object 1 UNINFLATED while object 2
    # still grows by `ccd_margin/2` — MuJoCo's heightfield collider sets
    # `obj2.margin = margin` and leaves the prism's at 0
    # (engine_collision_convex.c:1246). Everything else (tolerances, the
    # `+ ccd_margin` add-back) is unchanged; the hfield caller subtracts it.
    inflate1: Bool = True,
) -> Int:
    """Refine a 4-simplex until it ENCLOSES the origin. MuJoCo's `gjkIntersect`.

    Returns 1 (origin enclosed; `simplex` rewritten as a valid tetrahedron),
    0 (origin proven OUTSIDE the Minkowski difference — no collision), or -1
    (inconclusive; the caller should fall back to its distance subalgorithm).

    ⚠ THIS IS THE PIECE WHOSE ABSENCE BROKE EVERY ATTEMPT TO USE EPA ON
    PENETRATING PRIMITIVES. Without it our GJK terminated on a 2-SIMPLEX for
    pairs with millimetres of genuine overlap — a cylinder 5 mm into the RIM of
    a box — and EPA was then handed a seed that does not contain the origin.
    Two prior fixes failed for the same root reason: `polytope3` reconstructs a
    hexahedron that need not enclose (MuJoCo only ever VALIDATES with
    `testTetra`), and greedily growing the seed toward the most-violated face
    CYCLES, requesting the same support point twice. An inscribed polytope of
    the Minkowski difference need not contain the origin even when the origin
    is inside it, so no amount of patching downstream substitutes for arriving
    with a real enclosing tetrahedron.

    The mechanism: keep the four faces of the current tetrahedron, each with an
    ORIENTED signed distance to the origin. If the smallest is positive the
    origin is inside every face and we are done. Otherwise replace the vertex
    OPPOSITE the most-violated face with the support point along that face's
    normal, which is the only direction that can bring the origin inside. The
    trailing swap keeps the tetrahedron's orientation consistent so the signs
    stay meaningful.

    It also carries a real SEPARATION CERTIFICATE, which the engine has never
    had: if that support point does not reach the origin's side of the face
    plane, the whole Minkowski difference lies beyond it and the geoms provably
    do not overlap.
    """
    # ⚠ THE PERMUTATION IS FOUR SCALARS, NOT AN `Array[Int, 4]`.
    # `sidx[index]` and the trailing swap index it by a RUNTIME value, which is
    # the per-thread-array miscompile of
    # `feedback_metal_wide_per_thread_inlinearray_miscompute` — three elements
    # was enough to hit it the second time. `_sel4i` is the same if-chain fix
    # `_capsule_box_second_pos` took.
    var sidx0 = 0
    var sidx1 = 1
    var sidx2 = 2
    var sidx3 = 3

    # ⚠⚠ THERE USED TO BE A RE-ORIENTATION HERE AND MuJoCo HAS NONE.
    # It computed the scalar triple product of the incoming simplex and swapped
    # `sidx[0]`/`sidx[1]` when it was negative, on the reasoning that "MuJoCo
    # can skip this because its `gjk` builds the simplex by appending support
    # points and never reorders them; ours arrives via
    # `_closest_point_on_simplex`, which REDUCES and re-packs".
    #
    # THAT PREMISE IS GONE. `subdistance` is a one-for-one port now and the
    # caller compacts IN ORDER, exactly as the reference does — so our simplex
    # reaches this function with the same winding MuJoCo's does, and imposing
    # an orientation on top of it produces a DIFFERENT tetrahedron.
    #
    # ⚠ MEASURED on the ellipsoid pair of `test_ellipsoid_convex_vs_mujoco`,
    # arbitrated against `min over unit n of h(n)` rather than against MuJoCo:
    # the contact normal was 3.823 deg from the true minimising direction with
    # the swap and MuJoCo is 0.694 deg from it. The seed tetrahedron's closest
    # face was 1.68e-08 against the reference's 2.55e-06 — a different simplex
    # from the first line.
    for _ in range(GJK_MAX_ITERATIONS):
        # signed distance to each face; face i is the one OPPOSITE vertex i
        var f0 = _gjk_signed_distance[DTYPE](
            ws, wrow, base, sidx2, sidx1, sidx3
        )
        var f1 = _gjk_signed_distance[DTYPE](
            ws, wrow, base, sidx0, sidx2, sidx3
        )
        var f2 = _gjk_signed_distance[DTYPE](
            ws, wrow, base, sidx1, sidx0, sidx3
        )
        var f3 = _gjk_signed_distance[DTYPE](
            ws, wrow, base, sidx0, sidx1, sidx2
        )

        # origin exactly on an affine hull -> the signs cannot converge
        if (
            f0[0] == Scalar[DTYPE](0)
            or f1[0] == Scalar[DTYPE](0)
            or f2[0] == Scalar[DTYPE](0)
            or f3[0] == Scalar[DTYPE](0)
        ):
            return -1

        var i = 0 if f0[0] < f1[0] else 1
        var j = 2 if f2[0] < f3[0] else 3
        var di = f0[0] if i == 0 else f1[0]
        var dj = f2[0] if j == 2 else f3[0]
        var index = i if di < dj else j

        var best = di if di < dj else dj
        # ⚠⚠ THERE USED TO BE A RELATIVE FLOOR HERE AND MuJoCo HAS NONE.
        # `dist[index] > 0` is the whole test; a distance of EXACTLY zero is
        # already rejected above as "origin on the affine hull". The floor —
        # `best <= 1e-12 * max|simplex|` also returning -1 — was added because
        # a cylinder resting exactly on a box rim came back as a confident
        # enclosure and EPA then reported -0.09998 with a sideways normal.
        # That was the OLD EPA, which had no `testTetra` and no
        # `polytope2/3/4`; the faithful one validates the seed the reference's
        # way and answers "no contact" when it fails, so the floor is covering
        # for something that no longer exists. Leaving it in cost a real
        # difference: it rejected every THIN enclosure, so a 5 mm box/cylinder
        # overlap fell through to the distance subalgorithm and seeded EPA from
        # a TRIANGLE (`polytope3`) where MuJoCo seeds from the TETRAHEDRON.
        if best > Scalar[DTYPE](0):
            # enclosed: emit the tetrahedron in permutation order
            # Built in `CCD_WS_SPX2` first: the permutation can read a slot
            # it has already overwritten, so it cannot be done in place. That
            # is the reference's own `Vertex simplex[4]` local.
            for c in range(SPX_STRIDE):
                set_sv(ws, wrow, CCD_WS_SPX2, 0, c, sv(ws, wrow, base, sidx0, c))
                set_sv(ws, wrow, CCD_WS_SPX2, 1, c, sv(ws, wrow, base, sidx1, c))
                set_sv(ws, wrow, CCD_WS_SPX2, 2, c, sv(ws, wrow, base, sidx2, c))
                set_sv(ws, wrow, CCD_WS_SPX2, 3, c, sv(ws, wrow, base, sidx3, c))
            for v in range(4):
                for c in range(SPX_STRIDE):
                    set_sv(
                        ws, wrow, base, v, c, sv(ws, wrow, CCD_WS_SPX2, v, c)
                    )
            return 1

        var nx: Scalar[DTYPE]
        var ny: Scalar[DTYPE]
        var nz: Scalar[DTYPE]
        if index == 0:
            nx = f0[1]
            ny = f0[2]
            nz = f0[3]
        elif index == 1:
            nx = f1[1]
            ny = f1[2]
            nz = f1[3]
        elif index == 2:
            nx = f2[1]
            ny = f2[2]
            nz = f2[3]
        else:
            nx = f3[1]
            ny = f3[2]
            nz = f3[3]

        var w = _minkowski_support[DTYPE, NPRISM=NPRISM](
            type1, p1x, p1y, p1z, q1x, q1y, q1z, q1w,
            r1, hl1, hx1, hy1, hz1, mesh_verts, mesh_vert_edgeadr, mesh_edges, va1, mnv1,
            type2, p2x, p2y, p2z, q2x, q2y, q2z, q2w,
            r2, hl2, hx2, hy2, hz2, va2, mnv2,
            nx, ny, nz,
            warm1, warm2,
            prism,
            ccd_margin,
            shrink1,
            shrink2,
            inflate1,
        )
        var tgt = _sel4i(index, sidx0, sidx1, sidx2, sidx3)
        set_sv(ws, wrow, base, tgt, 0, w[0])
        set_sv(ws, wrow, base, tgt, 1, w[1])
        set_sv(ws, wrow, base, tgt, 2, w[2])
        set_sv(ws, wrow, base, tgt, 3, w[3])
        set_sv(ws, wrow, base, tgt, 4, w[4])
        set_sv(ws, wrow, base, tgt, 5, w[5])
        set_sv(ws, wrow, base, tgt, 6, w[6])
        set_sv(ws, wrow, base, tgt, 7, w[7])
        set_sv(ws, wrow, base, tgt, 8, w[8])
        set_sv(ws, wrow, base, tgt, 9, Scalar[DTYPE](warm1))
        set_sv(ws, wrow, base, tgt, 10, Scalar[DTYPE](warm2))

        # separation certificate
        if nx * w[0] + ny * w[1] + nz * w[2] < Scalar[DTYPE](0):
            return 0

        var a = (index + 1) & 3
        var b = (index + 2) & 3
        var va = _sel4i(a, sidx0, sidx1, sidx2, sidx3)
        var vb = _sel4i(b, sidx0, sidx1, sidx2, sidx3)
        if a == 0:
            sidx0 = vb
        elif a == 1:
            sidx1 = vb
        elif a == 2:
            sidx2 = vb
        else:
            sidx3 = vb
        if b == 0:
            sidx0 = va
        elif b == 1:
            sidx1 = va
        elif b == 2:
            sidx2 = va
        else:
            sidx3 = va

    return -1


def gjk_epa_witness[
    DTYPE: DType,
    L_MESH_VERTS: Layout,
    L_MESH_VERT_EDGEADR: Layout,
    L_MESH_EDGES: Layout,
    L_WS: Layout,
    NPRISM: Int = 1,
](
    type1: Int,
    p1x: Scalar[DTYPE],
    p1y: Scalar[DTYPE],
    p1z: Scalar[DTYPE],
    q1x: Scalar[DTYPE],
    q1y: Scalar[DTYPE],
    q1z: Scalar[DTYPE],
    q1w: Scalar[DTYPE],
    r1: Scalar[DTYPE],
    hl1: Scalar[DTYPE],
    hx1: Scalar[DTYPE],
    hy1: Scalar[DTYPE],
    hz1: Scalar[DTYPE],
    mesh_verts: LayoutTensor[
        DTYPE, L_MESH_VERTS, MutAnyOrigin
    ],
    mesh_vert_edgeadr: LayoutTensor[
        DTYPE, L_MESH_VERT_EDGEADR, MutAnyOrigin
    ],
    mesh_edges: LayoutTensor[
        DTYPE, L_MESH_EDGES, MutAnyOrigin
    ],
    va1: Int,
    mnv1: Int,
    type2: Int,
    p2x: Scalar[DTYPE],
    p2y: Scalar[DTYPE],
    p2z: Scalar[DTYPE],
    q2x: Scalar[DTYPE],
    q2y: Scalar[DTYPE],
    q2z: Scalar[DTYPE],
    q2w: Scalar[DTYPE],
    r2: Scalar[DTYPE],
    hl2: Scalar[DTYPE],
    hx2: Scalar[DTYPE],
    hy2: Scalar[DTYPE],
    hz2: Scalar[DTYPE],
    va2: Int,
    mnv2: Int,
    mut wf1: Array[Scalar[DTYPE], 9],
    mut wf2: Array[Scalar[DTYPE], 9],
    mut wx: Array[Scalar[DTYPE], 6],
    mut wf_ok: Int,
    mut wfi: Array[Int, 6],
    # ⚠ EPA'S POLYTOPE — MuJoCo's `config->buffer`. `ws[wrow, ...]` is the
    # caller's scratch row and is written unconditionally; nothing in it is
    # read across calls, so no caller has to clear it. One row per ENV is what
    # keeps it thread-local in the collision kernels. See `ccd_workspace`.
    ws: LayoutTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    ccd_tol: Scalar[DTYPE] = Scalar[DTYPE](MJ_CCD_TOLERANCE),
    ccd_iter: Int = MJ_CCD_ITERATIONS,
    ccd_margin: Scalar[DTYPE] = Scalar[DTYPE](0),
    # ⚠⚠ OPT-IN, AND THE DEFAULT MUST STAY "DISABLED". Negative means
    # "converge to the true distance", which is what every distance gate in the
    # tree asserts on (`test_gjk_float32_no_phantom_contacts` compares
    # separations of 7-17 cm). Only a caller that uses the result SOLELY for a
    # `dist < margin` contact test may pass a cutoff. See the exit in the loop.
    dist_cutoff: Scalar[DTYPE] = Scalar[DTYPE](-1),
    # ⚠ DEFAULTED SO THE TWELVE EXISTING CALL SITES ARE UNTOUCHED. Only
    # `hfield_convex.mojo` passes it, and only with `type1 == GEOM_HFIELD`;
    # every other caller collides two real geoms and never reads it.
    prism: Array[Scalar[DTYPE], NPRISM] = Array[
        Scalar[DTYPE], NPRISM
    ](fill=Scalar[DTYPE](0)),
    warm_slot: Int = -1,
    # The ROW the warm slots are read from and written to; -1 = `wrow`.
    # The block-per-env collision kernel gives every thread its own
    # workspace row (EPA's polytope, the clipper's polygons) but keeps the
    # warm slots on ONE row per env — lane 0's — so a pair's vertex from the
    # previous step is found whatever lane the pair lands on this step.
    # Two lanes of one env can race on a slot (a hash collision); any
    # value in a slot is safe (`ccd_workspace.mojo`), so that costs steps,
    # never a point.
    hw_row: Int = -1,
    # ⚠ AUD-32: `inflate1=False` leaves object 1 UNINFLATED while object 2
    # still grows by `ccd_margin/2` — MuJoCo's heightfield collider sets
    # `obj2.margin = margin` and leaves the prism's at 0
    # (engine_collision_convex.c:1246). Everything else (tolerances, the
    # `+ ccd_margin` add-back) is unchanged; the hfield caller subtracts it.
    inflate1: Bool = True,
) -> Tuple[
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
]:
    """`_gjk_epa_witness_run` with the mesh hill climb's warm vertices
    seeded from, and written back to, warm slot `warm_slot` of the CCD
    workspace row (`HW_WS_OFF + 2 * warm_slot`, object 1 then object 2) — the
    cross-step warm start of `ccd_workspace.mojo`. `warm_slot < 0` (the
    default, and every caller but `_sap_pair_narrow`) runs cold, as before.
    Only a MESH's slot is read or written: a box keeps `warm` as a corner
    code for EPA's repeated-support test and must start from -1.
    """
    var warm1 = -1
    var warm2 = -1
    var off = HW_WS_OFF + 2 * warm_slot
    var hrow = hw_row if hw_row >= 0 else wrow
    if warm_slot >= 0:
        if type1 == GEOM_MESH:
            var f = rebind[Scalar[DTYPE]](ws[hrow, off])
            if f >= Scalar[DTYPE](0) and f < Scalar[DTYPE](1e8):
                warm1 = Int(f)
        if type2 == GEOM_MESH:
            var f2 = rebind[Scalar[DTYPE]](ws[hrow, off + 1])
            if f2 >= Scalar[DTYPE](0) and f2 < Scalar[DTYPE](1e8):
                warm2 = Int(f2)
    # The winning face's support indices (hull vertex / box corner code) for
    # object 1 then object 2; -1 wherever EPA did not produce one. Constant
    # indices only — see `feedback_metal_wide_per_thread_inlinearray_miscompute`.
    wfi[0] = -1; wfi[1] = -1; wfi[2] = -1
    wfi[3] = -1; wfi[4] = -1; wfi[5] = -1
    var r = _gjk_epa_witness_run[DTYPE, NPRISM=NPRISM](
        type1,
        p1x,
        p1y,
        p1z,
        q1x,
        q1y,
        q1z,
        q1w,
        r1,
        hl1,
        hx1,
        hy1,
        hz1,
        mesh_verts,
        mesh_vert_edgeadr,
        mesh_edges,
        va1,
        mnv1,
        type2,
        p2x,
        p2y,
        p2z,
        q2x,
        q2y,
        q2z,
        q2w,
        r2,
        hl2,
        hx2,
        hy2,
        hz2,
        va2,
        mnv2,
        wf1,
        wf2,
        wx,
        wf_ok,
        wfi,
        ws,
        wrow,
        warm1,
        warm2,
        ccd_tol,
        ccd_iter,
        ccd_margin,
        dist_cutoff,
        prism,
        inflate1=inflate1,
    )
    if warm_slot >= 0:
        if type1 == GEOM_MESH:
            ws[hrow, off] = Scalar[DTYPE](warm1)
        if type2 == GEOM_MESH:
            ws[hrow, off + 1] = Scalar[DTYPE](warm2)
    return r^


def _gjk_epa_witness_run[
    DTYPE: DType,
    L_MESH_VERTS: Layout,
    L_MESH_VERT_EDGEADR: Layout,
    L_MESH_EDGES: Layout,
    L_WS: Layout,
    NPRISM: Int = 1,
](
    type1: Int,
    p1x: Scalar[DTYPE],
    p1y: Scalar[DTYPE],
    p1z: Scalar[DTYPE],
    q1x: Scalar[DTYPE],
    q1y: Scalar[DTYPE],
    q1z: Scalar[DTYPE],
    q1w: Scalar[DTYPE],
    r1: Scalar[DTYPE],
    hl1: Scalar[DTYPE],
    hx1: Scalar[DTYPE],
    hy1: Scalar[DTYPE],
    hz1: Scalar[DTYPE],
    mesh_verts: LayoutTensor[
        DTYPE, L_MESH_VERTS, MutAnyOrigin
    ],
    mesh_vert_edgeadr: LayoutTensor[
        DTYPE, L_MESH_VERT_EDGEADR, MutAnyOrigin
    ],
    mesh_edges: LayoutTensor[
        DTYPE, L_MESH_EDGES, MutAnyOrigin
    ],
    va1: Int,
    mnv1: Int,
    type2: Int,
    p2x: Scalar[DTYPE],
    p2y: Scalar[DTYPE],
    p2z: Scalar[DTYPE],
    q2x: Scalar[DTYPE],
    q2y: Scalar[DTYPE],
    q2z: Scalar[DTYPE],
    q2w: Scalar[DTYPE],
    r2: Scalar[DTYPE],
    hl2: Scalar[DTYPE],
    hx2: Scalar[DTYPE],
    hy2: Scalar[DTYPE],
    hz2: Scalar[DTYPE],
    va2: Int,
    mnv2: Int,
    mut wf1: Array[Scalar[DTYPE], 9],
    mut wf2: Array[Scalar[DTYPE], 9],
    mut wx: Array[Scalar[DTYPE], 6],
    mut wf_ok: Int,
    mut wfi: Array[Int, 6],
    # ⚠ EPA'S POLYTOPE — MuJoCo's `config->buffer`. `ws[wrow, ...]` is the
    # caller's scratch row and is written unconditionally; nothing in it is
    # read across calls, so no caller has to clear it. One row per ENV is what
    # keeps it thread-local in the collision kernels. See `ccd_workspace`.
    ws: LayoutTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    mut warm1: Int,
    mut warm2: Int,
    ccd_tol: Scalar[DTYPE] = Scalar[DTYPE](MJ_CCD_TOLERANCE),
    ccd_iter: Int = MJ_CCD_ITERATIONS,
    ccd_margin: Scalar[DTYPE] = Scalar[DTYPE](0),
    # ⚠⚠ OPT-IN, AND THE DEFAULT MUST STAY "DISABLED". Negative means
    # "converge to the true distance", which is what every distance gate in the
    # tree asserts on (`test_gjk_float32_no_phantom_contacts` compares
    # separations of 7-17 cm). Only a caller that uses the result SOLELY for a
    # `dist < margin` contact test may pass a cutoff. See the exit in the loop.
    dist_cutoff: Scalar[DTYPE] = Scalar[DTYPE](-1),
    # ⚠ DEFAULTED SO THE TWELVE EXISTING CALL SITES ARE UNTOUCHED. Only
    # `hfield_convex.mojo` passes it, and only with `type1 == GEOM_HFIELD`;
    # every other caller collides two real geoms and never reads it.
    prism: Array[Scalar[DTYPE], NPRISM] = Array[
        Scalar[DTYPE], NPRISM
    ](fill=Scalar[DTYPE](0)),
    # ⚠ AUD-32: `inflate1=False` leaves object 1 UNINFLATED while object 2
    # still grows by `ccd_margin/2` — MuJoCo's heightfield collider sets
    # `obj2.margin = margin` and leaves the prism's at 0
    # (engine_collision_convex.c:1246). Everything else (tolerances, the
    # `+ ccd_margin` add-back) is unchanged; the hfield caller subtracts it.
    inflate1: Bool = True,
) -> Tuple[
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
]:
    """GJK distance + EPA penetration depth between two convex shapes
    (verbatim from gjk_epa_gpu; mesh reads via record tensor).

    `wf1` / `wf2` receive the SUPPORT POINTS, on geom 1 and geom 2
    respectively, of the three vertices of the EPA face the answer came from —
    MuJoCo's `pt->verts[face->verts[i]].vert1 / .vert2`. `wx` receives the two
    witness points themselves (`status->x1`, then `status->x2`), whose
    difference is MuJoCo's `dir`. `wf_ok` is set to 1 when they are valid and
    left at 0 on every early return.

    ⚠ `dir` IS NOT `normal * depth` AND MUST NOT BE REBUILT THAT WAY. The
    returned normal has already been negated into the `gi -> gj` convention
    callers expect, while `multicontact` wants `x2 - x1` in the reference's own
    orientation; deriving one from the other is a sign guess in a routine where
    a sign error silently mirrors the whole manifold.

    ⚠ THAT FACE IS THE ONLY THING THAT IDENTIFIES THE CONTACT FEATURE, which is
    why it is plumbed out rather than recomputed. `multicontact` has to know
    whether the contact came from a vertex, an edge or a face of each geom, and
    the ONLY evidence for that is which support points the winning face's three
    vertices resolved to: three distinct points mean a face, two mean an edge,
    one means a vertex. It cannot be recovered from the returned witness point,
    which is already a barycentric BLEND of all three.

    Callers that do not need the feature use `gjk_epa`, which wraps this.
    """
    wf_ok = 0
    # ⚠ ONE HILL-CLIMB SEED PER OBJECT, LIVING FOR THE WHOLE GJK+EPA RUN. This
    # is MuJoCo's `mjCCDObj.meshindex`, whose lifetime is exactly one
    # `mjc_initCCDObj` .. collision call — so the scope here matches by
    # construction rather than by luck. Every support request inside GJK, inside
    # `_gjk_intersect`, and inside EPA resumes from where the last one landed;
    # -1 means "no previous answer", and `_support_mesh` starts at vertex 0.
    # ⚠ THEY MUST NOT BE SWAPPED at a call site: object 1's index is a local
    # vertex id in object 1's mesh and means nothing in object 2's. Crossing
    # them is not a correctness bug (the guard in `_support_mesh` clamps an
    # out-of-range seed, and any in-range seed still converges) but it throws
    # the speed-up away silently, which is the worse failure to debug.
    # The seeds the caller handed in (the cross-step slot, or -1): every
    # phase of the run starts its walks there.
    var seed1 = warm1
    var seed2 = warm2
    # ===== GJK Phase =====
    # ⚠⚠ TWO PHASES, AND THE FIRST IS WHERE EVERY SPHERE AND CAPSULE PAIR IS
    # ACTUALLY SOLVED. `mjc_ccd` opens with
    #
    #   if (obj1 or obj2 is SPHERE or CAPSULE) {
    #     full_margin_i = size[0] + 0.5*margin_i;  support -> point/line;
    #     margin_i = 0;  dist_cutoff += full_margin1 + full_margin2;
    #     gjk(...);  restore;
    #     if (dist > tolerance) { inflate(...); ...; return dist; }
    #     // deep penetration only: reset and fall through to GJK + EPA
    #     gjk_iterations = 0;  x1 = center1;  x2 = center2;
    #   }
    #   gjk(status, obj1, obj2);
    #
    # so a sphere is a POINT and a capsule a LINE, the radii come back through
    # `inflate()`, and EPA is reached only when the shrunken shapes themselves
    # overlap. This engine ran the ROUNDED supports straight into GJK + EPA,
    # which is a smooth Minkowski boundary where the reference had an exact
    # point-or-line distance query.
    #
    # ⚠ MEASURED: it is what `test_hfield_vs_mujoco`'s sphere and capsule need
    # (14 contacts against MuJoCo's 15 at float32 without it) and what
    # `test_narrow_phase_pairs`' capsule/cylinder manifold needs (2 perturbed
    # contacts against 5) — MuJoCo's perturbed re-queries move the witness
    # point off a LINE, ours slid along a rounded surface and were rejected as
    # non-distinct.
    var _sh1 = type1 == GEOM_SPHERE or type1 == GEOM_CAPSULE
    var _sh2 = type2 == GEOM_SPHERE or type2 == GEOM_CAPSULE
    var full_m1 = Scalar[DTYPE](0)
    var full_m2 = Scalar[DTYPE](0)
    if _sh1:
        full_m1 = r1 + (
            Scalar[DTYPE](0.5) * ccd_margin if inflate1 else Scalar[DTYPE](0)
        )
    if _sh2:
        full_m2 = r2 + Scalar[DTYPE](0.5) * ccd_margin
    var shrunk = _sh1 or _sh2
    var _g_t0: Int = 0
    var _g_t1: Int = 0
    var _g_t2: Int = 0
    var _g_kg: Int = 0
    var _g_ke: Int = 0
    var _g_gap: Float64 = 0
    var _g_upper: Float64 = 0
    comptime if _GJK_PROBE:
        _g_t0 = Int(perf_counter_ns())

    # ⚠ GJK'S SIMPLEX LIVES IN THE ROW, NOT ON THE STACK — see
    # `ccd_workspace.mojo`. It is zeroed here because `mjc_ccd` starts each
    # phase from a fresh `Vertex sv(ws, wrow, SPX, 0, 4)`.
    comptime SPX = CCD_WS_SPX
    for _c in range(4 * SPX_STRIDE):
        ws[wrow, SPX + _c] = Scalar[DTYPE](0)
    var nsimplex = 0

    # ⚠⚠ `x_k` STARTS AS THE DIFFERENCE OF THE TWO GEOM CENTRES, NOT AS A
    # SUPPORT POINT, AND THE FIRST SUPPORT IS TAKEN ALONG ITS NEGATION.
    #
    #     obj1->center(status->x1, obj1);  obj2->center(status->x2, obj2);
    #     sub3(x_k, x1_k, x2_k);
    #     ... gjkSupport(simplex + n, obj1, obj2, x_k, x_norm)
    #     -> dir_neg = x_k/|x_k|;  dir = -dir_neg
    #
    # This engine used to take a support along `+(p1 - p2)/|p1 - p2|` OUTSIDE
    # the loop and seed `v` with the point it returned — the OPPOSITE side of
    # the Minkowski difference, and one iteration ahead of the reference. Every
    # support after that was consistent (`-v/|v|`), so the loop looked right;
    # it just descended from a different starting vertex, and GJK's answer is
    # path-dependent whenever more than one face is within tolerance.
    #
    # ⚠ MEASURED, `test_narrow_phase_pairs`'s box/cylinder group: an
    # independent numpy replica of `gjk()` + `polytope3` + `epa()` reproduces
    # MuJoCo's contact to every printed digit, and its seed polytope's six face
    # distances were 2.95e-07 1.88e-05 4.51e-04 ... against ours 2.20e-05
    # 7.85e-06 7.56e-04 — different from the FIRST LINE, on the same pose, with
    # the subdistance already ported. The seed is where it parted.
    # ⚠⚠ THE WHOLE GJK RUN IS A PHASE. It executes once for an ordinary pair
    # and TWICE when a sphere or capsule is involved: first on the shrunken
    # shapes, then — only if those overlap — on the real ones. The declarations
    # below sit outside the phase loop because the classification after it
    # reads them.
    var vx = Scalar[DTYPE](0)
    var vy = Scalar[DTYPE](0)
    var vz = Scalar[DTYPE](0)
    var lam = Array[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    var min_norm2 = _gjk_min_norm2[DTYPE](type1, type2, ccd_margin, ccd_tol)
    var dist_sq = Scalar[DTYPE](0)
    var dist = Scalar[DTYPE](0)

    # ⚠⚠ A BOUNDED `for`, NOT `while True`. There are at most two phases — the
    # shrunken one and the real one — and spelling that as an unbounded loop
    # with a `break` and a `return` inside gave the METAL back end a control
    # flow it miscompiled: the heightfield GPU leg returned 0, 7, 11, 12, 14,
    # 16 or 17 contacts depending on unrelated edits while the float32 CPU leg
    # sat at the correct 15 throughout. The trip count is what fixed it.
    # ⚠ The baseline kernel was NOT near a stack ceiling — doubling
    # `MC_MAX_DEG` there changes nothing — so this was never a size problem.
    for _phase in range(2):
        nsimplex = 0
        warm1 = seed1
        warm2 = seed2
        vx = p1x - p2x
        vy = p1y - p2y
        vz = p1z - p2z


        # `lambda` rides OUTSIDE the loop because the witness points at the end are
        # `lincomb(lambda, n, ...)` over whatever the last completed iteration
        # left. `{1, 0, 0, 0}` is the reference's initial value, which is what a
        # break on the very first iteration uses.
        lam = Array[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
        lam[0] = Scalar[DTYPE](1)

        # MuJoCo's `min_norm2` (`engine_collision_gjk.c:218`), NOT a constant — see
        # `_gjk_min_norm2`. The old hardcoded `GJK_TOLERANCE` was a 1e-5 m floor.
        min_norm2 = _gjk_min_norm2[DTYPE](type1, type2, ccd_margin, ccd_tol)

        # `backup_gjk = !get_dist`, and `mjc_Convex` sets `dist_cutoff = 0`, so it
        # is ON — but the reference clears it after ONE call
        # (`k = status->gjk_iterations; backup_gjk = 0;`), so `gjkIntersect` is
        # tried at most once per GJK run. Ours re-entered it every time the simplex
        # reached four vertices.
        var backup_gjk = True
        for k in range(GJK_MAX_ITERATIONS):
            comptime if _GJK_PROBE:
                _g_kg += 1
            var v_dot_v = vx * vx + vy * vy + vz * vz
            if v_dot_v < min_norm2:
                break

            var inv_vlen = Scalar[DTYPE](1) / sqrt(v_dot_v)
            var ndx = -vx * inv_vlen
            var ndy = -vy * inv_vlen
            var ndz = -vz * inv_vlen

            var sn = _minkowski_support[DTYPE, NPRISM=NPRISM](
                type1,
                p1x,
                p1y,
                p1z,
                q1x,
                q1y,
                q1z,
                q1w,
                r1,
                hl1,
                hx1,
                hy1,
                hz1,
                mesh_verts,
                mesh_vert_edgeadr,
                mesh_edges,
                va1,
                mnv1,
                type2,
                p2x,
                p2y,
                p2z,
                q2x,
                q2y,
                q2z,
                q2w,
                r2,
                hl2,
                hx2,
                hy2,
                hz2,
                va2,
                mnv2,
                ndx,
                ndy,
                ndz,
                warm1,
                warm2,
                prism,
                ccd_margin,
                shrunk and _sh1,
                shrunk and _sh2,
                inflate1,
            )

            var w_dot = sn[0] * ndx + sn[1] * ndy + sn[2] * ndz
            var v_dot = vx * ndx + vy * ndy + vz * ndz

            # ── CUTOFF EXIT — MuJoCo's `dist_cutoff` arm of `mj_gjk`
            # (`engine_collision_gjk.c:225`). `nd = -v/|v|`, so `-w_dot` is
            # `dot(w, v)/|v|`, the standard GJK LOWER BOUND on the distance from
            # the origin to the Minkowski difference. Once that bound reaches the
            # cutoff the pair is proven at least `cutoff` apart and no further
            # iteration can change the caller's `dist < margin` answer.
            #
            # ⚠⚠ THIS IS SAFE WHERE THE `gi == 0` CERTIFICATE BELOW WAS NOT, AND
            # THE DIFFERENCE IS THE BOUND. That branch proved "separated" and
            # returned 1e30, which is only equivalent to "no contact" when the
            # margin is 0 — with a margin it silently lost every contact in the
            # band, 0 against MuJoCo's 5 (see the comment there). This exits only
            # when a LOWER BOUND on the true distance has reached the very
            # threshold the caller compares against, so it can cost iterations,
            # never a contact. A penetrating pair has the origin inside, hence
            # `dot(w, v) < 0` and `-w_dot < 0`, so it can never fire on one.
            #
            # ⚠ MuJoCo's OTHER early-out (`!get_dist`, one branch up) returns on
            # ANY separating hyperplane. That one is safe only because
            # `mjc_penetration` INFLATES both geoms by margin first, which we have
            # never ported — do not copy it. See
            # `feedback_copying_control_flow_without_its_precondition`.
            #
            # Measured, SO-ARM101: GJK runs ~15 iterations per call converging to a
            # distance nobody reads; its 4 pairs sit 0.9-7.6 cm apart with margin 0.
            # ⚠ `status->dist_cutoff += full_margin1 + full_margin2` for the
            # shrunken phase and is restored afterwards: the shrunken shapes
            # are farther apart than the real ones by exactly the two radii, so
            # the cutoff has to grow by the same amount or this exit would give
            # up on a pair that is genuinely in contact.
            var _cut = dist_cutoff
            if shrunk and dist_cutoff >= Scalar[DTYPE](0):
                _cut = dist_cutoff + full_m1 + full_m2
            if (
                dist_cutoff >= Scalar[DTYPE](0)
                and ccd_margin - w_dot >= _cut
            ):
                wf_ok = 0
                return (ccd_margin - w_dot, Scalar[DTYPE](0), Scalar[DTYPE](0),
                        Scalar[DTYPE](0), ndx, ndy, ndz)
            # ⚠⚠ THE FLOAT32 FLOOR IS NOT A LOOSENING, IT IS WHAT MAKES THE TEST
            # ABLE TO FIRE AT ALL. `w_dot - v_dot` is a difference of two dot
            # products of magnitude |v|, so at float32 its rounding floor is about
            # `1e-7 * |v|` — for robot-scale geometry, HUNDREDS of times above
            # `GJK_TOLERANCE = 1e-10`. Without the relative term the comparison
            # never succeeds, GJK runs to `GJK_MAX_ITERATIONS`, and it returns
            # whatever it is holding at the cap. Measured over a 256-pose sweep of
            # two real hulls, that produced three PHANTOM CONTACTS: `-0.0` returned
            # for pairs float64 places 7.2, 16.5 and 16.9 cm apart. A phantom
            # contact at 17 cm hands the solver a constraint row out of nowhere and
            # is indistinguishable downstream from a real one.
            #
            # ⚠ MUJOCO'S THRESHOLD HERE IS ZERO, AND THAT IS NOT A LOOPHOLE WE CAN
            # COPY. `engine_collision_gjk.c` sets it to 0 for mesh pairs because
            # "if both geoms are discrete, finite convergence is guaranteed" — an
            # exact-arithmetic guarantee that float64 nearly honours and float32
            # does not. It is relative to |v| so it is scale-invariant, unlike the
            # absolute constant it adds to.
            #
            # ⚠ FLOAT64 IS UNTOUCHED — the term is exactly 0 there, so every
            # float64 gate is bit-identical. Which also means none of them covers
            # this; `test_gjk_float32_no_phantom_contacts.mojo` does, and it is RED
            # without the term.
            # ⚠ THE SUPPORT IS WRITTEN INTO SLOT `n` AND `n` IS NOT INCREMENTED.
            # `gjkSupport(simplex + n, ...)` puts it there and the reference then
            # passes `n + 1` to `subdistance`; `n` only advances when the
            # compaction below decides how many vertices survived. Incrementing
            # here instead — which this loop used to do — makes every test between
            # this point and the compaction read a count one too high.
            var si = nsimplex
            set_sv(ws, wrow, SPX, si, 0, sn[0])
            set_sv(ws, wrow, SPX, si, 1, sn[1])
            set_sv(ws, wrow, SPX, si, 2, sn[2])
            set_sv(ws, wrow, SPX, si, 3, sn[3])
            set_sv(ws, wrow, SPX, si, 4, sn[4])
            set_sv(ws, wrow, SPX, si, 5, sn[5])
            set_sv(ws, wrow, SPX, si, 6, sn[6])
            set_sv(ws, wrow, SPX, si, 7, sn[7])
            set_sv(ws, wrow, SPX, si, 8, sn[8])
            set_sv(ws, wrow, SPX, si, 9, Scalar[DTYPE](warm1))
            set_sv(ws, wrow, SPX, si, 10, Scalar[DTYPE](warm2))

            var fw_gap = v_dot_v - (sn[0] * vx + sn[1] * vy + sn[2] * vz)
            if fw_gap < _gjk_epsilon[DTYPE](
                type1, type2, ccd_margin, ccd_tol, v_dot_v
            ):
                # `if (!k) n = 1;` — a first-iteration convergence leaves the
                # reference holding exactly one vertex, and `mjc_ccd` then refuses
                # to build a polytope from it (`nsimplex > 1`).
                if k == 0:
                    nsimplex = 1
                break

            # ⚠ THE SIMPLEX IS AT FOUR VERTICES — REFINE IT BEFORE REDUCING IT.
            # MuJoCo calls `gjkIntersect` at exactly this point
            # (`engine_collision_gjk.c:238`, `if (n == 3 && backup_gjk)`), BEFORE
            # the distance subalgorithm runs, because that subalgorithm's job is to
            # find the closest FEATURE and it will happily drop back to a triangle
            # or an edge. For a SEPARATED pair that is what you want; for a
            # PENETRATING one it discards the enclosure and hands EPA a seed that
            # does not contain the origin, which is the single root cause behind
            # three failed attempts to use EPA on penetrating primitives.
            if nsimplex == 3 and backup_gjk:
                var gi = _gjk_intersect[DTYPE, NPRISM=NPRISM](
                    ws, wrow, SPX,
                    type1, p1x, p1y, p1z, q1x, q1y, q1z, q1w,
                    r1, hl1, hx1, hy1, hz1, mesh_verts, mesh_vert_edgeadr, mesh_edges, va1, mnv1,
                    type2, p2x, p2y, p2z, q2x, q2y, q2z, q2w,
                    r2, hl2, hx2, hy2, hz2, va2, mnv2,
                    warm1, warm2,
                    prism,
                    ccd_margin,
                    shrunk and _sh1,
                    shrunk and _sh2,
                    inflate1,
                )
                if gi == 1:
                    # enclosed, and `simplex` now holds a valid tetrahedron.
                    # ⚠ THE REFERENCE RETURNS HERE with `status->nsimplex = 4` set
                    # by `gjkIntersect` itself, so the count has to be raised
                    # explicitly now that the loop no longer pre-increments it.
                    nsimplex = 4
                    vx = Scalar[DTYPE](0)
                    vy = Scalar[DTYPE](0)
                    vz = Scalar[DTYPE](0)
                    break
                elif gi == 0:
                    # ⚠⚠ PROVEN SEPARATED IS NOT THE SAME AS "FAR ENOUGH TO IGNORE",
                    # AND THIS USED TO RETURN 1e30 ON THAT ASSUMPTION. The reasoning
                    # was "no collision, so report a separation the caller's
                    # `dist < margin` test will reject" — which is right only when
                    # `margin` is 0. With a margin, a pair separated by LESS than it
                    # is a contact MuJoCo reports and we returned nothing for.
                    #
                    # Measured, cylinder over a box, `margin=0.01` on each
                    # (`includemargin` 0.02):
                    #
                    #     gap      ours   MuJoCo
                    #     -0.0005    5       5
                    #      0.0       0       5    <- lost
                    #      0.0005    0       5    <- lost
                    #      0.001     0       5    <- lost
                    #      0.002     1       5
                    #      0.02      0       0
                    #
                    # i.e. it fired exactly in the band that matters and nowhere
                    # else — the certificate needs a 4-simplex, which only forms at
                    # SMALL separations, so distant pairs never reached this branch
                    # and paid nothing for it either.
                    #
                    # ⚠ MUJOCO NEVER NEEDS THE DISTANCE because `mjc_penetration`
                    # sets `dist_cutoff = 0` and INFLATES both geoms by `margin`
                    # instead (`mjc_initCCDObj(&obj, m, d, g, margin)`), so a
                    # within-margin pair reads as penetrating. We do not inflate, so
                    # our caller compares a real distance against `margin` — and
                    # that means this path MUST produce one.
                    #
                    # Falling through costs a few more GJK iterations on
                    # near-touching non-colliding pairs and keeps the certificate's
                    # real value: it still prevents EPA being handed a non-enclosing
                    # seed, because `gi == 1` is what gates that.
                    #
                    # ⚠ NOTHING IS SAVED OR RESTORED HERE, DELIBERATELY.
                    # `_gjk_intersect` rewrites the simplex in place, so the obvious
                    # move is to snapshot it and put it back — and a 36-element
                    # per-thread `Array` to do that is exactly the shape that
                    # silently miscomputes on Metal
                    # (`feedback_metal_wide_per_thread_inlinearray_miscompute`).
                    # It was written that way first and cost CPU-vs-GPU parity:
                    # 9.5e-06 on `test_narrow_phase_pairs_gpu_matches_cpu`, which
                    # must be bit-exact. The snapshot is also unnecessary — every
                    # vertex `_gjk_intersect` writes is a genuine support point, so
                    # what it leaves behind is a valid simplex for the distance
                    # subalgorithm, just a different one. Verified: identical
                    # contact counts and distances across the whole margin band
                    # with and without the restore.
                    pass
                # gi == -1: inconclusive, fall through to the subalgorithm below.
                # Either way the reference clears `backup_gjk` on the way out.
                backup_gjk = False

            # `subdistance` — barycentric coordinates of the closest point.
            # ⚠ IT DOES NOT REDUCE THE SIMPLEX; the compaction below is the
            # reference's own, and the retention test is `lambda[i] != 0` EXACTLY.
            lam = _subdistance[DTYPE](ws, wrow, SPX, nsimplex + 1)

            var keep = 0
            for i in range(4):
                if lam[i] == Scalar[DTYPE](0):
                    continue
                if keep != i:
                    for c in range(SPX_STRIDE):
                        set_sv(
                            ws, wrow, SPX, keep, c, sv(ws, wrow, SPX, i, c)
                        )
                    lam[keep] = lam[i]
                keep += 1
            # "SHOULD NOT OCCUR"
            if keep < 1:
                wf_ok = 0
                return (
                    Scalar[DTYPE](1e30), Scalar[DTYPE](0), Scalar[DTYPE](0),
                    Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](0),
                    Scalar[DTYPE](1),
                )
            nsimplex = keep

            # the next iterate, `lincomb(lambda, n, simplex)`
            var xnx = Scalar[DTYPE](0)
            var xny = Scalar[DTYPE](0)
            var xnz = Scalar[DTYPE](0)
            for i in range(nsimplex):
                xnx += lam[i] * sv(ws, wrow, SPX, i, 0)
                xny += lam[i] * sv(ws, wrow, SPX, i, 1)
                xnz += lam[i] * sv(ws, wrow, SPX, i, 2)

            # ⚠⚠ `equal3(x_next, x_k)` — "x_k has converged to minimum". This break
            # leaves `x_k` AND `x_norm` at the values the top of the iteration
            # computed, which is why `vx/vy/vz` are not updated before it. The
            # routine this replaced had no equivalent: it could not compare against
            # the previous iterate because it returned a point rather than lambda.
            if (
                abs(xnx - vx) < Scalar[DTYPE](MJ_MINVAL)
                and abs(xny - vy) < Scalar[DTYPE](MJ_MINVAL)
                and abs(xnz - vz) < Scalar[DTYPE](MJ_MINVAL)
            ):
                break
            vx = xnx
            vy = xny
            vz = xnz

            if nsimplex == 4:
                vx = Scalar[DTYPE](0)
                vy = Scalar[DTYPE](0)
                vz = Scalar[DTYPE](0)
                break


        dist_sq = vx * vx + vy * vy + vz * vz
        dist = sqrt(dist_sq)
        if not shrunk:
            break

        # `if (status->dist > status->tolerance) { inflate(...); return; }` —
        # the shrunken shapes are APART, so the contact is entirely described
        # by their distance and the two radii, and EPA never runs.
        if dist > ccd_tol:
            var i1x = Scalar[DTYPE](0)
            var i1y = Scalar[DTYPE](0)
            var i1z = Scalar[DTYPE](0)
            var i2x = Scalar[DTYPE](0)
            var i2y = Scalar[DTYPE](0)
            var i2z = Scalar[DTYPE](0)
            for i in range(nsimplex):
                i1x += lam[i] * sv(ws, wrow, SPX, i, 3)
                i1y += lam[i] * sv(ws, wrow, SPX, i, 4)
                i1z += lam[i] * sv(ws, wrow, SPX, i, 5)
                i2x += lam[i] * sv(ws, wrow, SPX, i, 6)
                i2y += lam[i] * sv(ws, wrow, SPX, i, 7)
                i2z += lam[i] * sv(ws, wrow, SPX, i, 8)
            # `inflate`: n = normalize(x2 - x1); x1 += m1*n; x2 -= m2*n;
            #            dist -= (m1 + m2)
            var inx = i2x - i1x
            var iny = i2y - i1y
            var inz = i2z - i1z
            var iln = sqrt(inx * inx + iny * iny + inz * inz)
            if iln > Scalar[DTYPE](0):
                inx /= iln
                iny /= iln
                inz /= iln
            i1x += full_m1 * inx
            i1y += full_m1 * iny
            i1z += full_m1 * inz
            i2x -= full_m2 * inx
            i2y -= full_m2 * iny
            i2z -= full_m2 * inz
            dist -= full_m1 + full_m2
            # The normal is `normalize(x1 - x2)` on the INFLATED witnesses,
            # exactly as `mjc_penetration` builds it for the EPA path.
            var fnx = i1x - i2x
            var fny = i1y - i2y
            var fnz = i1z - i2z
            var fln = sqrt(fnx * fnx + fny * fny + fnz * fnz)
            if fln > Scalar[DTYPE](0):
                fnx /= fln
                fny /= fln
                fnz /= fln
            else:
                fnx = Scalar[DTYPE](0)
                fny = Scalar[DTYPE](0)
                fnz = Scalar[DTYPE](1)
            wf_ok = 0
            return (
                dist + ccd_margin,
                (i1x + i2x) * Scalar[DTYPE](0.5),
                (i1y + i2y) * Scalar[DTYPE](0.5),
                (i1z + i2z) * Scalar[DTYPE](0.5),
                fnx, fny, fnz,
            )

        # Deep penetration: reset and re-run on the REAL shapes.
        shrunk = False

    # Classify with the SAME quantity the loop converges on. `GJK_TOLERANCE` is
    # a threshold on |v|^2 in the loop above (`v_dot_v < GJK_TOLERANCE`), so
    # comparing |v| against it here meant any run that exited by converging to
    # the origin — |v| anywhere in [1e-10, 1e-5) — was then reported SEPARATED
    # at a hair's-breadth positive distance, and dropped by the caller's
    # `dist < margin` test. A deep penetration would vanish outright: the two
    # exits that mean "origin reached" (tetrahedron enclosed, |v| -> 0) have to
    # agree, or a single ULP decides whether a contact exists at all.
    # ⚠⚠ THE EPA GATE IS `dist <= ccd_tolerance`, NOT `dist^2 > min_norm2`.
    # `mjc_ccd` reads
    #
    #     if (status->dist <= config->tolerance && status->nsimplex > 1 ...)
    #
    # while `min_norm2` is only the GJK LOOP's break. This engine used the loop
    # threshold for both, and the two are the same number only for a SMOOTH
    # pair (`tol^2` either way). For a DISCRETE pair `min_norm2` is
    # `mjMINVAL^2`, so anything GJK left further than 1e-15 from the origin was
    # reported SEPARATED where the reference would have run EPA on it — a lost
    # contact whenever GJK's own convergence floor sits above 1e-15, which at
    # float32 is always.
    comptime if _GJK_PROBE:
        _g_t1 = Int(perf_counter_ns())
    if dist > ccd_tol:
        # Separated
        # ⚠⚠ `lincomb(x1_k, lambda, n, ...)`, NOT A UNIFORM AVERAGE. This used
        # to divide the SUM of the surviving support points by their count,
        # which is the barycentric combination only when every coordinate
        # happens to be equal. The reference weights by `lambda`, and those are
        # the coordinates of the closest point — the whole output of
        # `subdistance`, which the routine this replaced discarded.
        var w1x = Scalar[DTYPE](0)
        var w1y = Scalar[DTYPE](0)
        var w1z = Scalar[DTYPE](0)
        var w2x = Scalar[DTYPE](0)
        var w2y = Scalar[DTYPE](0)
        var w2z = Scalar[DTYPE](0)
        for i in range(nsimplex):
            w1x += lam[i] * sv(ws, wrow, SPX, i, 3)
            w1y += lam[i] * sv(ws, wrow, SPX, i, 4)
            w1z += lam[i] * sv(ws, wrow, SPX, i, 5)
            w2x += lam[i] * sv(ws, wrow, SPX, i, 6)
            w2y += lam[i] * sv(ws, wrow, SPX, i, 7)
            w2z += lam[i] * sv(ws, wrow, SPX, i, 8)
        var cx = (w1x + w2x) * Scalar[DTYPE](0.5)
        var cy = (w1y + w2y) * Scalar[DTYPE](0.5)
        var cz = (w1z + w2z) * Scalar[DTYPE](0.5)
        # ⚠ SIGN: this function returns `gi -> gj`, the convention EVERY caller
        # applies (the emit then negates once more to reach the record's
        # `body_b -> body_a`). GJK's `v` and EPA's face normal both point OUT of
        # the Minkowski difference, which is `gj -> gi`, so both return sites
        # negate. See the note at the EPA return.
        var nx = -vx / dist
        var ny = -vy / dist
        var nz = -vz / dist
        # ⚠⚠ `+ ccd_margin` — `mjc_Convex` REPORTS `margin + dist`. Every
        # distance this function computes is now the INFLATED pair's (each geom
        # grown by half the pair margin, see `_minkowski_support`), and MuJoCo
        # adds the margin back before storing it
        # (engine_collision_convex.c:115). With margin 0 the term vanishes and
        # every margin-free model is bit-identical.
        comptime if _GJK_PROBE:
            print("[gprobe]", "sep", 1, "gjk", _g_t1 - _g_t0, "kg", _g_kg, "poly", 0, "epa", 0, "ke", 0)
        return (dist + ccd_margin, cx, cy, cz, nx, ny, nz)

    # ===== EPA Phase =====
    # A one-for-one port of `polytope2` / `polytope3` / `polytope4` and `epa()`
    # (`engine_collision_gjk.c`). The polytope itself lives in `epa.mojo`; only
    # the three seed builders and the expansion loop are here, because they are
    # the only parts that call the support function.
    #
    # ⚠⚠ FOUR THINGS THE EPA THIS REPLACED DID THAT MuJoCo DOES NOT, all of
    # which change WHICH boundary face is returned and therefore the contact
    # NORMAL:
    #
    #   1. it rescanned EVERY face for the closest one; MuJoCo scans `map`, a
    #      candidate list a face joins only when `lower2 <= dist2 <= upper2`;
    #   2. it marked every globally visible face and compacted the array;
    #      MuJoCo walks `Face::adj` out from the closest face and never reuses
    #      a slot;
    #   3. it compared the CURRENT iteration's upper bound against the lower
    #      one; MuJoCo keeps `upper` as a running MINIMUM, so it stops no later
    #      and usually sooner;
    #   4. it had no "the closest face got worse, return the previous one" rule.
    #
    # ⚠⚠ AND THREE THINGS IT DID THAT HAVE NO COUNTERPART AT ALL, now gone: an
    # axis-aligned OCTAHEDRON seed when GJK's simplex did not enclose the
    # origin, a post-hoc UPPER-BOUND CHECK over seven sampled directions, and a
    # CENTRE-LINE normal with a support-extent depth. Each was covering for a
    # missing `polytope2/3/4`; the reference's answer in all three situations is
    # `status->dist = 0`, which `mjc_penetration` records as NO CONTACT because
    # it only writes a contact when `dist < 0`.
    #
    # ⚠ `status->dist = 0` IS SET BEFORE THE SEED IS BUILT, which is why
    # `polytope3`'s `status->dist > 10*mjMINVAL` guard around `testTetra` can
    # never fire from `mjc_ccd`. It is transcribed below as the dead branch it
    # is rather than silently dropped.

    var nverts = 0
    var nfaces = 0
    var nmap = 0
    var nhor = 0
    # `mjc_ccd`'s `ret`: 0 is "simplex not on boundary, run EPA", anything else
    # is one of the `mjEPA_*` codes and means no polytope and no contact.
    var ret = 0
    var need3 = False

    var md23 = epa_mindist[DTYPE]()
    var md4 = epa_mindist4[DTYPE]()
    var mval = epa_minval[DTYPE]()

    if nsimplex < 2:
        # `mjc_ccd` requires `status->nsimplex > 1` before it builds anything.
        ret = -1

    # ── polytope4 — GJK ended enclosing the origin ────────────────────────
    elif nsimplex >= 4:
        # ⚠ STAGE B (2026-09-26): the seeds carry their support indices.
        # MuJoCo's `insertVertex` copies `index1`/`index2` out of the GJK
        # simplex; ours was nine floats wide and the seeds got -1 here, which
        # left multicontact scanning the whole hull for every seed vertex of a
        # winning face. The simplex is now eleven wide, laid out exactly like
        # an EPA vertex, so this copy carries them — the same at polytope2/3.
        comptime assert SPX_STRIDE == EPA_V_STRIDE, (
            "a simplex vertex must copy into an EPA vertex slot for slot"
        )
        for i in range(4):
            for k in range(SPX_STRIDE):
                set_ev(ws, wrow, i, k, sv(ws, wrow, SPX, i, k))
        nverts = 4
        var c4x = Scalar[DTYPE](0)
        var c4y = Scalar[DTYPE](0)
        var c4z = Scalar[DTYPE](0)
        for i in range(4):
            c4x += ev(ws, wrow, i, 0)
            c4y += ev(ws, wrow, i, 1)
            c4z += ev(ws, wrow, i, 2)
        set_center(
            ws, wrow,
            c4x * Scalar[DTYPE](0.25),
            c4y * Scalar[DTYPE](0.25),
            c4z * Scalar[DTYPE](0.25),
        )

        # ⚠⚠ THE FOUR CALLS ARE WRITTEN OUT, NOT DRIVEN BY A TABLE, and that is
        # a GPU requirement rather than a style choice: a per-thread
        # `Array[Int, 24]` indexed by a RUNTIME loop variable is the
        # shape that silently miscomputes in a Metal kernel
        # (`feedback_metal_wide_per_thread_inlinearray_miscompute`). The
        # reference writes them out too.
        var fail4 = -1
        var d = attach_face(ws, wrow, 0, 0, 1, 2, 1, 3, 2)
        nfaces = 1
        if d < md4:
            fail4 = 0
        else:
            d = attach_face(ws, wrow, 1, 0, 3, 1, 2, 3, 0)
            nfaces = 2
            if d < md4:
                fail4 = 1
            else:
                d = attach_face(ws, wrow, 2, 0, 2, 3, 0, 3, 1)
                nfaces = 3
                if d < md4:
                    fail4 = 2
                else:
                    d = attach_face(ws, wrow, 3, 3, 2, 1, 2, 0, 1)
                    nfaces = 4
                    if d < md4:
                        fail4 = 3

        if fail4 >= 0:
            # `replaceSimplex3` — the origin is on this face, so drop to the
            # 2-simplex it spans and rebuild with `polytope3`. The triples are
            # the first three arguments of the call that failed:
            #   0 -> (0, 1, 2)   1 -> (0, 3, 1)   2 -> (0, 2, 3)   3 -> (3, 2, 1)
            # ⚠ ONE 9-FLOAT SCRATCH, and the write ORDER matters: two of the
            # four permutations read a slot they are about to overwrite.
            var ra = 0
            var rb = 1
            var rc = 2
            if fail4 == 1:
                ra = 0
                rb = 3
                rc = 1
            elif fail4 == 2:
                ra = 0
                rb = 2
                rc = 3
            elif fail4 == 3:
                ra = 3
                rb = 2
                rc = 1
            var tmp = Array[Scalar[DTYPE], SPX_STRIDE](fill=Scalar[DTYPE](0))
            for k in range(SPX_STRIDE):
                tmp[k] = sv(ws, wrow, SPX, rc, k)
            for k in range(SPX_STRIDE):
                set_sv(ws, wrow, SPX, 0, k, sv(ws, wrow, SPX, ra, k))
            for k in range(SPX_STRIDE):
                set_sv(ws, wrow, SPX, 1, k, sv(ws, wrow, SPX, rb, k))
            for k in range(SPX_STRIDE):
                set_sv(ws, wrow, SPX, 2, k, tmp[k])
            nsimplex = 3
            nverts = 0
            nfaces = 0
            nmap = 0
            need3 = True
        elif not test_tetra[DTYPE](
            ev(ws, wrow, 0, 0), ev(ws, wrow, 0, 1), ev(ws, wrow, 0, 2),
            ev(ws, wrow, 1, 0), ev(ws, wrow, 1, 1), ev(ws, wrow, 1, 2),
            ev(ws, wrow, 2, 0), ev(ws, wrow, 2, 1), ev(ws, wrow, 2, 2),
            ev(ws, wrow, 3, 0), ev(ws, wrow, 3, 1), ev(ws, wrow, 3, 2),
        ):
            ret = 1  # mjEPA_P4_MISSING_ORIGIN
        else:
            for i in range(4):
                set_emap(ws, wrow, i, i)
                set_efi(ws, wrow, i, i)
            nmap = 4

    # ── polytope2 — GJK ended on a segment ────────────────────────────────
    elif nsimplex == 2:
        var v1x = sv(ws, wrow, SPX, 0, 0)
        var v1y = sv(ws, wrow, SPX, 0, 1)
        var v1z = sv(ws, wrow, SPX, 0, 2)
        var v2x = sv(ws, wrow, SPX, 1, 0)
        var v2y = sv(ws, wrow, SPX, 1, 1)
        var v2z = sv(ws, wrow, SPX, 1, 2)
        set_center(
            ws, wrow,
            (v1x + v2x) * Scalar[DTYPE](0.5),
            (v1y + v2y) * Scalar[DTYPE](0.5),
            (v1z + v2z) * Scalar[DTYPE](0.5),
        )
        var dfx = v2x - v1x
        var dfy = v2y - v1y
        var dfz = v2z - v1z

        # the axis with the smallest component, so the cross product is largest
        var ex = Scalar[DTYPE](0)
        var ey = Scalar[DTYPE](0)
        var ez = Scalar[DTYPE](0)
        if abs(dfx) <= abs(dfy) and abs(dfx) <= abs(dfz):
            ex = Scalar[DTYPE](1)
        elif abs(dfy) <= abs(dfz):
            ey = Scalar[DTYPE](1)
        else:
            ez = Scalar[DTYPE](1)

        var d1x = ey * dfz - ez * dfy
        var d1y = ez * dfx - ex * dfz
        var d1z = ex * dfy - ey * dfx
        var rm = rotmat120[DTYPE](dfx, dfy, dfz)
        var d2x = rm[0] * d1x + rm[1] * d1y + rm[2] * d1z
        var d2y = rm[3] * d1x + rm[4] * d1y + rm[5] * d1z
        var d2z = rm[6] * d1x + rm[7] * d1y + rm[8] * d1z
        var d3x = rm[0] * d2x + rm[1] * d2y + rm[2] * d2z
        var d3y = rm[3] * d2x + rm[4] * d2y + rm[5] * d2z
        var d3z = rm[6] * d2x + rm[7] * d2y + rm[8] * d2z

        for i in range(2):
            for k in range(SPX_STRIDE):
                set_ev(ws, wrow, i, k, sv(ws, wrow, SPX, i, k))
        nverts = 2

        for s in range(3):
            var sx = d1x
            var sy = d1y
            var sz = d1z
            if s == 1:
                sx = d2x
                sy = d2y
                sz = d2z
            elif s == 2:
                sx = d3x
                sy = d3y
                sz = d3z
            var sn2 = sqrt(sx * sx + sy * sy + sz * sz)
            # `epaSupport` normalises by the norm it is handed and falls back
            # to (1, 0, 0) when that norm is below `mjMINVAL`.
            var ux = Scalar[DTYPE](1)
            var uy = Scalar[DTYPE](0)
            var uz = Scalar[DTYPE](0)
            if sn2 > mval:
                ux = sx / sn2
                uy = sy / sn2
                uz = sz / sn2
            var sp = _minkowski_support[DTYPE, NPRISM=NPRISM](
                type1, p1x, p1y, p1z, q1x, q1y, q1z, q1w,
                r1, hl1, hx1, hy1, hz1, mesh_verts, mesh_vert_edgeadr, mesh_edges, va1, mnv1,
                type2, p2x, p2y, p2z, q2x, q2y, q2z, q2w,
                r2, hl2, hx2, hy2, hz2, va2, mnv2,
                ux, uy, uz,
                warm1, warm2,
                prism,
                ccd_margin,
                inflate1=inflate1,
            )
            set_ev(ws, wrow, nverts, 0, sp[0])
            set_ev(ws, wrow, nverts, 1, sp[1])
            set_ev(ws, wrow, nverts, 2, sp[2])
            set_ev(ws, wrow, nverts, 3, sp[3])
            set_ev(ws, wrow, nverts, 4, sp[4])
            set_ev(ws, wrow, nverts, 5, sp[5])
            set_ev(ws, wrow, nverts, 6, sp[6])
            set_ev(ws, wrow, nverts, 7, sp[7])
            set_ev(ws, wrow, nverts, 8, sp[8])
            set_ev(ws, wrow, nverts, 9, Scalar[DTYPE](warm1))
            set_ev(ws, wrow, nverts, 10, Scalar[DTYPE](warm2))
            nverts += 1

        # Written out for the same reason as `polytope4`'s four.
        var fail2 = -1
        var d2a = attach_face(ws, wrow, 0, 0, 2, 3, 1, 3, 2)
        nfaces = 1
        if d2a < md23:
            fail2 = 0
        else:
            d2a = attach_face(ws, wrow, 1, 0, 4, 2, 2, 4, 0)
            nfaces = 2
            if d2a < md23:
                fail2 = 1
            else:
                d2a = attach_face(ws, wrow, 2, 0, 3, 4, 0, 5, 1)
                nfaces = 3
                if d2a < md23:
                    fail2 = 2
                else:
                    d2a = attach_face(ws, wrow, 3, 1, 3, 2, 5, 0, 4)
                    nfaces = 4
                    if d2a < md23:
                        fail2 = 3
                    else:
                        d2a = attach_face(ws, wrow, 4, 1, 2, 4, 3, 1, 5)
                        nfaces = 5
                        if d2a < md23:
                            fail2 = 4
                        else:
                            d2a = attach_face(
                                ws, wrow, 5, 1, 4, 3, 4, 2, 3
                            )
                            nfaces = 6
                            if d2a < md23:
                                fail2 = 5

        if fail2 >= 0:
            # The failing call's first three arguments. ⚠ These index the
            # POLYTOPE's vertices, not the simplex, so there is no aliasing.
            var sa = 0
            var sb = 2
            var sc = 3
            if fail2 == 1:
                sa = 0
                sb = 4
                sc = 2
            elif fail2 == 2:
                sa = 0
                sb = 3
                sc = 4
            elif fail2 == 3:
                sa = 1
                sb = 3
                sc = 2
            elif fail2 == 4:
                sa = 1
                sb = 2
                sc = 4
            elif fail2 == 5:
                sa = 1
                sb = 4
                sc = 3
            for k in range(SPX_STRIDE):
                set_sv(ws, wrow, SPX, 0, k, ev(ws, wrow, sa, k))
            for k in range(SPX_STRIDE):
                set_sv(ws, wrow, SPX, 1, k, ev(ws, wrow, sb, k))
            for k in range(SPX_STRIDE):
                set_sv(ws, wrow, SPX, 2, k, ev(ws, wrow, sc, k))
            nsimplex = 3
            nverts = 0
            nfaces = 0
            nmap = 0
            need3 = True
        elif ray_triangle[DTYPE](
            ev(ws, wrow, 0, 0), ev(ws, wrow, 0, 1), ev(ws, wrow, 0, 2),
            ev(ws, wrow, 1, 0), ev(ws, wrow, 1, 1), ev(ws, wrow, 1, 2),
            ev(ws, wrow, 2, 0), ev(ws, wrow, 2, 1), ev(ws, wrow, 2, 2),
            ev(ws, wrow, 3, 0), ev(ws, wrow, 3, 1), ev(ws, wrow, 3, 2),
            ev(ws, wrow, 4, 0), ev(ws, wrow, 4, 1), ev(ws, wrow, 4, 2),
        ) == 0:
            ret = 2  # mjEPA_P2_NONCONVEX
        else:
            for i in range(6):
                set_emap(ws, wrow, i, i)
                set_efi(ws, wrow, i, i)
            nmap = 6

    else:
        need3 = True

    # ── polytope3 — GJK ended on a triangle, or a seed dropped to one ─────
    if need3 and ret == 0:
        var w1x = sv(ws, wrow, SPX, 0, 0)
        var w1y = sv(ws, wrow, SPX, 0, 1)
        var w1z = sv(ws, wrow, SPX, 0, 2)
        var w2x = sv(ws, wrow, SPX, 1, 0)
        var w2y = sv(ws, wrow, SPX, 1, 1)
        var w2z = sv(ws, wrow, SPX, 1, 2)
        var w3x = sv(ws, wrow, SPX, 2, 0)
        var w3y = sv(ws, wrow, SPX, 2, 1)
        var w3z = sv(ws, wrow, SPX, 2, 2)
        set_center(
            ws, wrow,
            (w1x + w2x + w3x) / Scalar[DTYPE](3),
            (w1y + w2y + w3y) / Scalar[DTYPE](3),
            (w1z + w2z + w3z) / Scalar[DTYPE](3),
        )
        var a1x = w2x - w1x
        var a1y = w2y - w1y
        var a1z = w2z - w1z
        var a2x = w3x - w1x
        var a2y = w3y - w1y
        var a2z = w3z - w1z
        var tnx = a1y * a2z - a1z * a2y
        var tny = a1z * a2x - a1x * a2z
        var tnz = a1x * a2y - a1y * a2x
        var tn = sqrt(tnx * tnx + tny * tny + tnz * tnz)
        if tn < mval:
            ret = 3  # mjEPA_P3_BAD_NORMAL
        else:
            for i in range(3):
                for k in range(SPX_STRIDE):
                    set_ev(ws, wrow, i, k, sv(ws, wrow, SPX, i, k))
            nverts = 3

            # ⚠ v5 IS INSERTED BEFORE v4. The reference calls
            # `epaSupport(..., n_neg, ...)` first, so the support along -n
            # takes polytope index 3 and the one along +n takes index 4. Their
            # order decides every face index below.
            for s in range(2):
                var ux = -tnx / tn
                var uy = -tny / tn
                var uz = -tnz / tn
                if s == 1:
                    ux = tnx / tn
                    uy = tny / tn
                    uz = tnz / tn
                var sp = _minkowski_support[DTYPE, NPRISM=NPRISM](
                    type1, p1x, p1y, p1z, q1x, q1y, q1z, q1w,
                    r1, hl1, hx1, hy1, hz1, mesh_verts, mesh_vert_edgeadr, mesh_edges, va1, mnv1,
                    type2, p2x, p2y, p2z, q2x, q2y, q2z, q2w,
                    r2, hl2, hx2, hy2, hz2, va2, mnv2,
                    ux, uy, uz,
                    warm1, warm2,
                    prism,
                    ccd_margin,
                    inflate1=inflate1,
                )
                set_ev(ws, wrow, nverts, 0, sp[0])
                set_ev(ws, wrow, nverts, 1, sp[1])
                set_ev(ws, wrow, nverts, 2, sp[2])
                set_ev(ws, wrow, nverts, 3, sp[3])
                set_ev(ws, wrow, nverts, 4, sp[4])
                set_ev(ws, wrow, nverts, 5, sp[5])
                set_ev(ws, wrow, nverts, 6, sp[6])
                set_ev(ws, wrow, nverts, 7, sp[7])
                set_ev(ws, wrow, nverts, 8, sp[8])
                set_ev(ws, wrow, nverts, 9, Scalar[DTYPE](warm1))
                set_ev(ws, wrow, nverts, 10, Scalar[DTYPE](warm2))
                nverts += 1

            if tri_point_intersect[DTYPE](
                w1x, w1y, w1z, w2x, w2y, w2z, w3x, w3y, w3z,
                ev(ws, wrow, 4, 0), ev(ws, wrow, 4, 1), ev(ws, wrow, 4, 2),
            ):
                ret = 4  # mjEPA_P3_INVALID_V4
            elif tri_point_intersect[DTYPE](
                w1x, w1y, w1z, w2x, w2y, w2z, w3x, w3y, w3z,
                ev(ws, wrow, 3, 0), ev(ws, wrow, 3, 1), ev(ws, wrow, 3, 2),
            ):
                ret = 5  # mjEPA_P3_INVALID_V5
            else:
                # ⚠ THE `mjEPA_P3_MISSING_ORIGIN` GUARD IS DEAD FROM `mjc_ccd`.
                # It reads `status->dist > 10*mjMINVAL`, and `mjc_ccd` has
                # already written `status->dist = 0`. Transcribed as a comment
                # rather than as code that can never run.
                # Written out for the same reason as `polytope4`'s four.
                # `v1i, v2i, v3i` are 0, 1, 2 and — note the order —
                # `v5i` is 3 and `v4i` is 4.
                var d3 = attach_face(ws, wrow, 0, 4, 0, 1, 1, 3, 2)
                nfaces = 1
                if d3 < md23:
                    ret = 6  # mjEPA_P3_ORIGIN_ON_FACE
                else:
                    d3 = attach_face(ws, wrow, 1, 4, 2, 0, 2, 4, 0)
                    nfaces = 2
                    if d3 < md23:
                        ret = 6
                    else:
                        d3 = attach_face(ws, wrow, 2, 4, 1, 2, 0, 5, 1)
                        nfaces = 3
                        if d3 < md23:
                            ret = 6
                        else:
                            d3 = attach_face(ws, wrow, 3, 3, 1, 0, 5, 0, 4)
                            nfaces = 4
                            if d3 < md23:
                                ret = 6
                            else:
                                d3 = attach_face(
                                    ws, wrow, 4, 3, 0, 2, 3, 1, 5
                                )
                                nfaces = 5
                                if d3 < md23:
                                    ret = 6
                                else:
                                    d3 = attach_face(
                                        ws, wrow, 5, 3, 2, 1, 4, 2, 3
                                    )
                                    nfaces = 6
                                    if d3 < md23:
                                        ret = 6
                if ret == 0:
                    for i in range(6):
                        set_emap(ws, wrow, i, i)
                        set_efi(ws, wrow, i, i)
                    nmap = 6

    comptime if _GJK_PROBE:
        _g_t2 = Int(perf_counter_ns())
    # ── epa() ─────────────────────────────────────────────────────────────
    var face = -1
    var pface = -1
    var upper = Scalar[DTYPE](1e30)
    var upper2 = Scalar[DTYPE](1e30)
    var _epa_tol = _epa_tolerance[DTYPE](type1, type2, ccd_margin, ccd_tol)
    var _epa_iters = ccd_iter
    if _epa_iters > EPA_ITER_CAP:
        _epa_iters = EPA_ITER_CAP
    if _epa_iters < 1:
        _epa_iters = 1

    if ret == 0:
        for k in range(_epa_iters):
            comptime if _GJK_PROBE:
                _g_ke += 1
            pface = face

            # the face closest to the origin, over the CANDIDATE list
            var lower2 = Scalar[DTYPE](1e30)
            face = -1
            for i in range(nmap):
                var fi = emap(ws, wrow, i)
                var d2 = efd(ws, wrow, fi)
                if d2 < lower2:
                    face = fi
                    lower2 = d2

            # the lower bound moved backwards, or nothing is left
            if lower2 > upper2 or face < 0:
                face = pface
                break

            # `mju_warning("EPA: origin lies on affine hull of face")`
            if lower2 <= Scalar[DTYPE](0):
                break

            var lower = sqrt(lower2)
            var ux = Scalar[DTYPE](1)
            var uy = Scalar[DTYPE](0)
            var uz = Scalar[DTYPE](0)
            if lower > mval:
                ux = efv(ws, wrow, face, 0) / lower
                uy = efv(ws, wrow, face, 1) / lower
                uz = efv(ws, wrow, face, 2) / lower
            var w = _minkowski_support[DTYPE, NPRISM=NPRISM](
                type1, p1x, p1y, p1z, q1x, q1y, q1z, q1w,
                r1, hl1, hx1, hy1, hz1, mesh_verts, mesh_vert_edgeadr, mesh_edges, va1, mnv1,
                type2, p2x, p2y, p2z, q2x, q2y, q2z, q2w,
                r2, hl2, hx2, hy2, hz2, va2, mnv2,
                ux, uy, uz,
                warm1, warm2,
                prism,
                ccd_margin,
                inflate1=inflate1,
            )
            # ⚠ `epaSupport` INSERTS THE VERTEX BEFORE ANY TEST, and the
            # reference has no capacity check here because `5 + N` vertices
            # exactly covers a seed of five plus one per iteration. Ours is
            # sized by the same formula, so this can only fire if the two ever
            # drift apart.
            if nverts >= EPA_V_CAP:
                break
            var wi = nverts
            set_ev(ws, wrow, wi, 0, w[0])
            set_ev(ws, wrow, wi, 1, w[1])
            set_ev(ws, wrow, wi, 2, w[2])
            set_ev(ws, wrow, wi, 3, w[3])
            set_ev(ws, wrow, wi, 4, w[4])
            set_ev(ws, wrow, wi, 5, w[5])
            set_ev(ws, wrow, wi, 6, w[6])
            set_ev(ws, wrow, wi, 7, w[7])
            set_ev(ws, wrow, wi, 8, w[8])
            set_ev(ws, wrow, wi, 9, Scalar[DTYPE](warm1))
            set_ev(ws, wrow, wi, 10, Scalar[DTYPE](warm2))
            nverts += 1

            # ⚠⚠ `upper` IS A RUNNING MINIMUM ACROSS ITERATIONS, not the
            # current iteration's bound. It also gates which new faces are
            # allowed into `map` at all, through `upper2`.
            var upper_k = (
                efv(ws, wrow, face, 0) * w[0]
                + efv(ws, wrow, face, 1) * w[1]
                + efv(ws, wrow, face, 2) * w[2]
            ) / lower
            if upper_k < upper:
                upper = upper_k
                upper2 = upper * upper
            comptime if _GJK_PROBE:
                _g_gap = Float64(upper - lower)
                _g_upper = Float64(upper)
            if upper - lower < _epa_tol:
                # `if (k == 0 && upper < lower - 1e-10) face = NULL;` —
                # "terminate without contact when upper < lower on the first
                # iteration".
                #
                # ⚠⚠ THE 1e-10 IS ABSOLUTE AND LIVES IN A DOUBLE-PRECISION
                # WORLD. `upper` and `lower` are two distances of magnitude
                # `|v|`, so their difference has a float32 rounding floor near
                # `1e-7 * |v|` — for a geom 0.2 m from the origin that is
                # 2e-08, two hundred times the constant. A genuine contact
                # whose bounds cross by rounding then reads as NO CONTACT.
                # Measured on `test_hfield_vs_mujoco`: the box loses exactly
                # one of its three prism-face contacts at float32 and keeps all
                # three at float64, with `ret == 0`, `nfaces == 4` and
                # `face == -1` — EPA gave up here on its first iteration.
                #
                # ⚠ THE RELATIVE TERM IS FLOAT32-ONLY, so every float64 gate
                # is bit-identical. It is the same shape and the same reason as
                # `_gjk_epsilon`'s float32 floor.
                var _sep = Scalar[DTYPE](1e-10)
                comptime if DTYPE != DType.float64:
                    _sep += Scalar[DTYPE](1e-6) * lower
                if k == 0 and upper < lower - _sep:
                    face = -1
                break

            var vis_x = w[0]
            var vis_y = w[1]
            var vis_z = w[2]
            nhor = 0
            horizon(ws, wrow, face, vis_x, vis_y, vis_z, nmap, nhor)

            # "unrecoverable numerical issue; at least one face was deleted so
            # nedges is 3 or more"
            if nhor < 3:
                face = -1
                break

            var nf0 = nfaces
            var nedges = nhor
            if nedges > EPA_F_CAP - nfaces:
                # `mju_warning("EPA: out of memory for faces")` — the reference
                # keeps the face it has, which is a valid lower bound.
                break

            var bad = False
            for i in range(nedges):
                var cur = nf0 + i
                var nxt = nf0 + (i + 1) % nedges
                var prv = nf0 + nedges - 1 if i == 0 else cur - 1
                var hf = ehor(ws, wrow, i, 0)
                var he = ehor(ws, wrow, i, 1)
                var hv1 = ef(ws, wrow, hf, he)
                var hv2 = ef(ws, wrow, hf, (he + 1) % 3)
                set_eadj(ws, wrow, hf, he, cur)
                var d2 = attach_face(
                    ws, wrow, cur, wi, hv2, hv1, prv, hf, nxt
                )
                nfaces = cur + 1
                if d2 == Scalar[DTYPE](0):
                    face = -1
                    bad = True
                    break
                if d2 >= lower2 and d2 <= upper2:
                    set_emap(ws, wrow, nmap, cur)
                    set_efi(ws, wrow, cur, nmap)
                    nmap += 1
            nhor = 0
            if bad or nmap == 0 or face < 0:
                break

    # ---- what the reference does with the face it ended on ------------------
    comptime if _GJK_PROBE:
        var _g_t3 = Int(perf_counter_ns())
        print("[gprobe]", "sep", 0, "gjk", _g_t1 - _g_t0, "kg", _g_kg, "poly", _g_t2 - _g_t1, "epa", _g_t3 - _g_t2, "ke", _g_ke, "gap", _g_gap, "upper", _g_upper, "tol", Float64(_epa_tol))
    if face >= 0:
        var wit = epa_witness(ws, wrow, face)
        var i0 = ef(ws, wrow, face, 0)
        var i1 = ef(ws, wrow, face, 1)
        var i2 = ef(ws, wrow, face, 2)
        # The winning face's three vertices, as support points on each geom.
        # Written with CONSTANT indices on the destination and only the
        # polytope vertex id varying, so no per-thread array is indexed by a
        # runtime value on the way out — see
        # `feedback_metal_wide_per_thread_inlinearray_miscompute`.
        wf1[0] = ev(ws, wrow, i0, 3)
        wf1[1] = ev(ws, wrow, i0, 4)
        wf1[2] = ev(ws, wrow, i0, 5)
        wf1[3] = ev(ws, wrow, i1, 3)
        wf1[4] = ev(ws, wrow, i1, 4)
        wf1[5] = ev(ws, wrow, i1, 5)
        wf1[6] = ev(ws, wrow, i2, 3)
        wf1[7] = ev(ws, wrow, i2, 4)
        wf1[8] = ev(ws, wrow, i2, 5)
        wf2[0] = ev(ws, wrow, i0, 6)
        wf2[1] = ev(ws, wrow, i0, 7)
        wf2[2] = ev(ws, wrow, i0, 8)
        wf2[3] = ev(ws, wrow, i1, 6)
        wf2[4] = ev(ws, wrow, i1, 7)
        wf2[5] = ev(ws, wrow, i1, 8)
        wf2[6] = ev(ws, wrow, i2, 6)
        wf2[7] = ev(ws, wrow, i2, 7)
        wf2[8] = ev(ws, wrow, i2, 8)
        # Every polytope vertex carries its support indices, seeds included
        # (the simplex is laid out like an EPA vertex, "STAGE B" in
        # `polytope4`). A mesh's is its hull vertex, which is what
        # multicontact used to recover by scanning the whole hull.
        wfi[0] = Int(ev(ws, wrow, i0, 9))
        wfi[1] = Int(ev(ws, wrow, i1, 9))
        wfi[2] = Int(ev(ws, wrow, i2, 9))
        wfi[3] = Int(ev(ws, wrow, i0, 10))
        wfi[4] = Int(ev(ws, wrow, i1, 10))
        wfi[5] = Int(ev(ws, wrow, i2, 10))
        wx[0] = wit[0]
        wx[1] = wit[1]
        wx[2] = wit[2]
        wx[3] = wit[3]
        wx[4] = wit[4]
        wx[5] = wit[5]
        wf_ok = 1

        comptime if EPA_DBG:
            return (
                -sqrt(efd(ws, wrow, face)) + ccd_margin,
                Scalar[DTYPE](nverts),
                Scalar[DTYPE](nfaces),
                Scalar[DTYPE](face),
                Scalar[DTYPE](i0),
                Scalar[DTYPE](i1),
                Scalar[DTYPE](i2),
            )

        # ⚠ THE NORMAL IS `normalize(x1 - x2)`, NOT THE UNIT FACE NORMAL.
        # `mjc_penetration` (`engine_collision_convex.c`) builds it from the
        # two witness points; that is the same vector as `face->v / |face->v|`
        # in exact arithmetic and not the same bits, and this function's
        # callers compare against MuJoCo to 1e-12.
        var nx = wit[0] - wit[3]
        var ny = wit[1] - wit[4]
        var nz = wit[2] - wit[5]
        var nl = sqrt(nx * nx + ny * ny + nz * nz)
        if nl > Scalar[DTYPE](0):
            nx /= nl
            ny /= nl
            nz /= nl
        else:
            nx = Scalar[DTYPE](0)
            ny = Scalar[DTYPE](0)
            nz = Scalar[DTYPE](1)
        # `con[i].dist = margin + dist` with `dist = -sqrt(face->dist2)`.
        var pen = -sqrt(efd(ws, wrow, face))
        return (
            pen + ccd_margin,
            (wit[0] + wit[3]) * Scalar[DTYPE](0.5),
            (wit[1] + wit[4]) * Scalar[DTYPE](0.5),
            (wit[2] + wit[5]) * Scalar[DTYPE](0.5),
            nx,
            ny,
            nz,
        )

    # ⚠⚠ NO POLYTOPE, NO CONTACT — AND THAT IS THE REFERENCE'S ANSWER, NOT A
    # GIVE-UP. `mjc_ccd` leaves `status->dist` at the 0 it wrote before
    # building the seed, and `mjc_penetration` records a contact only when
    # `mjc_ccd` returns something NEGATIVE. Returning `1e30` is how this
    # function's callers spell the same thing: every one of them admits a
    # contact on `dist < margin`.
    wf_ok = 0
    return (
        Scalar[DTYPE](1e30),
        Scalar[DTYPE](0),
        Scalar[DTYPE](0),
        Scalar[DTYPE](0),
        Scalar[DTYPE](0),
        Scalar[DTYPE](0),
        Scalar[DTYPE](1),
    )


@always_inline
def gjk_epa[
    DTYPE: DType,
    L_MESH_VERTS: Layout,
    L_MESH_VERT_EDGEADR: Layout,
    L_MESH_EDGES: Layout,
    L_WS: Layout,
    NPRISM: Int = 1,
](
    type1: Int,
    p1x: Scalar[DTYPE], p1y: Scalar[DTYPE], p1z: Scalar[DTYPE],
    q1x: Scalar[DTYPE], q1y: Scalar[DTYPE], q1z: Scalar[DTYPE],
    q1w: Scalar[DTYPE],
    r1: Scalar[DTYPE], hl1: Scalar[DTYPE],
    hx1: Scalar[DTYPE], hy1: Scalar[DTYPE], hz1: Scalar[DTYPE],
    mesh_verts: LayoutTensor[
        DTYPE, L_MESH_VERTS, MutAnyOrigin
    ],
    mesh_vert_edgeadr: LayoutTensor[
        DTYPE, L_MESH_VERT_EDGEADR, MutAnyOrigin
    ],
    mesh_edges: LayoutTensor[
        DTYPE, L_MESH_EDGES, MutAnyOrigin
    ],
    va1: Int, mnv1: Int,
    type2: Int,
    p2x: Scalar[DTYPE], p2y: Scalar[DTYPE], p2z: Scalar[DTYPE],
    q2x: Scalar[DTYPE], q2y: Scalar[DTYPE], q2z: Scalar[DTYPE],
    q2w: Scalar[DTYPE],
    r2: Scalar[DTYPE], hl2: Scalar[DTYPE],
    hx2: Scalar[DTYPE], hy2: Scalar[DTYPE], hz2: Scalar[DTYPE],
    va2: Int, mnv2: Int,
    ws: LayoutTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    ccd_tol: Scalar[DTYPE] = Scalar[DTYPE](MJ_CCD_TOLERANCE),
    ccd_iter: Int = MJ_CCD_ITERATIONS,
    ccd_margin: Scalar[DTYPE] = Scalar[DTYPE](0),
    prism: Array[Scalar[DTYPE], NPRISM] = Array[
        Scalar[DTYPE], NPRISM
    ](fill=Scalar[DTYPE](0)),
    # Forwarded to `gjk_epa_witness`. A caller whose ONLY use of the result
    # is the `dist < margin` contact test passes its margin here; the GJK then
    # stops as soon as a lower bound proves the pair at least that far apart
    # (MuJoCo's `dist_cutoff = 0` on its margin-inflated shapes,
    # engine_collision_convex.c:104). -1 keeps the full distance. After `prism`, which
    # `hfield_convex` passes positionally.
    dist_cutoff: Scalar[DTYPE] = Scalar[DTYPE](-1),
    # ⚠ AUD-32: `inflate1=False` leaves object 1 UNINFLATED while object 2
    # still grows by `ccd_margin/2` — MuJoCo's heightfield collider sets
    # `obj2.margin = margin` and leaves the prism's at 0
    # (engine_collision_convex.c:1246). Everything else (tolerances, the
    # `+ ccd_margin` add-back) is unchanged; the hfield caller subtracts it.
    inflate1: Bool = True,
) -> Tuple[
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
]:
    """`gjk_epa_witness` for callers that only want `(dist, pos, normal)`.

    Kept as a separate entry point rather than making the witness arguments
    optional so that the sixteen existing call sites — two narrow phases, the
    perturbation loop and ten tests — are untouched by the native
    multi-contact work. The scratch arrays are dead stores everywhere the
    result is discarded.
    """
    var wf1 = Array[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
    var wf2 = Array[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
    var wx = Array[Scalar[DTYPE], 6](fill=Scalar[DTYPE](0))
    var wf_ok = 0
    var wfi = Array[Int, 6](fill=-1)
    return gjk_epa_witness[DTYPE, NPRISM=NPRISM](
        type1,
        p1x, p1y, p1z, q1x, q1y, q1z, q1w,
        r1, hl1, hx1, hy1, hz1,
        mesh_verts, mesh_vert_edgeadr, mesh_edges, va1, mnv1,
        type2,
        p2x, p2y, p2z, q2x, q2y, q2z, q2w,
        r2, hl2, hx2, hy2, hz2,
        va2, mnv2,
        wf1, wf2, wx, wf_ok, wfi,
        ws, wrow,
        ccd_tol, ccd_iter, ccd_margin,
        dist_cutoff,
        prism,
        inflate1=inflate1,
    )

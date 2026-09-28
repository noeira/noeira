"""Contact solver — a port of Box2D 2.3's b2ContactSolver (sequential impulses).

Reference: `references/pybox2d-2.3.10/Box2D/Dynamics/Contacts/b2ContactSolver.cpp`
(the Box2D that Gymnasium's LunarLander / BipedalWalker / CarRacing run).

Per world step, for the contacts collision detection produced:
  1. `init_velocity_single_env`  — InitializeVelocityConstraints + WarmStart:
     store each contact point in both bodies' frames (for the position
     solve) and the restitution velocity bias, then apply the stored
     impulses.
  2. `solve_velocity_single_env` — SolveVelocityConstraints, once per velocity
     iteration: friction FIRST, then the normal impulse, both with
     accumulated-impulse clamping, on every contact every iteration.
  3. `solve_position_single_env` — SolvePositionConstraints, once per position
     iteration: the separation is recomputed from the CURRENT poses and the
     correction moves positions AND angles (non-linear Gauss-Seidel).
     Returns whether every contact is within 3 * linear slop, for Box2D's
     early exit.

Contact convention: the normal points from body B (or the static ground,
`CONTACT_BODY_B == -1`) toward body A, and CONTACT_DEPTH is the penetration
(> 0). With that convention every Box2D formula carries over with A and B
swapped, so the impulse values are identical.

Known deviations from Box2D (noted where they apply):
  - contacts are solved one point at a time (Box2D's 2-point block solver
    needs manifold pairing the detectors do not produce),
  - detection zeroes the accumulated impulses every step, so warm starting
    applies zero impulses (Box2D matches manifold points across steps),
  - shapes have no skin radius (Box2D polygons carry 2 * linear slop).
"""

from layout import LayoutTensor, Layout
from std.math import cos, sin

from ..constants import (
    dtype,
    BODY_STATE_SIZE,
    CONTACT_DATA_SIZE,
    IDX_X,
    IDX_Y,
    IDX_ANGLE,
    IDX_VX,
    IDX_VY,
    IDX_OMEGA,
    IDX_INV_MASS,
    IDX_INV_INERTIA,
    CONTACT_BODY_A,
    CONTACT_BODY_B,
    CONTACT_POINT_X,
    CONTACT_POINT_Y,
    CONTACT_NORMAL_X,
    CONTACT_NORMAL_Y,
    CONTACT_DEPTH,
    CONTACT_NORMAL_IMPULSE,
    CONTACT_TANGENT_IMPULSE,
    CONTACT_LOCAL_AX,
    CONTACT_LOCAL_AY,
    CONTACT_LOCAL_BX,
    CONTACT_LOCAL_BY,
    CONTACT_VELOCITY_BIAS,
    DEFAULT_FRICTION,
    DEFAULT_RESTITUTION,
    DEFAULT_VELOCITY_ITERATIONS,
    DEFAULT_POSITION_ITERATIONS,
    B2_LINEAR_SLOP,
    B2_BAUMGARTE,
    B2_MAX_LINEAR_CORRECTION,
    B2_VELOCITY_THRESHOLD,
)
from ..traits.solver import ConstraintSolver


@always_inline
def _ld[
    BATCH: Int, SIZE: Int
](
    t: LayoutTensor[dtype, Layout.row_major(BATCH, SIZE), MutAnyOrigin],
    env: Int,
    i: Int,
) -> Scalar[dtype]:
    return rebind[Scalar[dtype]](t[env, i])


@always_inline
def _clamp(x: Scalar[dtype], lo: Scalar[dtype], hi: Scalar[dtype]) -> Scalar[dtype]:
    return lo if x < lo else (hi if x > hi else x)


struct ImpulseSolver(ConstraintSolver):
    """Box2D 2.3 contact solver (see the module docstring)."""

    comptime VELOCITY_ITERATIONS: Int = DEFAULT_VELOCITY_ITERATIONS
    comptime POSITION_ITERATIONS: Int = DEFAULT_POSITION_ITERATIONS

    var friction: Scalar[dtype]
    var restitution: Scalar[dtype]

    def __init__(
        out self,
        friction: Float64 = DEFAULT_FRICTION,
        restitution: Float64 = DEFAULT_RESTITUTION,
    ):
        """Contact material: Coulomb friction (Box2D mixes two fixtures as
        sqrt(f_a * f_b) — pass the mixed value) and restitution."""
        self.friction = Scalar[dtype](friction)
        self.restitution = Scalar[dtype](restitution)

    # =========================================================================
    # Trait methods — the [BATCH, NUM_BODIES, BODY_STATE_SIZE] bodies layout
    # is the flat state layout with BODIES_OFFSET = 0, so each call routes to
    # the single-env core below (ONE implementation).
    # =========================================================================

    def init_velocity[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
    ](
        self,
        mut bodies: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ],
        mut contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
    ):
        var state = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES * BODY_STATE_SIZE),
            MutAnyOrigin,
        ](bodies.ptr)
        for env in range(BATCH):
            Self.init_velocity_single_env[
                BATCH, NUM_BODIES, MAX_CONTACTS, NUM_BODIES * BODY_STATE_SIZE, 0
            ](env, state, contacts, Int(contact_counts[env]), self.restitution)

    def solve_velocity[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
    ](
        self,
        mut bodies: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ],
        mut contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
    ):
        var state = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES * BODY_STATE_SIZE),
            MutAnyOrigin,
        ](bodies.ptr)
        for env in range(BATCH):
            Self.solve_velocity_single_env[
                BATCH, NUM_BODIES, MAX_CONTACTS, NUM_BODIES * BODY_STATE_SIZE, 0
            ](env, state, contacts, Int(contact_counts[env]), self.friction)

    def solve_position[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
    ](
        self,
        mut bodies: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES, BODY_STATE_SIZE),
            MutAnyOrigin,
        ],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH), MutAnyOrigin
        ],
    ) -> Bool:
        var state = LayoutTensor[
            dtype,
            Layout.row_major(BATCH, NUM_BODIES * BODY_STATE_SIZE),
            MutAnyOrigin,
        ](bodies.ptr)
        var all_ok = True
        for env in range(BATCH):
            if not Self.solve_position_single_env[
                BATCH, NUM_BODIES, MAX_CONTACTS, NUM_BODIES * BODY_STATE_SIZE, 0
            ](env, state, contacts, Int(contact_counts[env])):
                all_ok = False
        return all_ok

    # =========================================================================
    # InitializeVelocityConstraints + WarmStart
    # =========================================================================

    @always_inline
    @staticmethod
    def init_velocity_single_env[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        contact_count: Int,
        restitution: Scalar[dtype],
    ):
        """Record each contact point in both bodies' frames and its
        restitution bias, then warm start. Call once per step, after
        detection and velocity integration, before the velocity
        iterations (positions have not moved since detection)."""
        for c in range(contact_count):
            var a = Int(contacts[env, c, CONTACT_BODY_A])
            var b = Int(contacts[env, c, CONTACT_BODY_B])
            var px = rebind[Scalar[dtype]](contacts[env, c, CONTACT_POINT_X])
            var py = rebind[Scalar[dtype]](contacts[env, c, CONTACT_POINT_Y])
            var nx = rebind[Scalar[dtype]](contacts[env, c, CONTACT_NORMAL_X])
            var ny = rebind[Scalar[dtype]](contacts[env, c, CONTACT_NORMAL_Y])

            var off_a = BODIES_OFFSET + a * BODY_STATE_SIZE
            var cax = _ld(state, env, off_a + IDX_X)
            var cay = _ld(state, env, off_a + IDX_Y)
            var aa = _ld(state, env, off_a + IDX_ANGLE)
            var rax = px - cax
            var ray = py - cay
            # local = R(-a) * r
            var ca = cos(aa)
            var sa = sin(aa)
            contacts[env, c, CONTACT_LOCAL_AX] = ca * rax + sa * ray
            contacts[env, c, CONTACT_LOCAL_AY] = -sa * rax + ca * ray

            var vax = _ld(state, env, off_a + IDX_VX)
            var vay = _ld(state, env, off_a + IDX_VY)
            var wa = _ld(state, env, off_a + IDX_OMEGA)
            var rel_x = vax - wa * ray
            var rel_y = vay + wa * rax
            if b >= 0:
                var off_b = BODIES_OFFSET + b * BODY_STATE_SIZE
                var cbx = _ld(state, env, off_b + IDX_X)
                var cby = _ld(state, env, off_b + IDX_Y)
                var ab = _ld(state, env, off_b + IDX_ANGLE)
                var rbx = px - cbx
                var rby = py - cby
                var cb = cos(ab)
                var sb = sin(ab)
                contacts[env, c, CONTACT_LOCAL_BX] = cb * rbx + sb * rby
                contacts[env, c, CONTACT_LOCAL_BY] = -sb * rbx + cb * rby
                var wb = _ld(state, env, off_b + IDX_OMEGA)
                rel_x -= _ld(state, env, off_b + IDX_VX) - wb * rby
                rel_y -= _ld(state, env, off_b + IDX_VY) + wb * rbx
            else:
                # Static ground: the anchor stays at the world point.
                contacts[env, c, CONTACT_LOCAL_BX] = px
                contacts[env, c, CONTACT_LOCAL_BY] = py

            # Restitution only above the velocity threshold.
            var v_rel = rel_x * nx + rel_y * ny
            var bias = Scalar[dtype](0)
            if v_rel < Scalar[dtype](-B2_VELOCITY_THRESHOLD):
                bias = -restitution * v_rel
            contacts[env, c, CONTACT_VELOCITY_BIAS] = bias

            # Warm start with the stored impulses.
            var jn = rebind[Scalar[dtype]](contacts[env, c, CONTACT_NORMAL_IMPULSE])
            var jt = rebind[Scalar[dtype]](
                contacts[env, c, CONTACT_TANGENT_IMPULSE]
            )
            if jn != Scalar[dtype](0) or jt != Scalar[dtype](0):
                var tx = -ny
                var ty = nx
                Self._apply_impulse[BATCH, STATE_SIZE, BODIES_OFFSET](
                    env, state, a, b,
                    jn * nx + jt * tx, jn * ny + jt * ty,
                    rax, ray, px, py,
                )

    # =========================================================================
    # SolveVelocityConstraints
    # =========================================================================

    @always_inline
    @staticmethod
    def solve_velocity_single_env[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        contact_count: Int,
        friction: Scalar[dtype],
    ):
        """One velocity iteration over all contacts: friction first ("non-
        penetration is more important than friction"), then the normal
        impulse, each clamped on its accumulated value."""
        for c in range(contact_count):
            var a = Int(contacts[env, c, CONTACT_BODY_A])
            var b = Int(contacts[env, c, CONTACT_BODY_B])
            var px = rebind[Scalar[dtype]](contacts[env, c, CONTACT_POINT_X])
            var py = rebind[Scalar[dtype]](contacts[env, c, CONTACT_POINT_Y])
            var nx = rebind[Scalar[dtype]](contacts[env, c, CONTACT_NORMAL_X])
            var ny = rebind[Scalar[dtype]](contacts[env, c, CONTACT_NORMAL_Y])
            var tx = -ny
            var ty = nx

            var off_a = BODIES_OFFSET + a * BODY_STATE_SIZE
            var ma = _ld(state, env, off_a + IDX_INV_MASS)
            var ia = _ld(state, env, off_a + IDX_INV_INERTIA)
            var rax = px - _ld(state, env, off_a + IDX_X)
            var ray = py - _ld(state, env, off_a + IDX_Y)
            var mb = Scalar[dtype](0)
            var ib = Scalar[dtype](0)
            var rbx = Scalar[dtype](0)
            var rby = Scalar[dtype](0)
            if b >= 0:
                var off_b = BODIES_OFFSET + b * BODY_STATE_SIZE
                mb = _ld(state, env, off_b + IDX_INV_MASS)
                ib = _ld(state, env, off_b + IDX_INV_INERTIA)
                rbx = px - _ld(state, env, off_b + IDX_X)
                rby = py - _ld(state, env, off_b + IDX_Y)

            # --- Tangent (friction) ---
            var rta = rax * ty - ray * tx
            var rtb = rbx * ty - rby * tx
            var kt = ma + mb + ia * rta * rta + ib * rtb * rtb
            var vt = Self._rel_vel_along[BATCH, STATE_SIZE, BODIES_OFFSET](
                env, state, a, b, rax, ray, rbx, rby, tx, ty
            )
            var lambda_t = (
                -vt / kt if kt > Scalar[dtype](0) else Scalar[dtype](0)
            )
            var max_friction = friction * rebind[Scalar[dtype]](
                contacts[env, c, CONTACT_NORMAL_IMPULSE]
            )
            var old_t = rebind[Scalar[dtype]](
                contacts[env, c, CONTACT_TANGENT_IMPULSE]
            )
            var new_t = _clamp(old_t + lambda_t, -max_friction, max_friction)
            contacts[env, c, CONTACT_TANGENT_IMPULSE] = new_t
            lambda_t = new_t - old_t
            Self._apply_impulse[BATCH, STATE_SIZE, BODIES_OFFSET](
                env, state, a, b, lambda_t * tx, lambda_t * ty,
                rax, ray, px, py,
            )

            # --- Normal ---
            var rna = rax * ny - ray * nx
            var rnb = rbx * ny - rby * nx
            var kn = ma + mb + ia * rna * rna + ib * rnb * rnb
            var vn = Self._rel_vel_along[BATCH, STATE_SIZE, BODIES_OFFSET](
                env, state, a, b, rax, ray, rbx, rby, nx, ny
            )
            var bias = rebind[Scalar[dtype]](
                contacts[env, c, CONTACT_VELOCITY_BIAS]
            )
            var lambda_n = (
                -(vn - bias) / kn if kn > Scalar[dtype](0) else Scalar[dtype](0)
            )
            var old_n = rebind[Scalar[dtype]](
                contacts[env, c, CONTACT_NORMAL_IMPULSE]
            )
            var new_n = max(old_n + lambda_n, Scalar[dtype](0))
            contacts[env, c, CONTACT_NORMAL_IMPULSE] = new_n
            lambda_n = new_n - old_n
            Self._apply_impulse[BATCH, STATE_SIZE, BODIES_OFFSET](
                env, state, a, b, lambda_n * nx, lambda_n * ny,
                rax, ray, px, py,
            )

    # =========================================================================
    # SolvePositionConstraints
    # =========================================================================

    @always_inline
    @staticmethod
    def solve_position_single_env[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        contact_count: Int,
        linear_slop: Scalar[dtype] = Scalar[dtype](B2_LINEAR_SLOP),
        baumgarte: Scalar[dtype] = Scalar[dtype](B2_BAUMGARTE),
        max_linear_correction: Scalar[dtype] = Scalar[dtype](
            B2_MAX_LINEAR_CORRECTION
        ),
    ) -> Bool:
        """One position iteration: recompute each separation from the current
        poses, correct by C = clamp(baumgarte * (s + slop), -max, 0) along the
        normal, moving positions and angles. True when every separation is
        >= -3 * linear slop (Box2D's early-exit test). The defaults are
        Box2D's (meters); an env in other units passes its own."""
        var min_separation = Scalar[dtype](0)
        for c in range(contact_count):
            var a = Int(contacts[env, c, CONTACT_BODY_A])
            var b = Int(contacts[env, c, CONTACT_BODY_B])
            var nx = rebind[Scalar[dtype]](contacts[env, c, CONTACT_NORMAL_X])
            var ny = rebind[Scalar[dtype]](contacts[env, c, CONTACT_NORMAL_Y])
            var depth0 = rebind[Scalar[dtype]](contacts[env, c, CONTACT_DEPTH])

            var off_a = BODIES_OFFSET + a * BODY_STATE_SIZE
            var cax = _ld(state, env, off_a + IDX_X)
            var cay = _ld(state, env, off_a + IDX_Y)
            var aa = _ld(state, env, off_a + IDX_ANGLE)
            var ma = _ld(state, env, off_a + IDX_INV_MASS)
            var ia = _ld(state, env, off_a + IDX_INV_INERTIA)
            var lax = rebind[Scalar[dtype]](contacts[env, c, CONTACT_LOCAL_AX])
            var lay = rebind[Scalar[dtype]](contacts[env, c, CONTACT_LOCAL_AY])
            var ca = cos(aa)
            var sa = sin(aa)
            var pax = cax + ca * lax - sa * lay
            var pay = cay + sa * lax + ca * lay

            var lbx = rebind[Scalar[dtype]](contacts[env, c, CONTACT_LOCAL_BX])
            var lby = rebind[Scalar[dtype]](contacts[env, c, CONTACT_LOCAL_BY])
            var pbx = lbx
            var pby = lby
            var off_b = 0
            var cbx = Scalar[dtype](0)
            var cby = Scalar[dtype](0)
            var ab = Scalar[dtype](0)
            var mb = Scalar[dtype](0)
            var ib = Scalar[dtype](0)
            if b >= 0:
                off_b = BODIES_OFFSET + b * BODY_STATE_SIZE
                cbx = _ld(state, env, off_b + IDX_X)
                cby = _ld(state, env, off_b + IDX_Y)
                ab = _ld(state, env, off_b + IDX_ANGLE)
                mb = _ld(state, env, off_b + IDX_INV_MASS)
                ib = _ld(state, env, off_b + IDX_INV_INERTIA)
                var cb = cos(ab)
                var sb = sin(ab)
                pbx = cbx + cb * lbx - sb * lby
                pby = cby + sb * lbx + cb * lby

            # Both anchors coincided at detection, where the separation was
            # -depth0; it changes by the anchors' relative motion along n.
            var separation = -depth0 + (pax - pbx) * nx + (pay - pby) * ny
            min_separation = min(min_separation, separation)
            var corr = _clamp(
                baumgarte * (separation + linear_slop),
                -max_linear_correction,
                Scalar[dtype](0),
            )
            var rax = pax - cax
            var ray = pay - cay
            var rbx = pax - cbx
            var rby = pay - cby
            var rna = rax * ny - ray * nx
            var rnb = rbx * ny - rby * nx
            var k = ma + mb + ia * rna * rna + ib * rnb * rnb
            var impulse = -corr / k if k > Scalar[dtype](0) else Scalar[dtype](0)
            var pxi = impulse * nx
            var pyi = impulse * ny
            state[env, off_a + IDX_X] = cax + ma * pxi
            state[env, off_a + IDX_Y] = cay + ma * pyi
            state[env, off_a + IDX_ANGLE] = aa + ia * (rax * pyi - ray * pxi)
            if b >= 0:
                state[env, off_b + IDX_X] = cbx - mb * pxi
                state[env, off_b + IDX_Y] = cby - mb * pyi
                state[env, off_b + IDX_ANGLE] = ab - ib * (rbx * pyi - rby * pxi)
        return min_separation >= Scalar[dtype](-3.0) * linear_slop

    # =========================================================================
    # Helpers
    # =========================================================================

    @always_inline
    @staticmethod
    def _rel_vel_along[
        BATCH: Int, STATE_SIZE: Int, BODIES_OFFSET: Int
    ](
        env: Int,
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        a: Int,
        b: Int,
        rax: Scalar[dtype],
        ray: Scalar[dtype],
        rbx: Scalar[dtype],
        rby: Scalar[dtype],
        dx: Scalar[dtype],
        dy: Scalar[dtype],
    ) -> Scalar[dtype]:
        """(v_A + w_A x r_A - v_B - w_B x r_B) . d at the contact point."""
        var off_a = BODIES_OFFSET + a * BODY_STATE_SIZE
        var wa = _ld(state, env, off_a + IDX_OMEGA)
        var rel_x = _ld(state, env, off_a + IDX_VX) - wa * ray
        var rel_y = _ld(state, env, off_a + IDX_VY) + wa * rax
        if b >= 0:
            var off_b = BODIES_OFFSET + b * BODY_STATE_SIZE
            var wb = _ld(state, env, off_b + IDX_OMEGA)
            rel_x -= _ld(state, env, off_b + IDX_VX) - wb * rby
            rel_y -= _ld(state, env, off_b + IDX_VY) + wb * rbx
        return rel_x * dx + rel_y * dy

    @always_inline
    @staticmethod
    def _apply_impulse[
        BATCH: Int, STATE_SIZE: Int, BODIES_OFFSET: Int
    ](
        env: Int,
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        a: Int,
        b: Int,
        jx: Scalar[dtype],
        jy: Scalar[dtype],
        rax: Scalar[dtype],
        ray: Scalar[dtype],
        px: Scalar[dtype],
        py: Scalar[dtype],
    ):
        """Apply impulse J at the contact point: +J on A, -J on B."""
        var off_a = BODIES_OFFSET + a * BODY_STATE_SIZE
        var ma = _ld(state, env, off_a + IDX_INV_MASS)
        var ia = _ld(state, env, off_a + IDX_INV_INERTIA)
        state[env, off_a + IDX_VX] = _ld(state, env, off_a + IDX_VX) + ma * jx
        state[env, off_a + IDX_VY] = _ld(state, env, off_a + IDX_VY) + ma * jy
        state[env, off_a + IDX_OMEGA] = _ld(
            state, env, off_a + IDX_OMEGA
        ) + ia * (rax * jy - ray * jx)
        if b >= 0:
            var off_b = BODIES_OFFSET + b * BODY_STATE_SIZE
            var mb = _ld(state, env, off_b + IDX_INV_MASS)
            var ib = _ld(state, env, off_b + IDX_INV_INERTIA)
            var rbx = px - _ld(state, env, off_b + IDX_X)
            var rby = py - _ld(state, env, off_b + IDX_Y)
            state[env, off_b + IDX_VX] = _ld(state, env, off_b + IDX_VX) - mb * jx
            state[env, off_b + IDX_VY] = _ld(state, env, off_b + IDX_VY) - mb * jy
            state[env, off_b + IDX_OMEGA] = _ld(
                state, env, off_b + IDX_OMEGA
            ) - ib * (rbx * jy - rby * jx)

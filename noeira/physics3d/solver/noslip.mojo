"""`mj_solNoSlip` — the friction-only post-solver sweep (BOTH cones).

WIRED IN: `solver/newton_solve.mojo` calls `noslip_pyramidal` from its
PYRAMIDAL per-env path (2026-08-03) and `noslip_elliptic` from its ELLIPTIC one
(2026-08-13), each under `comptime if NOSLIP_ITER > 0`, with `NOSLIP_ITER`
threaded from `<option noslip_iterations>` via `ModelDefFromXML` ->
`Phyics3dEnv` -> `EulerIntegrator` -> `solve_newton`.

⚠ THE TWO ARE DIFFERENT ALGORITHMS OVER DIFFERENT ROW LAYOUTS, not one routine
with a flag — see the header on each. The dispatch is therefore a real
obligation on the caller: routing a model to the wrong one would apply the
wrong friction law rather than fail. `newton_solve` dispatches inside the
already-cone-split solve bodies, so the two cannot be confused.

GATES
  * pyramidal: `tests/physics3d/test_noslip_vs_mujoco.mojo` (compiles, runs,
    and does not perturb an already-converged solve — the pass is INERT on
    that fixture and the file says so) and dm_control's dog, where it bites.
    ⚠ That file gates the PER-ENV path only. The blocked NVIDIA kernel is
    gated separately by `tests/physics3d/test_noslip_blocked_kernel.mojo`,
    because for a long time it ran neither and nothing noticed.
  * elliptic: `tests/physics3d/test_noslip_elliptic_vs_mujoco.mojo`, a
    3-capsule chain SLAMMED into the floor at 40 m/s while sliding, where
    MuJoCo-against-MuJoCo with only the option changed moves `qacc` by
    **9.8e+1** — the pass is first-order there, so the gate can fail. That
    file also records which fixtures do NOT work and why: a hard normal
    impulse is the ingredient, not contact count, and every gently resting
    fixture is inert to round-off.
    ⚠ That fixture is condim 3, so it only reaches `mju_QCQP2`. The other two
    dispatches ride on `tests/physics3d/test_elliptic_condim46_vs_mujoco.mojo`,
    whose spinning ball runs the pass at condim 4 and 6 — MuJoCo moves `qacc`
    by 3.0e+1 and 3.6e+2 there with only `noslip_iterations` changed, so
    `mju_QCQP3` and `mju_QCQP` are exercised rather than merely compiled.

WHAT IT IS

After the primal solve converges, MuJoCo optionally runs a Gauss-Seidel sweep
over the FRICTION dimensions only, with the normal forces held fixed, to
remove residual slip. dm_control's dog asks for it in the suite
(`<option timestep="0.005" noslip_iterations="4"/>`), and EVERY manipulation
model asks for it with the elliptic cone
(`<option cone="elliptic" noslip_iterations="5" noslip_tolerance="0"/>`).

It is NOT a refinement that can be skipped. Measured MuJoCo-against-MuJoCo with
only the option changed:

  * dog (pyramidal): `max|d(qvel)|` **2.9e-2 on the FIRST contacting step**.
  * `reach_site_features` (elliptic): `max|d(qacc)|` **7.4e+2 on step 1**, at
    55 contacts. Not a tail correction — the same order as the answer.

(Multi-step numbers prove nothing on their own — a contact-rich rollout is
chaotic and any perturbation grows. The first-step number is the one that
settles it, because there is nothing yet to amplify.)

VERSION DRIFT: none that reaches us. `mj_solNoSlip` is byte-identical across
MuJoCo 3.3.6, 3.6.0 and main; 3.11.0 refactors it into a static `solNoSlip`
with island support, hoists `dualFinish` to the caller (which still runs it
whenever `noslip_iterations > 0`, `engine_forward.c:1152`) and moves the
elliptic QCQP block into `solveQCQP`/`projectEllipsoid` helpers — all three
changes are structural, and the arithmetic is unchanged. So the runtime's
3.10.0, which no tree here matches, is bracketed rather than guessed at. See
`feedback_reference_tree_version_drift`. Transcribed from
`engine_solver.c:537` (3.6.0) / `:767` (3.11.0).

THE TWO SIMPLIFICATIONS THAT MAKE THIS TRACTABLE

MuJoCo works in the DUAL formulation and needs `efc_AR = J M^-1 J^T + R`, a
dense `nefc x nefc` matrix we do not build. Both of its uses collapse:

  * `residual(..., flg_subR=1)` at row j expands to

        b_j + (AR f)_j - R_j f_j
      = (J qacc_smooth - aref)_j + (J M^-1 J^T f)_j
      = J_j (qacc_smooth + M^-1 J^T f) - aref_j
      = J_j qacc - aref_j

    which is EXACTLY the primal solver's `jar[j]`. No dual vector needed.

  * `extractBlock(..., flg_subR=1)` is `A` with `R` subtracted off the diagonal
    and the diagonal clamped to 1e-10 — i.e. plain `J_j M^-1 J_k^T`, with NO
    `R` term at all.

So the only genuinely new quantity is `M^-1 J_j^T` per row, for the in-place
`qacc` update that keeps `jar` current between blocks. `m_inv` is already a
dense NV x NV on `Data`, so that is a matvec, not a solve.

WHY THE NORMAL FORCE IS PRESERVED EXACTLY

For a pyramidal contact the friction is carried by opposing edge PAIRS
`(f_j, f_{j+1})`, and the normal contribution of a pair is
`mid = (f_j + f_{j+1})/2`. Every branch below writes the pair as
`(mid + y, mid - y)` for some `y`, so `mid` — and with it the normal force —
is invariant by construction. That is what "with the normal forces frozen"
means operationally, and it is why this cannot destabilise a contact.

SCOPE, STATED RATHER THAN IMPLIED

  * `noslip_pyramidal` is PYRAMIDAL only BY CONSTRUCTION and
    `noslip_elliptic` is ELLIPTIC only BY CONSTRUCTION — the cone is baked
    into each one's arithmetic and row layout, so neither takes a cone type
    and neither has anything to check at runtime.
    ⚠ THE OBLIGATION IS ON THE CALLER: the wiring must dispatch on
    `CONE_TYPE`, because calling the wrong one applies the wrong friction law
    rather than failing.
  * `scale` and `tolerance` are CALLER-SUPPLIED for both, and both must be
    MuJoCo's or the iteration count diverges. `tolerance` is
    `m->opt.noslip_tolerance` — MuJoCo's default is 1e-6, dm_control's
    manipulation models set **0** ("run every iteration"). It is parsed and
    carried in `MODEL_META_IDX_NOSLIP_TOLERANCE`; it was a hardcoded 1e-6
    until 2026-08-13. ⚠ That change is a FIDELITY fix, not a measured one: no
    fixture here can tell 0 from 1e-6 (worst 8.9e-10, and exactly 0.0 on
    `reach_site_features`). `scale` is `1 / (stat.meaninertia * max(1, nv))`,
    and `stat.meaninertia` is the mean of the mass matrix DIAGONAL —
    `engine_setconst.c:1139-1146`:

        meaninertia = (1/nv) * sum_i qM[dof_Madr[i]]

    evaluated once at model-build time. Getting either wrong does not corrupt
    the sweep, it changes WHEN the loop stops, which is a subtler and more
    annoying divergence — hence spelling it out here.
  * `noslip_pyramidal` DOES now run on `solve_newton_blocked`, the NVIDIA
    production kernel, on BOTH of its branches. Until 2026-08-13 that kernel
    accepted `NOSLIP_ITER` and never read it, so the pass ran on its CPU
    branch (which delegates to `_newton_solve_env`) and silently vanished on
    its GPU one — two branches of one function computing different physics
    from identical inputs, on the path dm_control's dog is trained on.
    `noslip_elliptic` still does not, and cannot: the blocked kernel is
    PYRAMIDAL-only and `solve_newton` will not route an elliptic model to it.
    Gate: `tests/physics3d/test_noslip_blocked_kernel.mojo`.

    ⚠ THAT WIRING IS WHY THE ROW ARRAYS BELOW ARE POINTERS. The blocked kernel
    keeps `Je`/`bias_e`/`force`/... in THREADGROUP memory (and spills `Je` to
    global on models where it does not fit), while `_newton_solve_env` holds
    per-thread `Array`s. A signature naming either storage excludes the
    other caller, and copying into locals is exactly what that kernel exists
    to avoid. An address-space-parameterized pointer is what they share — see
    the note above `_minv_jt`. Do not "simplify" it back to one storage kind
    without re-reading that note; it would fork this routine in two.

⚠ THE REJECTION IS INSIDE `costChange`, NOT AT ITS CALL SITE. MuJoCo's helper
restores `force` from `oldforce` and returns 0 when the change comes out above
1e-10; the call sites just do `improvement -= costChange(...)` and look
rejection-free. Both functions here therefore test the returned change and skip
the write — reading the call site alone would have produced a solver that
happily accepts cost-increasing steps.

⚠⚠ EVERY SCALAR HERE IS `Scalar[DTYPE]`. IT USED TO BE `Float64` (2026-08-10)

This module originally widened to `Float64` internally — the natural choice,
since the arithmetic below is where MuJoCo's own solver is most sensitive to
rounding. That made the whole pass UNCOMPILABLE ON GPU:

    Function 'air.convert.f.f64.f.f32' has Metal-unsupported instructions
    Function 'noeira_physics3d_solver_nosl...' has Metal-unsupported ...
    LLVM ERROR: Failed to verify LLVM IR for Metal

and `Float64` is off the table on the NVIDIA path too. Dog was then the only
model in the tree asking for this pass (`dog_xml`, `dog_fetch_xml`), so that
made "dog on GPU" and "noslip in Float64" mutually exclusive. The port is the
resolution, and it now also covers every manipulation model. The CPU path is
UNAFFECTED — it instantiates at
`DTYPE = float64`, so its arithmetic is bit-identical to before.

⚠ WHAT THE GPU PATH GIVES UP, STATED RATHER THAN DISCOVERED LATER. At
float32 three things below get noisier, and all three change WHEN the sweep
stops rather than what it converges to:

  * `improvement` accumulates `0.5*d*d*a_ii + d*res` over every row, then is
    compared against `tolerance` (1e-6) after scaling. A float32 sum over
    hundreds of rows carries ~1e-7 relative noise, which is the same order as
    the threshold — so the GPU may run one more or one fewer iteration than
    the CPU on the same state.
  * `_COST_REJECT` (1e-10) gates a `change` built from differences of
    products. That is catastrophic cancellation at float32, so a block the
    CPU accepts may be rejected on GPU and vice versa.
  * `k1 = a00 + a11 - a01 - a10` is a four-way cancellation feeding a
    division. When it lands under `_MINVAL` the pair is left at `mid`.

None of that can destabilise a contact — the `(mid + y, mid - y)` invariant
below is structural, not numerical, so the NORMAL force is preserved exactly
whatever the arithmetic does. It does mean a CPU-vs-GPU gate on a CONTACTING
dog should expect iteration-count divergence, which is the same regime
`test_quadruped_gpu_vs_cpu` already declines to bound.
"""

from std.math import sqrt
from std.sys import simd_width_of
from std.collections import Array
from max.gpu.memory import AddressSpace
from layout import TileTensor, TensorLayout
from ..fields.scratch import Scratch
from ..dynamics.ldl import _dof_ancestors

from ..constraints.scalar_rows import SROW_FRICTION
from ..gpu.constants import (
    CONTACT_SIZE,
    CONTACT_IDX_CONDIM,
)
from ..constraints.qcqp import mj_qcqp2, mj_qcqp3, mj_qcqp5


# ⚠ These stay `Float64` DELIBERATELY: they are `comptime`, so they never
from ..fields import DimsLike, Dims
# reach the emitted IR — `Scalar[DTYPE](_MINVAL)` folds at compile time. What
# Metal rejects is a `double` VALUE or CONVERSION in the kernel body, not a
# compile-time constant that happens to be spelled in double precision.
# All three are representable at float32 (min normal ~1.2e-38).
# `mjMINVAL`.
comptime _MINVAL: Float64 = 1e-15
# `extractBlock`'s diagonal floor when `flg_subR` is set.
comptime _DIAG_FLOOR: Float64 = 1e-10
# `costChange`'s "this made things worse" threshold.
comptime _COST_REJECT: Float64 = 1e-10


# ⚠⚠ THE ROW ARRAYS ARE POINTERS, NOT `Array`s, AND THAT IS THE WHOLE
# REASON ONE COPY OF THIS ROUTINE CAN SERVE BOTH SOLVERS.
#
# `_newton_solve_env` holds `Je`/`bias_e`/... as per-thread `Array`s;
# `_newton_blocked_fields_kernel` holds the same rows in THREADGROUP memory
# (`Je_sh`, `bias_e_sh`, ... — and `Je` itself spills to GLOBAL on models where
# it does not fit, so its address space is not even fixed across builds). A
# signature naming either storage excludes the other caller, and the blocked
# kernel cannot copy them into locals: `Je` is `E_CAP * V_CAP` scalars — ~38 kB
# on dog — and this kernel exists precisely to keep per-thread local memory
# small (see the `je_spills` note in `newton_solve`).
#
# A pointer with the address space as a PARAMETER is what both storages have in
# common: `Array.unsafe_ptr()` is GENERIC, `LayoutTensor.ptr` is SHARED or
# GENERIC, and everything downstream is identical. Verified on Metal for both
# address spaces including WRITE-THROUGH, since a per-thread buffer is exactly
# the shape that miscomputes there
# (`feedback_metal_wide_per_thread_inlinearray_miscompute`).
#
# ⚠ Read-only pointers leave the origin unbound (`_`); `force` is written, so
# it takes a real `MutOrigin` parameter.


@always_inline
def _minv_jt[
    DTYPE: DType,
    V_CAP: Int,
    L_M_INV: TensorLayout,
    JE_AS: AddressSpace = AddressSpace.GENERIC,
    SPARSE: Bool = False,
    N_CAP: Int = 1,
    IX_CAP: Int = 1,
](
    env: Int,
    nv: Int,
    m_inv: TileTensor[DTYPE, L_M_INV, MutAnyOrigin],
    Je: Pointer[Scalar[DTYPE], _, address_space=JE_AS],
    row: Int,
    mut out_v: Scratch[Scalar[DTYPE], V_CAP],
    je_n: Scratch[Int, N_CAP],
    je_ix: Scratch[Int, IX_CAP],
):
    """`out_v = M^-1 J_row^T`.

    Recomputed per use rather than cached for every row. The cache would be
    `num_edges * nv` doubles — ~95 kB on dog — which is fine on the CPU path
    and far too much for per-env GPU local memory, and this module is meant to
    serve both. If the CPU path ever needs the speed, hoist it: `J` and `M` do
    not change during the sweep, so the result is loop-invariant.

    ⚠ HOISTED AND SPARSE ON THE CPU PATH NOW (2026-09-05): `noslip_pyramidal`
    fills a per-row cache once under `CACHE`, and under `SPARSE` the sum runs
    over the row's nonzero dofs (`je_n` / `je_ix`, the Newton's own lists) —
    every skipped term is a product with an exact zero, so the value is the
    dense one bit for bit. The GPU legs keep both off (PERFORMANCE.md §13.6:
    noslip was 64% of dog's solve and two thirds of the reassemble scenes').
    """
    for i in range(nv):
        var acc = Scalar[DTYPE](0)
        comptime if SPARSE:
            for a in range(je_n[row]):
                var k = je_ix[row * nv + a]
                acc += rebind[Scalar[DTYPE]](
                    m_inv[env, i * nv + k]
                ) * Je[unsafe_offset = row * nv + k]
        else:
            for k in range(nv):
                acc += rebind[Scalar[DTYPE]](
                    m_inv[env, i * nv + k]
                ) * Je[unsafe_offset = row * nv + k]
        out_v[i] = acc


@always_inline
def _dot_row[
    DTYPE: DType,
    V_CAP: Int,
    JE_AS: AddressSpace = AddressSpace.GENERIC,
    SPARSE: Bool = False,
    N_CAP: Int = 1,
    IX_CAP: Int = 1,
](
    Je: Pointer[Scalar[DTYPE], _, address_space=JE_AS],
    row: Int,
    v: Scratch[Scalar[DTYPE], V_CAP],
    nv: Int,
    je_n: Scratch[Int, N_CAP],
    je_ix: Scratch[Int, IX_CAP],
) -> Scalar[DTYPE]:
    """`J_row . v` — over the row's nonzero dofs under `SPARSE`, bit-exact."""
    var acc = Scalar[DTYPE](0)
    comptime if SPARSE:
        for a in range(je_n[row]):
            var i = je_ix[row * nv + a]
            acc += Je[unsafe_offset=row * nv + i] * v[i]
    else:
        for i in range(nv):
            acc += Je[unsafe_offset=row * nv + i] * v[i]
    return acc


@always_inline
def _refresh_jar[
    DTYPE: DType,
    E_CAP: Int,
    V_CAP: Int,
    JE_AS: AddressSpace = AddressSpace.GENERIC,
    ROW_AS: AddressSpace = AddressSpace.GENERIC,
    SPARSE: Bool = False,
    N_CAP: Int = 1,
    IX_CAP: Int = 1,
](
    num_edges: Int,
    Je: Pointer[Scalar[DTYPE], _, address_space=JE_AS],
    bias_e: Pointer[Scalar[DTYPE], _, address_space=ROW_AS],
    qacc: Scratch[Scalar[DTYPE], V_CAP],
    mut jar: Scratch[Scalar[DTYPE], E_CAP],
    nv: Int,
    je_n: Scratch[Int, N_CAP],
    je_ix: Scratch[Int, IX_CAP],
):
    """`jar[e] = J_e . qacc + bias_e` for every row.

    Recomputed in full after each block rather than updated incrementally.
    Both cost `E_CAP * NV`; the full recompute cannot drift, and this pass runs
    at most `noslip_iterations` times per step.
    """
    for e in range(num_edges):
        jar[e] = (
            _dot_row[DTYPE, V_CAP, JE_AS, SPARSE](Je, e, qacc, nv, je_n, je_ix)
            + bias_e[unsafe_offset=e]
        )


@always_inline
def _dot_a_row[
    DTYPE: DType,
    A_CAP: Int,
    SW_CAP: Int,
    ROW_AS: AddressSpace = AddressSpace.GENERIC,
](
    A_mat: Scratch[Scalar[DTYPE], A_CAP],
    s: Int,
    ns: Int,
    sw_ix: Scratch[Int, SW_CAP],
    force: Pointer[Scalar[DTYPE], _, address_space=ROW_AS],
) -> Scalar[DTYPE]:
    """`A_S[s, :] . f_S` — MuJoCo's `residual` restricted to the swept rows
    (the other rows' constant contribution is already in `b_S`), without the
    `R f` subtraction our `A` never carried."""
    var acc = Scalar[DTYPE](0)
    var base = s * ns
    for t in range(ns):
        acc += A_mat[base + t] * force[unsafe_offset = sw_ix[t]]
    return acc


@always_inline
def _cost_change[
    DTYPE: DType
](
    a00: Scalar[DTYPE], a01: Scalar[DTYPE],
    a10: Scalar[DTYPE], a11: Scalar[DTYPE],
    f0: Scalar[DTYPE], f1: Scalar[DTYPE],
    old0: Scalar[DTYPE], old1: Scalar[DTYPE],
    r0: Scalar[DTYPE], r1: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """`costChange` for a 2x2 block: `0.5 d^T A d + d . res`.

    The caller must REJECT the update when this is positive — MuJoCo restores
    the old forces and counts zero improvement. Returned unclamped so the
    caller can do exactly that.

    ⚠ This is the cancellation-sensitive expression of the module: `f0 - old0`
    and `f1 - old1` are small differences of comparable numbers, then squared
    against `A`. See the float32 note in the module docstring.
    """
    var d0 = f0 - old0
    var d1 = f1 - old1
    var quad = (
        d0 * (a00 * d0 + a01 * d1) + d1 * (a10 * d0 + a11 * d1)
    ) * Scalar[DTYPE](0.5)
    return quad + d0 * r0 + d1 * r1


@always_inline
def _axpy_self[
    O: MutOrigin, //,
    DTYPE: DType,
](
    p: Pointer[Scalar[DTYPE], O],
    xo: Int,
    ao: Int,
    alpha: Scalar[DTYPE],
    n: Int,
):
    """`p[xo+k] += alpha * p[ao+k]` for `k < n`, `W` lanes at a time — the
    one-pointer twin of `cholesky._axpy_seg` (two ranges of ONE buffer; the
    exclusivity checker refuses the same mutable pointer twice). The caller
    keeps the two ranges disjoint."""
    comptime W = 2 * simd_width_of[DTYPE]()
    var av = SIMD[DTYPE, W](alpha)
    var k = 0
    while k + W <= n:
        p.unsafe_store(
            xo + k,
            p.unsafe_load[width=W](xo + k)
            + av * p.unsafe_load[width=W](ao + k),
        )
        k += W
    while k < n:
        p[unsafe_offset = xo + k] = p[unsafe_offset = xo + k] + alpha * p[unsafe_offset = ao + k]
        k += 1


@always_inline
def _dot_self[
    O: MutOrigin, //,
    DTYPE: DType,
](
    p: Pointer[Scalar[DTYPE], O],
    ao: Int,
    bo: Int,
    n: Int,
) -> Scalar[DTYPE]:
    """`Σ_k p[ao+k] · p[bo+k]`, `W` lanes at a time — two ranges of ONE
    buffer, like `_axpy_self`. A scalar reduction is a serial chain of `n`
    dependent adds (~4 cycles each); `W` partial sums cut that `W`-fold —
    the `A_S` build stage of `noslip_pyramidal` went from 44 to 21 µs on dog
    with this one change (§13.20)."""
    comptime W = 2 * simd_width_of[DTYPE]()
    var acc = SIMD[DTYPE, W](0)
    var k = 0
    while k + W <= n:
        acc += p.unsafe_load[width=W](ao + k) * p.unsafe_load[width=W](bo + k)
        k += W
    var r = acc.reduce_add()
    while k < n:
        r += p[unsafe_offset = ao + k] * p[unsafe_offset = bo + k]
        k += 1
    return r


@always_inline
def _tree_solve_cols[
    DTYPE: DType,
    V_CAP: Int,
    RV_CAP: Int,
    L_M_INV: TensorLayout,
    L_D: TensorLayout,
](
    env: Int,
    nv: Int,
    n: Int,
    ldl_L: TileTensor[DTYPE, L_M_INV, MutAnyOrigin],
    ldl_D: TileTensor[DTYPE, L_D, MutAnyOrigin],
    par: Scratch[Int, V_CAP],
    mut Wk: Scratch[Scalar[DTYPE], RV_CAP],
):
    """`mj_solveLD` on `n` vectors IN PLACE, column-major: `Wk[k*n + r]` is
    dof `k` of vector `r`. The core of `_minv_apply_rows`, exposed so a
    caller that can scatter its rows straight into this layout skips the
    two transposes (`noslip_elliptic`'s cache build: 110 rows on
    reassemble3, §13.21)."""
    var wp = Wk.unsafe_ptr()
    for i in range(nv - 1, -1, -1):
        var ri = i * nv
        var j = par[i]
        while j >= 0:
            _axpy_self[DTYPE](
                wp, j * n, i * n, -rebind[Scalar[DTYPE]](ldl_L[env, ri + j]), n
            )
            j = par[j]
    for i in range(nv):
        var d_i = rebind[Scalar[DTYPE]](ldl_D[env, i])
        var inv = Scalar[DTYPE](0)
        if d_i > Scalar[DTYPE](1e-14) or d_i < Scalar[DTYPE](-1e-14):
            inv = Scalar[DTYPE](1) / d_i
        for r in range(n):
            Wk[i * n + r] = Wk[i * n + r] * inv
    for i in range(nv):
        var ri = i * nv
        var j = par[i]
        while j >= 0:
            _axpy_self[DTYPE](
                wp, i * n, j * n, -rebind[Scalar[DTYPE]](ldl_L[env, ri + j]), n
            )
            j = par[j]


@always_inline
def _minv_apply_rows[
    DTYPE: DType,
    V_CAP: Int,
    RV_CAP: Int,
    L_M_INV: TensorLayout,
    L_D: TensorLayout,
](
    env: Int,
    nv: Int,
    n: Int,
    m_inv: TileTensor[DTYPE, L_M_INV, MutAnyOrigin],
    ldl_L: TileTensor[DTYPE, L_M_INV, MutAnyOrigin],
    ldl_D: TileTensor[DTYPE, L_D, MutAnyOrigin],
    par: Scratch[Int, V_CAP],
    tree_ok: Bool,
    B: Scratch[Scalar[DTYPE], RV_CAP],
    mut X: Scratch[Scalar[DTYPE], RV_CAP],
    mut Wk: Scratch[Scalar[DTYPE], RV_CAP],
):
    """`X[r*nv + k] = (M⁻¹ B[r])[k]` for the `n` rows `B[r*nv ..]`, in ONE
    pass: `mj_solveLD` with `n` vectors (engine_core_smooth.c:2113).

    WHY ONE PASS. A tree solve per row is a serial chain walk — ~3 µs a row
    on dog's 79 dofs, and 32 rows made the noslip SLOWER than the dense
    inverse it replaced (+100 µs). Here the vectors live column-major in
    `Wk[k*n + r]`, so every chain step is one contiguous `n`-wide axpy over
    all rows at once (`_axpy_self`, SIMD): the same flops, no dependency
    chain, `W` lanes wide.

    `tree_ok` False (no kinematic tree, `NTREE == 0`) is the dense product
    against `m_inv`; the GPU legs never reach this function.
    """
    if tree_ok:
        for r in range(n):
            for k in range(nv):
                Wk[k * n + r] = B[r * nv + k]
        _tree_solve_cols[DTYPE, V_CAP, RV_CAP](env, nv, n, ldl_L, ldl_D, par, Wk)
        for r in range(n):
            for k in range(nv):
                X[r * nv + k] = Wk[k * n + r]
    else:
        for r in range(n):
            for i in range(nv):
                var acc = Scalar[DTYPE](0)
                for k in range(nv):
                    acc += rebind[Scalar[DTYPE]](m_inv[env, i * nv + k]) * B[r * nv + k]
                X[r * nv + i] = acc


@always_inline
def _minv_apply[
    DTYPE: DType,
    V_CAP: Int,
    L_M_INV: TensorLayout,
    L_D: TensorLayout,
    TREE: Bool = False,
    A_CAP: Int = 1,
](
    env: Int,
    nv: Int,
    m_inv: TileTensor[DTYPE, L_M_INV, MutAnyOrigin],
    ldl_L: TileTensor[DTYPE, L_M_INV, MutAnyOrigin],
    ldl_D: TileTensor[DTYPE, L_D, MutAnyOrigin],
    # The dof ancestor table (`_dof_ancestors`), read only under `TREE`.
    dep: Scratch[Int, V_CAP],
    anc: Scratch[Int, A_CAP],
    tree_ok: Bool,
    b: Scratch[Scalar[DTYPE], V_CAP],
    mut x: Scratch[Scalar[DTYPE], V_CAP],
):
    """`x = M⁻¹ b`, the one product the noslip needs of the mass matrix.

    Under `TREE` with `tree_ok` it is `mj_solveLD` (engine_core_smooth.c:2113)
    on the step's tree-ordered `LᵀDL` factor of `M` and the dof ancestor
    table — scatter up the tree, divide by `D`, gather — so the noslip needs no
    dense inverse at all and the integrator stops computing one on this path
    (PERFORMANCE.md §13.16: 141 µs of dog's 498 µs step, plus the dense
    `M⁻¹Jᵀ` products here). Otherwise it is the dense product against
    `m_inv`: the GPU legs, and a model with no kinematic tree (`NTREE == 0`,
    whose `scratch.L` holds the dense LDLᵀ in the other convention).

    ⚠ NOT BIT-EXACT against the dense product — a different operation order
    — so it is gated against MuJoCo, not against the old checksum.
    """
    comptime if TREE:
        if tree_ok:
            # Ancestor lists in place of the `par` chase, and `L` through
            # its pointer: the same per-element operation order as
            # `_ldl_solve_tree_env` (bit-exact with the chase, §13.26).
            var xp = x.unsafe_ptr()
            var Lp = ldl_L.ptr.unsafe_offset(env * nv * nv)
            for i in range(nv):
                xp[unsafe_offset=i] = b[i]
            for i in range(nv - 1, -1, -1):
                var yi = xp[unsafe_offset=i]
                if yi != Scalar[DTYPE](0):
                    var ri = i * nv
                    for a in range(dep[i]):
                        var j = anc[ri + a]
                        xp[unsafe_offset=j] = (
                            xp[unsafe_offset=j]
                            - Lp[unsafe_offset=ri + j] * yi
                        )
            for i in range(nv):
                var d_i = rebind[Scalar[DTYPE]](ldl_D[env, i])
                if d_i > Scalar[DTYPE](1e-14) or d_i < Scalar[DTYPE](-1e-14):
                    xp[unsafe_offset=i] = xp[unsafe_offset=i] / d_i
                else:
                    xp[unsafe_offset=i] = Scalar[DTYPE](0)
            for i in range(nv):
                var ri = i * nv
                var acc_i = xp[unsafe_offset=i]
                for a in range(dep[i] - 1, -1, -1):
                    var j = anc[ri + a]
                    acc_i = (
                        acc_i
                        - Lp[unsafe_offset=ri + j] * xp[unsafe_offset=j]
                    )
                xp[unsafe_offset=i] = acc_i
            return
    for i in range(nv):
        var acc = Scalar[DTYPE](0)
        for k in range(nv):
            acc += rebind[Scalar[DTYPE]](m_inv[env, i * nv + k]) * b[k]
        x[i] = acc


def noslip_pyramidal[
    FO: MutOrigin, //,
    DTYPE: DType,
    E_CAP: Int,
    V_CAP: Int,
    MC_CAP: Int,
    MAX_CONTACTS: Int,
    MAX_CONDIM: Int,
    L_CONTACTS: TensorLayout,
    L_M_INV: TensorLayout,
    L_D: TensorLayout,
    # GENERIC/GENERIC is the per-env caller (per-thread `Array`s); the
    # blocked kernel passes SHARED rows and a `Je` that is SHARED or GLOBAL
    # depending on whether it fit. See the note above `_minv_jt`.
    JE_AS: AddressSpace = AddressSpace.GENERIC,
    ROW_AS: AddressSpace = AddressSpace.GENERIC,
    # CPU-path knobs (both off on the GPU legs): walk each row's nonzero dofs
    # (`je_n` / `je_ix`, the pyramidal Newton's own lists) and cache
    # `M^-1 J^T` per row for the whole sweep instead of recomputing it per
    # row per iteration — the `E_CAP * V_CAP` slab the note on `_minv_jt`
    # says a per-env GPU frame cannot hold. Bit-exact either way.
    SPARSE: Bool = False,
    N_CAP: Int = 1,
    IX_CAP: Int = 1,
    CACHE: Bool = False,
    TREE: Bool = False,
](
    env: Int,
    nc: Int,
    num_edges: Int,
    # ⚠ Keyed on MAX_CONTACTS, not MC_CAP. `MC_CAP = _max_one[MAX_CONTACTS]()` and the
    # two agree everywhere except MAX_CONTACTS == 0, but a LayoutTensor
    # parameter is part of the TYPE — passing the caller's tensor with the
    # wrong one is a compile error, not a coercion.
    contacts: TileTensor[DTYPE, L_CONTACTS, MutAnyOrigin],
    m_inv: TileTensor[DTYPE, L_M_INV, MutAnyOrigin],
    # The step's tree LDL of `M` and the dof parents; read only under `TREE`
    # (see `_minv_apply`). The other legs pass placeholders and `False`.
    ldl_L: TileTensor[DTYPE, L_M_INV, MutAnyOrigin],
    ldl_D: TileTensor[DTYPE, L_D, MutAnyOrigin],
    par: Scratch[Int, V_CAP],
    tree_ok: Bool,
    # ⚠ `Je` is indexed with stride `V_CAP`, and both callers lay it out with
    # stride `NV`. Those agree because `V_CAP = _max_one[NV]()`, i.e. they
    # differ only at NV == 0, where there are no rows to sweep.
    Je: Pointer[Scalar[DTYPE], _, address_space=JE_AS],
    bias_e: Pointer[Scalar[DTYPE], _, address_space=ROW_AS],
    # ⚠ DTYPE, NOT `Int`, and the row kind is an enum — so this reads
    # `Int(kind_e[i])`. The two callers store it differently: the blocked
    # kernel's shared-memory slab is single-dtype (`kind_e_sh` is DTYPE), while
    # `_newton_solve_env` holds an `Scratch[Int, E_CAP]`. DTYPE is the side to
    # unify on because it makes the BLOCKED caller free — it passes
    # `kind_e_sh.ptr` straight through — and that kernel is the one whose
    # reason for existing is small per-thread local memory. The per-env caller
    # pays an E_CAP-scalar mirror instead, built at the call site under the same
    # `comptime if NOSLIP_ITER > 0` guard, so a model without the pass pays
    # nothing on either path.
    kind_e: Pointer[Scalar[DTYPE], _, address_space=ROW_AS],
    R_e: Pointer[Scalar[DTYPE], _, address_space=ROW_AS],
    floss_e: Pointer[Scalar[DTYPE], _, address_space=ROW_AS],
    qacc_smooth: Scratch[Scalar[DTYPE], V_CAP],
    # ⚠ `Scalar[DTYPE]`, not `Float64` — the caller must build these in DTYPE
    # too, or the conversion reappears at the CALL SITE and Metal rejects it
    # there instead. `newton_solve` computes `1/(meaninertia*max(1,nv))` from
    # a DTYPE tensor read, with `max(1,nv)` folded to a comptime constant.
    scale: Scalar[DTYPE],
    tolerance: Scalar[DTYPE],
    # ⚠⚠ RUNTIME, NOT A COMPILE-TIME PARAMETER — it was `MAX_ITER` until
    # 2026-08-25. `m->opt.noslip_iterations` is a plain `int` MuJoCo loops to,
    # and this routine only ever used it as `range(...)`; making it comptime
    # meant the COUNT had to be known when the integrator was instantiated, so
    # every caller that loads a model at runtime (the studio, and the fidelity
    # harnesses that mirror it) had no way to ask for the pass at all. It now
    # arrives beside `scale` and `tolerance`, the two other numbers the same
    # convergence test needs, all three read from model meta by the caller.
    max_iter: Int,
    mut qacc: Scratch[Scalar[DTYPE], V_CAP],
    mut jar: Scratch[Scalar[DTYPE], E_CAP],
    # The one array this routine WRITES through — hence a real `MutOrigin`
    # rather than the unbound `_` its read-only siblings use.
    force: Pointer[Scalar[DTYPE], FO, address_space=ROW_AS],
    mut qfrc: Scratch[Scalar[DTYPE], V_CAP],
    nv: Int,
    je_n: Scratch[Int, N_CAP],
    je_ix: Scratch[Int, IX_CAP],
):
    """One `mj_solNoSlip` call: up to `max_iter` friction-only sweeps.

    On entry `qacc`, `jar` and `force` are the primal solver's converged
    output. On exit `force` has been redistributed within each friction pair,
    and `qacc` / `qfrc` have been recomputed from the new forces exactly as
    `dualFinish` does — from the FINAL forces, not accumulated incrementally,
    so the in-sweep `qacc` updates cannot leave a residue.
    """
    comptime NE_PYR = 2 * (MAX_CONDIM - 1)
    comptime ZERO = Scalar[DTYPE](0)
    comptime HALF = Scalar[DTYPE](0.5)
    comptime TWO = Scalar[DTYPE](2.0)
    comptime MINVAL = Scalar[DTYPE](_MINVAL)
    comptime DIAG_FLOOR = Scalar[DTYPE](_DIAG_FLOOR)
    comptime COST_REJECT = Scalar[DTYPE](_COST_REJECT)

    var mj_a = Scratch[Scalar[DTYPE], V_CAP](nv, fill=ZERO)
    var mj_b = Scratch[Scalar[DTYPE], V_CAP](nv, fill=ZERO)
    # The dof ancestor table for the three tree solves below (`Z`, `b_S`,
    # `dualFinish`), built once per call from `par`. CPU only: the GPU legs
    # carry a one-element placeholder they never read.
    comptime AN_CAP = V_CAP * V_CAP if TREE else 1
    var dep = Scratch[Int, V_CAP](nv if TREE else 1, uninitialized=0)
    var anc = Scratch[Int, AN_CAP](nv * nv if TREE else 1, uninitialized=0)
    comptime if TREE:
        if tree_ok:
            _dof_ancestors[V_CAP, AN_CAP](nv, par, dep, anc)

    # ── The CPU path (`CACHE`): MuJoCo's shape, on the SWEPT rows only ────
    #
    # MuJoCo builds `efc_AR = J M⁻¹ Jᵀ + R` before `solNoSlip`
    # (`mj_projectConstraint`) and the pass then only READS it: a residual is
    # one row of `AR` against the current forces (`residual`), a block's `Ac`
    # is four entries (`extractBlock`), and the sweep never touches `qacc`.
    # The refreshing legs below (the GPU ones) instead recompute every row's
    # `jar` after every pair — `E * nnz(J)` per pair.
    #
    # Two things make it cheaper than MuJoCo's own: (1) only the rows the
    # sweep MOVES — dry-friction dofs and the friction edges of condim ≥ 3
    # contacts — need `M⁻¹ Jᵀ`, an `A` row and a residual. On dog that is 36
    # rows of 119: the limit and normal-direction rows carry forces the sweep
    # never changes, so their whole contribution to every residual is a
    # constant, `J_S M⁻¹ Jᵀ_¬S f_¬S`, folded into `b_S` through ONE extra
    # `M⁻¹` apply (`qacc_ns = qacc_smooth + M⁻¹ Jᵀ_¬S f_¬S`, then
    # `b_S = J_S qacc_ns + bias_S`). (2) `A_S` is symmetric and computed for
    # the lower triangle only. PERFORMANCE.md §13.20 has the stage numbers:
    # the all-rows `M⁻¹ Jᵀ` cache was 76 of the pass's ~130 µs on dog.
    #
    # ⚠ NOT BIT-EXACT against the refreshing legs — the same quantities in a
    # different summation order — and gated against MuJoCo
    # (`test_noslip_vs_mujoco`, `test_noslip_reaches_the_runtime_path`).
    # ⚠ `jar` is LEFT AS IT WAS ON ENTRY under `CACHE`: no caller reads it
    # after this routine (`_newton_solve_env` writes `qacc` and `force` back).
    # ⚠ `A_S` is HEAP (`CAP = 0`): a cap of `E_CAP²` would be megabytes of
    # stack on a model with a large contact budget; the live `ns²` is a few kB.
    comptime SW_CAP = E_CAP if CACHE else 1
    var sw_slot = Scratch[Int, SW_CAP](num_edges if CACHE else 1, fill=-1)
    var sw_ix = Scratch[Int, SW_CAP](num_edges if CACHE else 1, uninitialized=0)
    var ns = 0
    comptime if CACHE:
        # Slot order = sweep order: dry-friction rows, then the pyramid pairs.
        for i in range(num_edges):
            if Int(kind_e[unsafe_offset=i]) == SROW_FRICTION:
                sw_slot[i] = ns
                sw_ix[ns] = i
                ns += 1
        for c in range(nc):
            var dim = Int(contacts[env, c * CONTACT_SIZE + CONTACT_IDX_CONDIM])
            if dim < 3:
                continue
            var base = c * NE_PYR
            for k in range(dim - 1):
                var j1 = base + 2 * k + 1
                if j1 >= num_edges:
                    break
                for j in range(j1 - 1, j1 + 1):
                    if sw_slot[j] < 0:
                        sw_slot[j] = ns
                        sw_ix[ns] = j
                        ns += 1

    # `A_S = J_S M⁻¹ J_Sᵀ`, built MuJoCo's way (`mj_projectConstraint`:
    # `mj_solveM2` then `mju_sqrMatTDSparse`). With `M = Lᵀ D L`,
    # `A_S = Z_Sᵀ Z_S` for `Z_S = D^-½ L⁻ᵀ J_Sᵀ` — HALF a solve per row, and
    # the half that stays SPARSE: pushing a row's entries up their dof
    # ancestors never leaves the ancestor-closed support of a contact
    # Jacobian, so it is O(chain²) per row instead of a dense fill, and no
    # `M⁻¹ Jᵀ` cache exists at all (§13.20: that cache was 25 µs of the pass
    # on dog and the `J · cache` products another 10). Lower triangle only,
    # mirrored. `zj` holds `Z_S` row-major; on the dense fallback (a model
    # with no kinematic tree) it holds `M⁻¹ J_Sᵀ` and `A_S[s, t] = J_t · zj[s]`.
    comptime MJ_CAP = E_CAP * V_CAP if CACHE else 1
    # Uninitialized, then the LIVE `ns * nv` zeroed by hand: `fill=` would
    # memset the whole `E_CAP * V_CAP` slab (76 kB on dog) for 2k live floats.
    var zj = Scratch[Scalar[DTYPE], MJ_CAP](
        ns * nv if CACHE else 1, uninitialized=ZERO
    )
    comptime if CACHE:
        for q in range(ns * nv):
            zj[q] = ZERO
    # `CAP = 0` is the HEAP leg — only on the CPU path. The GPU legs get a
    # one-element static array they never read.
    comptime A_CAP = 0 if CACHE else 1
    var A_mat = Scratch[Scalar[DTYPE], A_CAP](
        ns * ns if CACHE else 1, uninitialized=ZERO
    )
    var b_s = Scratch[Scalar[DTYPE], SW_CAP](ns if CACHE else 1, uninitialized=ZERO)
    comptime if CACHE:
        var tree_path = False
        comptime if TREE:
            tree_path = tree_ok
        if tree_path:
            var Lp = ldl_L.ptr.unsafe_offset(env * nv * nv)
            var dis = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=ZERO)
            for k in range(nv):
                var d_k = rebind[Scalar[DTYPE]](ldl_D[env, k])
                dis[k] = (
                    Scalar[DTYPE](1) / sqrt(d_k)
                    if d_k > Scalar[DTYPE](1e-14) else ZERO
                )
            # COLUMN-MAJOR, all swept rows at once: `wk[k*ns + s]` is dof
            # `k` of row `s`. Row by row, the push up the ancestors is a
            # read-modify-write chain through one row's entries — every
            # hop waits on the store before it (~7 µs a call on dog for
            # 27 rows, unchanged by ancestor tables, §13.26). Here one hop
            # is one `ns`-wide axpy over all rows, the shape of
            # `_tree_solve_cols`, and the rows go back to row-major in one
            # transpose for the Gram below. Each entry receives the same
            # subtractions from the same `i` in the same order (all-zero
            # entries add an exact zero), so this is the same arithmetic
            # as the per-row walk. Heap (`CAP = 0`): the live `nv * ns`.
            var wk = Scratch[Scalar[DTYPE], 0](nv * ns, uninitialized=ZERO)
            var wp = wk.unsafe_ptr()
            comptime W_Z = 2 * simd_width_of[DTYPE]()
            var qz = 0
            while qz + W_Z <= nv * ns:
                wp.unsafe_store(qz, SIMD[DTYPE, W_Z](0))
                qz += W_Z
            while qz < nv * ns:
                wp[unsafe_offset=qz] = ZERO
                qz += 1
            for s in range(ns):
                var e = sw_ix[s]
                comptime if SPARSE:
                    for a in range(je_n[e]):
                        var k = je_ix[e * nv + a]
                        wp[unsafe_offset=k * ns + s] = Je[
                            unsafe_offset = e * nv + k
                        ]
                else:
                    for k in range(nv):
                        wp[unsafe_offset=k * ns + s] = Je[
                            unsafe_offset = e * nv + k
                        ]
            # `L⁻ᵀ`: for `i` descending, push block `i` up its ancestors —
            # the first half of `mj_solveLD`, `ns` lanes wide.
            for i in range(nv - 1, -1, -1):
                var ri = i * nv
                var j = par[i]
                while j >= 0:
                    _axpy_self[DTYPE](
                        wp, j * ns, i * ns, -Lp[unsafe_offset=ri + j], ns
                    )
                    j = par[j]
            # `D^-½`, then back to row-major `zj[s*nv + k]`.
            var zp0 = zj.unsafe_ptr()
            for k in range(nv):
                var dk = dis[k]
                var kb = k * ns
                for s in range(ns):
                    zp0[unsafe_offset=s * nv + k] = (
                        wp[unsafe_offset=kb + s] * dk
                    )
            var zp = zj.unsafe_ptr()
            for s in range(ns):
                var zs = s * nv
                for t in range(s + 1):
                    var acc = _dot_self[DTYPE](zp, zs, t * nv, nv)
                    A_mat[s * ns + t] = acc
                    A_mat[t * ns + s] = acc
        else:
            for s in range(ns):
                _minv_jt[DTYPE, V_CAP, JE_AS=JE_AS, SPARSE=SPARSE](
                    env, nv, m_inv, Je, sw_ix[s], mj_a, je_n, je_ix
                )
                for q in range(nv):
                    zj[s * nv + q] = mj_a[q]
            for s in range(ns):
                for t in range(s + 1):
                    var g = sw_ix[t]
                    var acc = ZERO
                    comptime if SPARSE:
                        for a in range(je_n[g]):
                            var k = je_ix[g * nv + a]
                            acc += Je[unsafe_offset = g * nv + k] * zj[s * nv + k]
                    else:
                        for k in range(nv):
                            acc += Je[unsafe_offset = g * nv + k] * zj[s * nv + k]
                    A_mat[s * ns + t] = acc
                    A_mat[t * ns + s] = acc

        # `b_S = J_S (qacc_smooth + M⁻¹ Jᵀ_¬S f_¬S) + bias_S`; `qfrc` is
        # scratch here — `dualFinish` below rewrites it from zero.
        for k in range(nv):
            qfrc[k] = ZERO
        for g in range(num_edges):
            if sw_slot[g] >= 0:
                continue
            var f = force[unsafe_offset=g]
            if f == ZERO:
                continue
            comptime if SPARSE:
                for a in range(je_n[g]):
                    var k = je_ix[g * nv + a]
                    qfrc[k] += Je[unsafe_offset = g * nv + k] * f
            else:
                for k in range(nv):
                    qfrc[k] += Je[unsafe_offset = g * nv + k] * f
        _minv_apply[DTYPE, V_CAP, TREE=TREE, A_CAP=AN_CAP](
            env, nv, m_inv, ldl_L, ldl_D, dep, anc, tree_ok, qfrc, mj_a
        )
        for k in range(nv):
            mj_a[k] += qacc_smooth[k]  # qacc_ns
        for s in range(ns):
            var e = sw_ix[s]
            b_s[s] = (
                _dot_row[DTYPE, V_CAP, JE_AS, SPARSE](Je, e, mj_a, nv, je_n, je_ix)
                + bias_e[unsafe_offset=e]
            )

    for it in range(max_iter):
        var improvement = ZERO

        # `iter == 0` correction: MuJoCo folds in the R-weighted force energy
        # once, so the first iteration's improvement is comparable with the
        # primal solver's own.
        if it == 0:
            for i in range(num_edges):
                var f = force[unsafe_offset=i]
                improvement += HALF * f * f * R_e[unsafe_offset=i]

        # ── sweep 1: dry-friction dof rows, box-clamped to +-frictionloss ──
        for i in range(num_edges):
            if Int(kind_e[unsafe_offset=i]) != SROW_FRICTION:
                continue
            var a_ii = ZERO
            var res = ZERO
            comptime if CACHE:
                # `ARdiaginv` + `residual`: the diagonal and one row-dot.
                var s_i = sw_slot[i]
                a_ii = A_mat[s_i * ns + s_i]
                res = b_s[s_i] + _dot_a_row[DTYPE, A_CAP, SW_CAP, ROW_AS](
                    A_mat, s_i, ns, sw_ix, force
                )
            else:
                _minv_jt[DTYPE, V_CAP, JE_AS=JE_AS, SPARSE=SPARSE](
                    env, nv, m_inv, Je, i, mj_a, je_n, je_ix
                )
                a_ii = _dot_row[DTYPE, V_CAP, JE_AS, SPARSE](Je, i, mj_a, nv, je_n, je_ix)
                res = jar[i]
            var arinv = Scalar[DTYPE](1.0) / (
                a_ii if a_ii > MINVAL else MINVAL
            )

            var old = force[unsafe_offset=i]
            var f = old - res * arinv
            var lim = floss_e[unsafe_offset=i]
            if f < -lim:
                f = -lim
            elif f > lim:
                f = lim

            var d = f - old
            if d != ZERO:
                force[unsafe_offset=i] = f
                comptime if not CACHE:
                    for k in range(nv):
                        qacc[k] += d * mj_a[k]
                    _refresh_jar[DTYPE, E_CAP, V_CAP, JE_AS, ROW_AS, SPARSE](
                        num_edges, Je, bias_e, qacc, jar, nv, je_n, je_ix
                    )
            # `0.5*d^2/ARinv` — and `1/ARinv` is `a_ii`, so no division here.
            improvement -= HALF * d * d * a_ii + d * res

        # ── sweep 2: contact friction, one opposing pyramid pair at a time ──
        for c in range(nc):
            var dim = Int(contacts[env, c * CONTACT_SIZE + CONTACT_IDX_CONDIM])
            if dim < 3:
                continue  # frictionless contact: no friction rows to sweep
            var base = c * NE_PYR
            for k in range(dim - 1):
                var j0 = base + 2 * k
                var j1 = j0 + 1
                if j1 >= num_edges:
                    break

                var a00 = ZERO
                var a01 = ZERO
                var a10 = ZERO
                var a11 = ZERO
                var r0 = ZERO
                var r1 = ZERO
                comptime if CACHE:
                    # `extractBlock` + `residual`: four entries of `A` and two
                    # row-dots against the current forces.
                    var s0 = sw_slot[j0]
                    var s1 = sw_slot[j1]
                    a00 = A_mat[s0 * ns + s0]
                    a01 = A_mat[s0 * ns + s1]
                    a10 = A_mat[s1 * ns + s0]
                    a11 = A_mat[s1 * ns + s1]
                    r0 = b_s[s0] + _dot_a_row[DTYPE, A_CAP, SW_CAP, ROW_AS](
                        A_mat, s0, ns, sw_ix, force
                    )
                    r1 = b_s[s1] + _dot_a_row[DTYPE, A_CAP, SW_CAP, ROW_AS](
                        A_mat, s1, ns, sw_ix, force
                    )
                else:
                    _minv_jt[DTYPE, V_CAP, JE_AS=JE_AS, SPARSE=SPARSE](
                        env, nv, m_inv, Je, j0, mj_a, je_n, je_ix
                    )
                    _minv_jt[DTYPE, V_CAP, JE_AS=JE_AS, SPARSE=SPARSE](
                        env, nv, m_inv, Je, j1, mj_b, je_n, je_ix
                    )
                    # `Ac` = A submatrix, diagonal clamped (flg_subR semantics:
                    # R is NOT part of it).
                    a00 = _dot_row[DTYPE, V_CAP, JE_AS, SPARSE](Je, j0, mj_a, nv, je_n, je_ix)
                    a01 = _dot_row[DTYPE, V_CAP, JE_AS, SPARSE](Je, j0, mj_b, nv, je_n, je_ix)
                    a10 = _dot_row[DTYPE, V_CAP, JE_AS, SPARSE](Je, j1, mj_a, nv, je_n, je_ix)
                    a11 = _dot_row[DTYPE, V_CAP, JE_AS, SPARSE](Je, j1, mj_b, nv, je_n, je_ix)
                    r0 = jar[j0]
                    r1 = jar[j1]
                if a00 < DIAG_FLOOR:
                    a00 = DIAG_FLOOR
                if a11 < DIAG_FLOOR:
                    a11 = DIAG_FLOOR

                var old0 = force[unsafe_offset=j0]
                var old1 = force[unsafe_offset=j1]

                # `bc = res - Ac * oldforce`
                var b0 = r0 - (a00 * old0 + a01 * old1)
                var b1 = r1 - (a10 * old0 + a11 * old1)

                # The pair is written as (mid + y, mid - y), so `mid` — the
                # NORMAL contribution — is invariant no matter which branch
                # fires. That is the whole "normal forces frozen" property,
                # and it is STRUCTURAL: it survives the float32 port intact,
                # because every branch below is symmetric about `mid` by
                # construction rather than by cancellation.
                var mid = HALF * (old0 + old1)
                var k1 = a00 + a11 - a01 - a10
                var k0 = mid * (a00 - a11) + b0 - b1

                var f0 = mid
                var f1 = mid
                if k1 >= MINVAL:
                    var y = -k0 / k1
                    if y < -mid:
                        f0 = ZERO
                        f1 = TWO * mid
                    elif y > mid:
                        f0 = TWO * mid
                        f1 = ZERO
                    else:
                        f0 = mid + y
                        f1 = mid - y

                var change = _cost_change[DTYPE](
                    a00, a01, a10, a11, f0, f1, old0, old1, r0, r1
                )
                if change > COST_REJECT:
                    # Made it worse — MuJoCo restores and counts nothing.
                    continue

                var d0 = f0 - old0
                var d1 = f1 - old1
                if d0 != ZERO or d1 != ZERO:
                    force[unsafe_offset=j0] = f0
                    force[unsafe_offset=j1] = f1
                    comptime if not CACHE:
                        for q in range(nv):
                            qacc[q] += d0 * mj_a[q] + d1 * mj_b[q]
                        _refresh_jar[DTYPE, E_CAP, V_CAP, SPARSE=SPARSE](
                            num_edges, Je, bias_e, qacc, jar, nv, je_n, je_ix
                        )
                improvement -= change

        improvement *= scale
        if improvement < tolerance:
            break

    # ── dualFinish ─────────────────────────────────────────────────────────
    # `qfrc_constraint = J^T f`, then `qacc = M^-1 qfrc + qacc_smooth`.
    #
    # ⚠ RECOMPUTED FROM THE FINAL FORCES, not carried over from the in-sweep
    # `qacc +=` updates. Those exist only to keep `jar` current between blocks;
    # MuJoCo recomputes here and so does this, so any drift they accumulated is
    # discarded rather than integrated.
    for i in range(nv):
        qfrc[i] = Scalar[DTYPE](0)
    for e in range(num_edges):
        var f = force[unsafe_offset=e]
        if f == Scalar[DTYPE](0):
            continue
        comptime if SPARSE:
            for a in range(je_n[e]):
                var i = je_ix[e * nv + a]
                qfrc[i] += Je[unsafe_offset=e * nv + i] * f
        else:
            for i in range(nv):
                qfrc[i] += Je[unsafe_offset=e * nv + i] * f
    comptime if TREE:
        _minv_apply[DTYPE, V_CAP, TREE=True, A_CAP=AN_CAP](
            env, nv, m_inv, ldl_L, ldl_D, dep, anc, tree_ok, qfrc, mj_a
        )
        for i in range(nv):
            qacc[i] = mj_a[i] + qacc_smooth[i]
    else:
        for i in range(nv):
            var acc = Scalar[DTYPE](0)
            for k in range(nv):
                acc += rebind[Scalar[DTYPE]](m_inv[env, i * nv + k]) * qfrc[k]
            qacc[i] = acc + qacc_smooth[i]


# =============================================================================
# ELLIPTIC cone
# =============================================================================


@always_inline
def _minv_dense[
    DTYPE: DType,
    R_CAP: Int,
    V_CAP: Int,
    NT: Int,
    D: DimsLike,
    L_M_INV: TensorLayout,
    SPARSE: Bool = False,
    N_CAP: Int = 1,
    IX_CAP: Int = 1,
](
    env: Int,
    dims: D,
    m_inv: TileTensor[DTYPE, L_M_INV, MutAnyOrigin],
    J: Scratch[Scalar[DTYPE], R_CAP * V_CAP],
    row: Int,
    mut out_v: Scratch[Scalar[DTYPE], NT * V_CAP],
    slot: Int,
    cn_n: Scratch[Int, N_CAP],
    cn_ix: Scratch[Int, IX_CAP],
    c: Int,
):
    """`out_v[slot] = M^-1 J_row^T` for a row of a dense `[R_CAP, nv]` Jacobian.

    The elliptic path stores its contact Jacobians as a flat `[MC_CAP*NT, nv]`
    tangential block plus a separate `[MC_CAP, nv]` normal one, rather than one
    interleaved edge list — hence a helper keyed on the array and a row index
    within it. `out_v` holds one column per tangential row of the contact
    being swept, so the whole `nt x nt` AR block can be built from it without
    recomputing the matvecs.
    """
    var nv = dims.get_nv()
    for i in range(nv):
        var acc = Scalar[DTYPE](0)
        comptime if SPARSE:
            # `cn_n` / `cn_ix`: contact `c`'s nonzero dofs (the elliptic
            # Newton's lists); the skipped terms are exact zeros.
            for a in range(cn_n[c]):
                var k = cn_ix[c * nv + a]
                acc += rebind[Scalar[DTYPE]](m_inv[env, i * nv + k]) * J[
                    row * nv + k
                ]
        else:
            for k in range(nv):
                acc += rebind[Scalar[DTYPE]](m_inv[env, i * nv + k]) * J[
                    row * nv + k
                ]
        out_v[slot * nv + i] = acc


@always_inline
def _dot_dense[
    DTYPE: DType,
    R_CAP: Int,
    V_CAP: Int,
    NT: Int,
    SPARSE: Bool = False,
    N_CAP: Int = 1,
    IX_CAP: Int = 1,
](
    J: Scratch[Scalar[DTYPE], R_CAP * V_CAP],
    row: Int,
    v: Scratch[Scalar[DTYPE], NT * V_CAP],
    slot: Int,
    nv: Int,
    cn_n: Scratch[Int, N_CAP],
    cn_ix: Scratch[Int, IX_CAP],
    c: Int,
) -> Scalar[DTYPE]:
    """`J_row . v[slot]` — over contact `c`'s nonzero dofs under `SPARSE`."""
    var acc = Scalar[DTYPE](0)
    comptime if SPARSE:
        for a in range(cn_n[c]):
            var i = cn_ix[c * nv + a]
            acc += J[row * nv + i] * v[slot * nv + i]
    else:
        for i in range(nv):
            acc += J[row * nv + i] * v[slot * nv + i]
    return acc


@always_inline
def _solve_contact_qcqp[
    DTYPE: DType, NT: Int, T_CAP: Int
](
    nt: Int,
    cb: Int,
    Ac: Array[Scalar[DTYPE], NT * NT],
    bc: Array[Scalar[DTYPE], NT],
    fr_c: Scratch[Scalar[DTYPE], T_CAP],
    fn_v: Scalar[DTYPE],
    mut vf: Array[Scalar[DTYPE], NT],
) -> Bool:
    """MuJoCo's QCQP dispatch by contact dimension. Returns `flg_active`.

        dim 3 -> nt 2 -> mju_QCQP2
        dim 4 -> nt 3 -> mju_QCQP3
        else  ->         mju_QCQP   (n = dim-1; only dim 6 -> nt 5 occurs)

    ⚠ THE `comptime if NT >= n` GUARDS ARE NOT DEFENSIVE, THEY ARE A COMPILE
    BUDGET. `mj_qcqp5` is a 20-iteration Cholesky loop; instantiating it for a
    condim-3 model would be pure compile time for a branch that cannot be
    reached, and this file is already inside the per-env solve body that Metal
    has to fit. A model whose `MAX_CONDIM` is 3 therefore compiles exactly what
    it compiled before this generalization.

    ⚠ MuJoCo's own `else` is a general `mju_QCQP(n)`, ours is fixed at 5.
    `condim` is one of {1, 3, 4, 6} in MuJoCo (`mjCGeom`), so `nt` is one of
    {0, 2, 3, 5} and nothing else is reachable. A hypothetical `nt == 4` still
    behaves: `mj_qcqp5` scales by `d` and unscales by it, so a padded row with
    `d = 0` contributes nothing and returns 0.
    """
    comptime ZERO = Scalar[DTYPE](0)
    if nt == 2:
        var A2 = Array[Scalar[DTYPE], 4](fill=ZERO)
        A2[0] = Ac[0]
        A2[1] = Ac[1]
        A2[2] = Ac[NT]
        A2[3] = Ac[NT + 1]
        var b2 = Array[Scalar[DTYPE], 2](fill=ZERO)
        b2[0] = bc[0]
        b2[1] = bc[1]
        var d2 = Array[Scalar[DTYPE], 2](fill=ZERO)
        d2[0] = fr_c[cb]
        d2[1] = fr_c[cb + 1]
        var f0 = ZERO
        var f1 = ZERO
        var act = mj_qcqp2[DTYPE](f0, f1, A2, b2, d2, fn_v)
        vf[0] = f0
        vf[1] = f1
        return act

    comptime if NT >= 3:
        if nt == 3:
            var A3 = Array[Scalar[DTYPE], 9](fill=ZERO)
            for t in range(3):
                for u in range(3):
                    A3[t * 3 + u] = Ac[t * NT + u]
            var b3 = Array[Scalar[DTYPE], 3](fill=ZERO)
            var d3 = Array[Scalar[DTYPE], 3](fill=ZERO)
            for t in range(3):
                b3[t] = bc[t]
                d3[t] = fr_c[cb + t]
            var f0 = ZERO
            var f1 = ZERO
            var f2 = ZERO
            var act = mj_qcqp3[DTYPE](f0, f1, f2, A3, b3, d3, fn_v)
            vf[0] = f0
            vf[1] = f1
            vf[2] = f2
            return act

    comptime if NT >= 4:
        var A5 = Array[Scalar[DTYPE], 25](fill=ZERO)
        for t in range(nt):
            for u in range(nt):
                A5[t * 5 + u] = Ac[t * NT + u]
        var b5 = Array[Scalar[DTYPE], 5](fill=ZERO)
        var d5 = Array[Scalar[DTYPE], 5](fill=ZERO)
        for t in range(nt):
            b5[t] = bc[t]
            d5[t] = fr_c[cb + t]
        var v5 = Array[Scalar[DTYPE], 5](fill=ZERO)
        var act = mj_qcqp5[DTYPE](v5, A5, b5, d5, fn_v)
        for t in range(nt):
            vf[t] = v5[t]
        return act

    return False



@always_inline
def _refresh_jar_elliptic[
    DTYPE: DType,
    MC_CAP: Int,
    NT: Int,
    T_CAP: Int,
    V_CAP: Int,
    S_CAP: Int,
    EQ_CAP: Int,
](
    nc: Int,
    ns: Int,
    neq_rows: Int,
    nt_c: Scratch[Int, MC_CAP],
    Jn_c: Scratch[Scalar[DTYPE], MC_CAP * V_CAP],
    Jt_c: Scratch[Scalar[DTYPE], T_CAP * V_CAP],
    pb_c: Scratch[Scalar[DTYPE], MC_CAP],
    bt_c: Scratch[Scalar[DTYPE], T_CAP],
    sr_dof: Scratch[Int, S_CAP],
    sr_sign: Scratch[Scalar[DTYPE], S_CAP],
    sr_bias: Scratch[Scalar[DTYPE], S_CAP],
    eq_J: Scratch[Scalar[DTYPE], EQ_CAP * V_CAP],
    eq_bias: Scratch[Scalar[DTYPE], EQ_CAP],
    qacc: Scratch[Scalar[DTYPE], V_CAP],
    mut jar_n: Scratch[Scalar[DTYPE], MC_CAP],
    mut jar_t: Scratch[Scalar[DTYPE], T_CAP],
    mut sr_jar: Scratch[Scalar[DTYPE], S_CAP],
    mut eq_jar: Scratch[Scalar[DTYPE], EQ_CAP],
    nv: Int,
):
    """`jar = J qacc + bias` for EVERY row of the elliptic system.

    ⚠ EVERY row, not just the friction ones. MuJoCo's `residual()` reads
    `efc_AR * efc_force` over the whole constraint block, so a friction update
    is visible to every subsequent block — including the normal and limit rows
    this sweep never writes. Refreshing only the rows the sweep touches would
    turn a Gauss-Seidel pass into a Jacobi one against the rest of the system.
    """
    for c in range(nc):
        var jn = pb_c[c]
        var nt = nt_c[c]
        for t in range(nt):
            jar_t[c * NT + t] = bt_c[c * NT + t]
        for i in range(nv):
            var qa = qacc[i]
            jn += Jn_c[c * nv + i] * qa
            for t in range(nt):
                jar_t[c * NT + t] += Jt_c[(c * NT + t) * nv + i] * qa
        jar_n[c] = jn
    for s in range(ns):
        sr_jar[s] = sr_bias[s] + sr_sign[s] * qacc[sr_dof[s]]
    for e in range(neq_rows):
        var je = eq_bias[e]
        for d in range(nv):
            je += eq_J[e * nv + d] * qacc[d]
        eq_jar[e] = je


def noslip_elliptic[
    DTYPE: DType,
    MC_CAP: Int,
    NT: Int,
    T_CAP: Int,
    V_CAP: Int,
    S_CAP: Int,
    EQ_CAP: Int,
    D: DimsLike,
    L_M_INV: TensorLayout,
    L_D: TensorLayout,
    # CPU-path knobs, both off on the GPU legs — see `noslip_pyramidal`.
    SPARSE: Bool = False,
    N_CAP: Int = 1,
    IX_CAP: Int = 1,
    CACHE: Bool = False,
    TREE: Bool = False,
](
    env: Int,
    nc: Int,
    ns: Int,
    neq_rows: Int,
    dims: D,
    m_inv: TileTensor[DTYPE, L_M_INV, MutAnyOrigin],
    # The step's tree LDL of `M` and the dof parents; read only under `TREE`
    # (see `_minv_apply`). The other legs pass placeholders and `False`.
    ldl_L: TileTensor[DTYPE, L_M_INV, MutAnyOrigin],
    ldl_D: TileTensor[DTYPE, L_D, MutAnyOrigin],
    par: Scratch[Int, V_CAP],
    tree_ok: Bool,
    # ── contact rows: one normal + `nt_c[c]` tangential, `nt_c[c] = dim-1` ──
    nt_c: Scratch[Int, MC_CAP],
    Jn_c: Scratch[Scalar[DTYPE], MC_CAP * V_CAP],
    Jt_c: Scratch[Scalar[DTYPE], T_CAP * V_CAP],
    fr_c: Scratch[Scalar[DTYPE], T_CAP],
    D_n_c: Scratch[Scalar[DTYPE], MC_CAP],
    D_t_c: Scratch[Scalar[DTYPE], T_CAP],
    pb_c: Scratch[Scalar[DTYPE], MC_CAP],
    bt_c: Scratch[Scalar[DTYPE], T_CAP],
    # ── scalar rows: joint limits + dry-friction dofs (J = sign * e_dof) ──
    sr_dof: Scratch[Int, S_CAP],
    sr_kind: Scratch[Int, S_CAP],
    sr_sign: Scratch[Scalar[DTYPE], S_CAP],
    sr_R: Scratch[Scalar[DTYPE], S_CAP],
    sr_bias: Scratch[Scalar[DTYPE], S_CAP],
    sr_floss: Scratch[Scalar[DTYPE], S_CAP],
    # ── equality rows: tendon + connect/weld (dense J) ──
    eq_J: Scratch[Scalar[DTYPE], EQ_CAP * V_CAP],
    eq_D: Scratch[Scalar[DTYPE], EQ_CAP],
    eq_bias: Scratch[Scalar[DTYPE], EQ_CAP],
    qacc_smooth: Scratch[Scalar[DTYPE], V_CAP],
    scale: Scalar[DTYPE],
    tolerance: Scalar[DTYPE],
    # ⚠ RUNTIME, for the reason `noslip_pyramidal`'s copy of this note gives.
    max_iter: Int,
    mut qacc: Scratch[Scalar[DTYPE], V_CAP],
    mut fn_a: Scratch[Scalar[DTYPE], MC_CAP],
    mut ft_a: Scratch[Scalar[DTYPE], T_CAP],
    mut jar_n_a: Scratch[Scalar[DTYPE], MC_CAP],
    mut jar_t_a: Scratch[Scalar[DTYPE], T_CAP],
    mut sr_f: Scratch[Scalar[DTYPE], S_CAP],
    mut sr_jar: Scratch[Scalar[DTYPE], S_CAP],
    mut eq_f: Scratch[Scalar[DTYPE], EQ_CAP],
    mut eq_jar: Scratch[Scalar[DTYPE], EQ_CAP],
    mut qfrc: Scratch[Scalar[DTYPE], V_CAP],
    cn_n: Scratch[Int, N_CAP],
    cn_ix: Scratch[Int, IX_CAP],
):
    """One `mj_solNoSlip` call on the ELLIPTIC cone: up to `max_iter` sweeps.

    Transcribed from the `mjCNSTR_CONTACT_ELLIPTIC` branch of MuJoCo's
    `mj_solNoSlip` (`engine_solver.c:653` in 3.6.0, `solveQCQP` in 3.11.0).

    HOW THIS DIFFERS FROM THE PYRAMIDAL BRANCH — the two share a name and
    almost nothing else:

      * ROW LAYOUT. A pyramidal contact of dim d has `2*(d-1)` rows, all of
        them friction, and the normal force is only implicit in their sum.
        An elliptic contact has ONE normal row followed by `d-1` tangential
        ones. So `force[i]` here IS the normal force, and freezing it needs no
        arithmetic trick at all: the sweep simply never writes that row. The
        pyramidal `(mid + y, mid - y)` construction exists precisely because
        it has no such row to leave alone.
      * SOLVE. Pyramidal minimises a 1-D quadratic in `y` over `[-mid, mid]`,
        which is closed form. Elliptic solves a QCQP over the friction
        ellipsoid `sum (f_j/mu_j)^2 <= f_n^2` — Newton's method on the dual
        variable, not a projection.
      * A FEASIBLE ANSWER IS STILL RE-PROJECTED. When the QCQP reports the
        constraint active, MuJoCo pushes `v` back onto the ellipsoid boundary
        ("in case QCQP is approximate"). That is not a clamp — it can move a
        point that is already inside — and skipping it leaves a slowly
        drifting friction force under sustained sliding.

    ⚠ ANY CONDIM, as of 2026-08-13. This used to say "condim 3 only, because
    the SOLVER is", which was true: the primal path cached exactly three
    Jacobian rows per contact and one isotropic `mu`, so a condim-4 contact
    lost its torsional row before this function was reached, and writing the
    dim-4/5 branches here would have been unreachable code. The primal path
    now carries `dim-1` rows with per-direction friction, so all three of
    MuJoCo's dispatches are live and reachable:

        nt == 2   mju_QCQP2    condim 3   slide x2
        nt == 3   mju_QCQP3    condim 4   slide x2 + torsion
        nt == 5   mju_QCQP     condim 6   slide x2 + torsion + roll x2

    `nt == 0` is a FRICTIONLESS contact and is skipped entirely — there is no
    friction row to sweep. The QCQP3/QCQP5 instantiations are `comptime if`-ed
    on `NT` so a condim-3 model compiles exactly what it compiled before;
    QCQP5's 20-iteration Cholesky loop is not free to instantiate.

    ⚠ THE `improvement` ACCUMULATOR SPANS EVERY ROW KIND. The iteration-0
    correction is `0.5 f^2 R` over ALL `nefc` rows — contact normal AND
    tangents AND limits AND dry friction AND equalities — because MuJoCo's
    `improvement` is comparable with the primal solver's cost. Summing it over
    only the rows the sweep writes makes the loop stop on a different
    iteration, which is the one failure mode here that produces a plausible
    wrong answer rather than an obviously wrong one.

    On entry `qacc`, the `jar_*` arrays and the force arrays are the primal
    solver's converged output. On exit the TANGENTIAL forces have been
    re-solved with the normal ones frozen, and `qacc` / `qfrc` are recomputed
    from the final forces exactly as `dualFinish` does.
    """
    var nv = dims.get_nv()
    comptime ZERO = Scalar[DTYPE](0)
    comptime HALF = Scalar[DTYPE](0.5)
    comptime ONE = Scalar[DTYPE](1)
    comptime MINVAL = Scalar[DTYPE](_MINVAL)
    comptime DIAG_FLOOR = Scalar[DTYPE](_DIAG_FLOOR)
    comptime COST_REJECT = Scalar[DTYPE](_COST_REJECT)

    # `M^-1 J_t^T`, one column per tangential row of the contact being swept.
    var MinvJ = Scratch[Scalar[DTYPE], NT * V_CAP](NT * nv, fill=ZERO)
    # `M^-1 J_t^T` for every tangent row, once — loop-invariant across the
    # sweeps, the same value `_minv_dense` produced per use.
    comptime MJC_CAP = T_CAP * V_CAP if CACHE else 1
    var minv_cache = Scratch[Scalar[DTYPE], MJC_CAP](
        nc * NT * nv if CACHE else 1, uninitialized=ZERO
    )
    comptime if CACHE:
        var tree_path = False
        comptime if TREE:
            tree_path = tree_ok
        if tree_path:
            # Every tangent row's `M⁻¹ J_tᵀ` in ONE tree solve. The rows'
            # nonzeros are scattered STRAIGHT into the column-major work slab
            # `_tree_solve_cols` wants (`Wk[k*n + r]`), so there is no
            # row-major staging copy, no whole-cap memset and no input
            # transpose (§13.21: the cache build was 17.5 µs of the pass on
            # reassemble3, most of it those copies). Only the live `n * nv`
            # is zeroed.
            var n_r = nc * NT
            var jt_work = Scratch[Scalar[DTYPE], MJC_CAP](
                n_r * nv, uninitialized=ZERO
            )
            var wp = jt_work.unsafe_ptr()
            var n_live = n_r * nv
            comptime W = 2 * simd_width_of[DTYPE]()
            var q = 0
            while q + W <= n_live:
                wp.unsafe_store(q, SIMD[DTYPE, W](0))
                q += W
            while q < n_live:
                wp[unsafe_offset=q] = ZERO
                q += 1
            for c in range(nc):
                for t in range(nt_c[c]):
                    var r = c * NT + t
                    comptime if SPARSE:
                        for a in range(cn_n[c]):
                            var k = cn_ix[c * nv + a]
                            wp[unsafe_offset = k * n_r + r] = Jt_c[r * nv + k]
                    else:
                        for k in range(nv):
                            wp[unsafe_offset = k * n_r + r] = Jt_c[r * nv + k]
            _tree_solve_cols[DTYPE, V_CAP, MJC_CAP](
                env, nv, n_r, ldl_L, ldl_D, par, jt_work
            )
            var mp = minv_cache.unsafe_ptr()
            var wp2 = jt_work.unsafe_ptr()
            for r in range(n_r):
                for k in range(nv):
                    mp[unsafe_offset = r * nv + k] = wp2[unsafe_offset = k * n_r + r]
        else:
            for c in range(nc):
                for t in range(nt_c[c]):
                    _minv_dense[DTYPE, T_CAP, V_CAP, NT, SPARSE=SPARSE](
                        env, dims, m_inv, Jt_c, c * NT + t, MinvJ, t, cn_n, cn_ix, c
                    )
                    for q in range(nv):
                        minv_cache[(c * NT + t) * nv + q] = MinvJ[t * nv + q]

    # Each contact's `Ac` block (`extractBlock`, flg_subR=1: `J_t M⁻¹ J_uᵀ`
    # with the diagonal floored) is loop-invariant across the sweeps — it was
    # rebuilt from `nt²` sparse dots on every contact of every iteration.
    # Built once here under `CACHE`, straight from `minv_cache`.
    comptime AC_CAP = MC_CAP * NT * NT if CACHE else 1
    var ac_cache = Scratch[Scalar[DTYPE], AC_CAP](
        nc * NT * NT if CACHE else 1, uninitialized=ZERO
    )
    comptime if CACHE:
        for c in range(nc):
            var nt = nt_c[c]
            var cb = c * NT
            for t in range(nt):
                for u in range(nt):
                    var acc = ZERO
                    comptime if SPARSE:
                        for a in range(cn_n[c]):
                            var i = cn_ix[c * nv + a]
                            acc += Jt_c[(cb + t) * nv + i] * minv_cache[(cb + u) * nv + i]
                    else:
                        for i in range(nv):
                            acc += Jt_c[(cb + t) * nv + i] * minv_cache[(cb + u) * nv + i]
                    if t == u and acc < DIAG_FLOOR:
                        acc = DIAG_FLOOR
                    ac_cache[c * NT * NT + t * NT + u] = acc
    # Friction-dof rows read a COLUMN of `M⁻¹` (`J = sign · e_dof`); under
    # `TREE` those columns are one bulk tree solve of unit vectors.
    comptime SC_CAP = S_CAP * V_CAP if TREE else 1
    var col_cache = Scratch[Scalar[DTYPE], SC_CAP](
        ns * nv if TREE else 1, uninitialized=ZERO
    )
    comptime if TREE:
        var col_rows = Scratch[Scalar[DTYPE], SC_CAP](ns * nv, fill=ZERO)
        var col_work = Scratch[Scalar[DTYPE], SC_CAP](ns * nv, uninitialized=ZERO)
        for s_r in range(ns):
            if sr_kind[s_r] == SROW_FRICTION:
                col_rows[s_r * nv + sr_dof[s_r]] = ONE
        _minv_apply_rows[DTYPE, V_CAP, SC_CAP](
            env, nv, ns, m_inv, ldl_L, ldl_D, par, tree_ok,
            col_rows, col_cache, col_work,
        )
    # The `nt x nt` AR block, its rhs, the solved force and the old one.
    var Ac = Array[Scalar[DTYPE], NT * NT](fill=ZERO)
    var bc = Array[Scalar[DTYPE], NT](fill=ZERO)
    var vf = Array[Scalar[DTYPE], NT](fill=ZERO)
    var oldf = Array[Scalar[DTYPE], NT](fill=ZERO)
    # This contact's tangential residuals, recomputed at the point of use.
    var jt_cur = Array[Scalar[DTYPE], NT](fill=ZERO)

    for it in range(max_iter):
        var improvement = ZERO

        # `iter == 0` correction: MuJoCo folds in the R-weighted force energy
        # once so the first iteration's improvement is comparable with the
        # primal solver's own. `efc_D = 1/R`, so a row's R is the reciprocal
        # of the stiffness the primal solve used for it.
        if it == 0:
            for c in range(nc):
                var f_n = fn_a[c]
                if D_n_c[c] > ZERO:
                    improvement += HALF * f_n * f_n / D_n_c[c]
                for t in range(nt_c[c]):
                    var d = D_t_c[c * NT + t]
                    if d > ZERO:
                        var f = ft_a[c * NT + t]
                        improvement += HALF * f * f / d
            for s in range(ns):
                improvement += HALF * sr_f[s] * sr_f[s] * sr_R[s]
            for e in range(neq_rows):
                if eq_D[e] > ZERO:
                    improvement += HALF * eq_f[e] * eq_f[e] / eq_D[e]

        # ── sweep 1: dry-friction dof rows, box-clamped to +-frictionloss ──
        # J = sign * e_dof, so `M^-1 J^T` is a COLUMN of m_inv scaled by the
        # sign and `a_ii` is a single diagonal entry — no matvec.
        for s in range(ns):
            if sr_kind[s] != SROW_FRICTION:
                continue
            var dof = sr_dof[s]
            var sgn = sr_sign[s]
            var a_ii = ZERO
            comptime if TREE:
                a_ii = col_cache[s * nv + dof]
            else:
                a_ii = rebind[Scalar[DTYPE]](m_inv[env, dof * nv + dof])
            var arinv = ONE / (a_ii if a_ii > MINVAL else MINVAL)

            # ⚠ COMPUTED HERE, NOT READ FROM A REFRESHED ARRAY. This is
            # `residual()` for this row and nothing else: MuJoCo recomputes the
            # residual of the row it is about to touch, it does not refresh the
            # whole system after every write. For a scalar row that is O(1) —
            # `J = sign * e_dof`, so the dot with `qacc` is one element.
            #
            # Gauss-Seidel is PRESERVED, which is the property the old
            # docstring was protecting: the value read is still built from the
            # `qacc` that every earlier update in this sweep has already moved.
            var res = sr_bias[s] + sr_sign[s] * qacc[sr_dof[s]]
            var old = sr_f[s]
            var f = old - res * arinv
            var lim = sr_floss[s]
            if f < -lim:
                f = -lim
            elif f > lim:
                f = lim

            var d = f - old
            if d != ZERO:
                sr_f[s] = f
                comptime if TREE:
                    for k in range(nv):
                        qacc[k] += d * sgn * col_cache[s * nv + k]
                else:
                    for k in range(nv):
                        qacc[k] += d * sgn * rebind[Scalar[DTYPE]](
                            m_inv[env, k * nv + dof]
                        )
                # (the full-system refresh that used to sit here is gone —
                # every consumer now recomputes its own row on demand)
            # ⚠ `/ arinv`, matching MuJoCo literally. That is `a_ii` clamped
            # from below, and the reciprocal round-trip is a 1-ulp difference
            # from using `a_ii` directly — kept because this feeds the
            # termination test, where the cheapest way to diverge is to be
            # almost right.
            improvement -= HALF * d * d / arinv + d * res

        # ── sweep 2: contact friction, the whole tangential block at once ──
        for c in range(nc):
            var nt = nt_c[c]
            # A FRICTIONLESS contact has no friction rows: MuJoCo's loop
            # `for j=0; j<dim-1` is empty and `costChange(..., dim-1=0)` is 0.
            if nt <= 0:
                continue
            var cb = c * NT

            # MuJoCo's order, and the order matters: the block is extracted
            # and `bc` formed BEFORE the zero-normal guard, because a contact
            # whose normal force has collapsed may still be carrying tangential
            # force that has to be zeroed AND accounted for.
            # `residual()` for THIS contact's tangential rows, from the
            # current `qacc` — same expression and same accumulation order the
            # full refresh used, so the values are identical, but O(nt * nv)
            # for one contact instead of O(nrows * nv) for the whole system.
            for t in range(nt):
                oldf[t] = ft_a[cb + t]
                var jv = bt_c[cb + t]
                comptime if SPARSE:
                    for a in range(cn_n[c]):
                        var i = cn_ix[c * nv + a]
                        jv += Jt_c[(cb + t) * nv + i] * qacc[i]
                else:
                    for i in range(nv):
                        jv += Jt_c[(cb + t) * nv + i] * qacc[i]
                jt_cur[t] = jv
                comptime if not CACHE:
                    _minv_dense[DTYPE, T_CAP, V_CAP, NT, SPARSE=SPARSE](
                        env, dims, m_inv, Jt_c, cb + t, MinvJ, t, cn_n, cn_ix, c
                    )

            # `Ac` = the AR submatrix with R subtracted off the diagonal and
            # the diagonal floored (`extractBlock`, flg_subR=1) — so R is not
            # in it at all. Under `CACHE` it is read from `ac_cache`.
            comptime if CACHE:
                for t in range(nt):
                    for u in range(nt):
                        Ac[t * NT + u] = ac_cache[c * NT * NT + t * NT + u]
            else:
                for t in range(nt):
                    for u in range(nt):
                        Ac[t * NT + u] = _dot_dense[
                            DTYPE, T_CAP, V_CAP, NT, SPARSE
                        ](Jt_c, cb + t, MinvJ, u, nv, cn_n, cn_ix, c)
                    if Ac[t * NT + t] < DIAG_FLOOR:
                        Ac[t * NT + t] = DIAG_FLOOR

            # `bc = res - Ac * oldforce`
            for t in range(nt):
                var b = jt_cur[t]
                for u in range(nt):
                    b -= Ac[t * NT + u] * oldf[u]
                bc[t] = b

            for t in range(nt):
                vf[t] = ZERO
            var fn_v = fn_a[c]
            if fn_v >= MINVAL:
                var active = _solve_contact_qcqp[DTYPE, NT, T_CAP](
                    nt, cb, Ac, bc, fr_c, fn_v, vf
                )
                if active:
                    # `projectEllipsoid(..., feasible=0)` — ALWAYS scales to
                    # the boundary, even from inside. Not a clamp.
                    var sq = ZERO
                    for t in range(nt):
                        var mu = fr_c[cb + t]
                        if mu > ZERO:
                            sq += vf[t] * vf[t] / (mu * mu)
                    var scl = sqrt(
                        fn_v * fn_v / (sq if sq > MINVAL else MINVAL)
                    )
                    for t in range(nt):
                        vf[t] *= scl

            # `costChange`: `0.5 d^T Ac d + d . res`.
            #
            # ⚠ This is the cancellation-sensitive expression of the module —
            # `vf[t] - oldf[t]` is a small difference of comparable numbers,
            # then squared against `Ac`.
            var change = ZERO
            var any_change = False
            for t in range(nt):
                var dt = vf[t] - oldf[t]
                if dt != ZERO:
                    any_change = True
                var quad = ZERO
                for u in range(nt):
                    quad += Ac[t * NT + u] * (vf[u] - oldf[u])
                change += HALF * dt * quad + dt * jt_cur[t]
            if change > COST_REJECT:
                # `costChange` restores `force` and returns 0 — the update is
                # dropped, not merely uncounted.
                continue

            if any_change:
                for t in range(nt):
                    var dt = vf[t] - oldf[t]
                    ft_a[cb + t] = vf[t]
                    comptime if CACHE:
                        var r0 = (cb + t) * nv
                        for q in range(nv):
                            qacc[q] += dt * minv_cache[r0 + q]
                    else:
                        for q in range(nv):
                            qacc[q] += dt * MinvJ[t * nv + q]
                # (full-system refresh removed — see sweep 1)
            improvement -= change

        improvement *= scale
        if improvement < tolerance:
            break

    # ── dualFinish: the FORCE writeback only ───────────────────────────────
    # `qfrc_constraint = J^T f` over EVERY row, not restricted to the friction
    # ones. The normal, limit and equality forces are unchanged by the sweep
    # but are still part of `qfrc_constraint`; dropping them would silently
    # release every contact this pass just refined.
    for i in range(nv):
        qfrc[i] = ZERO
    for c in range(nc):
        var f_n = fn_a[c]
        var n_i = cn_n[c] if SPARSE else nv
        for a in range(n_i):
            var i = cn_ix[c * nv + a] if SPARSE else a
            var acc = Jn_c[c * nv + i] * f_n
            for t in range(nt_c[c]):
                acc += Jt_c[(c * NT + t) * nv + i] * ft_a[c * NT + t]
            qfrc[i] += acc
    for s in range(ns):
        qfrc[sr_dof[s]] += sr_sign[s] * sr_f[s]
    for e in range(neq_rows):
        var fe = eq_f[e]
        if fe == ZERO:
            continue
        for d in range(nv):
            qfrc[d] += eq_J[e * nv + d] * fe
    # ⚠⚠ AND THAT IS WHERE IT STOPS — THE ACCELERATION IS NOT REBUILT.
    #
    # MuJoCo's `dualFinish` continues `qacc = M^-1 qfrc + qacc_smooth`, and so
    # did this. At float64 that is free. At float32 it is the single largest
    # error in the pass, because it THROWS AWAY A CONVERGED ANSWER AND REBUILDS
    # IT: the Newton solve already produced `qacc`, and the sweeps above have
    # already applied their own force changes to it incrementally
    # (`qacc += dt * MinvJ`, plus the dry-friction column update). Rebuilding
    # re-derives the WHOLE acceleration from the WHOLE force set in order to
    # express a tiny correction — measured force changes across all sweeps peak
    # at 2.9e-04 N — and pays the full conditioning cost of `M` to do it. On
    # `reassemble_5` that matrix's diagonal spans 3.9e-06 to 4.34, a ratio of
    # 1.1e6.
    #
    # ⚠⚠ MEASURED, one substep from a settled tower: the rebuild MOVED `qacc`
    # by 0.0866 at float32 against 0.00037 at float64 — 3.4% of |qacc| against
    # 0.003%, a thousandfold worse in relative terms, and it fired every
    # substep. It also explains the otherwise baffling iteration curve, which
    # is what led here: `noslip_iterations=1` was CATASTROPHIC (144.75 J of
    # peak brick kinetic energy) while 0 was clean (1.42e-04) and the shipped 5
    # partly recovered (1.58e-04). The rebuild fires at any count >= 1; only
    # repeated sweeps drag the forces back toward something it reproduces.
    #
    # ⚠ THE WARNING ABOVE STILL STANDS FOR `qfrc` AND DOES NOT APPLY HERE. A
    # from-scratch reconstruction has to re-express the normal, limit and
    # equality forces itself, which is why `qfrc` spans every row. `qacc` does
    # not need to: it still CARRIES them, because it was never discarded.
    #
    # ⚠ `qacc_smooth` IS ALREADY IN `qacc` for the same reason — it entered
    # with the Newton solution. Adding it again here would double it.

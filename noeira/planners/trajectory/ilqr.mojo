"""Planner-side trajectory optimization for iLQR (iterative Linear Quadratic Regulator).

Two structs:

* ``ILQRCPU`` — host-side reference implementation. Drives the
  ``RolloutJacobianCallbackCPU`` contract: rolls out, linearizes the
  whole trajectory, runs a Riccati backward pass with
  Levenberg-Marquardt regularization, then a backtracking line search.

* ``ILQRGPUBatched`` — batched GPU implementation for ``N_ENVS``
  independent iLQR problems planned in parallel. Global LM ``μ`` and
  line-search ``α`` (host scalars) — simplest orchestration; per-env
  tracking is a future cleanup. The Riccati backward pass runs as
  one block per env (single thread, sequential ``t = T-1 → 0`` in
  registers). Callback-facing ``LayoutTensor`` views are built with
  the callback's comptime dims (``CB.LATENT_DIM`` /
  ``CB.ACTION_DIM``) via ``rebind`` to bridge Mojo's unfolded-
  comptime layout-type mismatch. Helper kernels live in
  ``ilqr_kernels.mojo`` (1-D flat ``LayoutTensor`` views with
  explicit offset math + ``rebind[Scalar[dtype]](...)`` reads).
  Validated on a 1-D LQ oracle: ``N_ENVS = 2`` envs both converge
  to the analytic Riccati gain to ~1e-4 on float32.

iLQR works in **cost-space** (minimization), not reward-space — see
``jacobian_callback.mojo`` for the contract. The planner is gradient-
*based* (needs Jacobians of dynamics + Hessian of cost), unlike MPPI
which is gradient-free; hence the separate
``RolloutJacobianCallback{CPU,GPU}`` extension traits.

Algorithm references:
  Tassa et al., "Synthesis and Stabilization of Complex Behaviors
  through Online Trajectory Optimization", IROS 2012.
  Li & Todorov, "Iterative Linear Quadratic Regulator Design for
  Nonlinear Biological Movement Systems", ICINCO 2004.

Levenberg-Marquardt scheme follows Tassa: add ``μ·I`` to ``Q_uu``
before inversion, decrease ``μ`` on a successful line search,
increase it on failure. Line search is plain backtracking on
``α ∈ {1, 1/2, 1/4, …}`` with simple cost-improvement acceptance
(no Armijo ratio test in v1).
"""

from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from noeira.nn.constants import DT as dtype

from .jacobian_callback import (
    RolloutJacobianCallbackCPU,
    RolloutJacobianCallbackGPU,
)
from .ilqr_kernels import (
    ilqr_copy_z0_kernel,
    ilqr_reduce_cost_kernel,
    ilqr_apply_control_update_kernel,
    ilqr_backward_pass_kernel,
    ilqr_accept_kernel,
)
from noeira.core.tiles import Tile, tile

# =============================================================================
# Module-level scalar matrix helpers (CPU, List[Float64] interface)
# =============================================================================
#
# All matrices are row-major and flat in a List[Float64]. Helpers keep
# the algorithm body readable while ACTION_DIM / LATENT_DIM stay
# comptime in the struct body.


@always_inline
def _matmul(
    A: List[Float64],
    B: List[Float64],
    mut C_out: List[Float64],
    m: Int,
    k: Int,
    n: Int,
    c_offset: Int,
    a_offset: Int,
    b_offset: Int,
):
    """C[m,n] = A[m,k] @ B[k,n]."""
    for i in range(m):
        for j in range(n):
            var s: Float64 = 0.0
            for r in range(k):
                s += A[a_offset + i * k + r] * B[b_offset + r * n + j]
            C_out[c_offset + i * n + j] = s


@always_inline
def _matmul_T_A(
    A: List[Float64],
    B: List[Float64],
    mut C_out: List[Float64],
    k: Int,
    m: Int,
    n: Int,
    c_offset: Int,
    a_offset: Int,
    b_offset: Int,
):
    """C[m,n] = A[k,m]^T @ B[k,n]."""
    for i in range(m):
        for j in range(n):
            var s: Float64 = 0.0
            for r in range(k):
                s += A[a_offset + r * m + i] * B[b_offset + r * n + j]
            C_out[c_offset + i * n + j] = s


@always_inline
def _matvec(
    A: List[Float64],
    x: List[Float64],
    mut y_out: List[Float64],
    m: Int,
    n: Int,
    a_offset: Int,
    x_offset: Int,
    y_offset: Int,
):
    """y[m] = A[m,n] @ x[n]."""
    for i in range(m):
        var s: Float64 = 0.0
        for j in range(n):
            s += A[a_offset + i * n + j] * x[x_offset + j]
        y_out[y_offset + i] = s


@always_inline
def _matvec_T(
    A: List[Float64],
    x: List[Float64],
    mut y_out: List[Float64],
    m: Int,
    n: Int,
    a_offset: Int,
    x_offset: Int,
    y_offset: Int,
):
    """y[n] = A[m,n]^T @ x[m]."""
    for j in range(n):
        var s: Float64 = 0.0
        for i in range(m):
            s += A[a_offset + i * n + j] * x[x_offset + i]
        y_out[y_offset + j] = s


def _cholesky_solve(
    mut M: List[Float64],
    mut rhs: List[Float64],
    n: Int,
    nrhs: Int,
) raises -> Bool:
    """Solve ``M[n,n] @ X[n,nrhs] = rhs[n,nrhs]`` in place via LDL.

    On success ``rhs`` holds ``X`` and the routine returns ``True``.
    On failure (non-PD pivot) returns ``False`` — caller bumps ``μ``.

    ``M`` is destroyed (used as scratch for the LDL factors). Layout:
    row-major flat list.
    """
    var diag_eps: Float64 = 1e-12
    for j in range(n):
        var d = M[j * n + j]
        for k in range(j):
            d -= M[j * n + k] * M[j * n + k] * M[k * n + k]
        if d <= diag_eps:
            return False
        M[j * n + j] = d
        for i in range(j + 1, n):
            var s = M[i * n + j]
            for k in range(j):
                s -= M[i * n + k] * M[j * n + k] * M[k * n + k]
            M[i * n + j] = s / d

    # Solve L y = rhs (forward sub), D^{-1}, L^T x = y (back sub).
    for c in range(nrhs):
        for i in range(n):
            var s = rhs[i * nrhs + c]
            for k in range(i):
                s -= M[i * n + k] * rhs[k * nrhs + c]
            rhs[i * nrhs + c] = s
        for i in range(n):
            rhs[i * nrhs + c] /= M[i * n + i]
        for i in range(n - 1, -1, -1):
            var s = rhs[i * nrhs + c]
            for k in range(i + 1, n):
                s -= M[k * n + i] * rhs[k * nrhs + c]
            rhs[i * nrhs + c] = s
    return True


@always_inline
def _maxf(a: Float64, b: Float64) -> Float64:
    return a if a > b else b


# =============================================================================
# ILQRCPU
# =============================================================================


struct ILQRCPU[
    LATENT_DIM: Int,
    ACTION_DIM: Int,
    HORIZON: Int,
](Deinitable, Movable):
    """CPU iLQR planner.

    Plans an open-loop ``HORIZON``-step control sequence for a single
    initial state. Comptime-parametric on
    ``(LATENT_DIM, ACTION_DIM, HORIZON)``; runtime hyperparameters
    (``n_iters``, line-search depth, LM bounds, convergence tol) are
    ``__init__`` args so different problems can reuse the same comptime
    instantiation.
    """

    comptime _T = Self.HORIZON
    comptime _L = Self.LATENT_DIM
    comptime _A = Self.ACTION_DIM
    comptime _LL = Self.LATENT_DIM * Self.LATENT_DIM
    comptime _LA = Self.LATENT_DIM * Self.ACTION_DIM
    comptime _AA = Self.ACTION_DIM * Self.ACTION_DIM

    var n_iters: Int
    var n_line_search: Int
    var mu_init: Float64
    var mu_min: Float64
    var mu_max: Float64
    var mu_factor: Float64
    var cost_tol: Float64

    var U: List[Float64]
    """`(HORIZON, ACTION_DIM)` — current control sequence."""
    var z_seq: List[Float64]
    """`(HORIZON+1, LATENT_DIM)` — current state trajectory."""

    var A_seq: List[Float64]
    var B_seq: List[Float64]
    var l_z_seq: List[Float64]
    var l_u_seq: List[Float64]
    var l_zz_seq: List[Float64]
    var l_uu_seq: List[Float64]
    var l_zu_seq: List[Float64]

    var K_seq: List[Float64]
    var k_seq: List[Float64]

    var V_z: List[Float64]
    var V_zz: List[Float64]
    var Q_z: List[Float64]
    var Q_u: List[Float64]
    var Q_zz: List[Float64]
    var Q_uu: List[Float64]
    var Q_zu: List[Float64]
    var _tmp_LL: List[Float64]
    var _tmp_LA: List[Float64]
    var _tmp_AL: List[Float64]
    var _quu_solve: List[Float64]
    var _rhs_solve: List[Float64]

    var U_trial: List[Float64]
    var z_trial: List[Float64]

    var current_cost: Float64

    def __init__(
        out self,
        n_iters: Int = 10,
        n_line_search: Int = 10,
        mu_init: Float64 = 1.0,
        mu_min: Float64 = 1e-6,
        mu_max: Float64 = 1e10,
        mu_factor: Float64 = 2.0,
        cost_tol: Float64 = 1e-6,
    ) raises:
        if Self.HORIZON < 1:
            raise Error("ILQRCPU: HORIZON must be >= 1")
        if Self.LATENT_DIM < 1 or Self.ACTION_DIM < 1:
            raise Error("ILQRCPU: LATENT_DIM and ACTION_DIM must be >= 1")
        if n_iters < 1:
            raise Error("ILQRCPU: n_iters must be >= 1")
        if mu_factor <= 1.0:
            raise Error("ILQRCPU: mu_factor must be > 1.0")

        self.n_iters = n_iters
        self.n_line_search = n_line_search
        self.mu_init = mu_init
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.mu_factor = mu_factor
        self.cost_tol = cost_tol

        self.U = List[Float64](length=Self._T * Self._A, fill=0.0)
        self.z_seq = List[Float64](length=(Self._T + 1) * Self._L, fill=0.0)
        self.A_seq = List[Float64](length=Self._T * Self._LL, fill=0.0)
        self.B_seq = List[Float64](length=Self._T * Self._LA, fill=0.0)
        self.l_z_seq = List[Float64](length=Self._T * Self._L, fill=0.0)
        self.l_u_seq = List[Float64](length=Self._T * Self._A, fill=0.0)
        self.l_zz_seq = List[Float64](length=Self._T * Self._LL, fill=0.0)
        self.l_uu_seq = List[Float64](length=Self._T * Self._AA, fill=0.0)
        self.l_zu_seq = List[Float64](length=Self._T * Self._LA, fill=0.0)

        self.K_seq = List[Float64](length=Self._T * Self._LA, fill=0.0)
        self.k_seq = List[Float64](length=Self._T * Self._A, fill=0.0)

        self.V_z = List[Float64](length=Self._L, fill=0.0)
        self.V_zz = List[Float64](length=Self._LL, fill=0.0)
        self.Q_z = List[Float64](length=Self._L, fill=0.0)
        self.Q_u = List[Float64](length=Self._A, fill=0.0)
        self.Q_zz = List[Float64](length=Self._LL, fill=0.0)
        self.Q_uu = List[Float64](length=Self._AA, fill=0.0)
        self.Q_zu = List[Float64](length=Self._LA, fill=0.0)
        self._tmp_LL = List[Float64](length=Self._LL, fill=0.0)
        self._tmp_LA = List[Float64](length=Self._LA, fill=0.0)
        self._tmp_AL = List[Float64](length=Self._LA, fill=0.0)
        self._quu_solve = List[Float64](length=Self._AA, fill=0.0)
        self._rhs_solve = List[Float64](
            length=Self._A * (1 + Self._L), fill=0.0
        )

        self.U_trial = List[Float64](length=Self._T * Self._A, fill=0.0)
        self.z_trial = List[Float64](length=(Self._T + 1) * Self._L, fill=0.0)

        self.current_cost = 0.0

    def reset_warm_start(mut self):
        """Zero the control sequence."""
        for i in range(Self._T * Self._A):
            self.U[i] = 0.0

    def plan[
        CB: RolloutJacobianCallbackCPU
    ](
        mut self,
        mut callback: CB,
        z0: List[Float64],
        warm_start: Bool = False,
    ) raises -> Float64:
        """Run iLQR from initial state ``z0`` and return the final cost.

        On return, the optimized control sequence is in ``self.U`` as a
        flat ``(HORIZON, ACTION_DIM)`` row-major list. ``warm_start``
        toggles whether to reuse the previous solution.
        """
        if len(z0) != Self._L:
            raise Error("ILQRCPU.plan: z0 length must equal LATENT_DIM")

        if not warm_start:
            for i in range(Self._T * Self._A):
                self.U[i] = 0.0

        self.current_cost = self._rollout(callback, z0)

        var mu = self.mu_init

        for _outer in range(self.n_iters):
            self._linearize_all(callback)

            var bw_ok = self._backward_pass(mu)
            while not bw_ok:
                mu *= self.mu_factor
                if mu > self.mu_max:
                    return self.current_cost
                bw_ok = self._backward_pass(mu)

            var accepted = False
            var alpha: Float64 = 1.0
            var new_cost: Float64 = self.current_cost
            for _ls in range(self.n_line_search):
                new_cost = self._forward_with_gains(callback, z0, alpha)
                if new_cost < self.current_cost:
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted:
                mu *= self.mu_factor
                if mu > self.mu_max:
                    break
                continue

            var cost_change = self.current_cost - new_cost
            self._accept_trial(new_cost)
            mu = _maxf(mu / self.mu_factor, self.mu_min)
            if cost_change < self.cost_tol:
                break

        return self.current_cost

    # ────────────────────────────────────────────────────────────────

    def _rollout[
        CB: RolloutJacobianCallbackCPU
    ](mut self, mut callback: CB, z0: List[Float64],) raises -> Float64:
        for d in range(Self._L):
            self.z_seq[d] = z0[d]

        var z_step = List[Float64](length=Self._L, fill=0.0)
        var u_step = List[Float64](length=Self._A, fill=0.0)
        var z_next = List[Float64](length=Self._L, fill=0.0)

        var total: Float64 = 0.0
        for t in range(Self._T):
            for d in range(Self._L):
                z_step[d] = self.z_seq[t * Self._L + d]
            for d in range(Self._A):
                u_step[d] = self.U[t * Self._A + d]
            var c = callback.step_cpu(z_step, u_step, z_next)
            total += c
            for d in range(Self._L):
                self.z_seq[(t + 1) * Self._L + d] = z_next[d]

        for d in range(Self._L):
            z_step[d] = self.z_seq[Self._T * Self._L + d]
        var V_z_scratch = List[Float64](length=Self._L, fill=0.0)
        var V_zz_scratch = List[Float64](length=Self._LL, fill=0.0)
        total += callback.terminal_cpu(z_step, V_z_scratch, V_zz_scratch)
        return total

    def _linearize_all[
        CB: RolloutJacobianCallbackCPU
    ](mut self, mut callback: CB,) raises:
        var z_step = List[Float64](length=Self._L, fill=0.0)
        var u_step = List[Float64](length=Self._A, fill=0.0)
        var A_buf = List[Float64](length=Self._LL, fill=0.0)
        var B_buf = List[Float64](length=Self._LA, fill=0.0)
        var lz_buf = List[Float64](length=Self._L, fill=0.0)
        var lu_buf = List[Float64](length=Self._A, fill=0.0)
        var lzz_buf = List[Float64](length=Self._LL, fill=0.0)
        var luu_buf = List[Float64](length=Self._AA, fill=0.0)
        var lzu_buf = List[Float64](length=Self._LA, fill=0.0)

        for t in range(Self._T):
            for d in range(Self._L):
                z_step[d] = self.z_seq[t * Self._L + d]
            for d in range(Self._A):
                u_step[d] = self.U[t * Self._A + d]
            callback.linearize_cpu(
                z_step,
                u_step,
                A_buf,
                B_buf,
                lz_buf,
                lu_buf,
                lzz_buf,
                luu_buf,
                lzu_buf,
            )
            for d in range(Self._LL):
                self.A_seq[t * Self._LL + d] = A_buf[d]
            for d in range(Self._LA):
                self.B_seq[t * Self._LA + d] = B_buf[d]
            for d in range(Self._L):
                self.l_z_seq[t * Self._L + d] = lz_buf[d]
            for d in range(Self._A):
                self.l_u_seq[t * Self._A + d] = lu_buf[d]
            for d in range(Self._LL):
                self.l_zz_seq[t * Self._LL + d] = lzz_buf[d]
            for d in range(Self._AA):
                self.l_uu_seq[t * Self._AA + d] = luu_buf[d]
            for d in range(Self._LA):
                self.l_zu_seq[t * Self._LA + d] = lzu_buf[d]

        for d in range(Self._L):
            z_step[d] = self.z_seq[Self._T * Self._L + d]
        _ = callback.terminal_cpu(z_step, self.V_z, self.V_zz)

    def _backward_pass(mut self, mu: Float64) raises -> Bool:
        for t_rev in range(Self._T):
            var t = Self._T - 1 - t_rev

            # Q_z = l_z + A^T V_z
            _matvec_T(
                self.A_seq,
                self.V_z,
                self.Q_z,
                Self._L,
                Self._L,
                t * Self._LL,
                0,
                0,
            )
            for d in range(Self._L):
                self.Q_z[d] += self.l_z_seq[t * Self._L + d]

            # Q_u = l_u + B^T V_z
            _matvec_T(
                self.B_seq,
                self.V_z,
                self.Q_u,
                Self._L,
                Self._A,
                t * Self._LA,
                0,
                0,
            )
            for d in range(Self._A):
                self.Q_u[d] += self.l_u_seq[t * Self._A + d]

            # tmp_LL = V_zz @ A
            _matmul(
                self.V_zz,
                self.A_seq,
                self._tmp_LL,
                Self._L,
                Self._L,
                Self._L,
                0,
                0,
                t * Self._LL,
            )
            # Q_zz = l_zz + A^T (V_zz A)
            _matmul_T_A(
                self.A_seq,
                self._tmp_LL,
                self.Q_zz,
                Self._L,
                Self._L,
                Self._L,
                0,
                t * Self._LL,
                0,
            )
            for d in range(Self._LL):
                self.Q_zz[d] += self.l_zz_seq[t * Self._LL + d]

            # tmp_LA = V_zz @ B
            _matmul(
                self.V_zz,
                self.B_seq,
                self._tmp_LA,
                Self._L,
                Self._L,
                Self._A,
                0,
                0,
                t * Self._LA,
            )
            # Q_uu = l_uu + B^T (V_zz B)
            _matmul_T_A(
                self.B_seq,
                self._tmp_LA,
                self.Q_uu,
                Self._L,
                Self._A,
                Self._A,
                0,
                t * Self._LA,
                0,
            )
            for d in range(Self._AA):
                self.Q_uu[d] += self.l_uu_seq[t * Self._AA + d]

            # Q_zu = l_zu + A^T (V_zz B)
            _matmul_T_A(
                self.A_seq,
                self._tmp_LA,
                self.Q_zu,
                Self._L,
                Self._L,
                Self._A,
                0,
                t * Self._LL,
                0,
            )
            for d in range(Self._LA):
                self.Q_zu[d] += self.l_zu_seq[t * Self._LA + d]

            # Solve (Q_uu + muI) [k | K_row] = [-Q_u | -Q_uz]
            for i in range(Self._A):
                for j in range(Self._A):
                    self._quu_solve[i * Self._A + j] = self.Q_uu[
                        i * Self._A + j
                    ]
                self._quu_solve[i * Self._A + i] += mu

            for i in range(Self._A):
                self._rhs_solve[i * (1 + Self._L) + 0] = -self.Q_u[i]
                for j in range(Self._L):
                    self._rhs_solve[i * (1 + Self._L) + 1 + j] = -self.Q_zu[
                        j * Self._A + i
                    ]

            var ok = _cholesky_solve(
                self._quu_solve, self._rhs_solve, Self._A, 1 + Self._L
            )
            if not ok:
                return False

            for i in range(Self._A):
                self.k_seq[t * Self._A + i] = self._rhs_solve[
                    i * (1 + Self._L) + 0
                ]
                for j in range(Self._L):
                    self.K_seq[
                        t * Self._LA + i * Self._L + j
                    ] = self._rhs_solve[i * (1 + Self._L) + 1 + j]

            # V_z, V_zz update (Tassa Eqs. 11–12)
            _matmul(
                self.Q_uu,
                self.K_seq,
                self._tmp_AL,
                Self._A,
                Self._A,
                Self._L,
                0,
                0,
                t * Self._LA,
            )

            var term1 = List[Float64](length=Self._L, fill=0.0)
            _matvec_T(
                self.K_seq,
                self.Q_u,
                term1,
                Self._A,
                Self._L,
                t * Self._LA,
                0,
                0,
            )
            var Quu_k = List[Float64](length=Self._A, fill=0.0)
            _matvec(
                self.Q_uu,
                self.k_seq,
                Quu_k,
                Self._A,
                Self._A,
                0,
                t * Self._A,
                0,
            )
            var term2 = List[Float64](length=Self._L, fill=0.0)
            _matvec_T(
                self.K_seq,
                Quu_k,
                term2,
                Self._A,
                Self._L,
                t * Self._LA,
                0,
                0,
            )
            var term3 = List[Float64](length=Self._L, fill=0.0)
            _matvec(
                self.Q_zu,
                self.k_seq,
                term3,
                Self._L,
                Self._A,
                0,
                t * Self._A,
                0,
            )
            for d in range(Self._L):
                self.V_z[d] = self.Q_z[d] + term1[d] + term2[d] + term3[d]

            _matmul_T_A(
                self.K_seq,
                self._tmp_AL,
                self._tmp_LL,
                Self._A,
                Self._L,
                Self._L,
                0,
                t * Self._LA,
                0,
            )
            var cross = List[Float64](length=Self._LL, fill=0.0)
            _matmul(
                self.Q_zu,
                self.K_seq,
                cross,
                Self._L,
                Self._A,
                Self._L,
                0,
                0,
                t * Self._LA,
            )
            for i in range(Self._L):
                for j in range(Self._L):
                    self.V_zz[i * Self._L + j] = (
                        self.Q_zz[i * Self._L + j]
                        + self._tmp_LL[i * Self._L + j]
                        + cross[i * Self._L + j]
                        + cross[j * Self._L + i]
                    )

        return True

    def _forward_with_gains[
        CB: RolloutJacobianCallbackCPU
    ](
        mut self,
        mut callback: CB,
        z0: List[Float64],
        alpha: Float64,
    ) raises -> Float64:
        for d in range(Self._L):
            self.z_trial[d] = z0[d]

        var z_step = List[Float64](length=Self._L, fill=0.0)
        var u_step = List[Float64](length=Self._A, fill=0.0)
        var z_next = List[Float64](length=Self._L, fill=0.0)

        var total: Float64 = 0.0
        for t in range(Self._T):
            for d in range(Self._L):
                z_step[d] = self.z_trial[t * Self._L + d]
            for i in range(Self._A):
                var ff = self.k_seq[t * Self._A + i]
                var fb: Float64 = 0.0
                for j in range(Self._L):
                    fb += self.K_seq[t * Self._LA + i * Self._L + j] * (
                        self.z_trial[t * Self._L + j]
                        - self.z_seq[t * Self._L + j]
                    )
                u_step[i] = self.U[t * Self._A + i] + alpha * ff + fb
                self.U_trial[t * Self._A + i] = u_step[i]
            var c = callback.step_cpu(z_step, u_step, z_next)
            total += c
            for d in range(Self._L):
                self.z_trial[(t + 1) * Self._L + d] = z_next[d]

        for d in range(Self._L):
            z_step[d] = self.z_trial[Self._T * Self._L + d]
        var V_z_scratch = List[Float64](length=Self._L, fill=0.0)
        var V_zz_scratch = List[Float64](length=Self._LL, fill=0.0)
        total += callback.terminal_cpu(z_step, V_z_scratch, V_zz_scratch)
        return total

    def _accept_trial(mut self, new_cost: Float64):
        for i in range(Self._T * Self._A):
            self.U[i] = self.U_trial[i]
        for i in range((Self._T + 1) * Self._L):
            self.z_seq[i] = self.z_trial[i]
        self.current_cost = new_cost


# =============================================================================
# ILQRGPUBatched
# =============================================================================


struct ILQRGPUBatched[
    LATENT_DIM: Int,
    ACTION_DIM: Int,
    HORIZON: Int,
    N_ENVS: Int,
](Deinitable, Movable):
    """Batched GPU iLQR — ``N_ENVS`` independent problems planned in
    parallel.

    Buffer layout: timestep-major. ``z_seq_buf`` is ``[T+1, N, L]``;
    every per-step buffer is ``[T, N, …]``. Per-step kernel launches
    see contiguous ``(N, …)`` slices via offset ``LayoutTensor``
    views.

    Algorithm choices for v1:
      * Global LM ``μ`` and line-search ``α`` (host scalars) —
        simplest host orchestration. Per-env α tracking is a future
        cleanup.
      * Backward pass: one block per env (parallel over envs, single
        thread inside the block runs the sequential ``T → 0`` Riccati).
        ``ACTION_DIM`` is assumed small (≤ 8) so the in-block LDL
        fits in registers.
      * Line search: try ``α`` ∈ ``{1, 1/2, …}``; accept globally if
        sum-of-per-env-costs decreases.
    """

    comptime _T = Self.HORIZON
    comptime _L = Self.LATENT_DIM
    comptime _A = Self.ACTION_DIM
    comptime _N = Self.N_ENVS
    comptime _LL = Self.LATENT_DIM * Self.LATENT_DIM
    comptime _LA = Self.LATENT_DIM * Self.ACTION_DIM
    comptime _AA = Self.ACTION_DIM * Self.ACTION_DIM

    var n_iters: Int
    var n_line_search: Int
    var mu_init: Float64
    var mu_min: Float64
    var mu_max: Float64
    var mu_factor: Float64
    var cost_tol: Float64

    var U_buf: DeviceBuffer[dtype]
    var z_seq_buf: DeviceBuffer[dtype]
    var step_cost_buf: DeviceBuffer[dtype]
    var term_cost_buf: DeviceBuffer[dtype]
    var total_cost_buf: DeviceBuffer[dtype]

    var A_seq_buf: DeviceBuffer[dtype]
    var B_seq_buf: DeviceBuffer[dtype]
    var l_z_seq_buf: DeviceBuffer[dtype]
    var l_u_seq_buf: DeviceBuffer[dtype]
    var l_zz_seq_buf: DeviceBuffer[dtype]
    var l_uu_seq_buf: DeviceBuffer[dtype]
    var l_zu_seq_buf: DeviceBuffer[dtype]
    var V_z_term_buf: DeviceBuffer[dtype]
    var V_zz_term_buf: DeviceBuffer[dtype]

    var K_seq_buf: DeviceBuffer[dtype]
    var k_seq_buf: DeviceBuffer[dtype]
    var bw_ok_buf: DeviceBuffer[DType.int32]

    var U_trial_buf: DeviceBuffer[dtype]
    var z_trial_buf: DeviceBuffer[dtype]
    var trial_cost_buf: DeviceBuffer[dtype]

    var bw_ok_host: HostBuffer[DType.int32]
    var trial_cost_host: HostBuffer[dtype]
    var total_cost_host: HostBuffer[dtype]
    var zeros_U_host: HostBuffer[dtype]

    def __init__(
        out self,
        ctx: DeviceContext,
        n_iters: Int = 10,
        n_line_search: Int = 10,
        mu_init: Float64 = 1.0,
        mu_min: Float64 = 1e-6,
        mu_max: Float64 = 1e10,
        mu_factor: Float64 = 2.0,
        cost_tol: Float64 = 1e-6,
    ) raises:
        if Self._T < 1 or Self._L < 1 or Self._A < 1 or Self._N < 1:
            raise Error(
                "ILQRGPUBatched: HORIZON / LATENT / ACTION / N_ENVS must be"
                " >= 1"
            )
        if n_iters < 1:
            raise Error("ILQRGPUBatched: n_iters must be >= 1")
        if mu_factor <= 1.0:
            raise Error("ILQRGPUBatched: mu_factor must be > 1.0")

        self.n_iters = n_iters
        self.n_line_search = n_line_search
        self.mu_init = mu_init
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.mu_factor = mu_factor
        self.cost_tol = cost_tol

        self.U_buf = ctx.enqueue_create_buffer[dtype](
            Self._N * Self._T * Self._A
        )
        self.z_seq_buf = ctx.enqueue_create_buffer[dtype](
            Self._N * (Self._T + 1) * Self._L
        )
        self.step_cost_buf = ctx.enqueue_create_buffer[dtype](Self._N * Self._T)
        self.term_cost_buf = ctx.enqueue_create_buffer[dtype](Self._N)
        self.total_cost_buf = ctx.enqueue_create_buffer[dtype](Self._N)

        self.A_seq_buf = ctx.enqueue_create_buffer[dtype](
            Self._N * Self._T * Self._LL
        )
        self.B_seq_buf = ctx.enqueue_create_buffer[dtype](
            Self._N * Self._T * Self._LA
        )
        self.l_z_seq_buf = ctx.enqueue_create_buffer[dtype](
            Self._N * Self._T * Self._L
        )
        self.l_u_seq_buf = ctx.enqueue_create_buffer[dtype](
            Self._N * Self._T * Self._A
        )
        self.l_zz_seq_buf = ctx.enqueue_create_buffer[dtype](
            Self._N * Self._T * Self._LL
        )
        self.l_uu_seq_buf = ctx.enqueue_create_buffer[dtype](
            Self._N * Self._T * Self._AA
        )
        self.l_zu_seq_buf = ctx.enqueue_create_buffer[dtype](
            Self._N * Self._T * Self._LA
        )
        self.V_z_term_buf = ctx.enqueue_create_buffer[dtype](Self._N * Self._L)
        self.V_zz_term_buf = ctx.enqueue_create_buffer[dtype](
            Self._N * Self._LL
        )

        self.K_seq_buf = ctx.enqueue_create_buffer[dtype](
            Self._N * Self._T * Self._LA
        )
        self.k_seq_buf = ctx.enqueue_create_buffer[dtype](
            Self._N * Self._T * Self._A
        )
        self.bw_ok_buf = ctx.enqueue_create_buffer[DType.int32](Self._N)

        self.U_trial_buf = ctx.enqueue_create_buffer[dtype](
            Self._N * Self._T * Self._A
        )
        self.z_trial_buf = ctx.enqueue_create_buffer[dtype](
            Self._N * (Self._T + 1) * Self._L
        )
        self.trial_cost_buf = ctx.enqueue_create_buffer[dtype](Self._N)

        self.bw_ok_host = ctx.enqueue_create_host_buffer[DType.int32](Self._N)
        self.trial_cost_host = ctx.enqueue_create_host_buffer[dtype](Self._N)
        self.total_cost_host = ctx.enqueue_create_host_buffer[dtype](Self._N)
        self.zeros_U_host = ctx.enqueue_create_host_buffer[dtype](
            Self._N * Self._T * Self._A
        )
        for i in range(Self._N * Self._T * Self._A):
            self.zeros_U_host[i] = 0.0

        ctx.synchronize()

    def plan_gpu[
        CB: RolloutJacobianCallbackGPU
    ](
        mut self,
        ctx: DeviceContext,
        mut callback: CB,
        z0: Tile[dtype, Self._N, Self._L],
        warm_start: Bool = False,
    ) raises:
        """Run batched iLQR with global μ / α schedule.

        After return, ``self.U_buf`` holds optimized controls
        ``(N_ENVS, HORIZON, ACTION_DIM)`` and ``self.total_cost_buf``
        holds per-env final costs.
        """
        if not warm_start:
            ctx.enqueue_copy(self.U_buf, self.zeros_U_host)

        # Initial forward rollout (writes z_seq, step_cost, term_cost).
        self._copy_z0(ctx, z0)
        self._forward_rollout_initial(ctx, callback)
        self._reduce_total_cost(ctx)
        ctx.enqueue_copy(self.total_cost_host, self.total_cost_buf)
        ctx.synchronize()

        var current_total: Float64 = 0.0
        for i in range(Self._N):
            current_total += Float64(self.total_cost_host[i])

        var mu = self.mu_init

        for _outer in range(self.n_iters):
            self._linearize_all(ctx, callback)

            var bw_done = False
            for _bw_retry in range(64):
                self._backward_pass(ctx, mu)
                ctx.enqueue_copy(self.bw_ok_host, self.bw_ok_buf)
                ctx.synchronize()
                var all_ok = True
                for i in range(Self._N):
                    if Int(self.bw_ok_host[i]) == 0:
                        all_ok = False
                if all_ok:
                    bw_done = True
                    break
                mu *= self.mu_factor
                if mu > self.mu_max:
                    break
            if not bw_done:
                return

            var accepted = False
            var alpha: Float64 = 1.0
            var new_total: Float64 = current_total
            for _ls in range(self.n_line_search):
                self._copy_z0_trial(ctx, z0)
                self._forward_with_gains(ctx, callback, alpha)
                self._reduce_trial_cost(ctx)
                ctx.enqueue_copy(self.trial_cost_host, self.trial_cost_buf)
                ctx.synchronize()
                new_total = 0.0
                for i in range(Self._N):
                    new_total += Float64(self.trial_cost_host[i])
                if new_total < current_total:
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted:
                mu *= self.mu_factor
                if mu > self.mu_max:
                    break
                continue

            self._accept_all(ctx)
            var cost_change = current_total - new_total
            current_total = new_total
            mu = _maxf(mu / self.mu_factor, self.mu_min)
            if cost_change < self.cost_tol:
                break

    # ────────────────────────────────────────────────────────────────

    def _copy_z0(
        mut self,
        ctx: DeviceContext,
        z0: Tile[dtype, Self._N, Self._L],
    ) raises:
        var z_seq_view = tile[Self._N * Self._L](self.z_seq_buf)
        ctx.enqueue_function[ilqr_copy_z0_kernel[dtype, Self._N, Self._L]](
            z0,
            z_seq_view,
            grid_dim=Self._N,
            block_dim=Self._L,
        )

    def _copy_z0_trial(
        mut self,
        ctx: DeviceContext,
        z0: Tile[dtype, Self._N, Self._L],
    ) raises:
        var z_trial_view = tile[Self._N * Self._L](self.z_trial_buf)
        ctx.enqueue_function[ilqr_copy_z0_kernel[dtype, Self._N, Self._L]](
            z0,
            z_trial_view,
            grid_dim=Self._N,
            block_dim=Self._L,
        )

    def _forward_rollout_initial[
        CB: RolloutJacobianCallbackGPU
    ](mut self, ctx: DeviceContext, mut callback: CB,) raises:
        """Roll out ``z_seq[t+1] = f(z_seq[t], U[t])``, accumulate
        ``step_cost``, then terminal expansion on ``z_seq[T]``.
        """
        for t in range(Self._T):
            var z_t = rebind[
                Tile[dtype, Self._N, CB.LATENT_DIM]
            ](
                tile[Self._N, Self._L](self.z_seq_buf.create_sub_buffer[dtype]( t * Self._N * Self._L, Self._N * Self._L ))
            )
            var u_t = rebind[
                Tile[dtype, Self._N, CB.ACTION_DIM]
            ](
                tile[Self._N, Self._A](self.U_buf.create_sub_buffer[dtype]( t * Self._N * Self._A, Self._N * Self._A ))
            )
            var z_next = rebind[
                Tile[dtype, Self._N, CB.LATENT_DIM]
            ](
                tile[Self._N, Self._L](self.z_seq_buf.create_sub_buffer[dtype]( (t + 1) * Self._N * Self._L, Self._N * Self._L ))
            )
            var c_t = rebind[
                Tile[dtype, Self._N]
            ](
                tile[Self._N](self.step_cost_buf.create_sub_buffer[dtype]( t * Self._N, Self._N ))
            )
            callback.step_gpu[Self._N](ctx, z_t, u_t, z_next, c_t)

        self._terminal_on_z_seq(ctx, callback)

    def _terminal_on_z_seq[
        CB: RolloutJacobianCallbackGPU
    ](mut self, ctx: DeviceContext, mut callback: CB,) raises:
        var z_T = rebind[
            Tile[dtype, Self._N, CB.LATENT_DIM]
        ](
            tile[Self._N, Self._L](self.z_seq_buf.create_sub_buffer[dtype]( Self._T * Self._N * Self._L, Self._N * Self._L ))
        )
        var Vz = rebind[
            Tile[dtype, Self._N, CB.LATENT_DIM]
        ](
            tile[Self._N, Self._L](self.V_z_term_buf)
        )
        var Vzz = rebind[
            Tile[dtype, Self._N, CB.LATENT_DIM, CB.LATENT_DIM]
        ](
            tile[Self._N, Self._L, Self._L](self.V_zz_term_buf)
        )
        var term = tile[Self._N](self.term_cost_buf)
        callback.terminal_gpu[Self._N](ctx, z_T, Vz, Vzz, term)

    def _terminal_on_z_trial[
        CB: RolloutJacobianCallbackGPU
    ](mut self, ctx: DeviceContext, mut callback: CB,) raises:
        var z_T = rebind[
            Tile[dtype, Self._N, CB.LATENT_DIM]
        ](
            tile[Self._N, Self._L](self.z_trial_buf.create_sub_buffer[dtype]( Self._T * Self._N * Self._L, Self._N * Self._L ))
        )
        var Vz = rebind[
            Tile[dtype, Self._N, CB.LATENT_DIM]
        ](
            tile[Self._N, Self._L](self.V_z_term_buf)
        )
        var Vzz = rebind[
            Tile[dtype, Self._N, CB.LATENT_DIM, CB.LATENT_DIM]
        ](
            tile[Self._N, Self._L, Self._L](self.V_zz_term_buf)
        )
        var term = tile[Self._N](self.term_cost_buf)
        callback.terminal_gpu[Self._N](ctx, z_T, Vz, Vzz, term)

    def _reduce_total_cost(mut self, ctx: DeviceContext) raises:
        var step_view = tile[Self._T * Self._N](self.step_cost_buf)
        var term_view = tile[Self._N](self.term_cost_buf)
        var out_view = tile[Self._N](self.total_cost_buf)
        ctx.enqueue_function[ilqr_reduce_cost_kernel[dtype, Self._T, Self._N]](
            step_view,
            term_view,
            out_view,
            grid_dim=1,
            block_dim=Self._N,
        )

    def _reduce_trial_cost(mut self, ctx: DeviceContext) raises:
        var step_view = tile[Self._T * Self._N](self.step_cost_buf)
        var term_view = tile[Self._N](self.term_cost_buf)
        var out_view = tile[Self._N](self.trial_cost_buf)
        ctx.enqueue_function[ilqr_reduce_cost_kernel[dtype, Self._T, Self._N]](
            step_view,
            term_view,
            out_view,
            grid_dim=1,
            block_dim=Self._N,
        )

    def _linearize_all[
        CB: RolloutJacobianCallbackGPU
    ](mut self, ctx: DeviceContext, mut callback: CB,) raises:
        for t in range(Self._T):
            var z_t = rebind[
                Tile[dtype, Self._N, CB.LATENT_DIM]
            ](
                tile[Self._N, Self._L](self.z_seq_buf.create_sub_buffer[dtype]( t * Self._N * Self._L, Self._N * Self._L ))
            )
            var u_t = rebind[
                Tile[dtype, Self._N, CB.ACTION_DIM]
            ](
                tile[Self._N, Self._A](self.U_buf.create_sub_buffer[dtype]( t * Self._N * Self._A, Self._N * Self._A ))
            )
            var A_t = rebind[
                Tile[dtype, Self._N, CB.LATENT_DIM, CB.LATENT_DIM]
            ](
                tile[Self._N, Self._L, Self._L](self.A_seq_buf.create_sub_buffer[dtype]( t * Self._N * Self._LL, Self._N * Self._LL ))
            )
            var B_t = rebind[
                Tile[dtype, Self._N, CB.LATENT_DIM, CB.ACTION_DIM]
            ](
                tile[Self._N, Self._L, Self._A](self.B_seq_buf.create_sub_buffer[dtype]( t * Self._N * Self._LA, Self._N * Self._LA ))
            )
            var lz_t = rebind[
                Tile[dtype, Self._N, CB.LATENT_DIM]
            ](
                tile[Self._N, Self._L](self.l_z_seq_buf.create_sub_buffer[dtype]( t * Self._N * Self._L, Self._N * Self._L ))
            )
            var lu_t = rebind[
                Tile[dtype, Self._N, CB.ACTION_DIM]
            ](
                tile[Self._N, Self._A](self.l_u_seq_buf.create_sub_buffer[dtype]( t * Self._N * Self._A, Self._N * Self._A ))
            )
            var lzz_t = rebind[
                Tile[dtype, Self._N, CB.LATENT_DIM, CB.LATENT_DIM]
            ](
                tile[Self._N, Self._L, Self._L](self.l_zz_seq_buf.create_sub_buffer[dtype]( t * Self._N * Self._LL, Self._N * Self._LL ))
            )
            var luu_t = rebind[
                Tile[dtype, Self._N, CB.ACTION_DIM, CB.ACTION_DIM]
            ](
                tile[Self._N, Self._A, Self._A](self.l_uu_seq_buf.create_sub_buffer[dtype]( t * Self._N * Self._AA, Self._N * Self._AA ))
            )
            var lzu_t = rebind[
                Tile[dtype, Self._N, CB.LATENT_DIM, CB.ACTION_DIM]
            ](
                tile[Self._N, Self._L, Self._A](self.l_zu_seq_buf.create_sub_buffer[dtype]( t * Self._N * Self._LA, Self._N * Self._LA ))
            )
            callback.linearize_gpu[Self._N](
                ctx, z_t, u_t, A_t, B_t, lz_t, lu_t, lzz_t, luu_t, lzu_t
            )

        self._terminal_on_z_seq(ctx, callback)

    def _backward_pass(mut self, ctx: DeviceContext, mu: Float64) raises:
        var A_view = tile[Self._T * Self._N * Self._LL](self.A_seq_buf)
        var B_view = tile[Self._T * Self._N * Self._LA](self.B_seq_buf)
        var lz_view = tile[Self._T * Self._N * Self._L](self.l_z_seq_buf)
        var lu_view = tile[Self._T * Self._N * Self._A](self.l_u_seq_buf)
        var lzz_view = tile[Self._T * Self._N * Self._LL](self.l_zz_seq_buf)
        var luu_view = tile[Self._T * Self._N * Self._AA](self.l_uu_seq_buf)
        var lzu_view = tile[Self._T * Self._N * Self._LA](self.l_zu_seq_buf)
        var Vz_view = tile[Self._N * Self._L](self.V_z_term_buf)
        var Vzz_view = tile[Self._N * Self._LL](self.V_zz_term_buf)
        var K_view = tile[Self._T * Self._N * Self._LA](self.K_seq_buf)
        var k_view = tile[Self._T * Self._N * Self._A](self.k_seq_buf)
        var bw_view = tile[Self._N](self.bw_ok_buf)
        ctx.enqueue_function[
            ilqr_backward_pass_kernel[dtype, Self._N, Self._T, Self._L, Self._A]
        ](
            A_view,
            B_view,
            lz_view,
            lu_view,
            lzz_view,
            luu_view,
            lzu_view,
            Vz_view,
            Vzz_view,
            K_view,
            k_view,
            Scalar[dtype](mu),
            bw_view,
            grid_dim=Self._N,
            block_dim=1,
        )

    def _forward_with_gains[
        CB: RolloutJacobianCallbackGPU
    ](mut self, ctx: DeviceContext, mut callback: CB, alpha: Float64,) raises:
        var U_view = tile[Self._T * Self._N * Self._A](self.U_buf)
        var k_view = tile[Self._T * Self._N * Self._A](self.k_seq_buf)
        var K_view = tile[Self._T * Self._N * Self._LA](self.K_seq_buf)
        var zseq_view = tile[(Self._T + 1) * Self._N * Self._L](self.z_seq_buf)
        var ztrial_view = tile[(Self._T + 1) * Self._N * Self._L](self.z_trial_buf)
        var Utrial_view = tile[Self._T * Self._N * Self._A](self.U_trial_buf)
        for t in range(Self._T):
            ctx.enqueue_function[
                ilqr_apply_control_update_kernel[
                    dtype, Self._N, Self._T, Self._L, Self._A
                ]
            ](
                U_view,
                k_view,
                K_view,
                zseq_view,
                ztrial_view,
                Scalar[dtype](alpha),
                Utrial_view,
                Int32(t),
                grid_dim=Self._N,
                block_dim=Self._A,
            )
            var z_t = rebind[
                Tile[dtype, Self._N, CB.LATENT_DIM]
            ](
                tile[Self._N, Self._L](self.z_trial_buf.create_sub_buffer[dtype]( t * Self._N * Self._L, Self._N * Self._L ))
            )
            var u_t = rebind[
                Tile[dtype, Self._N, CB.ACTION_DIM]
            ](
                tile[Self._N, Self._A](self.U_trial_buf.create_sub_buffer[dtype]( t * Self._N * Self._A, Self._N * Self._A ))
            )
            var z_next = rebind[
                Tile[dtype, Self._N, CB.LATENT_DIM]
            ](
                tile[Self._N, Self._L](self.z_trial_buf.create_sub_buffer[dtype]( (t + 1) * Self._N * Self._L, Self._N * Self._L ))
            )
            var c_t = rebind[
                Tile[dtype, Self._N]
            ](
                tile[Self._N](self.step_cost_buf.create_sub_buffer[dtype]( t * Self._N, Self._N ))
            )
            callback.step_gpu[Self._N](ctx, z_t, u_t, z_next, c_t)

        self._terminal_on_z_trial(ctx, callback)

    def _accept_all(mut self, ctx: DeviceContext) raises:
        var Utrial_view = tile[Self._T * Self._N * Self._A](self.U_trial_buf)
        var ztrial_view = tile[(Self._T + 1) * Self._N * Self._L](self.z_trial_buf)
        var trial_view = tile[Self._N](self.trial_cost_buf)
        var U_view = tile[Self._T * Self._N * Self._A](self.U_buf)
        var zseq_view = tile[(Self._T + 1) * Self._N * Self._L](self.z_seq_buf)
        var total_view = tile[Self._N](self.total_cost_buf)
        ctx.enqueue_function[
            ilqr_accept_kernel[dtype, Self._N, Self._T, Self._L, Self._A]
        ](
            Utrial_view,
            ztrial_view,
            trial_view,
            U_view,
            zseq_view,
            total_view,
            grid_dim=Self._N,
            block_dim=1,
        )

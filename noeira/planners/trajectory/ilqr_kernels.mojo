"""Planner-side GPU kernels for iLQR.

All kernels take 1-D flat ``LayoutTensor`` views and do explicit
offset arithmetic. Reasons:

* Multi-D ``Layout.row_major`` ≥ 4 dims is uncharted in this codebase
  and the per-step matrices iLQR works with (e.g. ``A[T,N,L,L]``) need
  4-D indexing.
* All LT reads inside a kernel are rebound to ``Scalar[dtype]`` to
  avoid the ``SIMD[dtype, element_size]`` vs ``Scalar[dtype]``
  mismatch when different layouts feed the same expression (each LT
  has its own ``element_size`` comptime expression — structurally
  distinct even when numerically equal).

Buffer layouts (all timestep-major, kernel-private):
  * ``U`` / ``U_trial`` / ``k_seq`` / ``l_u_seq``: ``(T, N, A)``
  * ``z_seq`` / ``z_trial``: ``(T+1, N, L)``
  * ``l_z_seq``: ``(T, N, L)``
  * ``A_seq``: ``(T, N, L, L)``
  * ``B_seq``: ``(T, N, L, A)``
  * ``l_zz_seq``: ``(T, N, L, L)``
  * ``l_uu_seq``: ``(T, N, A, A)``
  * ``l_zu_seq``: ``(T, N, L, A)``
  * ``K_seq``: ``(T, N, A, L)``
  * ``V_z_term``: ``(N, L)``
  * ``V_zz_term``: ``(N, L, L)``
  * ``step_cost``: ``(T, N)``
  * ``term_cost`` / ``trial_cost`` / ``total_cost``: ``(N,)``
  * ``bw_ok``: ``(N,)`` int32

Kernels:

  * ``ilqr_copy_z0_kernel`` — write ``z0[N, L]`` into ``z_seq[0, *, *]``.
    Grid = ``N_ENVS``, block = ``LATENT_DIM``.
  * ``ilqr_reduce_cost_kernel`` — sum step_cost over horizon + add
    terminal cost. Single block, ``block_dim = N_ENVS``.
  * ``ilqr_apply_control_update_kernel`` — at timestep ``t``, compute
    ``u_trial[t] = U[t] + α·k[t] + K[t]·(z_trial[t] - z_seq[t])``.
    Grid = ``N_ENVS``, block = ``ACTION_DIM``.
  * ``ilqr_backward_pass_kernel`` — full Riccati sweep. Grid =
    ``N_ENVS``, block = 1. Each block runs the sequential ``T → 0``
    pass for one env in local scratch. Sets ``bw_ok[e] = 0`` on any
    LDL failure.
  * ``ilqr_accept_kernel`` — copy trial → current for one env. Grid =
    ``N_ENVS``, block = 1.
"""

from max.gpu import thread_idx, block_idx, block_dim
from noeira.core.tiles import Tile


# =============================================================================
# ilqr_copy_z0_kernel
# =============================================================================


def ilqr_copy_z0_kernel[
    dtype: DType,
    N_ENVS: Int,
    LATENT_DIM: Int,
](
    z0: Tile[dtype, N_ENVS, LATENT_DIM],
    z_seq: Tile[dtype, N_ENVS * LATENT_DIM],
):
    """Copy ``z0[e, d]`` → ``z_seq[0, e, d]``."""
    var e = Int(block_idx.x)
    var d = Int(thread_idx.x)
    if e >= N_ENVS or d >= LATENT_DIM:
        return
    z_seq[e * LATENT_DIM + d] = rebind[z_seq.ElementType](z0[e, d])


# =============================================================================
# ilqr_reduce_cost_kernel
# =============================================================================


def ilqr_reduce_cost_kernel[
    dtype: DType,
    HORIZON: Int,
    N_ENVS: Int,
](
    step_cost: Tile[dtype, HORIZON * N_ENVS],
    term_cost: Tile[dtype, N_ENVS],
    total_out: Tile[dtype, N_ENVS],
):
    """Per-env reduce: sum step_cost over horizon + terminal cost."""
    var e = Int(thread_idx.x)
    if e >= N_ENVS:
        return
    var s = Scalar[dtype](0.0)
    for t in range(HORIZON):
        s += rebind[Scalar[dtype]](step_cost[t * N_ENVS + e])
    s += rebind[Scalar[dtype]](term_cost[e])
    total_out[e] = rebind[total_out.ElementType](s)


# =============================================================================
# ilqr_apply_control_update_kernel
# =============================================================================


def ilqr_apply_control_update_kernel[
    dtype: DType,
    N_ENVS: Int,
    HORIZON: Int,
    LATENT_DIM: Int,
    ACTION_DIM: Int,
](
    U: Tile[dtype, HORIZON * N_ENVS * ACTION_DIM],
    k_seq: Tile[dtype, HORIZON * N_ENVS * ACTION_DIM],
    K_seq: Tile[dtype, HORIZON * N_ENVS * ACTION_DIM * LATENT_DIM],
    z_seq: Tile[dtype, (HORIZON + 1) * N_ENVS * LATENT_DIM],
    z_trial: Tile[dtype, (HORIZON + 1) * N_ENVS * LATENT_DIM],
    alpha: Scalar[dtype],
    U_trial: Tile[dtype, HORIZON * N_ENVS * ACTION_DIM],
    # ⚠ Int32, NOT Int — `Int`/`UInt` are not `DevicePassable`. A bare
    # `Int` still compiles in `pixi run build` and fails only where the
    # kernel is LAUNCHED; keep the `Int32(...)` casts at the call sites.
    t: Int32,
):
    var e = Int(block_idx.x)
    var i = Int(thread_idx.x)
    if e >= N_ENVS or i >= ACTION_DIM:
        return

    var tt = Int(t)
    var u_base = (tt * N_ENVS + e) * ACTION_DIM
    var k_idx = u_base + i
    var ff = rebind[Scalar[dtype]](k_seq[k_idx])

    var K_row_base = (tt * N_ENVS + e) * ACTION_DIM * LATENT_DIM + i * LATENT_DIM
    var z_base = (tt * N_ENVS + e) * LATENT_DIM
    var fb = Scalar[dtype](0.0)
    for j in range(LATENT_DIM):
        var k_val = rebind[Scalar[dtype]](K_seq[K_row_base + j])
        var zt = rebind[Scalar[dtype]](z_trial[z_base + j])
        var zs = rebind[Scalar[dtype]](z_seq[z_base + j])
        fb += k_val * (zt - zs)
    var u_old = rebind[Scalar[dtype]](U[u_base + i])
    U_trial[u_base + i] = rebind[U_trial.ElementType](u_old + alpha * ff + fb)


# =============================================================================
# ilqr_backward_pass_kernel
# =============================================================================


def ilqr_backward_pass_kernel[
    dtype: DType,
    N_ENVS: Int,
    HORIZON: Int,
    LATENT_DIM: Int,
    ACTION_DIM: Int,
](
    A_seq: Tile[dtype, HORIZON * N_ENVS * LATENT_DIM * LATENT_DIM],
    B_seq: Tile[dtype, HORIZON * N_ENVS * LATENT_DIM * ACTION_DIM],
    l_z_seq: Tile[dtype, HORIZON * N_ENVS * LATENT_DIM],
    l_u_seq: Tile[dtype, HORIZON * N_ENVS * ACTION_DIM],
    l_zz_seq: Tile[dtype, HORIZON * N_ENVS * LATENT_DIM * LATENT_DIM],
    l_uu_seq: Tile[dtype, HORIZON * N_ENVS * ACTION_DIM * ACTION_DIM],
    l_zu_seq: Tile[dtype, HORIZON * N_ENVS * LATENT_DIM * ACTION_DIM],
    V_z_term: Tile[dtype, N_ENVS * LATENT_DIM],
    V_zz_term: Tile[dtype, N_ENVS * LATENT_DIM * LATENT_DIM],
    K_seq: Tile[dtype, HORIZON * N_ENVS * ACTION_DIM * LATENT_DIM],
    k_seq: Tile[dtype, HORIZON * N_ENVS * ACTION_DIM],
    mu: Scalar[dtype],
    bw_ok: Tile[DType.int32, N_ENVS],
):
    """Single-thread-per-env Riccati backward pass with LM
    regularization. Sequential ``t = T-1 .. 0`` per block.
    """
    var e = Int(block_idx.x)
    if e >= N_ENVS:
        return

    comptime _LL = LATENT_DIM * LATENT_DIM
    comptime _AA = ACTION_DIM * ACTION_DIM
    comptime _LA = LATENT_DIM * ACTION_DIM
    comptime _RHS_W = 1 + LATENT_DIM

    var V_z = Array[Scalar[dtype], LATENT_DIM](fill=Scalar[dtype](0.0))
    var V_zz = Array[Scalar[dtype], _LL](fill=Scalar[dtype](0.0))
    for i in range(LATENT_DIM):
        V_z[i] = rebind[Scalar[dtype]](V_z_term[e * LATENT_DIM + i])
        for j in range(LATENT_DIM):
            V_zz[i * LATENT_DIM + j] = rebind[Scalar[dtype]](
                V_zz_term[(e * LATENT_DIM + i) * LATENT_DIM + j]
            )

    var Q_z = Array[Scalar[dtype], LATENT_DIM](fill=Scalar[dtype](0.0))
    var Q_u = Array[Scalar[dtype], ACTION_DIM](fill=Scalar[dtype](0.0))
    var Q_zz = Array[Scalar[dtype], _LL](fill=Scalar[dtype](0.0))
    var Q_uu = Array[Scalar[dtype], _AA](fill=Scalar[dtype](0.0))
    var Q_zu = Array[Scalar[dtype], _LA](fill=Scalar[dtype](0.0))
    var tmp_LL = Array[Scalar[dtype], _LL](fill=Scalar[dtype](0.0))
    var tmp_LA = Array[Scalar[dtype], _LA](fill=Scalar[dtype](0.0))
    var tmp_AL = Array[Scalar[dtype], _LA](fill=Scalar[dtype](0.0))
    var quu_solve = Array[Scalar[dtype], _AA](fill=Scalar[dtype](0.0))
    var rhs_solve = Array[Scalar[dtype], ACTION_DIM * _RHS_W](
        fill=Scalar[dtype](0.0)
    )

    var diag_eps = Scalar[dtype](1.0e-12)

    for t_rev in range(HORIZON):
        var t = HORIZON - 1 - t_rev
        var step_base = t * N_ENVS + e
        var A_off = step_base * _LL
        var B_off = step_base * _LA
        var lz_off = step_base * LATENT_DIM
        var lu_off = step_base * ACTION_DIM
        var lzz_off = step_base * _LL
        var luu_off = step_base * _AA
        var lzu_off = step_base * _LA
        var K_off = step_base * _LA
        var k_off = step_base * ACTION_DIM

        # ---- Q_z = l_z + A^T V_z ----
        for i in range(LATENT_DIM):
            var s = rebind[Scalar[dtype]](l_z_seq[lz_off + i])
            for r in range(LATENT_DIM):
                var a = rebind[Scalar[dtype]](A_seq[A_off + r * LATENT_DIM + i])
                s += a * V_z[r]
            Q_z[i] = s

        # ---- Q_u = l_u + B^T V_z ----
        for i in range(ACTION_DIM):
            var s = rebind[Scalar[dtype]](l_u_seq[lu_off + i])
            for r in range(LATENT_DIM):
                var b = rebind[Scalar[dtype]](B_seq[B_off + r * ACTION_DIM + i])
                s += b * V_z[r]
            Q_u[i] = s

        # ---- tmp_LL = V_zz @ A ----
        for i in range(LATENT_DIM):
            for j in range(LATENT_DIM):
                var s = Scalar[dtype](0.0)
                for r in range(LATENT_DIM):
                    var a = rebind[Scalar[dtype]](
                        A_seq[A_off + r * LATENT_DIM + j]
                    )
                    s += V_zz[i * LATENT_DIM + r] * a
                tmp_LL[i * LATENT_DIM + j] = s

        # ---- Q_zz = l_zz + A^T @ tmp_LL ----
        for i in range(LATENT_DIM):
            for j in range(LATENT_DIM):
                var s = rebind[Scalar[dtype]](
                    l_zz_seq[lzz_off + i * LATENT_DIM + j]
                )
                for r in range(LATENT_DIM):
                    var a = rebind[Scalar[dtype]](
                        A_seq[A_off + r * LATENT_DIM + i]
                    )
                    s += a * tmp_LL[r * LATENT_DIM + j]
                Q_zz[i * LATENT_DIM + j] = s

        # ---- tmp_LA = V_zz @ B ----
        for i in range(LATENT_DIM):
            for j in range(ACTION_DIM):
                var s = Scalar[dtype](0.0)
                for r in range(LATENT_DIM):
                    var b = rebind[Scalar[dtype]](
                        B_seq[B_off + r * ACTION_DIM + j]
                    )
                    s += V_zz[i * LATENT_DIM + r] * b
                tmp_LA[i * ACTION_DIM + j] = s

        # ---- Q_uu = l_uu + B^T @ tmp_LA ----
        for i in range(ACTION_DIM):
            for j in range(ACTION_DIM):
                var s = rebind[Scalar[dtype]](
                    l_uu_seq[luu_off + i * ACTION_DIM + j]
                )
                for r in range(LATENT_DIM):
                    var b = rebind[Scalar[dtype]](
                        B_seq[B_off + r * ACTION_DIM + i]
                    )
                    s += b * tmp_LA[r * ACTION_DIM + j]
                Q_uu[i * ACTION_DIM + j] = s

        # ---- Q_zu = l_zu + A^T @ tmp_LA ----
        for i in range(LATENT_DIM):
            for j in range(ACTION_DIM):
                var s = rebind[Scalar[dtype]](
                    l_zu_seq[lzu_off + i * ACTION_DIM + j]
                )
                for r in range(LATENT_DIM):
                    var a = rebind[Scalar[dtype]](
                        A_seq[A_off + r * LATENT_DIM + i]
                    )
                    s += a * tmp_LA[r * ACTION_DIM + j]
                Q_zu[i * ACTION_DIM + j] = s

        # ---- LDL solve of (Q_uu + μI) [k|K_row^T] = [-Q_u | -Q_uz] ----
        for i in range(ACTION_DIM):
            for j in range(ACTION_DIM):
                quu_solve[i * ACTION_DIM + j] = Q_uu[i * ACTION_DIM + j]
            quu_solve[i * ACTION_DIM + i] += mu

        for i in range(ACTION_DIM):
            rhs_solve[i * _RHS_W + 0] = -Q_u[i]
            for j in range(LATENT_DIM):
                rhs_solve[i * _RHS_W + 1 + j] = -Q_zu[j * ACTION_DIM + i]

        var pd_ok = True
        for j in range(ACTION_DIM):
            var d = quu_solve[j * ACTION_DIM + j]
            for kk in range(j):
                d -= (
                    quu_solve[j * ACTION_DIM + kk]
                    * quu_solve[j * ACTION_DIM + kk]
                    * quu_solve[kk * ACTION_DIM + kk]
                )
            if d <= diag_eps:
                pd_ok = False
            quu_solve[j * ACTION_DIM + j] = d
            for ii in range(j + 1, ACTION_DIM):
                var s = quu_solve[ii * ACTION_DIM + j]
                for kk in range(j):
                    s -= (
                        quu_solve[ii * ACTION_DIM + kk]
                        * quu_solve[j * ACTION_DIM + kk]
                        * quu_solve[kk * ACTION_DIM + kk]
                    )
                quu_solve[ii * ACTION_DIM + j] = s / d

        if not pd_ok:
            bw_ok[e] = rebind[bw_ok.ElementType](Scalar[DType.int32](0))
            return

        for c in range(_RHS_W):
            for i in range(ACTION_DIM):
                var s = rhs_solve[i * _RHS_W + c]
                for kk in range(i):
                    s -= (
                        quu_solve[i * ACTION_DIM + kk]
                        * rhs_solve[kk * _RHS_W + c]
                    )
                rhs_solve[i * _RHS_W + c] = s
            for i in range(ACTION_DIM):
                rhs_solve[i * _RHS_W + c] /= quu_solve[i * ACTION_DIM + i]
            for i in range(ACTION_DIM - 1, -1, -1):
                var s = rhs_solve[i * _RHS_W + c]
                for kk in range(i + 1, ACTION_DIM):
                    s -= (
                        quu_solve[kk * ACTION_DIM + i]
                        * rhs_solve[kk * _RHS_W + c]
                    )
                rhs_solve[i * _RHS_W + c] = s

        # Write k, K to global buffers.
        for i in range(ACTION_DIM):
            k_seq[k_off + i] = rebind[k_seq.ElementType](
                rhs_solve[i * _RHS_W + 0]
            )
            for j in range(LATENT_DIM):
                K_seq[K_off + i * LATENT_DIM + j] = rebind[K_seq.ElementType](
                    rhs_solve[i * _RHS_W + 1 + j]
                )

        # ---- Update V_z, V_zz (Tassa Eqs. 11–12) ----
        # tmp_AL = Q_uu @ K  (A,L). K row-major (A,L) ⇒ reads via offset.
        for i in range(ACTION_DIM):
            for j in range(LATENT_DIM):
                var s = Scalar[dtype](0.0)
                for r in range(ACTION_DIM):
                    var k_val = rebind[Scalar[dtype]](
                        K_seq[K_off + r * LATENT_DIM + j]
                    )
                    s += Q_uu[i * ACTION_DIM + r] * k_val
                tmp_AL[i * LATENT_DIM + j] = s

        var Quu_k = Array[Scalar[dtype], ACTION_DIM](
            fill=Scalar[dtype](0.0)
        )
        for i in range(ACTION_DIM):
            var s = Scalar[dtype](0.0)
            for r in range(ACTION_DIM):
                var k_val = rebind[Scalar[dtype]](k_seq[k_off + r])
                s += Q_uu[i * ACTION_DIM + r] * k_val
            Quu_k[i] = s

        # V_z = Q_z + K^T(Q_u + Quu_k) + Q_zu k
        for i in range(LATENT_DIM):
            var s = Q_z[i]
            for r in range(ACTION_DIM):
                var k_val = rebind[Scalar[dtype]](
                    K_seq[K_off + r * LATENT_DIM + i]
                )
                s += k_val * (Q_u[r] + Quu_k[r])
            for r in range(ACTION_DIM):
                var k_val = rebind[Scalar[dtype]](k_seq[k_off + r])
                s += Q_zu[i * ACTION_DIM + r] * k_val
            V_z[i] = s

        # tmp_LL := Q_zz + K^T (Q_uu K)
        for i in range(LATENT_DIM):
            for j in range(LATENT_DIM):
                var s = Q_zz[i * LATENT_DIM + j]
                for r in range(ACTION_DIM):
                    var k_val = rebind[Scalar[dtype]](
                        K_seq[K_off + r * LATENT_DIM + i]
                    )
                    s += k_val * tmp_AL[r * LATENT_DIM + j]
                tmp_LL[i * LATENT_DIM + j] = s

        # V_zz = tmp_LL + (Q_zu K) + (Q_zu K)^T
        for i in range(LATENT_DIM):
            for j in range(LATENT_DIM):
                var cij = Scalar[dtype](0.0)
                var cji = Scalar[dtype](0.0)
                for r in range(ACTION_DIM):
                    var k_ij = rebind[Scalar[dtype]](
                        K_seq[K_off + r * LATENT_DIM + j]
                    )
                    var k_ji = rebind[Scalar[dtype]](
                        K_seq[K_off + r * LATENT_DIM + i]
                    )
                    cij += Q_zu[i * ACTION_DIM + r] * k_ij
                    cji += Q_zu[j * ACTION_DIM + r] * k_ji
                V_zz[i * LATENT_DIM + j] = (
                    tmp_LL[i * LATENT_DIM + j] + cij + cji
                )

    bw_ok[e] = rebind[bw_ok.ElementType](Scalar[DType.int32](1))


# =============================================================================
# ilqr_accept_kernel
# =============================================================================


def ilqr_accept_kernel[
    dtype: DType,
    N_ENVS: Int,
    HORIZON: Int,
    LATENT_DIM: Int,
    ACTION_DIM: Int,
](
    U_trial: Tile[dtype, HORIZON * N_ENVS * ACTION_DIM],
    z_trial: Tile[dtype, (HORIZON + 1) * N_ENVS * LATENT_DIM],
    trial_cost: Tile[dtype, N_ENVS],
    U: Tile[dtype, HORIZON * N_ENVS * ACTION_DIM],
    z_seq: Tile[dtype, (HORIZON + 1) * N_ENVS * LATENT_DIM],
    total_cost: Tile[dtype, N_ENVS],
):
    """Per-env unconditional copy of trial → current."""
    var e = Int(block_idx.x)
    if e >= N_ENVS:
        return
    for t in range(HORIZON):
        var off = (t * N_ENVS + e) * ACTION_DIM
        for a in range(ACTION_DIM):
            U[off + a] = rebind[U.ElementType](U_trial[off + a])
    for t in range(HORIZON + 1):
        var off = (t * N_ENVS + e) * LATENT_DIM
        for d in range(LATENT_DIM):
            z_seq[off + d] = rebind[z_seq.ElementType](z_trial[off + d])
    total_cost[e] = rebind[total_cost.ElementType](trial_cost[e])

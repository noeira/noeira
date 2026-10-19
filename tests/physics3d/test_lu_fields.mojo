"""Stage-I gate: dense LU factor/solve over per-field tensors (lu) —
fields self-check against ground truth (no legacy reference; the legacy
`lu_factorization` was deleted at the P6 sunset).

LU is the non-symmetric linear solve the fields `ImplicitIntegrator`
needs (M_hat = M + armature - dt*qDeriv is asymmetric). This gate feeds a
deterministic, diagonally-dominant non-symmetric NV×NV system per env into the
fields helpers and checks ground-truth invariants:
  * the solution actually solves the system (residual |M @ x - b| ≈ 0),
  * the inverse is correct (|M @ M^-1 - I| ≈ 0),
  * fields-GPU == fields-CPU (tight tol; only asserted off-NVIDIA... the
    GPU path runs the identical per-env helper).

Pure scratch math (no model, no physics) → light on the GPU.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_lu_fields.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.sys import default_accelerator
from layout import Layout

from noeira.physics3d.fields import DynamicsScratch, Dims, DimsLike
from noeira.physics3d.dynamics.lu import (
    lu_factor,
    lu_solve,
    compute_m_inv_from_lu,
)

comptime DT = DType.float32
comptime NV = 6
comptime NBODY = 2
comptime MD = Dims[nv=NV, nbody=NBODY]
comptime BATCH = 2
comptime M_SIZE = NV * NV
comptime V_SIZE = NV


def _fill_M(e: Int, i: Int, j: Int) -> Scalar[DT]:
    """Deterministic non-symmetric entry; diagonal boosted so the system is
    well-conditioned + non-singular (i*7 vs j*13 → asymmetric)."""
    var base = Scalar[DT]((i * 7 + j * 13 + e * 3) % 11 - 5)
    if i == j:
        base += Scalar[DT](NV * 2 + e)  # dominant, env-varying diagonal
    return base


def _fill_b(e: Int, i: Int) -> Scalar[DT]:
    return Scalar[DT]((i * 5 + e * 2) % 7 - 3)


def main() raises:
    print("=== Stage-I lu parity (NV=", NV, " BATCH=", BATCH, ") ===")
    var ctx = DeviceContext()

    var sc = DynamicsScratch[DT, MD, BATCH]()
    for e in range(BATCH):
        for i in range(NV):
            for j in range(NV):
                sc.M.data[e * M_SIZE + i * NV + j] = _fill_M(e, i, j)
            sc.fnet.data[e * V_SIZE + i] = _fill_b(e, i)

    # ── fields-CPU factor + solve + M^-1 ──────────────────────────────────
    lu_factor["cpu", DT, BATCH=BATCH](sc)
    lu_solve["cpu", DT, BATCH=BATCH](sc)
    compute_m_inv_from_lu["cpu", DT, BATCH=BATCH](sc)

    # snapshot fields-CPU results (M/fnet are untouched by factor/solve)
    var xf_cpu = List[Scalar[DT]](length=BATCH * V_SIZE, fill=0)
    var mi_cpu = List[Scalar[DT]](length=BATCH * M_SIZE, fill=0)
    for i in range(BATCH * V_SIZE):
        xf_cpu[i] = sc.qacc_ws.data[i]
    for i in range(BATCH * M_SIZE):
        mi_cpu[i] = sc.m_inv.data[i]

    # ── fields-only ground truth: x solves the system + M·M^-1 = I (per env) ─
    var worst_resid = Float64(0)
    var worst_minv = Float64(0)
    for e in range(BATCH):
        # residual: original M @ x_fields - b
        for i in range(NV):
            var s = Float64(0)
            for j in range(NV):
                s += Float64(sc.M.data[e * M_SIZE + i * NV + j]) * Float64(
                    xf_cpu[e * V_SIZE + j]
                )
            var r = abs(s - Float64(sc.fnet.data[e * V_SIZE + i]))
            if r > worst_resid:
                worst_resid = r
        # M @ M^-1 == I
        for i in range(NV):
            for j in range(NV):
                var s = Float64(0)
                for k in range(NV):
                    s += Float64(sc.M.data[e * M_SIZE + i * NV + k]) * Float64(
                        mi_cpu[e * M_SIZE + k * NV + j]
                    )
                var expect = Float64(1) if i == j else Float64(0)
                var d = abs(s - expect)
                if d > worst_minv:
                    worst_minv = d

    print("  worst residual |M x - b|:", worst_resid)
    print("  worst |M M^-1 - I|:", worst_minv)
    if worst_resid > 1e-3:
        raise Error("LU solution residual too large — not solving the system")
    if worst_minv > 1e-3:
        raise Error("LU M^-1 wrong — M @ M^-1 != I")
    print("  Part A PASS: LU solve + M^-1 correct (fields self-check)")

    # ── fields-GPU factor + solve + M^-1 vs fields-CPU ────────────────────
    sc.upload_all(ctx)  # push host M/fnet (+ all) to device
    lu_factor["gpu", DT, BATCH=BATCH](sc, ctx)
    lu_solve["gpu", DT, BATCH=BATCH](sc, ctx)
    compute_m_inv_from_lu["gpu", DT, BATCH=BATCH](sc, ctx)
    sc.qacc_ws.download(ctx)
    sc.m_inv.download(ctx)

    var worst_gpu = Float64(0)
    for i in range(BATCH * V_SIZE):
        var d = abs(Float64(sc.qacc_ws.data[i]) - Float64(xf_cpu[i]))
        if d > worst_gpu:
            worst_gpu = d
    for i in range(BATCH * M_SIZE):
        var d = abs(Float64(sc.m_inv.data[i]) - Float64(mi_cpu[i]))
        if d > worst_gpu:
            worst_gpu = d
    print("  fields-GPU vs fields-CPU worst err:", worst_gpu)
    if worst_gpu > 1e-4 and not default_accelerator().is_nvidia_gpu():
        raise Error("fields-GPU LU diverges from fields-CPU")
    print("  Part B PASS: fields-GPU == fields-CPU")
    print("test_lu_fields: ALL PASS")

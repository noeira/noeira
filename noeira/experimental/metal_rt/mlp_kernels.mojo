"""Naive kernels for one training step of a 2-layer MLP (MSE loss, Adam).

    Z1 = X W1 + b1      A1 = relu(Z1)      Z2 = A1 W2 + b2
    L  = mean((Z2 - Y)^2)

Integer arguments are `Int64` (MAX's `enqueue_function` rejects `Int`:
"Int and UInt do not conform to DevicePassable"). Shapes: X[B,I], W1[I,H], b1[H], W2[H,O], b2[O], Y[B,O]; row-major.
One thread per output element, sums in a fixed order, so every result is
deterministic and two launchers running the same metallib in the same order
must agree bit for bit. 14 launches per step (`STEP_LAUNCHES`).
"""

from max.gpu import global_idx, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.math import sqrt
from std.memory import stack_allocation

comptime P = Pointer[Float32, MutAnyOrigin]
comptime TPB = 256
comptime STEP_LAUNCHES = 14


def fc_fwd(out_: P, x: P, w: P, b: P, rows_: Int64, k_: Int64, n_: Int64):
    """`out[r, j] = sum_k x[r, k] w[k, j] + b[j]`."""
    var rows = Int(rows_)
    var k = Int(k_)
    var n = Int(n_)
    var t = global_idx.x
    if t < rows * n:
        var r = t // n
        var j = t % n
        var acc = b[unsafe_offset=j]
        for kk in range(k):
            acc += x[unsafe_offset=r * k + kk] * w[unsafe_offset=kk * n + j]
        out_[unsafe_offset=t] = acc


def relu_fwd(a: P, z: P, n_: Int64):
    var n = Int(n_)
    var t = global_idx.x
    if t < n:
        var v = z[unsafe_offset=t]
        a[unsafe_offset=t] = v if v > 0.0 else 0.0


def mse_grad(dz: P, z: P, y: P, n_: Int64, scale: Float32):
    """`dz = scale * (z - y)`, `scale = 2 / n`."""
    var n = Int(n_)
    var t = global_idx.x
    if t < n:
        dz[unsafe_offset=t] = scale * (z[unsafe_offset=t] - y[unsafe_offset=t])


def mse_loss(losses: P, z: P, y: P, n_: Int64, step_: Int64):
    """`losses[step] = mean((z - y)^2)`; ONE threadgroup of TPB threads, strided
    partial sums then a fixed-shape tree: deterministic."""
    var n = Int(n_)
    var step = Int(step_)
    var sh = stack_allocation[
        TPB, Float32, address_space=AddressSpace.SHARED
    ]()
    var tid = thread_idx.x
    var acc: Float32 = 0.0
    var i = tid
    while i < n:
        var d = z[unsafe_offset=i] - y[unsafe_offset=i]
        acc += d * d
        i += TPB
    sh[unsafe_offset=tid] = acc
    barrier()
    var active = TPB // 2
    while active > 0:
        if tid < active:
            sh[unsafe_offset=tid] = sh[unsafe_offset=tid] + sh[
                unsafe_offset=tid + active
            ]
        barrier()
        active //= 2
    if tid == 0:
        losses[unsafe_offset=step] = sh[unsafe_offset=0] / Float32(n)


def grad_w(dw: P, a: P, dz: P, rows_: Int64, k_: Int64, n_: Int64):
    """`dw[kk, j] = sum_r a[r, kk] dz[r, j]`   (dW = A^T dZ)."""
    var rows = Int(rows_)
    var k = Int(k_)
    var n = Int(n_)
    var t = global_idx.x
    if t < k * n:
        var kk = t // n
        var j = t % n
        var acc: Float32 = 0.0
        for r in range(rows):
            acc += a[unsafe_offset=r * k + kk] * dz[unsafe_offset=r * n + j]
        dw[unsafe_offset=t] = acc


def grad_b(db: P, dz: P, rows_: Int64, n_: Int64):
    """`db[j] = sum_r dz[r, j]`."""
    var rows = Int(rows_)
    var n = Int(n_)
    var j = global_idx.x
    if j < n:
        var acc: Float32 = 0.0
        for r in range(rows):
            acc += dz[unsafe_offset=r * n + j]
        db[unsafe_offset=j] = acc


def grad_in_relu(dz1: P, dz2: P, w2: P, z1: P, rows_: Int64, h_: Int64, o_: Int64):
    """`dz1[r, j] = (sum_q dz2[r, q] w2[j, q]) * (z1[r, j] > 0)`."""
    var rows = Int(rows_)
    var h = Int(h_)
    var o = Int(o_)
    var t = global_idx.x
    if t < rows * h:
        var r = t // h
        var j = t % h
        var acc: Float32 = 0.0
        for q in range(o):
            acc += dz2[unsafe_offset=r * o + q] * w2[unsafe_offset=j * o + q]
        dz1[unsafe_offset=t] = acc if z1[unsafe_offset=t] > 0.0 else 0.0


def adam(
    p: P,
    g: P,
    m: P,
    v: P,
    n_: Int64,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    bc1: Float32,
    bc2: Float32,
):
    """Adam with host-computed bias corrections bc1 = 1/(1-beta1^t),
    bc2 = 1/(1-beta2^t)."""
    var n = Int(n_)
    var t = global_idx.x
    if t < n:
        var gt = g[unsafe_offset=t]
        var mt = beta1 * m[unsafe_offset=t] + (1.0 - beta1) * gt
        var vt = beta2 * v[unsafe_offset=t] + (1.0 - beta2) * gt * gt
        m[unsafe_offset=t] = mt
        v[unsafe_offset=t] = vt
        p[unsafe_offset=t] = p[unsafe_offset=t] - lr * (mt * bc1) / (
            sqrt(vt * bc2) + eps
        )

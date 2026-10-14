"""GPTQ (Frantar et al., 2023) onto the Q4_0 grid of `q4_0.mojo`.

    gptq_q4_0(w [n, k] fp32, h [k, k] fp64) -> codes [n, k], scales [n, k/32]

Round-to-nearest quantizes every weight on its own. GPTQ quantizes the k
columns of W one at a time and pushes each column's rounding error onto the
columns not yet quantized, weighted by the inverse Hessian of the layer's
inputs (H = Σ x xᵀ over calibration rows), so that the layer's OUTPUT error
tr((W − Ŵ) H (W − Ŵ)ᵀ) is what gets minimized, not the weight error.

## The steps (the reference GPTQ pipeline behind Kev's int4 weights)

1. Dead inputs: a column whose H diagonal is 0 never saw a nonzero input.
   Its weights are zeroed and its diagonal set to 1.
2. Damping: H += 1 % of mean(diag H) on the diagonal.
3. U = the upper Cholesky factor of H⁻¹ (`dense.hinv_upper_f64`, fp64), then
   rounded to fp32: the loop below runs in fp32, as the reference's does.
4. Blocks of 128 columns. In a block, for each column c:
     - at a group start (c % 32 == 0), the group's scale from the CURRENT,
       error-fed weights of columns c .. c+31 (`q4_0_group_scale`)
     - q = the code of w[:, c] at that scale (`q4_0_code`)
     - e = (w[:, c] − deq(q)) / U[c, c]
     - w[:, c:block end] −= e ⊗ U[c, c:block end]
   then the lazy update of every column after the block:
     w[:, end:] −= E @ U[block, end:]      (E: the block's e columns)

The scale is chosen per group from the weights as GPTQ has left them, so a
group's d can differ from RTN's for the same layer: GPTQ moves the grid, not
only the codes. Rows are independent (each row's error stays in that row),
so the loop is threaded over rows; within a row the order is the
reference's.

## What parity means

Floating-point order (the GEMM's accumulation, FMA contraction) flips a code
now and then where a value sits on a rounding boundary, and a flip changes
the error fed to the rest of that row. Parity with the reference is
therefore gated on the objective and on the codes up to such flips
(`tests/nn/quant/test_gptq.mojo`), not code for code.
"""

from max.algorithm import parallelize

from .dense import gemm_acc, gemm_acc_serial, hinv_upper_f64
from .q4_0 import Q4_GROUP, q4_0_group_scale, q4_0_code

comptime GPTQ_BLOCK = 128
comptime _ROWS = 32


struct GPTQStats(Copyable, Movable):
    """What one `gptq_q4_0` call saw: the dead inputs and the damping."""

    var dead: Int
    var damp: Float64

    def __init__(out self):
        self.dead = 0
        self.damp = 0


def gptq_q4_0(
    ref w: List[Float32],
    n: Int,
    k: Int,
    var h: List[Float64],
    mut q: List[UInt8],
    mut d: List[Float32],
    damp_frac: Float64 = 0.01,
) raises -> GPTQStats:
    """GPTQ of a row-major [n, k] fp32 weight on the Q4_0 grid, given the
    [k, k] Hessian of its inputs (consumed). `q` gets the [n, k] codes,
    `d` the [n, k / 32] scales (bf16-rounded fp32), as `quantize_rtn_q4_0`
    does; `pack_q4_0` turns them into blocks. The Hessian's scale does not
    matter (the damping is relative to it)."""
    if k % Q4_GROUP != 0:
        raise Error("gptq_q4_0: k = " + String(k) + " is not a multiple of 32")
    if len(w) < n * k or len(h) < k * k:
        raise Error("gptq_q4_0: w or h is smaller than [n, k] / [k, k]")
    var st = GPTQStats()
    var ww = List[Float32](capacity=n * k)
    for i in range(n * k):
        ww.append(w[i])
    var mean = Float64(0)
    for i in range(k):
        if h[i * k + i] == 0:
            st.dead += 1
            h[i * k + i] = 1.0
            for r in range(n):
                ww[r * k + i] = 0
        mean += h[i * k + i]
    st.damp = damp_frac * mean / Float64(k)
    for i in range(k):
        h[i * k + i] += st.damp
    var u64 = hinv_upper_f64(h^, k)
    var u = List[Float32](capacity=k * k)
    for i in range(k * k):
        u.append(Float32(u64[i]))
    _ = u64^

    var groups = k // Q4_GROUP
    q = List[UInt8](length=n * k, fill=0)
    d = List[Float32](length=n * groups, fill=0)
    var wp = rebind[Pointer[Float32, MutAnyOrigin]](ww.unsafe_ptr())
    var up = rebind[Pointer[Float32, MutAnyOrigin]](u.unsafe_ptr())
    var qp = rebind[Pointer[UInt8, MutAnyOrigin]](q.unsafe_ptr())
    var dp = rebind[Pointer[Float32, MutAnyOrigin]](d.unsafe_ptr())

    def _rows(t: Int) {imm}:
        var r0 = t * _ROWS
        var r1 = min(r0 + _ROWS, n)
        var err = List[Float32](length=_ROWS * GPTQ_BLOCK, fill=0)
        var ep = rebind[Pointer[Float32, MutAnyOrigin]](err.unsafe_ptr())
        for i1 in range(0, k, GPTQ_BLOCK):
            var i2 = min(i1 + GPTQ_BLOCK, k)
            var bs = i2 - i1
            for r in range(r0, r1):
                var row = r * k
                var dg = Float32(0)
                for c in range(i1, i2):
                    if c % Q4_GROUP == 0:
                        dg = q4_0_group_scale(wp.unsafe_offset(row + c))
                        dp[unsafe_offset = r * groups + c // Q4_GROUP] = dg
                    var x = wp[unsafe_offset = row + c]
                    var code = q4_0_code(x, dg)
                    qp[unsafe_offset = row + c] = code
                    var e = (x - (Float32(Int(code)) - 8.0) * dg) / up[
                        unsafe_offset = c * k + c
                    ]
                    for j in range(c, i2):
                        wp[unsafe_offset = row + j] -= e * up[unsafe_offset = c * k + j]
                    ep[unsafe_offset = (r - r0) * bs + (c - i1)] = e
            if i2 < k:
                gemm_acc_serial[DType.float32](
                    r1 - r0,
                    k - i2,
                    bs,
                    -1.0,
                    ep,
                    bs,
                    up.unsafe_offset(i1 * k + i2),
                    k,
                    wp.unsafe_offset(r0 * k + i2),
                    k,
                )

    parallelize(_rows, (n + _ROWS - 1) // _ROWS)
    _ = ww^
    _ = u^
    return st^


def q4_0_objective(
    ref w: List[Float32],
    ref q: List[UInt8],
    ref d: List[Float32],
    n: Int,
    k: Int,
    ref h: List[Float64],
) -> Float64:
    """tr((W − Ŵ) H (W − Ŵ)ᵀ) in fp64, Ŵ = (q − 8) d: the layer's summed
    squared output error over the calibration rows H was built from (H
    undamped). GPTQ minimizes it; RTN ignores it."""
    var groups = k // Q4_GROUP
    var delta = List[Float64](length=n * k, fill=0)
    for r in range(n):
        for c in range(k):
            var deq = (Float64(Int(q[r * k + c])) - 8.0) * Float64(
                d[r * groups + c // Q4_GROUP]
            )
            delta[r * k + c] = Float64(w[r * k + c]) - deq
    var dh = List[Float64](length=n * k, fill=0)
    var hh = h.copy()
    gemm_acc[DType.float64](
        n,
        k,
        k,
        1.0,
        rebind[Pointer[Float64, MutAnyOrigin]](delta.unsafe_ptr()),
        k,
        rebind[Pointer[Float64, MutAnyOrigin]](hh.unsafe_ptr()),
        k,
        rebind[Pointer[Float64, MutAnyOrigin]](dh.unsafe_ptr()),
        k,
    )
    var s = Float64(0)
    for i in range(n * k):
        s += dh[i] * delta[i]
    return s

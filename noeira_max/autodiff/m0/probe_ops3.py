"""M0 probe 3: on-disk compile cache across processes, float64 for the fused
ops, and how to name an op that has no typed binding."""
import sys, time
import numpy as np
from max import _core
from max._core import engine as _eng
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import Graph, TensorType, DeviceRef, ops
from max.engine import InferenceSession

print("max_cache_dir:", _eng.max_cache_dir())
dev = DeviceRef.CPU()

def tiny64():
    with Graph("tiny64", input_types=[TensorType(DType.float64, [5, 4], dev),
                                      TensorType(DType.float64, [4, 3], dev)]) as g:
        x, w = g.inputs
        g.output(ops.sum(ops.relu(ops.matmul(x, w)), axis=-1))
    return g

def fused64():
    B, T, C, V = 2, 5, 8, 11
    with Graph("fused64", input_types=[TensorType(DType.int64, [B, T], dev),
                                       TensorType(DType.float64, [V, C], dev),
                                       TensorType(DType.float64, [C], dev),
                                       TensorType(DType.float64, [C], dev)]) as g:
        idx, wte, gamma, beta = g.inputs
        x = ops.gather(wte, idx, axis=0)
        x = ops.layer_norm(x, gamma, beta, epsilon=1e-5)
        att = ops.matmul(x, ops.transpose(x, -1, -2))
        att = ops.band_part(att, num_lower=None, num_upper=0)
        att = ops.softmax(att)
        y = ops.gelu(ops.matmul(att, x))
        y = ops.gelu(y, approximate="tanh")
        c = ops.concat([y, y], axis=-1)
        l = ops.logsoftmax(c)
        g.output(l)
    return g

sess = InferenceSession(devices=[CPU()])
which = sys.argv[1]
g = tiny64() if which == "tiny" else fused64()
t0 = time.perf_counter()
try:
    m = sess.load(g)
    print(f"{which}: compile {time.perf_counter()-t0:.2f}s")
except Exception as e:
    print(f"{which}: COMPILE FAILED after {time.perf_counter()-t0:.2f}s: {type(e).__name__}: {str(e)[:400]}")
    sys.exit(0)
if which == "fused":
    rng = np.random.default_rng(0)
    out = m.execute(Buffer.from_numpy(rng.integers(0, 11, (2, 5)).astype(np.int64)),
                    Buffer.from_numpy(rng.standard_normal((11, 8))),
                    Buffer.from_numpy(rng.standard_normal(8)), Buffer.from_numpy(rng.standard_normal(8)))[0].to_numpy()
    print("  fused64 out dtype/shape:", out.dtype, out.shape, " finite:", np.isfinite(out).all(),
          " rows sum to 1 after exp:", np.allclose(np.exp(out).sum(-1), 1.0))
    blk = _core.Operation._from_cmlir(g._mlir_op).regions[0].front
    for o in blk:
        if type(o) is _core.Operation:
            print("  generic op asm head:", o.asm(skip_regions=True).split("(")[0][:120])
    print("  type(g._mlir_op):", type(g._mlir_op))

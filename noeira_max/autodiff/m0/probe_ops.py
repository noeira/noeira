"""M0 probe: which op classes does a freshly staged graph contain, and can we
append ops after the function returns?"""
import time
import numpy as np
from max import _core
from max._core.dialects import mo, rmo
from max.driver import CPU
from max.dtype import DType
from max.graph import Graph, TensorType, DeviceRef, ops
from max.experimental import compilation
from max.experimental import functional as F
from max.experimental.sharding import TensorLayout
from max.experimental.tensor import Tensor

def block_of(graph):
    op = _core.Operation._from_cmlir(graph._mlir_op)
    return op.regions[0].front

def describe(o):
    cls = type(o)
    props = {}
    for p in ("axis", "transpose_a", "transpose_b", "keep_dims"):
        if hasattr(o, p):
            try: props[p] = getattr(o, p)
            except Exception as e: props[p] = f"<{e}>"
    return f"{cls.__module__}.{cls.__name__:28s} nopnd={len(list(o.operands))} nres={len(list(o.results))} {props}"

print("=== (1) max.experimental + compilation.stage ===")
def f(x, w):
    return F.sum(F.relu(x @ w), axis=-1)
xs = TensorLayout(DType.float32, ["batch", 4], CPU())
ws = TensorLayout(DType.float32, [4, 3], CPU())
staged = compilation.stage(f)(xs, ws)
for o in block_of(staged.graph):
    print("  ", describe(o))

print("=== (2) max.graph, append after f, reuse forward intermediates ===")
dev = DeviceRef.CPU()
with Graph("g", input_types=[TensorType(DType.float64, [5, 4], dev),
                             TensorType(DType.float64, [4, 3], dev)]) as g:
    x, w = g.inputs
    h = ops.matmul(x, w)
    r = ops.relu(h)
    y = ops.sum(r, axis=-1)
    blk = block_of(g)
    fwd = list(blk)
    print("  forward ops:", [type(o).__name__ for o in fwd])
    # emit 'backward' after the fact: dW = x^T @ (1[r>0])
    mask = ops.cast(ops.greater(h, ops.constant(0.0, DType.float64, dev)), DType.float64)
    dw = ops.matmul(ops.transpose(x, -1, -2), mask)
    g.output(y, dw)
print("  all ops   :", [type(o).__name__ for o in block_of(g)])

from max.engine import InferenceSession
sess = InferenceSession(devices=[CPU()])
t0 = time.perf_counter(); model = sess.load(g); t1 = time.perf_counter()
print(f"  compile f64 graph: {t1-t0:.2f}s")
xv = np.random.randn(5, 4); wv = np.random.randn(4, 3)
from max.driver import Buffer
outs = model.execute(Buffer.from_numpy(xv), Buffer.from_numpy(wv))
yv, dwv = [o.to_numpy() for o in outs]
hv = xv @ wv
print("  y ok:", np.allclose(yv, np.maximum(hv, 0).sum(-1)), " dW ok:", np.allclose(dwv, xv.T @ (hv > 0)))

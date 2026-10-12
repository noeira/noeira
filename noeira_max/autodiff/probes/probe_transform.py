"""Probe: a 40-line reverse walk. f(x, w) = sum(relu(x@w) * (x@w)), with
x@w used twice (fan-out), rules keyed by MLIR op NAME, gradients checked in f64."""
import numpy as np
from max import _core
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import Graph, TensorType, DeviceRef, Value, ops
from max.engine import InferenceSession

def block_of(g): return _core.Operation._from_cmlir(g._mlir_op).regions[0].front
def op_name(o):
    head = o.asm(skip_regions=True, assume_verified=True).split("(")[0]
    return head.split("=")[-1].strip().strip('"')

dev = DeviceRef.CPU()
F64 = DType.float64
R = {  # name -> rule(op, operands as TensorValue, g) -> list of cotangents
  "rmo.matmul":        lambda o, a, b, g: [ops.matmul(g, ops.transpose(b, -1, -2)), ops.matmul(ops.transpose(a, -1, -2), g)],
  "rmo.mo.relu":       lambda o, x, g: [g * ops.cast(ops.greater(x, ops.constant(0.0, F64, dev)), F64)],
  "rmo.mul":           lambda o, a, b, g: [g * b, g * a],
  "rmo.mo.reduce.add": lambda o, x, g: [ops.broadcast_to(g, x.shape)],
}
with Graph("vg", input_types=[TensorType(F64, [5, 4], dev), TensorType(F64, [4, 3], dev)]) as gr:
    x, w = gr.inputs
    n0 = len(list(block_of(gr)))
    h = ops.matmul(x, w)
    y = ops.sum(ops.sum(ops.relu(h) * h, axis=-1), axis=0)       # [1, 1]
    fwd = list(block_of(gr))[n0:]
    print("forward op names:", [op_name(o) for o in fwd])
    ct = {y._mlir_value: ops.constant(np.ones((1, 1)), F64, dev)}
    for o in reversed(fwd):
        g = [ct.get(r) for r in o.results]
        if g[0] is None: continue
        name = op_name(o)
        if name not in R: raise NotImplementedError(f"no VJP rule for {name}")
        args = [Value.from_mlir(opnd.value) for opnd in o.operands]
        for opnd, c in zip(o.operands, R[name](o, *args, g[0])):
            k = opnd.value
            ct[k] = c if k not in ct else ct[k] + c                  # ACCUMULATE, never overwrite
    gr.output(y, ct[x._mlir_value], ct[w._mlir_value])

m = InferenceSession(devices=[CPU()]).load(gr)
rng = np.random.default_rng(0); xv = rng.standard_normal((5, 4)); wv = rng.standard_normal((4, 3))
yv, dx, dw = [t.to_numpy() for t in m.execute(Buffer.from_numpy(xv), Buffer.from_numpy(wv))]
hv = xv @ wv; dh = 2 * hv * (hv > 0)                               # d/dh relu(h)*h = 2h[h>0]
print("y  ok:", np.allclose(yv[0, 0], (np.maximum(hv, 0) * hv).sum()))
print("dx ok:", np.allclose(dx, dh @ wv.T), " dw ok:", np.allclose(dw, xv.T @ dh))

"""Probe: op classes of the nanoGPT op set, compile-time warm vs cold,
and whether the eager interpreter runs an rmo graph in float64."""
import time
import numpy as np
from max import _core, _interpreter
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import Graph, TensorType, DeviceRef, ops
from max.engine import InferenceSession

def block_of(graph):
    return _core.Operation._from_cmlir(graph._mlir_op).regions[0].front

def describe(o):
    cls = type(o)
    attrs = {}
    for p in ("axis", "new_shape", "approximate", "num_lower", "num_upper", "keep_dims", "transpose_a", "transpose_b"):
        if p in type(o).__dict__:
            try: attrs[p] = str(getattr(o, p))[:50]
            except Exception as e: attrs[p] = "<err>"
    res = [str(r.type)[:40] for r in o.results]
    return f"{cls.__module__.split('.')[-1]}.{cls.__name__:24s} in={len(list(o.operands))} {attrs} -> {res}"

dev = DeviceRef.CPU()
B, T, C, V = 2, 5, 8, 11
with Graph("gpt_ops", input_types=[TensorType(DType.int64, [B, T], dev),
                                   TensorType(DType.float32, [V, C], dev),
                                   TensorType(DType.float32, [C], dev),
                                   TensorType(DType.float32, [C], dev)]) as g:
    idx, wte, gamma, beta = g.inputs
    x = ops.gather(wte, idx, axis=0)                       # [B,T,C]
    x = ops.layer_norm(x, gamma, beta, epsilon=1e-5)
    q = ops.reshape(x, [B, T, 2, C // 2])
    q = ops.permute(q, [0, 2, 1, 3])                       # [B,2,T,C/2]
    att = ops.matmul(q, ops.transpose(q, -1, -2))          # batched
    att = ops.band_part(att, num_lower=None, num_upper=0)  # causal (keeps lower)
    att = ops.softmax(att)
    y = ops.matmul(att, q)
    y = ops.gelu(y)
    y2 = ops.gelu(y, approximate="tanh")
    s = ops.slice_tensor(y2, [slice(None), slice(0, 1), slice(None), slice(None)])
    c = ops.concat([s, s], axis=1)
    m = ops.mean(c, axis=-1)
    mx = ops.max(c, axis=-1)
    w = ops.where(ops.greater(m, mx), m, mx)
    l = ops.logsoftmax(ops.reshape(c, [B * 2 * T, C // 2]))
    out = ops.sum(ops.sqrt(ops.exp(w) + ops.rsqrt(ops.pow(w * w + 1.0, 2.0))), axis=0)
    g.output(out, l)
for o in block_of(g):
    print("  ", describe(o))

print("=== interpreter on rmo graph (f64) ===")
with Graph("tiny64", input_types=[TensorType(DType.float64, [5, 4], dev),
                                  TensorType(DType.float64, [4, 3], dev)]) as g2:
    x, w = g2.inputs
    y = ops.sum(ops.relu(ops.matmul(x, w)), axis=-1)
    g2.output(y)
try:
    print("  can_execute:", _interpreter.can_execute(g2))
    xv = np.random.randn(5, 4); wv = np.random.randn(4, 3)
    t0 = time.perf_counter()
    outs = _interpreter.execute(g2, [Buffer.from_numpy(xv), Buffer.from_numpy(wv)])
    t1 = time.perf_counter()
    yv = outs[0].to_numpy()
    print(f"  interp ok: shape={yv.shape} match={np.allclose(yv[:, 0], np.maximum(xv @ wv, 0).sum(-1))}  {1e3*(t1-t0):.1f} ms")
except Exception as e:
    print("  interpreter FAILED:", type(e).__name__, str(e)[:300])

print("=== compile times: cold, same graph again, new graph ===")
sess = InferenceSession(devices=[CPU()])
for label, gg in (("tiny64 #1", g2), ("tiny64 #2", g2), ("gpt_ops", g)):
    t0 = time.perf_counter(); sess.load(gg); t1 = time.perf_counter()
    print(f"  {label}: {t1-t0:.2f}s")

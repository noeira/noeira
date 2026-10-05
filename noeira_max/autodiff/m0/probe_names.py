"""M0 probe 6: op-name table for the M1 rule set, Value.owner, constant
decoding, generic-op attributes, and source locations."""
import numpy as np
from max import _core
from max._core.dialects import mo
from max.dtype import DType
from max.graph import Graph, TensorType, DeviceRef, ops

def block_of(g): return _core.Operation._from_cmlir(g._mlir_op).regions[0].front
def op_name(o):
    head = o.asm(skip_regions=True, assume_verified=True).split("(")[0]
    return head.split("=")[-1].strip().strip('"')

dev = DeviceRef.CPU(); F64 = DType.float64
with Graph("names", input_types=[TensorType(F64, ["n", 4], dev), TensorType(F64, [4, 3], dev),
                                 TensorType(DType.int64, ["n"], dev)]) as g:
    x, w, idx = g.inputs
    y = x
    for f in (ops.negate, ops.exp, ops.log, ops.log1p, ops.sqrt, ops.rsqrt, ops.tanh, ops.sigmoid,
              ops.relu, ops.abs, ops.sin, ops.cos, ops.erf, ops.silu, ops.gelu, ops.floor, ops.atanh):
        y = f(y)
    y = ops.gelu(y, approximate="tanh"); y = ops.gelu(y, approximate="quick")
    y = ops.cast(ops.cast(y, DType.float32), F64)
    y = ops.add(y, x); y = ops.sub(y, x); y = ops.mul(y, x); y = ops.div(y, x); y = ops.pow(y, x)
    y = ops.max(y, x); y = ops.min(y, x)
    c = ops.greater(y, x); y = ops.where(c, y, x)
    _ = ops.greater_equal(y, x); _ = ops.equal(y, x); _ = ops.not_equal(y, x)
    s = ops.sum(y, axis=1); m = ops.mean(y, axis=1); mx = ops.max(y, axis=1); mn = ops.min(y, axis=1)
    am = ops.argmax(y, axis=1); pr = ops.prod(y, axis=1)
    z = ops.matmul(y, w)
    t = ops.transpose(z, 0, 1); pm = ops.permute(z, [1, 0]); r = ops.reshape(z, [-1]); b = ops.broadcast_to(ops.sum(z, axis=0), [5, 3])
    sl = ops.slice_tensor(z, [slice(None), slice(0, 2)]); cc = ops.concat([z, z], axis=1)
    sp = ops.split(z, [1, 2], axis=1); ch = ops.chunk(cc, 2, axis=1)
    sq = ops.squeeze(ops.unsqueeze(z, 0), 0); st = ops.stack([z, z], axis=0)
    ga = ops.gather(w, idx, axis=0); sm = ops.softmax(z); ls = ops.logsoftmax(z)
    ln = ops.layer_norm(z, ops.constant(np.ones(3), F64, dev), ops.constant(np.zeros(3), F64, dev), epsilon=1e-5)
    sa = ops.scatter_nd_add(ops.constant(np.zeros((7, 3)), F64, dev), z, ops.unsqueeze(idx, -1))
    g.output(s, m, mx, mn, am, pr, t, pm, r, b, sl, cc, *sp, *ch, sq, st, ga, sm, ls, ln, sa)

seen = {}
for o in block_of(g):
    k = (type(o).__module__.split(".")[-1], type(o).__name__)
    seen.setdefault(op_name(o), k)
for name, (mod, cls) in seen.items():
    print(f"  {name:30s} {mod}.{cls}")

print("=== Value.owner / constant decoding / generic attrs / location ===")
blk = list(block_of(g))
perm_op = next(o for o in blk if op_name(o) == "rmo.mo.transpose")
perm_src = perm_op.operands[1].value.owner
print("  transpose operand[1] owner:", type(perm_src).__name__, op_name(perm_src))
attr = perm_src.value
print("  ConstantOp.value type:", type(attr).__name__)
for how in ("np.asarray", "memoryview", "list"):
    try:
        v = {"np.asarray": lambda a: np.asarray(a), "memoryview": lambda a: np.frombuffer(memoryview(a), dtype=np.int64),
             "list": lambda a: list(a)}[how](attr)
        print(f"   via {how}: {v}")
    except Exception as e:
        print(f"   via {how}: FAILED {type(e).__name__}: {str(e)[:80]}")
print("  attr public members:", [m for m in dir(attr) if not m.startswith("_")][:30])
cc_op = next(o for o in blk if op_name(o) == "rmo.concat")
print("  concat asm:", cc_op.asm(skip_regions=True, assume_verified=True)[:200])
print("  concat discardable:", list(cc_op.discardable_attributes))
import max._mlir.ir as ir
try:
    iop = ir.Operation._CAPICreate(cc_op._CAPIPtr)
    print("  ir bridge name:", iop.name, " attrs:", {a.name: str(a.attr) for a in iop.attributes})
except Exception as e:
    print("  ir bridge FAILED:", type(e).__name__, str(e)[:120])
print("  matmul loc:", next(o for o in blk if op_name(o) == "rmo.matmul").asm(enable_debug_info=True, skip_regions=True, assume_verified=True)[-160:])

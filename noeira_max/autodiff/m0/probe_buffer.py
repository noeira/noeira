"""M0 probe 4: does a compiled step mutate a BufferLayout argument in place,
across calls, with a device-side step counter? And does export_mef work?"""
import os, tempfile, time
import numpy as np
from max.driver import CPU
from max.dtype import DType
from max.experimental import compilation
from max.experimental import functional as F
from max.experimental.sharding import TensorLayout, BufferLayout
from max.experimental.tensor import Tensor

def sgd_step(p, step, g):
    # p, step: buffers the step stores through; g: read-only gradient
    lr = 0.1 / (step + 1.0)            # schedule computed from the device counter
    F.buffer_store(p, p - lr * g)
    F.buffer_store(step, step + 1.0)
    return F.sum(p * p)

p_spec = BufferLayout(DType.float32, [4], CPU())
s_spec = BufferLayout(DType.float32, [1], CPU())
g_spec = TensorLayout(DType.float32, [4], CPU())
t0 = time.perf_counter()
run = compilation.compile(sgd_step)(p_spec, s_spec, g_spec)
print(f"compile {time.perf_counter()-t0:.2f}s")
p = Tensor.ones([4], dtype=DType.float32, device=CPU())
s = Tensor.zeros([1], dtype=DType.float32, device=CPU())
g = Tensor.ones([4], dtype=DType.float32, device=CPU())
expect_p, expect_s = np.ones(4, np.float32), 0.0
for i in range(3):
    loss = run(p, s, g)
    lr = 0.1 / (expect_s + 1.0); expect_p = expect_p - lr; expect_s += 1.0
    print(f"  call {i}: p={p.to_numpy()} s={s.to_numpy()} loss={loss.to_numpy()}  expect p={expect_p} s={expect_s}")
path = os.path.join(tempfile.mkdtemp(), "sgd_step.mef")
run.export_mef(path)
print("export_mef:", os.path.getsize(path), "bytes")

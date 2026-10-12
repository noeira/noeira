from max.driver import CPU
from max.dtype import DType
from max.experimental.nn import Module, module_dataclass
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType

@module_dataclass
class VectorAdd(Module[[Tensor, Tensor], Tensor]):
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return a + b

dev = CPU()
m = VectorAdd().to(dev)
t = TensorType(dtype=DType.float32, shape=("n",), device=DeviceRef.from_device(dev))
m.compile(t, t).export_mef("graph.mef")
print("exported")

"""`TapeAdam` — Adam on a GPU tape's own parameters, launched through the
tape's `Dev` (`backend.mojo`), so it runs on a metal_rt tape where nn's
optimizers cannot (they launch through MAX).

It is nn's per-parameter Adam, not a re-derivation: the same host step state
(`t`, the running powers β₁ᵗ, β₂ᵗ, `bc = 1 - βᵗ`), the same walk (parameters
in NAME order, decay on matrices only) and the same kernel,
`nn.optimizer.adam._adam_update_kernel_rt`, one launch per parameter. On a
MAX tape it is bit-identical to an un-adopted `nn.optimizer.Adam`. (nn's
ADOPTED Adam computes `bc` on the device from a device-resident power; the
result can differ in the last bit.)

Modules as ops are not supported (a metal_rt tape refuses them anyway).
"""

from noeira.nn.constants import DT, TPB
from noeira.nn.optimizer.adam import _adam_update_kernel_rt
from .backend import ensure_gpu, fill_dev
from .ftensor import Tape


struct TapeAdam(Movable):
    var lr: Scalar[DT]
    var beta1: Scalar[DT]
    var beta2: Scalar[DT]
    var eps: Scalar[DT]
    var wd: Scalar[DT]
    var t: Int
    var _b1_pow: Scalar[DT]
    var _b2_pow: Scalar[DT]
    var bc1: Scalar[DT]
    var bc2: Scalar[DT]

    def __init__(
        out self,
        lr: Scalar[DT],
        beta1: Scalar[DT] = 0.9,
        beta2: Scalar[DT] = 0.999,
        eps: Scalar[DT] = 1e-8,
        wd: Scalar[DT] = 0.0,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = wd
        self.t = 0
        self._b1_pow = 1.0
        self._b2_pow = 1.0
        self.bc1 = 1.0
        self.bc2 = 1.0

    def step(mut self, tape: Tape) raises:
        ref d = tape.data[]
        if not d.ctx:
            raise Error("TapeAdam: a GPU tape (use nn's Adam on the CPU)")
        if len(d.mods) > 0:
            raise Error("TapeAdam: the tape owns modules; use nn's Adam")
        # nn's `Adam.begin_step`, verbatim.
        self.t += 1
        self._b1_pow = self._b1_pow * self.beta1
        self._b2_pow = self._b2_pow * self.beta2
        self.bc1 = Scalar[DT](1.0) - self._b1_pow
        self.bc2 = Scalar[DT](1.0) - self._b2_pow
        var c = d.dctx()
        var order = d._walk_order()
        for e in range(len(order)):
            var i = order[e]
            var n = d.nodes[i].size()
            if d.nodes[i].grad.n == 0:
                continue
            if not d.nodes[i].m.dev:
                ensure_gpu(d.nodes[i].m, c, n)
                fill_dev(d.nodes[i].m.dev.value(), n, Scalar[DT](0), c)
                ensure_gpu(d.nodes[i].v, c, n)
                fill_dev(d.nodes[i].v.dev.value(), n, Scalar[DT](0), c)
            var apply_decay = d.nodes[i].rows > 1
            c.enqueue_function[_adam_update_kernel_rt](
                d.nodes[i].val.dev.value(),
                d.nodes[i].grad.dev.value(),
                d.nodes[i].m.dev.value(),
                d.nodes[i].v.dev.value(),
                Int64(n),
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                self.bc1,
                self.bc2,
                self.wd,
                Int64(apply_decay),
                grid_dim=(n + TPB - 1) // TPB,
                block_dim=TPB,
            )
            d.nodes[i].val.version += 1

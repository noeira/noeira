"""ElementOp implementations consumed by `Elementwise[DIM, OP]`.

Each `*_op.mojo` defines one struct implementing `ElementOp` — pure
math, no state. Aliasing primitives (e.g. `Tanh[DIM] = Elementwise[DIM,
TanhOp]`) live in `primitives/*.mojo` alongside their old hand-written
forms during the Phase 2 migration; once all consumers move to the
alias, the hand-written modules are deleted.
"""

from .tanh_op import TanhOp
from .relu_op import ReLUOp
from .symlog_op import SymlogOp
from .swish_op import SwishOp
from .mish_op import MishOp
from .sigmoid_op import SigmoidOp
from .gelu_tanh_op import GELUTanhOp
from .gelu_op import GELUOp
from .stop_grad_op import StopGradOp
from .sum_op import SumOp
from .mean_op import MeanOp

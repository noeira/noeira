"""`F` — the named-function spelling of the functional mode, for code that
reads better as calls than as operators (`import noeira.nn.functional.F as F`).

Each is the composition its body shows; none is a new op. `linear` is
`x @ w + b` because that IS `Linear`'s CPU forward: one cblas sgemm, then
the bias row added — the same arithmetic, so the same bits.
"""

from .ftensor import FTensor
from .opcodes import (
    EW_NONE, EW_RELU, EW_TANH, EW_SIGMOID, EW_GELU, EW_GELU_TANH, EW_MISH,
    EW_SWISH, EW_ELU,
)


def linear(x: FTensor, w: FTensor, b: FTensor) raises -> FTensor:
    """`x[B, IN] @ w[IN, OUT] + b[1, OUT]` — `Linear`'s weight layout. ONE
    fused node (`tape.linear`); `x @ w + b` is the same value as two."""
    return x._new(x.tape[].linear(x.id, w.id, b.id, EW_NONE))


def linear(x: FTensor, w: FTensor, b: FTensor, act: StaticString) raises -> FTensor:
    """`act(x @ w + b)` as ONE fused node — `LinearAct`'s launches. `act`:
    "relu", "tanh", "sigmoid", "gelu", "gelu_tanh", "mish", "swish", "elu"."""
    var tag: Int
    if act == "relu":
        tag = EW_RELU
    elif act == "tanh":
        tag = EW_TANH
    elif act == "sigmoid":
        tag = EW_SIGMOID
    elif act == "gelu":
        tag = EW_GELU
    elif act == "gelu_tanh":
        tag = EW_GELU_TANH
    elif act == "mish":
        tag = EW_MISH
    elif act == "swish":
        tag = EW_SWISH
    elif act == "elu":
        tag = EW_ELU
    else:
        raise Error(String("F.linear: unknown activation ", act))
    return x._new(x.tape[].linear(x.id, w.id, b.id, tag))


def mse(pred: FTensor, target: FTensor) raises -> FTensor:
    """Mean of the squared error over every element (`mse_forward`)."""
    return ((pred - target) ** 2).mean()


def relu(x: FTensor) raises -> FTensor:
    return x.relu()


def tanh(x: FTensor) raises -> FTensor:
    return x.tanh()


def sigmoid(x: FTensor) raises -> FTensor:
    return x.sigmoid()


def gelu(x: FTensor) raises -> FTensor:
    return x.gelu()

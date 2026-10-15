"""`F` — the named-function spelling of the functional mode, for code that
reads better as calls than as operators (`import noeira.nn.functional.F as F`).

Each is the composition its body shows; none is a new op. `linear` is
`x @ w + b` because that IS `Linear`'s CPU forward: one cblas sgemm, then
the bias row added — the same arithmetic, so the same bits.
"""

from .ftensor import FTensor


def linear(x: FTensor, w: FTensor, b: FTensor) raises -> FTensor:
    """`x[B, IN] @ w[IN, OUT] + b[1, OUT]` — `Linear`'s weight layout."""
    return x @ w + b


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

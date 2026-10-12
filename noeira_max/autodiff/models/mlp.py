"""The MLP of the nn examples' MNIST twin: 784-256-128-10, ReLU."""

from __future__ import annotations

import numpy as np
from max.graph import TensorValue, ops

from .common import cross_entropy, dense

DIMS = (784, 256, 128, 10)


def init(rng: np.random.Generator, dims=DIMS, dtype=np.float64) -> dict:
    """He-uniform weights (``U(-b, b)``, ``b = sqrt(6 / fan_in)``), zero
    biases: the twin's Kaiming init."""
    params = {}
    for i, (fan_in, fan_out) in enumerate(zip(dims[:-1], dims[1:])):
        bound = np.sqrt(6.0 / fan_in)
        params[f"w{i}"] = rng.uniform(-bound, bound, (fan_in, fan_out)).astype(dtype)
        params[f"b{i}"] = np.zeros(fan_out, dtype)
    return params


def forward(p: dict, x: TensorValue, depth: int = len(DIMS) - 1) -> TensorValue:
    h = x
    for i in range(depth):
        h = dense(h, p[f"w{i}"], p[f"b{i}"])
        if i < depth - 1:
            h = ops.relu(h)
    return h


def loss(p: dict, x: TensorValue, y: TensorValue) -> TensorValue:
    return cross_entropy(forward(p, x), y)

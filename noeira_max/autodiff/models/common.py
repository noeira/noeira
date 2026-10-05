"""Pieces shared by the models: a dense layer and the cross-entropy loss."""

from __future__ import annotations

import numpy as np
from max.dtype import DType
from max.graph import TensorValue, ops


def dense(x: TensorValue, w: TensorValue, b: TensorValue) -> TensorValue:
    return ops.matmul(x, w) + b


def cross_entropy(logits: TensorValue, targets: TensorValue) -> TensorValue:
    """Mean cross-entropy over every leading position, as ``[1, 1]``.

    ``logits`` is ``[..., classes]``, ``targets`` the integer labels
    ``[...]``. The one-hot is built in the graph from the labels; a fused
    loss would be RFC K6.
    """
    classes = int(logits.shape[-1])
    flat = ops.reshape(logits, [-1, classes])
    labels = ops.reshape(targets, [-1, 1])
    choices = ops.constant(np.arange(classes), DType.int64, targets.device)
    one_hot = ops.cast(ops.equal(labels, choices), logits.dtype)
    picked = ops.sum(ops.logsoftmax(flat) * one_hot, axis=1)
    return -ops.mean(picked, axis=0)

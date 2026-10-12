"""Pieces shared by the models: a dense layer, the cross-entropy loss, and
the kernel-backed layer norm and attention."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from max.dtype import DType
from max.graph import Graph, TensorType, TensorValue, ops

from .. import _graph

KERNELS = Path(__file__).resolve().parent.parent / "kernels"
"""The Mojo custom-op package (``kernels/``)."""


def dense(x: TensorValue, w: TensorValue, b: TensorValue) -> TensorValue:
    return ops.matmul(x, w) + b


def cross_entropy(logits: TensorValue, targets: TensorValue) -> TensorValue:
    """Mean cross-entropy over every leading position, as ``[1, 1]``.

    ``logits`` is ``[..., classes]``, ``targets`` the integer labels
    ``[...]``. The one-hot is built in the graph from the labels; MAX has no
    fused cross-entropy kernel.
    """
    classes = int(logits.shape[-1])
    flat = ops.reshape(logits, [-1, classes])
    labels = ops.reshape(targets, [-1, 1])
    choices = ops.constant(np.arange(classes), DType.int64, targets.device)
    one_hot = ops.cast(ops.equal(labels, choices), logits.dtype)
    picked = ops.sum(ops.logsoftmax(flat) * one_hot, axis=1)
    return -ops.mean(picked, axis=0)


def layer_norm_kernel(x: TensorValue, gamma: TensorValue, beta: TensorValue,
                      eps: float) -> TensorValue:
    """``ops.layer_norm`` over the last axis, as the Mojo custom op
    ``noeira_layer_norm_fwd``. It also returns the mean and the reciprocal
    standard deviation of each row, which the op's VJP rule
    (``rules/custom.py``) hands to ``noeira_layer_norm_bwd``.
    """
    _graph.import_kernels(Graph.current, KERNELS)
    d = x.shape[-1]
    rows = ops.reshape(x, [-1, d])
    n = rows.shape[0]
    eps_value = ops.constant(np.array([eps], x.dtype.to_numpy()), x.dtype, x.device)
    y, _, _ = ops.custom(
        "noeira_layer_norm_fwd",
        device=x.device,
        values=[rows, gamma, beta, eps_value],
        out_types=[
            TensorType(x.dtype, [n, d], x.device),
            TensorType(x.dtype, [n, 1], x.device),
            TensorType(x.dtype, [n, 1], x.device),
        ],
    )
    return ops.reshape(y.tensor, x.shape)


def attention_kernel(qkv: TensorValue, heads: int) -> TensorValue:
    """Causal attention over ``qkv [B, T, 3·C]`` (the QKV projection's own
    output), as the Mojo custom op ``noeira_attention_fwd``: ``[B, T, C]``,
    heads merged. It also returns each query row's log-sum-exp, which the
    op's VJP rule (``rules/custom.py``) hands to ``noeira_attention_bwd``.
    Every size must be static: noeira's kernels take them as parameters.
    """
    _graph.import_kernels(Graph.current, KERNELS)
    b, t, c3 = (int(d) for d in qkv.shape)
    c = c3 // 3
    if qkv.device.is_gpu() and (c // heads) % 32:
        # Caught here, the graph compiler would report it as a MAX bug.
        raise ValueError(f"noeira's GPU attention kernels need a head dim that is a multiple "
                         f"of 32 (16 on Apple GPUs), got {c // heads}")
    o, _ = ops.custom(
        "noeira_attention_fwd",
        device=qkv.device,
        values=[qkv],
        out_types=[
            TensorType(qkv.dtype, [b, t, c], qkv.device),
            TensorType(qkv.dtype, [b * heads * t], qkv.device),
        ],
        parameters={"B": b, "H": heads, "S": t, "HD": c // heads},
    )
    return o.tensor

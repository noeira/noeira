"""A character GPT with the block of the torch twin
(``tools/nn/torch_nn_reference.py``): pre-LN, causal multi-head attention,
tanh GELU, learned positions, the head tied to the token embedding, and
dropout where the twin has it (the embedding sum, each attention projection
and each MLP output).

Dropout draws random numbers, so a graph using it must set a seed
(``build_train_step(..., uses_seed=True)`` does, from a seed buffer).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from max.dtype import DType
from max.graph import TensorType, TensorValue, ops

from .common import cross_entropy, dense


@dataclass(frozen=True)
class Config:
    vocab: int = 65
    seq: int = 16
    dim: int = 32
    heads: int = 2
    layers: int = 2
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads


def init(cfg: Config, rng: np.random.Generator, dtype=np.float64) -> dict:
    """``N(0, 0.02)`` for every matrix (and the positions), the residual
    projections scaled by ``1/sqrt(2 * layers)``, LayerNorm ``(1, 0)``,
    zero biases."""
    d, f = cfg.dim, 4 * cfg.dim
    residual = 1.0 / math.sqrt(2 * cfg.layers)

    def normal(*shape, scale=1.0):
        return (rng.standard_normal(shape) * 0.02 * scale).astype(dtype)

    p = {"wte": normal(cfg.vocab, d), "wpe": normal(cfg.seq, d)}
    for l in range(cfg.layers):
        p |= {
            f"h{l}.ln1.w": np.ones(d, dtype), f"h{l}.ln1.b": np.zeros(d, dtype),
            f"h{l}.qkv.w": normal(d, 3 * d), f"h{l}.qkv.b": np.zeros(3 * d, dtype),
            f"h{l}.proj.w": normal(d, d, scale=residual),
            f"h{l}.proj.b": np.zeros(d, dtype),
            f"h{l}.ln2.w": np.ones(d, dtype), f"h{l}.ln2.b": np.zeros(d, dtype),
            f"h{l}.fc1.w": normal(d, f), f"h{l}.fc1.b": np.zeros(f, dtype),
            f"h{l}.fc2.w": normal(f, d, scale=residual),
            f"h{l}.fc2.b": np.zeros(d, dtype),
        }
    p |= {"lnf.w": np.ones(d, dtype), "lnf.b": np.zeros(d, dtype)}
    return p


def dropout(x: TensorValue, p: float) -> TensorValue:
    """Inverted dropout: keeps each entry with probability ``1 - p``."""
    if p == 0.0:
        return x
    draw = ops.random.uniform(TensorType(DType.float32, x.shape, x.device))
    keep = ops.cast(ops.greater_equal(draw, p), x.dtype)
    return x * keep * (1.0 / (1.0 - p))


def sample_batch(
    corpus: TensorValue, batch: int, seq: int
) -> tuple[TensorValue, TensorValue]:
    """Random windows of a device-resident ``int64`` corpus, drawn in the
    graph: the inputs ``[batch, seq]`` and their next-token targets. A step
    that samples its own batch has the same inputs every call."""
    n = int(corpus.shape[0])
    draw = ops.random.uniform(
        TensorType(DType.float32, [batch, 1], corpus.device),
        range=(0.0, float(n - seq - 1)),
    )
    starts = ops.cast(ops.floor(draw), DType.int64)
    offsets = ops.constant(np.arange(seq + 1)[None, :], DType.int64, corpus.device)
    window = ops.gather(corpus, starts + offsets, axis=0)
    inputs = ops.slice_tensor(window, [slice(None), slice(0, seq)])
    targets = ops.slice_tensor(window, [slice(None), slice(1, seq + 1)])
    return inputs, targets


def _heads(x: TensorValue, cfg: Config) -> TensorValue:
    """``[B, T, C]`` -> ``[B, H, T, D]``."""
    b, t = x.shape[0], x.shape[1]
    return ops.permute(ops.reshape(x, [b, t, cfg.heads, cfg.head_dim]), [0, 2, 1, 3])


def _merge(x: TensorValue, cfg: Config) -> TensorValue:
    """``[B, H, T, D]`` -> ``[B, T, C]``."""
    x = ops.permute(x, [0, 2, 1, 3])
    return ops.reshape(x, [x.shape[0], x.shape[1], cfg.dim])


def _attention(p: dict, l: int, x: TensorValue, cfg: Config) -> TensorValue:
    qkv = dense(x, p[f"h{l}.qkv.w"], p[f"h{l}.qkv.b"])
    q, k, v = (_heads(t, cfg) for t in ops.split(qkv, [cfg.dim] * 3, axis=2))
    scores = ops.matmul(q, ops.transpose(k, -1, -2)) * (1.0 / math.sqrt(cfg.head_dim))
    # The causal mask is an additive -inf bias, NOT `where(mask, scores,
    # -inf)`: MAX 26.6 on CPU returns all NaN for that select form once the
    # batch dims exceed 1 (tests/test_max_findings.py).
    t = int(x.shape[1])
    causal = np.where(np.tril(np.ones((t, t), dtype=bool)), 0.0, -np.inf)
    bias = ops.constant(causal.astype(x.dtype.to_numpy()), x.dtype, x.device)
    weights = ops.softmax(scores + bias)
    y = _merge(ops.matmul(weights, v), cfg)
    return dropout(dense(y, p[f"h{l}.proj.w"], p[f"h{l}.proj.b"]), cfg.dropout)


def _mlp(p: dict, l: int, x: TensorValue, cfg: Config) -> TensorValue:
    h = ops.gelu(dense(x, p[f"h{l}.fc1.w"], p[f"h{l}.fc1.b"]), approximate="tanh")
    return dropout(dense(h, p[f"h{l}.fc2.w"], p[f"h{l}.fc2.b"]), cfg.dropout)


def forward(p: dict, idx: TensorValue, cfg: Config) -> TensorValue:
    """Token indices ``[B, T]`` -> logits ``[B, T, vocab]``."""
    x = dropout(ops.gather(p["wte"], idx, axis=0) + p["wpe"], cfg.dropout)
    for l in range(cfg.layers):
        x = x + _attention(
            p, l, ops.layer_norm(x, p[f"h{l}.ln1.w"], p[f"h{l}.ln1.b"], 1e-5), cfg
        )
        x = x + _mlp(
            p, l, ops.layer_norm(x, p[f"h{l}.ln2.w"], p[f"h{l}.ln2.b"], 1e-5), cfg
        )
    x = ops.layer_norm(x, p["lnf.w"], p["lnf.b"], 1e-5)
    return ops.matmul(x, ops.transpose(p["wte"], 0, 1))  # tied head


def loss(p: dict, idx: TensorValue, targets: TensorValue, cfg: Config) -> TensorValue:
    return cross_entropy(forward(p, idx, cfg), targets)


def decays(name: str, shape: tuple) -> bool:
    """The twin's AdamW groups: decay every matrix except the positions."""
    return len(shape) >= 2 and name != "wpe"

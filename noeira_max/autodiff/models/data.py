"""Fixed batches from the datasets the nn examples cache in
``~/.cache/noeira`` (``NOEIRA_CACHE`` overrides it), read with numpy only."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

CACHE = Path(os.environ.get("NOEIRA_CACHE", "~/.cache/noeira")).expanduser()


def mnist(n_batches: int, batch: int, dtype=np.float64) -> list[tuple[np.ndarray, np.ndarray]]:
    """The first ``n_batches * batch`` training images, scaled to [0, 1]."""
    root = CACHE / "mnist"
    images = np.frombuffer((root / "train-images-idx3-ubyte").read_bytes(), np.uint8, offset=16)
    labels = np.frombuffer((root / "train-labels-idx1-ubyte").read_bytes(), np.uint8, offset=8)
    n = n_batches * batch
    x = images[: n * 784].reshape(n, 784).astype(dtype) / 255.0
    y = labels[:n].astype(np.int64)
    return [(x[i : i + batch], y[i : i + batch]) for i in range(0, n, batch)]


def shakespeare_vocab() -> tuple[str, np.ndarray]:
    text = (CACHE / "tinyshakespeare" / "input.txt").read_text()
    chars = sorted(set(text))
    lookup = np.zeros(max(map(ord, chars)) + 1, np.int64)
    lookup[[ord(c) for c in chars]] = np.arange(len(chars))
    return text, lookup[np.frombuffer(text.encode("latin-1"), np.uint8)]


def shakespeare(
    n_batches: int, batch: int, seq: int, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Random windows of the encoded text: inputs and next-character targets."""
    _, tokens = shakespeare_vocab()
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_batches):
        starts = rng.integers(0, len(tokens) - seq - 1, batch)
        window = np.stack([tokens[s : s + seq + 1] for s in starts])
        out.append((window[:, :-1].copy(), window[:, 1:].copy()))
    return out

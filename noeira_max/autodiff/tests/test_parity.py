"""Model-level parity with PyTorch: the MLP and a tiny GPT,
from identical initial weights, 50 SGD steps on fixed batches, float64 on
CPU. Loss curves and final parameters are compared.

The MAX side is one compiled graph per step function: forward, backward
through ``value_and_grad``, and the SGD update, parameters in and out.

    noeira_max/autodiff/run.sh -m unittest noeira_max.autodiff.tests.test_parity -v
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType

from noeira_max.autodiff import value_and_grad
from noeira_max.autodiff.models import data, gpt, mlp

from .test_golden import TORCH_PYTHON

TORCH_SIDE = Path(__file__).with_name("parity_torch.py")
STEPS = 50
# Agreement is limited by float64 rounding compounding over 50 steps.
LOSS_RTOL = 1e-9
PARAM_RTOL = 1e-8


def sgd_step_graph(loss_fn, init: dict, data_types: list[TensorType], lr: float):
    """``(params, *data) -> (loss, params - lr * grads)`` as one graph."""
    names = list(init)
    dev = DeviceRef.CPU()
    param_types = [TensorType(DType.float64, init[n].shape, dev) for n in names]
    with Graph("sgd_step", input_types=param_types + data_types) as graph:
        params = dict(zip(names, graph.inputs[: len(names)]))
        batch = graph.inputs[len(names):]
        loss, grads = value_and_grad(loss_fn)(params, *batch)
        graph.output(loss, *(params[n] - lr * grads[n] for n in names))
    return graph


def train(graph, init: dict, batches, steps: int):
    start = time.perf_counter()
    model = InferenceSession(devices=[CPU()]).load(graph)
    compile_seconds = time.perf_counter() - start
    names = list(init)
    params = [init[n] for n in names]
    losses = []
    for s in range(steps):
        x, y = batches[s % len(batches)]
        out = model.execute(*map(Buffer.from_numpy, [*params, x, y]))
        losses.append(out[0].to_numpy().item())
        params = [b.to_numpy() for b in out[1:]]
    return losses, dict(zip(names, params)), compile_seconds


def torch_parity(
    model: str, init, batches, losses, final, steps: int, lr: float,
    config=(), **fields,
) -> dict:
    """Retrains ``model`` in torch (``parity_torch.py``, act-ref env) from
    ``init`` on ``batches`` and compares with our ``losses`` and ``final``
    parameters. ``fields`` carry the optimizer's settings."""
    with tempfile.TemporaryDirectory() as tmp:
        npz, report = Path(tmp) / "parity.npz", Path(tmp) / "torch.json"
        arrays = {f"init.{n}": v for n, v in init.items()}
        arrays |= {f"final.{n}": v for n, v in final.items()}
        for s in range(steps):
            arrays[f"x{s}"], arrays[f"y{s}"] = batches[s % len(batches)]
        np.savez(
            npz, names=np.array(list(init)), steps=steps, lr=lr,
            losses=np.array(losses), config=np.array(config), **fields, **arrays,
        )
        env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
        proc = subprocess.run(
            [str(TORCH_PYTHON), str(TORCH_SIDE), model, str(npz), str(report)],
            env=env, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"torch side failed:\n{proc.stderr[-3000:]}")
        return json.loads(report.read_text())


def report(tag: str, losses, result) -> tuple[float, float]:
    worst_loss = max(result["loss_rel_diff"])
    worst_param = max(result["param_rel_diff"].values())
    print(
        f"\n[{tag}] loss {losses[0]:.6f} -> {losses[-1]:.6f} "
        f"(torch {result['torch_losses'][-1]:.6f}); worst loss rel. diff "
        f"{worst_loss:.1e}, worst parameter rel. diff {worst_param:.1e}",
        flush=True,
    )
    return worst_loss, worst_param


@unittest.skipUnless(TORCH_PYTHON.exists(), f"no act-ref env at {TORCH_PYTHON}")
class ParityTest(unittest.TestCase):
    def _compare(self, model: str, init, batches, losses, final, lr, config=()):
        result = torch_parity(model, init, batches, losses, final, STEPS, lr, config)
        worst_loss, worst_param = report(model, losses, result)
        self.assertLess(losses[-1], losses[0], "the model did not train")
        self.assertLess(worst_loss, LOSS_RTOL)
        self.assertLess(worst_param, PARAM_RTOL)

    def test_mlp(self):
        init = mlp.init(np.random.default_rng(0))
        batches = data.mnist(STEPS, 64)
        dev = DeviceRef.CPU()
        graph = sgd_step_graph(
            mlp.loss, init,
            [TensorType(DType.float64, [64, 784], dev), TensorType(DType.int64, [64], dev)],
            lr=0.1,
        )
        losses, final, seconds = train(graph, init, batches, STEPS)
        print(f"\n[mlp] train-step graph compiled in {seconds:.1f} s", flush=True)
        self._compare("mlp", init, batches, losses, final, lr=0.1)

    def test_gpt(self):
        cfg = gpt.Config()
        init = gpt.init(cfg, np.random.default_rng(0))
        batches = data.shakespeare(STEPS, 4, cfg.seq)
        dev = DeviceRef.CPU()
        tokens = TensorType(DType.int64, [4, cfg.seq], dev)
        graph = sgd_step_graph(
            lambda p, x, y: gpt.loss(p, x, y, cfg), init, [tokens, tokens], lr=0.3
        )
        losses, final, seconds = train(graph, init, batches, STEPS)
        print(f"\n[gpt] train-step graph compiled in {seconds:.1f} s", flush=True)
        config = (cfg.vocab, cfg.seq, cfg.dim, cfg.heads, cfg.layers)
        self._compare("gpt", init, batches, losses, final, lr=0.3, config=config)


if __name__ == "__main__":
    unittest.main()

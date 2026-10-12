"""The whole train step as one compiled graph: parameters,
moments and the step counter are buffers updated in place; the schedule,
bias correction and clipping run on the device.

    noeira_max/autodiff/run.sh -m unittest noeira_max.autodiff.tests.test_train_step -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef, TensorType

from noeira_max.autodiff.models import data, gpt, mlp
from noeira_max.autodiff.optim import SGD, AdamW, WarmupCosine
from noeira_max.autodiff.train import SEED_STRIDE, CompiledStep, build_train_step, load_mef

from .test_golden import TORCH_PYTHON
from .test_parity import report, torch_parity

DEV = DeviceRef.CPU()
STEPS = 30
LOSS_RTOL = 1e-9
PARAM_RTOL = 1e-8


def _run(step, init, opt, batches, steps, **kwargs):
    """Runs ``steps`` steps; returns losses, diagnostics and the compiled step."""
    compiled = CompiledStep(step, init, opt, CPU(), **kwargs)
    losses, diags = [], []
    for s in range(steps):
        out = compiled(*(Buffer.from_numpy(a) for a in batches[s % len(batches)]))
        losses.append(out[0].to_numpy().item())
        diags.append([o.to_numpy().item() for o in out[1:]])
    return losses, diags, compiled


def _adamw_fields(opt: AdamW, names, shapes) -> dict:
    sched = opt.schedule
    return dict(
        optim="adamw", betas=np.array(opt.betas), eps=opt.eps,
        weight_decay=opt.weight_decay,
        decay=np.array([opt.decay(n, s) for n, s in zip(names, shapes)]),
        clip=opt.clip_norm or -1.0, warmup=sched.warmup, total=sched.total,
        min_scale=sched.min_scale,
    )


@unittest.skipUnless(TORCH_PYTHON.exists(), f"no act-ref env at {TORCH_PYTHON}")
class AdamWParityTest(unittest.TestCase):
    """The twin's recipe (AdamW groups, warmup + cosine, clip 1) against
    ``torch.optim.AdamW`` + ``clip_grad_norm_``, float64, 30 steps."""

    def _check(self, tag, model, loss_fn, init, opt, data_types, batches, config=()):
        step = build_train_step(loss_fn, init, opt, data_types, DEV, diagnostics=True)
        losses, diags, compiled = _run(step, init, opt, batches, STEPS)
        print(f"\n[{tag}] train step compiled in {compiled.compile_seconds:.1f} s", flush=True)
        names, shapes = list(init), [v.shape for v in init.values()]
        result = torch_parity(
            model, init, batches, losses, compiled.host_params(), STEPS, opt.lr,
            config, **_adamw_fields(opt, names, shapes),
        )
        worst_loss, worst_param = report(tag, losses, result)
        self.assertLess(worst_loss, LOSS_RTOL)
        self.assertLess(worst_param, PARAM_RTOL)

        # The schedule ran on the device, and clipping really clipped.
        grad_norm, lr = zip(*diags)  # diagnostics are sorted by name
        expected = [opt.lr * opt.schedule.reference(s) for s in range(STEPS)]
        np.testing.assert_allclose(lr, expected, rtol=1e-12)
        self.assertGreater(max(grad_norm), opt.clip_norm, "clipping never triggered")
        self.assertEqual(compiled.host_state()["step"].item(), STEPS)

    def test_mlp(self):
        init = mlp.init(np.random.default_rng(0))
        opt = AdamW(
            lr=1e-3, schedule=WarmupCosine(5, STEPS, 0.1), clip_norm=1.0,
            betas=(0.9, 0.99), weight_decay=0.1,
            decay=lambda name, shape: len(shape) >= 2,
        )
        types = [TensorType(DType.float64, [64, 784], DEV), TensorType(DType.int64, [64], DEV)]
        self._check("mlp adamw", "mlp", mlp.loss, init, opt, types, data.mnist(STEPS, 64))

    def test_gpt(self):
        cfg = gpt.Config()
        init = gpt.init(cfg, np.random.default_rng(0))
        opt = AdamW(
            lr=3e-3, schedule=WarmupCosine(5, STEPS, 0.1), clip_norm=1.0,
            betas=(0.9, 0.99), weight_decay=0.1, decay=gpt.decays,
        )
        tokens = TensorType(DType.int64, [4, cfg.seq], DEV)
        config = (cfg.vocab, cfg.seq, cfg.dim, cfg.heads, cfg.layers)
        self._check(
            "gpt adamw", "gpt", lambda p, x, y: gpt.loss(p, x, y, cfg), init, opt,
            [tokens, tokens], data.shakespeare(STEPS, 4, cfg.seq), config,
        )


class TrainStepTest(unittest.TestCase):
    def test_mef_round_trip_keeps_in_place_updates(self):
        """A step exported to a MEF and read back trains bit-identically."""
        init = mlp.init(np.random.default_rng(1), dims=(784, 32, 10))
        opt = SGD(lr=0.1)
        types = [TensorType(DType.float64, [16, 784], DEV), TensorType(DType.int64, [16], DEV)]
        loss = lambda p, x, y: mlp.cross_entropy(mlp.forward(p, x, depth=2), y)  # noqa: E731
        step = build_train_step(loss, init, opt, types, DEV, name="mef_step")
        batches = data.mnist(3, 16)
        losses, _, compiled = _run(step, init, opt, batches, 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step.mef"
            compiled.export_mef(path)
            reloaded = load_mef(path, step, init, opt, CPU())
            again = [
                reloaded(*map(Buffer.from_numpy, b))[0].to_numpy().item() for b in batches
            ]
        self.assertEqual(again, losses)
        self.assertLess(losses[-1], losses[0])
        for name, value in compiled.host_params().items():
            np.testing.assert_array_equal(reloaded.host_params()[name], value)
        self.assertEqual(reloaded.host_state()["step"].item(), 3)

    def test_experimental_api_train_step(self):
        """A train step written with ``max.experimental``: the
        parameters and optimizer state are ``BufferLayout`` specs given to
        ``compilation.compile``, the step is traced over ``max.experimental``
        tensors, and it stores in place. It must match ``build_train_step``
        (the ``max.graph`` path) exactly."""
        from max.experimental import compilation
        from max.experimental.sharding import BufferLayout, TensorLayout
        from max.experimental.tensor import Tensor

        from noeira_max.autodiff import value_and_grad

        init = mlp.init(np.random.default_rng(2), dims=(784, 32, 10))
        opt = AdamW(lr=1e-2, betas=(0.9, 0.99), weight_decay=0.1)
        loss_fn = lambda p, x, y: mlp.cross_entropy(mlp.forward(p, x, depth=2), y)  # noqa: E731

        def train_step(params, opt_state, x, y):
            loss, grads = value_and_grad(loss_fn)(params, x, y)
            opt.apply(params, opt_state, grads)
            return loss

        def buffers(arrays):
            return {n: BufferLayout(DType.float64, a.shape, CPU()) for n, a in arrays.items()}

        state0 = opt.init(init)
        step = compilation.compile(train_step)(
            buffers(init), buffers(state0),
            TensorLayout(DType.float64, [16, 784], CPU()), TensorLayout(DType.int64, [16], CPU()),
        )
        params = {n: Tensor.from_dlpack(a.copy()) for n, a in init.items()}
        state = {n: Tensor.from_dlpack(a.copy()) for n, a in state0.items()}
        batches = data.mnist(3, 16)
        losses = [
            step(params, state, Tensor.from_dlpack(x), Tensor.from_dlpack(y)).to_numpy().item()
            for x, y in batches
        ]

        types = [TensorType(DType.float64, [16, 784], DEV), TensorType(DType.int64, [16], DEV)]
        reference = build_train_step(loss_fn, init, opt, types, DEV, name="rfc_reference")
        expected, _, compiled = _run(reference, init, opt, batches, 3)
        np.testing.assert_allclose(losses, expected, rtol=1e-12)
        for name, value in compiled.host_params().items():
            np.testing.assert_allclose(params[name].to_numpy(), value, rtol=1e-12)
        self.assertEqual(state["step"].to_numpy().item(), 3)

    def test_dropout_draws_new_masks_every_step(self):
        """The seed is a buffer bumped in the graph: with the
        parameters frozen (lr 0) and one batch, every step's loss differs, and
        a fresh run from the same seed repeats the sequence."""
        cfg = gpt.Config(layers=1, dropout=0.5)
        init = gpt.init(cfg, np.random.default_rng(0))
        opt = SGD(lr=0.0)
        tokens = TensorType(DType.int64, [4, cfg.seq], DEV)
        step = build_train_step(
            lambda p, x, y: gpt.loss(p, x, y, cfg), init, opt, [tokens, tokens], DEV,
            uses_seed=True, name="dropout_step",
        )
        batch = data.shakespeare(1, 4, cfg.seq)
        first, _, compiled = _run(step, init, opt, batch, 4)
        second, _, _ = _run(step, init, opt, batch, 4)
        self.assertEqual(len(set(first)), 4, first)
        self.assertEqual(first, second)
        self.assertEqual(compiled.seed.to_numpy().item(), 4 * SEED_STRIDE)

    def test_step_samples_its_own_batch(self):
        """A step whose only inputs are buffers and the corpus (what capture
        needs) trains: the loss falls over 40 steps of AdamW."""
        cfg = gpt.Config(layers=1)
        init = gpt.init(cfg, np.random.default_rng(0))
        _, tokens = data.shakespeare_vocab()
        corpus = tokens[: len(tokens) * 9 // 10]
        opt = AdamW(lr=1e-2, betas=(0.9, 0.99), weight_decay=0.1, decay=gpt.decays)
        step = build_train_step(
            lambda p, c: gpt.loss(p, *gpt.sample_batch(c, 8, cfg.seq), cfg),
            init, opt, [TensorType(DType.int64, [len(corpus)], DEV)], DEV,
            uses_seed=True, name="sampling_step",
        )
        losses, _, _ = _run(step, init, opt, [(corpus,)], 40)
        self.assertTrue(all(np.isfinite(losses)))
        self.assertLess(np.mean(losses[-5:]), np.mean(losses[:5]) - 0.3, losses)


if __name__ == "__main__":
    unittest.main()

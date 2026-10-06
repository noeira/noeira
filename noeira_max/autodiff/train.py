"""A whole train step as one MAX graph: forward, backward, optimizer update.

The parameters, the optimizer state (moments and the step counter) and,
when the step draws random numbers, a seed are ``BufferType`` inputs that the
graph stores into: one execution is one step, nothing is copied between
steps, and only the loss leaves the step, as a device tensor. A step whose
batch is drawn in the graph from a device-resident corpus has the same inputs
on every call, which is what device-graph capture needs.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.engine import read as read_mef
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops

from .transform import value_and_grad

# Each step advances the seed by this much: one random op per seed value, and
# no step needs more than this many random ops.
SEED_STRIDE = 1 << 16


@dataclass
class TrainStep:
    """A built (not yet compiled) train-step graph and its buffer layout."""

    graph: Graph
    param_names: list[str]
    state_names: list[str]
    uses_seed: bool
    diagnostics: list[str]
    """Extra scalar outputs after the loss (learning rate, gradient norm)."""


def build_train_step(
    loss_fn: Callable[..., object],
    params: Mapping[str, np.ndarray],
    optimizer,
    data_types: Sequence[TensorType],
    device: DeviceRef,
    *,
    uses_seed: bool = False,
    diagnostics: bool = False,
    name: str = "train_step",
) -> TrainStep:
    """``loss_fn(params, *data)`` must return a scalar loss. With
    ``uses_seed``, a ``[1]`` uint64 seed buffer comes after the optimizer
    state; ``loss_fn`` may then draw random numbers (dropout, batch sampling)
    and every step draws new ones."""
    param_names = list(params)
    state = optimizer.init(params)
    state_names = list(state)

    def buffer_type(value: np.ndarray) -> BufferType:
        return BufferType(DType.from_numpy(value.dtype), value.shape, device)

    input_types = [buffer_type(params[n]) for n in param_names]
    input_types += [buffer_type(state[n]) for n in state_names]
    if uses_seed:
        input_types.append(BufferType(DType.uint64, [1], device))
    input_types += list(data_types)

    with Graph(name, input_types=input_types) as graph:
        inputs = list(graph.inputs)
        p_buffers = dict(zip(param_names, inputs[: len(param_names)]))
        del inputs[: len(param_names)]
        s_buffers = dict(zip(state_names, inputs[: len(state_names)]))
        del inputs[: len(state_names)]
        if uses_seed:
            seed_buffer = inputs.pop(0)
            seed = ops.buffer_load(seed_buffer)
            ops.random.set_seed(seed)
        loss, grads = value_and_grad(loss_fn)(p_buffers, *inputs)
        stats = optimizer.apply(p_buffers, s_buffers, grads)
        if uses_seed:
            ops.buffer_store(seed_buffer, seed + ops.constant(SEED_STRIDE, DType.uint64, device))
        extra = sorted(stats) if diagnostics else []
        graph.output(loss, *(ops.reshape(stats[k], [1]) for k in extra))
    return TrainStep(graph, param_names, state_names, uses_seed, extra)


class CompiledStep:
    """A compiled train step, with its parameter and state buffers on the
    device. Calling it runs one step in place and returns the loss buffer."""

    def __init__(
        self,
        step: TrainStep,
        params: Mapping[str, np.ndarray],
        optimizer,
        device: Device,
        session: InferenceSession | None = None,
        *,
        model: Model | None = None,
        seed: int = 0,
    ):
        self.step = step
        self.device = device
        self.session = session or InferenceSession(devices=[device])
        self.compile_seconds = 0.0
        if model is None:
            start = time.perf_counter()
            self.compiled = self.session.compile(step.graph)
            self.compile_seconds = time.perf_counter() - start
            model = self.session.init(self.compiled)
        self.model = model
        state = optimizer.init(params)
        self.params = {n: _to_device(params[n], device) for n in step.param_names}
        self.state = {n: _to_device(state[n], device) for n in step.state_names}
        self.buffers = [*self.params.values(), *self.state.values()]
        if step.uses_seed:
            self.seed = _to_device(np.array([seed], np.uint64), device)
            self.buffers.append(self.seed)

    def __call__(self, *data: Buffer) -> list[Buffer]:
        """One step. Returns the loss (and diagnostics), still on the device."""
        return self.model.execute(*self.buffers, *data)

    def export_mef(self, path: str | Path) -> None:
        self.compiled.export_mef(path)

    def host_params(self) -> dict[str, np.ndarray]:
        return {n: b.to_numpy().copy() for n, b in self.params.items()}

    def host_state(self) -> dict[str, np.ndarray]:
        return {n: b.to_numpy().copy() for n, b in self.state.items()}


def load_mef(
    path: str | Path,
    step: TrainStep,
    params: Mapping[str, np.ndarray],
    optimizer,
    device: Device,
    session: InferenceSession | None = None,
) -> CompiledStep:
    """A :class:`CompiledStep` from an exported MEF, without compiling."""
    session = session or InferenceSession(devices=[device])
    model = session.init(read_mef(str(path)))
    return CompiledStep(step, params, optimizer, device, session, model=model)


def _to_device(value: np.ndarray, device: Device) -> Buffer:
    """A device buffer holding a COPY of ``value``.

    ``Buffer.from_numpy`` shares the array's memory and, on the CPU,
    ``.to(device)`` returns the same buffer: a step updating it in place
    would then rewrite the caller's array (the initial weights, here).
    """
    return Buffer.from_numpy(np.array(value, copy=True, order="C")).to(device)

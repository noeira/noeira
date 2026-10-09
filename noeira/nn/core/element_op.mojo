"""ElementOp — per-element forward + backward op for `Elementwise[DIM, OP]`.

Phase 1.3. Captures the per-lane math of an elementwise activation /
scaling / log / etc. operation. The `Elementwise[DIM, OP]` template
provides one CPU SIMD body + one GPU kernel; OP supplies the math.

  - `forward_scalar(x)` / `forward_simd[W](x)`: y = f(x). Used by the
    forward kernel; SIMD form is for the CPU vectorised inner loop.
  - `backward_scalar(c, go)` / `backward_simd[W](c, go)`: gi = f'(c) ⊙ go,
    where `c` is the cached value passed by the orchestrator. The
    interpretation of `c` depends on `owns_cache`:
      - `owns_cache = True`  → `c = y`  (output-cache; Tanh / Sigmoid).
      - `owns_cache = False` → `c = x`  (input-alias;  ReLU / Mish / Symlog).

  - `comptime owns_cache: Bool`: tells `Elementwise[DIM, OP]` whether
    its backward needs `y` (output) or `x` (input) as the cached value.
    If True, the `Elementwise` struct owns a CPU `List` / GPU
    `DeviceBuffer` cache and writes `y` into it on forward. If False,
    the struct aliases the forward input pointer (zero-copy — the
    orchestrator keeps the input slab live until backward, matching
    ReLU's existing pattern).

SPIKE verified 2026-05-22 (`tests/nn/spikes/spike_element_op_simd.mojo`):
Mojo nightly accepts `@staticmethod def forward_simd[W: Int]` on traits
and dispatches correctly across multiple W specialisations.
"""

from ..constants import DT


trait ElementOp(Movable & Deinitable):
    """Per-element math (scalar + SIMD specialisations) for the
    `Elementwise[DIM, OP]` template."""

    comptime owns_cache: Bool
    """If True, backward reads `y = f(x)` from an owned cache buffer.
    If False, backward reads `x` from the forward input pointer (alias)."""

    comptime is_relu: Bool = False
    """True for `ReLUOp` only: `LinearAct` then fuses bias + activation into
    a cuBLASLt `RELU_AUX_BIAS` epilogue on NVIDIA (`NN_LT_RELU`)."""

    @staticmethod
    def forward_scalar(x: Scalar[DT]) -> Scalar[DT]:
        ...

    @staticmethod
    def forward_simd[W: Int](x: SIMD[DT, W]) -> SIMD[DT, W]:
        ...

    @staticmethod
    def backward_scalar(c: Scalar[DT], go: Scalar[DT]) -> Scalar[DT]:
        """Backward gradient: gi = f'(c) ⊙ go. c = y when owns_cache, c = x when not.
        """
        ...

    @staticmethod
    def backward_simd[W: Int](c: SIMD[DT, W], go: SIMD[DT, W]) -> SIMD[DT, W]:
        ...

    @staticmethod
    def display_label() -> String:
        """Display name surfaced by `Elementwise.display_label` for graph
        exporters. Default generic; activation ops override (e.g. "GELU")."""
        return String("act")

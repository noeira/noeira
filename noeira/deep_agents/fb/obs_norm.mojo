"""Per-dimension observation standardisation, and the sidecar that keeps
training and evaluation from disagreeing about it.

BFM-Zero runs a `BatchNorm1d(affine=False, momentum=0.01)` on every observation
entering F, B, the actor and the critics (`agents/normalizers.py`); we ran raw
`qpos | qvel`. On walker that is not cosmetic — `qvel` spans roughly an order of
magnitude more than `qpos`, so the first `Linear` sees one block of inputs that
dominates the other before a single weight is learned.
`docs/BFM_ZERO_SHOT_RL.md` §16.3.

**Fixed statistics, not a running estimate.** Their normaliser is running
because their data is generated online and its distribution moves. Ours is a
frozen store, so the dataset mean and std ARE the converged values of that
running estimate — computing them once is the same thing, minus a per-step
kernel and minus a moving target under the TD bootstrap.

⚠⚠ **THE FAILURE THIS MODULE EXISTS TO PREVENT: train normalised, evaluate
raw.** Nothing raises. The actor receives inputs from a distribution it never
saw, emits plausible-looking actions, and the eval reports a number that is
neither the normalised policy's nor the unnormalised one's. It would look
exactly like "this arm of the sweep did not help".

The countermeasure is structural: the statistics travel WITH the checkpoint as
`<ckpt>.norm`, written by `save`, and evaluation calls `try_load` on that exact
path. An arm trained without normalisation writes no sidecar and evaluation
finds none; an arm trained with one cannot be evaluated without it unless the
file is deliberately deleted. Neither script owns a copy of the statistics and
neither has a flag that could disagree with the other.
"""

from noeira.core.bytes import string_from_bytes

from std.math import sqrt

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.checkpoint import _split_lines
from noeira.io.fileio import write_text_atomic


struct ObsNorm[N: Int](Copyable & Deinitable):
    """`(x - mu) / sd` per dimension, with `mu`/`sd` fixed at fit time."""

    var mu: List[Float64]
    var sd: List[Float64]

    def __init__(out self):
        self.mu = List[Float64](length=Self.N, fill=0.0)
        self.sd = List[Float64](length=Self.N, fill=1.0)

    def __init__(out self, *, deinit move: Self):
        self.mu = move.mu^
        self.sd = move.sd^

    @staticmethod
    def fit(ref rows: Tensor, n_rows: Int) raises -> Self:
        """Mean and std over `n_rows` contiguous `[N]` rows of a HOST tensor.

        ⚠ A dimension whose std is below 1e-6 is left ALONE (`sd = 1`), not
        divided by. Walker's store has none, but a constant column is exactly
        what a padded or unused observation slot looks like, and dividing it by
        its own noise would turn a dead input into the largest signal in the
        batch.
        """
        if n_rows <= 1:
            raise Error("ObsNorm.fit: need at least 2 rows, got "
                        + String(n_rows))
        var out = Self()
        for k in range(Self.N):
            var acc = Float64(0)
            for r in range(n_rows):
                acc += Float64(rows.data[r * Self.N + k])
            out.mu[k] = acc / Float64(n_rows)
        for k in range(Self.N):
            var acc = Float64(0)
            for r in range(n_rows):
                var d = Float64(rows.data[r * Self.N + k]) - out.mu[k]
                acc += d * d
            var s = sqrt(acc / Float64(n_rows))
            out.sd[k] = 1.0 if s < 1e-6 else s
        return out^

    def apply_rows(self, mut rows: Tensor, n_rows: Int):
        """Standardise `n_rows` `[N]` rows of a HOST tensor, in place."""
        for r in range(n_rows):
            for k in range(Self.N):
                rows.data[r * Self.N + k] = Scalar[DT](
                    (Float64(rows.data[r * Self.N + k]) - self.mu[k]) / self.sd[k]
                )

    def apply_row(self, mut row: Tensor):
        """Standardise ONE `[N]` observation in place — the rollout path."""
        for k in range(Self.N):
            row.data[k] = Scalar[DT](
                (Float64(row.data[k]) - self.mu[k]) / self.sd[k]
            )

    def save(self, path: String) raises:
        """Write `<ckpt>.norm`. Called for EVERY checkpoint, so any rung of a
        run can be evaluated on its own without knowing which run wrote it."""
        var s = String(Self.N) + "\n"
        for k in range(Self.N):
            s += String(self.mu[k]) + " " + String(self.sd[k]) + "\n"
        # Atomic: a crash mid-save must not leave the checkpoint beside it
        # with a truncated sidecar, which `try_load` would reject — or, cut on
        # a line boundary, accept with the wrong width's rows missing.
        write_text_atomic(path, s)

    @staticmethod
    def try_load(path: String) raises -> Optional[Self]:
        """`None` when the sidecar is absent — the run trained on raw inputs.

        ⚠ A PRESENT-but-wrong-width sidecar raises rather than being ignored.
        Silently falling back to raw inputs there would reproduce precisely the
        train/eval mismatch this module exists to prevent.
        """
        var content = _read_or_empty(path)
        if content.byte_length() == 0:
            return None
        var lines = _split_lines(content)
        if len(lines) < 1:
            raise Error("ObsNorm: sidecar " + path + " is empty")
        var n = atol(lines[0])
        if n != Self.N:
            raise Error(
                "ObsNorm: sidecar " + path + " holds " + String(n)
                + " dimensions, this build expects " + String(Self.N)
            )
        if len(lines) < Self.N + 1:
            raise Error(
                "ObsNorm: sidecar " + path + " declares " + String(Self.N)
                + " dimensions but carries " + String(len(lines) - 1)
                + " rows"
            )
        var out = Self()
        for k in range(Self.N):
            var parts = _split_ws(lines[k + 1])
            if len(parts) < 2:
                raise Error(
                    "ObsNorm: sidecar " + path + " line " + String(k + 1)
                    + " is not 'mean std'"
                )
            out.mu[k] = atof(parts[0])
            out.sd[k] = atof(parts[1])
            if out.sd[k] <= 0.0:
                raise Error(
                    "ObsNorm: sidecar " + path + " dimension " + String(k)
                    + " has std " + String(out.sd[k])
                )
        return out^


def _read_or_empty(path: String) -> String:
    """Empty string when the file is absent — the ONLY tolerated failure. Any
    other malformation is raised by `try_load` rather than swallowed here."""
    try:
        with open(path, "r") as f:
            return String(f.read())
    except:
        return String("")


def _split_ws(s: String) -> List[String]:
    var out = List[String]()
    # ⚠ BYTES — see `core/bytes.mojo`.
    var cur = List[UInt8]()
    var bytes = s.as_bytes()
    for i in range(len(bytes)):
        var c = bytes[i]
        if c == UInt8(ord(" ")) or c == UInt8(ord("\t")):
            if cur.byte_length() > 0:
                out.append(string_from_bytes(cur))
                cur = List[UInt8]()
        else:
            cur.append(c)
    if len(cur) > 0:
        out.append(string_from_bytes(cur))
    return out^

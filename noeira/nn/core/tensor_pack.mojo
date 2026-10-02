"""TensorPack[N] — an owned bag of N `Tensor` storages with a subscript that
returns an origin-erased ref. Used by orchestrators for their inter-module
buffer pools (Sequential's `act`/`grd`) AND as the multi-arity input / grad_input
container for ≥2-ary leaves (no §B0 origin-union constraint — these are
independent storages, not origin-pinned views).
"""

from noeira.nn.constants import DT
from .tensor import Tensor, TensorImpl


struct TensorPack[N: Int, ADT: DType = DT](
    Defaultable & Movable & Deinitable
):
    """`ADT` is the element (activation) dtype — `DT` (fp32) by default, so the
    bare `TensorPack[N]` is unchanged; bf16-flow combinators use the child's
    `ADT` for their inter-module buffer pools."""

    var tensors: List[TensorImpl[Self.ADT]]

    def __init__(out self):
        self.tensors = List[TensorImpl[Self.ADT]]()
        comptime for i in range(Self.N):
            self.tensors.append(TensorImpl[Self.ADT]())

    def release(mut self):
        """`TensorImpl.release` on every slot (see it)."""
        for i in range(len(self.tensors)):
            self.tensors[i].release()

    def __getitem__(
        mut self, index: Int
    ) raises -> ref [MutAnyOrigin] TensorImpl[Self.ADT]:
        # MUST be `MutAnyOrigin`, NOT `MutUntrackedOrigin`. The returned ref
        # points into `self.tensors`' heap buffer; the wildcard's lifetime-
        # PINNING is what keeps `self` (the pack) alive for as long as the ref
        # is live. `MutUntrackedOrigin` does NOT pin — the ref then dangles
        # the moment it's bound to a var or re-read (proven: a stored+reloaded
        # element reads garbage under MutUntracked, correct under MutAny). The
        # bypass-exclusivity property both share is necessary for passing two
        # elements as ref+mut call args; only the wildcard ALSO supplies the
        # load-bearing pinning. This is the rare case where the wildcard's
        # lifetime extension is the safety mechanism, not a footgun (cf. a
        # FIELD-stored erased pointer, which no origin pins — design review
        # §7.9). Do not "narrow" this to MutUntracked.
        return self.tensors[index]

"""Joint targets <-> the normalised action a batched env takes.

A family with `NORMALIZED_ACTIONS` maps an action in [-1, 1] affinely onto
each actuator's `ctrlrange`. A recorder whose demonstrator thinks in joint
targets (a leader arm, an IK expert) writes the INVERSE of that map, so a
recorded row is the action the policy will be asked to produce.
"""


struct CtrlRange(Movable):
    var lo: List[Float64]
    var hi: List[Float64]

    def __init__(out self, var lo: List[Float64], var hi: List[Float64]) raises:
        if len(lo) != len(hi):
            raise Error(
                "CtrlRange: " + String(len(lo)) + " lows against "
                + String(len(hi)) + " highs"
            )
        self.lo = lo^
        self.hi = hi^

    def __len__(self) -> Int:
        return len(self.lo)

    def normalize(self, i: Int, q: Float64) -> Float64:
        """Actuator `i`'s target `q` -> [-1, 1], clamped. A zero-width range
        maps to 0.

        (Arithmetic kept exactly as the recorders wrote it, so a
        re-recorded `.demo` is byte-identical.)"""
        var span = self.hi[i] - self.lo[i]
        var a = 2.0 * (q - self.lo[i]) / span - 1.0 if span != 0.0 else 0.0
        if a > 1.0:
            a = 1.0
        if a < -1.0:
            a = -1.0
        return a

    def denormalize(self, i: Int, a: Float64) -> Float64:
        """[-1, 1] -> actuator `i`'s target, the env's own affine map."""
        return self.lo[i] + (a + 1.0) * 0.5 * (self.hi[i] - self.lo[i])

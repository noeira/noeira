"""The BC policy's SHAPE and its normalisation — written once, used by both.

    from noeira.deep_agents.bc.policy import BcNet, BcNorm, bc_forward_host

A trainer fits it (`examples/libero/libero_bc_train.mojo` on LIBERO) and a
driver runs it (`examples/libero/libero_eval_batched.mojo`). A network the trainer and the driver each
spell for themselves is the shape `_a_rule_written_inline_twice_drifts` names:
one side gains a layer, the checkpoint still LOADS by name and size for the
layers that match, and the policy quietly scores worse than it trained.

⚠ THE NORMALISATION IS PART OF THE POLICY. The fit standardises its inputs
with the TRAINING rows' mean and spread, so a driver feeding raw metres to the
same weights is not running the policy that was trained. `BcNorm` is written
beside the checkpoint by the trainer and refused by the loader when it does not
match the checkpoint's width.
"""

from std.os.path import exists

from noeira.io.fileio import write_text_atomic

from noeira.nn.constants import DT
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.activations import ReLU
from noeira.nn.primitives.linear import Linear


comptime BC_HID: Int = 256
"""The hidden width of both layers. ⚠ CHANGING IT INVALIDATES EVERY
CHECKPOINT: `load_params` validates names and SIZES, so an old file raises
rather than loading a different network."""


comptime BcNet[OBS: Int, ACT: Int] = Sequential[
    Linear[OBS, BC_HID], ReLU[BC_HID], Linear[BC_HID, BC_HID], ReLU[BC_HID],
    Linear[BC_HID, ACT],
]
"""`OBS -> 256 -> 256 -> ACT`, no output squash — recorded actions (LIBERO's
OSC_POSE, a family's normalised targets) are already in [-1, 1] and a `tanh` head would put the fit's error where the
gradient vanishes. The driver clamps at inference instead."""


struct BcNorm(Copyable, Movable):
    """The input standardisation a checkpoint was trained under."""

    var mu: List[Float64]
    var sd: List[Float64]
    var act_dim: Int

    def __init__(out self):
        self.mu = List[Float64]()
        self.sd = List[Float64]()
        self.act_dim = 0

    def __init__(out self, var mu: List[Float64], var sd: List[Float64],
                 act_dim: Int):
        self.mu = mu^
        self.sd = sd^
        self.act_dim = act_dim

    def obs_dim(self) -> Int:
        return len(self.mu)

    def apply(self, raw: List[Float64], mut out: List[Float64]) raises:
        """`(x - mu) / sd`, APPENDING — the trainer's own transform."""
        if len(raw) != len(self.mu):
            raise Error(
                "bc policy: the observation is " + String(len(raw))
                + " words and the policy was trained on " + String(len(self.mu))
            )
        for j in range(len(raw)):
            out.append((raw[j] - self.mu[j]) / self.sd[j])


def write_bc_norm(
    path: String, mu: List[Float64], sd: List[Float64], act_dim: Int
) raises:
    """The sidecar `<checkpoint>.norm` the trainer writes."""
    var text = String("# bc normalisation — GENERATED\n")
    text += "obs_dim=" + String(len(mu)) + "\n"
    text += "act_dim=" + String(act_dim) + "\n"
    for j in range(len(mu)):
        text += "mu=" + String(mu[j]) + "\n"
    for j in range(len(sd)):
        text += "sd=" + String(sd[j]) + "\n"
    # Atomic: the checkpoint beside it is, and a policy whose sidecar was cut
    # short by a crash is refused on load — or worse, loaded with the wrong
    # width if the cut fell on a line boundary.
    write_text_atomic(path, text)


def load_bc_norm(
    path: String, obs_dim: Int, act_dim: Int,
    retrain_hint: String = "Re-run the BC trainer.",
) raises -> BcNorm:
    """⚠ REFUSES A SIDECAR THAT IS NOT THIS MODEL'S. A policy run under the
    wrong normalisation produces plausible actions and a wrong rate, which is
    the failure this check exists to make impossible."""
    if not exists(path):
        raise Error(
            "bc policy: no normalisation beside the checkpoint at " + path
            + " — a checkpoint without it is a network fed raw metres where it"
            " was trained on standardised ones. " + retrain_hint
        )
    var text: String
    with open(path, "r") as fh:
        text = fh.read()
    var mu = List[Float64]()
    var sd = List[Float64]()
    var got_obs = -1
    var got_act = -1
    var lines = text.split("\n")
    for i in range(len(lines)):
        var l = String(String(lines[i]).strip())
        if l.byte_length() == 0 or l.startswith("#"):
            continue
        var eq = l.find("=")
        if eq < 0:
            raise Error(path + ": malformed line '" + l + "'")
        var key = String(l[byte=0:eq])
        var val = String(l[byte = eq + 1 : l.byte_length()])
        if key == "obs_dim":
            got_obs = Int(val)
        elif key == "act_dim":
            got_act = Int(val)
        elif key == "mu":
            mu.append(Float64(val))
        elif key == "sd":
            sd.append(Float64(val))
        else:
            raise Error(path + ": unknown key '" + key + "'")
    if got_obs != obs_dim or got_act != act_dim:
        raise Error(
            path + ": trained for obs " + String(got_obs) + " act "
            + String(got_act) + ", this model is obs " + String(obs_dim)
            + " act " + String(act_dim)
        )
    if len(mu) != obs_dim or len(sd) != obs_dim:
        raise Error(
            path + ": " + String(len(mu)) + " means and " + String(len(sd))
            + " spreads for " + String(obs_dim) + " words"
        )
    for j in range(obs_dim):
        if sd[j] <= 0.0:
            raise Error(path + ": spread " + String(j) + " is not positive")
    return BcNorm(mu^, sd^, act_dim)

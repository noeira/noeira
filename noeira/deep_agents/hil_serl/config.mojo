"""HIL-SERL's knobs on an off-policy run: `--demos`, `--demo-filter`,
`--bc-weight`, `--bc-q-ratio`, `--bc-only`.

A driver keeps one `HilSerlConfig`, offers every flag to `try_parse` in its
own argument loop, calls `validate` after the loop, `print_banner` with its
other settings, `log_config` on its remote logger, and `apply_hil_serl`
(`setup.mojo`) once the trainer is set up.
"""

from noeira.nn.constants import DT
from noeira.core.logger import Logger
from noeira.deep_agents.demos.filter import DemoFilter


struct HilSerlConfig(Copyable, Movable):
    var demos: String
    """`--demos a.demo[,b.demo]` — HIL-SERL / RLPD. The files' transitions
    are loaded into the replay BEFORE the loop and pinned as a prefix, and
    half of every minibatch is drawn from them for the whole run — the 50/50
    demo/online sampling of HIL-SERL's `train_rlpd.py`. A file comes from a
    recorder (a human on the leader arm in the sim, a scripted expert, or a
    human / expert correcting a checkpoint — the intervention half)."""
    var filter: DemoFilter
    """`--demo-filter all|success|intervened` — which rows."""
    var bc_weight: Scalar[DT]
    """`--bc-weight λ` — a behaviour-cloning penalty on the DEMO half of every
    batch (`SACActorLoss.set_bc`). Symmetric sampling alone left the tower
    lift policy parked (eval 356/355 at 25k/50k with 7677 expert rows
    pinned); TD3+BC's normalisation puts λ near mean|Q| / 2.5 ≈ 40 there.
    Needs `--demos`; 0 (the default) is plain RLPD."""
    var bc_q_ratio: Scalar[DT]
    """`--bc-q-ratio r` — the BC weight tracks the critic: after every
    diagnostics flush it becomes max(`--bc-weight`, r · mean|Q|)
    (`SACTrainer.set_bc_q_ratio`). With λ fixed the BC term's share of the
    actor gradient shrinks as Q grows: the tower lift policy peaked at 25k
    (eval 414, mean_q 58) and had lost its grasp by 50k (400, mean_q 87).
    0.7 reproduces that run's 25k balance (40 / 58) and holds it. Needs
    `--bc-weight > 0`, which is the floor; 0 (the default) is a fixed λ."""
    var bc_only: Bool
    """`--bc-only` — zero the SAC half of the actor loss
    (`SACActorLoss.set_q_weight(0)`): the actor fits the demo half of every
    batch alone while the critics train on ITS rollouts, so the checkpoint is
    an imitator with critics that have seen the task — the first half of a
    warm start (`--init CKPT` is the second). Needs `--bc-weight > 0`."""

    def __init__(out self):
        self.demos = String("")
        self.filter = DemoFilter()
        self.bc_weight = Scalar[DT](0.0)
        self.bc_q_ratio = Scalar[DT](0.0)
        self.bc_only = False

    def active(self) -> Bool:
        return self.demos.byte_length() > 0

    def try_parse(
        mut self, flag: String, value: String, has_value: Bool,
        who: String = "hil-serl",
    ) raises -> Bool:
        """True when `flag` is one of ours (its value, if any, is consumed).
        `--bc-only` takes no value; the others need one and are left to the
        caller's unknown-argument error without it."""
        if flag == "--bc-only":
            self.bc_only = True
            return True
        if not has_value:
            return False
        if flag == "--demos":
            self.demos = value
            return True
        if flag == "--bc-weight":
            self.bc_weight = Scalar[DT](Float64(value))
            return True
        if flag == "--bc-q-ratio":
            self.bc_q_ratio = Scalar[DT](Float64(value))
            return True
        if flag == "--demo-filter":
            self.filter = DemoFilter.parse(value, who)
            return True
        return False

    def validate(self, who: String = "hil-serl") raises:
        if not self.active() and self.bc_weight > Scalar[DT](0):
            raise Error(who + ": --bc-weight needs --demos")
        if self.bc_only and self.bc_weight <= Scalar[DT](0):
            raise Error(who + ": --bc-only needs --bc-weight > 0")
        if self.bc_q_ratio < Scalar[DT](0):
            raise Error(who + ": --bc-q-ratio must be >= 0")
        if self.bc_q_ratio > Scalar[DT](0) and self.bc_weight <= Scalar[DT](0):
            raise Error(who + ": --bc-q-ratio needs --bc-weight > 0 (the floor)")

    def paths(self, who: String = "hil-serl") raises -> List[String]:
        """`--demos` split on commas. ⚠ An empty piece is refused: `a.demo,`
        is a typo, not a second file."""
        var out = List[String]()
        var parts = self.demos.split(",")
        for i in range(len(parts)):
            var piece = String(String(parts[i]).strip())
            if piece.byte_length() == 0:
                raise Error(
                    who + ": empty element in '" + self.demos + "'. A trailing"
                    " or doubled comma is a typo, not an empty file."
                )
            out.append(piece^)
        return out^

    def print_banner(self, warmup: Int, default_warmup: Int):
        """⚠ THE WARMUP STILL APPLIES with demos: `--warmup` gates both the
        uniform-random actions and the first gradient step, and HIL-SERL runs
        `random_steps=0, training_starts=100` because the demos already cover
        the space."""
        if not self.active():
            return
        print("  demos    :", self.demos, " filter:", self.filter.name(),
              " -> half of every minibatch (RLPD)")
        if warmup >= default_warmup:
            print("  ⚠ --warmup is", warmup, "with demos loaded; HIL-SERL"
                  " starts learning after ~100 steps. Consider --warmup 1000.")

    def log_config[L: Logger](self, mut logger: L):
        logger.set_config("demos", self.demos)
        logger.set_config("demo_filter", self.filter.name())
        logger.set_config("bc_weight", String(self.bc_weight))
        logger.set_config("bc_q_ratio", String(self.bc_q_ratio))
        logger.set_config("bc_only", String(self.bc_only))

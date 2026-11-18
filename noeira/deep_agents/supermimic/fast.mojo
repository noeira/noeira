"""The fast-training options of the agents' loop, and their schedule state.

What the public "minutes" recipes change in the loop, as options that are all
off by default (`docs/SUPERMIMIC_FAST_TRAINING_PLAN.md` §4):

    pi_ema        deploy a running average of the policy      InstantTrack `--pi_ema 0.995`
    wm_batches    world-model batches per update              InstantTrack `--wm_batches 5`
    pi_lr_mult    the policy's lr = lr × this, ramped back    BlinkMimic `--pi_lr_mult 4
                  to × 1 over `lr_decay_len` updates once       --lr_decay_ret 0.75 --lr_decay_len 300
                  the training return reaches `lr_decay_ret`    --lr_decay_by 1400`
                  (at update `lr_decay_by` at the latest)
    action_noise  collection's action noise                   the official code dropped it (7e94b07)
    ref_skip      action = decoder output + the action that   BlinkMimic `--pi_ref_skip`
                  commands the target pose
    analytic_term imagination terminates on the real          the official code's 98ee3ec; the
                  thresholds, not on the learned head         recipes' `soft_termination`
    v_max         V capped at the reward ceiling in the       the official code's 133c0f0
                  bootstrap and the critic's targets

Both agents hold one `SMFastState` and ask it for the policy's lr multiplier
and the world-model batch count; the averaged policy and the reference skip
live in the trainer (`SuperMimicTrainer.set_ema`, `.set_ref_skip`).

Counts are in UPDATES, the recipes' "iterations" (one update per 2.5 env
steps at UTD 0.4), not in this loop's iterations (one env step each).
"""


comptime SM_FAST_CHECK: Int = 50
"""Updates between two tests of the lr ramp's trigger (BlinkMimic tests it at
each log line, every 50 iterations, on the episodes finished since the last)."""


@fieldwise_init
struct SMFastOpts(Copyable, Movable):
    var pi_ema: Float64           # the averaged policy's decay per update; 0 = off
    var wm_batches: Int           # world-model batches per update; 0 = the config's `DEF_WM_BATCHES`
    var pi_lr_mult: Float64       # the policy's lr multiplier; 1 = off
    var lr_decay_ret: Float64     # training return that starts the ramp back to 1; 0 = no trigger
    var lr_decay_len: Int         # the ramp's length, in updates
    var lr_decay_by: Int          # the ramp starts at this update at the latest; 0 = no deadline
    var action_noise: Float64     # collection's action noise std; < 0 = the schedule's (0.1 with noise)
    var ref_skip: Bool
    var analytic_term: Bool
    var v_max: Float64            # the value ceiling; 0 = none (10 = 0.1 / (1 − 0.99) for this reward)

    @staticmethod
    def off() -> Self:
        """The reproduction's loop."""
        return Self(0.0, 0, 1.0, 0.0, 300, 0, -1.0, False, False, 0.0)

    @staticmethod
    def blink() -> Self:
        """BlinkMimic's `PRESET[main]` values for these knobs, with the
        analytic termination its code line always has. Its lr 4e-4 is the
        schedule's `lr`, its 1024³ world model and batch 2048 are the
        config's widths. The value ceiling is the official code's, not
        BlinkMimic's: off here."""
        return Self(0.995, 5, 4.0, 0.75, 300, 1400, -1.0, True, True, 0.0)


struct SMFastState(Copyable, Movable):
    var opts: SMFastOpts
    var decay_at: Int             # the update at which the lr ramp started; -1 = not yet
    var ret_sum: Float64          # normalised returns of the episodes finished since the last test
    var ret_n: Int
    var last_return: Float64      # their mean at the last test with episodes (the trigger's reading)

    def __init__(out self, opts: SMFastOpts):
        self.opts = opts.copy()
        self.decay_at = -1
        self.ret_sum = 0.0
        self.ret_n = 0
        self.last_return = 0.0

    def note_episode(mut self, norm_return: Float64):
        """A finished training episode's normalised return Σr / (0.1·T)."""
        self.ret_sum += norm_return
        self.ret_n += 1

    def wm_batches(self, config_default: Int) -> Int:
        return self.opts.wm_batches if self.opts.wm_batches > 0 else config_default

    def pi_mult(self, updates: Int) -> Float64:
        """The policy's lr multiplier at `updates` done: `pi_lr_mult` until
        the ramp starts, then linearly to 1 over `lr_decay_len` updates."""
        var m = self.opts.pi_lr_mult
        if self.decay_at < 0:
            return m
        var u = Float64(updates - self.decay_at) / Float64(max(self.opts.lr_decay_len, 1))
        return m + (1.0 - m) * min(1.0, max(u, 0.0))

    def tick(mut self, updates: Int) -> Bool:
        """After update number `updates`: every `SM_FAST_CHECK` updates, test
        the ramp's trigger on the episodes finished since the last test.
        Returns True on the update that starts the ramp."""
        if updates % SM_FAST_CHECK != 0:
            return False
        if self.ret_n > 0:
            self.last_return = self.ret_sum / Float64(self.ret_n)
        self.ret_sum = 0.0
        self.ret_n = 0
        if self.decay_at >= 0 or self.opts.pi_lr_mult == 1.0:
            return False
        var reached = self.opts.lr_decay_ret > 0.0 and self.last_return >= self.opts.lr_decay_ret
        var deadline = self.opts.lr_decay_by > 0 and updates >= self.opts.lr_decay_by
        if reached or deadline:
            self.decay_at = updates
            return True
        return False

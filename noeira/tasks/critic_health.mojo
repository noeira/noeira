"""THE CRITIC'S HEALTH, READ BACK OUT OF A RUN'S OWN CSV.

    var h = critic_health(csv_path, 0.99)
    peak_mean_q, peak_critic_loss, q_fixed_point = h[0], h[1], h[2]

## ⚠⚠ WHY THIS IS NOT A HOOK IN THE TRAINING LOOP

`mean_q` and `critic_loss` are produced inside the shared off-policy driver
(`deep_agents/training/driver_offpolicy.mojo`), which every algorithm in the
tree runs through. A per-iteration health hook there would be a change to
everyone's driver to serve one family's diagnosis. The metrics are already
written to the run's CSV, so reading that file back after `close()` costs one
file read and touches nothing shared.

⚠ AFTER `close()` IS THE ONLY POINT THE FILE IS WHOLE. `CsvLogger` streams
rows as they happen; the last of them reach disk when the queue drains.
"""

def critic_health(csv_path: String, gamma: Float64) raises -> Tuple[
    Float64, Float64, Float64
]:
    """`(peak mean_q, peak critic_loss, the run's own Q fixed point)`, read
    back out of the CSV this run just wrote.

    ## ⚠⚠ THE END-OF-RUN RATIO PASSES A RUN THAT WAS ALREADY DESTROYED

    A diverging critic does not stay diverged. Two `gather` runs at a config
    identical down to every logged `cfg/*` word — same weights, same margins,
    same `tau`, same 48.2 warmup baseline:

        run          peak mean_q   peak critic_loss   PEAK   final   success
        converged          48.876             0.892   1.00x   1.00x    0.5625
        diverged         4372.959         27214.555 273.38x   1.05x    0.0000

    Both END at the fixed point. The diverged one got there by blowing up at
    step 24k, peaking at 180k and DECAYING back over the remaining 800k steps,
    with the policy it had been training destroyed on the way. A check that
    reads only the last row calls that run healthy — and the 1.05x it printed
    is why 34 GPU-minutes of failure read as a task that is merely hard.

    ⚠ THE PEAK IS THE STATISTIC, NOT THE FINAL VALUE. The peak ratio
    separates the two runs 273-fold and `critic_loss` by four orders of
    magnitude (0.89 vs 27215); the FINAL ratio separates them by 5%. Those
    numbers are read off the two real CSVs, not reconstructed.

    ## ⚠ THE MIDDLE OF THE RANGE IS NOT EMPTY, AND THIS FILE USED TO SAY IT WAS

    The 10x threshold was first justified as sitting "in the middle of an
    empty decade" — 1.00x on the converged run, 89x on the destroyed one,
    nothing between. A later `lift` run at `updates_per_step 16` peaked at
    **4.19x** with `critic_loss` 33.9: it did not diverge, it overshot and
    decayed, and it learned nothing (`reward_mean` 0.3931 -> 0.3949 over 1M
    steps). So the decade is populated, the claim was an artefact of two data
    points, and a run between 2x and 10x is worth SAYING rather than passing
    silently — it is not a diverged run and it is not a healthy one.

    ⚠ THE FIXED POINT IS THE RUN'S OWN. `reward_mean / (1 - gamma)` from the
    last row that carries one, so the check needs no per-task calibration —
    the same reason the baseline comes from the run's own warmup.
    """
    var peak_q = 0.0
    var peak_loss = 0.0
    var last_r = 0.0
    # ⚠ `with` IS A SCOPE — a `var` bound inside it does not survive the
    # block, so the text is hoisted and assigned rather than declared there.
    var text: String
    with open(csv_path, "r") as f:
        text = f.read()
    var lines = text.split("\n")
    for i in range(len(lines)):
        # step,wall_time_ms,name,value — the header and any short/blank
        # trailing line are skipped by the field count, not by index, so a
        # partially flushed file reads as far as it is intact.
        var c = lines[i].split(",")
        if len(c) != 4:
            continue
        var name = String(c[2])
        # ⚠ `mean_reward` is the pre-2026-09-22 name of `reward_mean`; the
        # runs this is read against were mostly logged under it.
        if name == "mean_reward":
            name = String("reward_mean")
        if name != "mean_q" and name != "critic_loss" and name != "reward_mean":
            continue
        var v: Float64
        try:
            v = Float64(String(c[3]))
        except:
            continue        # the header row's "value", and nothing else
        if name == "mean_q":
            if v > peak_q:
                peak_q = v
        elif name == "critic_loss":
            if v > peak_loss:
                peak_loss = v
        else:
            last_r = v
    return (peak_q, peak_loss, last_r / (1.0 - gamma))



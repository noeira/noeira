"""`critic_health` SEPARATES A CONVERGED RUN FROM A DESTROYED ONE.

    pixi run mojo run -I . tests/tasks/test_critic_health.mojo

## ⚠⚠ WHAT THIS GATE EXISTS TO CATCH

Two `gather` runs at a configuration identical down to every logged `cfg/*`
word — same shaping weights, same margins, same `tau`, same 48.2 warmup
baseline — landed at 0.5625 and 0.0. Both FINISHED with `mean_q` within 5% of
their fixed point, because a diverged critic decays back. Any check that reads
the last row calls both healthy.

The fixtures below are those two runs, synthesised to the shape the real CSVs
have, with the peaks the real ones carried:

    run          peak mean_q   peak critic_loss   final ratio
    converged           48.9                1.0         1.00x
    diverged          4373.0            27215.0         1.05x

⚠ THE ANTI-VACUITY LEG IS THE CONVERGED ONE. A check that flagged everything
would also "catch" the diverged run, so the gate fails if the converged
fixture trips it — that is the half that is easy to lose and impossible to
notice, since the alarm only ever fires on runs that already look bad.
"""

from std.os import remove

from noeira.tasks.critic_health import critic_health


def write_csv(path: String, peak_q: Float64, peak_loss: Float64,
              final_r: Float64,
              reward_name: String = String("reward_mean")) raises:
    """A CSV with the four-field shape `CsvLogger` writes.

    ⚠ THE PEAK IS BURIED IN THE MIDDLE and the LAST rows are healthy — which
    is the whole point. A reader that takes the final value sees only the
    tail; the peak is three quarters of the way back, where the real diverged
    run's was.
    """
    var s = String("step,wall_time_ms,name,value\n")
    for i in range(200):
        var q = 20.0
        var l = 0.5
        if i == 140:               # the spike, mid-run and long since decayed
            q = peak_q
            l = peak_loss
        s += String(i * 32) + ",0.1,mean_q," + String(q) + "\n"
        s += String(i * 32) + ",0.1,critic_loss," + String(l) + "\n"
        s += (
            String(i * 32) + ",0.1," + reward_name + "," + String(final_r)
            + "\n"
        )
    with open(path, "w") as f:
        f.write(s)


def main() raises:
    print("=== critic_health: converged vs destroyed ===")
    var fails = 0

    # The run that reached 0.5625. mean_reward 0.488 -> fixed point 48.8.
    var ok_path = String("/tmp/_ch_converged.csv")
    write_csv(ok_path, 48.9, 1.0, 0.488)
    var h_ok = critic_health(ok_path, 0.99)
    print("  converged: peak_q", h_ok[0], " peak_loss", h_ok[1],
          " fixed point", h_ok[2], " ratio", h_ok[0] / h_ok[2])
    if h_ok[0] > 10.0 * h_ok[2]:
        print("  FAIL: the converged run tripped the divergence threshold —"
              " the check flags everything and discriminates nothing")
        fails += 1

    # The run that reached 0.0. Same fixed point; peak 89x it.
    var bad_path = String("/tmp/_ch_diverged.csv")
    write_csv(bad_path, 4373.0, 27215.0, 0.488)
    var h_bad = critic_health(bad_path, 0.99)
    print("  diverged : peak_q", h_bad[0], " peak_loss", h_bad[1],
          " fixed point", h_bad[2], " ratio", h_bad[0] / h_bad[2])
    if h_bad[0] <= 10.0 * h_bad[2]:
        print("  FAIL: the destroyed run read as healthy — a 4373 peak"
              " against a fixed point of 48.8 is the failure this exists for")
        fails += 1

    # ⚠ AND THE TWO MUST DIFFER. Identical readings on both fixtures is what
    # a function that ignores its argument returns, and it would pass both
    # legs above if the threshold happened to sit right.
    if h_ok[0] == h_bad[0] or h_ok[1] == h_bad[1]:
        print("  FAIL: both runs read the same peak — the file is not being"
              " read")
        fails += 1

    # ⚠ THE FIXED POINT IS THE RUN'S OWN, so a run with a different reward
    # scale must move it; a hard-coded 48.8 would pass everything above.
    var scaled = String("/tmp/_ch_scaled.csv")
    write_csv(scaled, 48.9, 1.0, 4.88)
    var h_s = critic_health(scaled, 0.99)
    print("  10x reward: fixed point", h_s[2], "(expected 488)")
    if h_s[2] < 487.0 or h_s[2] > 489.0:
        print("  FAIL: the fixed point does not track the run's own"
              " mean_reward")
        fails += 1
    # At ten times the reward the SAME peak is now a tenth of the fixed
    # point — so the threshold is relative, not an absolute Q ceiling.
    if h_s[0] > 10.0 * h_s[2]:
        print("  FAIL: a healthy peak on a 10x-reward run tripped the"
              " threshold — the check is reading an absolute Q, not a ratio")
        fails += 1

    # ⚠⚠ THE OVERSHOOT BAND, which exists because the middle of the range
    # turned out to be populated. A `lift` run peaked at 4.19x with
    # `critic_loss` 33.9 — it did not diverge, it overshot and decayed, and
    # it learned nothing (`mean_reward` 0.3931 -> 0.3949 over 1M steps). A
    # single 10x gate passes that run silently while it is neither healthy
    # nor diverged.
    var mid_path = String("/tmp/_ch_overshoot.csv")
    write_csv(mid_path, 165.3, 33.9, 0.395)
    var h_mid = critic_health(mid_path, 0.99)
    var r_mid = h_mid[0] / h_mid[2]
    print("  overshoot: peak_q", h_mid[0], " fixed point", h_mid[2],
          " ratio", r_mid)
    if r_mid <= 2.0 or r_mid > 10.0:
        print("  FAIL: the 4.19x run does not land in the overshoot band"
              " (2x, 10x] — it would be reported as healthy or as diverged,"
              " and it is neither")
        fails += 1
    # ⚠ AND THE HEALTHY RUN MUST STAY BELOW THE NEW LOWER EDGE, or the band
    # swallows the runs that actually learned.
    if h_ok[0] / h_ok[2] > 2.0:
        print("  FAIL: the converged run trips the overshoot band")
        fails += 1
    remove(mid_path)

    # A run logged before 2026-09-22 names the batch reward `mean_reward`.
    # Those are the runs this is read against, so the old name must give the
    # SAME fixed point — not a silent 0 that makes every peak look diverged.
    var legacy = String("/tmp/_ch_legacy.csv")
    write_csv(legacy, 48.9, 1.0, 0.488, String("mean_reward"))
    var h_l = critic_health(legacy, 0.99)
    print("  legacy `mean_reward` CSV: fixed point", h_l[2], "(expected 48.8)")
    if h_l[2] < 48.7 or h_l[2] > 48.9:
        print("  FAIL: a pre-rename CSV lost its reward")
        fails += 1
    remove(legacy)

    remove(ok_path)
    remove(bad_path)
    remove(scaled)

    print()
    if fails == 0:
        print("=== PASS ===")
    else:
        raise Error("critic_health: " + String(fails) + " check(s) failed")

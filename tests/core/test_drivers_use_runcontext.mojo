"""The retrofitted drivers name their paths from a run, not from a constant.

Run: pixi run mojo run -I . tests/core/test_drivers_use_runcontext.mojo

⚠⚠ THIS IS A SOURCE GATE, AND IT IS ON PURPOSE. Three of the four drivers below
cannot be built on the laptop this is developed on — `bfm_zero_train_gpu` is
NVIDIA-only (its own header says so, and the G1 batched env does not compile for
Metal), and `act_so101_train_gpu` dies in `ld` with a symbol-name-too-long
assertion that reproduces at HEAD. A runtime gate for them would run nowhere.
What CAN be checked everywhere is that the constants came out and the run went
in, which is the whole of the retrofit.

The shape it prevents coming back:

    comptime DEFAULT_CKPT = "act_so101_best_gpu.ckpt"
    var best_ckpt = String("/tmp/act_so101_best_gpu.ckpt")
    comptime CKPT_PATH: StaticString = "fb_walker_all_d128.ckpt"

Every run of a driver overwrote the previous run's checkpoint by construction.
`checkpoints/` holds 219 flat entries because of it.

⚠ COMMENT LINES ARE EXCLUDED FROM THE NEGATIVE CHECK. The retrofit deliberately
KEPT the old names in `⚠` notes explaining what they were and why they went —
that history is worth more than the grep is, so the gate reads code only.
"""

from noeira.core.kv import split_on


comptime DRIVERS = (
    # the task-family SAC run lives in the shared driver; `sac_task_gpu.mojo`
    # and `sac_tower_gpu.mojo` are one-call entries into it (since 3311b28a0)
    "noeira/tasks/sac_family_driver.mojo"
    "|examples/so101/act_so101_train_gpu.mojo"
    "|examples/fb/fb_train_gpu.mojo"
    "|examples/fb/fb_train_cpr_gpu.mojo"
    "|examples/fb/fb_online_walker_gpu.mojo"
    "|examples/fb/fb_online_cpr_walker_gpu.mojo"
    "|examples/g1/bfm_zero_train_gpu.mojo"
)
"""⚠⚠ SEVEN, AND THE FIRST PASS OF THIS GATE LISTED FOUR. The FB family carried
**five** copies of the same hand-rolled `--tag` block — `fb_train_gpu`,
`fb_train_cpr_gpu`, `fb_online_walker_gpu`, `fb_online_cpr_walker_gpu` and
`bfm_zero_train_gpu` — and the retrofit found two of them by grepping for the
two spellings it already knew instead of for the SHAPE. That is
`_a_rule_written_inline_twice_drifts` committed while citing it.

The list is here, in the gate, so the next driver that grows a `--tag` has one
obvious place to be added."""

comptime BANNED = (
    "/tmp/act_so101_best_gpu.ckpt"
    "|/tmp/act_so101_last_gpu.ckpt"
    "|fb_walker_all_d128.ckpt"
    "|fb_walker_all_d128_metrics.csv"
    '|var ckpt_path = "fb_walker_'
    '|ckpt = "fb_online_walker_'
    '|agent.save_state(tag +'
    '|comptime CKPT_PREFIX'
    '|comptime CSV_PREFIX'
    '|comptime CKPT_PATH'
    '|comptime CSV_PATH'
    '|comptime RUN_NAME'
)


def _code_lines(path: String) raises -> List[String]:
    """Every line that is not blank and not a `#` comment."""
    var text: String
    with open(path, "r") as fh:
        text = fh.read()
    var out = List[String]()
    var lines = split_on(text, String("\n"))
    for i in range(len(lines)):
        var s = String(lines[i].strip())
        if s.byte_length() == 0 or s.startswith("#"):
            continue
        out.append(s)
    return out^


def main() raises:
    print("=" * 62)
    print("Retrofitted drivers — paths come from RunContext")
    print("=" * 62)

    var drivers = split_on(String(DRIVERS), String("|"))
    var banned = split_on(String(BANNED), String("|"))

    var files = 0
    var code_lines = 0
    var missing_run = 0
    var leaks = 0

    for d in drivers:
        var lines = _code_lines(d)
        files += 1
        code_lines += len(lines)

        # ── positive: the run exists and is closed ───────────────────────
        var has_ctor = False
        var has_close = False
        for i in range(len(lines)):
            if lines[i].find("RunContext(") >= 0:
                has_ctor = True
            # `finish_run` (core/run_session) closes the run too, after the
            # logger has the outcome — the order the inline form got wrong.
            if lines[i].find("run.close()") >= 0 or lines[i].find("finish_run(") >= 0:
                has_close = True
        if not has_ctor or not has_close:
            missing_run += 1
            print(
                "    ", d, "RunContext(", has_ctor, ") run.close(", has_close, ")"
            )

        # ── negative: no legacy path constant survives IN CODE ───────────
        for b in banned:
            for i in range(len(lines)):
                if lines[i].find(b) >= 0:
                    leaks += 1
                    print("    ", d, "still has:", b)

    print(
        "  drivers:", files, "files /", code_lines, "code lines,",
        missing_run, "without a run,", leaks, "legacy constants",
    )
    # ⚠ VACUITY IS THE DEFAULT FAILURE: a glob that stopped resolving, or a
    # reader that returned nothing, looks exactly like a clean pass.
    if files != 7 or code_lines < 3000:
        raise Error(
            "gate went vacuous: " + String(files) + " files, "
            + String(code_lines) + " code lines"
        )
    if missing_run != 0:
        raise Error(String(missing_run) + " drivers do not open/close a run")
    if leaks != 0:
        raise Error(String(leaks) + " legacy path constants survive in code")
    print("[PASS] drivers use RunContext")

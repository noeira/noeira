# +--------------------------------------------------------------------------+ #
# | An example that writes a checkpoint writes it into a run
# +--------------------------------------------------------------------------+ #
"""A SOURCE gate over `examples/`.

    pixi run mojo run -I . tests/core/test_examples_open_a_run.mojo

## ⚠⚠ Why

`RunContext` landed in P0c and by 2026-09-22 SEVEN entry points used it, out of
about seventy that produce checkpoints. The rest kept a `comptime` path —

    comptime CHECKPOINT_PATH = "sac_ant_nn.ckpt"

— so every run of that example overwrote the previous one's weights, kept no
record of whether the run was any good, and had nothing for `project-push` to
send. `tests/core/test_drivers_use_runcontext.mojo` pins the SEVEN by name and
checks their banned constants; this one is the rule for everyone else, and it
finds a file nobody remembered to add to a list.

THE RULE: a file under `examples/` that WRITES a checkpoint (`save_state(`,
`.save(`, `save_params`, `save_trainables`, or a `checkpoint_path=` handed to a
driver) must mention `RunContext(`.

## ⚠ What is deliberately NOT in scope

Readers (eval, viewer, gif, deploy, inspect) only LOAD; they take
`--ckpt <run_id|path>` through `core/run.resolve_checkpoint` instead, and
loading is not what this gate looks for. `EXEMPT` below lists the few writers
that are not runs — profiling harnesses and dataset collectors — each with its
reason, because an exemption without one becomes the hole the rule leaks
through.
"""

from noeira.io.fileio import read_file_bytes
from noeira.io.proc import run_capture


comptime WRITES = (
    "save_state(|.save(|save_params|save_trainables|checkpoint_path="
)

comptime EXEMPT = (
    # Profiling harnesses: they write a scratch file to measure the write, and
    # produce a number for a document rather than an artifact to promote.
    "examples/fb/fb_train_profile_gpu.mojo"
    "|examples/half_cheetah/mbpo_half_cheetah_profile_nn_gpu.mojo"
    "|examples/half_cheetah/sac_half_cheetah_profile_graph_nn.mojo"
    # Dataset collectors: they read a policy ladder and write a STORE, never a
    # checkpoint.
    "|examples/fb/collect_walker_all.mojo"
    "|examples/fb/collect_walker_sac.mojo"
    # `buf.save(...)` — an offline replay BUFFER, the input to a run rather
    # than its artifact.
    "|examples/dreamer4/pong_reward_collect_buffer.mojo"
    "|examples/lewm/lewm_pong_collect_buffer.mojo"
    # `norm.save(...)` — it EXPORTS an existing run's normalisation for the
    # deploy path; it trains nothing.
    "|examples/so101/act_so101_export_norm.mojo"
)

comptime MIN_FILES = 300
"""`examples/` holds ~400 .mojo files; a scan that reaches fewer did not run
where it should. VACUITY IS THE DEFAULT FAILURE for a gate like this."""


def _read(path: String) raises -> String:
    var bytes = read_file_bytes(path)
    bytes.append(0)
    return String(unsafe_from_utf8_ptr=bytes.unsafe_ptr())


def _split(s: String, sep: String) raises -> List[String]:
    var out = List[String]()
    for p in s.split(sep):
        out.append(String(p))
    return out^


def main() raises:
    print("=" * 62)
    print("examples that write a checkpoint open a run")
    print("=" * 62)
    var listing = run_capture(
        String("find examples -name '*.mojo' -type f | sort"), 1 << 22
    )
    var files = List[String]()
    for f in listing.split("\n"):
        var s = String(String(f).strip())
        if s.byte_length() > 0:
            files.append(s)

    var patterns = _split(String(WRITES), String("|"))
    var exempt = _split(String(EXEMPT), String("|"))
    var writers = 0
    var offenders = 0

    for path in files:
        var p = String(path)
        var skip = False
        for e in exempt:
            if String(e) == p:
                skip = True
        if skip:
            continue
        var text = _read(p)
        # Code lines only: the retrofit KEPT the old constants in `⚠` notes
        # explaining what they were, and that history is worth more than the
        # grep is.
        var code = String("")
        for line in text.split("\n"):
            var l = String(String(line).strip())
            if l.byte_length() > 0 and not l.startswith("#"):
                code += l + "\n"
        var writes = False
        for w in patterns:
            if code.find(String(w)) >= 0:
                writes = True
        if not writes:
            continue
        writers += 1
        if code.find(String("RunContext(")) < 0:
            offenders += 1
            print("    " + p + " writes a checkpoint outside a run")

    print(
        "  " + String(len(files)) + " example files, " + String(writers)
        + " write a checkpoint, " + String(offenders) + " without a RunContext"
    )
    if len(files) < MIN_FILES:
        raise Error(
            "gate went vacuous: " + String(len(files)) + " files scanned"
        )
    if writers < 50:
        raise Error(
            "gate went vacuous: only " + String(writers) + " writers found —"
            " the patterns stopped matching"
        )
    if offenders != 0:
        raise Error(
            String(offenders) + " example(s) write a checkpoint with no run:"
            " every run of them overwrites the last one's weights and"
            " `project-push` has nothing to send"
        )
    print("[PASS] examples open a run")

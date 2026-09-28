"""`RunContext` — the id, the clock, the record, and the status transitions.

Run: pixi run mojo run -I . tests/core/test_run_context.mojo

The pain this closes, stated as it is: a run had two identities that never met.
The dashboard minted a `run_id`; on disk the same run was a **compile-time
constant** —

    comptime DEFAULT_CKPT = "act_so101_best_gpu.ckpt"

— so every run of a driver overwrote the previous one's checkpoint by
construction, and nothing on disk recorded whether the run that wrote it was
any good. `checkpoints/` holds 219 flat entries as a result.

⚠ THE FILESYSTEM HALF WRITES UNDER A PER-RUN TEMP ROOT and never touches the
repo's own `runs/`. A gate that wrote where the tool writes would be one
`--root` typo away from deleting a real run.
"""

from std.time import perf_counter_ns

from noeira.core.kv import kv_lines
from noeira.core.logger import CompositeLogger, CsvLogger, Logger, RemoteLogger
from noeira.core.run_session import finish_run, run_logger
from noeira.io.artifact_sink import ArtifactSink
from noeira.core.run import (
    RunContext,
    run_id_of_checkpoint,
    resolve_checkpoint,
    civil_from_days,
    date_utc,
    derive_run_id,
    iso8601_utc,
    load_run,
    parse_run,
    register_run,
    resume_args_for,
    run_command,
    shell_word,
    slugify,
)
from noeira.io.proc import run_capture


def _root() -> String:
    return String("/tmp/noeira_run_gate_") + String(perf_counter_ns())


# =============================================================================
# The clock — pure, and gated by value
# =============================================================================


def test_iso8601_against_known_epochs() raises:
    """⚠ THE ERA-BASED CONVERSION EXISTS BECAUSE THE LEAP RULE IS ONLY PERIODIC
    OVER 400 YEARS. 2000 is a leap year and 2100 is not; a "365 or 366" loop
    gets the second one wrong, and nothing in a training run would notice."""
    var cases = [
        (0, String("1970-01-01T00:00:00Z")),
        (951782400, String("2000-02-29T00:00:00Z")),   # leap, /400
        (4107542400, String("2100-03-01T00:00:00Z")),  # NOT leap, /100
        (1000000000, String("2001-09-09T01:46:40Z")),
        (1767225600, String("2026-01-01T00:00:00Z")),
        (1788000000, String("2026-08-29T10:40:00Z")),
    ]
    var compared = 0
    var differing = 0
    for c in cases:
        compared += 1
        var got = iso8601_utc(c[0])
        if got != c[1]:
            differing += 1
            print("    epoch", c[0], "->", got, "want", c[1])
    print("  clock:", compared, "epochs compared,", differing, "differing")
    if compared != 6 or differing != 0:
        raise Error("iso8601_utc wrong on " + String(differing))
    if date_utc(1788000000) != String("2026-08-29"):
        raise Error("date_utc: " + date_utc(1788000000))


def test_civil_from_days_edges() raises:
    var d0 = civil_from_days(0)
    if d0[0] != 1970 or d0[1] != 1 or d0[2] != 1:
        raise Error("day 0 is not 1970-01-01")
    var leap = civil_from_days(11016)  # 2000-02-29
    if leap[0] != 2000 or leap[1] != 2 or leap[2] != 29:
        raise Error("2000-02-29 missed")
    print("  civil: day 0 and the 2000 leap day both exact")


# =============================================================================
# The identifier — deterministic, and derived from things that differ
# =============================================================================


def test_run_id_is_deterministic_and_separating() raises:
    """⚠⚠ DERIVED, NOT RANDOM. Two runs on one box differ in `pid`/`start_ns`;
    two boxes in the same second differ in `host`. Reproducibility is what lets
    *which box wrote this?* have an answer at all — and it is why a collision
    would be a BUG in the derivation rather than bad luck."""
    var d = String("2026-09-02")
    var s = String("act-reach")
    var a = derive_run_id(d, s, String("vast-5090-de81"), 4242, 1000)
    var again = derive_run_id(d, s, String("vast-5090-de81"), 4242, 1000)
    if a != again:
        raise Error("same inputs gave two ids: " + a + " / " + again)
    if not a.startswith("2026-09-02_act-reach_") or a.byte_length() != 29:
        raise Error("id shape wrong: " + a + " (" + String(a.byte_length()) + ")")

    var others = [
        derive_run_id(d, s, String("laptop"), 4242, 1000),        # host
        derive_run_id(d, s, String("vast-5090-de81"), 4243, 1000),  # pid
        derive_run_id(d, s, String("vast-5090-de81"), 4242, 1001),  # start_ns
    ]
    var collisions = 0
    for o in others:
        if o == a:
            collisions += 1
            print("    collided:", o)
    print("  id: 1 repeat identical, 3 one-field changes,", collisions, "collisions")
    if collisions != 0:
        raise Error("the id does not separate on " + String(collisions) + " axes")


def test_slugify() raises:
    var cases = [
        (String("act_so101_train_gpu"), String("act-so101-train-gpu")),
        (String("SAC Task GPU"), String("sac-task-gpu")),
        (String("  --weird--  "), String("weird")),
        (String("vast.ai-5090"), String("vast-ai-5090")),
    ]
    var differing = 0
    for c in cases:
        if slugify(c[0]) != c[1]:
            differing += 1
            print("    ", c[0], "->", slugify(c[0]), "want", c[1])
    print("  slugify:", len(cases), "compared,", differing, "differing")
    if differing != 0:
        raise Error("slugify wrong on " + String(differing))


# =============================================================================
# The directory and the record
# =============================================================================


def test_run_kv_exists_at_t0_and_says_running() raises:
    """⚠⚠ THE RECORD IS WRITTEN BEFORE ANY TRAINING. A run that registers itself
    only on success leaves exactly the orphan checkpoint this layer abolishes —
    so this asserts the file is on disk with `status=running` while the run is
    still, as far as anything knows, about to start."""
    var root = _root()
    var run = RunContext(
        project=String("so101"),
        driver=String("examples/so101/act_so101_train_gpu.mojo"),
        env=String("family:so101_tabletop"),
        task=String("so101_reach_brick"),
        seed=12345,
        root=root,
    )
    var rec = load_run(run.kv_path())
    if rec.status != String("running"):
        raise Error("status at t=0 is '" + rec.status + "'")
    if not rec.is_stale():
        raise Error("a running run is not reported stale")
    if rec.run_id != run.id or rec.project != String("so101"):
        raise Error("identity did not survive: " + rec.run_id)
    if rec.seed != 12345 or rec.task != String("so101_reach_brick"):
        raise Error("fields did not survive")
    if rec.finished.byte_length() != 0:
        raise Error("finished was written before close(): " + rec.finished)
    # ⚠ The slug defaults from the driver's basename — this is the path that
    # kills `comptime DEFAULT_CKPT`, so it is asserted rather than assumed.
    if run.slug != String("act-so101-train-gpu"):
        raise Error("slug: " + run.slug)
    var ck = run.checkpoint_path(String("best"))
    if ck != run.dir + "/checkpoints/best.ckpt":
        raise Error("checkpoint_path: " + ck)
    print("  t=0:", rec.status, "/ seed", rec.seed, "/ id", rec.run_id)
    _ = run^


def test_status_transitions_and_close_idempotence() raises:
    var root = _root()
    var run = RunContext(
        project=String("p"), driver=String("d.mojo"), root=root
    )
    run.set_outcome(String("success_rate=0.82 val_l1=0.031"))
    run.set_tag(String("meilleur reach à ce jour, testé 8/10"))
    run.set_config(String("lr"), String("3e-4"))
    run.set_config(String("batch"), String("64"))
    run.close()
    var done = load_run(run.kv_path())
    if done.status != String("done") or done.finished.byte_length() == 0:
        raise Error("close(): status=" + done.status + " finished=" + done.finished)
    if done.is_stale():
        raise Error("a closed run reported stale")
    if len(done.config) != 2 or done.config[0] != String("lr:3e-4"):
        raise Error("config lines: " + String(len(done.config)))
    # ⚠ NON-ASCII IN `tag=` IS THE CASE `core/kv` WAS FIXED FOR — a byte-wise
    # reader that used `chr` per byte returned mojibake for exactly this.
    if done.tag != String("meilleur reach à ce jour, testé 8/10"):
        raise Error("tag came back as: " + done.tag)
    # ⚠ `outcome` CARRIES AN `=`, which only works because split_once cuts at
    # the FIRST one.
    if done.outcome != String("success_rate=0.82 val_l1=0.031"):
        raise Error("outcome: " + done.outcome)

    var before = done.finished
    run.close(String("killed"))
    var after = load_run(run.kv_path())
    if after.status != String("done") or after.finished != before:
        raise Error("close() was not idempotent: " + after.status)
    print("  transitions: running -> done, close() idempotent, tag/outcome exact")
    _ = run^


def test_a_stated_terminal_status_survives_close() raises:
    """⚠ THE INTERRUPT PATH (P0e) STATES `killed` BEFORE `close()` REPORTS THE
    DEFAULT. A killed run filed as `done` is worse than no record at all."""
    var root = _root()
    var run = RunContext(
        project=String("p"), driver=String("d.mojo"), root=root
    )
    run.set_status(String("killed"))
    run.close()
    var rec = load_run(run.kv_path())
    if rec.status != String("killed"):
        raise Error("close() overwrote a stated status with " + rec.status)
    print("  stated end: killed survived close()")
    _ = run^


def test_an_unknown_key_raises() raises:
    """⚠⚠ THE `tasks/spec.mojo` POLICY, NOT `data/manifest.mojo`'S. A manifest
    ignores unknown keys so a store from a newer build stays readable. Here a
    typo'd key is a LIE about what a run was."""
    var good = String("schema_version=1\nrun_id=r\nstatus=done\n")
    _ = parse_run(good, String("gate"))
    var bad = String("schema_version=1\nrun_id=r\nstatuss=done\n")
    var refused = False
    try:
        _ = parse_run(bad, String("gate"))
    except:
        refused = True
    if not refused:
        raise Error("a typo'd key was accepted — status= would be silently lost")
    var future = String("schema_version=3\nrun_id=r\n")
    var refused2 = False
    try:
        _ = parse_run(future, String("gate"))
    except:
        refused2 = True
    if not refused2:
        raise Error("a future schema_version was accepted")
    print("  strictness: unknown key and future schema both refused")


def test_the_record_is_rewritten_not_appended() raises:
    """⚠ ONE RENDERER, CALLED FROM EVERY MUTATOR. Appending the changed line is
    how a file ends up with two `status=` lines and a reader that believes the
    first — `_a_rule_written_inline_twice_drifts`."""
    var root = _root()
    var run = RunContext(
        project=String("p"), driver=String("d.mojo"), root=root
    )
    run.set_outcome(String("a"))
    run.set_outcome(String("b"))
    run.set_tag(String("t"))
    run.close()
    var text: String
    with open(run.kv_path(), "r") as fh:
        text = fh.read()
    var ls = kv_lines(text, String("gate"))
    var status_lines = 0
    var outcome_lines = 0
    for i in range(len(ls)):
        if ls[i].key == "status":
            status_lines += 1
        elif ls[i].key == "outcome":
            outcome_lines += 1
    print(
        "  single-valued keys:", status_lines, "status,", outcome_lines,
        "outcome (want 1, 1) across", len(ls), "lines",
    )
    if status_lines != 1 or outcome_lines != 1:
        raise Error("the record was appended to, not rewritten")
    _ = run^


def test_register_run_seeds_the_config_then_announces() raises:
    """⚠⚠ WHY `register()` IS HERE AND NOT IN `RunContext.__init__`. `/runs`
    carries the config, which is assembled from the run — and making
    `RunContext` generic over `Logger` to hold one would put a type parameter
    on every driver signature and every struct that stores a run.

    ⚠ ASSERT BEFORE `close()`. The remote half registers itself on close, so a
    check made after passes whether or not this function did anything — the
    vacuity that `test_run_lifecycle` already caught once.
    """
    var root = _root()
    var run = RunContext(
        project=String("so101"),
        driver=String("examples/so101/act_so101_train_gpu.mojo"),
        env=String("family:so101_tabletop"),
        task=String("so101_reach_brick"),
        seed=7,
        root=root,
    )
    var remote = RemoteLogger(
        server_url=String("http://127.0.0.1:9"), run_id=run.id
    )
    var lg = CompositeLogger(CsvLogger(run.metrics_path()), remote)
    register_run(run, lg)
    if not lg.b.registered():
        raise Error("register_run did not announce the run")
    # The dashboard's config must carry the run's identity, not the driver's
    # hand-assembled guess at it.
    var payload = lg.b._register_payload()
    var want = [
        String('"run_id":"') + run.id + '"',
        String('"project":"so101"'),
        String('"task":"so101_reach_brick"'),
        String('"seed":"7"'),
    ]
    var missing = 0
    for w in want:
        if payload.find(w) < 0:
            missing += 1
            print("    absent:", w)
    print("  register_run:", len(want) - missing, "of", len(want), "config fields")
    if missing != 0:
        raise Error("register_run seeded " + String(missing) + " fields short")
    lg.close()
    run.close()
    _ = run^


def _write(path: String, text: String) raises:
    with open(path, "w") as fh:
        fh.write(text)


def test_close_records_what_the_run_left() raises:
    """⚠⚠ `add_artifact` HAD ZERO CALLERS, so every `run.kv` on disk carried no
    `artifact=` line and `project-push` — which pushes what `run.kv` lists —
    pushed nothing. `close()` now reads the run directory itself. Asserted:
    checkpoints, eval files, the CSV and its config sidecar are recorded; a
    `.tmp` (an atomic write caught mid-flight) is not; and a checkpoint
    recorded twice is ONE entry describing the latest bytes."""
    var root = _root()
    var run = RunContext(project=String("p"), driver=String("d.mojo"), root=root)
    var lg = run_logger(run, env_path=root + "/no.env")
    lg.set_config(String("algorithm"), String("SAC"))
    register_run(run, lg)
    lg.log_scalar(String("loss"), 1.5, 1)
    _write(run.checkpoint_path(String("best")), String("v1"))
    run.add_artifact(String("checkpoints/best.ckpt"))
    _write(run.checkpoint_path(String("best")), String("v2 is longer"))
    _write(run.checkpoint_path(String("last")) + ".tmp", String("partial"))
    _write(run.eval_dir() + "/report.kv", String("success_rate=0.5\n"))
    lg.close()
    run.close()
    var rec = load_run(run.kv_path())
    var rels = List[String]()
    for a in rec.artifacts:
        var s0 = String(a)
        rels.append(String(s0[byte = 0 : s0.find(":")]))
    print("  recorded:", len(rels), "artifacts")
    var want = [
        String("checkpoints/best.ckpt"), String("eval/report.kv"),
        String("metrics.csv"), String("metrics.config.kv"),
    ]
    for w in want:
        var n = 0
        for r in rels:
            if r == w:
                n += 1
        if n != 1:
            raise Error(w + " recorded " + String(n) + " times, want 1")
    for r in rels:
        if r.endswith(".tmp"):
            raise Error("a .tmp file was recorded: " + r)
    # The best entry describes the SECOND write (12 bytes), not the first.
    for a in rec.artifacts:
        if String(a).startswith("checkpoints/best.ckpt:") and String(a).find(":12:") < 0:
            raise Error("best.ckpt entry is stale: " + String(a))
    # And the config reached its sidecar.
    with open(run.dir + "/metrics.config.kv", "r") as fh:
        var cfg = fh.read()
        if cfg.find("algorithm=SAC") < 0 or cfg.find("project=p") < 0:
            raise Error("config sidecar: " + cfg)
    _ = run^


@fieldwise_init
struct _Rec(Logger):
    """Records the order of the calls `finish_run` makes."""
    var calls: List[String]

    def log_scalar(mut self, name: String, value: Float64, step: Int) raises:
        pass

    def log_scalars(
        mut self, names: List[String], values: List[Float64], step: Int
    ) raises:
        pass

    def flush(mut self) raises:
        pass

    def register(mut self) raises:
        pass

    def finish(mut self, status: String, outcome: String) raises:
        self.calls.append("finish:" + status + ":" + outcome)

    def close(mut self) raises:
        self.calls.append(String("close"))

    def set_config(mut self, key: String, value: String):
        pass

    def is_active(self) -> Bool:
        return True


def test_finish_run_sends_the_verdict_before_closing() raises:
    """⚠⚠ THE ORDER IS THE BUG IT FIXES. The SAC family driver closed its logger
    and then set the outcome; `close()` had already sent `/finish` with `done`
    and an empty outcome, and the first finish wins. So: the logger must see
    `finish(outcome)` BEFORE `close`, and run.kv must carry the same outcome."""
    var root = _root()
    var run = RunContext(project=String("p"), driver=String("d.mojo"), root=root)
    var lg = _Rec(List[String]())
    var none = Optional[ArtifactSink](None)
    finish_run(run, lg, none, String("eval_return=812.5"))
    print("  finish_run calls:", lg.calls[0], "then", lg.calls[1])
    if len(lg.calls) != 2 or lg.calls[0] != "finish:done:eval_return=812.5" or lg.calls[1] != "close":
        raise Error("finish_run order: " + lg.calls[0])
    var rec = load_run(run.kv_path())
    if rec.outcome != "eval_return=812.5" or rec.status != "done" or rec.finished.byte_length() == 0:
        raise Error("run.kv: status=" + rec.status + " outcome=" + rec.outcome)
    _ = run^


def test_run_id_of_checkpoint() raises:
    var cases = [
        (String("runs/2026-09-22_bfm-zero_ab12cd34/checkpoints/step_5.ckpt"), String("2026-09-22_bfm-zero_ab12cd34")),
        (String("projects/g1/runs/X_y_z/checkpoints/final.ckpt"), String("X_y_z")),
        (String("/tmp/act_so101_best.ckpt"), String("")),
        (String(""), String("")),
    ]
    for c in cases:
        var got = run_id_of_checkpoint(c[0])
        if got != c[1]:
            raise Error("run_id_of_checkpoint(" + c[0] + ") = '" + got + "'")
    print("  run_id_of_checkpoint:", len(cases), "cases")


def test_resolve_checkpoint() raises:
    """A file is used as is; a run id finds `<run>/checkpoints/<name>.ckpt`;
    anything else RAISES rather than falling back to a default path."""
    var root = _root()
    var run = RunContext(project=String("p"), driver=String("d.mojo"), root=root)
    _write(run.checkpoint_path(String("last")), String("x"))
    var f = run.checkpoint_path(String("last"))
    if resolve_checkpoint(f) != f:
        raise Error("a file path was not returned as is")
    # A run under the flat root, by id: make one there by symlink-free copy.
    var flat = String("runs/") + run.id
    _ = run_capture(String("mkdir -p ") + flat + "/checkpoints && cp " + f + " " + flat + "/checkpoints/", 4096)
    var got = resolve_checkpoint(run.id)
    _ = run_capture(String("rm -rf runs/") + run.id + " 2>&1", 4096)
    if got != flat + "/checkpoints/last.ckpt":
        raise Error("run id resolved to " + got)
    var raised = False
    try:
        _ = resolve_checkpoint(String("no_such_run_xyz"))
    except:
        raised = True
    if not raised:
        raise Error("an unknown ref did not raise")
    print("  resolve_checkpoint: file, run id, unknown -> raise")
    _ = run^


# =============================================================================
# A run is a recipe: its command, its source, how it continues
# =============================================================================


def test_shell_word_round_trips_through_sh() raises:
    """⚠ CHECKED BY THE SHELL, NOT BY A SECOND COPY OF THE RULE. Each word is
    handed to `/bin/sh` and must come back byte-identical — a quoting rule
    compared with itself would pass whatever it got wrong."""
    var cases = [
        String("--steps"), String("200000"), String("so101_reach_clear"),
        String("runs/a-b_c/checkpoints/last.ckpt"), String("a b"),
        String("it's"), String("$HOME"), String("x;rm"), String(""),
        String("--note=8/10 on the arm"),
    ]
    for c in cases:
        var w = shell_word(c)
        var back = run_capture(String("printf %s ") + w, 4096)
        if back != c:
            raise Error("shell_word(" + c + ") = " + w + " came back as " + back)
    if shell_word(String("--steps")) != String("--steps"):
        raise Error("a plain flag was quoted: " + shell_word(String("--steps")))
    if shell_word(String("x\ny")) != String("'x y'"):
        raise Error("newline: " + shell_word(String("x\ny")))
    print("  shell_word:", len(cases), "words round-trip through /bin/sh")


def test_the_record_carries_its_command() raises:
    """argv, the pixi env and the schema survive `run.kv`, and the command is
    rebuilt from them by the one builder."""
    var root = _root()
    var run = RunContext(
        project=String("so101"),
        driver=String("examples/tasks/sac_task_gpu.mojo"),
        seed=3,
        root=root,
    )
    # This test's own argv is empty; set the list a driver would have had.
    run.args = [
        String("so101_lift_brick"), String("--steps"), String("2000"),
        shell_word(String("a b")), shell_word(String("it's")),
    ]
    run.pixi_env = String("nvidia")
    run.set_tag(String("recipe"))  # any mutator rewrites run.kv
    var rec = load_run(run.kv_path())
    if rec.schema_version != 2:
        raise Error("schema_version read back " + String(rec.schema_version))
    if len(rec.args) != 5 or rec.args[3] != String("'a b'"):
        raise Error("args did not survive: " + String(len(rec.args)))
    var cmd = run_command(rec.driver, rec.pixi_env, rec.args)
    var want = (
        "pixi run -e nvidia mojo run -I . examples/tasks/sac_task_gpu.mojo"
        " so101_lift_brick --steps 2000 'a b' 'it'\\''s'"
    )
    if cmd != want or run.reproduce_command() != want:
        raise Error("command:\n  got  " + cmd + "\n  want " + want)
    # The default env adds no `-e` (a box with one env has no name for it).
    if run_command(rec.driver, String("default"), List[String]()) != String(
        "pixi run mojo run -I . examples/tasks/sac_task_gpu.mojo"
    ):
        raise Error("default env grew a -e")
    print("  recipe:", cmd)
    run.close()
    _ = run^


def _words(s: String) -> List[String]:
    var out = List[String]()
    for w in s.split(" "):
        if String(w).byte_length() > 0:
            out.append(String(w))
    return out^


def _expect_words(got: List[String], want: String, what: String) raises:
    var g = String("")
    for x in got:
        g += (" " if g.byte_length() > 0 else "") + x
    if g != want:
        raise Error(what + ":\n  got  " + g + "\n  want " + want)


def test_resume_args_for_both_driver_shapes() raises:
    """The two declared shapes: BFM's `--resume {ckpt}` (flag with a value)
    and the SO-ARM driver's `--resume --ckpt {ckpt}` (bare flag + valued
    flag). A run that was ITSELF a resume must not carry two `--resume`s."""
    var ck = String("runs/r1/checkpoints/last.ckpt")
    _expect_words(
        resume_args_for(
            _words("--steps 5 --resume old.ckpt --tag x"),
            String("--resume {ckpt}"), ck,
        ),
        "--steps 5 --tag x --resume " + ck, "BFM shape",
    )
    _expect_words(
        resume_args_for(
            _words("--resume --ckpt a.ckpt --steps 3"),
            String("--resume --ckpt {ckpt}"), ck,
        ),
        "--steps 3 --resume --ckpt " + ck, "SO-ARM shape",
    )
    # A checkpoint path with a space is quoted, not split.
    _expect_words(
        resume_args_for(List[String](), String("--resume {ckpt}"),
                        String("my runs/last.ckpt")),
        "--resume 'my runs/last.ckpt'", "quoted ckpt",
    )
    print("  resume_args_for: both templates, and a resumed run's own flags")


def test_a_dirty_tree_saves_its_patch() raises:
    """⚠ CONDITIONAL ON THE TREE, AND IT SAYS WHICH BRANCH RAN. On a dirty
    tree the patch must be exactly `git diff HEAD --binary`; on a clean one
    there must be none."""
    var root = _root()
    var run = RunContext(
        project=String("so101"),
        driver=String("examples/tasks/sac_task_gpu.mojo"),
        root=root,
    )
    if not run.dirty:
        if run.source_patch.byte_length() != 0:
            raise Error("clean tree wrote a patch: " + run.source_patch)
        print("  source.patch: tree CLEAN here — only the no-patch branch ran")
    elif run.source_patch.startswith("too_large:"):
        print("  source.patch: tree dirty past the cap —", run.source_patch)
    else:
        if run.source_patch != String("source.patch"):
            raise Error("dirty tree, source_patch=" + run.source_patch)
        var mine = run_capture(
            String("wc -c < ") + run.dir + "/source.patch", 64
        ).strip()
        var head = run_capture(
            String("git diff HEAD --binary | wc -c"), 64
        ).strip()
        if String(mine) != String(head):
            raise Error("patch is " + String(mine) + " bytes, git diff "
                        + String(head))
        print("  source.patch: dirty tree,", String(mine), "bytes = git diff")
    run.close()
    var rec = load_run(run.kv_path())
    if run.source_patch == String("source.patch"):
        var listed = False
        for a in rec.artifacts:
            if a.startswith("source.patch:"):
                listed = True
        if not listed:
            raise Error("source.patch is not an artifact — it would not be pushed")
    _ = run^


def test_register_run_sends_the_command() raises:
    var root = _root()
    var run = RunContext(
        project=String("so101"),
        driver=String("examples/tasks/sac_task_gpu.mojo"),
        root=root,
    )
    run.set_resume_args(String("--resume {ckpt}"))
    var remote = RemoteLogger(
        server_url=String("http://127.0.0.1:9"), run_id=run.id
    )
    var lg = CompositeLogger(CsvLogger(run.metrics_path()), remote)
    register_run(run, lg)
    var payload = lg.b._register_payload()
    for w in [String('"command":"pixi run'), String('"dirty":"'),
              String('"resume_args":"--resume {ckpt}"')]:
        if payload.find(w) < 0:
            raise Error("register payload lacks " + w)
    var raised = False
    try:
        run.set_resume_args(String("--resume"))
    except:
        raised = True
    if not raised:
        raise Error("a resume template without {ckpt} was accepted")
    print("  register_run: command, dirty and resume_args reach the dashboard")
    lg.close()
    run.close()
    _ = run^


def test_a_schema_1_record_still_reads() raises:
    var text = String(
        "schema_version=1\nrun_id=r\nproject=p\ndriver=d.mojo\n"
        "status=done\n"
    )
    var rec = parse_run(text, String("old"))
    if rec.schema_version != 1 or len(rec.args) != 0:
        raise Error("schema 1 record misread")
    var raised = False
    try:
        _ = parse_run(String("schema_version=3\nrun_id=r\n"), String("new"))
    except:
        raised = True
    if not raised:
        raise Error("a schema 3 record was accepted")
    print("  schema: 1 still reads (and is known to have no arguments); 3 refused")


def main() raises:
    print("=" * 62)
    print("RunContext — one identifier, and a record written at t=0")
    print("=" * 62)
    test_iso8601_against_known_epochs()
    test_civil_from_days_edges()
    test_run_id_is_deterministic_and_separating()
    test_slugify()
    test_run_kv_exists_at_t0_and_says_running()
    test_status_transitions_and_close_idempotence()
    test_a_stated_terminal_status_survives_close()
    test_an_unknown_key_raises()
    test_the_record_is_rewritten_not_appended()
    test_register_run_seeds_the_config_then_announces()
    test_close_records_what_the_run_left()
    test_finish_run_sends_the_verdict_before_closing()
    test_run_id_of_checkpoint()
    test_resolve_checkpoint()
    test_shell_word_round_trips_through_sh()
    test_the_record_carries_its_command()
    test_resume_args_for_both_driver_shapes()
    test_a_dirty_tree_saves_its_patch()
    test_register_run_sends_the_command()
    test_a_schema_1_record_still_reads()
    # ⚠ THE PREFIX IS A LITERAL, NOT A VARIABLE. An `rm -rf` assembled from a
    # String is one empty value away from a very bad day; this one cannot
    # widen, and every root above is minted under exactly this prefix.
    _ = run_capture(String("rm -rf /tmp/noeira_run_gate_* 2>&1"), 4096)
    print("[PASS] run context")

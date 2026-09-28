# +--------------------------------------------------------------------------+ #
# | Every checkpoint a driver writes must be offered to the artifact sink
# +--------------------------------------------------------------------------+ #
"""A SOURCE gate over `noeira/deep_agents/training/`.

    pixi run build-http                              # ONCE
    pixi run mojo run -I . tests/deep_agents/test_checkpoints_announce.mojo

## ⚠⚠ Why a source gate rather than a behavioural one

`trainer.save_state(checkpoint_path)` appears eighteen times across four driver
files. A site that saves and forgets to announce produces an artifact that
simply never leaves the box — and there is NOTHING TO SEE: the run trains, the
checkpoint is on disk, the dashboard shows the run, and only the box dying
reveals that the weights were never uploaded.

That is not a defect a behavioural test can reach. Covering it that way would
mean running all eighteen code paths — several of which are NVIDIA-only, and
one of which is a CUDA-graph capture. So the gate reads the source and checks
the one mechanical rule that makes the drift impossible.

This is the same shape as `tests/core/test_drivers_use_runcontext.mojo`, and it
exists for the same reason that one does: the previous version of this mistake
— five hand-rolled `--tag` blocks in the FB family, of which a patch touched
two — was found by accident rather than by a gate.

## ⚠ The file list is IN THE GATE, not derived

A gate that globbed `training/*.mojo` would pass forever the day someone adds a
fifth driver, because a file that is not scanned cannot fail. The list is
written down, and the count of scanned save sites is PRINTED beside the count
of failures — "0 violations" is also what scanning nothing prints.
"""

from noeira.deep_agents.training.checkpoint import (
    announce_checkpoint,
    offered_path,
)
from noeira.io.artifact_sink import ArtifactSink
from noeira.io.fileio import read_file_bytes
from noeira.io.proc import quote_arg, run_capture


def drivers() -> List[String]:
    """⚠ A LIST, NOT A `comptime` TUPLE: a tuple cannot be indexed by a runtime
    loop variable. Written down rather than globbed — see the header."""
    var out = List[String]()
    out.append(String("noeira/deep_agents/training/driver_offpolicy.mojo"))
    out.append(
        String("noeira/deep_agents/training/driver_offpolicy_discrete.mojo")
    )
    out.append(String("noeira/deep_agents/training/driver_onpolicy.mojo"))
    out.append(
        String("noeira/deep_agents/training/driver_onpolicy_discrete.mojo")
    )
    return out^

comptime SAVE = "trainer.save_state(checkpoint_path)"
comptime ANNOUNCE = "announce_checkpoint(checkpoint_path, artifacts, run_dir)"

comptime EXPECT_SITES = 18
"""⚠ PINNED. If a driver gains or loses a save site this gate FAILS rather than
silently scanning a different amount of code — the number is the evidence that
the scan reached what it was written against."""


def _read(path: String) raises -> String:
    var bytes = read_file_bytes(path)
    bytes.append(0)
    return String(unsafe_from_utf8_ptr=bytes.unsafe_ptr())


def _lines(text: String) raises -> List[String]:
    var out = List[String]()
    for line in text.split("\n"):
        out.append(String(line))
    return out^


def main() raises:
    print("=== every save_state is announced ===")

    var sites = 0
    var violations = 0
    var files_scanned = 0

    var files = drivers()
    for i in range(len(files)):
        var path = String(files[i])
        var text = _read(path)
        files_scanned += 1
        var lines = _lines(text)

        for n in range(len(lines)):
            var line = String(lines[n]).strip()
            # ⚠ The docstring in driver_offpolicy MENTIONS the call. Only a
            # line that IS the call counts, or the gate would demand an
            # announce inside a comment.
            if String(line) != String(SAVE):
                continue
            sites += 1

            # The next line that is not blank must be the announce.
            var found = String("")
            for m in range(n + 1, len(lines)):
                var nxt = String(lines[m]).strip()
                if String(nxt).byte_length() == 0:
                    continue
                found = String(nxt)
                break
            if found.find(String(ANNOUNCE)) < 0:
                violations += 1
                print(
                    "    "
                    + path
                    + ":"
                    + String(n + 1)
                    + " saves a checkpoint and does not announce it"
                )
                print("      next line was: " + found)

    # ⚠ THE SCANNED COUNT IS PRINTED BESIDE THE FAILURE COUNT. "0 violations"
    # is also what a gate that read nothing reports.
    print(
        "  "
        + String(files_scanned)
        + " drivers scanned, "
        + String(sites)
        + " save sites, "
        + String(violations)
        + " unannounced"
    )

    if files_scanned != len(files):
        raise Error("a driver in the list was not read")
    if sites != EXPECT_SITES:
        raise Error(
            "expected "
            + String(EXPECT_SITES)
            + " save sites, found "
            + String(sites)
            + " — a driver gained or lost one, so update EXPECT_SITES"
            " DELIBERATELY rather than letting the scan drift"
        )
    if violations != 0:
        raise Error(
            String(violations)
            + " checkpoint(s) are written and never offered to the sink."
            " The artifact would stay on the box with nothing to see."
        )

    # ⚠ And the rule itself must live in ONE place. A driver that inlined the
    # offer instead of calling the helper would pass the check above while
    # reintroducing exactly the drift it exists to prevent.
    var inlined = 0
    for i in range(len(files)):
        var text = _read(String(files[i]))
        if text.find(String(".offer(")) >= 0:
            inlined += 1
            print("    " + String(files[i]) + " calls sink.offer() directly")
    if inlined != 0:
        raise Error(
            String(inlined)
            + " driver(s) inline the offer instead of calling"
            " announce_checkpoint — that is the rule written twice"
        )
    print("  and no driver inlines the offer; the rule has one home")

    # ── the facades forward it too, or the driver never sees it ─────
    #
    # ⚠⚠ A DRIVER PARAMETER NOTHING PASSES IS A NO-OP WITH GOOD DOCUMENTATION.
    # Almost every example calls a FACADE (`SACAgent.train`, `DQNAgent.train`)
    # rather than the driver directly, so `artifacts` has to be forwarded at
    # twenty-two sites across eleven `agent.mojo` files. One that forwards
    # `checkpoint_path` and not `artifacts` compiles, runs, trains, saves — and
    # never uploads, with nothing to see.
    #
    # Discovered the hard way: the first wiring pass changed the drivers only,
    # and the build failed with "unexpected keyword argument 'artifacts'"
    # because the facade in between knew nothing about it.
    var fwd = run_capture(
        String(
            "grep -rl 'checkpoint_path=checkpoint_path,'"
            " noeira/deep_agents/*/agent.mojo 2>/dev/null"
        ),
        1 << 20,
    )
    var facades = 0
    var unforwarded = 0
    for line in fwd.split("\n"):
        var f = String(line).strip()
        if String(f).byte_length() == 0:
            continue
        facades += 1
        var text = _read(String(f))
        var n_ckpt = 0
        var n_art = 0
        for l2 in text.split("\n"):
            var t = String(l2).strip()
            if t == "checkpoint_path=checkpoint_path,":
                n_ckpt += 1
            elif t == "artifacts=artifacts,":
                n_art += 1
        if n_art != n_ckpt:
            unforwarded += 1
            print(
                "    "
                + String(f)
                + " forwards checkpoint_path "
                + String(n_ckpt)
                + "x but artifacts "
                + String(n_art)
                + "x"
            )
    print(
        "  "
        + String(facades)
        + " facades forward a checkpoint_path, "
        + String(unforwarded)
        + " of them drop `artifacts`"
    )
    if facades == 0:
        raise Error(
            "no facade forwards checkpoint_path — the grep found nothing, so"
            " this check proved nothing"
        )
    if unforwarded != 0:
        raise Error(
            String(unforwarded)
            + " facade(s) forward the checkpoint path and not the sink. Those"
            " agents train, save, and never upload."
        )

    # ── facades with their OWN loop announce their own saves ───────
    #
    # TD-MPC2 and DreamerV3 do not call a shared driver: their facade runs the
    # loop and saves itself, so there is no `artifacts=` to forward and the
    # check above cannot see them. Each such save must be followed by the
    # announce, exactly as in the drivers.
    var own = run_capture(
        String(
            "grep -rlE 'self\\.save(_state)?\\(checkpoint_path\\)'"
            " noeira/deep_agents/*/agent*.mojo 2>/dev/null"
        ),
        1 << 20,
    )
    var own_files = 0
    var own_sites = 0
    var own_bad = 0
    for line in own.split("\n"):
        var f = String(String(line).strip())
        if f.byte_length() == 0:
            continue
        own_files += 1
        var ls = _lines(_read(f))
        for n in range(len(ls)):
            var t = String(String(ls[n]).strip())
            if t != "self.save(checkpoint_path)" and t != "self.save_state(checkpoint_path)":
                continue
            own_sites += 1
            var nxt = String("")
            for m in range(n + 1, len(ls)):
                var u = String(String(ls[m]).strip())
                if u.byte_length() > 0:
                    nxt = u
                    break
            if nxt.find("announce_checkpoint(checkpoint_path, artifacts, run_dir)") < 0:
                own_bad += 1
                print("    " + f + ":" + String(n + 1) + " saves and does not announce")
    print(
        "  " + String(own_files) + " facades save in their own loop, "
        + String(own_sites) + " sites, " + String(own_bad) + " unannounced"
    )
    if own_files == 0 or own_sites == 0:
        raise Error("no facade saves in its own loop — the grep found nothing")
    if own_bad != 0:
        raise Error(String(own_bad) + " facade save(s) never reach the sink")

    # ── the rule the source gate CANNOT see ─────────────────────────
    #
    # ⚠⚠ A SOURCE GATE PROVES THE CALL IS THERE, NOT THAT IT DOES ANYTHING.
    # A mutant that emptied `announce_checkpoint`'s body survived everything
    # above — eighteen sites still called it, and it still did nothing. So the
    # path decision is split into `offered_path`, which is pure, and gated
    # here by value.
    var run = String("/tmp/p/runs/2026-09-10_sac_abcd1234")
    var cases = [
        # (checkpoint path, run dir, expected artifact path)
        (run + "/checkpoints/best.ckpt", run, String("checkpoints/best.ckpt")),
        (run + "/checkpoints/last.ckpt", run, String("checkpoints/last.ckpt")),
        (run + "/metrics.csv", run, String("metrics.csv")),
        # ⚠ Outside the run: DROPPED. A driver still writing to a comptime
        # constant would otherwise file its checkpoint under a run it does not
        # belong to.
        (String("/tmp/other/best.ckpt"), run, String("")),
        (String("checkpoints/best.ckpt"), run, String("")),
        # ⚠ A path that merely SHARES A PREFIX with the run dir is not inside
        # it. Without the trailing separator, `<run>_2/best.ckpt` would be
        # filed under `<run>` as `_2/best.ckpt`.
        (run + "_2/checkpoints/best.ckpt", run, String("")),
        # The run dir itself is not an artifact.
        (run, run, String("")),
        (run + "/", run, String("")),
        # No run dir at all — a driver with no RunContext.
        (run + "/checkpoints/best.ckpt", String(""), String("")),
        (String(""), run, String("")),
    ]
    var compared = 0
    var wrong = 0
    for c in cases:
        compared += 1
        var got = offered_path(c[0], c[1])
        if got != c[2]:
            wrong += 1
            print(
                "    offered_path("
                + c[0]
                + ", "
                + c[1]
                + ") = '"
                + got
                + "', want '"
                + c[2]
                + "'"
            )
    print(
        "  offered_path: "
        + String(compared)
        + " compared, "
        + String(wrong)
        + " differing"
    )
    if wrong != 0 or compared != 10:
        raise Error(
            "offered_path: " + String(wrong) + " of " + String(compared)
            + " wrong"
        )

    # ── and the three lines that JOIN the two ───────────────────────
    #
    # ⚠⚠ `offered_path` is gated by value and the eighteen call sites are
    # gated by source, and a mutant that emptied `announce_checkpoint`'s body
    # still survived BOTH. Nothing above reaches the plumbing between them.
    #
    # The sink points at the discard port, so the upload fails immediately and
    # deterministically — what is being measured is that the artifact reached
    # the sink AT ALL, which the failure count proves and a no-op cannot fake.
    var dead_run = String("/tmp/noeira_announce_gate")
    var s1 = ArtifactSink(
        run_id=String("gate"),
        run_dir=dead_run,
        base_url=String("http://127.0.0.1:9"),
        api_key=String("k"),
    )
    announce_checkpoint(dead_run + "/checkpoints/best.ckpt", s1, dead_run)
    s1.close(drain_ms=4000)
    if s1.failed() + s1.abandoned() != 1:
        raise Error(
            "announce_checkpoint did not reach the sink: failed="
            + String(s1.failed())
            + " abandoned="
            + String(s1.abandoned())
            + " (want exactly 1 accounted for)"
        )

    # ...and the same call with a path OUTSIDE the run reaches it not at all.
    var s2 = ArtifactSink(
        run_id=String("gate"),
        run_dir=dead_run,
        base_url=String("http://127.0.0.1:9"),
        api_key=String("k"),
    )
    announce_checkpoint(String("/tmp/elsewhere/best.ckpt"), s2, dead_run)
    s2.close(drain_ms=2000)
    if s2.failed() + s2.abandoned() + s2.uploaded() != 0:
        raise Error(
            "a checkpoint outside the run directory was offered anyway:"
            " failed="
            + String(s2.failed())
            + " abandoned="
            + String(s2.abandoned())
        )
    print(
        "  announce_checkpoint reaches the sink for a path inside the run,"
        " and not at all for one outside it"
    )

    print("[PASS] checkpoints announce (" + String(sites) + " sites)")


# MUTANTS THIS FILE WAS CHECKED AGAINST (each must turn it red):
#   C1  a save site drops its announce            -> the source scan
#   C2  a driver inlines .offer() instead         -> the source scan
#   C3  announce_checkpoint's body is emptied     -> the sink check
#   C4  offered_path returns the ABSOLUTE path    -> offered_path by value
#   C5  the trailing-separator guard is removed   -> offered_path by value
#   C6  a path outside the run is kept anyway     -> offered_path by value
#   D1  a facade forwards checkpoint_path only    -> the facade scan
#
# ⚠⚠ C3 SURVIVED TWICE, AND THE FILE GREW TWICE BECAUSE OF IT. A SOURCE gate
# proves the call is THERE; it cannot prove the call DOES anything. Splitting
# `offered_path` out made the decision checkable by value and killed C4-C6 —
# and C3 survived even that, because the three lines JOINING the pure decision
# to `sink.offer` were still covered by nothing. Only pointing a real sink at
# the discard port and counting what reached it closes that.
#
# The general shape: a gate that checks structure and a gate that checks a pure
# function can both be green while the wire between them is cut.
#
# ⚠ D1 EXISTS BECAUSE THE FIRST WIRING PASS FORGOT THE FACADES ENTIRELY. The
# drivers took the new argument and the build failed with "unexpected keyword
# argument 'artifacts'" — which was lucky. Had the facades used **kwargs or a
# forwarding wrapper, they would have compiled and silently never uploaded.

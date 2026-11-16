# +--------------------------------------------------------------------------+ #
# | Private projects — two boxes, one platform, and nobody's edit erased
# +--------------------------------------------------------------------------+ #
"""Gate `core/project_sync.mojo` (who wins) and `data/project_sync.mojo` (doing it).

    pixi run mojo run -I . tests/core/test_project_sync.mojo

Hermetic: `tools/io/mock_monitor_server.py` plays the Worker and R2. Two
project roots under /tmp stand in for two machines.

⚠⚠ THE CHECK THAT MATTERS IS THE CONFLICT. A sync between machines has one
failure that costs something: box A pushing over an edit box B made after A
last synced, and nobody finding out. Checks 5 and 6 are that case from both
directions, and each asserts the BYTES that survived, not only the count a
report printed.

Part A is the decision table by value; part B runs it against the fixture.
"""

from std.collections import Dict
from std.os import makedirs
from std.os.path import exists
from std.time import sleep

from noeira.core.project_sync import (
    ACT_AHEAD,
    ACT_BEHIND,
    ACT_CONFLICT,
    ACT_DELETED_LOCALLY,
    ACT_DOWNLOAD,
    ACT_LOCAL_ONLY,
    ACT_REMOTE_ONLY,
    ACT_SAME,
    ACT_UPLOAD,
    load_base,
    pull_action,
    push_action,
    save_base,
    sync_refusal,
)
from noeira.data.project_sync import pull_definition, push_definition
from noeira.data.remote import RemoteCatalog
from noeira.io.fileio import remove_file, write_text_atomic
from noeira.io.http import HttpClient, http_shim_available
from noeira.io.proc import run_capture
from noeira.io.sha256 import sha256_file


comptime WORK = "/tmp/noeira_project_sync_gate"
comptime PORT_FILE = "/tmp/noeira_project_sync_gate_port"
comptime LOG_FILE = "/tmp/noeira_project_sync_gate_log"
comptime SLUG = "so101-tower"

comptime H1 = "1111111111111111111111111111111111111111111111111111111111111111"
comptime H2 = "2222222222222222222222222222222222222222222222222222222222222222"
comptime H3 = "3333333333333333333333333333333333333333333333333333333333333333"


def _expect(got: String, want: String, what: String) raises:
    if got != want:
        raise Error(what + ": expected '" + want + "', got '" + got + "'")


# =============================================================================
# Part A — the decision, by value
# =============================================================================


def part_a() raises -> Int:
    var n = 0
    var e = String("")

    # push: (local, remote, base, force)
    _expect(push_action(H1, H1, H1, False), ACT_SAME, "push same"); n += 1
    _expect(push_action(H1, e, e, False), ACT_UPLOAD, "push new file"); n += 1
    _expect(push_action(H2, H1, H1, False), ACT_UPLOAD, "push my edit over what I saw"); n += 1
    # ⚠⚠ THE ONE THAT MATTERS: the platform moved (H3) since I synced (H1),
    # and I edited too (H2). Uploading would erase their H3.
    _expect(push_action(H2, H3, H1, False), ACT_CONFLICT, "push both edited"); n += 1
    _expect(push_action(H2, H3, H1, True), ACT_UPLOAD, "push --force"); n += 1
    _expect(push_action(H1, H3, H1, False), ACT_BEHIND, "push, platform newer"); n += 1
    # Two boxes created the same path independently, with different bytes.
    _expect(push_action(H1, H2, e, False), ACT_CONFLICT, "push, both new"); n += 1
    _expect(push_action(e, H1, e, False), ACT_REMOTE_ONLY, "push, remote only"); n += 1
    _expect(push_action(e, H1, H1, False), ACT_DELETED_LOCALLY, "push, deleted here"); n += 1
    _expect(push_action(H1, e, H1, False), ACT_UPLOAD, "push, deleted on platform"); n += 1

    # pull
    _expect(pull_action(H1, H1, H1, False), ACT_SAME, "pull same"); n += 1
    _expect(pull_action(e, H1, e, False), ACT_DOWNLOAD, "pull fresh box"); n += 1
    _expect(pull_action(H1, H2, H1, False), ACT_DOWNLOAD, "pull fast-forward"); n += 1
    _expect(pull_action(H2, H3, H1, False), ACT_CONFLICT, "pull both edited"); n += 1
    _expect(pull_action(H2, H3, H1, True), ACT_DOWNLOAD, "pull --force"); n += 1
    _expect(pull_action(H2, H1, H1, False), ACT_AHEAD, "pull, edited here"); n += 1
    _expect(pull_action(H1, e, e, False), ACT_LOCAL_ONLY, "pull, local only"); n += 1
    _expect(pull_action(e, H1, H1, False), ACT_DELETED_LOCALLY, "pull, deleted here"); n += 1
    _expect(pull_action(H1, H2, e, False), ACT_CONFLICT, "pull, both new"); n += 1
    print("  A1 push/pull decision table: " + String(n) + " cases")

    # What counts as definition.
    var ok = List[String]()
    ok.append(String("project.kv"))
    ok.append(String("calibration/follower.json"))
    ok.append(String("policies/act.kv"))
    ok.append(String("tasks/cube_in_bowl.task"))
    for p in ok:
        _expect(sync_refusal(p), String(""), String("definition path ") + p)
        n += 1
    var bad = List[String]()
    bad.append(String("policies/act.ckpt"))
    bad.append(String("runs/r1/run.kv"))
    bad.append(String("artifacts/x.json"))
    bad.append(String("datasets/trial-01/meta/info.json"))
    bad.append(String(".DS_Store"))
    bad.append(String("calibration/.hidden"))
    bad.append(String("../escape.txt"))
    bad.append(String("calibration/follower (copy).json"))
    bad.append(String("data/episodes.h5"))
    bad.append(String("a/b/c/d/e/f/g/h/i.kv"))
    bad.append(String(""))
    var refused = 0
    for p in bad:
        if sync_refusal(p).byte_length() > 0:
            refused += 1
        else:
            print("    ACCEPTED: '" + p + "'")
    # ⚠ Both numbers. "0 accepted" is also what a function refusing everything
    # prints, and the accepted list above is what tells them apart.
    print(
        "  A2 definition paths: " + String(len(ok)) + " accepted, "
        + String(refused) + " of " + String(len(bad)) + " refused"
    )
    if refused != len(bad):
        raise Error("sync_refusal accepted a path it must refuse")
    n += 1

    # `.sync.kv` round-trips.
    var dir = String(WORK) + "/base_rt"
    makedirs(dir, exist_ok=True)
    var b = Dict[String, String]()
    b[String("project.kv")] = String(H1)
    b[String("calibration/follower.json")] = String(H2)
    save_base(dir, b)
    var back = load_base(dir)
    if len(back) != 2 or back[String("project.kv")] != H1 or back[
        String("calibration/follower.json")
    ] != H2:
        raise Error(".sync.kv did not round-trip")
    print("  A3 .sync.kv round-trips (2 entries)")
    n += 1
    return n


# =============================================================================
# Part B — two boxes against the fixture
# =============================================================================


def _start_server() raises -> String:
    for p in [String(PORT_FILE), String(LOG_FILE)]:
        try:
            remove_file(p)
        except:
            pass
    _ = run_capture(
        "python3 tools/io/mock_monitor_server.py " + String(PORT_FILE) + " "
        + String(LOG_FILE) + " 180 > /tmp/noeira_project_sync_gate_server.log 2>&1 &"
    )
    for _ in range(100):
        if exists(PORT_FILE):
            var f = open(String(PORT_FILE), "r")
            var port = String(f.read().strip())
            f.close()
            if port.byte_length() > 0:
                return "http://127.0.0.1:" + port
        sleep(0.1)
    raise Error("the mock monitor never wrote " + String(PORT_FILE))


def _puts() raises -> Int:
    """Object PUTs so far — a byte transfer, not a catalog call."""
    var f = open(String(LOG_FILE), "r")
    var text = String(f.read())
    f.close()
    var n = 0
    for line in text.split("\n"):
        var parts = String(line).split(" ")
        if len(parts) >= 3 and String(parts[1]) == "PUT" and String(parts[2]).startswith("/r2/"):
            n += 1
    return n


def _put(root: String, rel: String, text: String) raises:
    var full = root + "/" + SLUG + "/" + rel
    makedirs(String(full[byte=0 : full.rfind("/")]), exist_ok=True)
    write_text_atomic(full, text)


def _sha(root: String, rel: String) raises -> String:
    return sha256_file(root + "/" + SLUG + "/" + rel)


def _remote_sha(mut cat: RemoteCatalog, rel: String) raises -> String:
    var doc = cat.project_files(String(SLUG))
    var files = doc.field(doc.root(), String("files"))
    for i in range(doc.size(files)):
        var row = doc.at(files, i)
        if doc.string(doc.field(row, String("path"))) == rel:
            return doc.string(doc.field(row, String("sha256")))
    return String("")


def _check(cond: Bool, what: String) raises:
    if not cond:
        raise Error(what)


def part_b(base_url: String) raises -> Int:
    var n = 0
    var A = String(WORK) + "/box_a"
    var B = String(WORK) + "/box_b"
    var cat = RemoteCatalog(base_url, String("gate-key"))

    _put(A, String("project.kv"), String("schema_version=1\nname=so101-tower\ndescription=tower rig\n"))
    _put(A, String("calibration/follower.json"), String("{\"wrist_roll\": [682, 3412]}\n"))
    _put(A, String("policies/act.kv"), String("name=act\nrun_id=r1\n"))
    _put(A, String("policies/act.ckpt"), String("WEIGHTS"))
    _put(A, String("runs/r1/run.kv"), String("run_id=r1\n"))
    _put(A, String("datasets/trial-01/meta/info.json"), String("{\"fps\": 30}\n"))
    _put(A, String(".DS_Store"), String("x"))
    var big = String("")
    for _ in range(5000):
        big += "z"
    _put(A, String("notes/big.json"), big)

    # ── 1. first push: three files, and the rest skipped out loud ──────
    var r1 = push_definition(cat, String(SLUG), A, String("tower rig"))
    r1.print_all(String("pushed"))
    _check(r1.moved == 3 and r1.conflicts == 0, "B1: expected 3 pushed")
    # ckpt, .DS_Store, the 5 KB file over the fixture's 4096 cap. `runs/` is
    # not listed per file.
    _check(len(r1.skipped) == 3, "B1: expected 3 skipped, got " + String(len(r1.skipped)))
    _check(_remote_sha(cat, String("policies/act.ckpt")) == "", "B1: weights reached the platform")
    _check(
        _remote_sha(cat, String("datasets/trial-01/meta/info.json")) == "",
        "B1: a recorded dataset's metadata reached the platform as definition",
    )
    # ⚠ The upload must be RECORDED. Without it the next pull, after anyone
    # else's push, plans this box's own files as edited-on-both-sides and
    # reports a conflict nobody made.
    var a_base = load_base(A + "/" + SLUG)
    _check(len(a_base) == 3, "B1: box A's .sync.kv should hold 3 files, got " + String(len(a_base)))
    _check(
        a_base.get(String("project.kv"), String("")) == _sha(A, String("project.kv")),
        "B1: .sync.kv does not record what was uploaded",
    )
    print("  B1 first push: 3 pushed, 3 skipped (ckpt, hidden, over cap), runs/ untouched")
    n += 1

    # ── 2. a second push moves no bytes ────────────────────────────────
    var puts0 = _puts()
    var r2 = push_definition(cat, String(SLUG), A)
    _check(r2.moved == 0 and r2.same == 3, "B2: a no-change push must be all `same`")
    _check(_puts() == puts0, "B2: a no-change push transferred bytes")
    print("  B2 re-push: 0 moved, 3 same, 0 object PUTs")
    n += 1

    # ── 2b. --dry-run: the push's plan, and nothing written ────────────
    # (`project-push --dry-run --definition-only` listed runs and checkpoints instead, 2026-11-16)
    _put(A, String("notes/plan.md"), String("draft\n"))
    var base_before = load_base(A + "/" + SLUG)
    var puts1 = _puts()
    var rd = push_definition(cat, String(SLUG), A, dry=True)
    rd.print_all(String("would be pushed"))
    _check(rd.moved == 1 and rd.same == 3, "B2b: the dry run should plan 1 upload and 3 same")
    _check(_puts() == puts1, "B2b: a DRY RUN transferred bytes")
    _check(_remote_sha(cat, String("notes/plan.md")) == "", "B2b: a dry run put a file on the platform")
    var base_after = load_base(A + "/" + SLUG)
    _check(
        len(base_after) == len(base_before) and String("notes/plan.md") not in base_after,
        "B2b: a dry run wrote .sync.kv",
    )
    # a project the platform has never seen: planned in full, and NOT created
    var C = String(WORK) + "/box_c"
    makedirs(C + "/ghost", exist_ok=True)
    write_text_atomic(C + "/ghost/project.kv", String("schema_version=1\nname=ghost\ndescription=x\n"))
    var rg = push_definition(cat, String("ghost"), C, dry=True)
    _check(rg.moved == 1, "B2b: an unknown project's dry run should plan its one file")
    var created = True
    try:
        _ = cat.project_files(String("ghost"))
    except:
        created = False
    _check(not created, "B2b: a dry run CREATED the project on the platform")
    _check(not exists(C + "/ghost/.sync.kv"), "B2b: a dry run wrote the unknown project's .sync.kv")
    remove_file(A + "/" + SLUG + "/notes/plan.md")
    print("  B2b dry run: 1 would upload, 0 object PUTs, platform and .sync.kv unchanged; unknown project not created")
    n += 1

    # ── 3. a fresh box pulls the definition, byte for byte ─────────────
    var r3 = pull_definition(cat, String(SLUG), B)
    _check(r3.moved == 3, "B3: expected 3 pulled, got " + String(r3.moved))
    for rel in [String("project.kv"), String("calibration/follower.json"), String("policies/act.kv")]:
        _check(_sha(B, rel) == _sha(A, rel), "B3: bytes differ for " + rel)
    _check(not exists(B + "/" + SLUG + "/policies/act.ckpt"), "B3: weights pulled")
    _check(len(load_base(B + "/" + SLUG)) == 3, "B3: box B's .sync.kv should hold 3 files")
    print("  B3 fresh-box pull: 3 files, identical sha256, no weights, base of 3")
    n += 1

    # ── 4. box B edits and pushes: a plain upload ──────────────────────
    _put(B, String("calibration/follower.json"), String("{\"wrist_roll\": [700, 3394]}\n"))
    var b_sha = _sha(B, String("calibration/follower.json"))
    var r4 = push_definition(cat, String(SLUG), B)
    _check(r4.moved == 1 and r4.conflicts == 0, "B4: expected box B's edit to upload")
    print("  B4 box B edits calibration and pushes: 1 uploaded")
    n += 1

    # ── 5. ⚠⚠ box A edited the same file without pulling: REFUSED ─────
    _put(A, String("calibration/follower.json"), String("{\"wrist_roll\": [0, 4095]}\n"))
    var a_sha = _sha(A, String("calibration/follower.json"))
    var r5 = push_definition(cat, String(SLUG), A)
    _check(r5.conflicts == 1 and r5.moved == 0, "B5: box A's push must conflict")
    _check(
        _remote_sha(cat, String("calibration/follower.json")) == b_sha,
        "B5: BOX B'S EDIT WAS ERASED on the platform",
    )
    print("  B5 box A pushes a stale edit: conflict, the platform still holds box B's bytes")
    n += 1

    # ── 6. ...and pulling does not overwrite box A's edit either ──────
    var r6 = pull_definition(cat, String(SLUG), A)
    _check(r6.conflicts == 1 and r6.moved == 0, "B6: box A's pull must conflict")
    _check(
        _sha(A, String("calibration/follower.json")) == a_sha,
        "B6: BOX A'S EDIT WAS OVERWRITTEN by a pull",
    )
    print("  B6 box A pulls: conflict, box A's own bytes untouched")
    n += 1

    # ── 7. --force on pull: the platform wins ──────────────────────────
    var r7 = pull_definition(cat, String(SLUG), A, force=True)
    _check(r7.moved == 1, "B7: --force pull should download 1")
    _check(_sha(A, String("calibration/follower.json")) == b_sha, "B7: --force did not take the platform's bytes")
    print("  B7 box A pulls --force: takes box B's calibration")
    n += 1

    # ── 8. an unedited file fast-forwards ──────────────────────────────
    _put(B, String("policies/act.kv"), String("name=act\nrun_id=r2\n"))
    _ = push_definition(cat, String(SLUG), B)
    var r8 = pull_definition(cat, String(SLUG), A)
    _check(r8.moved == 1 and r8.conflicts == 0, "B8: an unedited file should just download")
    _check(_sha(A, String("policies/act.kv")) == _sha(B, String("policies/act.kv")), "B8: bytes differ")
    print("  B8 box B re-promotes; box A pulls: 1 downloaded, no conflict")
    n += 1

    # ── 9. a local deletion is neither propagated nor undone ───────────
    remove_file(A + "/" + SLUG + "/policies/act.kv")
    var r9 = pull_definition(cat, String(SLUG), A)
    _check(not exists(A + "/" + SLUG + "/policies/act.kv"), "B9: a pull resurrected a deleted file")
    _ = push_definition(cat, String(SLUG), A)
    _check(_remote_sha(cat, String("policies/act.kv")).byte_length() == 64, "B9: a push deleted a file on the platform")
    _check(r9.other >= 1, "B9: the deletion was not reported")
    print("  B9 box A deletes a file: pull does not restore it, push does not delete it")
    n += 1

    # ── 10. a path the platform should never have sent is refused ──────
    var http = HttpClient(5000, 5000)
    for bad in [String("../escape.txt"), String("runs/r9/evil.kv")]:
        var body = String("{\"slug\":\"") + SLUG + "\",\"path\":\"" + bad + "\",\"sha256\":\"" + H3 + "\"}"
        var payload = List[UInt8]()
        for i in range(body.byte_length()):
            payload.append(body.as_bytes()[i])
        _ = http.request(String("POST"), base_url + "/__inject_file", payload^, String("application/json"), 200)
    var r10 = pull_definition(cat, String(SLUG), A)
    _check(not exists(A + "/escape.txt"), "B10: a traversal path from the server was WRITTEN")
    _check(not exists(A + "/" + SLUG + "/runs/r9/evil.kv"), "B10: run data from the server was written")
    var refused = 0
    for s in r10.skipped:
        if s.find("refused from the platform") >= 0:
            refused += 1
    _check(refused == 2, "B10: expected 2 refused server paths, got " + String(refused))
    print("  B10 server sends ../escape.txt and runs/...: both refused, nothing written")
    n += 1

    # ── 11. bytes that do not hash to the platform's row are not written ─
    var tb = String("{\"slug\":\"") + SLUG + "\",\"path\":\"calibration/tampered.json\",\"sha256\":\"" + H3 + "\",\"body\":\"not the right bytes\"}"
    var tp = List[UInt8]()
    for i in range(tb.byte_length()):
        tp.append(tb.as_bytes()[i])
    _ = http.request(String("POST"), base_url + "/__inject_file", tp^, String("application/json"), 200)
    var raised = False
    try:
        _ = pull_definition(cat, String(SLUG), A)
    except e:
        raised = String(e).find("not written") >= 0
    _check(raised, "B11: a pull accepted bytes that do not match the row's sha256")
    _check(
        not exists(A + "/" + SLUG + "/calibration/tampered.json"),
        "B11: mismatching bytes were written",
    )
    print("  B11 served bytes do not match the row's sha256: refused, not written")
    n += 1
    return n


def main() raises:
    print("[project-sync] gate")
    _ = run_capture(String("rm -rf ") + WORK)
    makedirs(String(WORK), exist_ok=True)
    var a = part_a()
    if not http_shim_available():
        raise Error("the http shim is not built — pixi run build-http")
    var url = _start_server()
    var b = 0
    try:
        b = part_b(url)
    finally:
        try:
            var h = HttpClient(2000, 2000)
            _ = h.request(String("GET"), url + "/__shutdown")
        except:
            pass
    print("  " + String(a + b) + " checks, 0 failures")
    print("[PASS] project sync (" + String(a + b) + " checks)")

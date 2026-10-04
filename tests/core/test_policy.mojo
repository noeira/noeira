# +--------------------------------------------------------------------------+ #
# | Promotion: a role, its weights, and the history it keeps for free
# +--------------------------------------------------------------------------+ #
"""Gate `noeira/core/policy.mojo` and `project-promote` — §8.

    pixi run mojo run -I . tests/core/test_policy.mojo

## What could be wrong, and what each check is for

* **The record round-trips wrong.** `policies/<name>.kv` is the answer to
  "which run is deployed right now?"; a field that does not survive a write is
  a question that gets the wrong answer months later.
* **⚠⚠ The weights are a SYMLINK.** §8 names three ways that breaks, all of
  which surface much later as the same confusing failure: `project-pull` to
  another machine brings the link and not its target, rsync/sftp either follow
  it or copy a dangling link, and `project-prune` turns it into a dead pointer.
  So the gate checks the INODE, not just that a file is there.
* **⚠ The materialised bytes differ from the source.** A hard link cannot
  differ, but the `cp` fallback can be short — and truncated weights are the
  worst possible thing to hand a real robot.
* **`supersedes` is lost.** Then each promotion overwrites the history that
  made it reviewable, which is the one thing a naming convention never gave.
* **Promoting does not repoint the role.** The whole point of decision 2 is
  that the deploy path names a role and never a run; a promotion that leaves
  the old weights in place is a deploy that silently keeps using them.
"""

from std.os.path import exists

from noeira.core.policy import (
    PolicyRecord,
    describe_policy,
    resolve_policy,
    load_policy,
    parse_policy,
    policy_ckpt_path,
    policy_kv_path,
    write_policy,
)
from noeira.io.fileio import write_file_atomic
from noeira.io.proc import quote_arg, run_capture
from noeira.io.sha256 import sha256_file


comptime TMP = "/tmp/noeira_policy_gate"


def _blob(path: String, seed: Int, n: Int) raises:
    var cut = path.rfind("/")
    if cut > 0:
        _ = run_capture("mkdir -p " + quote_arg(String(path[byte=0:cut])))
    var bytes = List[UInt8]()
    for i in range(n):
        bytes.append(UInt8((i * 17 + seed * 5 + 3) & 0xFF))
    write_file_atomic(path, bytes)


def _inode(path: String) raises -> String:
    var line = String(run_capture("ls -i " + quote_arg(path), 1 << 12).strip())
    var parts = line.split(" ")
    return String(parts[0]) if len(parts) > 0 else String("")


def _is_symlink(path: String) raises -> Bool:
    # ⚠ `|| true`: `test -L` EXITS 1 when the answer is "no", and `run_capture`
    # raises on a non-zero child. Without it the gate fails on exactly the
    # outcome it is hoping for.
    return (
        run_capture(
            "test -L " + quote_arg(path) + " && echo YES || true", 1 << 12
        ).find("YES")
        >= 0
    )


def _promote(rid: String, which: String, name: String, note: String) raises -> String:
    return run_capture(
        "NOEIRA_PROJECTS=" + quote_arg(String(TMP) + "/projects")
        + " pixi run mojo run -I . tools/project/project_cli.mojo promote "
        + quote_arg(rid) + " " + quote_arg(which) + " --as " + quote_arg(name)
        + (" --note " + quote_arg(note) if note.byte_length() > 0 else "")
        # ⚠⚠ `--no-push` IS NOT OPTIONAL HERE. `project-promote` records every
        # promotion on the monitor named in `.env`, which for a developer
        # running this is their REAL one — a gate that writes rows into
        # production is a defect, not a thorough test. The POST path is
        # exercised against the fixture in the worker's own suite.
        + " --no-push"
        # ⚠ `|| true`: check 8 promotes a checkpoint that does not exist ON
        # PURPOSE, and the CLI is supposed to exit non-zero for it. Without
        # this the gate cannot observe its own refusal.
        + " 2>&1 || true",
        1 << 20,
    )


def _make_run(rid: String, ckpt_seed: Int) raises -> String:
    var d = String(TMP) + "/projects/so101/runs/" + rid
    _blob(d + "/checkpoints/best.ckpt", ckpt_seed, 8192)
    _blob(d + "/checkpoints/last.ckpt", ckpt_seed + 100, 4096)
    var kv = (
        "schema_version=1\nrun_id=" + rid + "\nproject=so101\ndriver=gate\n"
        "status=done\noutcome=success_rate=0.5\n"
    )
    var bytes = List[UInt8]()
    for i in range(kv.byte_length()):
        bytes.append(kv.as_bytes()[i])
    write_file_atomic(d + "/run.kv", bytes)
    return d


def main() raises:
    print("=== promotion (§8) ===")
    var checks = 0

    # ── 1. the record round-trips ───────────────────────────────────
    var p = PolicyRecord(String("reach"))
    p.run = String("2026-09-02_act-reach_a3f21c8b")
    p.checkpoint = String("best")
    p.sha256 = String("9f1c2b")
    p.bytes = 215_000_000
    p.promoted = String("2026-09-02T14:40:00Z")
    p.note = String("8/10 on the real arm, gripper ok")
    p.supersedes = String("2026-08-30_act-reach_91b0f2aa")
    var back = parse_policy(p.render(), String("gate"))
    var fields = 0
    var wrong = 0
    for pair in [
        (back.name, p.name),
        (back.run, p.run),
        (back.checkpoint, p.checkpoint),
        (back.sha256, p.sha256),
        (back.promoted, p.promoted),
        (back.note, p.note),
        (back.supersedes, p.supersedes),
    ]:
        fields += 1
        if pair[0] != pair[1]:
            wrong += 1
            print("    '" + pair[0] + "' != '" + pair[1] + "'")
    if back.bytes != p.bytes:
        wrong += 1
        print("    bytes " + String(back.bytes) + " != " + String(p.bytes))
    fields += 1
    print("  record: " + String(fields) + " fields compared, " + String(wrong) + " differing")
    if wrong != 0 or fields != 8:
        raise Error("policy record round trip: " + String(wrong) + " wrong")
    checks += 1

    # ── the CLI, end to end ─────────────────────────────────────────
    _ = run_capture("rm -rf " + quote_arg(String(TMP)))
    _ = run_capture("mkdir -p " + quote_arg(String(TMP) + "/projects/so101"))
    var pkv = "schema_version=1\nname=so101\ncreated=2026-09-10T00:00:00Z\n"
    var pb = List[UInt8]()
    for i in range(pkv.byte_length()):
        pb.append(pkv.as_bytes()[i])
    write_file_atomic(String(TMP) + "/projects/so101/project.kv", pb)

    var rid_a = String("2026-09-10_gate-a_aaaa1111")
    var dir_a = _make_run(rid_a, 1)
    var out = _promote(rid_a, String("best"), String("reach"), String("first one that worked"))
    var pdir = String(TMP) + "/projects/so101"
    var role = policy_ckpt_path(pdir, String("reach"))
    if not exists(role):
        raise Error("promote did not materialise the role:\n" + out)

    # ── 2. ⚠⚠ HARD LINK, NOT A SYMLINK ──────────────────────────────
    if _is_symlink(role):
        raise Error(
            "the role is a SYMLINK. It breaks on project-pull, under rsync,"
            " and the moment project-prune removes the run."
        )
    var src_a = dir_a + "/checkpoints/best.ckpt"
    var same_inode = _inode(role) == _inode(src_a)
    print(
        "  materialised: not a symlink; "
        + ("hard-linked (same inode)" if same_inode else "copied")
    )
    checks += 1

    # ── 3. ⚠ the bytes at the role equal the bytes in the run ───────
    if sha256_file(role) != sha256_file(src_a):
        raise Error("the materialised policy does not match the checkpoint")
    print("  bytes: the role hashes to the run's checkpoint")
    checks += 1

    # ── 4. the record names the run, and the FIRST has no predecessor ─
    var rec = load_policy(policy_kv_path(pdir, String("reach")))
    if rec.run != rid_a or rec.checkpoint != "best":
        raise Error("the record does not name the run it came from: " + rec.run)
    if rec.supersedes.byte_length() != 0:
        raise Error("a first promotion claims to supersede " + rec.supersedes)
    if rec.note.find("first one that worked") < 0:
        raise Error("the note was lost: '" + rec.note + "'")
    print("  record: names the run, keeps the note, supersedes nothing yet")
    checks += 2

    # ── 5. ⚠⚠ RE-PROMOTING REPOINTS THE ROLE AND RECORDS THE HISTORY ─
    #
    # If the role still held the old weights, a deploy would keep using them
    # while the record claimed otherwise — the exact lie this layer exists to
    # make impossible.
    var rid_b = String("2026-09-10_gate-b_bbbb2222")
    var dir_b = _make_run(rid_b, 42)
    _ = _promote(rid_b, String("best"), String("reach"), String("better, 9/10"))
    var rec2 = load_policy(policy_kv_path(pdir, String("reach")))
    if rec2.run != rid_b:
        raise Error("re-promotion did not update the record: " + rec2.run)
    if rec2.supersedes != rid_a:
        raise Error(
            "supersedes should be '" + rid_a + "', is '" + rec2.supersedes + "'"
        )
    if sha256_file(role) != sha256_file(dir_b + "/checkpoints/best.ckpt"):
        raise Error(
            "the ROLE still holds the old weights after a re-promotion — a"
            " deploy would keep using them"
        )
    if sha256_file(role) == sha256_file(src_a):
        raise Error("the role did not change at all")
    print("  re-promote: the role moved to the new run, supersedes " + rec2.supersedes)
    checks += 3

    # ── 6. the old run is untouched ─────────────────────────────────
    #
    # ⚠ Promotion COPIES a role out; it must not move or consume the run's own
    # checkpoint, or the run record becomes a lie about what it produced.
    if not exists(src_a):
        raise Error("promotion consumed the source checkpoint")
    print("  the promoted run keeps its own checkpoint")
    checks += 1

    # ── 7. a second role is independent ─────────────────────────────
    _ = _promote(rid_a, String("last"), String("settle"), String(""))
    var settle = policy_ckpt_path(pdir, String("settle"))
    if not exists(settle):
        raise Error("a second role was not materialised")
    if sha256_file(settle) != sha256_file(dir_a + "/checkpoints/last.ckpt"):
        raise Error("the second role holds the wrong checkpoint")
    if sha256_file(role) != sha256_file(dir_b + "/checkpoints/best.ckpt"):
        raise Error("promoting `settle` disturbed `reach`")
    print("  a second role is independent of the first")
    checks += 2

    # ── 8. a checkpoint that does not exist is REFUSED, and says what is ─
    var bad = _promote(rid_a, String("step_999999"), String("nope"), String(""))
    if bad.find("no checkpoint") < 0:
        raise Error("promoting a missing checkpoint was not refused:\n" + bad)
    if bad.find("best.ckpt") < 0:
        raise Error(
            "the refusal did not say what IS there, which is the whole"
            " difference between a useful error and 'not found':\n" + bad
        )
    if exists(policy_ckpt_path(pdir, String("nope"))):
        raise Error("a refused promotion still materialised a role")
    print("  a missing checkpoint is refused, and the refusal lists what exists")
    checks += 2

    # ── 9. ⚠⚠ THE DEPLOY PATH RESOLVES THE ROLE, NOT A RUN ──────────
    #
    # This is decision 2, and it is the only reason the rest of the file
    # matters: `deploy_reach_real` asks for "reach" and gets whatever was last
    # promoted, with no run id anywhere on the deploy path.
    var proot = String(TMP) + "/projects"
    var got = resolve_policy(
        String("so101"), String("reach"), String("FLAT.ckpt"), proot
    )
    if got != role:
        raise Error("resolve_policy returned '" + got + "', want the role " + role)
    # ...and an UNPROMOTED role falls back rather than raising. A deploy path is
    # the one place where failing closed is worse than failing open.
    var fb = resolve_policy(
        String("so101"), String("no_such_role"), String("FLAT.ckpt"), proot
    )
    if fb != "FLAT.ckpt":
        raise Error("an unpromoted role did not fall back: '" + fb + "'")
    print("  deploy: the role resolves; an unpromoted one falls back")
    checks += 2

    # ── 10. and the provenance says WHICH run, with the human's note ─
    var prov = describe_policy(String("so101"), String("reach"), proot)
    if prov.find(rid_b) < 0:
        raise Error("the provenance line does not name the run: " + prov)
    if prov.find("better, 9/10") < 0:
        raise Error("the provenance line lost the note: " + prov)
    if describe_policy(String("so101"), String("nope"), proot) != "":
        raise Error("an unpromoted role described itself anyway")
    print("  deploy: the provenance names the run and carries the note")
    checks += 2

    # ── 11. ⚠ a record whose WEIGHTS are gone must not stay silent ───
    _ = run_capture("rm -f " + quote_arg(role))
    var shown = run_capture(
        "NOEIRA_PROJECTS=" + quote_arg(proot)
        + " pixi run mojo run -I . tools/project/project_cli.mojo show so101"
        + " 2>&1 || true",
        1 << 20,
    )
    if shown.find("MISSING WEIGHTS") < 0:
        raise Error(
            "a policy record with no weights was listed as if it were fine —"
            " a deploy would find nothing at the role:\n" + shown
        )
    print("  a record whose weights are gone is flagged, not listed as fine")
    checks += 1

    # ── 12. ⚠ `--no-push` really does keep it local ─────────────────
    #
    # Every promotion above passed it, so if the flag were ignored this whole
    # file would have been writing into a real account. That makes this the
    # check the rest of the file depends on.
    var quiet = _promote(rid_a, String("best"), String("checkflag"), String(""))
    if quiet.find("monitor") >= 0:
        raise Error(
            "--no-push was ignored; the gate is talking to a real monitor:\n"
            + quiet
        )
    print("  --no-push keeps a promotion local, as every check above assumed")
    checks += 1

    # ── 13. the normalization travels with the weights ──────────────
    #
    # A policy's checkpoint is not deployable without its run's norm.json;
    # promotion is what carries it to `policies/<name>.norm.json`, where
    # `project-push` syncs it and a deployment elsewhere finds it.
    var rid_n = String("2026-09-15_act-so101_n0rm0001")
    var dn = _make_run(rid_n, 91)
    _blob(dn + "/checkpoints/norm.json", 5, 300)
    _ = _promote(rid_n, String("best"), String("deployed"), String(""))
    var pol_norm = String(TMP) + "/projects/so101/policies/deployed.norm.json"
    if not exists(pol_norm) or sha256_file(pol_norm) != sha256_file(dn + "/checkpoints/norm.json"):
        raise Error("promote did not carry the run's norm.json to " + pol_norm)
    # ...and a later promotion of a run WITHOUT one must not leave the old
    # statistics beside new weights — the one mix-up nothing downstream detects.
    _ = _promote(rid_a, String("best"), String("deployed"), String(""))
    if exists(pol_norm):
        raise Error("promoting a run with no norm.json left the previous policy's statistics in place")
    print("  norm.json is carried to policies/<role>.norm.json, and removed when the new holder has none")
    checks += 2

    # ── 14. ⚠ an FB checkpoint's OWN sidecar travels too ────────────
    #
    # FB/BFM runs write no norm.json: `ObsNorm` lives beside each checkpoint
    # as `<ckpt>.norm`, and the G1 viewers read `ckpt + ".norm"`. A role
    # without it runs on RAW inputs after one warning line.
    var rid_s = String("2026-09-28_bfm-zero-g3_s1dec4r0")
    var ds = _make_run(rid_s, 77)
    _blob(ds + "/checkpoints/best.ckpt.norm", 9, 500)
    _ = _promote(rid_s, String("best"), String("bfm"), String(""))
    var pol_side = String(TMP) + "/projects/so101/policies/bfm.ckpt.norm"
    if not exists(pol_side) or sha256_file(pol_side) != sha256_file(ds + "/checkpoints/best.ckpt.norm"):
        raise Error("promote did not carry the checkpoint's .norm sidecar to " + pol_side)
    # ⚠ the SIDECAR OF THE CHECKPOINT PROMOTED, not any sidecar in the run
    _blob(ds + "/checkpoints/last.ckpt.norm", 13, 500)
    _ = _promote(rid_s, String("last"), String("bfm"), String(""))
    if sha256_file(pol_side) != sha256_file(ds + "/checkpoints/last.ckpt.norm"):
        raise Error("re-promoting `last` kept `best`'s sidecar at " + pol_side)
    # ...and a holder without one must not inherit the previous statistics
    _ = _promote(rid_a, String("best"), String("bfm"), String(""))
    if exists(pol_side):
        raise Error("promoting a checkpoint with no .norm left the previous holder's beside new weights")
    print("  <ckpt>.norm is carried to policies/<role>.ckpt.norm, per checkpoint, and removed when absent")
    checks += 3

    _ = run_capture("rm -rf " + quote_arg(String(TMP)))
    print("[PASS] policy (" + String(checks) + " checks)")


# MUTANTS THIS FILE WAS CHECKED AGAINST (each must turn it red):
#   E1  render() drops `supersedes`            -> check 1
#   E2  render() drops `note`                  -> check 1
#   E3  promote uses `ln -s` (a symlink)       -> check 2
#   E4  promote reads supersedes AFTER writing -> check 5
#   E5  promote does not replace an existing role -> check 5
#   E6  promote `mv`s instead of linking       -> check 6
#   E7  a missing checkpoint is not refused    -> check 8
#   E8  resolve_policy ignores the role         -> check 9
#   E9  resolve_policy raises instead of falling back -> check 9
#   E10 project-show hides missing weights      -> check 11
#   E11 --no-push is ignored                    -> check 12
#   E12 promote does not copy norm.json        -> check 13
#   E13 promote leaves a stale norm.json        -> check 13
#   E14 promote does not copy <ckpt>.norm       -> check 14
#   E15 promote copies best's sidecar for `last` -> check 14
#   E16 promote leaves a stale <ckpt>.norm      -> check 14

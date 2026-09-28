"""`project` — init, list, show, tag and prune, over `projects/<name>/`.

    pixi run project-init so101 --description "SO-101 tabletop manipulation"
    pixi run project-list
    pixi run project-show so101
    pixi run run-tag <run_id> "meilleur reach à ce jour, testé 8/10"
    pixi run project-prune so101 --older-than 30 [--apply]
    pixi run project-promote <run_id> best --as reach --note "8/10 on the arm"
        (add --no-push to keep it local; the box is the record either way)
    pixi run project-push so101 [<run_id>] [--kind checkpoint|eval|log] [--force]
    pixi run project-pull so101 [<run_id>] [--kind checkpoint] [--force]
    pixi run project-pull so101 --weights   # definition + promoted policies' weights
    pixi run project-list --remote
    pixi run project-show so101 --remote    # the project's runs on the monitor
    pixi run project-rerun <run_id> [--exec]    # how to run it again, exactly
    pixi run project-resume <run_id> [--from last] [--dry-run]

⚠⚠ PROJECTS ARE PRIVATE AND NOT IN GIT. `projects/` is ignored, so a project's
DEFINITION (`project.kv`, calibration, task files, policy records) reaches a
second box through the platform: `project-push so101` sends it before the run
artifacts, and `project-pull so101` with no run id recreates it on a fresh
box. Who wins when two boxes edited the same file is `core/project_sync.mojo`;
a conflict overwrites nothing unless `--force` is given. Put flags AFTER the
positional arguments.

⚠⚠ EVERYTHING HERE IS LOCAL AND COMPLETE. `docs/PROJECT_LAYER_PLAN.md` §10: the
OSS half is a project on ONE machine with nothing crippled, and the paid half is
the same project on SEVERAL. A local-only user who cannot tag, find or prune
their runs will not become a paying one — they will keep their own directory
convention and never adopt the layer at all.

⚠ `project-push` / `project-pull` ARE THE ONE PAID-HALF PAIR (§10, decision 6):
everything above works on a single machine with no account. They need
`NOEIRA_CLOUD_URL` + `NOEIRA_CLOUD_API_KEY` in `.env` — no new credential.

⚠ `project-prune` IS DRY-RUN BY DEFAULT and needs `--apply`. It deletes run
directories; the one thing worse than 219 flat checkpoints is a tool that
removes the one you wanted.
"""

from std.sys import argv
from std.os.path import exists

from noeira.core.project import (
    ProjectSpec,
    check_refs,
    load_project,
    projects_root,
    project_exists,
)
from noeira.core.policy import (
    PolicyRecord,
    load_policy,
    policy_ckpt_path,
    policy_kv_path,
    write_policy,
)
from noeira.core.run import (
    RunRecord,
    epoch_seconds,
    git_commit,
    git_dirty,
    iso8601_utc,
    load_run,
    resume_args_for,
    run_command,
)
from noeira.data.policy_weights import pull_policy_weights
from noeira.data.project_sync import pull_definition, push_definition
from noeira.data.remote import RemoteCatalog
from noeira.io.fetch import fetch_to_cache
from noeira.io.fileio import file_size
from noeira.io.json import JsonDoc
from noeira.io.proc import quote_arg, run_capture, run_system
from noeira.io.sha256 import sha256_file


def _flag(name: String, dflt: String) raises -> String:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def _has(name: String) -> Bool:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            return True
    return False


def _positional(n: Int) -> String:
    """The n-th argument that is not a flag or a flag's value."""
    var av = argv()
    var seen = 0
    var i = 1
    while i < len(av):
        var a = String(av[i])
        if a.startswith("--"):
            i += 2
            continue
        if seen == n:
            return a
        seen += 1
        i += 1
    return String("")


def _ls(path: String) raises -> List[String]:
    """Entries of a directory, sorted. Empty when it does not exist."""
    var out = List[String]()
    if not exists(path):
        return out^
    var txt = run_capture(
        String("ls -1 ") + quote_arg(path) + " 2>/dev/null", 1 << 20
    )
    for line in txt.split("\n"):
        var s = String(line.strip())
        if s.byte_length() > 0:
            out.append(s)
    return out^


# =============================================================================


def cmd_init() raises:
    var name = _positional(1)
    if name.byte_length() == 0:
        raise Error("usage: project-init <name> [--description ...]")
    var root = projects_root()
    if project_exists(name, root):
        raise Error(
            "project '" + name + "' already exists at " + root + "/" + name
            + "/project.kv — refusing to overwrite a definition"
        )
    _ = run_capture(
        String("mkdir -p ") + quote_arg(root + "/" + name + "/runs")
        + " " + quote_arg(root + "/" + name + "/policies") + " 2>&1", 4096
    )
    var p = ProjectSpec(name, root)
    p.created = iso8601_utc(epoch_seconds())
    p.description = _flag(String("--description"), String(""))
    p.write()
    print("created", p.kv_path())
    print()
    print("⚠ projects/ is not in git. Sync the definition through noeira cloud:")
    print("    pixi run project-push", name)
    print()
    print("Add what the project uses, as pinned references:")
    print("    ref=family:<name>:<path>@<commit>")
    print("    ref=task:<name>:<path>@<commit>")


def cmd_list_remote() raises:
    var cat = RemoteCatalog.from_env()
    var doc = cat.list_projects()
    var root = doc.root()
    var n = doc.size(root)
    for i in range(n):
        var row = doc.at(root, i)
        var slug = doc.string(doc.field(row, String("slug")))
        var runs = doc.integer(doc.field(row, String("runCount")))
        var desc = doc.string(doc.field(row, String("description")))
        var here = project_exists(slug, projects_root())
        print(
            "  " + slug, "—", String(runs), "runs",
            ("  " + desc) if desc else "",
            "" if here else "   (not on this box: project-pull " + slug + ")",
        )
    print()
    print(" ", n, "project(s) on the platform")


def cmd_list() raises:
    if _has("--remote"):
        cmd_list_remote()
        return
    var root = projects_root()
    var names = _ls(root)
    var shown = 0
    for n in names:
        if not project_exists(n, root):
            continue
        var p = load_project(n, root)
        var runs = _ls(p.runs_dir())
        shown += 1
        print(
            "  " + p.name, "—", String(len(runs)), "runs,",
            String(len(p.refs)), "refs" ,
            ("  " + p.description) if p.description else "",
        )
    if shown == 0:
        print("  no projects under " + root + "/ — `pixi run project-init <name>`")
    else:
        print()
        print(" ", shown, "project(s) under", root + "/")


def _field_or(ref doc: JsonDoc, row: Int, name: String) -> String:
    try:
        var f = doc.field(row, name)
        if f < 0:
            return String("")
        return doc.string(f)
    except:
        return String("")


def cmd_show_remote(name: String) raises:
    """`project-show <name> --remote`: the project's runs as the MONITOR has
    them — including runs trained on boxes this one has never seen, which is
    the list to pick a run from before `project-pull <name> <run_id>`."""
    var cat = RemoteCatalog.from_env()
    var doc = cat.runs_of(name)
    var root = doc.root()
    var n = doc.size(root)
    print("project", name, "on the platform:", n, "run(s)")
    for i in range(n):
        var row = doc.at(root, i)
        var rid = _field_or(doc, row, String("runId"))
        var st = _field_or(doc, row, String("status"))
        var oc = _field_or(doc, row, String("outcome"))
        var tg = _field_or(doc, row, String("tag"))
        var here = len(_run_dirs(name, rid)) > 0
        print(
            "   ", st, rid,
            ("  " + oc) if oc else "",
            ("  #" + tg) if tg else "",
            "" if here else "   (not on this box)",
        )


def cmd_show() raises:
    var name = _positional(1)
    if name.byte_length() == 0:
        raise Error("usage: project-show <name> [--remote]")
    if _has("--remote"):
        cmd_show_remote(name)
        return
    if not project_exists(name, projects_root()):
        # ⚠ NO DEFINITION IS NOT NO RUNS. `runs_root_for` files a run in the
        # flat `runs/` root until someone runs `project-init`, so a project
        # can have dozens of runs and no `project.kv` — refusing to show them
        # hid exactly the runs nobody had organised yet.
        print("project", name, "— not project-init'ed on this box (no refs, no policies)")
        var dirs = _run_dirs(name, String(""))
        print("  runs:", len(dirs), "(in the flat runs/ root)")
        for d in dirs:
            var rec = load_run(String(d) + "/run.kv")
            print(
                "   ", rec.status, rec.run_id,
                ("  " + rec.outcome) if rec.outcome else "",
                ("  #" + rec.tag) if rec.tag else "",
            )
        return
    var p = load_project(name, projects_root())
    print("project", p.name, "  created", p.created)
    if p.description:
        print("  ", p.description)
    print()
    print("  refs:")
    var rep = check_refs(p, verbose=True)
    for i in range(len(p.refs)):
        var r = p.refs[i].copy()
        print("   ", r.kind, r.name, "->", r.path, "@" + r.commit if r.commit else " (unpinned)")
    print(
        "   ", rep.checked, "checked,", rep.missing, "missing,", rep.drifted,
        "drifted",
    )
    print()
    # ⚠ POLICIES BEFORE RUNS. "Which checkpoint is deployed right now?" is the
    # question a person opens this for, and it is answered by three lines; the
    # run listing below can be hundreds.
    var pols = _ls(p.policies_dir())
    var n_pol = 0
    for f in pols:
        if not String(f).endswith(".kv"):
            continue
        n_pol += 1
        var rec = load_policy(p.policies_dir() + "/" + String(f))
        print(
            "  policy " + rec.name + " -> " + rec.run + " (" + rec.checkpoint
            + ")  promoted " + rec.promoted
        )
        if rec.note.byte_length() > 0:
            print("      " + rec.note)
        if rec.supersedes.byte_length() > 0:
            print("      supersedes " + rec.supersedes)
        if not exists(policy_ckpt_path(p.dir(), rec.name)):
            # ⚠ A RECORD WITHOUT ITS WEIGHTS IS THE ONE FAILURE THIS LAYER MUST
            # NOT HIDE — a deploy would find nothing at the role.
            print("      ⚠ MISSING WEIGHTS at " + policy_ckpt_path(p.dir(), rec.name))
    if n_pol == 0:
        print("  policies: none — `pixi run project-promote <run_id> best --as <name>`")
    print()
    var runs = _ls(p.runs_dir())
    print("  runs:", len(runs))
    var stale = 0
    for rid in runs:
        var kv = p.runs_dir() + "/" + rid + "/run.kv"
        if not exists(kv):
            continue
        var rec = load_run(kv)
        if rec.is_stale():
            stale += 1
        print(
            "   ", rec.status, rid,
            ("  " + rec.outcome) if rec.outcome else "",
            ("  #" + rec.tag) if rec.tag else "",
        )
    if stale > 0:
        # ⚠ A RECORD STILL SAYING `running` IS A CRASHED RUN — the process died
        # before `close()`, so nothing local could have said otherwise.
        print()
        print("   ⚠", stale, "run(s) still say status=running: they crashed or")
        print("     are still going. Nothing infers this on their behalf.")


def cmd_tag() raises:
    var rid = _positional(1)
    var text = _positional(2)
    if rid.byte_length() == 0 or text.byte_length() == 0:
        raise Error('usage: run-tag <run_id> "free text"')
    var root = projects_root()
    # ⚠ SEARCH BOTH ROOTS. A run made before its project existed is under the
    # flat `runs/`, and the id is the same either way.
    var candidates = List[String]()
    candidates.append(String("runs/") + rid + "/run.kv")
    for n in _ls(root):
        candidates.append(root + "/" + n + "/runs/" + rid + "/run.kv")
    for c in candidates:
        if exists(c):
            _tag_file(c, text)
            print("tagged", c)
            return
    raise Error("no run '" + rid + "' under runs/ or " + root + "/*/runs/")


def _tag_file(path: String, text: String) raises:
    """⚠ REWRITE THE RECORD, DO NOT APPEND A LINE. Two `tag=` lines is a record
    that disagrees with itself and a reader that believes the first."""
    from noeira.core.kv import KvWriter, kv_lines
    from noeira.io.fileio import write_text_atomic
    var txt: String
    with open(path, "r") as fh:
        txt = fh.read()
    var ls = kv_lines(txt, path)
    var w = KvWriter(String("run spec"))
    var wrote = False
    for i in range(len(ls)):
        if ls[i].key == "tag":
            w.add(String("tag"), text)
            wrote = True
        else:
            w.add(ls[i].key, ls[i].value)
    if not wrote:
        w.add(String("tag"), text)
    write_text_atomic(path, w^.done())


def cmd_prune() raises:
    """Delete run directories that nothing is holding on to.

    ⚠⚠ DRY RUN BY DEFAULT. It needs `--apply`. `runs/` will rot exactly as
    `checkpoints/` did, but the one thing worse than 219 flat checkpoints is a
    tool that removed the one you wanted.

    What is NEVER deleted, and why each one:
      * `status=running` — it may be a LIVE run. Deleting under a training
        process is how a six-hour job dies at hour five.
      * a non-empty `tag=` — a human wrote it down, which is the only signal
        anywhere that a run mattered.
      * a `:local` artifact — ⚠ that is precisely the "lost the files" pain
        this layer exists to fix, reintroduced by the fix for it (§14 Q2).
        Once P3 uploads them the artifact line says `:uploaded` and the bytes
        survive the delete.
    """
    var name = _positional(1)
    if name.byte_length() == 0:
        raise Error("usage: project-prune <name> [--older-than DAYS] [--apply]")
    var older = atol(_flag(String("--older-than"), String(30)))
    var apply = _has("--apply")
    var p = load_project(name, projects_root())
    var now = epoch_seconds()
    var cutoff = now - older * 86400

    var scanned = 0
    var kept = 0
    var removed = 0
    var bytes_freed = 0
    for rid in _ls(p.runs_dir()):
        var d = p.runs_dir() + "/" + rid
        var kv = d + "/run.kv"
        if not exists(kv):
            continue
        scanned += 1
        var rec = load_run(kv)
        var why = String("")
        if rec.is_stale():
            why = String("status=running (may be live)")
        elif rec.tag.byte_length() > 0:
            why = String("tagged: ") + rec.tag
        else:
            for a in rec.artifacts:
                if a.endswith(":local"):
                    why = String("has a :local artifact")
                    break
        if why.byte_length() == 0 and _mtime(kv) >= cutoff:
            why = String("newer than ") + String(older) + "d"
        if why.byte_length() > 0:
            kept += 1
            print("  keep  ", rid, "—", why)
            continue
        var sz = _du_bytes(d)
        removed += 1
        bytes_freed += sz
        print("  DELETE", rid, "—", String(sz // 1048576), "MB")
        if apply:
            _ = run_capture(String("rm -rf ") + quote_arg(d) + " 2>&1", 4096)
    print()
    print(
        " ", scanned, "scanned,", kept, "kept,", removed,
        "to delete —", bytes_freed // 1048576, "MB",
    )
    if removed > 0 and not apply:
        print("  DRY RUN. Re-run with --apply to actually delete.")


def _mtime(path: String) raises -> Int:
    var s = run_capture(
        String("stat -f %m ") + quote_arg(path) + " 2>/dev/null"
        + " || stat -c %Y " + quote_arg(path) + " 2>/dev/null", 64
    )
    var v = String(s.strip())
    return atol(v) if v.byte_length() > 0 else 0


def _du_bytes(path: String) raises -> Int:
    var s = run_capture(
        String("du -sk ") + quote_arg(path) + " 2>/dev/null | cut -f1", 64
    )
    var v = String(s.strip())
    return (atol(v) * 1024) if v.byte_length() > 0 else 0


# =============================================================================
# promote — the only way a checkpoint leaves runs/
# =============================================================================


def _find_run(rid: String) raises -> Tuple[String, String]:
    """`(run_dir, project)` for a run id, searching both roots.

    ⚠ SEARCH THE FLAT `runs/` TOO. A run made before its project existed lives
    there and its id is the same either way — the same rule `run-tag` follows.
    A flat run has no project, and promoting from one is refused below rather
    than guessed at.
    """
    var root = projects_root()
    if exists(String("runs/") + rid + "/run.kv"):
        return (String("runs/") + rid, String(""))
    for n in _ls(root):
        var d = root + "/" + n + "/runs/" + rid
        if exists(d + "/run.kv"):
            return (d, String(n))
    raise Error("no run '" + rid + "' under runs/ or " + root + "/*/runs/")


def cmd_promote() raises:
    """Give a run's checkpoint a NAME, and make that name the deploy target.

    ⚠⚠ THIS IS THE ONLY WAY A CHECKPOINT LEAVES `runs/` (§8). Deployment points
    at `policies/<name>.ckpt`, a ROLE, and never at a run id — so a better
    checkpoint is one `project-promote` away and no deploy script is edited.

    ⚠ HARD-LINK, FALLING BACK TO COPY, NEVER SYMLINK. A symlink into `runs/`
    breaks on `project-pull` to another machine, breaks under rsync/sftp, and
    becomes a dead pointer the moment `project-prune` removes the run.
    """
    var rid = _positional(1)
    var which = _positional(2)
    if which.byte_length() == 0:
        which = String("best")
    var name = _flag(String("--as"), String(""))
    var note = _flag(String("--note"), String(""))
    if rid.byte_length() == 0 or name.byte_length() == 0:
        raise Error(
            'usage: project-promote <run_id> [best|last|step_N] --as <name>'
            ' [--note "why"]'
        )

    var found = _find_run(rid)
    var run_dir = found[0]
    var project = found[1]
    var rec = load_run(run_dir + "/run.kv")
    if project.byte_length() == 0:
        project = rec.project
    if project.byte_length() == 0:
        raise Error(
            "run '" + rid + "' belongs to no project, so there is no"
            " policies/ to promote into. Give it one, or promote a run made"
            " under a project."
        )

    var src = run_dir + "/checkpoints/" + which + ".ckpt"
    if not exists(src):
        # ⚠ NAME WHAT IS ACTUALLY THERE. "not found" on a checkpoint the user
        # is sure they saved is the least useful message a tool can print.
        var have = _ls(run_dir + "/checkpoints")
        var listing = String("")
        for h in have:
            listing += ("\n    " + String(h))
        raise Error(
            "no checkpoint '" + which + "' in " + run_dir + "/checkpoints"
            + ("; that directory holds:" + listing if len(have) > 0
               else " (the directory is empty)")
        )

    var pdir = projects_root() + "/" + project
    _ = run_capture("mkdir -p " + quote_arg(pdir + "/policies"))
    var dst = policy_ckpt_path(pdir, name)

    # ⚠ THE PREVIOUS HOLDER OF THE ROLE, read BEFORE it is overwritten. That is
    # what `supersedes=` records, and it is a history nobody has to maintain.
    var previous = String("")
    var kv = policy_kv_path(pdir, name)
    if exists(kv):
        previous = load_policy(kv).run

    var sha = sha256_file(src)
    var size = file_size(src)

    # ⚠ `ln` FIRST, `cp` ONLY IF IT FAILS (a different filesystem). `ln -f`
    # replaces an existing role atomically enough for this purpose; a symlink
    # is never attempted, deliberately.
    _ = run_capture(
        "rm -f " + quote_arg(dst) + " && ln " + quote_arg(src) + " "
        + quote_arg(dst) + " 2>/dev/null || cp " + quote_arg(src) + " "
        + quote_arg(dst)
    )
    if not exists(dst):
        raise Error("could not materialise " + dst + " from " + src)
    # ⚠ VERIFY THE MATERIALISED BYTES, not the source's. A hard link cannot
    # differ, but a `cp` fallback can be short, and a policy whose weights are
    # truncated is the worst possible thing to hand a real robot.
    var dst_sha = sha256_file(dst)
    if dst_sha != sha:
        raise Error(
            "the materialised policy does not match its source:\n  " + src
            + "  " + sha + "\n  " + dst + "  " + dst_sha
        )

    # ⚠⚠ THE NORMALIZATION TRAVELS WITH THE WEIGHTS. A checkpoint alone is not
    # deployable: its statistics live in the training store (9 GB). The run's
    # `checkpoints/norm.json` is copied to `policies/<name>.norm.json` — a
    # small .json, so `project-push` syncs it as definition, and a deployment
    # on another machine finds it beside the weights.
    var norm_src = run_dir + "/checkpoints/norm.json"
    var norm_dst = pdir + "/policies/" + name + ".norm.json"
    if exists(norm_src):
        _ = run_capture(
            "cp " + quote_arg(norm_src) + " " + quote_arg(norm_dst) + ".tmp && mv "
            + quote_arg(norm_dst) + ".tmp " + quote_arg(norm_dst)
        )
        print("normalization  " + norm_dst)
    else:
        _ = run_capture("rm -f " + quote_arg(norm_dst))
        print(
            "  ⚠ no " + norm_src + " — a deployment will need --store. Write it"
            " with examples/so101/act_so101_export_norm.mojo --store <store>"
            " --run " + rid + ", then promote again."
        )

    var pol = PolicyRecord(name)
    pol.run = rec.run_id if rec.run_id else rid
    pol.checkpoint = which
    pol.sha256 = sha
    pol.bytes = size
    pol.promoted = iso8601_utc(epoch_seconds())
    pol.note = note
    # ⚠ A RUN DOES NOT SUPERSEDE ITSELF. Re-promoting the same run — a
    # corrected note, a different checkpoint of it — would otherwise write
    # `supersedes=<its own id>`, which reads as a history and is not one.
    pol.supersedes = String("") if previous == pol.run else previous
    write_policy(pdir, pol)

    # And name the policy in project.kv, so `project-show` finds it without
    # listing a directory.
    #
    # ⚠ BOTH CALLS TAKE THE ROOT. They default to `projects/` in the CWD, not to
    # `projects_root()` — so omitting it silently reads a DIFFERENT project
    # than the one being promoted into, or fails to find one at all under
    # NOEIRA_PROJECTS. That is exactly what the first version of this did.
    var root_dir = projects_root()
    if project_exists(project, root_dir):
        var spec = load_project(project, root_dir)
        var known = False
        for i in range(len(spec.policies)):
            if spec.policies[i] == name:
                known = True
                break
        if not known:
            spec.policies.append(name)
            spec.write()

    # ⚠⚠ THE POST COMES LAST AND CANNOT FAIL THE PROMOTION. The weights are
    # already linked and the `.kv` already written; the box is the record and
    # the monitor is its index (§10). A promotion that exists only locally is a
    # normal state — `--no-push` or an offline box — and one that existed only
    # on the server would be a lie. So this reports and carries on.
    var pushed = String("")
    if not _has("--no-push"):
        try:
            var cat = RemoteCatalog.from_env()
            _ = cat.record_policy(
                project, name, pol.run, which, sha, size, note,
                pol.supersedes, pol.promoted,
            )
            pushed = String("recorded on the monitor")
        except e:
            pushed = String("NOT recorded on the monitor: ") + String(e)

    print("promoted " + rid + " (" + which + ") as '" + name + "'")
    print("  role     " + dst)
    print("  record   " + policy_kv_path(pdir, name))
    print("  sha256   " + sha + "  (" + String(size // 1_000_000) + " MB)")
    # ⚠ PRINT WHAT WAS WRITTEN, NOT WHAT WAS READ. These differ for a
    # re-promotion of the same run, and a summary that disagrees with the
    # record it just wrote is worse than no summary.
    if pol.supersedes.byte_length() > 0:
        print("  supersedes " + pol.supersedes)
    if rec.outcome.byte_length() > 0:
        print("  the run said: " + rec.outcome)
    if pushed.byte_length() > 0:
        # ⚠ PRINTED EITHER WAY. A promotion the monitor never heard about is
        # invisible on a second machine, and silence about that is how someone
        # later concludes the history is complete when it is not.
        print("  " + pushed)


# =============================================================================
# push / pull — the two commands that need an account
# =============================================================================


def _artifact_rel(line: String) -> String:
    """`checkpoints/best.ckpt` out of `checkpoints/best.ckpt:sha256:ab..:215:local`.

    ⚠ `split_once` CUTS AT THE FIRST SEPARATOR, which is why the path comes
    first in the record: a path may not contain `:` (the monitor refuses one
    too), so the first colon is always the end of it.
    """
    var cut = line.find(":")
    return String(line) if cut < 0 else String(line[byte=0:cut])


def _run_dirs(project: String, only: String) raises -> List[String]:
    """Every run of `project` on this box: under `projects/<p>/runs/`, AND in
    the flat `runs/` root when its `run.kv` says `project=<p>`.

    ⚠ THE FLAT ROOT IS WHERE MOST RUNS ARE. `runs_root_for` falls back to it
    whenever the named project has not been `project-init`ed, which on
    2026-09-22 was 47 runs of so101, libero and g1 — none of which a
    project-scoped listing could see, and so none of which could be pushed.
    """
    var out = List[String]()
    var base = projects_root() + "/" + project + "/runs"
    for n in _ls(base):
        if only.byte_length() > 0 and n != only:
            continue
        if exists(base + "/" + n + "/run.kv"):
            out.append(base + "/" + n)
    for n in _ls(String("runs")):
        if only.byte_length() > 0 and n != only:
            continue
        var kv = String("runs/") + n + "/run.kv"
        if not exists(kv):
            continue
        try:
            if load_run(kv).project == project:
                out.append(String("runs/") + n)
        except:
            pass  # a record this build cannot read is reported by project-show
    return out^


def _kind_of(rel: String) -> String:
    if rel.startswith("checkpoints/"):
        return String("checkpoint")
    if rel.startswith("eval/"):
        return String("eval")
    return String("log")


def _run_files(dir: String, rec: RunRecord) raises -> List[String]:
    """What a push sends for one run, as paths relative to the run directory.

    The `artifact=` lines of `run.kv`; when there are none — a run that died
    before `close()`, or one closed by a build older than
    `RunContext.record_artifacts` — the same directories `close()` would have
    read. Then the run's own record: `run.kv`, `metrics.csv` and
    `metrics.config.kv`, so a run pulled onto another box is a run there
    (`project-show` lists what has a `run.kv`) and its curves come with it.
    """
    var out = List[String]()
    for line in rec.artifacts:
        out.append(_artifact_rel(String(line)))
    if len(out) == 0:
        var listing = run_capture(
            String("cd ") + quote_arg(dir)
            + " && find checkpoints eval -type f ! -name '*.tmp' 2>/dev/null"
            + " | sort; true",
            1 << 20,
        )
        for l in listing.split("\n"):
            var rel = String(String(l).strip())
            if rel.byte_length() > 0:
                out.append(rel)
    for f in [String("run.kv"), String("metrics.csv"), String("metrics.config.kv")]:
        var seen = False
        for r in out:
            if r == f:
                seen = True
        if not seen and exists(dir + "/" + f):
            out.append(f)
    return out^


def _run_config(dir: String, rec: RunRecord) raises -> Tuple[List[String], List[String]]:
    """The config a live run registers with (`core/run.register_run` + the
    logger's `set_config` calls), rebuilt from the run's files so the upsert
    does not replace it with less."""
    var k = List[String]()
    var v = List[String]()

    def put(key: String, val: String) {mut k, mut v}:
        if val.byte_length() == 0:
            return
        for i in range(len(k)):
            if k[i] == key:
                v[i] = val
                return
        k.append(key)
        v.append(val)

    var cfg = dir + "/metrics.config.kv"
    if exists(cfg):
        with open(cfg, "r") as fh:
            for line in fh.read().split("\n"):
                var l = String(line)
                var cut = l.find("=")
                if cut > 0:
                    put(String(l[byte=0:cut]), String(l[byte = cut + 1 :]))
    for c in rec.config:
        var cs = String(c)
        var cut = cs.find(":")
        if cut > 0:
            put(String(cs[byte=0:cut]), String(cs[byte = cut + 1 :]))
    put(String("run_id"), rec.run_id)
    put(String("project"), rec.project)
    put(String("driver"), rec.driver)
    put(String("env"), rec.env)
    put(String("task"), rec.task)
    put(String("source_commit"), rec.source_commit)
    put(String("seed"), String(rec.seed))
    put(String("host"), rec.host)
    return (k^, v^)


def _ready_digests(mut cat: RemoteCatalog, run_id: String) -> List[String]:
    """The sha256 of every artifact the monitor already holds for this run.

    ⚠⚠ THE SERVER IS THE AUTHORITY ON WHAT ARRIVED, NOT THE LOCAL `:local`
    MARKER. `ArtifactSink` uploads from a background thread and never rewrites
    `run.kv` — plumbing a completion back across the thread boundary would be
    a second source of truth for a fact the monitor already has. So a push
    asks, and skips what is already there with a matching digest. That also
    makes a push after a CRASH correct for free: whatever completed is
    skipped, whatever did not is re-sent.

    A failure to ask returns empty, which re-sends. Re-sending is wasteful;
    skipping something that is not there is wrong.
    """
    var out = List[String]()
    try:
        var doc = cat.artifacts_of(run_id)
        var root = doc.root()
        var n = doc.size(root)
        for i in range(n):
            var row = doc.at(root, i)
            var st = doc.string(doc.field(row, String("status")))
            if st != "ready":
                continue
            var sha = doc.field(row, String("sha256"))
            if sha >= 0:
                out.append(doc.string(sha))
    except:
        pass
    return out^


def _push_dry_run(project: String, only: String, kind: String) raises:
    """`project-push --dry-run`: every run, its status, its config size and
    the files a push would consider — with no network and no credentials."""
    var dirs = _run_dirs(project, only)
    var n_files = 0
    for d in dirs:
        var dir = String(d)
        var rec = load_run(dir + "/run.kv")
        var cfg = _run_config(dir, rec)
        print(
            "  run " + rec.run_id + "  [" + rec.status + "]  "
            + String(len(cfg[0])) + " config keys  (" + dir + ")"
        )
        for rel in _run_files(dir, rec):
            var k = _kind_of(rel)
            if kind.byte_length() > 0 and k != kind:
                continue
            n_files += 1
            var here = exists(dir + "/" + rel)
            print(
                "      " + k + "  " + rel
                + ("" if here else "   (MISSING locally)")
            )
    print(String(len(dirs)) + " runs, " + String(n_files) + " files — dry run, nothing sent")


def cmd_push() raises:
    """Send this box's runs of a project to the monitor: the run itself, its
    checkpoints and eval files, and its record (`run.kv`, `metrics.csv`,
    `metrics.config.kv`).

    ⚠ IT IS A RESUME, NOT A RE-SEND. Anything the monitor already holds with a
    matching sha256 is skipped, so running it twice costs two catalog reads and
    no bytes — the same "cache, not toll booth" rule `RemoteCatalog.pull` uses
    for datasets.

    ⚠ THE RUN IS REGISTERED FIRST. `/artifacts` refuses a run the monitor has
    never seen, so without this a run trained offline could never be pushed.
    `--kind checkpoint|eval|log` limits what is sent.
    """
    var project = _positional(1)
    if project.byte_length() == 0:
        raise Error(
            "usage: project-push <project> [<run_id>] [--kind K] [--force]"
            " [--definition-only] [--dry-run]"
        )
    var only = _positional(2)
    var kind = _flag(String("--kind"), String(""))
    # `--dry-run`: what would be registered and sent, with no request made.
    var dry = _has("--dry-run")
    if dry:
        _push_dry_run(project, only, kind)
        return
    var cat = RemoteCatalog.from_env()

    # ⚠ THE DEFINITION FIRST, and only when no single run was named. A box
    # that pushes a run whose project the platform has never seen would
    # otherwise leave the training box that pulls it with checkpoints and no
    # calibration to run them with.
    if only.byte_length() == 0:
        var root = projects_root()
        if project_exists(project, root):
            var spec = load_project(project, root)
            print("definition  " + root + "/" + project + "/")
            var rep = push_definition(
                cat, project, root, spec.description, _has("--force")
            )
            rep.print_all(String("pushed"))
        else:
            # Runs can exist without a definition (the flat `runs/` root);
            # the platform creates the project from the run's config.
            print("definition  none — " + project + " was never project-init'ed here")
        if _has("--definition-only"):
            return
        print()

    var dirs = _run_dirs(project, only)
    if len(dirs) == 0:
        if only.byte_length() > 0:
            raise Error("no run " + only + " with a run.kv for " + project)
        print("runs        none with a run.kv for " + project + " yet")
        return
    print("runs")
    var sent = 0
    var skipped = 0
    var missing = 0
    var considered = 0
    var registered = 0

    for d in dirs:
        var dir = String(d)
        var rec = load_run(dir + "/run.kv")
        var cfg = _run_config(dir, rec)
        try:
            cat.upsert_run(rec.run_id, rec.run_id, cfg[0], cfg[1])
            registered += 1
            if rec.status == "done" or rec.status == "killed" or rec.status == "crashed":
                cat.finish_run(rec.run_id, rec.status, rec.outcome)
        except e:
            print("  skip     " + rec.run_id + "  (could not register: " + String(e) + ")")
            continue
        var have = _ready_digests(cat, rec.run_id)
        for rel in _run_files(dir, rec):
            var k = _kind_of(rel)
            if kind.byte_length() > 0 and k != kind:
                continue
            considered += 1
            var local = dir + "/" + rel
            if not exists(local):
                missing += 1
                print("  missing  " + rec.run_id + "  " + rel)
                continue
            var sha = sha256_file(local)
            var known = False
            for h in have:
                if String(h) == sha:
                    known = True
                    break
            if known:
                skipped += 1
                continue
            print(
                "  push     " + rec.run_id + "  " + rel + "  ("
                + String(file_size(local) // 1_000_000) + " MB)"
            )
            _ = cat.push_artifact(rec.run_id, rel, local, k)
            sent += 1

    # ⚠ PRINT WHAT WAS CONSIDERED BESIDE WHAT WAS SENT. "0 pushed" is also what
    # a run with nothing to send prints, and those are different facts.
    print(
        String(len(dirs)) + " runs (" + String(registered) + " registered), "
        + String(considered) + " files considered: " + String(sent)
        + " pushed, " + String(skipped) + " already there, "
        + String(missing) + " missing locally"
    )


def cmd_pull() raises:
    """Bring a run's artifacts back onto this box.

    ⚠ NO BYTES MOVE for a file that is already here with the right digest —
    `fetch_to_cache` compares before it transfers.
    """
    var project = _positional(1)
    var rid = _positional(2)
    if project.byte_length() == 0:
        raise Error("usage: project-pull <project> [<run_id>] [--kind K] [--force]")
    if rid.byte_length() == 0:
        # ⚠ NO RUN ID = THE DEFINITION. This is the first command on a fresh
        # training box: it recreates `projects/<project>/` from the platform.
        var cat0 = RemoteCatalog.from_env()
        var root = projects_root()
        print("definition  " + root + "/" + project + "/")
        var rep = pull_definition(cat0, project, root, _has("--force"))
        rep.print_all(String("pulled"))
        if _has("--weights"):
            # ⚠ OPT-IN: a training box pulling a project wants its calibration,
            # not hundreds of MB of policy weights it will not run.
            print()
            print("policy weights")
            var wr = pull_policy_weights(cat0, root + "/" + project)
            for l in wr.lines:
                print("  " + l)
            print(
                String(wr.pulled) + " pulled, " + String(wr.present) + " already here, "
                + String(wr.failed) + " not available"
            )
        return
    var kind = _flag(String("--kind"), String(""))
    var dest_root = projects_root() + "/" + project + "/runs/" + rid
    _ = run_capture("mkdir -p " + quote_arg(dest_root))

    var cat = RemoteCatalog.from_env()
    var doc = cat.artifacts_of(rid)
    var root = doc.root()
    var n = doc.size(root)
    var pulled = 0
    var pending = 0
    for i in range(n):
        var row = doc.at(root, i)
        var rel = doc.string(doc.field(row, String("path")))
        var st = doc.string(doc.field(row, String("status")))
        var k = doc.string(doc.field(row, String("kind")))
        if kind.byte_length() > 0 and k != kind:
            continue
        if st != "ready":
            # ⚠ A pending row is an upload that never completed. Its bytes are
            # not there, and offering a download would 404.
            pending += 1
            print("  pending  " + rel + " (never completed; not pulled)")
            continue
        var meta = cat.describe_artifact(doc.string(doc.field(row, String("id"))))
        var dest = dest_root + "/" + rel
        var cut = dest.rfind("/")
        if cut > 0:
            _ = run_capture("mkdir -p " + quote_arg(String(dest[byte=0:cut])))
        _ = fetch_to_cache(
            meta.download_url, dest, meta.sha256, meta.size_bytes, rel
        )
        pulled += 1
    print(
        String(n)
        + " artifacts listed for "
        + rid
        + ": "
        + String(pulled)
        + " pulled, "
        + String(pending)
        + " still pending on the server"
    )


# =============================================================================
# rerun / resume — a run is a recipe, not only a record
# =============================================================================


def _load_found(rid: String) raises -> Tuple[String, RunRecord]:
    var found = _find_run(rid)
    var dir = found[0]
    return (dir, load_run(dir + "/run.kv"))


def _require_recipe(rid: String, rec: RunRecord) raises:
    """⚠ A record from before schema 2 has no `arg=` lines — its arguments
    were never written, and "no arguments" would be a guess presented as a
    fact. Refuse rather than rerun it with its defaults."""
    if rec.schema_version < 2:
        raise Error(
            "run " + rid + " was recorded before 2026-09-23 (run.kv schema "
            + String(rec.schema_version) + "): its command-line arguments"
            " were not saved, so it cannot be rerun as it ran. Its driver"
            " was " + rec.driver + "."
        )


def _source_verdict(rec: RunRecord) -> String:
    """Whether THIS checkout is the code the run ran — the one question a
    rerun has to answer before its curve is compared with the original."""
    var head = git_commit()
    if head != rec.source_commit:
        return (
            String("NOT exact here: HEAD is ") + head + ", the run was at "
            + rec.source_commit
        )
    if rec.dirty or git_dirty():
        if rec.source_patch == "source.patch" and not git_dirty():
            return String("NOT exact here: the run had uncommitted changes"
                          " (apply its source.patch)")
        return String("NOT exact here: the run or this tree has uncommitted"
                      " changes")
    return String("exact: same commit, both trees clean")


def cmd_rerun() raises:
    """Print how to run a run again, exactly; `--exec` runs it HERE."""
    var rid = _positional(1)
    if rid.byte_length() == 0:
        raise Error("usage: project-rerun <run_id> [--exec]")
    var lr = _load_found(rid)
    var dir = lr[0]
    ref rec = lr[1]
    _require_recipe(rid, rec)
    var cmd = run_command(rec.driver, rec.pixi_env, rec.args)
    print("run", rid, "—", rec.driver)
    var src = rec.source_commit
    if rec.source_patch == "source.patch":
        src += " + source.patch (uncommitted changes, saved with the run)"
    elif rec.source_patch.startswith("too_large:"):
        src += " + uncommitted changes NOT saved (" + rec.source_patch
        src += " bytes): this run cannot be reproduced exactly"
    elif rec.dirty:
        src += " + uncommitted changes NOT saved"
    print("  source :", src)
    print("  here   :", _source_verdict(rec))
    print()
    print("  In a separate worktree (leaves this checkout alone):")
    var wt = String("../noeira-rerun-") + rid
    print("    git worktree add", wt, rec.source_commit)
    if rec.source_patch == "source.patch":
        var abs_patch = String(run_capture(
            String("cd ") + quote_arg(dir) + " && pwd", 4096
        ).strip()) + "/source.patch"
        print("    git -C", wt, "apply", quote_arg(abs_patch))
    print("    cd", wt, "&&", cmd)
    print()
    print("  Or on this checkout:")
    print("    " + cmd)
    if _has(String("--exec")):
        print()
        print("  running it here (" + _source_verdict(rec) + ")")
        var code = run_system(cmd)
        if code != 0:
            raise Error("the rerun exited with status " + String(code))


def cmd_resume() raises:
    """Continue a run from one of its checkpoints, with its own arguments.

    ⚠ ONLY FOR A DRIVER THAT DECLARED `resume_args` (`RunContext.
    set_resume_args`). Inventing a flag for one that did not would start a
    FRESH run and call it a continuation — the failure this refuses.
    """
    var rid = _positional(1)
    if rid.byte_length() == 0:
        raise Error(
            "usage: project-resume <run_id> [--from last] [--dry-run]"
        )
    var name = _flag(String("--from"), String("last"))
    var lr = _load_found(rid)
    var dir = lr[0]
    ref rec = lr[1]
    _require_recipe(rid, rec)
    if rec.resume_args.byte_length() == 0:
        raise Error(
            rec.driver + " declares no way to continue from a checkpoint"
            " (no resume_args in its run.kv). `pixi run project-rerun "
            + rid + "` starts it again from scratch."
        )
    var ckpt = dir + "/checkpoints/" + name + ".ckpt"
    if not exists(ckpt):
        var proj = rec.project if rec.project.byte_length() > 0 else String(
            "<project>"
        )
        raise Error(
            "no " + ckpt + " on this box. If the run trained elsewhere:"
            " pixi run project-pull " + proj + " " + rid + " --kind checkpoint"
        )
    var args = resume_args_for(rec.args, rec.resume_args, ckpt)
    var cmd = run_command(rec.driver, rec.pixi_env, args)
    print("resume", rid, "from", ckpt)
    print("  source :", _source_verdict(rec))
    print("  " + cmd)
    if _has(String("--dry-run")):
        return
    var code = run_system(cmd)
    if code != 0:
        raise Error("the resumed run exited with status " + String(code))


def main() raises:
    var cmd = _positional(0)
    if cmd == "init":
        cmd_init()
    elif cmd == "list":
        cmd_list()
    elif cmd == "show":
        cmd_show()
    elif cmd == "tag":
        cmd_tag()
    elif cmd == "prune":
        cmd_prune()
    elif cmd == "promote":
        cmd_promote()
    elif cmd == "push":
        cmd_push()
    elif cmd == "pull":
        cmd_pull()
    elif cmd == "rerun":
        cmd_rerun()
    elif cmd == "resume":
        cmd_resume()
    else:
        print(
            "usage: project_cli"
            " <init|list|show|tag|prune|promote|push|pull|rerun|resume> ..."
        )
        print("  see the module docstring, or docs/PROJECT_LAYER_PLAN.md §10")

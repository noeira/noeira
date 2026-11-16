# +--------------------------------------------------------------------------+ #
# | Private projects — executing a sync plan against the platform
# +--------------------------------------------------------------------------+ #
"""`push_definition` / `pull_definition`: `core/project_sync.mojo`'s plans, run.

    var cat = RemoteCatalog.from_env()
    var rep = push_definition(cat, String("so101-tower"), projects_root())
    rep.print_all(String("pushed"))

⚠ THE PLAN IS DECIDED BEFORE ANYTHING MOVES, AND `.sync.kv` IS WRITTEN AFTER,
EVEN IF A TRANSFER FAILED HALFWAY. A file that did go through is synced, and
recording it means the next run does not re-plan it as a conflict against
its own upload.

⚠ NOTHING HERE DECIDES WHO WINS. That is `push_action` / `pull_action`, pure
and gated by value; this file only carries their answers out.
"""

from std.collections import Dict
from std.os import makedirs
from std.os.path import exists

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
    SyncAction,
    load_base,
    plan_pull,
    plan_push,
    save_base,
    scan_definition,
    sync_refusal,
)
from noeira.data.remote import RemoteCatalog
from noeira.io.fileio import write_file_atomic
from noeira.io.http import HttpClient
from noeira.io.json import JsonDoc
from noeira.io.sha256 import sha256_hex


struct SyncReport(Movable):
    """What a sync did, file by file, and what it left out."""

    var lines: List[String]
    """One line per file that was not simply `same`, in path order."""
    var skipped: List[String]
    var moved: Int
    var same: Int
    var conflicts: Int
    var other: Int
    """One of behind / ahead / remote-only / local-only / deleted-locally."""

    def __init__(out self):
        self.lines = List[String]()
        self.skipped = List[String]()
        self.moved = 0
        self.same = 0
        self.conflicts = 0
        self.other = 0

    def __init__(out self, *, deinit move: Self):
        self.lines = move.lines^
        self.skipped = move.skipped^
        self.moved = move.moved
        self.same = move.same
        self.conflicts = move.conflicts
        self.other = move.other

    def total(self) -> Int:
        return self.moved + self.same + self.conflicts + self.other

    def print_all(self, verb: String):
        for l in self.lines:
            print("  " + l)
        for s in self.skipped:
            print("  skipped   " + s)
        # ⚠ THE CONSIDERED COUNT BESIDE THE MOVED ONE. "0 pushed" is also what
        # a scan that found no files prints, and those are different facts.
        print(
            String(self.total()) + " definition files: " + String(self.moved)
            + " " + verb + ", " + String(self.same) + " already in sync, "
            + String(self.conflicts) + " conflicts, " + String(self.other)
            + " left alone, " + String(len(self.skipped)) + " skipped"
        )
        if self.conflicts > 0:
            print(
                "  ⚠ a conflict means BOTH sides changed since this box last"
                " synced. Nothing was overwritten. Keep one side with --force"
                " (push: this box wins; pull: the platform wins)."
            )


def _remote_maps(
    ref doc: JsonDoc,
    mut shas: Dict[String, String],
    mut sizes: Dict[String, Int],
    mut urls: Dict[String, String],
) raises -> Int:
    """Fill the maps from `GET /projects/<slug>/files`. Returns max_file_bytes."""
    var root = doc.root()
    var files = doc.field(root, String("files"))
    for i in range(doc.size(files)):
        var row = doc.at(files, i)
        var p = doc.string(doc.field(row, String("path")))
        shas[p] = doc.string(doc.field(row, String("sha256")))
        sizes[p] = doc.integer(doc.field(row, String("sizeBytes")))
        urls[p] = doc.string(doc.field(row, String("download_url")))
    var mx = doc.field(root, String("max_file_bytes"))
    return doc.integer(mx) if mx >= 0 else 0


def _describe(ref a: SyncAction) -> String:
    """The line a person reads for one file."""
    var tag = a.action
    while tag.byte_length() < 16:
        tag += " "
    return tag + a.path


def push_definition(
    mut cat: RemoteCatalog,
    slug: String,
    root: String,
    description: String = String(""),
    force: Bool = False,
    dry: Bool = False,
) raises -> SyncReport:
    """Send `<root>/<slug>/`'s definition files to the platform.

    ⚠ THE PROJECT IS UPSERTED FIRST, so a project that exists only on this box
    — the normal state after `project-init` — needs no separate step.

    `dry=True` (`project-push --dry-run`): the same plan against the platform's
    file list (one GET), and NOTHING WRITTEN — no upsert, no upload, no
    `.sync.kv`. The report's `moved` counts what a real push would upload.
    ⚠ A project the platform has never seen plans every file as an upload,
    with no cap known yet (the real push reads the cap after creating it).
    """
    var dir = root + "/" + slug
    if not exists(dir + "/project.kv"):
        raise Error(
            "no project '" + slug + "' under " + root + "/ — nothing to push"
        )
    var rshas = Dict[String, String]()
    var rsizes = Dict[String, Int]()
    var rurls = Dict[String, String]()
    var max_bytes = 0
    var rep = SyncReport()
    if dry:
        var known = True
        try:
            var doc = cat.project_files(slug)
            max_bytes = _remote_maps(doc, rshas, rsizes, rurls)
        except e:
            if String(e).find("is not on the platform") < 0:
                raise Error(String(e))
            known = False
        if not known:
            rep.lines.append(
                String("(not on the platform yet: a push creates it; the platform's size cap is not known)")
            )
    else:
        cat.upsert_project(slug, description)
        var doc = cat.project_files(slug)
        max_bytes = _remote_maps(doc, rshas, rsizes, rurls)

    var scan = scan_definition(dir, max_bytes)
    var base = load_base(dir)
    var plan = plan_push(scan.shas, rshas, base, force)

    for i in range(len(scan.skipped_paths)):
        rep.skipped.append(scan.skipped_paths[i] + " — " + scan.skipped_reasons[i])

    try:
        for a in plan:
            if a.action == ACT_SAME:
                rep.same += 1
                if a.local_sha.byte_length() > 0:
                    base[a.path] = a.local_sha
                elif a.path in base:
                    _ = base.pop(a.path, String(""))
                continue
            if a.action == ACT_UPLOAD and dry:
                rep.moved += 1
                rep.lines.append(_describe(a))
                continue
            if a.action == ACT_UPLOAD:
                var got = cat.push_project_file(
                    slug,
                    a.path,
                    dir + "/" + a.path,
                    a.local_sha,
                    scan.sizes.get(a.path, 0),
                    a.remote_sha,
                )
                if got.startswith("conflict:"):
                    # ⚠ Lost the race the plan could not see: another box
                    # pushed between our list and our write.
                    rep.conflicts += 1
                    rep.lines.append(
                        _describe(a) + "  (changed on the platform during this push)"
                    )
                    continue
                rep.moved += 1
                base[a.path] = a.local_sha
                rep.lines.append(_describe(a))
                continue
            if a.action == ACT_CONFLICT:
                rep.conflicts += 1
            else:
                rep.other += 1
            rep.lines.append(_describe(a) + _hint_push(a.action))
    finally:
        if not dry:
            save_base(dir, base)
    return rep^


def _hint_push(action: String) -> String:
    if action == ACT_BEHIND:
        return String("  (newer on the platform — pull it)")
    if action == ACT_REMOTE_ONLY:
        return String("  (only on the platform — pull brings it)")
    if action == ACT_DELETED_LOCALLY:
        return String("  (removed here; still on the platform)")
    return String("")


def pull_definition(
    mut cat: RemoteCatalog,
    slug: String,
    root: String,
    force: Bool = False,
) raises -> SyncReport:
    """Bring `<slug>`'s definition files onto this box, creating the project
    directory if it is not here yet — the fresh-GPU-box case."""
    var doc = cat.project_files(slug)
    var rshas = Dict[String, String]()
    var rsizes = Dict[String, Int]()
    var rurls = Dict[String, String]()
    _ = _remote_maps(doc, rshas, rsizes, rurls)

    var dir = root + "/" + slug
    makedirs(dir, exist_ok=True)
    var rep = SyncReport()

    # ⚠⚠ A PATH FROM THE SERVER IS CHECKED BEFORE IT IS PLANNED. It becomes a
    # file written under `dir`, and "our own Worker validated it" is a claim
    # about code this box is not running.
    var refused = List[String]()
    for k in rshas.keys():
        var why = sync_refusal(k)
        if why.byte_length() > 0:
            refused.append(k)
            rep.skipped.append(k + " — refused from the platform: " + why)
    for k in refused:
        _ = rshas.pop(k)

    var scan = scan_definition(dir, 0)
    var base = load_base(dir)
    var plan = plan_pull(scan.shas, rshas, base, force)
    var http = HttpClient(30000, 10000)

    try:
        for a in plan:
            if a.action == ACT_SAME:
                rep.same += 1
                if a.remote_sha.byte_length() > 0:
                    base[a.path] = a.remote_sha
                elif a.path in base:
                    _ = base.pop(a.path, String(""))
                continue
            if a.action == ACT_DOWNLOAD:
                var dest = dir + "/" + a.path
                var cut = dest.rfind("/")
                makedirs(String(dest[byte=0:cut]), exist_ok=True)
                _fetch_verified(
                    http, rurls.get(a.path, String("")), dest, a.remote_sha, a.path
                )
                rep.moved += 1
                base[a.path] = a.remote_sha
                rep.lines.append(_describe(a))
                continue
            if a.action == ACT_CONFLICT:
                rep.conflicts += 1
            else:
                rep.other += 1
            rep.lines.append(_describe(a) + _hint_pull(a.action))
    finally:
        save_base(dir, base)
    return rep^


def _fetch_verified(
    mut http: HttpClient, url: String, dest: String, sha: String, label: String
) raises:
    """GET a definition file into memory, verify it, then write it atomically.

    ⚠ NOT `fetch_to_cache`. That one is built for 215 MB checkpoints — resume,
    stall guard, a progress meter — and on a replaced 60-byte file it prints
    "cached file failed its sha256 - refetching", which reads as corruption
    for what is a deliberate update. These files are capped at 1 MB, so the
    whole body fits in memory and is checked BEFORE anything touches `dest`.
    """
    var r = http.request(String("GET"), url, List[UInt8](), String(""), 200)
    var body = r^.take_body()
    var got = sha256_hex(body)
    if got != sha:
        raise Error(
            label + ": the platform served bytes hashing to " + got
            + ", its own row says " + sha + " — not written"
        )
    write_file_atomic(dest, body)


def _hint_pull(action: String) -> String:
    if action == ACT_AHEAD:
        return String("  (edited here — push it)")
    if action == ACT_LOCAL_ONLY:
        return String("  (only on this box — push sends it)")
    if action == ACT_DELETED_LOCALLY:
        return String("  (removed here; not restored — --force restores it)")
    return String("")

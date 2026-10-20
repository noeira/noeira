"""A project — a directory, a `project.kv`, and a naming authority.

`docs/PROJECT_LAYER_PLAN.md` §3-§4. It owns nothing computational: it is a
container for a definition (which env, which tasks, which datasets) plus the
runs made under it.

⚠⚠ NOTHING HERE IS IN GIT — `projects/` is gitignored, because a project is
private data, not code. `project.kv` and the pointers beside it are still a
few KB of text saying exactly what a run was, and leaving that recipe on one
laptop's filesystem is the pain being fixed: they travel through noeira cloud
(`project-push` / `project-pull`, `core/project_sync.mojo`) instead of git.
Runs are append-only and per-machine; their weights move only with
`--weights`.

⚠ IT REFERENCES `.family` AND `.task`; IT NEVER REDEFINES THEM. The task layer
owns those. A project that copied them would fork the definition, and the fork
is invisible until two runs disagree about what a task was.

## The `ref=` line, and why it carries a commit

    ref=family:so101_tabletop:noeira/tasks/families/so101_tabletop.family@081b53c0
    ref=task:so101_reach_brick:noeira/tasks/tasks/so101_reach_brick.task@081b53c0

`<kind>:<name>:<path>@<commit>`, one repeating key — the same colon-separated
shape `tasks/spec.mojo`'s `region=` already uses, and `split_once` reads it
without losing a path that contains `:`.

⚠⚠ THE `@<commit>` IS THE WHOLE POINT AND THE FIRST PASS OMITTED IT. Decision 11
chose a reference over a copy on the reversibility argument — a reference can be
materialised later, whereas un-forking a drifted copy is a merge. But an
UNPINNED reference resolves against whatever the working tree happens to be, so
a `.family` that drifted makes `run.kv` a lie about what was trained. That is
this layer's own failure mode, reintroduced one level up.

⚠ THE PIN IS CHECKED, NOT ENFORCED. `check_refs` reports drift; it does not
refuse to run. A box at a different commit is the normal case during
development, and a project layer that blocked a run over it would be turned off.
"""

from std.os import getenv
from std.os.path import exists

from noeira.core.dotenv import load_dotenv
from noeira.core.kv import KvLine, KvWriter, kv_lines, kv_write, split_once
from noeira.io.proc import quote_arg, run_capture
from noeira.io.fileio import write_text_atomic


comptime SCHEMA_VERSION = 1
comptime DEFAULT_ROOT = "projects"
"""⚠ `NOEIRA_PROJECTS` in `.env` overrides it (§4)."""


comptime _KNOWN_KEYS = "schema_version name created description ref dataset policy"


struct Ref(Copyable, ImplicitlyCopyable, Movable):
    """One `ref=` line: what it is, what it is called, where it lives, and the
    commit the project last saw it at."""

    var kind: String
    var name: String
    var path: String
    var commit: String

    def __init__(
        out self, kind: String, name: String, path: String, commit: String
    ):
        self.kind = kind
        self.name = name
        self.path = path
        self.commit = commit

    def __init__(out self, *, copy: Self):
        self.kind = copy.kind
        self.name = copy.name
        self.path = copy.path
        self.commit = copy.commit

    def __init__(out self, *, deinit move: Self):
        self.kind = move.kind^
        self.name = move.name^
        self.path = move.path^
        self.commit = move.commit^

    def encode(self) -> String:
        var s = self.kind + ":" + self.name + ":" + self.path
        return s + "@" + self.commit if self.commit else s^


def parse_ref(value: String, what: String) raises -> Ref:
    """`<kind>:<name>:<path>[@<commit>]`.

    ⚠ SPLIT-ONCE TWICE, NOT A GREEDY SPLIT ON `:`. A path may contain a colon,
    and only the first two are separators — the same reason `tasks/spec.mojo`
    reads `region=` this way.
    """
    var a = split_once(value, String(":"))
    if len(a) != 2:
        raise Error(what + ": ref has no kind: '" + value + "'")
    var b = split_once(a[1], String(":"))
    if len(b) != 2:
        raise Error(what + ": ref has no name: '" + value + "'")
    var path = b[1]
    var commit = String("")
    # ⚠ THE `@` IS SPLIT FROM THE RIGHT, because a path may contain one and a
    # commit may not.
    var bytes = path.as_bytes()
    var at = -1
    for i in range(path.byte_length()):
        if Int(bytes[i]) == 0x40:  # "@"
            at = i
    if at >= 0:
        commit = String(path[byte = at + 1 : path.byte_length()])
        var head = String(path[byte=0:at])
        path = head^
    return Ref(a[0], b[0], path, commit)


struct ProjectSpec(Movable):
    """`project.kv` — the definition, as read or as being written."""

    var name: String
    var created: String
    var description: String
    var refs: List[Ref]
    var datasets: List[String]
    var policies: List[String]
    var root: String
    """The projects root this was found under; `dir()` needs it."""

    def __init__(out self, name: String, root: String = String("")):
        self.name = name
        self.created = String("")
        self.description = String("")
        self.refs = List[Ref]()
        self.datasets = List[String]()
        self.policies = List[String]()
        self.root = root if root else String(DEFAULT_ROOT)

    def __init__(out self, *, deinit move: Self):
        self.name = move.name^
        self.created = move.created^
        self.description = move.description^
        self.refs = move.refs^
        self.datasets = move.datasets^
        self.policies = move.policies^
        self.root = move.root^

    def dir(self) -> String:
        return self.root + "/" + self.name

    def kv_path(self) -> String:
        return self.dir() + "/project.kv"

    def runs_dir(self) -> String:
        return self.dir() + "/runs"

    def policies_dir(self) -> String:
        return self.dir() + "/policies"

    def add_ref(mut self, kind: String, name: String, path: String, commit: String):
        for i in range(len(self.refs)):
            if self.refs[i].kind == kind and self.refs[i].name == name:
                self.refs[i] = Ref(kind, name, path, commit)
                return
        self.refs.append(Ref(kind, name, path, commit))

    def ref_of(self, kind: String, name: String) raises -> Ref:
        for i in range(len(self.refs)):
            if self.refs[i].kind == kind and self.refs[i].name == name:
                return self.refs[i].copy()
        raise Error(
            self.name + ": no " + kind + " named '" + name + "' in "
            + self.kv_path()
        )

    def tasks(self) -> List[String]:
        var out = List[String]()
        for i in range(len(self.refs)):
            if self.refs[i].kind == "task":
                out.append(self.refs[i].name)
        return out^

    def render(self) raises -> String:
        var w = KvWriter(String("project spec"))
        w.add(String("schema_version"), String(SCHEMA_VERSION))
        w.add(String("name"), self.name)
        w.add(String("created"), self.created)
        w.add(String("description"), self.description)
        for i in range(len(self.refs)):
            w.add(String("ref"), self.refs[i].encode())
        for i in range(len(self.datasets)):
            w.add(String("dataset"), self.datasets[i])
        for i in range(len(self.policies)):
            w.add(String("policy"), self.policies[i])
        return w^.done()

    def write(self) raises:
        write_text_atomic(self.kv_path(), self.render())


def parse_project(text: String, what: String, root: String = String("")) raises -> ProjectSpec:
    """⚠ AN UNKNOWN KEY RAISES — the `tasks/spec.mojo` policy. A project
    definition is hand-edited, and a typo'd key is a task set that silently
    lost a member."""
    var p = ProjectSpec(String(""), root)
    var ls = kv_lines(text, what)
    var seen_name = False
    for i in range(len(ls)):
        var k = ls[i].key
        var v = ls[i].value
        if k == "name":
            p.name = v
            seen_name = True
        elif k == "created":
            p.created = v
        elif k == "description":
            p.description = v
        elif k == "ref":
            p.refs.append(parse_ref(v, what))
        elif k == "dataset":
            p.datasets.append(v)
        elif k == "policy":
            p.policies.append(v)
        elif k == "schema_version":
            if v != String(SCHEMA_VERSION):
                raise Error(
                    what + ": schema_version " + v + ", this build writes "
                    + String(SCHEMA_VERSION)
                )
        else:
            raise Error(
                what + ": unknown key '" + k + "' on line "
                + String(ls[i].lineno) + ". Known keys are: "
                + String(_KNOWN_KEYS)
            )
    if not seen_name:
        raise Error(what + ": no name=")
    return p^


def load_project(name: String, root: String = String("")) raises -> ProjectSpec:
    # ⚠ `projects_root()`, NOT `DEFAULT_ROOT`. `project_exists` already used
    # the resolved root, so with `NOEIRA_PROJECTS` set the two disagreed: a
    # project that "exists" could not be loaded.
    var r = root if root else projects_root()
    var path = r + "/" + name + "/project.kv"
    with open(path, "r") as fh:
        return parse_project(fh.read(), path, r)


# =============================================================================
# Resolution and drift
# =============================================================================


def projects_root() -> String:
    """`NOEIRA_PROJECTS` from the environment, then from `.env`, else
    `projects`.

    ⚠ `.env` IS NOT THE ENVIRONMENT. Nothing exports it, so a root set there —
    which is where §4 says to set it — used to be ignored by every tool, and
    each would quietly create its own `./projects` beside the real one.
    """
    var v = getenv("NOEIRA_PROJECTS")
    if v.byte_length() > 0:
        return v
    try:
        var env = load_dotenv(String(".env"))
        if "NOEIRA_PROJECTS" in env:
            var e = env["NOEIRA_PROJECTS"]
            if e.byte_length() > 0:
                return e
    except:
        pass
    return String(DEFAULT_ROOT)


def project_dataset_dir(project: String, dataset: String) raises -> String:
    """`<projects root>/<project>/datasets/<dataset>` — where a recording goes.

    ⚠ THE PROJECT MUST EXIST. A typo in `--project` would otherwise create a
    second, unregistered project directory holding an hour of demonstrations.

    ⚠ `datasets/` IS NOT SYNCED AS DEFINITION (`core/project_sync.mojo`): a
    recording's home on other machines is the Hub, via `hf-push-dataset`.
    """
    var root = projects_root()
    if not project_exists(project, root):
        raise Error(
            "no project '" + project + "' under " + root + "/ — create it with"
            " `pixi run project-init " + project + "`, or pull it"
        )
    if (
        dataset.byte_length() == 0
        or dataset.find("/") >= 0
        or dataset.startswith(".")
        or dataset.find(" ") >= 0
    ):
        raise Error(
            "dataset name '" + dataset + "': use one path segment, no spaces,"
            " not starting with '.'"
        )
    return root + "/" + project + "/datasets/" + dataset


def project_exists(name: String, root: String = String("")) -> Bool:
    var r = root if root else projects_root()
    return exists(r + "/" + name + "/project.kv")


def runs_root_for(project: String) -> String:
    """Where a run of `project` goes.

    ⚠⚠ IT FALLS BACK TO `runs/` WHEN THE PROJECT DOES NOT EXIST, and that is
    what lets the seven retrofitted drivers stay unedited. A driver names its
    project; the project layer becomes ACTIVE for it the moment someone runs
    `project-init`, and until then the run lands in the flat `runs/` P0 created.
    No driver has to know which world it is in.
    """
    if project.byte_length() > 0 and project_exists(project):
        return projects_root() + "/" + project + "/runs"
    return String("runs")


def last_commit_touching(path: String) -> String:
    """The short commit that last changed `path`, or empty."""
    try:
        return String(
            run_capture(
                String("git log -1 --format=%h -- ") + quote_arg(path)
                + " 2>/dev/null",
                64,
            ).strip()
        )
    except:
        return String("")


struct RefReport(Copyable, ImplicitlyCopyable, Movable):
    var checked: Int
    var missing: Int
    var drifted: Int

    def __init__(out self):
        self.checked = 0
        self.missing = 0
        self.drifted = 0

    def __init__(out self, *, copy: Self):
        self.checked = copy.checked
        self.missing = copy.missing
        self.drifted = copy.drifted

    def __init__(out self, *, deinit move: Self):
        self.checked = move.checked
        self.missing = move.missing
        self.drifted = move.drifted

    def ok(self) -> Bool:
        return self.missing == 0 and self.drifted == 0


def check_refs(ref spec: ProjectSpec, verbose: Bool = True) raises -> RefReport:
    """Does every `ref=` still point at the file the project pinned?

    ⚠ REPORTS, DOES NOT REFUSE. A box at a different commit is the normal case
    during development, and a project layer that blocked a run over drift would
    be turned off within a day. The value is that `run.kv` can say the pin was
    checked and what it said.

    ⚠⚠ DRIFT IS `git log -1 -- <path>`, NOT HEAD. Comparing against HEAD would
    flag every ref on every unrelated commit; the question is whether THAT FILE
    changed since the project last saw it.
    """
    var r = RefReport()
    for i in range(len(spec.refs)):
        r.checked += 1
        var rf = spec.refs[i].copy()
        if not exists(rf.path):
            r.missing += 1
            if verbose:
                print("    missing:", rf.kind, rf.name, "->", rf.path)
            continue
        if rf.commit.byte_length() == 0:
            continue
        var now = last_commit_touching(rf.path)
        if now.byte_length() > 0 and now != rf.commit:
            r.drifted += 1
            if verbose:
                print(
                    "    drifted:", rf.name, "pinned", rf.commit,
                    "but last changed in", now,
                )
    return r^

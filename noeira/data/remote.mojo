# +--------------------------------------------------------------------------+ #
# | Remote dataset catalog client (R2 via noeira-cloud)
# +--------------------------------------------------------------------------+ #
"""Resolve a dataset name to bytes on disk, pulling from R2 if needed.

Design in `docs/DATA_PLATFORM_PLAN.md` §6b. The monitor Worker is a catalog
and an authorizer only: it returns metadata plus a presigned S3 URL, and the
transfer happens directly against R2. Nothing multi-GB passes through the
Worker.

Credentials come from `.env` exactly as `core/logger.mojo`'s `RemoteLogger`
does — `NOEIRA_CLOUD_URL` + `NOEIRA_CLOUD_API_KEY`, sent as
`Authorization: Bearer <key>`. Same Worker, same key, same convention.

⚠ The catalog routes live at `<base>/datasets`, NOT `<base>/api/datasets`.
`/api/*` on that Worker is guarded by the dashboard's browser-session
middleware, which answers `{"error":"Unauthorized"}` 401 to an API key;
`/runs` and `/ingest` are top-level for the same reason.

**`pull` is a cache, not a toll booth.** If the destination already exists and
its sha256 matches the catalog, no bytes move. That is why the catalog carries
the hash, and why it also carries the generating recipe (`seed` +
`source_commit`): a state-only dataset can be regenerated (walker's 10 M
transitions are 992 MiB but ~2 min of CPU) rather than fetched.
"""

from noeira.core.bytes import string_from_bytes
from noeira.core.dotenv import load_dotenv
from noeira.io.fetch import fetch_to_cache, sha256_file, upload_file
from noeira.io.fileio import file_size
from noeira.io.http import HttpClient
from noeira.io.json import JsonDoc, JsonWriter, parse_json


comptime ENV_URL_KEY = "NOEIRA_CLOUD_URL"
comptime ENV_API_KEY = "NOEIRA_CLOUD_API_KEY"


struct DatasetMeta(Copyable):
    """One catalog row, plus the presigned URL `describe` hands back.

    Typed rather than a `JsonDoc` handle: every field here is read by name at
    a call site, and `meta["sizeBytes"]` returning silently-absent was exactly
    the failure the `PythonObject` version could not rule out.
    """

    var id: String
    var name: String
    var status: String
    var size_bytes: Int
    var sha256: String
    var download_url: String

    def __init__(out self):
        self.id = String("")
        self.name = String("")
        self.status = String("")
        self.size_bytes = 0
        self.sha256 = String("")
        self.download_url = String("")

    def __init__(out self, *, copy: Self):
        self.id = copy.id
        self.name = copy.name
        self.status = copy.status
        self.size_bytes = copy.size_bytes
        self.sha256 = copy.sha256
        self.download_url = copy.download_url

    def __init__(out self, *, deinit move: Self):
        self.id = move.id^
        self.name = move.name^
        self.status = move.status^
        self.size_bytes = move.size_bytes
        self.sha256 = move.sha256^
        self.download_url = move.download_url^


def _opt_string(ref doc: JsonDoc, node: Int, name: String) raises -> String:
    """A string member, or "" when absent or null. `field` already returns -1
    for a missing key, so this is the one place that decides that an absent
    optional is not an error."""
    var n = doc.field(node, name)
    if n < 0 or doc.kind_of(n) != 3:  # J_STRING
        return String("")
    return doc.string(n)


def _opt_int(ref doc: JsonDoc, node: Int, name: String) raises -> Int:
    var n = doc.field(node, name)
    if n < 0 or doc.kind_of(n) != 2:  # J_NUMBER
        return 0
    return doc.integer(n)


struct CatalogAnswer(Movable):
    """A status and a parsed body, for the routes where more than one status
    is a normal answer (`/projects/<slug>/files/*`: 200, 201 and 409)."""

    var status: Int
    var doc: JsonDoc
    var text: String

    def __init__(out self, status: Int, var doc: JsonDoc, var text: String):
        self.status = status
        self.doc = doc^
        self.text = text^

    def __init__(out self, *, deinit move: Self):
        self.status = move.status
        self.doc = move.doc^
        self.text = move.text^

    def take_doc(deinit self) -> JsonDoc:
        return self.doc^


struct RemoteCatalog(Movable & Deinitable):
    """Client for the monitor's `/datasets` routes."""

    var base_url: String
    var api_key: String
    var _http: HttpClient
    """One client for the catalog's lifetime — the register / upload /
    complete sequence in `push` is three calls to one host."""

    def __init__(out self, var base_url: String, var api_key: String) raises:
        self.base_url = String(base_url.removesuffix("/"))
        self.api_key = api_key^
        self._http = HttpClient(30000, 10000)
        self._http.bearer(self.api_key)

    def __init__(out self, *, deinit move: Self):
        self.base_url = move.base_url^
        self.api_key = move.api_key^
        self._http = move._http^

    @staticmethod
    def from_env(path: String = String(".env")) raises -> Self:
        """Load `NOEIRA_CLOUD_URL` / `NOEIRA_CLOUD_API_KEY` from a dotenv file.

        Raises with the missing key named rather than failing later as a 401 —
        an empty key produces a bare `Authorization: Bearer ` header, which is
        indistinguishable at the server from a wrong key.
        """
        var env = load_dotenv(path)
        if ENV_URL_KEY not in env:
            raise Error(
                "RemoteCatalog.from_env: " + String(ENV_URL_KEY)
                + " missing from " + path
            )
        if ENV_API_KEY not in env:
            raise Error(
                "RemoteCatalog.from_env: " + String(ENV_API_KEY)
                + " missing from " + path
            )
        var url = env[ENV_URL_KEY]
        var key = env[ENV_API_KEY]
        if key.byte_length() == 0:
            raise Error(
                "RemoteCatalog.from_env: " + String(ENV_API_KEY)
                + " is empty in " + path
            )
        return Self(url^, key^)

    def _request(
        mut self,
        method: String,
        path: String,
        body: String,
        expect: Int,
    ) raises -> JsonDoc:
        """One HTTP call, returning parsed JSON.

        Errors carry the STATUS and the SERVER'S BODY. That matters here: the
        monitor answers `{"error":"Unauthorized"}` from the dashboard session
        middleware, `{"error":"Missing API key"}` when no Bearer header
        arrived, and `{"error":"Invalid API key"}` when the key did not
        verify. Those three point at three different faults, so swallowing the
        body would throw away the diagnosis — which is why `expect` is passed
        down to `HttpClient.request`, whose raise embeds the body verbatim.

        An empty `body` means "no request body", matching the old
        `body is None`.
        """
        var url = self.base_url + path
        var payload = List[UInt8]()
        var ctype = String("")
        if body.byte_length() > 0:
            for i in range(body.byte_length()):
                payload.append(body.as_bytes()[i])
            ctype = String("application/json")
        var r = self._http.request(method, url, payload^, ctype, expect)
        # ⚠ Move the body out UNCONDITIONALLY. Moving it on only one branch
        # leaves the response partially destroyed on the other, which Mojo
        # rejects as "destroyed out of the middle of a value".
        var raw = r^.take_body()
        if len(raw) == 0:
            return JsonDoc()  # 204 / empty 200: an empty doc, not a parse error
        return parse_json(raw^)

    def _call(
        mut self, method: String, path: String, body: String
    ) raises -> CatalogAnswer:
        """`_request` without an expected status: the caller reads it.

        ⚠ A NON-JSON BODY IS KEPT AS TEXT, NOT RAISED. A 502 from Cloudflare
        is HTML, and the caller's error message is the only place it will be
        seen.
        """
        var url = self.base_url + path
        var payload = List[UInt8]()
        var ctype = String("")
        if body.byte_length() > 0:
            for i in range(body.byte_length()):
                payload.append(body.as_bytes()[i])
            ctype = String("application/json")
        var r = self._http.request(method, url, payload^, ctype, -1)
        var status = r.status
        var raw = r^.take_body()
        var text = string_from_bytes(raw)
        if len(raw) == 0:
            return CatalogAnswer(status, JsonDoc(), text^)
        try:
            return CatalogAnswer(status, parse_json(raw^), text^)
        except:
            return CatalogAnswer(status, JsonDoc(), text^)

    # ── project definition files (private projects) ───────────────────

    def list_projects(mut self) raises -> JsonDoc:
        """This account's projects on the platform, with run counts."""
        return self._request(String("GET"), String("/projects"), String(""), 200)

    def upsert_project(mut self, slug: String, description: String) raises:
        """Create the project on the platform if needed; set its description.

        ⚠ ONLY THE DESCRIPTION. The display NAME is editable on the dashboard,
        and `project.kv` has no display name — its `name=` is the slug — so
        sending one would reset a person's edit on every push.
        """
        var w = JsonWriter()
        w.begin_object()
        w.member(String("description"), description)
        w.end_object()
        _ = self._request(String("PUT"), String("/projects/") + slug, w.done(), 200)

    def project_files(mut self, slug: String) raises -> JsonDoc:
        """`{project, description, max_file_bytes, files: [{path, sha256,
        sizeBytes, download_url}]}`. Raises naming the project on a 404."""
        var a = self._call(
            String("GET"), String("/projects/") + slug + "/files", String("")
        )
        if a.status == 404:
            raise Error(
                "project '" + slug + "' is not on the platform — push it from"
                " the box that has it: pixi run project-push " + slug
            )
        if a.status != 200:
            raise Error(
                "GET /projects/" + slug + "/files -> " + String(a.status)
                + ": " + a.text
            )
        return a^.take_doc()

    def push_project_file(
        mut self,
        slug: String,
        rel_path: String,
        local_path: String,
        sha256: String,
        size_bytes: Int,
        base_sha256: String,
    ) raises -> String:
        """Upload one definition file over `base_sha256`. Returns `uploaded`,
        `unchanged`, or `conflict:<sha the platform holds>`.

        ⚠⚠ A CONFLICT IS A RETURN VALUE, NOT A RAISE. It is an expected answer
        — another box edited the file — and one conflicting file must not stop
        the rest of a push. Everything else unexpected raises with the body.

        ⚠ `sha256` IS HASHED BY THE CALLER, BEFORE THE UPLOAD, and the Worker
        re-hashes what arrives. A file edited between the two fails there with
        a 422 rather than being filed under a digest it does not have.
        """
        var w = JsonWriter()
        w.begin_object()
        w.member(String("path"), rel_path)
        w.member(String("sha256"), sha256)
        w.member(String("size_bytes"), size_bytes)
        w.member(String("base_sha256"), base_sha256)
        w.end_object()
        var body = w.done()
        var base = String("/projects/") + slug + "/files/"

        var pre = self._call(String("POST"), base + "presign", body)
        if pre.status == 200:
            return String("unchanged")
        if pre.status == 409:
            return String("conflict:") + _opt_string(
                pre.doc, pre.doc.root(), String("current_sha256")
            )
        if pre.status != 201:
            raise Error(
                "POST " + base + "presign (" + rel_path + ") -> "
                + String(pre.status) + ": " + pre.text
            )
        var url = _opt_string(pre.doc, pre.doc.root(), String("upload_url"))
        if url.byte_length() == 0:
            raise Error("presign answered 201 without an upload_url")
        var put = upload_file(url, local_path, rel_path, quiet=True)
        if put < 200 or put >= 300:
            raise Error("PUT " + rel_path + " to storage -> " + String(put))

        var done = self._call(String("POST"), base + "complete", body)
        if done.status == 200:
            return String("uploaded")
        if done.status == 409:
            return String("conflict:") + _opt_string(
                done.doc, done.doc.root(), String("current_sha256")
            )
        raise Error(
            "POST " + base + "complete (" + rel_path + ") -> "
            + String(done.status) + ": " + done.text
        )

    # ── project datasets (recordings mirrored on the platform) ────────

    def upsert_dataset(
        mut self, slug: String, name: String, n_episodes: Int, n_frames: Int, fps: Int
    ) raises:
        var w = JsonWriter()
        w.begin_object()
        w.member(String("n_episodes"), n_episodes)
        w.member(String("n_frames"), n_frames)
        w.member(String("fps"), fps)
        w.end_object()
        var a = self._call(
            String("PUT"), String("/projects/") + slug + "/datasets/" + name, w.done()
        )
        if a.status == 404:
            raise Error(
                "project '" + slug + "' is not on the platform — push it first:"
                " pixi run project-push " + slug
            )
        if a.status != 200:
            raise Error("PUT dataset " + name + " -> " + String(a.status) + ": " + a.text)

    def dataset_files(mut self, slug: String, name: String) raises -> JsonDoc:
        """`{dataset, files: [{path, sha256, sizeBytes, status, download_url}]}`.
        An unknown dataset answers an EMPTY doc rather than raising: a first
        push has nothing to compare against yet."""
        var a = self._call(
            String("GET"),
            String("/projects/") + slug + "/datasets/" + name + "/files",
            String(""),
        )
        if a.status == 404:
            return JsonDoc()
        if a.status != 200:
            raise Error("GET dataset files -> " + String(a.status) + ": " + a.text)
        return a^.take_doc()

    def list_datasets(mut self, slug: String) raises -> JsonDoc:
        return self._request(
            String("GET"), String("/projects/") + slug + "/datasets", String(""), 200
        )

    def push_dataset_file(
        mut self,
        slug: String,
        name: String,
        rel_path: String,
        local_path: String,
        sha256: String,
        size_bytes: Int,
    ) raises -> String:
        """Upload one dataset file. Returns `uploaded` or `unchanged`.

        ⚠ `unchanged` IS DECIDED BY THE SERVER (same path, ready, same sha and
        size) — decision 18 — so a watch loop re-offering every file costs one
        small request per file and no bytes.
        """
        var w = JsonWriter()
        w.begin_object()
        w.member(String("path"), rel_path)
        w.member(String("sha256"), sha256)
        w.member(String("size_bytes"), size_bytes)
        w.end_object()
        var body = w.done()
        var base = String("/projects/") + slug + "/datasets/" + name + "/files"
        var reg = self._call(String("POST"), base, body)
        if reg.status == 200:
            return String("unchanged")
        if reg.status != 201:
            raise Error(
                "POST " + base + " (" + rel_path + ") -> " + String(reg.status)
                + ": " + reg.text
            )
        var url = _opt_string(reg.doc, reg.doc.root(), String("upload_url"))
        _ = upload_file(url, local_path, rel_path, quiet=True)
        var done = self._call(String("POST"), base + "/complete", body)
        if done.status != 200:
            raise Error(
                "POST " + base + "/complete (" + rel_path + ") -> "
                + String(done.status) + ": " + done.text
            )
        return String("uploaded")

    # ── read ──────────────────────────────────────────────────────────

    def list_datasets(mut self) raises -> JsonDoc:
        """The raw catalog listing.

        Left as a document rather than a typed list: the shape of this route
        is the monitor's to decide, and a struct here would have to be kept in
        step with it for a call nothing but a smoke test makes.
        """
        return self._request(String("GET"), String("/datasets"), String(""), 200)

    def describe(mut self, id: String) raises -> DatasetMeta:
        """Catalog row + a presigned download URL. Raises 409 if the upload
        never completed — a `pending` row is not served, so a crashed upload
        cannot advertise a truncated object."""
        var doc = self._request(
            String("GET"), String("/datasets/") + id, String(""), 200
        )
        var root = doc.root()
        var m = DatasetMeta()
        m.id = _opt_string(doc, root, String("id"))
        m.name = _opt_string(doc, root, String("name"))
        m.status = _opt_string(doc, root, String("status"))
        m.size_bytes = _opt_int(doc, root, String("sizeBytes"))
        m.sha256 = _opt_string(doc, root, String("sha256"))
        m.download_url = _opt_string(doc, root, String("download_url"))
        if m.download_url.byte_length() == 0:
            raise Error(
                "describe(" + id + "): the catalog returned no download_url"
            )
        return m^

    def pull(
        mut self, id: String, dest: String, label: String = String("")
    ) raises -> String:
        """Ensure `dest` holds the dataset. Returns `dest`.

        No bytes move if `dest` already matches the catalog's sha256.
        """
        var meta = self.describe(id)
        var lbl = label if label.byte_length() > 0 else id
        return fetch_to_cache(
            meta.download_url, dest, meta.sha256, meta.size_bytes, lbl
        )

    # ── write ─────────────────────────────────────────────────────────

    def push_artifact(
        mut self,
        run_id: String,
        rel_path: String,
        local_path: String,
        kind: String = String("other"),
    ) raises -> String:
        """Register, upload, then confirm ONE run artifact. Returns its id.

        The same three steps as `push` above and for the same reason: the row
        stays `pending` between register and complete, so an interrupted
        transfer leaves a row the catalog refuses to serve rather than one
        advertising a truncated object.

        ⚠ `rel_path` IS RELATIVE TO THE RUN DIRECTORY ("checkpoints/best.ckpt")
        and `local_path` is where those bytes actually are. They are separate
        arguments because they are separate facts: the first is the identity
        the artifact keeps forever — it is the same string the local `run.kv`
        carries on its `artifact=` line, which is what makes a later
        `project-push` a resume rather than a re-send — while the second is an
        accident of where the run happened to be working.

        ⚠ NOTHING IS PRINTED. This is called from `ArtifactSink`'s worker
        thread, and `http_sink.mojo` states the rule: a worker that prints can
        interleave into training output from a second thread. Hence
        `quiet=True` below, which also turns off libcurl's progress meter.

        ⚠ RE-REGISTERING THE SAME `rel_path` IS THE SUPERSEDE PATH, not an
        error. The monitor keeps one row per (run, path) — within a run,
        `best.ckpt` is a role — so this returns the SAME id and resets the row
        to pending, which is exactly what a newer `best` should do.
        """
        var w = JsonWriter()
        w.begin_object()
        w.member(String("run_id"), run_id)
        w.member(String("path"), rel_path)
        w.member(String("kind"), kind)
        w.end_object()

        var reg = self._request(
            String("POST"), String("/artifacts"), w.done(), 201
        )
        var root = reg.root()
        var id = _opt_string(reg, root, String("id"))
        var upload_url = _opt_string(reg, root, String("upload_url"))
        if id.byte_length() == 0 or upload_url.byte_length() == 0:
            raise Error(
                "push_artifact: the registration answered without an id or an"
                " upload_url"
            )

        _ = upload_file(upload_url, local_path, rel_path, quiet=True)

        # ⚠ HASHED AFTER THE UPLOAD, NOT BEFORE. A checkpoint written by a run
        # that is still training can change under us; hashing first and
        # uploading second would store a digest for bytes that never went. This
        # way the digest describes a file that has already been read once —
        # still not atomic, but wrong in the direction that a later `pull`
        # DETECTS rather than one it silently trusts.
        var size = file_size(local_path)
        var sha = sha256_file(local_path)

        var done = JsonWriter()
        done.begin_object()
        done.member(String("size_bytes"), size)
        done.member(String("sha256"), sha)
        done.end_object()
        _ = self._request(
            String("POST"),
            String("/artifacts/") + id + "/complete",
            done.done(),
            200,
        )
        return id^

    def record_policy(
        mut self,
        project: String,
        name: String,
        run_id: String,
        checkpoint: String,
        sha256: String,
        bytes: Int,
        note: String,
        supersedes: String,
        promoted_at: String,
    ) raises -> String:
        """Record a promotion. Returns the row id.

        ⚠⚠ THE BOX IS THE RECORD AND THIS IS THE INDEX (§10). The weights are
        already hard-linked and `policies/<name>.kv` already written by the
        time this is called, so a caller must NOT let a failure here fail the
        promotion — `project-promote` reports it and carries on. A promotion
        that only exists locally is a normal state; one that exists only on the
        server would be a lie.

        ⚠ `promoted_at` IS THE BOX'S CLOCK, sent explicitly. A promotion made
        offline and pushed a week later belongs in the history where the PERSON
        put it, so the server does not substitute its own `now()`.
        """
        var w = JsonWriter()
        w.begin_object()
        w.member(String("project"), project)
        w.member(String("name"), name)
        w.member(String("run_id"), run_id)
        w.member(String("checkpoint"), checkpoint)
        if sha256.byte_length() == 64:
            w.member(String("sha256"), sha256)
        if bytes > 0:
            w.member(String("bytes"), bytes)
        w.member(String("note"), note)
        w.member(String("supersedes"), supersedes)
        w.member(String("promoted_at"), promoted_at)
        w.end_object()
        var doc = self._request(
            String("POST"), String("/policies"), w.done(), 201
        )
        return _opt_string(doc, doc.root(), String("id"))

    def policies_of(
        mut self, project: String, current_only: Bool = False
    ) raises -> JsonDoc:
        """A project's promotion history, newest first (`current_only` gives
        just who fills each role now)."""
        var q = String("/policies?project=") + project
        if current_only:
            q += "&current=1"
        return self._request(String("GET"), q, String(""), 200)

    def describe_artifact(mut self, id: String) raises -> DatasetMeta:
        """Artifact row + a presigned download URL.

        ⚠ REUSES `DatasetMeta` RATHER THAN A NEAR-IDENTICAL TWIN. The fields a
        transfer needs are the same four — id, status, size, digest, url — and
        a second struct differing only in its name is a second thing to keep in
        step with the Worker. `name` carries the artifact's relative path,
        which is its identity here.

        ⚠ RAISES 409 IF THE UPLOAD NEVER COMPLETED. A pending row is not
        served, so a crashed upload cannot advertise a truncated object.
        """
        var doc = self._request(
            String("GET"), String("/artifacts/") + id, String(""), 200
        )
        var root = doc.root()
        var m = DatasetMeta()
        m.id = _opt_string(doc, root, String("id"))
        m.name = _opt_string(doc, root, String("path"))
        m.status = _opt_string(doc, root, String("status"))
        m.size_bytes = _opt_int(doc, root, String("sizeBytes"))
        m.sha256 = _opt_string(doc, root, String("sha256"))
        m.download_url = _opt_string(doc, root, String("download_url"))
        if m.download_url.byte_length() == 0:
            raise Error(
                "describe_artifact(" + id + "): no download_url in the reply"
            )
        return m^

    def upsert_run(
        mut self,
        run_id: String,
        run_name: String,
        keys: List[String],
        vals: List[String],
    ) raises:
        """`POST /runs` — the registration a live `RemoteLogger` sends, from a
        run's files instead of its process.

        ⚠⚠ WHY `project-push` NEEDS IT. `POST /artifacts` answers 404 "unknown
        run" for a run the monitor never saw, so a run trained offline — no
        `.env`, `ACT_NO_MONITOR`, a box with no network — could never be
        pushed at all. The server's `/runs` is an UPSERT, so re-registering a
        run it already has is harmless AS LONG AS the config sent is the one
        it registered with; the caller sends `run.kv` + `metrics.config.kv`,
        which is exactly that.
        """
        var w = JsonWriter()
        w.begin_object()
        w.member(String("run_id"), run_id)
        w.member(String("run_name"), run_name)
        w.key(String("config"))
        w.begin_object()
        for i in range(min(len(keys), len(vals))):
            w.member(keys[i], vals[i])
        w.end_object()
        w.end_object()
        _ = self._request(String("POST"), String("/runs"), w.done(), 201)

    def finish_run(
        mut self, run_id: String, status: String, outcome: String
    ) raises:
        """`POST /runs/<id>/finish`. Only for a terminal status the RUN wrote
        (`done|killed|crashed`); a `running` record is the server's to call
        stale, never the client's."""
        var w = JsonWriter()
        w.begin_object()
        w.member(String("status"), status)
        w.member(String("outcome"), outcome)
        w.end_object()
        _ = self._request(
            String("POST"), String("/runs/") + run_id + "/finish", w.done(), 200
        )

    def runs_of(mut self, project: String) raises -> JsonDoc:
        """This account's runs of `project` on the monitor, newest first —
        `GET /runs?project=`. Rows carry `runId`, `status`, `outcome`, `tag`,
        `task`. An unknown project is an empty list, not an error."""
        return self._request(
            String("GET"), String("/runs?project=") + project, String(""), 200
        )

    def artifacts_of(mut self, run_id: String) raises -> JsonDoc:
        """Every artifact row for a run, pending ones included.

        ⚠ THE PENDING ROWS ARE THE POINT for `project-push`: they are precisely
        the ones whose bytes may still only exist on this box.
        """
        return self._request(
            String("GET"),
            String("/artifacts?run_id=") + run_id,
            String(""),
            200,
        )

    def push(
        mut self,
        path: String,
        name: String,
        version: String = String("v1"),
        env_id: String = String(""),
        n_rows: Int = 0,
        n_episodes: Int = 0,
        seed: Int = 0,
        source_commit: String = String(""),
        columns: String = String(""),
    ) raises -> String:
        """Register, upload, then confirm. Returns the dataset id.

        The three steps are separate on purpose: the row stays `pending`
        between register and complete, so an interrupted upload leaves a row
        that `describe` refuses to serve rather than one advertising a
        truncated object.
        """
        var w = JsonWriter()
        w.begin_object()
        w.member(String("name"), name)
        w.member(String("version"), version)
        if env_id.byte_length() > 0:
            w.member(String("env_id"), env_id)
        if n_rows > 0:
            w.member(String("n_rows"), n_rows)
        if n_episodes > 0:
            w.member(String("n_episodes"), n_episodes)
        if seed != 0:
            w.member(String("seed"), seed)
        if source_commit.byte_length() > 0:
            w.member(String("source_commit"), source_commit)
        if columns.byte_length() > 0:
            w.member(String("columns"), columns)
        w.end_object()

        var reg = self._request(
            String("POST"), String("/datasets"), w.done(), 201
        )
        var root = reg.root()
        var id = _opt_string(reg, root, String("id"))
        var upload_url = _opt_string(reg, root, String("upload_url"))
        if id.byte_length() == 0 or upload_url.byte_length() == 0:
            raise Error(
                "push: the registration answered without an id or an"
                " upload_url"
            )

        _ = upload_file(upload_url, path, id)

        var size = file_size(path)
        var sha = sha256_file(path)

        var done = JsonWriter()
        done.begin_object()
        done.member(String("size_bytes"), size)
        done.member(String("sha256"), sha)
        done.end_object()
        _ = self._request(
            String("POST"),
            String("/datasets/") + id + "/complete",
            done.done(),
            200,
        )
        return id^

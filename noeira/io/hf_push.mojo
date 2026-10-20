# +--------------------------------------------------------------------------+ #
# | Writing to the HuggingFace Hub, with libcurl and nothing else
# +--------------------------------------------------------------------------+ #
"""Create a repo, upload files, commit — the write half of `io/hf.mojo`.

    var files = List[HubUpload]()
    files.append(HubUpload(String("meta/info.json"), local + "/meta/info.json"))
    files.append(HubUpload(String("data/chunk-000/file-000.parquet"), ...))

    var p = HubPush(String("DenisLabs/my-dataset"))
    p.create_repo(private=True)
    p.push(files, String("Add episodes 0-49"))

`huggingface_hub` is one `pip` dependency and one Python process. The Hub's
write API is four JSON endpoints and a file PUT, and `io/http.mojo` already
streams a file body, so this is protocol plumbing rather than a port.

## The protocol, in the order it happens

    1. POST /api/repos/create                          {"name","organization","type","private"}
    2. POST /api/datasets/<repo>/preupload/<rev>        per file: path, size, base64 sample
       -> per file: uploadMode "lfs" | "regular", shouldIgnore
    3. POST /datasets/<repo>.git/info/lfs/objects/batch {"operation":"upload","objects":[{oid,size}]}
       -> per object: actions.upload.href (+ headers), or nothing if the Hub
          already has those bytes
    4. PUT  <href>                                     the file, streamed
    5. POST /api/datasets/<repo>/commit/<rev>          NDJSON: header, then one
                                                       lfsFile / file line each

Step 3 returning NO action for an object is SUCCESS, not an error: it means
the Hub already stores that content, and step 4 is skipped. A dataset
re-pushed after changing one episode uploads one file.

⚠ **THE PUT IN STEP 4 MUST NOT CARRY THE `Authorization` HEADER.** The href is
a presigned URL whose signature already IS the authorization; S3 answers a
request carrying both with `400 Only one auth mechanism allowed`. That is why
`_transfer` builds its own client rather than reusing the one holding the
token — the bug it prevents is a 400 from a service that is not HuggingFace,
against a URL that does not appear in this file.

⚠ **`oid` IS THE SHA-256 OF THE FILE, AND IT IS ALSO WHAT THE COMMIT SENDS.**
Steps 3 and 5 must use the same digest; a file rewritten between them commits
a pointer to bytes nobody uploaded, and the repo then has a file that 404s on
download. `HubUpload` hashes ONCE, at construction, and both steps read that.

## Multipart: negotiated, not triggered by size

⚠ **THE HUB PICKS THE TRANSFER FROM WHAT YOU ADVERTISE, NOT FROM THE FILE
SIZE.** Measured against the live Hub on 2026-08-31:

    transfers: ["basic", "multipart"]  ->  multipart for EVERY size,
                                           1 MB included, chunk_size 16 MB
    transfers: ["basic"]               ->  a single presigned PUT for every
                                           size, up to 5 GB

So this advertises **`basic` only** and multipart never arises. The mitigation
this file used to describe — roll the videos smaller to stay under a
"multipart threshold" — was based on a wrong model: there is no size
threshold, and a 1 MB parquet would have gone multipart just as readily as a
200 MB mp4.

⚠ **5 GB IS A REAL WALL, AND THE HUB SAYS SO CLEARLY.** Past it the batch is
rejected with *"You need to configure your repository to enable upload of
files > 5GB. Run `hf lfs-enable-largefiles`"*. That is the same S3 single-PUT
limit `io/fetch.mojo:168` already refuses to exceed for R2. `_transfer` checks
the size up front so the failure names the file rather than arriving as a
batch-level 400 with no filename in it.

⚠ A `chunk_size` in the response is therefore a CONTRACT VIOLATION, not an
expected branch — it means the Hub chose a transfer nobody advertised — so
`upload_lfs` still raises on it rather than PUTting the whole file at a
multipart href.
"""

from std.os import getenv

from .base64 import b64_encode_n
from .fileio import file_size, read_file_bytes, read_file_range
from .hf import hf_client, hf_token
from .http import HttpClient, HttpResponse
from .json import J_ARRAY, JsonDoc, JsonWriter, json_quote, parse_json
from .sha256 import sha256_file


comptime HF_ENDPOINT = "https://huggingface.co"

comptime SAMPLE_BYTES = 512
"""What `preupload` wants to look at. `huggingface_hub/lfs.py:114` reads
exactly this many bytes, and the server uses them to sniff the file type."""

comptime LFS_JSON = "application/vnd.git-lfs+json"

comptime LFS_SINGLE_PUT_LIMIT = 5_000_000_000
"""Measured: at 5,100 MB the batch endpoint answers 400 and names
`hf lfs-enable-largefiles`. Same wall as R2's, which `io/fetch.mojo` already
refuses to cross."""


def hf_whoami(var token: String = String("")) raises -> String:
    """The namespace the token belongs to — its user or org `name`.

    ⚠ PARSE THE JSON. Two callers hand-rolled `find('"name"')` on the response
    and BOTH picked up a nested field, yielding the namespace `,` and then a
    403 against a repo id that looked almost right (`,/noeira-roundtrip`).
    The second one was written after the first had been fixed, which is what
    `_a_rule_written_inline_twice_drifts` predicts. One implementation, here.
    """
    if token == "":
        token = hf_token()
    var c = HttpClient(0, 30000)
    c.bearer(token)
    var r = c.get(String(HF_ENDPOINT) + "/api/whoami-v2")
    if not r.ok():
        raise Error(
            "hf: whoami -> " + String(r.status) + ": " + r.text()
            + "  (is HF_TOKEN set, and does it have `write` scope?)"
        )
    var doc = _doc(r^)
    var n = doc.field(doc.root(), String("name"))
    if n < 0:
        raise Error("hf: whoami returned no `name`")
    return doc.string(n)


def _doc(var r: HttpResponse) raises -> JsonDoc:
    """Parse a response body as JSON, consuming it.

    ⚠ `take_body` is `deinit self` — the body cannot be moved out of a
    still-live response (`r.body^` is "destroyed out of the middle of a
    value"), so the response is consumed here rather than borrowed.
    """
    return parse_json(r^.take_body())


struct HubUpload(Copyable, Movable):
    """One file on its way into a repo.

    The digest is computed ONCE here. Both the LFS batch and the commit send
    it, and they must agree — see the module docstring.
    """

    var repo_path: String
    """Path inside the repo, e.g. `data/chunk-000/file-000.parquet`."""
    var local_path: String
    var size: Int
    var oid: String
    """Lowercase hex SHA-256. The git-LFS object id."""
    var mode: String
    """`""` until `preupload` answers, then `"lfs"` or `"regular"`."""
    var ignored: Bool
    """The repo's `.gitignore` excludes it; the commit must skip it."""
    var uploaded: Bool
    """False when the Hub already had these bytes and step 4 was skipped."""

    def __init__(out self, var repo_path: String, var local_path: String) raises:
        self.size = file_size(local_path)
        self.oid = sha256_file(local_path)
        self.repo_path = repo_path^
        self.local_path = local_path^
        self.mode = String("")
        self.ignored = False
        self.uploaded = False

    def __init__(out self, *, copy: Self):
        self.repo_path = copy.repo_path.copy()
        self.local_path = copy.local_path.copy()
        self.size = copy.size
        self.oid = copy.oid.copy()
        self.mode = copy.mode.copy()
        self.ignored = copy.ignored
        self.uploaded = copy.uploaded

    def __init__(out self, *, deinit move: Self):
        self.repo_path = move.repo_path^
        self.local_path = move.local_path^
        self.size = move.size
        self.oid = move.oid^
        self.mode = move.mode^
        self.ignored = move.ignored
        self.uploaded = move.uploaded

    def sample_b64(self) raises -> String:
        """The first `SAMPLE_BYTES` bytes, base64'd.

        ⚠ CLAMPED TO THE FILE SIZE. `read_file_range` demands EXACTLY the
        count it was asked for and raises otherwise, so an unclamped 512-byte
        read fails on every file shorter than that — which is most of a
        dataset's metadata, `meta/info.json` included.
        """
        var want = SAMPLE_BYTES if self.size > SAMPLE_BYTES else self.size
        if want <= 0:
            return String("")
        var head = read_file_range(self.local_path, 0, want)
        return b64_encode_n(head, len(head))


struct HubPush(Movable & Deinitable):
    """A token-carrying client pinned to one repo."""

    var client: HttpClient
    var repo: String
    var kind: String
    """`datasets` or `models`. Only the plural form appears in URLs."""
    var revision: String
    var endpoint: String
    var verbose: Bool

    def __init__(
        out self,
        var repo: String,
        var kind: String = String("datasets"),
        var revision: String = String("main"),
        var token: String = String(""),
        verbose: Bool = True,
    ) raises:
        if token == "":
            token = hf_token()
        self.client = HttpClient(0, 30000)
        self.client.stall_guard(1024, 60)
        self.client.bearer(token)
        self.client.max_body(1 << 26)
        self.repo = repo^
        self.kind = kind^
        self.revision = revision^
        self.endpoint = String(HF_ENDPOINT)
        self.verbose = verbose

    def __init__(out self, *, deinit move: Self):
        self.client = move.client^
        self.repo = move.repo^
        self.kind = move.kind^
        self.revision = move.revision^
        self.endpoint = move.endpoint^
        self.verbose = move.verbose

    def __deinit__(deinit self):
        pass

    def _api(self, tail: String) -> String:
        return self.endpoint + "/api/" + self.kind + "/" + self.repo + tail

    def _namespace(self) raises -> Tuple[String, String]:
        """`Org/name` -> (`Org`, `name`), or (``, `name`).

        ⚠ ONE PLACE, DELIBERATELY. Both `/api/repos/create` and
        `/api/repos/delete` take the namespace SPLIT OUT — `{"name": ...,
        "organization": ...}` — not the `Org/name` id the rest of the API
        uses. Sending the joined id to delete answers **404**, which reads as
        "no such repo" rather than "wrong shape", so the repo silently
        survives. Writing the split twice is how the two drift.
        """
        var slash = self.repo.find("/")
        if slash <= 0:
            return (String(""), self.repo.copy())
        return (
            String(self.repo[byte=0:slash]),
            String(self.repo[byte = slash + 1 : self.repo.byte_length()]),
        )

    def _singular(self) raises -> String:
        """`datasets` -> `dataset`. The repo endpoints want the singular."""
        return String(self.kind[byte=0 : self.kind.byte_length() - 1])

    def _log(self, msg: String):
        if self.verbose:
            print("  [hub] " + msg)

    # ── 1. the repo ───────────────────────────────────────────────────

    def create_repo(mut self, private: Bool = True) raises -> Bool:
        """Create the repo. Returns False when it already existed.

        ⚠ AN EXISTING REPO ANSWERS 409, AND THAT IS NOT AN ERROR HERE. Every
        push after the first hits it, so treating it as failure would make the
        second recording session fail. `private` is IGNORED by the Hub on an
        existing repo — it does not retroactively hide a public one.
        """
        var ns = self._namespace()
        var org = ns[0].copy()
        var name = ns[1].copy()
        var singular = self._singular()
        var w = JsonWriter()
        w.begin_object()
        w.member(String("name"), name)
        if org != "":
            w.member(String("organization"), org)
        w.member(String("type"), singular)
        w.key(String("private"))
        w.boolean(private)
        w.end_object()

        var r = self.client.post_json(
            self.endpoint + "/api/repos/create", w.done()
        )
        if r.status == 409:
            self._log("repo " + self.repo + " already exists")
            return False
        if not r.ok():
            raise Error(
                "hf_push: creating " + self.repo + " -> " + String(r.status)
                + ": " + r.text()
            )
        self._log(
            "created " + self.kind + "/" + self.repo
            + (" (private)" if private else " (PUBLIC)")
        )
        return True

    def delete_repo(mut self) raises -> Bool:
        """Delete the repo. Returns False if it was not there.

        ⚠ IRREVERSIBLE, AND IT TAKES THE WHOLE REPO. Nothing in the recording
        path calls this; it exists for the scratch repos `tools/hf/probe_push`
        creates, which would otherwise pile up on the account.
        """
        var ns = self._namespace()
        var w = JsonWriter()
        w.begin_object()
        w.member(String("name"), ns[1])
        if ns[0] != "":
            w.member(String("organization"), ns[0])
        w.member(String("type"), self._singular())
        w.end_object()

        var body = List[UInt8]()
        var text = w.done()
        for i in range(text.byte_length()):
            body.append(text.as_bytes()[i])
        var r = self.client.request(
            String("DELETE"),
            self.endpoint + "/api/repos/delete",
            body,
            String("application/json"),
        )
        if r.status == 404:
            self._log("nothing to delete: " + self.repo)
            return False
        if not r.ok():
            raise Error(
                "hf_push: deleting " + self.repo + " -> " + String(r.status)
                + ": " + r.text()
            )
        self._log("deleted " + self.kind + "/" + self.repo)
        return True

    # ── 2. preupload: lfs or regular? ─────────────────────────────────

    def preupload(mut self, mut files: List[HubUpload]) raises:
        """Ask the Hub how each file should be sent, and fill in `mode`."""
        if len(files) == 0:
            return
        var w = JsonWriter()
        w.begin_object()
        w.key(String("files"))
        w.begin_array()
        for i in range(len(files)):
            w.begin_object()
            w.member(String("path"), files[i].repo_path)
            w.member(String("sample"), files[i].sample_b64())
            w.member(String("size"), files[i].size)
            w.end_object()
        w.end_array()
        w.end_object()

        var r = self.client.post_json(
            self._api("/preupload/" + self.revision), w.done()
        )
        if not r.ok():
            raise Error(
                "hf_push: preupload -> " + String(r.status) + ": " + r.text()
            )
        var doc = _doc(r^)
        var arr = doc.field(doc.root(), String("files"))
        if doc.kind_of(arr) != J_ARRAY:
            raise Error("hf_push: preupload returned no `files` array")

        # ⚠ MATCH BY PATH, NOT BY POSITION. The response is not promised in
        # request order, and a mode applied to the wrong file sends a 200 MB
        # mp4 inline as base64.
        var seen = 0
        for i in range(doc.size(arr)):
            var ent = doc.at(arr, i)
            var path = doc.string(doc.field(ent, String("path")))
            var mode = doc.string(doc.field(ent, String("uploadMode")))
            var ig = doc.field(ent, String("shouldIgnore"))
            for f in range(len(files)):
                if files[f].repo_path != path:
                    continue
                files[f].mode = mode
                if ig >= 0:
                    files[f].ignored = doc.boolean(ig)
                seen += 1
                break
        if seen != len(files):
            raise Error(
                "hf_push: preupload answered for " + String(seen) + " of "
                + String(len(files)) + " files"
            )
        var n_lfs = 0
        for i in range(len(files)):
            if files[i].mode == "lfs":
                n_lfs += 1
        self._log(
            "preupload: " + String(n_lfs) + " lfs, "
            + String(len(files) - n_lfs) + " regular"
        )

    # ── 3 + 4. the LFS objects ────────────────────────────────────────

    def upload_lfs(mut self, mut files: List[HubUpload]) raises:
        """Batch, then PUT each object the Hub asks for."""
        var want = List[Int]()
        for i in range(len(files)):
            if files[i].mode == "lfs" and not files[i].ignored:
                want.append(i)
        if len(want) == 0:
            return

        var w = JsonWriter()
        w.begin_object()
        w.member(String("operation"), String("upload"))
        # ⚠ `basic` ALONE, AND THAT IS THE WHOLE MULTIPART STORY. See the
        # module docstring: advertising `multipart` makes the Hub choose it
        # for EVERY object, 1 MB included. Advertising only `basic` gets a
        # single presigned PUT at every size up to 5 GB. Adding "multipart"
        # back here does not add a capability, it removes one.
        w.key(String("transfers"))
        w.begin_array()
        w.string(String("basic"))
        w.end_array()
        w.member(String("hash_algo"), String("sha256"))
        w.key(String("ref"))
        w.begin_object()
        w.member(String("name"), self.revision)
        w.end_object()
        w.key(String("objects"))
        w.begin_array()
        for k in range(len(want)):
            w.begin_object()
            w.member(String("oid"), files[want[k]].oid)
            w.member(String("size"), files[want[k]].size)
            w.end_object()
        w.end_array()
        w.end_object()

        # The batch endpoint speaks its own media type, on both sides.
        self.client.header(String("Accept"), String(LFS_JSON))
        var body = List[UInt8]()
        var text = w.done()
        for i in range(text.byte_length()):
            body.append(text.as_bytes()[i])
        var r = self.client.request(
            String("POST"),
            self.endpoint + "/" + self.kind + "/" + self.repo
            + ".git/info/lfs/objects/batch",
            body,
            String(LFS_JSON),
        )
        self.client.header(String("Accept"), String("*/*"))
        if not r.ok():
            raise Error(
                "hf_push: lfs batch -> " + String(r.status) + ": " + r.text()
            )

        var doc = _doc(r^)
        var arr = doc.field(doc.root(), String("objects"))
        if doc.kind_of(arr) != J_ARRAY:
            raise Error("hf_push: the lfs batch returned no `objects` array")

        for i in range(doc.size(arr)):
            var ent = doc.at(arr, i)
            var oid = doc.string(doc.field(ent, String("oid")))
            var idx = -1
            for k in range(len(want)):
                if files[want[k]].oid == oid:
                    idx = want[k]
                    break
            if idx < 0:
                raise Error(
                    "hf_push: the lfs batch answered for an object we did not"
                    " ask about: " + oid
                )

            var err = doc.field(ent, String("error"))
            if err >= 0:
                raise Error(
                    "hf_push: the Hub refused " + files[idx].repo_path + ": "
                    + doc.string(doc.field(err, String("message")))
                )

            var actions = doc.field(ent, String("actions"))
            if actions < 0:
                # ⚠ NO ACTIONS MEANS THE HUB ALREADY HAS THESE BYTES. Success.
                files[idx].uploaded = False
                self._log("already on the Hub: " + files[idx].repo_path)
                continue

            var up = doc.field(actions, String("upload"))
            if up < 0:
                raise Error(
                    "hf_push: no upload action for " + files[idx].repo_path
                )

            var hdr = doc.field(up, String("header"))
            if hdr >= 0 and doc.field(hdr, String("chunk_size")) >= 0:
                raise Error(
                    "hf_push: the Hub answered MULTIPART for "
                    + files[idx].repo_path + " even though only `basic` was"
                    " advertised. That is a protocol change, not a size"
                    " problem — re-run tools/hf/probe_push.mojo and read what"
                    " the batch endpoint now negotiates."
                )

            var href = doc.string(doc.field(up, String("href")))
            self._transfer(files[idx], href)
            files[idx].uploaded = True

    def _transfer(self, ref f: HubUpload, var href: String) raises:
        """PUT one object at its presigned URL.

        ⚠ A FRESH CLIENT, DELIBERATELY. See the module docstring: the presigned
        signature is the authorization, and S3 rejects a request that also
        carries `Authorization: Bearer`.
        """
        if f.size > LFS_SINGLE_PUT_LIMIT:
            raise Error(
                "hf_push: " + f.repo_path + " is "
                + String(f.size // 1000000) + " MB. The Hub refuses a single"
                " object past 5 GB unless the repo has large-file support"
                " enabled (`hf lfs-enable-largefiles`), and this client only"
                " speaks the single-PUT transfer. For a LeRobot dataset the"
                " lever is `video_files_size_in_mb` in meta/info.json."
            )
        var c = HttpClient(0, 30000)
        c.stall_guard(1024, 60)
        var label = f.repo_path.copy() if self.verbose else String("")
        var r = c.upload(href, f.local_path.copy(), String("PUT"), label)
        if not r.ok():
            raise Error(
                "hf_push: PUT of " + f.repo_path + " -> " + String(r.status)
                + ": " + r.text()
            )

    # ── 5. the commit ─────────────────────────────────────────────────

    def commit(
        mut self,
        ref files: List[HubUpload],
        var message: String,
        var description: String = String(""),
    ) raises -> String:
        """POST the NDJSON commit. Returns the commit URL."""
        var nd = String("")

        var h = JsonWriter()
        h.begin_object()
        h.member(String("key"), String("header"))
        h.key(String("value"))
        h.begin_object()
        h.member(String("summary"), message)
        h.member(String("description"), description)
        h.end_object()
        h.end_object()
        nd += h.done()
        nd += "\n"

        var n_lfs = 0
        var n_reg = 0
        for i in range(len(files)):
            if files[i].ignored:
                continue
            var w = JsonWriter()
            w.begin_object()
            if files[i].mode == "lfs":
                w.member(String("key"), String("lfsFile"))
                w.key(String("value"))
                w.begin_object()
                w.member(String("path"), files[i].repo_path)
                w.member(String("algo"), String("sha256"))
                w.member(String("oid"), files[i].oid)
                w.member(String("size"), files[i].size)
                w.end_object()
                n_lfs += 1
            else:
                # ⚠ A REGULAR FILE TRAVELS INSIDE THE COMMIT, base64'd. That
                # is fine for `meta/info.json` and fatal for anything large,
                # which is exactly what `uploadMode` is for — so trust the
                # server's answer here rather than a size threshold of ours.
                var bytes = read_file_bytes(files[i].local_path)
                w.member(String("key"), String("file"))
                w.key(String("value"))
                w.begin_object()
                w.member(String("path"), files[i].repo_path)
                w.member(String("content"), b64_encode_n(bytes, len(bytes)))
                w.member(String("encoding"), String("base64"))
                w.end_object()
                n_reg += 1
            w.end_object()
            nd += w.done()
            nd += "\n"

        var body = List[UInt8]()
        for i in range(nd.byte_length()):
            body.append(nd.as_bytes()[i])
        var r = self.client.request(
            String("POST"),
            self._api("/commit/" + self.revision),
            body,
            String("application/x-ndjson"),
        )
        if not r.ok():
            raise Error(
                "hf_push: commit -> " + String(r.status) + ": " + r.text()
            )
        var doc = _doc(r^)
        var url_node = doc.field(doc.root(), String("commitUrl"))
        var url = doc.string(url_node) if url_node >= 0 else String("")
        self._log(
            "committed " + String(n_lfs) + " lfs + " + String(n_reg)
            + " regular file(s)"
        )
        return url^

    # ── the whole thing ───────────────────────────────────────────────

    def push(
        mut self,
        mut files: List[HubUpload],
        var message: String,
        var description: String = String(""),
    ) raises -> String:
        self.preupload(files)
        self.upload_lfs(files)
        return self.commit(files, message^, description^)

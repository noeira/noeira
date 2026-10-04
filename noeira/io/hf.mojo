# +--------------------------------------------------------------------------+ #
# | The HuggingFace Hub, with `curl` and nothing else
# +--------------------------------------------------------------------------+ #
"""Cache locations and file downloads for the Hub, without `huggingface_hub`.

    from noeira.io.hf import hf_download_file, HF_MODEL

    var path = hf_download_file(
        String("timm/resnet18.tv_in1k"), String("model.safetensors"), HF_MODEL
    )

`huggingface_hub` is one `pip` dependency and one Python process; `curl` plus
the Hub's public API is neither, and the JSON reader this needs already exists
for `meta/info.json`.

    GET https://huggingface.co/api/{models,datasets}/<repo>/tree/<rev>?recursive=1
      -> [ {"type": "file"|"directory", "path": "...", "size": N, ...}, ... ]
    GET https://huggingface.co/[datasets/]<repo>/resolve/<rev>/<path>
      -> the bytes (302 to the CDN, hence `curl -L`)

⚠ The two endpoints are NOT symmetric: a model's `resolve` URL has no kind
segment (`huggingface.co/<repo>/resolve/...`) while a dataset's does
(`huggingface.co/datasets/<repo>/resolve/...`). The API path carries the kind
in both cases. Getting that wrong yields an HTML 404 page saved as your
weights — `curl -f` catches it, which is why the flag is not optional.

⚠ Files land at `<dest>.part` and are renamed on success, and a file whose size
already matches the listing is SKIPPED — so an interrupted 47 MB download
resumes at file granularity rather than starting over. Size is a weak check
next to a hash, but it is the one the listing gives for free on every file,
LFS or not.

`noeira/data/lerobot.mojo` holds the dataset-shaped operations built on this
(whole-repo download, the `huggingface_hub` snapshot cache, resolution order).
"""

from noeira.core.bytes import string_from_bytes

from std.os import getenv, makedirs
from std.os.path import exists

from .fileio import file_size, read_file_bytes, remove_file, rename_over
from .http import HttpClient
from .json import J_ARRAY, parse_json


comptime HF_MODEL = "models"
comptime HF_DATASET = "datasets"


def home_dir() raises -> String:
    var h = getenv("HOME")
    if h == "":
        raise Error("hf: $HOME is unset; pass an explicit path")
    return h^


def hf_hub_cache() raises -> String:
    """The `huggingface_hub` cache directory, respecting its env vars."""
    var v = getenv("HF_HUB_CACHE")
    if v != "":
        return v^
    var home = getenv("HF_HOME")
    if home != "":
        return home + "/hub"
    return home_dir() + "/.cache/huggingface/hub"


def noeira_cache() raises -> String:
    var v = getenv("NOEIRA_CACHE")
    if v != "":
        return v^
    return home_dir() + "/.cache/noeira"


def repo_slug(repo: String) -> String:
    """`Org/name` -> `Org__name`, the on-disk directory name."""
    return repo.replace("/", "__")


def hf_token() raises -> String:
    """`HF_TOKEN` from the environment, or the `hf` CLI's token file.

    ⚠ THE TOKEN NEVER TOUCHES A SHELL — same rule as `io/hf.mojo`'s reader.
    It becomes a header on a libcurl handle, so no quoting question arises and
    a token with an odd byte in it does not have to be refused.
    """
    var t = getenv("HF_TOKEN")
    if t != "":
        return t^
    var home = getenv("HOME")
    if home != "":
        var path = home + "/.cache/huggingface/token"
        try:
            var b = read_file_bytes(path)
            # The CLI writes the token with no trailing newline, but do not
            # rely on that — a stray \n in a header is a protocol error.
            var out = String("")
            for i in range(len(b)):
                var c = Int(b[i])
                if c == 0x0A or c == 0x0D or c == 0x20:
                    continue
                out += chr(c)
            if out != "":
                return out^
        except:
            pass
    raise Error(
        "hf_push: no token. Set HF_TOKEN (a `.env` in the project root is"
        " loaded by `load_dotenv`) or log in with `hf auth login`."
    )



def hf_client(var token: String = String("")) raises -> HttpClient:
    """A client carrying a Hub token when there is one.

    ⚠ THE TOKEN NO LONGER TOUCHES A SHELL. It used to be interpolated into a
    `curl` command line that `popen` handed to `/bin/sh`, which made
    `quote_arg` load-bearing on a SECRET — a token with a quote in it had to
    be rejected outright rather than sent. It is now a header on a libcurl
    handle and no quoting question arises.
    """
    var c = HttpClient(0, 30000)
    c.stall_guard(1024, 60)
    if token == "":
        # Optional: public repos need none, so a missing token is not an
        # error here — it becomes a 401 only on a private repo.
        try:
            token = hf_token()
        except:
            token = String("")
    if token != "":
        c.bearer(token^)
    return c^


def path_prefix(s: String, n: Int) -> String:
    """The first `n` BYTES of `s`. Mojo has no string slicing, and these are
    ASCII repo paths."""
    # ⚠ BYTES, not `chr` per byte — see `core/bytes.mojo`. HF responses carry
    # repo and file names that are not guaranteed ASCII.
    var b = s.as_bytes()
    var o = List[UInt8]()
    for i in range(n):
        o.append(b[i])
    return string_from_bytes(o)


def hf_tree(
    repo: String, kind: StaticString = HF_MODEL,
    revision: String = String("main"),
    var token: String = String(""),
) raises -> String:
    """The raw tree listing JSON for a repo."""
    var api = (
        "https://huggingface.co/api/" + String(kind) + "/" + repo + "/tree/"
        + revision + "?recursive=1"
    )
    var c = hf_client(token^)
    c.max_body(1 << 26)  # a big repo's recursive listing
    var r = c.get(api)
    if not r.ok():
        raise Error(
            "hf: the tree API answered " + String(r.status) + " for '" + repo
            + "': " + r.text()
        )
    return r.text()


def hf_file_size(
    repo: String,
    rel: String,
    kind: StaticString = HF_MODEL,
    revision: String = String("main"),
) raises -> Int:
    """The listed size of one file, or -1 if the repo does not have it."""
    var listing = hf_tree(repo, kind, revision)
    var lbytes = List[UInt8]()
    for i in range(listing.byte_length()):
        lbytes.append(listing.as_bytes()[i])
    var doc = parse_json(lbytes^)
    var arr = doc.root()
    if doc.kind_of(arr) != J_ARRAY:
        raise Error(
            "hf: the Hub tree API did not return a list for '" + repo
            + "' — is the repo name right, and is it public (or HF_TOKEN set)?"
        )
    for i in range(doc.size(arr)):
        var ent = doc.at(arr, i)
        if doc.string(doc.field(ent, String("type"))) != "file":
            continue
        if doc.string(doc.field(ent, String("path"))) != rel:
            continue
        var sn = doc.field(ent, String("size"))
        return doc.integer(sn) if sn >= 0 else -1
    return -1


def hf_download_file(
    repo: String,
    rel: String,
    kind: StaticString = HF_MODEL,
    var dest: String = String(""),
    revision: String = String("main"),
    verbose: Bool = True,
) raises -> String:
    """Download one file from a Hub repo; return its local path.

    Skips the download when a local file of the listed size is already there.
    """
    if dest == "":
        dest = (
            noeira_cache() + "/hub/" + repo_slug(repo) + "/" + revision + "/"
            + rel
        )
    var slash = dest.rfind("/")
    if slash > 0:
        makedirs(path_prefix(dest, slash), exist_ok=True)

    var size = hf_file_size(repo, rel, kind, revision)
    if size < 0:
        raise Error(
            "hf: '" + repo + "' @ " + revision + " has no file '" + rel + "'"
        )
    if exists(dest) and file_size(dest) == size:
        if verbose:
            print(
                "  " + repo + "/" + rel + " already cached ("
                + String(size) + " bytes) -> " + dest
            )
        return dest^

    var kind_seg = String("") if kind == HF_MODEL else String(kind) + "/"
    var url = (
        "https://huggingface.co/" + kind_seg + repo + "/resolve/" + revision
        + "/" + rel
    )
    if verbose:
        print(
            "  downloading " + repo + "/" + rel + "  (" + String(size)
            + " bytes)"
        )
    var c = hf_client()
    var r = c.download(
        url, dest + ".part", 0, rel if verbose else String("")
    )
    if not r.ok():
        raise Error(
            "hf: GET " + url + " -> " + String(r.status) + ": " + r.text()
        )
    # ⚠ A NON-2xx IS NOT THE ONLY WAY TO GET THE WRONG BYTES. A connection cut
    # mid-transfer ends a 200 response early, and nothing about the status
    # says so. The size the listing gave is what separates the two — which is
    # why this check survived the move off `curl`, where it guarded the same
    # gap in `curl -f`.
    var got = file_size(dest + ".part")
    if got != size:
        raise Error(
            "hf: '" + rel + "' downloaded " + String(got) + " bytes, the Hub"
            " listing says " + String(size) + " — the transfer was truncated"
        )
    rename_over(dest + ".part", String(dest))
    return dest^


def url_path_encode(s: String) -> String:
    """Percent-encode a repo-relative path for a `resolve/` URL.

    `/` is left alone (it is the path separator) and so is the unreserved
    set; everything else is escaped. The AMASS dump needs this: its clip
    names hold spaces, parentheses and commas (`A6- lift box t2_poses…`,
    `C11 -  run turn left (90)_poses…`), and an unescaped space ends the
    request line at the space — the server then answers 400 for a file that
    is there.
    """
    comptime HEX = "0123456789ABCDEF"
    var b = s.as_bytes()
    var out = String("")
    for i in range(s.byte_length()):
        var c = b[i]
        var safe = (
            (c >= 48 and c <= 57)          # 0-9
            or (c >= 65 and c <= 90)       # A-Z
            or (c >= 97 and c <= 122)      # a-z
            or c == 45 or c == 46 or c == 95 or c == 126  # - . _ ~
            or c == 47                     # /
        )
        if safe:
            out += String(s[byte=i:i + 1])
        else:
            out += "%"
            out += String(HEX[byte=Int(c >> 4):Int(c >> 4) + 1])
            out += String(HEX[byte=Int(c & 15):Int(c & 15) + 1])
    return out^


def hf_repo_blobs(
    repo: String, kind: StaticString = HF_MODEL,
    revision: String = String("main"),
    var token: String = String(""),
    max_body: Int = 1 << 27,
) raises -> String:
    """Every file in a repo WITH ITS SIZE, in one response.

    ⚠ THIS EXISTS BECAUSE `hf_tree` CANNOT LIST A BIG REPO. The tree API
    pages at 1000 entries and hands the cursor back in a `link` response
    header, which `HttpResponse` does not carry — and the AMASS dump has
    17 771 files, so a tree listing silently returns the first 1000 and a
    caller that trusts it imports 6 % of the data. `?blobs=true` on the
    repo-info endpoint answers all of them at once (4.5 MB for AMASS).
    """
    var api = (
        "https://huggingface.co/api/" + String(kind) + "/" + repo
        + "/revision/" + revision + "?blobs=true"
    )
    var c = hf_client(token^)
    c.max_body(max_body)
    var r = c.get(api)
    if not r.ok():
        raise Error(
            "hf: the blobs API answered " + String(r.status) + " for '" + repo
            + "': " + r.text()
        )
    return r.text()


def hf_download_sized(
    repo: String,
    rel: String,
    size: Int,
    kind: StaticString = HF_MODEL,
    var dest: String = String(""),
    revision: String = String("main"),
    verbose: Bool = False,
) raises -> String:
    """`hf_download_file` for a caller that ALREADY KNOWS the size.

    `hf_download_file` calls `hf_file_size`, which lists the whole repo tree
    — once per file. Over the AMASS dump that is 17 717 listings of a repo
    the lister cannot even page through. Here the size comes from the
    caller's index and is used for exactly what it was used for before:
    skipping a file already on disk at the right length.
    """
    if dest == "":
        dest = (
            noeira_cache() + "/hub/" + repo_slug(repo) + "/" + revision + "/"
            + rel
        )
    var slash = dest.rfind("/")
    if slash > 0:
        makedirs(path_prefix(dest, slash), exist_ok=True)
    if exists(dest) and file_size(dest) == size:
        return dest^

    var kind_seg = String("") if kind == HF_MODEL else String(kind) + "/"
    var url = (
        "https://huggingface.co/" + kind_seg + repo + "/resolve/" + revision
        + "/" + url_path_encode(rel)
    )
    var c = hf_client()
    var r = c.download(url, dest + ".part", 0, rel if verbose else String(""))
    if not r.ok():
        raise Error("hf: GET " + url + " -> " + String(r.status) + ": " + r.text())
    var got = file_size(dest + ".part")
    if got != size:
        remove_file(dest + ".part")
        raise Error(
            "hf: '" + rel + "' arrived as " + String(got) + " bytes, the index"
            " says " + String(size) + " — a short read is not a small file"
        )
    rename_over(dest + ".part", dest)
    return dest^

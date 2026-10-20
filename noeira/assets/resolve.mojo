# +--------------------------------------------------------------------------+ #
# | Turning a pack declaration into files on this box
# +--------------------------------------------------------------------------+ #
"""Fetch, verify, extract and materialise an asset pack.

    pull_pack(pack, into_dir)     # idempotent; no bytes move if already there

## The three stages, and why they are separate

    1. FETCH     <cache>/assets/<sha256>.archive     verified against sha256
    2. EXTRACT   <cache>/assets/<sha256>/            once, marked by .ok
    3. LINK      <dest>/                             hard link, else copy

⚠⚠ THE CACHE IS KEYED BY THE ARCHIVE'S SHA256, NOT BY THE PACK NAME. Two envs
declaring the same bytes share one download; a pack that is re-cut under the
same version gets a different key and cannot be silently served from a stale
cache. A name-keyed cache has the opposite property on both counts.

⚠ HARD LINK, FALLING BACK TO COPY, NEVER SYMLINK — the same rule as policy
promotion, for the same reasons: a symlink into a cache breaks under rsync,
breaks when the cache is cleared, and turns into a dead pointer that looks like
a missing file much later.

## ⚠⚠ The credential is chosen by the PROVIDER, and never by the URL

`resolve_url` is the whole point of this module. A private pack is not "a URL
with a token bolted on": for `noeira` the API key goes to OUR Worker, which
answers with a presigned URL, and the transfer itself carries no credential at
all. That is why the provider has to be declared rather than inferred — two
`https://` URLs can need different secrets, or none.
"""

from std.os import getenv, makedirs
from std.os.path import exists

from ..core.dotenv import load_dotenv
from ..data.remote import RemoteCatalog
from ..io.fetch import fetch_to_cache
from ..io.fileio import file_size
from ..io.hf import noeira_cache
from ..io.proc import quote_arg, run_capture
from .pack import PROVIDER_HF, PROVIDER_HTTPS, PROVIDER_NOEIRA, Pack


def cache_root() raises -> String:
    """`~/.cache/noeira/assets`, or under `NOEIRA_CACHE`."""
    return noeira_cache() + "/assets"


def _env_or_dotenv(name: String) -> String:
    """The variable from the environment, falling back to `.env`.

    ⚠ `.env` IS NOT THE ENVIRONMENT, and this tree has been bitten by assuming
    it is. `NOEIRA_CLOUD_API_KEY` lives in a dotenv file that nothing exports, so
    a resolver that only read `getenv` would work in a shell where someone had
    sourced it and fail everywhere else.
    """
    var v = getenv(name)
    if v != "":
        return v^
    try:
        var env = load_dotenv(String(".env"))
        if name in env:
            return env[name]
    except:
        pass
    return String("")


struct Resolved(Movable):
    """Where to GET the bytes, and what to send with the request."""

    var url: String
    var bearer: String
    """Empty means send nothing. ⚠ For `noeira` this is ALWAYS empty: the URL
    is already presigned, so the credential stays with our Worker."""

    def __init__(out self, url: String, bearer: String = String("")):
        self.url = url
        self.bearer = bearer

    def __init__(out self, *, deinit move: Self):
        self.url = move.url^
        self.bearer = move.bearer^


def resolve_url(ref pack: Pack) raises -> Resolved:
    """Where this pack's bytes are, and the credential for reaching them.

    ⚠ THE PROVIDER DECIDES, NOT THE SCHEME. See the module header.
    """
    # An explicit override wins for every provider: a fourth host does not
    # need a fourth provider, only a different variable name.
    var override = String("")
    if pack.token_env.byte_length() > 0:
        override = _env_or_dotenv(pack.token_env)
        if override.byte_length() == 0:
            raise Error(
                "asset pack '" + pack.ref() + "': token_env="
                + pack.token_env + " is set but that variable is empty in the"
                " environment and in .env"
            )

    if pack.provider == PROVIDER_HTTPS:
        return Resolved(String(pack.url), override)

    if pack.provider == PROVIDER_HF:
        var tok = override
        if tok.byte_length() == 0:
            # ⚠ A MISSING HF TOKEN IS NOT AN ERROR. Public packs are the point
            # (§9: an asset pack is third-party data that must work for anyone
            # who clones), so it becomes a 401 only on a private repo.
            tok = _env_or_dotenv(String("HF_TOKEN"))
        return Resolved(String(pack.url), tok)

    if pack.provider == PROVIDER_NOEIRA:
        # ⚠⚠ THE ONLY SHAPE THAT NEVER SENDS A SECRET TO THE STORAGE HOST. The
        # key authenticates a catalog call to our Worker; the Worker answers
        # with a presigned URL; the transfer that follows is anonymous.
        var cat = RemoteCatalog.from_env()
        var meta = cat.describe(pack.id)
        if meta.sha256.byte_length() == 64 and meta.sha256 != pack.sha256:
            # ⚠ REFUSED BEFORE THE TRANSFER. The catalog and the declaration
            # disagreeing means the pack was re-cut without its `assets.kv`
            # being updated, and downloading 19 MB to fail the check at the end
            # tells you the same thing far more slowly.
            raise Error(
                "asset pack '" + pack.ref() + "': the catalog holds sha256 "
                + meta.sha256 + " but assets.kv declares " + pack.sha256
                + ". The pack was re-cut without updating the declaration."
            )
        return Resolved(meta.download_url, String(""))

    raise Error(
        "asset pack '" + pack.ref() + "': unknown provider '" + pack.provider
        + "'"
    )


def archive_path(ref pack: Pack) raises -> String:
    return cache_root() + "/" + pack.sha256 + ".archive"


def extract_dir(ref pack: Pack) raises -> String:
    return cache_root() + "/" + pack.sha256


def pack_status(ref pack: Pack, into_dir: String) raises -> String:
    """`present`, `cached` (extracted but not linked), or `absent`."""
    var dest = into_dir + "/" + pack.dest
    if exists(dest):
        return String("present")
    if exists(extract_dir(pack) + "/.ok"):
        return String("cached")
    return String("absent")


def pull_pack(ref pack: Pack, into_dir: String, force: Bool = False) raises -> String:
    """Ensure the pack's files are at `<into_dir>/<dest>`. Returns the status.

    ⚠ IDEMPOTENT AND CHEAP WHEN SATISFIED. An `assets-pull` on a box that
    already has everything makes no request at all — which is what lets it be a
    preflight someone actually runs before a training job rather than a cost
    they learn to skip.
    """
    var dest = into_dir + "/" + pack.dest
    if exists(dest) and not force:
        return String("present")

    var root = cache_root()
    makedirs(root, exist_ok=True)
    var archive = archive_path(pack)
    var xdir = extract_dir(pack)

    # ── 1. fetch, verified ──────────────────────────────────────────
    if not exists(xdir + "/.ok") or force:
        var r = resolve_url(pack)
        _ = fetch_to_cache(
            r.url,
            archive,
            pack.sha256,
            pack.bytes,
            pack.ref(),
            bearer=r.bearer,
        )

        # ── 2. extract, once ────────────────────────────────────────
        #
        # ⚠ EXTRACTED INTO A TEMP DIRECTORY AND RENAMED. A crash halfway
        # through would otherwise leave a partial tree that the `.ok` marker
        # was never written for — but which a future `force=False` run would
        # still link from if the marker logic ever moved.
        var tmp = xdir + ".tmp"
        _ = run_capture("rm -rf " + quote_arg(tmp))
        makedirs(tmp, exist_ok=True)
        # `tar -xf` auto-detects the compression on both bsdtar (macOS) and
        # GNU tar, so the archive may be .tar.zst, .tar.gz or plain .tar
        # without this having to know which.
        var out = run_capture(
            "tar -xf " + quote_arg(archive) + " -C " + quote_arg(tmp)
            + " 2>&1 || echo __TAR_FAILED__",
            1 << 20,
        )
        if out.find("__TAR_FAILED__") >= 0:
            _ = run_capture("rm -rf " + quote_arg(tmp))
            raise Error(
                "asset pack '" + pack.ref() + "': could not extract "
                + archive + "\n" + out
            )
        _ = run_capture(
            "rm -rf " + quote_arg(xdir) + " && mv " + quote_arg(tmp) + " "
            + quote_arg(xdir)
        )
        var f = open(xdir + "/.ok", "w")
        f.write(pack.sha256)
        f.close()

    # ── 3. materialise ──────────────────────────────────────────────
    materialise(xdir, dest)
    return String("pulled")


def materialise(src_dir: String, dest: String) raises:
    """Put the extracted tree at `dest`, by hard link where possible.

    ⚠ NEVER A SYMLINK — the same rule as `project-promote`. A symlink into
    `~/.cache` breaks under rsync, breaks the moment the cache is cleared, and
    fails much later as what looks like a missing mesh.

    ⚠ THE `.ok` MARKER IS NOT COPIED. It is cache bookkeeping, and a stray
    dotfile inside an env's asset directory is the kind of thing that ends up
    in a glob somewhere.
    """
    var parent_cut = dest.rfind("/")
    if parent_cut > 0:
        makedirs(String(dest[byte=0:parent_cut]), exist_ok=True)
    _ = run_capture("rm -rf " + quote_arg(dest))
    makedirs(dest, exist_ok=True)
    # `cp -al` hard-links a whole tree; plain `cp -a` is the cross-filesystem
    # fallback. The `/.` copies the CONTENTS, not the directory itself.
    var out = run_capture(
        "cp -al " + quote_arg(src_dir) + "/. " + quote_arg(dest)
        + " 2>/dev/null || cp -a " + quote_arg(src_dir) + "/. "
        + quote_arg(dest) + " 2>&1 || echo __CP_FAILED__",
        1 << 20,
    )
    if out.find("__CP_FAILED__") >= 0:
        raise Error("could not materialise " + dest + " from " + src_dir)
    _ = run_capture("rm -f " + quote_arg(dest + "/.ok"))


def pack_bytes_on_disk(ref pack: Pack, into_dir: String) raises -> Int:
    """Archive bytes held in the cache for this pack, 0 if absent."""
    var a = archive_path(pack)
    return file_size(a) if exists(a) else 0

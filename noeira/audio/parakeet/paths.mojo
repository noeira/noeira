"""Where the published checkpoint lives on this machine."""

from std.os.path import exists

from noeira.io.hf import hf_hub_cache


comptime PK_REPO_DIR = "models--nvidia--parakeet-tdt-0.6b-v3"


def parakeet_snapshot_dir() raises -> String:
    """The Hub cache's snapshot of `nvidia/parakeet-tdt-0.6b-v3` at `main`
    (`model.safetensors`, `tokenizer.json`, the configs). Raises with the
    download command when it is not there."""
    var repo = hf_hub_cache() + "/" + PK_REPO_DIR
    var ref_path = repo + "/refs/main"
    if not exists(ref_path):
        raise Error(
            "parakeet: no checkpoint in the Hub cache (" + repo + "). Fetch it:"
            " hf download nvidia/parakeet-tdt-0.6b-v3 model.safetensors"
            " tokenizer.json config.json"
        )
    var f = open(ref_path, "r")
    var rev = String(f.read().strip())
    f.close()
    return repo + "/snapshots/" + rev

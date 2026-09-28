# +--------------------------------------------------------------------------+ #
# | Import a LeRobot v3 dataset into a TrajectoryStore — no Python
# +--------------------------------------------------------------------------+ #
"""Download (if needed) and convert a LeRobot v3.0 dataset for ACT training.

    pixi run mojo run -I . examples/so101/act_so101_import_dataset.mojo \\
        --repo DenisLabs/record-test_20260828_092736

    pixi run mojo run -I . examples/so101/act_so101_import_dataset.mojo \\
        --root /path/to/snapshot --out /tmp/store.h5 --height 480 --width 640

Replaces `tools/act/lerobot_v3_to_store.py`. The Python script needed
`huggingface_hub`, `pyarrow`, `imageio`, `Pillow`, `numpy` and `h5py`; this
needs `curl` and `ffmpeg` on PATH, both of which the pixi environment provides.

Output goes to `~/.cache/noeira/act_so101/<Org>__<name>_<H>x<W>.h5`, which is
the path `examples/so101/act_so101_train_gpu.mojo` expects in `ACT_STORE`.

Options
-------
--repo REPO        HuggingFace dataset repo id (downloaded if not cached)
--root DIR         a local dataset directory; skips resolution entirely
--project P --dataset D
                   a project's recording: `projects/P/datasets/D`, output to
                   `~/.cache/noeira/act_so101/P__D_<H>x<W>.h5`
--out PATH         output .h5 (default: the cache path above)
--height / --width resize target (default 240x320; the recording is 480x640)
--undistort DIR    bring each camera's fisheye frame to the SIM's pinhole
                   first (`vision/fisheye.mojo`), from DIR/camera_<name>.txt —
                   `examples/vision/calibrate_fisheye.mojo`; e.g.
                   projects/so101-tower/cameras. The output name gains `_undist`.
--revision REV     branch or commit (default `main`)
--force            rebuild even if the output already exists
--no-download      fail rather than fetch anything over the network

⚠ REBUILDING IS NOT FREE — the 50-episode recording is a 700 MB download and a
6.8 GB store, and every frame is H.264-decoded and resampled. Existing output
is left alone unless `--force` says otherwise.
"""

from std.os import makedirs
from std.os.path import exists
from std.sys import argv

from noeira.core.project import project_dataset_dir
from noeira.io.fileio import write_text_atomic
from noeira.io.json import JsonWriter
from noeira.data.lerobot import (
    LeRobotInfo,
    import_lerobot_v3, noeira_cache, repo_slug, resolve_dataset_root,
)


def _opt(args: List[String], name: String, fallback: String) raises -> String:
    for i in range(len(args) - 1):
        if args[i] == name:
            return String(args[i + 1])
    return fallback


def _flag(args: List[String], name: String) -> Bool:
    for i in range(len(args)):
        if args[i] == name:
            return True
    return False


def main() raises:
    var raw = argv()
    var args = List[String]()
    for i in range(1, len(raw)):
        args.append(String(raw[i]))

    var repo = _opt(args, String("--repo"), String(""))
    var root = _opt(args, String("--root"), String(""))
    var project = _opt(args, String("--project"), String(""))
    var dataset = _opt(args, String("--dataset"), String(""))
    var slug = String("")
    if project != "" or dataset != "":
        if project == "" or dataset == "" or repo != "" or root != "":
            raise Error("--project and --dataset go together, and replace --repo / --root")
        root = project_dataset_dir(project, dataset)
        slug = project + "__" + dataset
    var revision = _opt(args, String("--revision"), String("main"))
    # `source_commit` records what was ASKED FOR, and only when it was asked
    # for: "main" is a moving reference, not a commit, so writing it into the
    # manifest would claim provenance the store does not have.
    var pinned = _opt(args, String("--revision"), String(""))
    var height = atol(_opt(args, String("--height"), String("240")))
    var width = atol(_opt(args, String("--width"), String("320")))
    var force = _flag(args, String("--force"))
    var undistort = _opt(args, String("--undistort"), String(""))
    var download = not _flag(args, String("--no-download"))

    if repo == "" and root == "":
        raise Error(
            "need --repo <org/name> or --root <dir>; see the docstring at the"
            " top of this file"
        )

    var out = _opt(args, String("--out"), String(""))
    if out == "":
        if repo == "" and slug == "":
            raise Error("--out is required when only --root is given")
        if slug == "":
            slug = repo_slug(repo)
        var dir = noeira_cache() + "/act_so101"
        makedirs(dir, exist_ok=True)
        out = (
            dir + "/" + slug + "_" + String(height) + "x" + String(width)
            + ("_undist" if undistort != "" else "") + ".h5"
        )

    print("LeRobot v3 -> TrajectoryStore")
    print("  out: " + out)
    if exists(out) and not force:
        print("  already present — pass --force to rebuild")
        return

    if root == "":
        root = resolve_dataset_root(repo, revision, download, True)
    print("  root: " + root)
    print("")

    import_lerobot_v3(
        root,
        out,
        height,
        width,
        String("lerobot/") + repo if repo != "" else String(""),
        pinned,
        True,
        undistort,
    )

    # ⚠ THE SIDE FILE `<out>.json`: which camera fills which slot. The store
    # itself records an `images` column of N cameras and no names, and the
    # Python converter this replaced wrote this file; without it a deployment
    # (and `norm.json`) can only say "slot 0". Slot order is the importer's:
    # `LeRobotInfo.cameras`, alphabetical by feature key.
    var info = LeRobotInfo(root)
    var jw = JsonWriter()
    jw.begin_object()
    jw.member(String("source"), (String("hf:") + repo) if repo != "" else root)
    jw.key(String("cameras"))
    jw.begin_array()
    for c in info.cameras:
        jw.string(c)
    jw.end_array()
    jw.member(String("height"), height)
    jw.member(String("width"), width)
    jw.member(String("fps"), info.fps)
    jw.member(String("state_dim"), info.state_dim)
    jw.member(String("action_dim"), info.action_dim)
    jw.end_object()
    if out.endswith(".h5"):
        var side = String(out[byte=0 : out.byte_length() - 3]) + ".json"
        write_text_atomic(side, jw.done())
        print("  cameras: " + side)

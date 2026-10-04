# +--------------------------------------------------------------------------+ #
# | noeira I/O bindings
# +--------------------------------------------------------------------------+ #
"""I/O backends for noeira datasets.

- ``hdf5``    — a Mojo FFI over libhdf5; the `TrajectoryStore` file format.
- ``parquet`` — a native Parquet reader (thrift + snappy + RLE + PLAIN).
- ``video``   — sequential H.264 decode by piping `ffmpeg`.
- ``json``    — a small JSON reader, and a writer.
- ``safetensors`` — the safetensors format, read and written natively.
- ``fileio``  — chunked, atomic whole-file I/O (the ~2 GiB syscall cap).
- ``http``    — an HTTP client over libcurl (needs `pixi run build-http`).
- ``sha256``  — FIPS 180-4, streaming; the catalog's checksums.
- ``base64``  — RFC 4648 §4; the Hub's `preupload` and `commit` bodies.
- ``hf``      — HuggingFace Hub cache paths + file download, over ``http``.
- ``image``   — Pillow-compatible bilinear and NEAREST resize.
- ``wav``     — 16-bit PCM WAV read/write, mono/resample (`noeira/ai/speech`).
- ``png``     — a PNG decoder AND encoder (8-bit, non-interlaced),
                Pillow-exact both ways.
- ``tar``     — a ustar / pax / GNU reader, for dataset archives.
- ``proc``    — `popen`/`fread`/`pclose`, since the stdlib has no subprocess.
- ``fetch``   — resumable HTTP download to a local cache.
- ``serial``  — raw-mode tty over libc (the SO-101 Feetech bus).

⚠ MOST OF THESE EXIST TO GET PYTHON OUT OF THE DATA PATH. Importing a LeRobot
v3 dataset used to need `huggingface_hub`, `pyarrow`, `imageio`, `Pillow`,
`numpy` and `h5py` in a separate process; `noeira/data/lerobot.mojo` now does
it with these plus `curl` and `ffmpeg` on PATH, and produces a byte-identical
store (`tests/data/test_lerobot_import.mojo`).

`http` closes the last of it. Every network call — the metrics logger, the
dataset catalog, the Hub downloader, the MNIST and TinyShakespeare loaders —
went through Python `urllib` or a `curl` subprocess; all of them are now one
libcurl handle behind a fixed-arity C shim, with `sha256` and the `json`
writer replacing the `hashlib` and `json.dumps` those calls also needed.
⚠ That is what took CPython out of a TRAINING binary: `RemoteLogger.flush`
was the reason `pixi.toml` has to pin `libpython3.13`.

`png` + `tar` + the parquet reader's BYTE_ARRAY path finish the dataset
loaders. CIFAR-10 needed `huggingface_hub`, `pyarrow`, `PIL` and `numpy` — the
last three in a SUBPROCESS, because libarrow cannot safely load inside Mojo's
embedded interpreter — and LeWM-PushT needed `huggingface_hub` + `zstandard`
to stream a 13 GB `.zst`. All of those are now oracles for the gates rather
than steps in the path.

`png` also retired `PIL` from the places that had nothing to do with datasets:
the 3D renderer's texture loader (`render/png_loader.mojo`), Procgen's sprite
loader and both Craftax ones — 1,443 asset images, 248 of them palettes, all
gated byte-for-byte against Pillow — and the 16 Procgen render demos, which
used to build their output image one pixel at a time through the interpreter.
⚠ THE ONLY REMAINING `Python.import_module` IN THE ENGINE IS THE GYMNASIUM /
MuJoCo WRAPPER LAYER, where calling Python IS the point.

`safetensors` closes the other half: ImageNet ResNet18 weights used to need
`torch` + `torchvision` and a dump step, and now come straight off the Hub
(`nn/models/resnet18_torch.mojo`), bit-identical to what torchvision loads
(`tests/nn/test_safetensors_resnet18_torch.mojo`, 11,190,912 values).
"""

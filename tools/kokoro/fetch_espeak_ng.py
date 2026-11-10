#!/usr/bin/env python3
"""Install the espeak-ng library and data the Mojo binding loads at run time.

    python3 tools/kokoro/fetch_espeak_ng.py [--dest ~/.cache/noeira/espeak-ng]

Standard library only: any `python3` runs it, once per machine. Nothing here is
in the inference path — `noeira/audio/espeak.mojo` dlopens what this writes.

Where it comes from: the `espeakng_loader` 0.2.4 wheel on PyPI, the same one
the deployment's kokoro-onnx uses on the Mac and on the Orin. It bundles a
self-contained build of espeak-ng 1.52.0 (it links only the system C and C++
runtimes) and its `espeak-ng-data`. Using exactly that build is the point:
espeak-ng's dictionaries change between releases, and the Orin's system
espeak-ng is 1.49, which phonemizes differently.

The wheel is pinned by URL AND sha256 per platform, and nothing is written
unless the digest matches. Output:

    <dest>/libespeak-ng.dylib | libespeak-ng.so    the library
    <dest>/espeak-ng-data/                         its data
    <dest>/VERSION                                 "espeak-ng 1.52.0 (espeakng_loader 0.2.4)"

⚠ LICENCE. espeak-ng is GPL-3.0-or-later. This fetches it onto the machine as a
separate runtime library; nothing of it enters the repository (MIT).
"""

import argparse
import hashlib
import io
import os
import platform
import shutil
import sys
import urllib.request
import zipfile

WHEELS = {
    ("Darwin", "arm64"): (
        "https://files.pythonhosted.org/packages/a8/26/258c0cd43b9bc1043301c5f61767d6a6c3b679df82790c9cb43a3277b865/"
        "espeakng_loader-0.2.4-py3-none-macosx_11_0_arm64.whl",
        "d27cdca31112226e7299d8562e889d3e38a1e48055c9ee381b45d669072ee59f",
        "libespeak-ng.dylib",
    ),
    ("Linux", "aarch64"): (
        "https://files.pythonhosted.org/packages/d9/ad/1b768d8daffc2996e07bbcb6f534d8de3202cd75fce1f1c45eced1ce6465/"
        "espeakng_loader-0.2.4-py3-none-manylinux_2_28_aarch64.whl",
        "d1e798141b46a050cdb75fcf3c17db969bb2c40394f3f4a48910655d547508b9",
        "libespeak-ng.so",
    ),
    ("Linux", "x86_64"): (
        "https://files.pythonhosted.org/packages/de/1e/25ec5ab07528c0fbb215a61800a38eca05c8a99445515a02d7fa5debcb32/"
        "espeakng_loader-0.2.4-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "08721baf27d13d461f6be6eed9a65277e70d68234ff484fd8b9897b222cdcb6d",
        "libespeak-ng.so",
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.join(
        os.environ.get("NOEIRA_CACHE", os.path.expanduser("~/.cache/noeira")), "espeak-ng"))
    args = ap.parse_args()

    key = (platform.system(), platform.machine())
    if key not in WHEELS:
        sys.exit(f"no pinned espeak-ng wheel for {key}; known: {sorted(WHEELS)}")
    url, sha256, lib_name = WHEELS[key]

    print(f"fetching {url.rsplit('/', 1)[1]}")
    with urllib.request.urlopen(url, timeout=120) as r:
        blob = r.read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != sha256:
        sys.exit(f"sha256 mismatch: got {digest}, pinned {sha256} — nothing written")

    tmp = args.dest + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        n_data = 0
        for info in z.infolist():
            name = info.filename
            if name == f"espeakng_loader/{lib_name}":
                with open(os.path.join(tmp, lib_name), "wb") as f:
                    f.write(z.read(info))
            elif name.startswith("espeakng_loader/espeak-ng-data/") and not info.is_dir():
                rel = name[len("espeakng_loader/"):]
                path = os.path.join(tmp, rel)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(z.read(info))
                n_data += 1
    if not os.path.exists(os.path.join(tmp, lib_name)) or n_data == 0:
        sys.exit(f"the wheel did not hold {lib_name} and espeak-ng-data — nothing installed")
    with open(os.path.join(tmp, "VERSION"), "w") as f:
        f.write("espeak-ng 1.52.0 (espeakng_loader 0.2.4)\n")

    shutil.rmtree(args.dest, ignore_errors=True)
    os.rename(tmp, args.dest)
    print(f"{args.dest}: {lib_name} and {n_data} data files")


if __name__ == "__main__":
    main()

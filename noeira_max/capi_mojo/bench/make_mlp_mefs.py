"""The MLP shapes of ``benchmark_interop.mojo`` as MEFs, for ``bench_capi.mojo``:
the same graphs and weights as ``MLPInference``, compiled for the accelerator,
plus each shape's input and the output MAX computes from it (raw float32), so
the Mojo side can check what it reads back.

    pixi run -e default python noeira_max/capi_mojo/bench/make_mlp_mefs.py OUT_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

from noeira_max.mlp_inference import MLPInference

# name, input dim, hidden, output dim, batch: benchmark_interop.mojo's sweep.
SHAPES = (
    ("actor-b1", 17, "256,256", 6, 1),
    ("actor-b64", 17, "256,256", 6, 64),
    ("actor-b1024", 17, "256,256", 6, 1024),
    ("wide-b1", 256, "512,512", 64, 1),
    ("wide-b1024", 256, "512,512", 64, 1024),
)


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, in_dim, hidden, out_dim, batch in SHAPES:
        mlp = MLPInference(in_dim, hidden, out_dim, batch, device="gpu")
        if not mlp.on_gpu:
            raise SystemExit("no accelerator: the M1.2 MEFs are GPU models")
        mlp.compiled.export_mef(out / f"{name}.mef")
        x = mlp.make_host_input()
        x.tofile(out / f"{name}.in")
        mlp.infer(x).astype("float32").tofile(out / f"{name}.out")
        print(f"wrote {name}.mef ({mlp.info()})", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))

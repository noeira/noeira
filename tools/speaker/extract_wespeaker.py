"""WeSpeaker ResNet34 (VoxCeleb) -> the weights noeira's speaker port loads.

    python tools/speaker/extract_wespeaker.py --onnx voxceleb_resnet34.onnx --out resnet34.safetensors

The published ONNX file (https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34, CC BY 4.0, the
licence of VoxCeleb) has its batch norms folded into the convolutions: 36 `Conv` with a bias, a
statistics pooling, one linear layer and a mean vector subtracted from the embedding. This walks the
graph and writes each tensor under the name of its place in the network (docs/ROBOT_BRAIN_PLAN.md B4):

    stem.w [32,1,3,3]  stem.b
    s<stage>.b<block>.c1.w / .c1.b      3x3, the block's stride
    s<stage>.b<block>.c2.w / .c2.b      3x3, stride 1
    s<stage>.b<block>.sc.w / .sc.b      1x1 shortcut, only where the shape changes
    seg.w [256,5120]  seg.b  mean_vec [256]

A convolution is recognised by its SHAPE, not by its position: in a block that changes shape the
shortcut is the 1x1, the first 3x3 is the one whose input channels differ or whose stride is 2. Every
shape and stride is asserted against the ResNet34 layout (blocks 3, 4, 6, 3; channels 32, 64, 128, 256;
strides 1, 2, 2, 2), so a file with another layout stops here and not as a wrong embedding.

Needs `onnx` and `numpy` only. The safetensors file is written by hand (an 8-byte header length, the
JSON header, the tensors' bytes), float32, little-endian.
"""

import argparse
import json
import struct

import numpy as np
import onnx
from onnx import numpy_helper

BLOCKS = [3, 4, 6, 3]
CHANNELS = [32, 64, 128, 256]
STRIDES = [1, 2, 2, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g = onnx.load(a.onnx).graph
    init = {t.name: numpy_helper.to_array(t).astype(np.float32) for t in g.initializer}
    convs = []
    for n in g.node:
        if n.op_type == "Conv":
            at = {x.name: list(x.ints) for x in n.attribute if x.ints}
            convs.append((init[n.input[1]], init[n.input[2]], at["strides"], at["pads"]))
    assert len(convs) == 36, len(convs)
    out = {}
    w, b, s, p = convs[0]
    assert w.shape == (32, 1, 3, 3) and s == [1, 1] and p == [1, 1, 1, 1], (w.shape, s, p)
    out["stem.w"], out["stem.b"] = w, b
    k = 1
    cin = 32
    for st in range(4):
        for bl in range(BLOCKS[st]):
            cout = CHANNELS[st]
            stride = STRIDES[st] if bl == 0 else 1
            changes = stride != 1 or cin != cout
            group = convs[k:k + (3 if changes else 2)]
            k += len(group)
            name = "s%d.b%d." % (st + 1, bl)
            threes = [c for c in group if c[0].shape[2:] == (3, 3)]
            ones = [c for c in group if c[0].shape[2:] == (1, 1)]
            assert len(threes) == 2 and len(ones) == (1 if changes else 0), (name, [c[0].shape for c in group])
            c1, c2 = threes
            assert c1[0].shape == (cout, cin, 3, 3) and c1[2] == [stride, stride] and c1[3] == [1, 1, 1, 1], (name, c1[0].shape, c1[2])
            assert c2[0].shape == (cout, cout, 3, 3) and c2[2] == [1, 1] and c2[3] == [1, 1, 1, 1], (name, c2[0].shape, c2[2])
            out[name + "c1.w"], out[name + "c1.b"] = c1[0], c1[1]
            out[name + "c2.w"], out[name + "c2.b"] = c2[0], c2[1]
            if changes:
                sc = ones[0]
                assert sc[0].shape == (cout, cin, 1, 1) and sc[2] == [stride, stride] and sc[3] == [0, 0, 0, 0], (name, sc[0].shape, sc[2], sc[3])
                out[name + "sc.w"], out[name + "sc.b"] = sc[0], sc[1]
            cin = cout
    assert k == 36, k
    gemm = [n for n in g.node if n.op_type == "Gemm"]
    assert len(gemm) == 1
    out["seg.w"], out["seg.b"] = init[gemm[0].input[1]], init[gemm[0].input[2]]
    assert out["seg.w"].shape == (256, 5120), out["seg.w"].shape
    out["mean_vec"] = init["mean_vec"]
    # the pooling's constants, read from the graph rather than assumed
    consts = {}
    for n in g.node:
        if n.op_type == "Constant":
            consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    sqrt_in = [n for n in g.node if n.op_type == "Sqrt"][0].input[0]
    add = [n for n in g.node if n.output[0] == sqrt_in][0]
    eps = float([consts[i] for i in add.input if i in consts][0])
    header = {"__metadata__": {"source": "Wespeaker/wespeaker-voxceleb-resnet34 voxceleb_resnet34.onnx",
                               "licence": "CC BY 4.0", "pool_eps": repr(eps), "embedding": "256", "mels": "80"}}
    blob = b""
    for name in sorted(out):
        arr = np.ascontiguousarray(out[name], dtype="<f4")
        header[name] = {"dtype": "F32", "shape": list(arr.shape), "data_offsets": [len(blob), len(blob) + arr.nbytes]}
        blob += arr.tobytes()
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(a.out, "wb") as f:
        f.write(struct.pack("<Q", len(hj)) + hj + blob)
    print(len(out), "tensors,", sum(v.size for v in out.values()), "values, pooling eps", eps, "->", a.out)


if __name__ == "__main__":
    main()

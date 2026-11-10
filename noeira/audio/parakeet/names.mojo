# +--------------------------------------------------------------------------+ #
# | Parakeet-TDT — our parameter names <-> the checkpoint's
# +--------------------------------------------------------------------------+ #
"""`parakeet_name_map()` — one entry per parameter `ParakeetTDT` walks.

The checkpoint is transformers' `ParakeetForTDT` state dict
(`nvidia/parakeet-tdt-0.6b-v3`, `model.safetensors`, 723 tensors). Of those, 699
are fp32 weights and each is named here exactly once; the other 24 are
`encoder.layers.N.conv.norm.num_batches_tracked` (int64 counters BatchNorm keeps
for training), which nothing reads. `ParakeetTDT.load` checks both counts.

Three kinds of entry:

- `TN_TRANSPOSE`: a `nn.Linear` weight, or a kernel-1 convolution weight
  (`[out, in, 1]`, `[out, in, 1, 1]`) — theirs `[out, in]`, ours `[in, out]`.
- `TN_PLAIN`: norms, biases, the embedding, and the one-channel convolutions
  (`[C, 1, K]` / `[C, 1, 3, 3]`), whose layout is already ours (`[C, TAPS]`).
- `TN_ZEROS`: the bias our `Linear` / `PkConv` has and their module was built
  without (`attention_bias: false`, `convolution_bias: false`).

⚠ The subsampling's `layers.N` indices count the ReLUs too: the five modules
with weights are 0 (conv), 2 and 5 (depthwise), 3 and 6 (pointwise).
"""

from noeira.nn.core.torch_names import TorchNameMap, TN_TRANSPOSE, TN_ZEROS
from .config import (
    PK_D, PK_FF, PK_HEADS, PK_HEAD_DIM, PK_LAYERS, PK_CONV_K, PK_SUB_CH,
    PK_SUB_FLAT, PK_DEC, PK_DEC_LAYERS, PK_VOCAB, PK_JOINT,
)


def _linear(
    mut m: TorchNameMap, ours: String, theirs: String, out_f: Int, in_f: Int,
    has_bias: Bool,
) raises:
    """A `nn.Linear`: weight transposed, bias plain or (absent) zeros."""
    m.add_linear(ours + ".weight", theirs + ".weight", out_f, in_f)
    var b: List[Int] = [out_f]
    if has_bias:
        m.add(ours + ".bias", theirs + ".bias", b)
    else:
        m.add(ours + ".bias", String(""), b, TN_ZEROS)


def _pointwise(
    mut m: TorchNameMap, ours: String, theirs: String, out_f: Int, in_f: Int,
    trailing: Int, has_bias: Bool,
) raises:
    """A kernel-1 convolution, `[out, in]` plus `trailing` unit dims."""
    var s: List[Int] = [out_f, in_f]
    for _ in range(trailing):
        s.append(1)
    m.add(ours + ".weight", theirs + ".weight", s, TN_TRANSPOSE)
    var b: List[Int] = [out_f]
    if has_bias:
        m.add(ours + ".bias", theirs + ".bias", b)
    else:
        m.add(ours + ".bias", String(""), b, TN_ZEROS)


def _norm(mut m: TorchNameMap, ours: String, theirs: String, dim: Int) raises:
    """`LayerNorm` / `BatchNorm1d` affine: their weight / bias, our gamma / beta."""
    var s: List[Int] = [dim]
    m.add(ours + ".gamma", theirs + ".weight", s)
    m.add(ours + ".beta", theirs + ".bias", s)


def parakeet_name_map() raises -> TorchNameMap:
    var m = TorchNameMap()

    # ── subsampling ──────────────────────────────────────────────────────
    var sub = String("encoder.subsampling.")
    var k33: List[Int] = [PK_SUB_CH, 1, 3, 3]
    var cb: List[Int] = [PK_SUB_CH]
    m.add(String("sub.conv0.weight"), sub + "layers.0.weight", k33)
    m.add(String("sub.conv0.bias"), sub + "layers.0.bias", cb)
    m.add(String("sub.dw1.weight"), sub + "layers.2.weight", k33)
    m.add(String("sub.dw1.bias"), sub + "layers.2.bias", cb)
    _pointwise(m, String("sub.pw1"), sub + "layers.3", PK_SUB_CH, PK_SUB_CH, 2, True)
    m.add(String("sub.dw2.weight"), sub + "layers.5.weight", k33)
    m.add(String("sub.dw2.bias"), sub + "layers.5.bias", cb)
    _pointwise(m, String("sub.pw2"), sub + "layers.6", PK_SUB_CH, PK_SUB_CH, 2, True)
    _linear(m, String("sub.linear"), sub + "linear", PK_D, PK_SUB_FLAT, True)

    # ── the 24 conformer blocks ──────────────────────────────────────────
    var uv: List[Int] = [PK_HEADS, PK_HEAD_DIM]
    var dw: List[Int] = [PK_D, 1, PK_CONV_K]
    var d1: List[Int] = [PK_D]
    for i in range(PK_LAYERS):
        var o = String("layers." + String(i) + ".")
        var t = String("encoder.layers." + String(i) + ".")
        _norm(m, o + "norm_ff1", t + "norm_feed_forward1", PK_D)
        _linear(m, o + "ff1_a", t + "feed_forward1.linear1", PK_FF, PK_D, False)
        _linear(m, o + "ff1_b", t + "feed_forward1.linear2", PK_D, PK_FF, False)
        _norm(m, o + "norm_att", t + "norm_self_att", PK_D)
        _linear(m, o + "q", t + "self_attn.q_proj", PK_D, PK_D, False)
        _linear(m, o + "k", t + "self_attn.k_proj", PK_D, PK_D, False)
        _linear(m, o + "v", t + "self_attn.v_proj", PK_D, PK_D, False)
        _linear(m, o + "o", t + "self_attn.o_proj", PK_D, PK_D, False)
        _linear(m, o + "rel_k", t + "self_attn.relative_k_proj", PK_D, PK_D, False)
        m.add(o + "bias_u", t + "self_attn.bias_u", uv)
        m.add(o + "bias_v", t + "self_attn.bias_v", uv)
        _norm(m, o + "norm_conv", t + "norm_conv", PK_D)
        _pointwise(m, o + "pw1", t + "conv.pointwise_conv1", 2 * PK_D, PK_D, 1, False)
        m.add(o + "dw.weight", t + "conv.depthwise_conv.weight", dw)
        m.add(o + "dw.bias", String(""), d1, TN_ZEROS)
        _norm(m, o + "bn", t + "conv.norm", PK_D)
        m.add(o + "bn.running_mean", t + "conv.norm.running_mean", d1)
        m.add(o + "bn.running_var", t + "conv.norm.running_var", d1)
        _pointwise(m, o + "pw2", t + "conv.pointwise_conv2", PK_D, PK_D, 1, False)
        _norm(m, o + "norm_ff2", t + "norm_feed_forward2", PK_D)
        _linear(m, o + "ff2_a", t + "feed_forward2.linear1", PK_FF, PK_D, False)
        _linear(m, o + "ff2_b", t + "feed_forward2.linear2", PK_D, PK_FF, False)
        _norm(m, o + "norm_out", t + "norm_out", PK_D)

    # ── encoder projection, prediction network, joint ────────────────────
    _linear(m, String("enc_proj"), String("encoder_projector"), PK_DEC, PK_D, True)
    var emb: List[Int] = [PK_VOCAB, PK_DEC]
    m.add(String("embedding.weight"), String("decoder.embedding.weight"), emb)
    var gb: List[Int] = [4 * PK_DEC]
    for l in range(PK_DEC_LAYERS):
        var o = String("lstm." + String(l) + ".")
        var sfx = String("_l" + String(l))
        m.add_linear(o + "ih.weight", String("decoder.lstm.weight_ih") + sfx, 4 * PK_DEC, PK_DEC)
        m.add(o + "ih.bias", String("decoder.lstm.bias_ih") + sfx, gb)
        m.add_linear(o + "hh.weight", String("decoder.lstm.weight_hh") + sfx, 4 * PK_DEC, PK_DEC)
        m.add(o + "hh.bias", String("decoder.lstm.bias_hh") + sfx, gb)
    _linear(m, String("dec_proj"), String("decoder.decoder_projector"), PK_DEC, PK_DEC, True)
    _linear(m, String("joint"), String("joint.head"), PK_JOINT, PK_DEC, True)
    return m^

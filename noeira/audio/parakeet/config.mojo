# +--------------------------------------------------------------------------+ #
# | Parakeet-TDT-0.6B-v3 — the numbers
# +--------------------------------------------------------------------------+ #
"""`nvidia/parakeet-tdt-0.6b-v3`'s `config.json`, as compile-time constants.

    encoder   FastConformer: 128 mels -> conv subsampling x8 (256 channels) ->
              24 blocks, hidden 1024, 8 heads x 128, feed-forward 4096,
              depthwise conv kernel 9, relative-position attention
    decoder   prediction network: embedding 8193 x 640, 2-layer LSTM 640
    joint     relu(encoder_proj + decoder) -> 8193 tokens + 5 durations

One frame of the encoder is 80 ms of audio (10 ms mel hop x 8).
"""

comptime PK_MELS: Int = 128
comptime PK_D: Int = 1024
comptime PK_HEADS: Int = 8
comptime PK_HEAD_DIM: Int = 128
comptime PK_FF: Int = 4096
comptime PK_LAYERS: Int = 24
comptime PK_CONV_K: Int = 9
comptime PK_SUB_CH: Int = 256
comptime PK_SUB_FREQ: Int = 16
"""Mel bins left after three stride-2 convolutions: 128 / 8."""
comptime PK_SUB_FLAT: Int = PK_SUB_CH * PK_SUB_FREQ

comptime PK_DEC: Int = 640
comptime PK_DEC_LAYERS: Int = 2
comptime PK_VOCAB: Int = 8193
comptime PK_BLANK: Int = 8192
comptime PK_DURATIONS: Int = 5
"""The durations are `[0, 1, 2, 3, 4]` frames, so a duration's index is its value."""
comptime PK_JOINT: Int = PK_VOCAB + PK_DURATIONS
comptime PK_MAX_SYMBOLS: Int = 10

comptime PK_BN_EPS: Float64 = 1e-5

comptime PK_FILE_TENSORS: Int = 723
"""Tensors in `model.safetensors`: 699 fp32 weights and 24 int64
`num_batches_tracked` counters, which inference does not read."""
comptime PK_FILE_WEIGHTS: Int = 699


def pk_sub_len(n: Int) -> Int:
    """Length after one subsampling convolution (kernel 3, stride 2, padding 1)."""
    return (n - 1) // 2 + 1


def pk_encoder_len(mel_frames: Int) -> Int:
    """Encoder frames for `mel_frames` mel frames: three stride-2 convolutions."""
    return pk_sub_len(pk_sub_len(pk_sub_len(mel_frames)))

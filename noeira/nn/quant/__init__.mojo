"""Weight quantization: frozen int4 linears.

`q4_0.mojo` is llama.cpp's Q4_0 format, the one MAX's int4 GEMM consumes:
round-to-nearest on its grid (`quantize_rtn_q4_0`), pack / unpack between
blocks and (codes, fp16 scale bits), and dequantization on the host and the
device. `qlinear.mojo`'s `QLinear` is a frozen Q4_0 linear: a forward and a
grad_input, no weight gradient, MAX's int4 GEMM on NVIDIA and a dequantized
fp32 GEMM elsewhere.
"""

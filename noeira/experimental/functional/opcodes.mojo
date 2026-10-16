"""Op tags of the functional tape, shared by its CPU (`tape.mojo`) and GPU
(`gpu.mojo`) paths, and the binary ops' one scalar definition."""

from noeira.nn.constants import DT


# ── op tags ──────────────────────────────────────────────────────────────
comptime OP_LEAF = 0
comptime OP_ADD = 1
comptime OP_SUB = 2
comptime OP_MUL = 3
comptime OP_DIV = 4
comptime OP_MATMUL = 5
comptime OP_SCALE = 6  # x * c
comptime OP_SHIFT = 7  # x + c
comptime OP_DIVC = 8  # x / c
comptime OP_RSUBC = 9  # c - x
comptime OP_EW = 10  # elementwise activation, `sub` = EW_* tag
comptime OP_SUM = 11  # all elements -> [1, 1]
comptime OP_MEAN = 12  # all elements -> [1, 1]
comptime OP_TRANSPOSE = 13
comptime OP_SUM_ROWS = 14  # [R, C] -> [1, C]
comptime OP_LINEAR = 16  # x @ w + b (+ act): a = x, b = w, x3 = bias, `sub` = EW_* or EW_NONE
comptime OP_MODULE = 15  # a comptime nn Module, `sub` = its entry (module_op.mojo)

# ── elementwise activations: the comptime nn's `ElementOp`s ─────────────
comptime EW_RELU = 0
comptime EW_TANH = 1
comptime EW_SIGMOID = 2
comptime EW_GELU = 3
comptime EW_GELU_TANH = 4
comptime EW_MISH = 5
comptime EW_SWISH = 6
comptime EW_ELU = 7
comptime EW_NONE = -1  # OP_LINEAR without an activation


@always_inline
def bin_op[OP: Int](x: Scalar[DT], y: Scalar[DT]) -> Scalar[DT]:
    comptime if OP == OP_ADD:
        return x + y
    elif OP == OP_SUB:
        return x - y
    elif OP == OP_MUL:
        return x * y
    else:
        return x / y

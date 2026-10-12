"""Mojo `DType` <-> the C API's `M_Dtype` (`include/max/c/types.h`)."""

comptime M_BOOL: Int32 = 1
comptime M_INT8: Int32 = (3 << 1) | (1 << 7) | 1
comptime M_UINT8: Int32 = (3 << 1) | (1 << 7)
comptime M_INT32: Int32 = (5 << 1) | (1 << 7) | 1
comptime M_UINT32: Int32 = (5 << 1) | (1 << 7)
comptime M_INT64: Int32 = (6 << 1) | (1 << 7) | 1
comptime M_UINT64: Int32 = (6 << 1) | (1 << 7)
comptime M_FLOAT16: Int32 = 15 | (1 << 6)
comptime M_BFLOAT16: Int32 = 16 | (1 << 6)
comptime M_FLOAT32: Int32 = 17 | (1 << 6)
comptime M_FLOAT64: Int32 = 18 | (1 << 6)


def m_dtype(dtype: DType) raises -> Int32:
    if dtype == DType.float32:
        return M_FLOAT32
    if dtype == DType.float64:
        return M_FLOAT64
    if dtype == DType.float16:
        return M_FLOAT16
    if dtype == DType.bfloat16:
        return M_BFLOAT16
    if dtype == DType.int64:
        return M_INT64
    if dtype == DType.uint64:
        return M_UINT64
    if dtype == DType.int32:
        return M_INT32
    if dtype == DType.uint32:
        return M_UINT32
    if dtype == DType.int8:
        return M_INT8
    if dtype == DType.uint8:
        return M_UINT8
    if dtype == DType.bool:
        return M_BOOL
    raise Error("maxrt: no M_Dtype for " + String(dtype))


def from_m_dtype(code: Int32) raises -> DType:
    if code == M_FLOAT32:
        return DType.float32
    if code == M_FLOAT64:
        return DType.float64
    if code == M_FLOAT16:
        return DType.float16
    if code == M_BFLOAT16:
        return DType.bfloat16
    if code == M_INT64:
        return DType.int64
    if code == M_UINT64:
        return DType.uint64
    if code == M_INT32:
        return DType.int32
    if code == M_UINT32:
        return DType.uint32
    if code == M_INT8:
        return DType.int8
    if code == M_UINT8:
        return DType.uint8
    if code == M_BOOL:
        return DType.bool
    raise Error("maxrt: unknown M_Dtype " + String(code))

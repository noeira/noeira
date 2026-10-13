"""Sequential Q4_0 calibration of SmolVLA's text tower: GPTQ (or RTN) layer
by layer, and the blocks file the int4 policy loads.

    calibrate_tower_q4(prefill, tower, cache, xs, method)  -> blocks, stats
    save_q4_blocks(path, names, blocks, ...)               -> the file
    SmolVLAPolicy.vlm_q4_blocks = path; load               -> the int4 VLM

## The sequence (the reference GPTQ pipeline's)

For layer i = 0 .. L-1, with layers 0 .. i-1 already quantized:

    1. taps on layer i (`hessian_start`), every calibration input through
       the WHOLE prefill (layers after i compute and are ignored)
    2. per projection: its weight [OUT, IN], its Hessian, RTN and GPTQ codes
       and both objectives tr((W − Ŵ) H (W − Ŵ)ᵀ) on that Hessian; the
       chosen method's blocks are kept
    3. the dequantized weights are written back into layer i's fp32
       `Linear`s, so layer i+1 is calibrated on the QUANTIZED layer i's
       output, as the reference does

The tower stays fp32 throughout: after the last layer it holds deq(Ŵ)
everywhere, which on the CPU and Metal is the int4 tower's function exactly
(`QLinear` dequantizes there) and on NVIDIA its twin up to MAX's bf16
activations. Re-running the whole prefill per layer costs L prefills per
input instead of one; calibration runs once, and it needs no per-layer
entry into the fused loop.

The calibration inputs `xs` are the prefill's inputs ([B, P·W] prefix
embeddings, one entry per call), built by the caller from TRAINING
observations (`SmolVLAPolicy.build_prefix_from_segment` leaves them in
`prefix_buf`). Every prefix row is a real token in this port (`N_LANG` is
the instruction's exact length, no padding), so every row enters H.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.torch_names import TorchNameMap
from noeira.nn.quant.q4_0 import (
    Q4_GROUP,
    Q4_BLOCK,
    quantize_rtn_q4_0,
    pack_q4_0,
    f16_bits,
    dequantize_q4_0,
)
from noeira.nn.quant.gptq import gptq_q4_0, q4_0_objective
from noeira.nn.quant.hessian import HessianAccumulator
from noeira.io.safetensors import SafeTensorsWriter

from .fused import SmolVLAPrefill
from .kv_cache import SmolVLAKVCache
from .layer_weights import (
    DecoderLayerWeights,
    LORA_Q,
    LORA_DOWN,
    q4_proj_name,
    q4_hess_slot,
    q4_blocks_key,
)
from .text import SmolVLMTextLayers

comptime Q4_RTN = 0
comptime Q4_GPTQ = 1


struct Q4ProjStat(Copyable, Movable):
    """One projection's calibration: both objectives on its Hessian."""

    var layer: Int
    var which: Int
    var rows: Int
    var dead: Int
    var obj_rtn: Float64
    var obj_gptq: Float64

    def __init__(out self, layer: Int, which: Int):
        self.layer = layer
        self.which = which
        self.rows = 0
        self.dead = 0
        self.obj_rtn = 0
        self.obj_gptq = 0


def _blocks(ref q: List[UInt8], ref d: List[Float32], n: Int, k: Int) raises -> List[UInt8]:
    var bits = List[UInt16](capacity=len(d))
    for i in range(len(d)):
        bits.append(f16_bits(d[i]))
    return pack_q4_0(q, bits, n, k)


def quantize_layer_q4[
    W: Int, FF: Int, QW: Int, KVW: Int, KV_IN: Int, LORA_R: Int, Q4: Bool, //,
    target: StaticString,
](
    mut layer: DecoderLayerWeights[W, FF, QW, KVW, KV_IN, LORA_R, Q4],
    index: Int,
    var hs: List[HessianAccumulator],
    method: Int,
    mut blocks: List[List[UInt8]],
    mut stats: List[Q4ProjStat],
    ctx: Optional[DeviceContext] = None,
    damp_frac: Float64 = 0.01,
) raises:
    """Steps 2 and 3 of the module docstring for one layer: the seven
    projections' blocks appended to `blocks` (LORA_Q .. LORA_DOWN), their
    objectives to `stats`, and deq(Ŵ) written into the layer."""
    for which in range(LORA_Q, LORA_DOWN + 1):
        var dims = layer.proj_dims(which)
        var k = dims[0]
        var n = dims[1]
        var w = layer.base_weight(which)
        var slot = q4_hess_slot(which)
        if hs[slot].n != k:
            raise Error("quantize_layer_q4: the Hessian of projection "
                        + String(which) + " is " + String(hs[slot].n)
                        + " wide, its input " + String(k))
        var st = Q4ProjStat(index, which)
        st.rows = hs[slot].rows
        var qr = List[UInt8]()
        var dr = List[Float32]()
        quantize_rtn_q4_0(w, n, k, qr, dr)
        st.obj_rtn = q4_0_objective(w, qr, dr, n, k, hs[slot].h)
        var qg = List[UInt8]()
        var dg = List[Float32]()
        var gs = gptq_q4_0(w, n, k, hs[slot].h.copy(), qg, dg, damp_frac)
        st.dead = gs.dead
        st.obj_gptq = q4_0_objective(w, qg, dg, n, k, hs[slot].h)
        var b = _blocks(qg, dg, n, k) if method == Q4_GPTQ else _blocks(qr, dr, n, k)
        layer.set_base_weight[target](which, dequantize_q4_0(b, n, k), ctx)
        blocks.append(b^)
        stats.append(st^)


def calibrate_tower_q4[
    P: Int,
    SUFFIX: Int,
    B: Int,
    LAYERS: Int,
    W: Int,
    FF: Int,
    HEADS: Int,
    N_KV: Int,
    HD: Int,
    THETA: Float64,
    RECORD: Bool,
    LORA_R: Int,
    Q4: Bool,
    //,
    target: StaticString,
](
    mut prefill: SmolVLAPrefill[
        P, SUFFIX, B, LAYERS, W, FF, HEADS, N_KV, HD, THETA, RECORD, LORA_R, Q4
    ],
    mut tower: SmolVLMTextLayers[LAYERS, W, FF, N_KV * HD, LORA_R, Q4],
    mut cache: SmolVLAKVCache[LAYERS, P, SUFFIX, N_KV, HD, B],
    ref xs: List[List[Float32]],
    method: Int,
    mut blocks: List[List[UInt8]],
    ctx: Optional[DeviceContext] = None,
    verbose: Bool = False,
    damp_frac: Float64 = 0.01,
) raises -> List[Q4ProjStat]:
    """The module docstring's sequence over the whole tower. `xs` are the
    prefill inputs (B·P·W floats each); `blocks` gets L·7 entries in layer,
    then LORA_Q .. LORA_DOWN order. The tower ends holding deq(Ŵ)."""
    comptime assert LORA_R == 0 and not Q4, (
        "calibrate_tower_q4: the fp32 tower without adapters"
    )
    comptime XN = B * P * W
    var x = Tensor.alloc(XN)
    var out = Tensor.alloc(XN)
    comptime if target == "gpu":
        x = Tensor.alloc_gpu(ctx.value(), XN)
        x.ensure(XN)
        out = Tensor.alloc_gpu(ctx.value(), XN)
    var stats = List[Q4ProjStat]()
    blocks = List[List[UInt8]]()
    for i in range(LAYERS):
        tower.layers[i].hessian_start()
        for j in range(len(xs)):
            if len(xs[j]) != XN:
                raise Error("calibrate_tower_q4: input " + String(j) + " has "
                            + String(len(xs[j])) + " values, the prefill reads "
                            + String(XN))
            for e in range(XN):
                x.data[e] = Scalar[DT](xs[j][e])
            comptime if target == "gpu":
                x.upload_resident(ctx.value())
            cache.reset()
            prefill.run[target](tower, cache, x, out, ctx)
        var hs = tower.layers[i].hessian_take()
        var first = len(stats)
        quantize_layer_q4[target](
            tower.layers[i], i, hs^, method, blocks, stats, ctx, damp_frac
        )
        if verbose:
            var line = String("  layer ") + String(i) + ": rows " + String(stats[first].rows)
            for s in range(first, len(stats)):
                line += (" | " + q4_proj_name(stats[s].which) + " GPTQ/RTN "
                         + String(stats[s].obj_gptq / stats[s].obj_rtn))
            print(line)
    return stats^


def save_q4_blocks(
    path: String,
    ref names: TorchNameMap,
    ref blocks: List[List[UInt8]],
    layers: Int,
    dims: List[Tuple[Int, Int]],
    method: Int,
    calibration: String,
) raises:
    """The blocks file `SmolVLAPolicy.vlm_q4_blocks` / `load_q4_blocks`
    read: one U8 [OUT, IN / 32 · 18] tensor per linear at `q4_blocks_key`,
    plus metadata naming the method and the calibration. `dims` is the
    seven projections' (IN, OUT), LORA_Q .. LORA_DOWN."""
    if len(blocks) != layers * 7 or len(dims) != 7:
        raise Error("save_q4_blocks: expected " + String(layers * 7)
                    + " linears and 7 shapes")
    var wr = SafeTensorsWriter()
    wr.add_metadata(String("q4_method"), String("gptq") if method == Q4_GPTQ else String("rtn"))
    wr.add_metadata(String("q4_format"), String("Q4_0 blocks [out, in/32*18], noeira nn/quant/q4_0.mojo"))
    wr.add_metadata(String("q4_calibration"), calibration)
    for i in range(layers):
        for which in range(7):
            var k = dims[which][0]
            var n = dims[which][1]
            var key = q4_blocks_key(
                names, String("layers.") + String(i) + "." + q4_proj_name(which)
            )
            var shape: List[Int] = [n, (k // Q4_GROUP) * Q4_BLOCK]
            ref b = blocks[i * 7 + which]
            wr.add_u8(key, shape, b, len(b))
    wr.save(path)

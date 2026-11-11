# +--------------------------------------------------------------------------+ #
# | SAM 3 — the DETR decoder (200 queries + presence) and the scoring
# +--------------------------------------------------------------------------+ #
"""`Sam3Decoder`: `Sam3DetrDecoder` (`detector_model.detr_decoder.*`), the box
refinement and `Sam3DotProductScoring` (`detector_model.dot_product_scoring.*`)
— what `Sam3Model.forward` makes of the DETR encoder's output for a TEXT
prompt (no geometry prompt: its encoder does not run).

    var dec = Sam3Decoder(ctx, sam3_checkpoint())
    var out = dec.forward(fused, pos72, text, text_len)
    # out.logits [200], out.boxes [200 x 4] xyxy in [0, 1], out.presence, out.queries (device [200, 256])

## A layer (post-norm; LayerNorm eps 1e-5)

    q = presence token + 200 queries;  qpos = ref_point_head(sine(boxes)), 0 for the presence row
    q = LN(q + self_attn(q + qpos, q + qpos, q))
    q = LN(q + text_attn(q + qpos, text, text))                        # padding masked
    q = LN(q + vision_attn(q + qpos, enc + pos, enc, bias = RPB))     # 8 heads x 32
    q = LN(q + fc2(relu(fc1(q))))

then `boxes = sigmoid(box_head(output_norm(q[1:])) + inverse_sigmoid(boxes))`,
`presence = clamp(presence_head(presence_norm(q[0])), +-10)`.

**sine(boxes)**: per box (cx, cy, w, h) the unnormalised sine encoding of each
coordinate (128 features, `2 pi x / 1e4^(2 (j // 2) / 128)`, even sin, odd cos),
concatenated in the order **y, x, w, h**.

**RPB** (box relative-position bias, log mode): for each query and each grid
column c, `(c / 72 - x0, c / 72 - x1)`, each `sign(8 d) log2(|8 d| + 1) / 3`,
through `box_rpb_embed_x` (2 -> 256 -> 8 heads); rows the same with y0, y1 and
`box_rpb_embed_y`; bias[h, q, r 72 + c] = y-part[q, r, h] + x-part[q, c, h];
the presence row 0.

## The scoring

`text' = LN(text + text_mlp(text))`, mean over the real tokens, `text_proj`;
`logit = clamp(query_proj(output_norm(q[1:])) . text' / 16, +-12)`.

The presence head (its last layer has ONE output: `max_matmul` is wrong at
N = 1 off NVIDIA) and the pooled dot products run on the host in fp32, as torch
does them. On NVIDIA (`DEV_BOXES`) the boxes stay on the device: the sine
encoding, the RPB deltas and the refinement are kernels, every layer's boxes
land in one buffer and the presence rows in another, both read back ONCE after
the last layer (the host round trip per layer — two synchronisations — was
most of the decoder's time). Off NVIDIA the box arithmetic runs on the host.
"""

from std.math import exp, log, log2, sin, cos, pow, sqrt
from std.sys import default_accelerator
from max.gpu import thread_idx, block_idx
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from .ops import (
    Sam3Weights, TPB, Ptr, blocks, linear, layer_norm, mha, add_kernel, add_into_kernel, relu_kernel,
)


comptime DEC_PREFIX = "detector_model.detr_decoder."
comptime SCORE_PREFIX = "detector_model.dot_product_scoring."
comptime Q: Int = 200
comptime QP: Int = Q + 1
comptime DM: Int = 256
comptime NH: Int = 8
comptime HD: Int = 32
comptime FF: Int = 2048
comptime NL: Int = 6
comptime G: Int = 72
comptime NV: Int = G * G
comptime T_LEN: Int = 32
comptime EPS: Float32 = 1e-5
comptime NPF: Int = 128
"""Sine features per box coordinate."""
comptime DEV_BOXES: Bool = default_accelerator().is_nvidia_gpu()


@fieldwise_init
struct Sam3Detections(Movable):
    var logits: List[Float32]
    """[200] class logits (sigmoid = the query's score)."""
    var boxes: List[Float32]
    """[200 x 4] xyxy, normalised to [0, 1]."""
    var presence: Float32
    """The presence logit (sigmoid = whether the concept is in the picture)."""
    var queries: DeviceBuffer[DT]
    """[200, 256] the last layer's queries after `output_layer_norm` (the mask decoder's input)."""
    var layer_queries: List[DeviceBuffer[DT]]
    """Each layer's, when asked (`keep_taps`)."""
    var layer_boxes: List[List[Float32]]
    """The reference boxes each layer READ (cxcywh)."""
    var layer_presence: List[Float32]


def rpb_kernel(dy: Ptr, dx: Ptr, bias: Ptr):
    """`bias[h, 1 + q, r G + c] = dy[(q G + r) NH + h] + dx[(q G + c) NH + h]`, row 0
    zero. Grid `(ceil(NV / TPB), QP, NH)`."""
    var n = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var qq = Int(block_idx.y)
    var h = Int(block_idx.z)
    if n >= NV:
        return
    var out = (h * QP + qq) * NV + n
    if qq == 0:
        bias[unsafe_offset=out] = Scalar[DT](0)
        return
    var q = qq - 1
    var r = n // G
    var c = n - r * G
    bias[unsafe_offset=out] = dy[unsafe_offset=(q * G + r) * NH + h] + dx[unsafe_offset=(q * G + c) * NH + h]


def sine_boxes_kernel(boxes: Ptr, dim_t: Ptr, dst: Ptr, xy_n: Int64):
    """`_sine_boxes` on the device: `dst[q, slot 128 + j]`, slots y, x, w, h
    (x, y with `xy_n`). The same fp32 product and quotient as the host's. Grid
    `(ceil(512 / TPB), Q)`."""
    var e = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var q = Int(block_idx.y)
    if e >= 4 * NPF:
        return
    var slot = e // NPF
    var j = e - slot * NPF
    var src = slot
    if xy_n == 0 and slot < 2:
        src = 1 - slot
    var v = boxes[unsafe_offset=q * 4 + src] * Scalar[DT](6.283185307179586) / dim_t[unsafe_offset=j]
    dst[unsafe_offset=(q * 4 + slot) * NPF + j] = sin(v) if j % 2 == 0 else cos(v)


def rpb_inputs_kernel(boxes: Ptr, dy: Ptr, dx: Ptr):
    """`_rpb_inputs` on the device: `dy`, `dx` [Q G, 2]. Grid `(ceil(G / TPB), Q)`."""
    var j = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var q = Int(block_idx.y)
    if j >= G:
        return
    var cx = boxes[unsafe_offset=q * 4]
    var cy = boxes[unsafe_offset=q * 4 + 1]
    var w = boxes[unsafe_offset=q * 4 + 2]
    var h = boxes[unsafe_offset=q * 4 + 3]
    var x0 = cx - Scalar[DT](0.5) * w
    var y0 = cy - Scalar[DT](0.5) * h
    var x1 = cx + Scalar[DT](0.5) * w
    var y1 = cy + Scalar[DT](0.5) * h
    var cj = Scalar[DT](j) / Scalar[DT](G)
    var b = (q * G + j) * 2
    dy[unsafe_offset=b] = _logd(cj - y0)
    dy[unsafe_offset=b + 1] = _logd(cj - y1)
    dx[unsafe_offset=b] = _logd(cj - x0)
    dx[unsafe_offset=b + 1] = _logd(cj - x1)


def box_update_kernel(delta: Ptr, src: Ptr, dst: Ptr):
    """`dst = sigmoid(delta + inverse_sigmoid(src))` over [Q, 4]. Grid `ceil(4 Q / TPB)`."""
    var i = Int(block_idx.x) * TPB + Int(thread_idx.x)
    if i < Q * 4:
        dst[unsafe_offset=i] = _sigmoid(delta[unsafe_offset=i] + _inverse_sigmoid(src[unsafe_offset=i]))


def pad_row0_kernel(src: Ptr, dst: Ptr):
    """`dst[0] = 0`, `dst[1 + q] = src[q]` ([Q, DM] -> [QP, DM])."""
    var i = Int(block_idx.x) * TPB + Int(thread_idx.x)
    if i < QP * DM:
        dst[unsafe_offset=i] = Scalar[DT](0) if i < DM else src[unsafe_offset=i - DM]


struct Sam3Decoder(Movable):
    var wts: Sam3Weights
    var sco: Sam3Weights
    var ref0: List[Float32]
    var dim_t: DeviceBuffer[DT]
    """The sine encoding's 128 divisors (the host's values)."""
    var host: Dict[String, List[Float32]]
    """Small tensors read on the host (the presence head, text_proj)."""
    var debug: Int
    """Falsification only: bit 1, no RPB; bit 2, the sine box encoding in x, y order."""

    def __init__(out self, var ctx: DeviceContext, path: String) raises:
        var c2 = ctx
        self.wts = Sam3Weights(ctx^, path, DEC_PREFIX)
        self.sco = Sam3Weights(c2^, path, SCORE_PREFIX)
        self.debug = 0
        self.host = Dict[String, List[Float32]]()
        self.ref0 = List[Float32]()
        self.dim_t = self.wts.ctx.enqueue_create_buffer[DT](1)  # set below (methods need every field)
        var rp = self._down(self.wts.w("reference_points.weight"), Q * 4)
        for v in rp:
            self.ref0.append(Float32(1.0) / (Float32(1.0) + exp(-v)))
        var dt = List[Float32](length=NPF, fill=0.0)
        for j in range(NPF):
            dt[j] = Float32(pow(10000.0, Float64(Float32(2 * (j // 2)) / Float32(NPF))))
        self.dim_t = self._up(dt)
        var names: List[String] = [
            "presence_layer_norm.weight", "presence_layer_norm.bias", "presence_head.layer1.weight",
            "presence_head.layer1.bias", "presence_head.layer2.weight", "presence_head.layer2.bias",
            "presence_head.layer3.weight", "presence_head.layer3.bias",
        ]
        for nm in names:
            var b = self.wts.w(nm)
            self.host[nm] = self._down(b, len(b))
        var snames: List[String] = ["text_proj.weight", "text_proj.bias"]
        for nm in snames:
            var b = self.sco.w(nm)
            self.host["s." + nm] = self._down(b, len(b))

    def _down(self, x: DeviceBuffer[DT], n: Int) raises -> List[Float32]:
        var h = self.wts.ctx.enqueue_create_host_buffer[DT](n)
        self.wts.ctx.enqueue_copy(h, x.create_sub_buffer[DT](0, n))
        self.wts.ctx.synchronize()
        var out = List[Float32](capacity=n)
        for i in range(n):
            out.append(h[i])
        return out^

    def _up(self, v: List[Float32]) raises -> DeviceBuffer[DT]:
        var h = self.wts.ctx.enqueue_create_host_buffer[DT](len(v))
        for i in range(len(v)):
            h[i] = v[i]
        var d = self.wts.ctx.enqueue_create_buffer[DT](len(v))
        self.wts.ctx.enqueue_copy(d, h)
        return d^

    def new(self, n: Int) raises -> DeviceBuffer[DT]:
        return self.wts.ctx.enqueue_create_buffer[DT](n)

    def _mlp(mut self, name: String, x: DeviceBuffer[DT], t: Int, k: Int, layers: Int) raises -> DeviceBuffer[DT]:
        """`Sam3DecoderMLP`: relu between layers, none after the last."""
        var h = linear(self.wts, name + ".layer1", x, t, k)
        var n1 = len(h) // t
        self.wts.ctx.enqueue_function[relu_kernel](h, Int64(t * n1), grid_dim=blocks(t * n1), block_dim=TPB)
        if layers == 2:
            return linear(self.wts, name + ".layer2", h, t, n1)
        var h2 = linear(self.wts, name + ".layer2", h, t, n1)
        self.wts.ctx.enqueue_function[relu_kernel](h2, Int64(t * n1), grid_dim=blocks(t * n1), block_dim=TPB)
        return linear(self.wts, name + ".layer3", h2, t, n1)

    def _attn(
        mut self, name: String, q_in: DeviceBuffer[DT], tq: Int, k_in: DeviceBuffer[DT], v_in: DeviceBuffer[DT], tk: Int,
        key_len: Int, bias: Optional[DeviceBuffer[DT]] = None,
    ) raises -> DeviceBuffer[DT]:
        var q = linear(self.wts, name + ".q_proj", q_in, tq, DM)
        var k = linear(self.wts, name + ".k_proj", k_in, tk, DM)
        var v = linear(self.wts, name + ".v_proj", v_in, tk, DM)
        var a = mha(self.wts.ctx, q, k, v, tq, tk, NH, HD, key_len, bias)
        return linear(self.wts, name + ".o_proj", a, tq, DM)

    def _post(mut self, name: String, x: DeviceBuffer[DT], delta: DeviceBuffer[DT], rows: Int) raises -> DeviceBuffer[DT]:
        """`LN(x + delta)`."""
        var s = self.new(rows * DM)
        self.wts.ctx.enqueue_function[add_into_kernel](s, x, delta, Int64(rows * DM), grid_dim=blocks(rows * DM), block_dim=TPB)
        var y = self.new(rows * DM)
        layer_norm(self.wts, name, s, y, rows, DM, EPS)
        return y^

    def forward(
        mut self, fused: DeviceBuffer[DT], pos: DeviceBuffer[DT], text: DeviceBuffer[DT], text_len: Int,
        keep_taps: Bool = False,
    ) raises -> Sam3Detections:
        ref ctx = self.wts.ctx
        var hidden = self.new(QP * DM)
        ctx.enqueue_copy(hidden.create_sub_buffer[DT](0, DM), self.wts.w("presence_token.weight"))
        ctx.enqueue_copy(hidden.create_sub_buffer[DT](DM, Q * DM), self.wts.w("query_embed.weight"))
        var kpos = self.new(NV * DM)
        ctx.enqueue_function[add_into_kernel](kpos, fused, pos, Int64(NV * DM), grid_dim=blocks(NV * DM), block_dim=TPB)
        var boxes = self.ref0.copy()
        # DEV_BOXES: slot li holds the boxes layer li reads, slot NL the final ones
        var btaps = self.new((NL + 1) * Q * 4)
        var pres_rows = self.new(NL * DM)
        comptime if DEV_BOXES:
            ctx.enqueue_copy(btaps.create_sub_buffer[DT](0, Q * 4), self._up(self.ref0))
        var layer_q = List[DeviceBuffer[DT]]()
        var layer_b = List[List[Float32]]()
        var layer_p = List[Float32]()
        var last_q = self.new(Q * DM)
        var presence = Float32(0)
        for li in range(NL):
            var p = String("layers.") + String(li) + "."
            var bd = btaps.create_sub_buffer[DT](li * Q * 4, Q * 4)
            var sine: DeviceBuffer[DT]
            comptime if DEV_BOXES:
                sine = self.new(Q * 4 * NPF)
                ctx.enqueue_function[sine_boxes_kernel](
                    bd, self.dim_t, sine, Int64(1 if (self.debug & 2) != 0 else 0),
                    grid_dim=(blocks(4 * NPF), Q), block_dim=TPB,
                )
            else:
                layer_b.append(boxes.copy())
                sine = self._up(_sine_boxes(boxes, (self.debug & 2) != 0))
            var qpos = self._mlp("ref_point_head", sine, Q, 2 * DM, 2)
            var qp = self.new(QP * DM)
            ctx.enqueue_function[pad_row0_kernel](qpos, qp, grid_dim=blocks(QP * DM), block_dim=TPB)
            var bias = Optional[DeviceBuffer[DT]](None)
            if (self.debug & 1) == 0:
                var dyi: DeviceBuffer[DT]
                var dxi: DeviceBuffer[DT]
                comptime if DEV_BOXES:
                    dyi = self.new(Q * G * 2)
                    dxi = self.new(Q * G * 2)
                    ctx.enqueue_function[rpb_inputs_kernel](bd, dyi, dxi, grid_dim=(blocks(G), Q), block_dim=TPB)
                else:
                    var dl = _rpb_inputs(boxes)
                    dyi = self._up(dl[0])
                    dxi = self._up(dl[1])
                var dy = self._mlp("box_rpb_embed_y", dyi, Q * G, 2, 2)
                var dx = self._mlp("box_rpb_embed_x", dxi, Q * G, 2, 2)
                var b = self.new(NH * QP * NV)
                ctx.enqueue_function[rpb_kernel](dy, dx, b, grid_dim=(blocks(NV), QP, NH), block_dim=TPB)
                bias = Optional(b^)
            # self-attention
            var qk = self.new(QP * DM)
            ctx.enqueue_function[add_into_kernel](qk, hidden, qp, Int64(QP * DM), grid_dim=blocks(QP * DM), block_dim=TPB)
            var o = self._attn(p + "self_attn", qk, QP, qk, hidden, QP, QP)
            hidden = self._post(p + "self_attn_layer_norm", hidden, o, QP)
            # text cross-attention
            ctx.enqueue_function[add_into_kernel](qk, hidden, qp, Int64(QP * DM), grid_dim=blocks(QP * DM), block_dim=TPB)
            o = self._attn(p + "text_cross_attn", qk, QP, text, text, T_LEN, text_len)
            hidden = self._post(p + "text_cross_attn_layer_norm", hidden, o, QP)
            # vision cross-attention, with the box bias
            ctx.enqueue_function[add_into_kernel](qk, hidden, qp, Int64(QP * DM), grid_dim=blocks(QP * DM), block_dim=TPB)
            o = self._attn(p + "vision_cross_attn", qk, QP, kpos, fused, NV, NV, bias)
            hidden = self._post(p + "vision_cross_attn_layer_norm", hidden, o, QP)
            # MLP
            var f1 = linear(self.wts, p + "mlp.fc1", hidden, QP, DM)
            ctx.enqueue_function[relu_kernel](f1, Int64(QP * FF), grid_dim=blocks(QP * FF), block_dim=TPB)
            var f2 = linear(self.wts, p + "mlp.fc2", f1, QP, FF)
            hidden = self._post(p + "mlp_layer_norm", hidden, f2, QP)
            # box refinement
            var qn = self.new(Q * DM)
            layer_norm(self.wts, "output_layer_norm", hidden.create_sub_buffer[DT](DM, Q * DM), qn, Q, DM, EPS)
            comptime if DEV_BOXES:
                var dd = self._mlp("box_head", qn, Q, DM, 3)
                ctx.enqueue_function[box_update_kernel](
                    dd, bd, btaps.create_sub_buffer[DT]((li + 1) * Q * 4, Q * 4), grid_dim=blocks(Q * 4), block_dim=TPB,
                )
                ctx.enqueue_copy(pres_rows.create_sub_buffer[DT](li * DM, DM), hidden.create_sub_buffer[DT](0, DM))
            else:
                var delta = self._down(self._mlp("box_head", qn, Q, DM, 3), Q * 4)
                var nb = List[Float32](capacity=Q * 4)
                for i in range(Q * 4):
                    nb.append(_sigmoid(delta[i] + _inverse_sigmoid(boxes[i])))
                boxes = nb^
                # presence
                presence = self._presence(self._down(hidden, DM))
                layer_p.append(presence)
            if keep_taps:
                layer_q.append(qn)
            last_q = qn
        comptime if DEV_BOXES:
            # the one read-back: every layer's boxes, then the presence rows
            var all_b = self._down(btaps, (NL + 1) * Q * 4)
            for li in range(NL + 1):
                var one = List[Float32](capacity=Q * 4)
                for i in range(Q * 4):
                    one.append(all_b[li * Q * 4 + i])
                if li < NL:
                    layer_b.append(one^)
                else:
                    boxes = one^
            var pr = self._down(pres_rows, NL * DM)
            for li in range(NL):
                var row = List[Float32](capacity=DM)
                for d in range(DM):
                    row.append(pr[li * DM + d])
                presence = self._presence(row)
                layer_p.append(presence)
        var logits = self._scores(last_q, text, text_len)
        var xyxy = List[Float32](capacity=Q * 4)
        for i in range(Q):
            var cx = boxes[i * 4]
            var cy = boxes[i * 4 + 1]
            var w = boxes[i * 4 + 2]
            var h = boxes[i * 4 + 3]
            xyxy.append(cx - Float32(0.5) * w)
            xyxy.append(cy - Float32(0.5) * h)
            xyxy.append(cx + Float32(0.5) * w)
            xyxy.append(cy + Float32(0.5) * h)
        return Sam3Detections(logits^, xyxy^, presence, last_q, layer_q^, layer_b^, layer_p^)

    def _presence(self, row: List[Float32]) raises -> Float32:
        """`clamp(presence_head(presence_layer_norm(row)), +-10)` on the host."""
        var x = _ln_host(row, self.host["presence_layer_norm.weight"], self.host["presence_layer_norm.bias"])
        var h1 = _lin_host(x, self.host["presence_head.layer1.weight"], self.host["presence_head.layer1.bias"], True)
        var h2 = _lin_host(h1, self.host["presence_head.layer2.weight"], self.host["presence_head.layer2.bias"], True)
        var o = _lin_host(h2, self.host["presence_head.layer3.weight"], self.host["presence_head.layer3.bias"], False)
        return max(Float32(-10.0), min(Float32(10.0), o[0]))

    def _scores(mut self, queries: DeviceBuffer[DT], text: DeviceBuffer[DT], text_len: Int) raises -> List[Float32]:
        """`Sam3DotProductScoring` on the last layer's queries."""
        ref ctx = self.sco.ctx
        var t1 = linear(self.sco, "text_mlp.layer1", text, T_LEN, DM)
        ctx.enqueue_function[relu_kernel](t1, Int64(T_LEN * FF), grid_dim=blocks(T_LEN * FF), block_dim=TPB)
        var t2 = linear(self.sco, "text_mlp.layer2", t1, T_LEN, FF)
        ctx.enqueue_function[add_kernel](t2, text, Int64(T_LEN * DM), grid_dim=blocks(T_LEN * DM), block_dim=TPB)
        var tn = self.new(T_LEN * DM)
        layer_norm(self.sco, "text_mlp_out_norm", t2, tn, T_LEN, DM, EPS)
        var th = self._down(tn, T_LEN * DM)
        var pooled = List[Float32](length=DM, fill=0.0)
        for t in range(text_len):
            for d in range(DM):
                pooled[d] += th[t * DM + d]
        for d in range(DM):
            pooled[d] = pooled[d] / Float32(text_len)
        var pt = _lin_host(pooled, self.host["s.text_proj.weight"], self.host["s.text_proj.bias"], False)
        var pq = self._down(linear(self.sco, "query_proj", queries, Q, DM), Q * DM)
        var out = List[Float32](capacity=Q)
        var scale = Float32(1.0) / Float32(16.0)
        for q in range(Q):
            var s = Float32(0)
            for d in range(DM):
                s += pq[q * DM + d] * pt[d]
            out.append(max(Float32(-12.0), min(Float32(12.0), s * scale)))
        return out^


def _sigmoid(x: Float32) -> Float32:
    return Float32(1.0) / (Float32(1.0) + exp(-x))


def _inverse_sigmoid(x: Float32) -> Float32:
    """`inverse_sigmoid(x, eps=1e-3)`: log(max(x, eps) / max(1 - x, eps)), x in [0, 1]."""
    var c = max(Float32(0.0), min(Float32(1.0), x))
    var x1 = max(c, Float32(1e-3))
    var x2 = max(Float32(1.0) - c, Float32(1e-3))
    return log(x1 / x2)


def _sine_boxes(boxes: List[Float32], xy_order: Bool) -> List[Float32]:
    """`encode_boxes`: [Q, 512], each box's y, x, w, h encodings (128 each)."""
    comptime NPF = 128
    var dim_t = List[Float32](length=NPF, fill=0.0)
    for j in range(NPF):
        dim_t[j] = Float32(pow(10000.0, Float64(Float32(2 * (j // 2)) / Float32(NPF))))
    var two_pi = Float32(6.283185307179586)
    var out = List[Float32](length=Q * 4 * NPF, fill=0.0)
    var order: List[Int] = [0, 1, 2, 3] if xy_order else [1, 0, 2, 3]   # y, x, w, h
    for q in range(Q):
        for slot in range(4):
            var e = boxes[q * 4 + order[slot]] * two_pi
            for j in range(NPF):
                var v = e / dim_t[j]
                out[(q * 4 + slot) * NPF + j] = Float32(sin(Float64(v))) if j % 2 == 0 else Float32(cos(Float64(v)))
    return out^


def _rpb_inputs(boxes: List[Float32]) -> Tuple[List[Float32], List[Float32]]:
    """The log-scaled deltas `[Q G, 2]` for rows (y0, y1) and columns (x0, x1)."""
    var dy = List[Float32](length=Q * G * 2, fill=0.0)
    var dx = List[Float32](length=Q * G * 2, fill=0.0)
    for q in range(Q):
        var cx = boxes[q * 4]
        var cy = boxes[q * 4 + 1]
        var w = boxes[q * 4 + 2]
        var h = boxes[q * 4 + 3]
        var x0 = cx - Float32(0.5) * w
        var y0 = cy - Float32(0.5) * h
        var x1 = cx + Float32(0.5) * w
        var y1 = cy + Float32(0.5) * h
        for j in range(G):
            var cj = Float32(j) / Float32(G)
            var b = (q * G + j) * 2
            dy[b] = _logd(cj - y0)
            dy[b + 1] = _logd(cj - y1)
            dx[b] = _logd(cj - x0)
            dx[b + 1] = _logd(cj - x1)
    return (dy^, dx^)


def _logd(d: Float32) -> Float32:
    """`sign(8 d) log2(|8 d| + 1) / log2(8)`."""
    var e = d * Float32(8.0)
    var s = Float32(1.0) if e > 0 else (Float32(-1.0) if e < 0 else Float32(0.0))
    return s * log2(abs(e) + Float32(1.0)) / Float32(3.0)


def _ln_host(x: List[Float32], w: List[Float32], b: List[Float32]) -> List[Float32]:
    var n = len(x)
    var m = Float32(0)
    for v in x:
        m += v
    m = m / Float32(n)
    var var_ = Float32(0)
    for v in x:
        var_ += (v - m) * (v - m)
    var inv = Float32(1.0) / sqrt(var_ / Float32(n) + EPS)
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append((x[i] - m) * inv * w[i] + b[i])
    return out^


def _lin_host(x: List[Float32], w: List[Float32], b: List[Float32], relu: Bool) -> List[Float32]:
    """`x W^T + b` (W [n, k] row-major), relu'd when asked."""
    var k = len(x)
    var n = len(b)
    var out = List[Float32](capacity=n)
    for o in range(n):
        var s = Float32(0)
        for i in range(k):
            s += x[i] * w[o * k + i]
        s += b[o]
        out.append(max(s, Float32(0)) if relu else s)
    return out^

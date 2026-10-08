"""Kev's backbone (Qwen3.5 hybrid: Gated DeltaNet + gated attention) as ONE MAX graph — prefill only, ragged rows.

    bb = Qwen35Backbone("~/.cache/noeira/local-ai/kev9b-gptq-q4g32")       # MLX-format GPTQ Q4_0 checkpoint
    hs = bb.hidden([[t0, t1, ...], [...]])                                 # final hidden states per row, fp32
    hp, hr = bb.hidden_shared(prefix, [suffix_1, ...])                     # the prefix computed ONCE

Why: Kev on the Jetson Orin spends ~43 % of a decision in MLX's naive 6-bit matmul and ~50 % in mlx-lm's per-token
DeltaNet loop (off Apple GPUs it has no fused kernel). This graph uses, from MAX 26.6:
  - the fused DeltaNet ops `gated_delta_conv1d_fwd` + `gated_delta_recurrence_fwd` (max.nn.state_space), called as
    MAX's own qwen3_5 layer calls them;
  - MAX's Q4_0 int4 tensor-core GEMM through OUR custom op `noeira_q4_0_matmul` (kernels/q4_0_matmul.mojo: tile
    picked by M; MAX's own `qmatmul_b4_g32` would hit 4096x4096 configs that do not compile in 26.6), fed by MAX's
    `GGUF_gpu_repack_q4_0`.
Everything else mirrors mlx-lm's `qwen3_5` op for op (the reference Kev runs): RMSNorm in fp32 -> bf16, partial
RoPE (64 of 256 dims, theta 1e7, half-split), gated attention (sigmoid gate), gated RMSNorm with a fp32 SiLU gate,
SwiGLU MLP, bf16 residuals. MAX's qwen3_5 ARCHITECTURE is not reused: it reads NVFP4/FP8 checkpoints only and
needs the serving stack (paged KV cache, state-cache pool) that a three-row prefill does not.

A Kev request's STATE — the prefix every question row starts with — is computed ONCE, by a PREFIX graph (one row
from zero state) that also returns each attention layer's K and V and leaves its final DeltaNet states in its pool
buffers. A ROWS graph then runs all question rows in one ragged call: each DeltaNet pool slot starts from the
prefix's state (one store per pool, before the pool's one in-place op), attention sees the prefix's K/V plus the
query's own row causally, and positions continue from the prefix.
⚠ ON THE ORIN, RUN WITH `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=0`. With MAX 26.6's default VMM memory manager
(a 15 GB reserved chunk, mapped as it grows), calls hit CUDA_ERROR_ILLEGAL_ADDRESS at random once the device cache
filled (3.75-4.25 GB in use): never in device-sync mode, never at small sizes, in this two-graph form and in a
single-graph one. With VMM off, 16 alternating calls of both paths ran clean. The step-3 graph (state recomputed per
row) never filled the cache.

The checkpoint is read straight from its safetensors header (NumPy memory map: no MLX, no torch needed — the Orin).
Q4 linears are MLX affine 4-bit g32 with bias = -8 * scale (gptq_kev.py); each is converted to llama.cpp Q4_0
bytes, refusing any whose biases are not exactly -8 * scale. 8-bit layers (in_proj_a/b) are dequantized to bf16;
the embedding stays 8-bit on the HOST and only the looked-up rows are dequantized.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.state_space import gated_delta_conv1d_fwd, gated_delta_recurrence_fwd

KERNELS = Path(__file__).resolve().parent / "kernels"
PFX = "language_model.model."
BF16, F32 = DType.bfloat16, DType.float32


# ── checkpoint I/O ───────────────────────────────────────────────────────────


class SafeTensors:
    """Lazy reader: a memory map + the header; tensors materialize (as fp32 for bf16/f16) on access."""

    def __init__(self, path: Path):
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            self.hdr = {k: v for k, v in json.loads(f.read(n)).items() if k != "__metadata__"}
        self.mm = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)

    def __contains__(self, k):
        return k in self.hdr

    def raw(self, k):
        v = self.hdr[k]
        a, b = v["data_offsets"]
        return self.mm[a:b], v["dtype"], v["shape"]

    def __getitem__(self, k):
        raw, dt, shape = self.raw(k)
        if dt == "BF16":
            arr = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dt == "F16":
            arr = raw.view(np.float16).astype(np.float32)
        elif dt == "F32":
            arr = raw.view(np.float32)
        elif dt == "U32":
            arr = raw.view(np.uint32)
        elif dt == "U8":
            arr = raw
        else:
            raise ValueError(f"{k}: dtype {dt}")
        return arr.reshape(shape)


def q4_0_bytes(wq: np.ndarray, scales: np.ndarray, biases: np.ndarray) -> np.ndarray:
    """MLX affine 4-bit g32 (bias = -8 * scale) -> llama.cpp Q4_0 [N, K/32 * 18]: fp16 d, then byte j = q[j] | q[j+16] << 4."""
    if not np.array_equal(biases, -8.0 * scales):
        raise ValueError("not a symmetric Q4_0 grid: biases != -8 * scales (quantize with gptq_kev.py)")
    N, K = wq.shape[0], wq.shape[1] * 8
    q = ((wq[..., None] >> np.arange(0, 32, 4, dtype=np.uint32)) & 0xF).astype(np.uint8).reshape(N, K // 32, 32)
    qs = q[..., :16] | (q[..., 16:] << 4)
    d = scales.astype(np.float16).view(np.uint8).reshape(N, K // 32, 2)
    return np.ascontiguousarray(np.concatenate([d, qs], axis=2).reshape(N, K // 32 * 18))


def deq8(wq: np.ndarray, scales: np.ndarray, biases: np.ndarray, gs: int) -> np.ndarray:
    """MLX affine 8-bit -> fp32 rows."""
    R, K = wq.shape[0], wq.shape[1] * 4
    q = ((wq[..., None] >> np.arange(0, 32, 8, dtype=np.uint32)) & 0xFF).astype(np.float32).reshape(R, K // gs, gs)
    return (q * scales[..., None] + biases[..., None]).reshape(R, K)


# ── the backbone ─────────────────────────────────────────────────────────────


class Qwen35Backbone:
    def __init__(self, ckpt: str | Path):
        ckpt = Path(ckpt).expanduser()
        cfg = json.loads((ckpt / "config.json").read_text())
        self.q = cfg["quantization"]
        t = cfg.get("text_config", cfg)
        self.D, self.L = t["hidden_size"], t["num_hidden_layers"]
        self.every = t["full_attention_interval"]
        self.nv, self.nk = t["linear_num_value_heads"], t["linear_num_key_heads"]
        self.kd, self.vd = t["linear_key_head_dim"], t["linear_value_head_dim"]
        self.conv_dim = 2 * self.nk * self.kd + self.nv * self.vd
        self.ck = t["linear_conv_kernel_dim"]
        self.H, self.Hkv, self.hd = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
        rp = t.get("rope_parameters") or {}
        self.rot = int(self.hd * rp.get("partial_rotary_factor", t.get("partial_rotary_factor", 0.25)))
        self.theta = float(rp.get("rope_theta", t.get("rope_theta") or 1e7))
        self.eps = t.get("rms_norm_eps", 1e-6)
        self.st = SafeTensors(ckpt / "model.safetensors")
        self.dev = Accelerator()
        self.dref = DeviceRef.GPU()
        self.session = InferenceSession(devices=[self.dev])
        self._repack_graphs = {}
        self.names, self.types, self.bufs = [], [], []
        self._load_weights()
        self._pool_cache = {}
        self.prefix_model = self.session.load(self._build_graph("prefix"))
        self.rows_model = self.session.load(self._build_graph("rows"))

    # weights -> device buffers (graph inputs, in a fixed order) ───────────────

    def _is_linear(self, i):
        return (i + 1) % self.every != 0

    def _add(self, name, arr: np.ndarray, dtype: DType = F32):
        arr = np.ascontiguousarray(arr.astype(np.float32) if dtype == F32 else arr)
        self.names.append(name)
        self.types.append(TensorType(dtype, list(arr.shape), device=self.dref))
        self.bufs.append(Buffer.from_numpy(arr).to(self.dev))

    def _repack(self, raw: np.ndarray) -> Buffer:
        shape = tuple(raw.shape)
        if shape not in self._repack_graphs:
            with Graph(f"repack_{shape[0]}x{shape[1]}", input_types=[TensorType(DType.uint8, list(shape), device=self.dref)]) as g:
                g.output(ops.custom("GGUF_gpu_repack_q4_0", self.dref, [g.inputs[0].tensor],
                                    out_types=[TensorType(DType.uint8, list(shape), device=self.dref)])[0].tensor)
            self._repack_graphs[shape] = self.session.load(g)
        return self._repack_graphs[shape].execute(Buffer.from_numpy(raw).to(self.dev))[0]

    def _linear(self, name):
        k = PFX + name
        spec = self.q.get(k)
        bits = spec["bits"] if isinstance(spec, dict) else self.q["bits"]
        gs = spec["group_size"] if isinstance(spec, dict) else self.q["group_size"]
        wq, s, b = self.st[k + ".weight"], self.st[k + ".scales"], self.st[k + ".biases"]
        if bits == 4:
            assert gs == 32, f"{name}: Q4_0 needs group 32"
            buf = self._repack(q4_0_bytes(wq, s, b))
            self.names.append(name)
            self.types.append(TensorType(DType.uint8, [buf.shape[0], buf.shape[1]], device=self.dref))
            self.bufs.append(buf)
            return ("q4", wq.shape[0])
        self._add(name, deq8(wq, s, b, gs))  # 8-bit: dequantized, bf16 in the graph
        return ("dense", wq.shape[0])

    def _load_weights(self):
        self.plan = []
        for i in range(self.L):
            lp = f"layers.{i}."
            layer = {"linear": self._is_linear(i)}
            self._add(lp + "input_layernorm", self.st[PFX + lp + "input_layernorm.weight"])
            if layer["linear"]:
                la = lp + "linear_attn."
                for n in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
                    layer[n] = self._linear(la + n)
                self._add(la + "conv1d", self.st[PFX + la + "conv1d.weight"].reshape(self.conv_dim, self.ck))
                self._add(la + "A_log", self.st[PFX + la + "A_log"])
                self._add(la + "dt_bias", self.st[PFX + la + "dt_bias"])
                self._add(la + "norm", self.st[PFX + la + "norm.weight"])
            else:
                sa = lp + "self_attn."
                for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    layer[n] = self._linear(sa + n)
                self._add(sa + "q_norm", self.st[PFX + sa + "q_norm.weight"])
                self._add(sa + "k_norm", self.st[PFX + sa + "k_norm.weight"])
            self._add(lp + "post_attention_layernorm", self.st[PFX + lp + "post_attention_layernorm.weight"])
            for n in ("gate_proj", "up_proj", "down_proj"):
                layer[n] = self._linear(lp + "mlp." + n)
            self.plan.append(layer)
        self._add("norm", self.st[PFX + "norm.weight"])
        e = PFX + "embed_tokens."
        self.emb = (self.st[e + "weight"], self.st[e + "scales"], self.st[e + "biases"])
        ebits = self.q.get("embed_bits", 0)
        assert ebits == 8, "the host embedding lookup reads an 8-bit table"
        self.emb_gs = self.q["group_size"]

    # graph ────────────────────────────────────────────────────────────────────

    def _build_graph(self, kind: str) -> Graph:
        """`prefix`: ONE row from zero state — its hidden states, each attention layer's K and V, and its final
        DeltaNet states left in the pool buffers. `rows`: B rows, each continuing from those states (copied into
        every slot first) and attending to the prefix's K/V plus its own row, causally."""
        T, P, B, B1 = "T", "P", "B", "B1"
        d = self.dref
        nlin = sum(1 for p in self.plan if p["linear"])
        natt = len(self.plan) - nlin
        n = 1 if kind == "prefix" else B
        L = T if kind == "rows" else P
        dyn = [TensorType(F32, [L, self.D], device=d), TensorType(DType.int32, [L], device=d)]
        if kind == "rows":
            dyn.append(TensorType(DType.int32, [T], device=d))  # row id
        dyn += [TensorType(DType.uint32, [2 if kind == "prefix" else B1], device=d),
                TensorType(DType.uint32, [1 if kind == "prefix" else B], device=d)]
        conv_shape, rec_shape = [self.conv_dim, self.ck - 1], [self.nv, self.kd, self.vd]
        pools, extra = [], []
        for _ in range(nlin):
            pools += [BufferType(F32, [n] + conv_shape, device=d), BufferType(F32, [n] + rec_shape, device=d)]
        if kind == "rows":
            for _ in range(nlin):  # the prefix's final states (its pool buffers, read as tensors)
                extra += [TensorType(F32, [1] + conv_shape, device=d), TensorType(F32, [1] + rec_shape, device=d)]
            for _ in range(natt):  # the prefix's K and V per attention layer
                extra += [TensorType(BF16, [P, self.Hkv, self.hd], device=d)] * 2
        with Graph(f"kev_qwen35_{kind}", input_types=dyn + pools + extra + self.types,
                   custom_extensions=[KERNELS]) as g:
            vals = list(g.inputs)
            nd = len(dyn)
            if kind == "rows":
                h0, pos, rid, offs, slots = (v.tensor for v in vals[:nd])
            else:
                h0, pos, offs, slots = (v.tensor for v in vals[:nd])
                rid = None
            pool_vals = [v.buffer for v in vals[nd:nd + 2 * nlin]]
            ex = [v.tensor for v in vals[nd + 2 * nlin:nd + 2 * nlin + len(extra)]]
            W = dict(zip(self.names, (v.tensor for v in vals[nd + 2 * nlin + len(extra):])))
            self._W = W
            # every slot starts from zero (prefix) or from the prefix's final state (rows); ONE store per pool,
            # before the pool's one in-place op
            for k, pool in enumerate(pool_vals):
                src = ex[k] if kind == "rows" else ops.constant(0.0, F32, device=d)
                ops.buffer_store(pool, ops.broadcast_to(src, pool.shape))
            kv = ex[2 * nlin:]
            x = ops.cast(h0, BF16)
            cos, sin = self._rope_tables(pos)
            mask = self._mask(pos, rid, kv[0].shape[0] if kind == "rows" else None)
            out_kv = []
            li = ai = 0
            for i, layer in enumerate(self.plan):
                lp = f"layers.{i}."
                xn = self._rms(x, W[lp + "input_layernorm"])
                if layer["linear"]:
                    r = self._deltanet(xn, lp + "linear_attn.", layer, pool_vals[2 * li], pool_vals[2 * li + 1],
                                       slots, offs)
                    li += 1
                else:
                    pkv = (kv[2 * ai], kv[2 * ai + 1]) if kind == "rows" else None
                    r, k_, v_ = self._attention(xn, lp + "self_attn.", layer, cos, sin, mask, pkv)
                    out_kv += [k_, v_]
                    ai += 1
                h = x + r
                hn = self._rms(h, W[lp + "post_attention_layernorm"])
                g_ = self._mm(hn, lp + "mlp.gate_proj", layer["gate_proj"])
                u_ = self._mm(hn, lp + "mlp.up_proj", layer["up_proj"])
                x = h + self._mm(ops.silu(g_) * u_, lp + "mlp.down_proj", layer["down_proj"])
            hid = ops.cast(self._rms(x, W["norm"]), F32)
            g.output(*([hid] + out_kv if kind == "prefix" else [hid]))
        return g

    def _mm(self, x, name, kind):
        mode, n = kind
        if mode == "q4":
            return ops.custom("noeira_q4_0_matmul", self.dref, [x, self._W[name]],
                              out_types=[TensorType(BF16, [x.shape[0], n], device=self.dref)])[0].tensor
        return ops.matmul(x, ops.transpose(ops.cast(self._W[name], BF16), 0, 1))

    def _rms(self, x, w):
        xf = ops.cast(x, F32)
        y = xf * ops.rsqrt(ops.mean(xf * xf, axis=-1) + self.eps) * w
        return ops.cast(y, BF16)

    def _rope_tables(self, pos):
        half = self.rot // 2
        inv = (1.0 / (self.theta ** (np.arange(0, self.rot, 2, dtype=np.float64) / self.rot))).astype(np.float32)
        ang = ops.unsqueeze(ops.cast(pos, F32), 1) * ops.constant(inv.reshape(1, half), F32, device=self.dref)
        return ops.unsqueeze(ops.cos(ang), 1), ops.unsqueeze(ops.sin(ang), 1)  # [T, 1, half]

    def _rope(self, x, cos, sin):  # x [T, h, hd] bf16; non-traditional: pairs (i, i + rot/2) in the first `rot` dims
        half = self.rot // 2
        xf = ops.cast(x, F32)
        x1, x2, rest = xf[:, :, :half], xf[:, :, half:self.rot], xf[:, :, self.rot:]
        out = ops.concat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], axis=-1)
        return ops.cast(out, BF16)

    def _mask(self, pos, rid, P):
        causal = ops.greater_equal(ops.unsqueeze(pos, 1), ops.unsqueeze(pos, 0))
        if rid is None:
            return causal  # [P, P]: one row
        own = ops.logical_and(ops.equal(ops.unsqueeze(rid, 1), ops.unsqueeze(rid, 0)), causal)
        T = pos.shape[0]
        # [T, P + T]: every prefix key, then the query's own row causally
        allp = ops.broadcast_to(ops.constant(True, DType.bool, device=self.dref), [T, P])
        return ops.concat([allp, own], axis=1)

    def _attention(self, xn, sp, layer, cos, sin, mask, prefix_kv):
        """The block's output, and THIS call's K and V (post-norm, post-RoPE) for a prefix graph to hand on."""
        T = xn.shape[0]
        qg = ops.reshape(self._mm(xn, sp + "q_proj", layer["q_proj"]), [T, self.H, 2 * self.hd])
        q, gate = qg[:, :, :self.hd], qg[:, :, self.hd:]
        gate = ops.reshape(gate, [T, self.H * self.hd])
        k = ops.reshape(self._mm(xn, sp + "k_proj", layer["k_proj"]), [T, self.Hkv, self.hd])
        v = ops.reshape(self._mm(xn, sp + "v_proj", layer["v_proj"]), [T, self.Hkv, self.hd])
        q = self._rope(self._rms(q, self._W[sp + "q_norm"]), cos, sin)
        k = self._rope(self._rms(k, self._W[sp + "k_norm"]), cos, sin)
        ka, va = (k, v) if prefix_kv is None else (ops.concat([prefix_kv[0], k], 0), ops.concat([prefix_kv[1], v], 0))
        rep = self.H // self.Hkv

        def heads(t):  # [S, Hkv, hd] -> [H, S, hd], query head h reads kv head h // rep (repeat_interleave: CPU only)
            S = t.shape[0]
            t = ops.broadcast_to(ops.unsqueeze(t, 2), [S, self.Hkv, rep, self.hd])
            return ops.transpose(ops.reshape(t, [S, self.H, self.hd]), 0, 1)

        kh, vh = heads(ka), heads(va)
        qh = ops.transpose(q, 0, 1)
        s = ops.matmul(ops.cast(qh, F32), ops.transpose(ops.cast(kh, F32), 1, 2)) * (self.hd ** -0.5)
        s = ops.where(ops.unsqueeze(mask, 0), s, ops.constant(-1e30, F32, device=self.dref))
        p = ops.softmax(s)
        o = ops.cast(ops.matmul(p, ops.cast(vh, F32)), BF16)  # [H, T, hd]
        o = ops.reshape(ops.transpose(o, 0, 1), [T, self.H * self.hd])
        return self._mm(o * ops.sigmoid(gate), sp + "o_proj", layer["o_proj"]), k, v

    def _deltanet(self, xn, lp, layer, conv_pool, rec_pool, slots, offs):
        T = xn.shape[0]
        W = self._W
        qkv = ops.cast(self._mm(xn, lp + "in_proj_qkv", layer["in_proj_qkv"]), F32)
        z = self._mm(xn, lp + "in_proj_z", layer["in_proj_z"])
        b = ops.cast(self._mm(xn, lp + "in_proj_b", layer["in_proj_b"]), F32)
        a = ops.cast(self._mm(xn, lp + "in_proj_a", layer["in_proj_a"]), F32)
        x_sp = a + W[lp + "dt_bias"]
        softplus = ops.where(x_sp > 20.0, x_sp, ops.log(1.0 + ops.exp(x_sp)))
        decay = ops.exp(-ops.exp(W[lp + "A_log"]) * softplus)
        beta = ops.sigmoid(b)
        conv = gated_delta_conv1d_fwd(qkv_input_ragged=qkv, conv_weight=W[lp + "conv1d"], conv_state=conv_pool,
                                      slot_idx=slots, input_row_offsets=offs)
        conv = ops.silu(conv)
        y = gated_delta_recurrence_fwd(qkv_conv_output=conv, decay_per_token=decay, beta_per_token=beta,
                                       recurrent_state=rec_pool, slot_idx=slots, input_row_offsets=offs)
        y = ops.cast(ops.reshape(ops.rebind(y, [T, self.nv * self.vd], "deltanet out"), [T, self.nv, self.vd]), BF16)
        yn = ops.cast(self._rms(y, W[lp + "norm"]), F32)
        zg = ops.silu(ops.cast(ops.reshape(z, [T, self.nv, self.vd]), F32))
        o = ops.cast(yn * zg, BF16)
        return self._mm(ops.reshape(o, [T, self.nv * self.vd]), lp + "out_proj", layer["out_proj"])

    # run ─────────────────────────────────────────────────────────────────────

    def embed(self, ids: np.ndarray) -> np.ndarray:
        wq, s, b = self.emb
        return deq8(np.asarray(wq[ids]), np.asarray(s[ids]), np.asarray(b[ids]), self.emb_gs)

    def _pools(self, n):
        """Persistent pool buffers for `n` slots: every graph overwrites them before use, so no zeros are uploaded."""
        if n not in self._pool_cache:
            bufs = []
            for p in self.plan:
                if p["linear"]:
                    bufs.append(Buffer.from_numpy(np.empty((n, self.conv_dim, self.ck - 1), np.float32)).to(self.dev))
                    bufs.append(Buffer.from_numpy(np.empty((n, self.nv, self.kd, self.vd), np.float32)).to(self.dev))
            self._pool_cache[n] = bufs
        return self._pool_cache[n]

    def _prefix(self, ids: list[int]):
        """One row from zero state: (hidden fp32, [K, V per attention layer]); its final states stay in _pools(1)."""
        P = len(ids)
        dyn = [Buffer.from_numpy(a).to(self.dev) for a in
               (self.embed(np.asarray(ids, dtype=np.int64)).astype(np.float32), np.arange(P, dtype=np.int32),
                np.array([0, P], np.uint32), np.zeros(1, np.uint32))]
        out = self.prefix_model.execute(*dyn, *self._pools(1), *self.bufs)
        return out[0], out[1:]

    def hidden(self, rows: list[list[int]]) -> list[np.ndarray]:
        """Final hidden states (fp32) of each token row, each row computed whole (one prefix-graph call per row)."""
        return [self._prefix(r)[0].to_numpy() for r in rows]

    def hidden_shared(self, prefix: list[int], rows: list[list[int]]) -> tuple[np.ndarray, list[np.ndarray]]:
        """The rows `prefix + r` for each r in `rows`, the prefix computed ONCE: (prefix states, suffix states)."""
        assert prefix, "hidden_shared needs a non-empty prefix (use hidden() for whole rows)"
        hp, kv = self._prefix(prefix)
        P, lens, nb = len(prefix), [len(r) for r in rows], len(rows)
        ids = np.concatenate([np.asarray(r, dtype=np.int64) for r in rows])
        pos = np.concatenate([np.arange(P, P + n, dtype=np.int32) for n in lens])
        rid = np.concatenate([np.full(n, i, dtype=np.int32) for i, n in enumerate(lens)])
        offs = np.concatenate([[0], np.cumsum(lens)]).astype(np.uint32)
        dyn = [Buffer.from_numpy(a).to(self.dev) for a in
               (self.embed(ids).astype(np.float32), pos, rid, offs, np.arange(nb, dtype=np.uint32))]
        out = self.rows_model.execute(*dyn, *self._pools(nb), *self._pools(1), *kv, *self.bufs)[0].to_numpy()
        return hp.to_numpy(), np.split(out, np.cumsum(lens)[:-1])

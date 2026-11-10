# +--------------------------------------------------------------------------+ #
# | The Parakeet encoder ON THE DEVICE, against transformers' ParakeetForTDT
# +--------------------------------------------------------------------------+ #
"""`test_parakeet_reference.mojo`'s comparisons with the encoder on the GPU.

    pixi run -e apple mojo build -I . tests/audio/test_parakeet_gpu.mojo -o build/test_parakeet_gpu
    ./build/test_parakeet_gpu [ref_root] [case ...]            (-e nvidia on a CUDA machine)

The device path shares the `Linear` / `LayerNorm` leaves with the CPU path and
replaces every plain loop with a kernel (`kernels.mojo`), so it is gated on the
REFERENCE again rather than on the CPU path: subsampling, each block from the
reference's own input, the chain, the encoder projection, then the greedy
trace and the transcript (decoded on the device too), which must be equal. Same
dumps, same compiled maximum, same error measure (max |d| over the valid
frames / max |reference|).

Falsification, with what a device-loaded model still allows (its weights have
no host copy to edit): the padding mask dropped, and the positional table
shifted by one offset. Each must push block 0 over `STAGE_BOUND`.

## Two sets of bounds, because NVIDIA's GEMMs are TF32 by default

On Metal, and on NVIDIA built with `-D NN_CUBLAS_FP32=1`, every GEMM is fp32
and the bounds are the CPU gate's (2e-5 a stage, 5e-5 the chain; measured
2.9e-6 at worst on an Orin). By default nn's large GEMMs on NVIDIA run on the
tensor cores in TF32 (10-bit mantissa products, `cublas_tf32`): 2.4x faster on
the Orin (52 ms against 127 ms for the encoder) and 4e-4 a stage, 8e-4 the
chain. So the default NVIDIA build is held to `1.5e-3` / `3e-3` — still under
the smallest defect (4.2e-3) — and, like every build, to the EQUAL trace and
transcript. The fp32 build is what proves the device code; the TF32 bound only
says the precision loss stayed where it was measured.

## Half-precision weights: `-D PK_FP16=1` (NVIDIA)

Builds `ParakeetSTT`'s `WEIGHTS = "fp16"` form: the blocks' GEMM weights stored
as float16. Its reference is NOT the float32 dump — rounding the weights moves
the encoder output by up to 1e-2 by itself, and a bound that loose would pass
the positions-off-by-one defect (4e-3) — but the SAME model with the SAME
weights rounded, dumped by `dump_parakeet_reference.py --weights float16` into
`~/.cache/noeira/parakeet/ref16` (this build's default root). What is left to
bound is then only what the port adds: the activation operand cast to float16.
"""

from std.os import getenv
from std.os.path import exists
from std.sys import argv, default_accelerator
from std.sys.defines import is_defined, get_defined_int
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.io.npy import load_npy_f64
from noeira.audio.parakeet.config import PK_D, PK_DEC, PK_LAYERS, PK_MELS, PK_VOCAB
from noeira.audio.parakeet.decode import PkDecodeWork, PkTrace, pk_greedy_decode
from noeira.audio.parakeet.encoder import (
    PkEncoderWork, pk_subsample, pk_block, pk_encode,
)
from noeira.audio.parakeet.paths import parakeet_snapshot_dir
from noeira.audio.parakeet.tokenizer import ParakeetTokenizer
from noeira.audio.parakeet.weights import ParakeetTDT
from noeira.nn.core.tensor import Tensor


comptime F_MAX: Int = get_defined_int["PK_F_MAX", 801]()
"""Mel frames compiled for: 801 (8 s) unless built with `-D PK_F_MAX=401` (4 s)."""
comptime T_MAX: Int = PkEncoderWork[F_MAX].T
comptime WEIGHTS: StaticString = "fp16" if is_defined["PK_FP16"]() else "fp32"
comptime TF32: Bool = default_accelerator().is_nvidia_gpu() and not is_defined["NN_CUBLAS_FP32"]()
comptime REDUCED: Bool = TF32 or WEIGHTS == "fp16"
"""Some operand of the GEMMs is rounded to ~11 bits (TF32, or float16)."""
comptime STAGE_BOUND: Float64 = 1.5e-3 if REDUCED else 2.0e-5
comptime CHAIN_BOUND: Float64 = 3.0e-3 if REDUCED else 5.0e-5


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def rel_error(ref got: Tensor, ref want: List[Float64], n: Int) -> Float64:
    """Largest |got - want| over the first `n` host values, over the largest |want|."""
    var worst = 0.0
    var scale = 0.0
    for i in range(n):
        var d = _abs(Float64(got.data[i]) - want[i])
        if d > worst:
            worst = d
        var a = _abs(want[i])
        if a > scale:
            scale = a
    return worst / scale if scale > 0.0 else 1.0e30


def layer_name(i: Int) -> String:
    return String("layer_") + (String("0") if i < 10 else String("")) + String(i)


def set_rows(mut t: Tensor, ref vals: List[Float64], width: Int, rows: Int):
    """`vals` (`[r, width]`) into the top of `t`'s HOST copy, zero below."""
    for i in range(len(vals)):
        t.data[i] = Float32(vals[i])
    for i in range(len(vals), rows * width):
        t.data[i] = 0.0


def check(what: String, err: Float64, bound: Float64) -> Int:
    var ok = err <= bound
    print("  " + ("PASS" if ok else "FAIL") + "  " + what + "  rel err = " + String(err))
    return 0 if ok else 1


def caught(what: String, err: Float64) -> Int:
    var ok = err > STAGE_BOUND
    print(
        "  " + ("PASS" if ok else "FAIL") + "  defect: " + what + "  rel err = "
        + String(err) + ("  (caught)" if ok else "  NOT CAUGHT")
    )
    return 0 if ok else 1


def same_trace(ref t: PkTrace, ref tokens: List[Float64], ref durations: List[Float64]) -> Bool:
    if len(t.tokens) != len(tokens) or len(t.durations) != len(durations):
        return False
    for i in range(len(tokens)):
        if t.tokens[i] != Int(tokens[i]) or t.durations[i] != Int(durations[i]):
            return False
    return True


def read_line(path: String) raises -> String:
    var f = open(path, "r")
    var s = String(f.read().strip())
    f.close()
    return s^


def main() raises:
    var args = argv()
    var root = getenv("HOME") + "/.cache/noeira/parakeet/" + (
        String("ref16") if WEIGHTS == "fp16" else String("ref")
    )
    if len(args) >= 2:
        root = String(args[1])
    var cases = List[String]()
    for i in range(2, len(args)):
        cases.append(String(args[i]))
    if len(cases) == 0:
        cases = [String("fr_10"), String("fr_10_pad8"), String("en_16")]
    if not exists(root + "/" + cases[0] + "/layer_23.npy"):
        raise Error(
            "no reference dump at " + root + "/" + cases[0]
            + " — generate it with tools/parakeet/dump_parakeet_reference.py"
        )

    print("Parakeet encoder ON THE DEVICE vs transformers ParakeetForTDT")
    print("  reference dumps: " + root + "   compiled maximum: " + String(F_MAX)
          + " mel frames, " + String(T_MAX) + " encoder frames")
    print("  weights: " + String(WEIGHTS) + "   GEMMs: "
          + (String("TF32 (NVIDIA default)") if TF32 else String("fp32"))
          + "   bounds: stage " + String(STAGE_BOUND) + ", chain " + String(CHAIN_BOUND))
    var fails = 0
    var ctx = DeviceContext()
    var t_load = perf_counter_ns()
    var model = ParakeetTDT.make["gpu"](ctx)
    model.load["gpu", WEIGHTS](parakeet_snapshot_dir() + "/model.safetensors", ctx)
    print("  PASS  load on the device in " + String(Int(perf_counter_ns() - t_load) // 1000000)
          + " ms: every parameter filled, every fp32 tensor of the file read")
    var tok = ParakeetTokenizer.load(parakeet_snapshot_dir() + "/tokenizer.json", PK_VOCAB)
    var ws = PkEncoderWork[F_MAX].make["gpu", WEIGHTS](ctx)
    var dw = PkDecodeWork.make["gpu"](ctx)
    var compared = 0

    for c in range(len(cases)):
        var dir = root + "/" + cases[c]
        var feats = load_npy_f64(dir + "/features.npy")
        var n_mel = Int(load_npy_f64(dir + "/feature_len.npy")[0])
        var want_len = Int(load_npy_f64(dir + "/encoder_len.npy")[0])
        var want_sub = load_npy_f64(dir + "/subsampling.npy")
        print("  case " + cases[c] + ": " + String(n_mel) + " mel frames, reference "
              + String(want_len) + " valid of " + String(len(want_sub) // PK_D) + " encoder frames")

        # subsampling from the reference's features
        set_rows(ws.feats, feats, PK_MELS, F_MAX)
        var valid = pk_subsample[F_MAX, "gpu"](model, ws, n_mel, ctx)
        ws.x.download(ctx)
        if valid != want_len:
            print("  FAIL  valid encoder frames: " + String(valid) + ", reference " + String(want_len))
            fails += 1
        fails += check(String("subsampling"), rel_error(ws.x, want_sub, valid * PK_D), STAGE_BOUND)

        # each block in isolation
        var prev = want_sub.copy()
        var case_worst = 0.0
        var worst_layer = 0
        for i in range(PK_LAYERS):
            var want = load_npy_f64(dir + "/" + layer_name(i) + ".npy")
            set_rows(ws.x, prev, PK_D, T_MAX)
            ws.x.upload_resident(ctx)
            pk_block[F_MAX, "gpu", WEIGHTS](model, ws, i, valid, ctx)
            ws.x.download(ctx)
            var e = rel_error(ws.x, want, valid * PK_D)
            if e > case_worst:
                case_worst = e
                worst_layer = i
            compared += valid * PK_D
            prev = want^
        fails += check(
            String("24 blocks, each from the reference's input (worst: block ")
            + String(worst_layer) + ")", case_worst, STAGE_BOUND,
        )

        # the chain, twice: the second run is the warm time
        set_rows(ws.feats, feats, PK_MELS, F_MAX)
        var valid2 = pk_encode[F_MAX, "gpu", WEIGHTS](model, ws, n_mel, ctx)
        var t0 = perf_counter_ns()
        valid2 = pk_encode[F_MAX, "gpu", WEIGHTS](model, ws, n_mel, ctx)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        ws.x.download(ctx)
        ws.proj.download(ctx)
        var want_last = load_npy_f64(dir + "/layer_23.npy")
        var want_proj = load_npy_f64(dir + "/encoder_proj.npy")
        fails += check(String("chain: block 23 output"), rel_error(ws.x, want_last, valid2 * PK_D), CHAIN_BOUND)
        fails += check(String("chain: encoder projection"), rel_error(ws.proj, want_proj, valid2 * PK_DEC), CHAIN_BOUND)

        # decoding on the device, from the projection where it is
        var want_tokens = load_npy_f64(dir + "/tokens.npy")
        var want_durs = load_npy_f64(dir + "/durations.npy")
        var want_text = read_line(dir + "/transcript.txt")
        var t2 = perf_counter_ns()
        var trace = pk_greedy_decode["gpu", T_MAX](model, ws.proj, valid2, T_MAX, dw, ctx)
        var t3 = perf_counter_ns()
        var text = tok.decode(trace.text_ids())
        var trace_ok = same_trace(trace, want_tokens, want_durs)
        var text_ok = text == want_text
        print("  " + ("PASS" if trace_ok else "FAIL") + "  token/duration trace: "
              + String(len(trace.tokens)) + " steps (reference " + String(len(want_tokens)) + ")")
        print("  " + ("PASS" if text_ok else "FAIL") + "  transcript: '" + text + "'")
        print("        encoder (warm, incl. the upload) " + String(Float64(Int(t1 - t0)) / 1.0e6)
              + " ms, decode on the device " + String(Float64(Int(t3 - t2)) / 1.0e6) + " ms")
        if not trace_ok:
            fails += 1
        if not text_ok:
            fails += 1

    if compared == 0:
        raise Error("gate: nothing compared — this run proved nothing")

    # ── falsification, on the first case (block 0, reference input) ──────
    var dir0 = root + "/" + cases[0]
    var sub0 = load_npy_f64(dir0 + "/subsampling.npy")
    var want0 = load_npy_f64(dir0 + "/layer_00.npy")
    var valid0 = Int(load_npy_f64(dir0 + "/encoder_len.npy")[0])

    # (a) no padding mask: every compiled frame treated as valid
    set_rows(ws.x, sub0, PK_D, T_MAX)
    ws.x.upload_resident(ctx)
    pk_block[F_MAX, "gpu", WEIGHTS](model, ws, 0, T_MAX, ctx)
    ws.x.download(ctx)
    fails += caught(String("padding mask dropped (all frames valid)"), rel_error(ws.x, want0, valid0 * PK_D))

    # (b) the positional table one offset off
    var saved_pos = ws.pos.data.copy()
    for i in range(len(saved_pos) - PK_D):
        ws.pos.data[i] = saved_pos[i + PK_D]
    ws.pos.upload_resident(ctx)
    ws.relk_ready = False  # the projected tables are a cache of `pos`
    set_rows(ws.x, sub0, PK_D, T_MAX)
    ws.x.upload_resident(ctx)
    pk_block[F_MAX, "gpu", WEIGHTS](model, ws, 0, valid0, ctx)
    ws.x.download(ctx)
    fails += caught(String("relative positions off by one"), rel_error(ws.x, want0, valid0 * PK_D))
    for i in range(len(saved_pos)):
        ws.pos.data[i] = saved_pos[i]
    ws.pos.upload_resident(ctx)
    ws.relk_ready = False

    # and the unperturbed block passes again
    set_rows(ws.x, sub0, PK_D, T_MAX)
    ws.x.upload_resident(ctx)
    pk_block[F_MAX, "gpu", WEIGHTS](model, ws, 0, valid0, ctx)
    ws.x.download(ctx)
    fails += check(String("block 0 after the restores"), rel_error(ws.x, want0, valid0 * PK_D), STAGE_BOUND)

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("parakeet device gate failed")

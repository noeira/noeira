# +--------------------------------------------------------------------------+ #
# | The Mojo Parakeet encoder, against transformers' ParakeetForTDT
# +--------------------------------------------------------------------------+ #
"""Gates `noeira/audio/parakeet/` stage by stage on the reference dump.

    pixi run -e detect python tools/parakeet/dump_parakeet_reference.py ...   (see the manifest)
    pixi run mojo run -I . tests/audio/test_parakeet_reference.mojo [ref_root] [case ...]

One compiled maximum (`F_MAX` = 801 mel frames, 8 s, 101 encoder frames) for
every case, the plain dumps and the reference's own padded one alike.

1. **Load**: `ParakeetTDT.load` raises unless every parameter was filled and
   every fp32 tensor of the file was read.
2. **Positional table** against `pos_embed.npy`.
3. **Subsampling**, from the REFERENCE's features, against `subsampling.npy`.
4. **Each block in isolation**: the reference's `layer_{i-1}` goes in, block `i`
   runs, the result is compared with `layer_i`. An error in one block cannot
   hide behind, or be blamed on, the blocks before it.
5. **The chain**: subsampling, the 24 blocks and the projection run on our own
   intermediate values, compared with `layer_23` and `encoder_proj`.
6. **Decoding**: the greedy token/duration trace from OUR encoder output must
   EQUAL the reference's (`tokens.npy`, `durations.npy`), element for element,
   and the decoded text must equal `transcript.txt`.

Errors are the largest |ours - theirs| over the VALID frames divided by the
largest |theirs| of that stage (the stages' scales run from 0.1 to 4700).
Both sides are float32 with the operations in a different order, so this is a
bound: `STAGE_BOUND` for one stage, `CHAIN_BOUND` after all of them.

Then the FALSIFICATION, on weights and inputs (no switch in the code under
test): each encoder defect must push its stage over `STAGE_BOUND`, and each
decoder defect must change the trace.
"""

from std.os import getenv
from std.os.path import exists
from std.sys import argv

from noeira.io.npy import load_npy_f64
from noeira.audio.parakeet.config import (
    PK_D, PK_DEC, PK_JOINT, PK_LAYERS, PK_MELS, PK_VOCAB,
)
from noeira.audio.parakeet.decode import PkDecodeWork, PkTrace, pk_greedy_decode
from noeira.audio.parakeet.tokenizer import ParakeetTokenizer
from noeira.audio.parakeet.encoder import (
    PkEncoderWork, pk_subsample, pk_block, pk_encode, pk_pos_table,
)
from noeira.audio.parakeet.paths import parakeet_snapshot_dir
from noeira.audio.parakeet.weights import ParakeetTDT
from noeira.nn.core.tensor import Tensor


comptime F_MAX: Int = 801
comptime T_MAX: Int = PkEncoderWork[F_MAX].T
comptime STAGE_BOUND: Float64 = 2.0e-5
"""Measured: 1.7e-6 at worst (subsampling), 8.5e-7 for a block."""
comptime CHAIN_BOUND: Float64 = 5.0e-5
"""Measured: 1.9e-6 after the 24 blocks. The smallest defect below is 4e-3."""
comptime POS_BOUND: Float64 = 2.0e-5
"""Measured 7.7e-6 at T = 101: a float32 angle of up to 100 rad."""


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def rel_error(ref got: Tensor, ref want: List[Float64], n: Int) -> Float64:
    """Largest |got - want| over the first `n` values, over the largest |want|."""
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
    """`vals` (`[r, width]`) into the top of `t` (`[rows, width]`), zero below."""
    for i in range(len(vals)):
        t.data[i] = Float32(vals[i])
    for i in range(len(vals), rows * width):
        t.data[i] = 0.0


def check(what: String, err: Float64, bound: Float64) -> Int:
    var ok = err <= bound
    print("  " + ("PASS" if ok else "FAIL") + "  " + what + "  rel err = " + String(err))
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


def caught(what: String, err: Float64) -> Int:
    var ok = err > STAGE_BOUND
    print(
        "  " + ("PASS" if ok else "FAIL") + "  defect: " + what + "  rel err = "
        + String(err) + ("  (caught)" if ok else "  NOT CAUGHT")
    )
    return 0 if ok else 1


def main() raises:
    var args = argv()
    var root = getenv("HOME") + "/.cache/noeira/parakeet/ref"
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

    print("Parakeet encoder vs transformers ParakeetForTDT")
    print("  reference dumps: " + root + "   compiled maximum: " + String(F_MAX)
          + " mel frames, " + String(T_MAX) + " encoder frames")
    var fails = 0

    # ── 1. load ──────────────────────────────────────────────────────────
    var model = ParakeetTDT.make["cpu"]()
    model.load["cpu"](parakeet_snapshot_dir() + "/model.safetensors")
    print("  PASS  load: every parameter filled, every fp32 tensor of the file read")

    var tok = ParakeetTokenizer.load(parakeet_snapshot_dir() + "/tokenizer.json", PK_VOCAB)
    var ws = PkEncoderWork[F_MAX]()
    var dw = PkDecodeWork()
    var isolated_worst = 0.0
    var compared = 0

    for c in range(len(cases)):
        var dir = root + "/" + cases[c]
        var feats = load_npy_f64(dir + "/features.npy")
        var n_mel = Int(load_npy_f64(dir + "/feature_len.npy")[0])
        var want_len = Int(load_npy_f64(dir + "/encoder_len.npy")[0])
        var want_sub = load_npy_f64(dir + "/subsampling.npy")
        var ref_t = len(want_sub) // PK_D
        print("  case " + cases[c] + ": " + String(n_mel) + " mel frames, reference "
              + String(want_len) + " valid of " + String(ref_t) + " encoder frames")

        # 2. positional table, at the length the reference built it for
        var want_pos = load_npy_f64(dir + "/pos_embed.npy")
        var pos = pk_pos_table(ref_t)
        var pos_worst = 0.0
        if len(pos) != len(want_pos):
            raise Error(cases[c] + ": pos_embed.npy is not [2T - 1, 1024]")
        for i in range(len(pos)):
            var d = _abs(Float64(pos[i]) - want_pos[i])
            if d > pos_worst:
                pos_worst = d
        fails += check(String("positional table (max |d|)"), pos_worst, POS_BOUND)

        # 3. subsampling from the reference's features
        set_rows(ws.feats, feats, PK_MELS, F_MAX)
        var valid = pk_subsample[F_MAX](model, ws, n_mel)
        if valid != want_len:
            print("  FAIL  valid encoder frames: " + String(valid) + ", reference " + String(want_len))
            fails += 1
        fails += check(String("subsampling"), rel_error(ws.x, want_sub, valid * PK_D), STAGE_BOUND)

        # 4. each block in isolation
        var prev = want_sub.copy()
        var case_worst = 0.0
        var worst_layer = 0
        for i in range(PK_LAYERS):
            var want = load_npy_f64(dir + "/" + layer_name(i) + ".npy")
            set_rows(ws.x, prev, PK_D, T_MAX)
            pk_block[F_MAX](model, ws, i, valid)
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
        if case_worst > isolated_worst:
            isolated_worst = case_worst

        # 5. the chain, on our own intermediates
        set_rows(ws.feats, feats, PK_MELS, F_MAX)
        var valid2 = pk_encode[F_MAX](model, ws, n_mel)
        var want_last = load_npy_f64(dir + "/layer_23.npy")
        var want_proj = load_npy_f64(dir + "/encoder_proj.npy")
        fails += check(String("chain: block 23 output"), rel_error(ws.x, want_last, valid2 * PK_D), CHAIN_BOUND)
        fails += check(String("chain: encoder projection"), rel_error(ws.proj, want_proj, valid2 * PK_DEC), CHAIN_BOUND)

        # 6. greedy TDT decoding from our own encoder output
        var want_tokens = load_npy_f64(dir + "/tokens.npy")
        var want_durs = load_npy_f64(dir + "/durations.npy")
        var want_text = read_line(dir + "/transcript.txt")
        var trace = pk_greedy_decode(model, ws.proj, valid2, T_MAX, dw)
        var text = tok.decode(trace.text_ids())
        var trace_ok = same_trace(trace, want_tokens, want_durs)
        var text_ok = text == want_text
        print("  " + ("PASS" if trace_ok else "FAIL") + "  token/duration trace: "
              + String(len(trace.tokens)) + " steps (reference " + String(len(want_tokens)) + ")")
        print("  " + ("PASS" if text_ok else "FAIL") + "  transcript: '" + text + "'")
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

    # (a) the two attention biases exchanged
    for i in range(PK_D):
        var u = model.layers[0].bias_u.val.data[i]
        model.layers[0].bias_u.val.data[i] = model.layers[0].bias_v.val.data[i]
        model.layers[0].bias_v.val.data[i] = u
    set_rows(ws.x, sub0, PK_D, T_MAX)
    pk_block[F_MAX](model, ws, 0, valid0)
    fails += caught(String("bias_u and bias_v exchanged"), rel_error(ws.x, want0, valid0 * PK_D))
    for i in range(PK_D):
        var u = model.layers[0].bias_u.val.data[i]
        model.layers[0].bias_u.val.data[i] = model.layers[0].bias_v.val.data[i]
        model.layers[0].bias_v.val.data[i] = u

    # (b) the positional table one offset off
    var saved_pos = ws.pos.data.copy()
    for i in range(len(saved_pos) - PK_D):
        ws.pos.data[i] = saved_pos[i + PK_D]
    ws.relk_ready = False  # the projected tables are a cache of `pos`
    set_rows(ws.x, sub0, PK_D, T_MAX)
    pk_block[F_MAX](model, ws, 0, valid0)
    fails += caught(String("relative positions off by one"), rel_error(ws.x, want0, valid0 * PK_D))
    for i in range(len(saved_pos)):
        ws.pos.data[i] = saved_pos[i]
    ws.relk_ready = False

    # (c) BatchNorm without its running variance
    var saved_var = model.layers[0].bn.running_var.t.data.copy()
    for i in range(PK_D):
        model.layers[0].bn.running_var.t.data[i] = 1.0
    set_rows(ws.x, sub0, PK_D, T_MAX)
    pk_block[F_MAX](model, ws, 0, valid0)
    fails += caught(String("BatchNorm running variance = 1"), rel_error(ws.x, want0, valid0 * PK_D))
    for i in range(PK_D):
        model.layers[0].bn.running_var.t.data[i] = saved_var[i]

    # (d) no padding mask: every compiled frame treated as valid
    set_rows(ws.x, sub0, PK_D, T_MAX)
    pk_block[F_MAX](model, ws, 0, T_MAX)
    fails += caught(String("padding mask dropped (all frames valid)"), rel_error(ws.x, want0, valid0 * PK_D))

    # and the unperturbed block passes again: the restores above are complete
    set_rows(ws.x, sub0, PK_D, T_MAX)
    pk_block[F_MAX](model, ws, 0, valid0)
    fails += check(String("block 0 after the restores"), rel_error(ws.x, want0, valid0 * PK_D), STAGE_BOUND)

    # ── decoder falsification, on the first case's reference projection ──
    var proj0 = load_npy_f64(dir0 + "/encoder_proj.npy")
    var tokens0 = load_npy_f64(dir0 + "/tokens.npy")
    var durs0 = load_npy_f64(dir0 + "/durations.npy")
    var enc0 = Tensor.alloc(T_MAX * PK_DEC)
    set_rows(enc0, proj0, PK_DEC, T_MAX)
    var base = pk_greedy_decode(model, enc0, valid0, T_MAX, dw)
    var base_ok = same_trace(base, tokens0, durs0)
    print("  " + ("PASS" if base_ok else "FAIL") + "  trace from the reference's projection")
    if not base_ok:
        fails += 1

    # (e) the first LSTM layer forgets: its recurrent weights zeroed.
    # ⚠ Layer 0, not layer 1: zeroing layer 1's recurrence leaves this clip's
    # trace UNCHANGED in the reference too (checked in torch), so it would be a
    # defect no correct decoder could be caught by.
    var saved_hh = model.lstm[0].hh.weight.val.data.copy()
    for i in range(len(saved_hh)):
        model.lstm[0].hh.weight.val.data[i] = 0.0
    var t_e = pk_greedy_decode(model, enc0, valid0, T_MAX, dw)
    var e_caught = not same_trace(t_e, tokens0, durs0)
    print("  " + ("PASS" if e_caught else "FAIL") + "  defect: LSTM layer 0 without recurrence -> '"
          + tok.decode(t_e.text_ids()) + "'" + ("  (caught)" if e_caught else "  NOT CAUGHT"))
    if not e_caught:
        fails += 1
    for i in range(len(saved_hh)):
        model.lstm[0].hh.weight.val.data[i] = saved_hh[i]

    # (f) durations 1 and 2 exchanged in the joint head (ours is [in, out])
    var d1 = PK_VOCAB + 1
    for r in range(PK_DEC):
        var w1 = model.joint.weight.val.data[r * PK_JOINT + d1]
        model.joint.weight.val.data[r * PK_JOINT + d1] = model.joint.weight.val.data[r * PK_JOINT + d1 + 1]
        model.joint.weight.val.data[r * PK_JOINT + d1 + 1] = w1
    var b1 = model.joint.bias.val.data[d1]
    model.joint.bias.val.data[d1] = model.joint.bias.val.data[d1 + 1]
    model.joint.bias.val.data[d1 + 1] = b1
    var t_f = pk_greedy_decode(model, enc0, valid0, T_MAX, dw)
    var f_caught = not same_trace(t_f, tokens0, durs0)
    print("  " + ("PASS" if f_caught else "FAIL") + "  defect: durations 1 and 2 exchanged"
          + ("  (caught)" if f_caught else "  NOT CAUGHT"))
    if not f_caught:
        fails += 1

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("parakeet encoder gate failed")

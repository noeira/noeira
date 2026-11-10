"""LFM3's backward with LoRA (`noeira.llm.lfm3.train`) against torch autograd
through d1-2B's own code (`tools/d1/dump_d1_grad_reference.py`).

    build on the Orin: build-jetson tests/llm/test_lfm3_backward_gpu.mojo
    ./test_lfm3_backward_gpu [d1 dir] [grad dir]   (defaults: ~/.cache/noeira/local-ai/d1/d1-2B, ~/.cache/noeira/d1/grad)

Per case, with the reference's 142 adapters (rank 16, scale 2, B nonzero):
the adapted forward's slot log-softmax and loss, then every gradient — the
input embeddings' and all 284 adapter matrices' — max |d| / max |reference|.
"""

from std.os import getenv
from std.os.path import exists
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.io.json import load_json
from noeira.io.npy import load_npy_f64
from noeira.io.safetensors import SafeTensors
from noeira.llm.lfm3 import Lfm3Text
from noeira.llm.lfm3.train import slot_loss, backward
from noeira.nn.constants import DT


comptime FWD_BOUND: Float64 = 1e-4
comptime GRAD_BOUND: Float64 = 3e-3
"""fp32 against fp32 through 24 layers and back: measured <= 2.6e-4 on the
embeddings, <= 1.0e-3 on an adapter (mean 1.3e-4 / 2.2e-4; the worst always in
layer 23, nearest the loss, which reads the logits' own ~3e-4 fp32 differences).
Three times the worst. Needs a WELL-CONDITIONED loss: the reference's targets
are each slot's third most likely token (`dump_d1_grad_reference.py`)."""


def rel_err(got: List[Float32], want: List[Float32]) -> Float64:
    var d = 0.0
    var m = 0.0
    for i in range(len(want)):
        d = max(d, abs(Float64(got[i]) - Float64(want[i])))
        m = max(m, abs(Float64(want[i])))
    return d / m if m > 0.0 else d


def check(mut fails: Int, ok: Bool, what: String):
    if ok:
        print("  PASS ", what)
    else:
        print("  FAIL ", what)
        fails += 1


def main() raises:
    var args = argv()
    var home = getenv("HOME")
    var model_dir = String(args[1]) if len(args) > 1 else home + "/.cache/noeira/local-ai/d1/d1-2B"
    var grad_dir = String(args[2]) if len(args) > 2 else home + "/.cache/noeira/d1/grad"
    var ref_dir = home + "/.cache/noeira/d1/ref"
    print("LFM3 backward with LoRA vs torch autograd through d1's own code")
    var lm = Lfm3Text.load(DeviceContext(), model_dir)
    var fails = 0
    var cases: List[String] = ["room_en", "single_fr"]
    for cname in cases:
        var gdir = grad_dir + "/" + cname
        lm.lora_index = Dict[String, Int]()
        lm.lora_a = List[DeviceBuffer[DT]]()
        lm.lora_b = List[DeviceBuffer[DT]]()
        lm.lora_da = List[DeviceBuffer[DT]]()
        lm.lora_db = List[DeviceBuffer[DT]]()
        lm.lora_shape = List[Tuple[Int, Int]]()
        lm.load_lora(gdir + "/lora.safetensors", 16, 2.0)
        var meta = load_json(gdir + "/meta.json")
        var ids_f = load_npy_f64(ref_dir + "/" + cname + "/row0_ids.npy")
        var gr_f = load_npy_f64(ref_dir + "/" + cname + "/row0_question.npy")
        var ids = List[Int]()
        var groups = List[Int]()
        for i in range(len(ids_f)):
            ids.append(Int(ids_f[i]))
            groups.append(Int(gr_f[i]))
        var slots = List[Int]()
        var targets = List[Int]()
        var sn = meta.field(meta.root(), "slots")
        var tn = meta.field(meta.root(), "targets")
        for i in range(meta.size(sn)):
            slots.append(meta.integer(meta.at(sn, i)))
            targets.append(meta.integer(meta.at(tn, i)))
        var state_len = meta.integer(meta.field(meta.root(), "state_len"))
        var t = len(ids)
        print("  ", cname, ":", t, "tokens,", len(lm.lora_a), "adapters")

        # the adapted forward layer by layer, when the reference has them (localises a divergence)
        if exists(gdir + "/layer0.npy"):
            var ltaps = List[DeviceBuffer[DT]]()
            _ = lm.forward(ids, groups, state_len, ltaps, keep_taps=True)
            var line = String("     adapted forward per layer:")
            for l in range(len(lm.cfg.layer_types)):
                var want = load_npy_f64(gdir + "/layer" + String(l) + ".npy")
                var got = lm.download(ltaps[l + 1], t * lm.cfg.hidden)
                var d = 0.0
                var m = 0.0
                for i in range(len(want)):
                    d = max(d, abs(Float64(got[i]) - want[i]))
                    m = max(m, abs(want[i]))
                line += String(" ") + String(l) + ":" + String(Float32(d / m))
            print(line)

        var t0 = perf_counter_ns()
        lm.training = True
        var taps = List[DeviceBuffer[DT]]()
        var h = lm.forward(ids, groups, state_len, taps)
        var lt = slot_loss(lm, h, t, slots, targets)
        var dembed = backward(lm, lt[1], t)
        lm.ctx.synchronize()
        var ms = Float64(perf_counter_ns() - t0) / 1e6
        lm.training = False

        var lz = lm.slot_logz(h, t, slots)
        var want_lz = load_npy_f64(gdir + "/slot_logz.npy")
        var lzw = List[Float32]()
        for x in want_lz:
            lzw.append(Float32(x))
        check(fails, rel_err(lz, lzw) < FWD_BOUND, String("adapted forward, slots' log-softmax  rel err = ") + String(rel_err(lz, lzw)))
        var want_loss = meta.number(meta.field(meta.root(), "loss"))
        check(fails, abs(lt[0] - want_loss) < 1e-4 * max(1.0, abs(want_loss)),
              String("loss ") + String(lt[0]) + " against " + String(want_loss))

        var st = SafeTensors(gdir + "/grads.safetensors")
        var e = rel_err(lm.download(dembed, t * lm.cfg.hidden), st.read_f32("embeds"))
        check(fails, e < GRAD_BOUND, String("d loss / d embeddings  rel err = ") + String(e))
        var worst = 0.0
        var worst_at = String("")
        var mean = 0.0
        var n = 0
        for entry in lm.lora_index.items():
            var key = entry.key
            var j = entry.value
            var r = lm.lora_rank
            var shp = lm.lora_shape[j]
            var ea = rel_err(lm.download(lm.lora_da[j], r * shp[0]), st.read_f32(key + ".lora_A"))
            var eb = rel_err(lm.download(lm.lora_db[j], shp[1] * r), st.read_f32(key + ".lora_B"))
            mean += ea + eb
            n += 2
            if ea > worst:
                worst = ea
                worst_at = key + ".lora_A"
            if eb > worst:
                worst = eb
                worst_at = key + ".lora_B"
        check(fails, worst < GRAD_BOUND, String(n) + " adapter gradients, worst " + worst_at + "  rel err = " + String(worst)
              + " (mean " + String(mean / Float64(n)) + ")")
        print("     forward + loss + backward:", ms, "ms (fp32)")
    # falsification: three backward defects on the first case must land far past the bound
    var gdir = grad_dir + "/" + cases[0]
    var meta = load_json(gdir + "/meta.json")
    var ids_f = load_npy_f64(ref_dir + "/" + cases[0] + "/row0_ids.npy")
    var gr_f = load_npy_f64(ref_dir + "/" + cases[0] + "/row0_question.npy")
    var ids = List[Int]()
    var groups = List[Int]()
    for i in range(len(ids_f)):
        ids.append(Int(ids_f[i]))
        groups.append(Int(gr_f[i]))
    var slots = List[Int]()
    var targets = List[Int]()
    var sn = meta.field(meta.root(), "slots")
    var tn = meta.field(meta.root(), "targets")
    for i in range(meta.size(sn)):
        slots.append(meta.integer(meta.at(sn, i)))
        targets.append(meta.integer(meta.at(tn, i)))
    var state_len = meta.integer(meta.field(meta.root(), "state_len"))
    var st = SafeTensors(gdir + "/grads.safetensors")
    var want_e = st.read_f32("embeds")
    var names: List[String] = ["softmax VJP skipped", "RoPE gradient rotated the wrong way", "q norm skipped in the backward"]
    var bits: List[Int] = [8, 16, 32]
    for i in range(3):
        lm.debug = bits[i]
        lm.training = True
        var taps = List[DeviceBuffer[DT]]()
        var h = lm.forward(ids, groups, state_len, taps)
        var lt = slot_loss(lm, h, len(ids), slots, targets)
        var d = backward(lm, lt[1], len(ids))
        lm.training = False
        var e = rel_err(lm.download(d, len(ids) * lm.cfg.hidden), want_e)
        check(fails, e > 10.0 * GRAD_BOUND, String("defect caught: ") + names[i] + "  d embeddings rel err = " + String(e))
    lm.debug = 0
    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")

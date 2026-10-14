"""LoRA against PEFT itself, and PEFT's adapter files both ways (CPU).

    pixi run -e act-ref python tools/peft/lora_peft_reference.py \\
        --out tests/fixtures/peft                # PEFT's files (committed)
    pixi run mojo run -I . tests/nn/peft/test_peft_io.mojo
    pixi run -e act-ref python tools/peft/lora_peft_reference.py \\
        --verify /tmp/peft_ours                  # PEFT loads ours

The fixture is PEFT's own: `nn.Linear(13, 7)` at the module path `proj`,
wrapped with r = 4 and lora_alpha = 10, a seeded nonzero lora_B, and
autograd's y, dx, dA and dB for one (x, g).

  [1] parity: PEFT's adapter loaded into `LoRALinear[13, 7, 4]` over the same
      base: forward, grad_input, dA and dB against PEFT's within relative
      1e-5, plain (scale 2.5) and rsLoRA (scale 5)
  [2] writer: noeira's file of the adapter it loaded has PEFT's keys, shapes
      and dtype and BIT-identical values (two transposes), and its config
      reads back with PEFT's r, lora_alpha, use_rslora and target_modules
  [3] bf16: the same adapter written as bf16 is BYTE-identical to PEFT's own
      bf16 save, i.e. torch's rounding
  [4] guards: an r = 4 file into an R = 5 adapter, the plain file into an
      rsLoRA adapter, and a DoRA config, each raise
  [5] the reverse bundle: one Adam step moves the adapter off the fixture,
      then the base, the adapter and (x, y) go to /tmp/peft_ours for PEFT's
      `--verify`
"""

from std.math import abs
from std.os import makedirs
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.optimizer.adam import Adam
from noeira.nn.peft.lora import LoRALinear
from noeira.nn.peft.peft_io import (
    PeftAdapterWriter, read_peft_config, load_peft_lora, peft_key,
)
from noeira.io.safetensors import SafeTensors, SafeTensorsWriter, dtype_name
from noeira.io.fileio import read_file_bytes, write_text_atomic
from noeira.core.bytes import string_from_bytes

comptime FIX = "tests/fixtures/peft"
comptime OURS = "/tmp/peft_ours"
comptime IN = 13
comptime OUT = 7
comptime R = 4
comptime T = 5


def _err(ref ours: List[Float64], ref ref_: List[Float64]) -> Float64:
    """max |ours - ref| / max |ref|."""
    var d = Float64(0)
    var m = Float64(0)
    for i in range(len(ref_)):
        d = max(d, abs(ours[i] - ref_[i]))
        m = max(m, abs(ref_[i]))
    return d / m


def _f64(ref v: List[Float32]) -> List[Float64]:
    var out = List[Float64](capacity=len(v))
    for i in range(len(v)):
        out.append(Float64(v[i]))
    return out^


def _load_base[RS: Bool](mut m: LoRALinear[IN, OUT, R, 16, RS]) raises:
    var st = SafeTensors(String(FIX) + "/base.safetensors")
    var w = st.read_f32("proj.weight")  # [OUT, IN]
    var b = st.read_f32("proj.bias")
    for i in range(IN):
        for o in range(OUT):
            m.base.weight.val.data[i * OUT + o] = w[o * IN + i]
    for o in range(OUT):
        m.base.bias.val.data[o] = b[o]
    m.base.weight.val.version += 1
    m.base.bias.val.version += 1


# ── [1] ─────────────────────────────────────────────────────────────────
def parity[RS: Bool](adapter_dir: String, suffix: String, label: String) raises:
    comptime M = LoRALinear[IN, OUT, R, 16, RS]
    comptime RP = M.Adapter.R_PAD
    var m = M.make["cpu", Deterministic]()
    _load_base(m)
    var cfg = read_peft_config(String(FIX) + "/" + adapter_dir)
    var st = SafeTensors(String(FIX) + "/" + adapter_dir + "/adapter_model.safetensors")
    load_peft_lora["cpu"](st, cfg, String("proj"), m.adapter)
    var io = SafeTensors(String(FIX) + "/io.safetensors")
    var xv = io.read_f32("x")
    var gv = io.read_f32("g")
    var x = Tensor.alloc(T * IN)
    var g = Tensor.alloc(T * OUT)
    for i in range(T * IN):
        x.data[i] = xv[i]
    for i in range(T * OUT):
        g.data[i] = gv[i]
    var y = Tensor.alloc(T * OUT)
    var gx = Tensor.alloc(T * IN)
    m.zero_grad["cpu"](None)
    m.forward["cpu", T](TensorRefs[1](x), y)
    m.vjp["cpu", T](TensorRefs[1](x), g, TensorRefs[1](gx))
    var y_ours = List[Float64]()
    for i in range(T * OUT):
        y_ours.append(Float64(y.data[i]))
    var dx_ours = List[Float64]()
    for i in range(T * IN):
        dx_ours.append(Float64(gx.data[i]))
    var da_ours = List[Float64]()  # PEFT's [R, IN] order
    for j in range(R):
        for i in range(IN):
            da_ours.append(Float64(m.adapter.lora_A.grd.data[i * RP + j]))
    var db_ours = List[Float64]()  # PEFT's [OUT, R] order
    for o in range(OUT):
        for j in range(R):
            db_ours.append(Float64(m.adapter.lora_B.grd.data[j * OUT + o]))
    var e_y = _err(y_ours, _f64(io.read_f32("y" + suffix)))
    var e_x = _err(dx_ours, _f64(io.read_f32("dx" + suffix)))
    var e_a = _err(da_ours, _f64(io.read_f32("dA" + suffix)))
    var e_b = _err(db_ours, _f64(io.read_f32("dB" + suffix)))
    print("  [1]", label, ": scale", m.adapter.scale(), " vs PEFT, max |diff| / max |peft|:  y",
          e_y, " dx", e_x, " dA", e_a, " dB", e_b)
    assert_true(
        max(max(e_y, e_x), max(e_a, e_b)) < 1e-5,
        "noeira's LoRA disagrees with PEFT's",
    )


# ── [2] / [3] ───────────────────────────────────────────────────────────
def _same_tensor(ref a: SafeTensors, ref b: SafeTensors, key: String) raises -> Bool:
    if not a.has(key) or not b.has(key):
        return False
    if a.dtype_of(key) != b.dtype_of(key) or a.shape_str(key) != b.shape_str(key):
        return False
    var ra = a.read_raw(key)
    var rb = b.read_raw(key)
    if len(ra) != len(rb):
        return False
    for i in range(len(ra)):
        if ra[i] != rb[i]:
            return False
    return True


def writer[BF16: Bool](ref_dir: String, label: String) raises:
    comptime M = LoRALinear[IN, OUT, R]
    var m = M.make["cpu", Deterministic]()
    var cfg = read_peft_config(String(FIX) + "/adapter")
    var st = SafeTensors(String(FIX) + "/adapter/adapter_model.safetensors")
    load_peft_lora["cpu"](st, cfg, String("proj"), m.adapter)
    var w = PeftAdapterWriter(r=R, lora_alpha=cfg.lora_alpha, bf16=BF16)
    w.add["cpu"](String("proj"), m.adapter)
    var dir = String(OURS) + "/roundtrip_" + label
    w.save(dir)
    var ours = SafeTensors(dir + "/adapter_model.safetensors")
    var theirs = SafeTensors(String(FIX) + "/" + ref_dir + "/adapter_model.safetensors")
    var ka = peft_key(String("proj"), String("lora_A"))
    var kb = peft_key(String("proj"), String("lora_B"))
    var same_a = _same_tensor(ours, theirs, ka)
    var same_b = _same_tensor(ours, theirs, kb)
    var keys_ok = ours.size() == 2 and theirs.size() == 2
    var back = read_peft_config(dir)
    var cfg_ok = (
        back.r == cfg.r and back.lora_alpha == cfg.lora_alpha
        and back.use_rslora == cfg.use_rslora
        and len(back.target_modules) == 1 and back.target_modules[0] == "proj"
        and len(cfg.target_modules) == 1 and cfg.target_modules[0] == "proj"
    )
    print("  [", label, "] noeira's file vs PEFT's (", dtype_name(ours.dtype_of(ka)), "): ",
          ka, " ", ours.shape_str(ka), " identical ", same_a, ", ", kb, " ",
          ours.shape_str(kb), " identical ", same_b, ", two keys ", keys_ok,
          "; config r ", back.r, " alpha ", back.lora_alpha, " rslora ", back.use_rslora,
          " same ", cfg_ok, sep="")
    assert_true(same_a and same_b and keys_ok, "noeira's adapter file differs from PEFT's")
    assert_true(cfg_ok, "the config did not round-trip")


# ── [4] ─────────────────────────────────────────────────────────────────
def guards() raises:
    var cfg = read_peft_config(String(FIX) + "/adapter")
    var st = SafeTensors(String(FIX) + "/adapter/adapter_model.safetensors")
    var wrong_r = False
    var m5 = LoRALinear[IN, OUT, 5].make["cpu", Deterministic]()
    try:
        load_peft_lora["cpu"](st, cfg, String("proj"), m5.adapter)
    except e:
        wrong_r = True
        print("        r 4 into R 5:", e)
    var wrong_rs = False
    var mrs = LoRALinear[IN, OUT, R, 16, True].make["cpu", Deterministic]()
    try:
        load_peft_lora["cpu"](st, cfg, String("proj"), mrs.adapter)
    except e:
        wrong_rs = True
        print("        plain into rsLoRA:", e)
    var text = string_from_bytes(read_file_bytes(String(FIX) + "/adapter/adapter_config.json"))
    var dora_text = text.replace('"use_dora": false', '"use_dora": true')
    var ddir = String(OURS) + "/dora"
    makedirs(ddir, exist_ok=True)
    write_text_atomic(ddir + "/adapter_config.json", dora_text)
    var dora = False
    try:
        _ = read_peft_config(ddir)
    except e:
        dora = True
        print("        DoRA config:", e)
    print("  [4] refused: wrong rank", wrong_r, ", wrong scale rule", wrong_rs,
          ", DoRA", dora, " (the DoRA config differs:", dora_text != text, ")")
    assert_true(dora_text != text, "the DoRA edit did not apply: [4] is vacuous")
    assert_true(wrong_r and wrong_rs and dora, "a mismatched adapter loaded")


# ── [5] ─────────────────────────────────────────────────────────────────
def reverse_bundle() raises:
    comptime M = LoRALinear[IN, OUT, R]
    var m = M.make["cpu", Deterministic]()
    _load_base(m)
    var cfg = read_peft_config(String(FIX) + "/adapter")
    var st = SafeTensors(String(FIX) + "/adapter/adapter_model.safetensors")
    load_peft_lora["cpu"](st, cfg, String("proj"), m.adapter)
    var io = SafeTensors(String(FIX) + "/io.safetensors")
    var xv = io.read_f32("x")
    var gv = io.read_f32("g")
    var x = Tensor.alloc(T * IN)
    var g = Tensor.alloc(T * OUT)
    for i in range(T * IN):
        x.data[i] = xv[i]
    for i in range(T * OUT):
        g.data[i] = gv[i]
    var y = Tensor.alloc(T * OUT)
    var gx = Tensor.alloc(T * IN)
    var opt = Adam(lr=5e-2)
    m.zero_grad["cpu"](None)
    m.forward["cpu", T](TensorRefs[1](x), y)
    m.vjp["cpu", T](TensorRefs[1](x), g, TensorRefs[1](gx))
    opt.step["cpu"](m)
    m.forward["cpu", T](TensorRefs[1](x), y)
    var w = PeftAdapterWriter(r=R, lora_alpha=cfg.lora_alpha)
    w.add["cpu"](String("proj"), m.adapter)
    w.save(String(OURS) + "/adapter")
    var base = SafeTensorsWriter()
    var wt = List[Float32]()
    for o in range(OUT):
        for i in range(IN):
            wt.append(Float32(m.base.weight.val.data[i * OUT + o]))
    var bt = List[Float32]()
    for o in range(OUT):
        bt.append(Float32(m.base.bias.val.data[o]))
    var sw = List[Int]()
    sw.append(OUT)
    sw.append(IN)
    var sb = List[Int]()
    sb.append(OUT)
    base.add_f32(String("proj.weight"), sw, wt, OUT * IN)
    base.add_f32(String("proj.bias"), sb, bt, OUT)
    base.save(String(OURS) + "/base.safetensors")
    var iow = SafeTensorsWriter()
    var sx = List[Int]()
    sx.append(T)
    sx.append(IN)
    var sy = List[Int]()
    sy.append(T)
    sy.append(OUT)
    var xl = List[Float32]()
    for i in range(T * IN):
        xl.append(Float32(x.data[i]))
    var yl = List[Float32]()
    for i in range(T * OUT):
        yl.append(Float32(y.data[i]))
    iow.add_f32(String("x"), sx, xl, T * IN)
    iow.add_f32(String("y"), sy, yl, T * OUT)
    iow.save(String(OURS) + "/io.safetensors")
    var moved = 0
    var a0 = st.read_f32(peft_key(String("proj"), String("lora_A")))
    var a1 = SafeTensors(String(OURS) + "/adapter/adapter_model.safetensors").read_f32(
        peft_key(String("proj"), String("lora_A"))
    )
    for i in range(len(a0)):
        if a0[i] != a1[i]:
            moved += 1
    print("  [5] after one Adam step,", moved, "of", len(a0),
          "lora_A values differ from the fixture; bundle in", OURS,
          "-> run the --verify step")
    assert_true(moved > 0, "the reverse bundle is the fixture itself: PEFT would verify PEFT")


def main() raises:
    print("LoRA vs PEFT, and PEFT's adapter files (CPU)")
    makedirs(String(OURS), exist_ok=True)
    parity[False](String("adapter"), String(""), String("plain"))
    parity[True](String("adapter_rslora"), String("_rs"), String("rsLoRA"))
    writer[False](String("adapter"), String("2"))
    writer[True](String("adapter_bf16"), String("3"))
    guards()
    reverse_bundle()
    print("PASSED")

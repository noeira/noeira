"""`KokoroService`: Kokoro on its own thread, behind `TextToSpeechService`.

    pixi run -e apple mojo run -I . tests/audio/test_kokoro_service_gpu.mojo    (or -e nvidia; the Orin: build)

Needs the converted checkpoint and espeak-ng (see `tests/manifests/kokoro-ref.txt`).

1. The service's WAV equals the one `KokoroDevice.speak` gives on the caller's
   own thread, byte for byte — the thread hop changes nothing.
2. A caller polling at 50 Hz is never held: the worst `poll` is timed, and
   the frames counted while the model speaks.
3. `cancel` then a new `start`: the reply is the new text's.
4. Falsification: another voice's WAV differs (so 1 is not vacuous), and a
   checkpoint folder that does not exist ends in `SVC_FAILED` with its reason.
"""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.ai.local_service import SVC_LOADING, SVC_READY, SVC_FAILED
from noeira.audio.espeak import EspeakNG
from noeira.audio.kokoro.device import KokoroDevice
from noeira.audio.kokoro.service import KokoroService
from noeira.core.concurrent.thread import sleep_us
from noeira.io.wav import encode_wav, wav_from_float32


comptime TEXT = "I am going to the fridge, then I will come back."
comptime TEXT2 = "Je vais au frigo."
comptime FRAME_US = 20_000


def check(mut fails: Int, ok: Bool, what: String):
    if ok:
        print("  ok:", what)
    else:
        print("  FAIL:", what)
        fails += 1


def direct(mut k: KokoroDevice, mut es: EspeakNG, text: String, voice: String) raises -> List[UInt8]:
    var s = k.speak(es, text, voice)
    var w = wav_from_float32(s.audio, 24000)
    return encode_wav(w)


def same(ref a: List[UInt8], ref b: List[UInt8]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def wait_ready(ref svc: KokoroService, timeout_s: Int) -> Int:
    var t0 = perf_counter_ns()
    while svc.state() == SVC_LOADING and (perf_counter_ns() - t0) < timeout_s * 1_000_000_000:
        _ = sleep_us(5000)
    return svc.state()


def main() raises:
    print("KokoroService (in-process TTS on a worker thread)")
    var fails = 0

    # the reference: the model on THIS thread
    var es = EspeakNG.open()
    var k = KokoroDevice.load(DeviceContext())
    _ = direct(k, es, String("Hello."), String("af_heart"))   # warm
    var ref_en = direct(k, es, TEXT, String("af_heart"))
    var ref_fr = direct(k, es, TEXT2, String("ff_siwis"))
    var ref_other = direct(k, es, TEXT, String("am_michael"))
    check(fails, not same(ref_en, ref_other), "falsification: another voice gives another WAV")

    var svc = KokoroService.from_spec(String("native"), String("af_heart"))
    var t0 = perf_counter_ns()
    var st = wait_ready(svc, 120)
    check(fails, st == SVC_READY, String("READY after ") + String((perf_counter_ns() - t0) // 1_000_000)
          + " ms (load + one warm reply on the worker: " + String(svc.load_ms()) + " ms)")

    # a 50 Hz caller
    svc.start(TEXT)
    var frames = 0
    var worst_us = 0
    var t1 = perf_counter_ns()
    while True:
        var f0 = perf_counter_ns()
        var done = svc.poll()
        var us = (perf_counter_ns() - f0) // 1000
        worst_us = max(worst_us, us)
        frames += 1
        if done:
            break
        if frames > 50 * 60:
            break
        _ = sleep_us(FRAME_US)
    var wall_ms = (perf_counter_ns() - t1) // 1_000_000
    var wav = svc.result_wav_bytes()
    check(fails, same(wav, ref_en), String("the service's WAV equals the direct one (") + String(len(wav)) + " bytes)")
    check(fails, worst_us < 2000, String("worst poll ") + String(worst_us) + " us over " + String(frames)
          + " frames (" + String(wall_ms) + " ms; the model's own " + String(svc.latency_ms()) + " ms)")

    # cancel, then a new call (another voice)
    svc.start(TEXT)
    svc.cancel()
    svc.voice = String("ff_siwis")
    svc.start(TEXT2)
    while not svc.poll(10):
        pass
    var wav2 = svc.result_wav_bytes()
    check(fails, same(wav2, ref_fr), "after a cancel, the reply is the new call's (French, ff_siwis)")

    # a checkpoint that is not there
    var bad = KokoroService.from_spec(String("native:/nonexistent/kokoro"), String("af_heart"))
    var st2 = wait_ready(bad, 60)
    check(fails, st2 == SVC_FAILED, "a missing checkpoint ends in FAILED")
    bad.start(String("Hello."))
    while not bad.poll(10):
        pass
    var why = String("")
    try:
        _ = bad.result_wav_bytes()
    except e:
        why = String(e)
    check(fails, why.find("failed to load") >= 0, String("and says why: ") + why)

    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")

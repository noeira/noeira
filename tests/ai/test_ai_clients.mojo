# +--------------------------------------------------------------------------+ #
# | The AI clients, against a mock server that speaks each wire format
# +--------------------------------------------------------------------------+ #
"""Gate `noeira/ai/` (chat, jev, speech, transport) and `noeira/io/wav.mojo`.

    pixi run build-http                       # ONCE
    pixi run mojo run -I . tests/ai/test_ai_clients.mojo

Hermetic: `tools/ai/mock_ai_server.py` runs on a loopback port. No key, no
network, no spend. The fixture REFUSES a request with the wrong shape (a 400
naming the fault), so every check below gates what the client SENDS as well
as what it parses.

⚠ THE POINT IS THE INVARIANTS A LIVE API ONLY ENFORCES LATER:

* **An Anthropic assistant turn goes back verbatim** — its `thinking` block
  included. A turn rebuilt from text would pass every single-turn test and
  400 (or silently lose reasoning) on the second turn of a real agent.
* **Parallel tool results share ONE user message** for Anthropic, and are
  one `role: tool` message each for OpenAI.
* **No `temperature` reaches a current Claude model** (it is a 400 there).
* **Tool-call arguments round-trip as JSON text** — an integral number stays
  an integer (`"k":2`, not `2.0`).
* **A 529 is retried; a 422 is not**, and the 422's body reaches the caller.
* **WAV survives encode → multipart → the server's own parser**, resampled
  and down-mixed to 16 kHz mono on the way.
"""

from std.os.path import exists
from std.time import sleep

from noeira.ai.chat import (
    ChatClient, Conversation, PROVIDER_ANTHROPIC, PROVIDER_OPENAI,
)
from noeira.ai.jev import JevClient, JevQuestions
from noeira.ai.speech import STT_MULTIPART, STT_RAW, SpeechToText, TextToSpeech
from noeira.io.fileio import remove_file
from noeira.io.http import HttpClient, http_shim_available
from noeira.io.json import parse_json
from noeira.io.proc import run_capture
from noeira.io.wav import WavAudio, decode_wav, encode_wav


comptime PORT_FILE = "/tmp/noeira_ai_gate_port"


def _start_server() raises -> String:
    try:
        remove_file(String(PORT_FILE))
    except:
        pass
    _ = run_capture(
        "python3 tools/ai/mock_ai_server.py " + String(PORT_FILE)
        + " 120 > /tmp/noeira_ai_gate_server.log 2>&1 &"
    )
    for _ in range(100):
        if exists(PORT_FILE):
            var f = open(String(PORT_FILE), "r")
            var port = String(f.read().strip())
            f.close()
            if port.byte_length() > 0:
                return "http://127.0.0.1:" + port
        sleep(0.1)
    raise Error("the mock server never started — see /tmp/noeira_ai_gate_server.log")


def _check(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAIL: " + what)


comptime POSE_SCHEMA = (
    '{"type":"object","properties":{"name":{"type":"string"},'
    '"k":{"type":"number"}},"required":["name"]}'
)


def _agent_round_trip(mut client: ChatClient, label: String) raises -> Int:
    """Two turns: image + tools -> two parallel tool calls -> results -> text."""
    var conv = Conversation("You are a robot planner.")
    conv.tool("get_pose", "Pose of a named object", String(POSE_SCHEMA))
    var png: List[UInt8] = [0x89, 0x50, 0x4E, 0x47]
    conv.user_image("Where are the objects?", png)
    var r1 = conv.send(client)
    _check(r1.wants_tools(), label + ": first turn asked for no tool")
    _check(len(r1.tool_calls) == 2, label + ": expected 2 parallel tool calls")
    _check(r1.text == "I see 1 image(s).", label + ": text/image lost: " + r1.text)
    _check(
        r1.tool_calls[0].arguments_json == '{"name":"cube","k":2}',
        label + ": arguments did not round-trip: " + r1.tool_calls[0].arguments_json,
    )
    var args = r1.tool_calls[1].args()
    _check(args.number(args.field(args.root(), "k")) == 0.5, label + ": args() parse")
    _check(r1.input_tokens == 40 and r1.output_tokens == 30, label + ": usage")
    for i in range(len(r1.tool_calls)):
        var call = r1.tool_calls[i].copy()
        conv.tool_result(call.id, call.name + "=" + String(i))
    var r2 = conv.send(client)
    _check(not r2.wants_tools(), label + ": second turn still wants tools")
    _check(
        r2.text == "done: get_pose=0 | get_pose=1",
        label + ": tool results mis-grouped: " + r2.text,
    )
    _check(len(conv.messages) == 5, label + ": history length")
    print("  " + label + ": 2-turn tool agent, image, parallel calls   ok")
    return 9


def main() raises:
    print("=== ai clients ===")
    if not http_shim_available():
        raise Error("the HTTP shim is not built — run `pixi run build-http` first")
    var base = _start_server()
    print("  fixture at " + base)
    var checks = 0

    # ── 1. Claude: verbatim replay, grouped tool results, no temperature ──
    var claude = ChatClient(
        PROVIDER_ANTHROPIC, base + "/v1", String("claude-opus-5-5"), String("test-key")
    )
    checks += _agent_round_trip(claude, String("anthropic"))
    var probe = HttpClient()
    var last = parse_json(probe.get(base + "/__last", 200).take_body())
    var hdrs = last.field(last.root(), "headers")
    _check(
        last.string(last.field(hdrs, "anthropic-beta")) == "server-side-fallback-2026-07-01",
        "refusal-fallback beta header missing",
    )
    checks += 1

    # ── 2. the same agent over the OpenAI wire format ────────────────────
    var oai = ChatClient(
        PROVIDER_OPENAI, base + "/v1", String("qwen3-vl"), String("test-key")
    )
    oai.temperature = 0.2
    checks += _agent_round_trip(oai, String("openai"))

    # ── 3. Jev: every question type, answers + confidence ────────────────
    var jev = JevClient(String("test-key"), url=base + "/v1/systemone")
    var q = JevQuestions()
    q.noul("holding", "Is the gripper holding the cube?")
    q.choice(
        "next", "What should the arm do next?",
        ["grasp", "lift", "place"],
        ["close the jaws", "raise the cube", "lower it onto the target"],
    )
    q.score("risk", "Collision risk?", ["none", "low", "high"])
    var a = jev.decide('{"cube_in_jaws": true, "gap_mm": 12}', q)
    _check(a.model == "jev-1.13.0", "jev model")
    _check(a.noul("holding") == 0.93, "jev noul")
    _check(a.choice("next") == "lift", "jev choice")
    _check(a.confidence("next") == 0.85, "jev confidence")
    _check(a.probability("next", "grasp") == 0.1, "jev probability")
    _check(a.score("risk") == 1.4, "jev score")
    _check(a.probability("risk", "1") == 0.6, "jev score level probability")
    _check(a.confidence("holding") == 0.93, "jev noul confidence = max(p, 1-p)")
    _check(a.input_tokens == 210, "jev usage")
    var t = jev.decide_text("My card was charged twice.", q)
    _check(t.noul("holding") == 0.93, "jev text state")
    last = parse_json(probe.get(base + "/__last", 200).take_body())
    var sent = String(last.string(last.field(last.root(), "body")))
    _check('"state":"My card was charged twice."' in sent, "text state not a JSON string")
    checks += 11
    print("  jev: noul / choice / score, JSON + text state          ok")

    # ── 4. retry what is retryable, surface what is not ─────────────────
    var flaky = JevClient(String("test-key"), url=base + "/flaky/v1/systemone")
    flaky.retries = 2
    var fa = flaky.decide("{}", q)
    _check(fa.choice("next") == "lift", "529 x2 then 200 was not retried through")
    var bad = JevClient(String("test-key"), url=base + "/bad/v1/systemone")
    var msg = String("")
    try:
        _ = bad.decide("{}", q)
    except e:
        msg = String(e)
    _check("422" in msg and "questions: field required" in msg, "422 body lost: " + msg)
    checks += 2
    print("  transport: 529 retried, 422 raised with the body        ok")

    # ── 5. WAV codec + both STT upload shapes + TTS ──────────────────────
    var samples = List[Int16]()
    for i in range(48000):  # 0.5 s of 48 kHz stereo
        samples.append(Int16((i * 13) % 4000 - 2000))
    var stereo48 = WavAudio(48000, 2, samples^)
    var wav_bytes = encode_wav(stereo48)
    var back = decode_wav(wav_bytes)
    _check(back.sample_rate == 48000 and back.channels == 2, "wav header round-trip")
    _check(back.samples == stereo48.samples, "wav samples round-trip")
    _check(back.frames() == 24000, "wav frames")

    var stt = SpeechToText(
        STT_MULTIPART, base + "/v1/audio/transcriptions", String("whisper-1"), String("k")
    )
    var tr = stt.transcribe(stereo48)
    _check(tr.text == "8000@16000", "multipart STT (48k stereo -> 16k mono): " + tr.text)
    var hf = SpeechToText(STT_RAW, base + "/hf/whisper", String("w"), String("k"))
    var tr2 = hf.transcribe(stereo48)
    _check(tr2.text == "8000@16000", "raw-body STT: " + tr2.text)
    var tts = TextToSpeech(base + "/v1/audio/speech", String("m"), String("v"), String("k"))
    var spoken = tts.speak("Cube placed.")
    _check(spoken.sample_rate == 16000 and spoken.frames() == 4000, "TTS WAV decode")
    checks += 6
    print("  speech: wav codec, multipart + raw STT, TTS            ok")

    _ = probe.get(base + "/__shutdown")
    print("=== ai clients: " + String(checks) + " checks passed ===")

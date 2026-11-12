# +--------------------------------------------------------------------------+ #
# | Lfm3ChatService: the robot's spoken replies through the worker thread
# +--------------------------------------------------------------------------+ #
"""The chat service the voice loop drives (`noeira.llm.lfm3.service`), against
the direct API it wraps (docs/NN_MODEL_PORTS_PLAN.md §12.7 step 4).

    pixi run -e nvidia mojo build -D LFM3_BF16=1 -I . tests/llm/test_lfm3_chat_service_gpu.mojo -o build/t
    ./build/t [LFM3-1.7B dir]

1. Driven as `G1VoiceLoopOf` drives it: `start`, `poll` + `done` every 10 ms,
   `result`. The reply must EQUAL what `Lfm3Chat` gives directly with the same
   settings (int8, keep_system, Liquid's sampler and seed): the request's
   framing, the template and the history replay are the service's only work,
   and this is where a mistake in them would show.
2. A history (user, assistant, user) and a second request with the SAME system
   prompt (the service rewinds to it) — both equal to the direct API's.
3. What it refuses comes back from `result` (tools at `start`; a
   conversation ending with an assistant message from the worker), and the
   next request is served.

Prints each reply's latency and the time to its first token.
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.ai.chat import ChatMessage, ChatService, ToolSpec
from noeira.core.concurrent.thread import sleep_us
from noeira.llm.lfm3 import Lfm3Chat, ChatSettings
from noeira.llm.lfm3.service import Lfm3ChatService


comptime SYSTEM = (
    "You are a small humanoid robot in a furnished room with a fridge, a door, a table and a sofa. Reply in ONE short"
    " sentence, in the language you were addressed in. You can walk to those places, turn, squat and raise your arms,"
    " but you cannot pick things up, open the door or sit down. Be warm and brief."
)


def check(mut fails: Int, ok: Bool, what: String):
    if ok:
        print("  PASS ", what)
    else:
        print("  FAIL ", what)
        fails += 1


def drive(mut svc: Lfm3ChatService, ref msgs: List[ChatMessage], system: String) raises -> String:
    """As the voice loop does: start, then poll + done once a 'frame' (10 ms)."""
    var t0 = perf_counter_ns()
    svc.start(msgs, system, List[ToolSpec](), False)
    var frames = 0
    while True:
        _ = svc.poll()
        if svc.done():
            break
        _ = sleep_us(10_000)
        frames += 1
    var r = svc.result()
    print("        ", Int(Float64(perf_counter_ns() - t0) / 1e6), "ms wall,", frames, "frames,", r.input_tokens, "tokens in,",
          r.output_tokens, "out, first token after", Int(r.first_token_ms), "ms,", r.stop_reason, ": '" + r.text + "'")
    return r.text


def direct(mut chat: Lfm3Chat, ref msgs: List[ChatMessage], system: String, max_tokens: Int) raises -> String:
    chat.reset(system)
    for ref m in msgs:
        if m.role == "user":
            chat.user(m.text)
        else:
            chat.assistant(m.text)
    var s = ChatSettings()
    s.max_tokens = max_tokens
    chat.start(s)
    while chat.next():
        pass
    return chat.reply_text()


def main() raises:
    var args = argv()
    var dir = String(args[1]) if len(args) > 1 else getenv("HOME") + "/.cache/noeira/local-ai/lfm3/LFM3-1.7B"
    var fails = 0
    print("Lfm3ChatService (int8) vs Lfm3Chat")
    var t0 = perf_counter_ns()
    var svc = Lfm3ChatService.from_spec(String("native:") + dir)
    svc.set_reply_limits(120)
    var chat = Lfm3Chat.load(DeviceContext(), dir, 4096, True)
    chat.keep_system = True

    var a = List[ChatMessage]()
    a.append(ChatMessage.user(String("Bonjour ! Qu'est-ce que tu peux faire pour moi ?")))
    var got = drive(svc, a, SYSTEM)
    print("  service ready + first reply after", Int(Float64(perf_counter_ns() - t0) / 1e6), "ms (load", svc.load_ms(), "ms)")
    check(fails, got.byte_length() > 0 and got == direct(chat, a, SYSTEM, 120), "a reply, equal to the direct API's")

    var b = List[ChatMessage]()
    b.append(ChatMessage.user(String("Can you bring me a soda from the fridge?")))
    b.append(ChatMessage.assistant(String("I can't carry things, but I can walk to the fridge with you.")))
    b.append(ChatMessage.user(String("Okay, then where is the sofa?")))
    got = drive(svc, b, SYSTEM)
    check(fails, got.byte_length() > 0 and got == direct(chat, b, SYSTEM, 120), "a history (user, assistant, user), equal to the direct API's")

    var c = List[ChatMessage]()
    c.append(ChatMessage.user(String("Tu peux t'asseoir sur le canapé ?")))
    got = drive(svc, c, SYSTEM)  # the same system prompt again: the service rewinds to it
    check(fails, got.byte_length() > 0 and got == direct(chat, c, SYSTEM, 120), "the system prompt reused (rewound), equal to the direct API's")

    var other = String("You are a pirate. Answer in one short sentence.")
    got = drive(svc, a, other)
    check(fails, got.byte_length() > 0 and got == direct(chat, a, other, 120), "another system prompt, equal to the direct API's")

    # refusals
    var tools = List[ToolSpec]()
    tools.append(ToolSpec(String("walk"), String("walk somewhere"), String("{}")))
    var refused = False
    try:
        svc.start(a, SYSTEM, tools, False)
    except:
        refused = True
    check(fails, refused, "tools are refused at start")
    var bad = List[ChatMessage]()
    bad.append(ChatMessage.user(String("Hello")))
    bad.append(ChatMessage.assistant(String("Hi!")))
    svc.start(bad, SYSTEM, List[ToolSpec](), False)
    while not svc.done():
        _ = svc.poll(5)
    var raised = False
    try:
        _ = svc.result()
    except e:
        raised = True
        print("        (refused: " + String(e) + ")")
    check(fails, raised, "a conversation ending with the assistant fails at result")
    got = drive(svc, a, SYSTEM)
    check(fails, got.byte_length() > 0, "and the next request is served")

    print("")
    if fails > 0:
        print("FAILED:", fails)
        raise Error("lfm3 chat service gate failed")
    print("ALL PASS")

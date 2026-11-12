# +--------------------------------------------------------------------------+ #
# | Chat with LFM3-1.7B in the terminal, in this process (no server, no Python)
# +--------------------------------------------------------------------------+ #
"""Type a line, read the robot's reply as it is generated; an empty line (or end of input) quits.
The conversation stays on one cache, so a turn costs its own tokens only.

    pixi run -e nvidia mojo build -D LFM3_BF16=1 -D LFM3_INT8=1 -I . examples/ai/lfm3_chat.mojo -o build/lfm3_chat
    ./build/lfm3_chat [LFM3-1.7B dir] [--greedy] [--think]

`-D LFM3_INT8=1`: int8 weights (1.7 GB, ~41 tokens/s on the Orin; without it bf16,
3.5 GB, ~24 tokens/s).

Liquid's chat settings (temperature 0.1, top_k 50, repetition penalty 1.1),
thinking off (`--think`: on, one turn only — the template drops old thinking).
The default folder is where the Orin keeps it: ~/.cache/noeira/local-ai/lfm3/LFM3-1.7B.
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.llm.lfm3 import Lfm3Chat, ChatSettings


comptime SYSTEM = (
    "You are Noé, a small humanoid robot in a living room. You speak with the people around you, briefly and "
    "kindly, in the language they use. Room: the fridge is 3.1 m ahead, the door 3.8 m to the left, the table "
    "2.0 m to the right, the sofa 4.5 m behind. You are standing in the middle of the room."
)


def main() raises:
    var args = argv()
    var dir = getenv("HOME") + "/.cache/noeira/local-ai/lfm3/LFM3-1.7B"
    var settings = ChatSettings()
    var think = False
    for i in range(1, len(args)):
        var a = String(args[i])
        if a == "--greedy":
            settings.greedy = True
        elif a == "--think":
            think = True
        else:
            dir = a
    var chat = Lfm3Chat.load(DeviceContext(), dir)
    chat.thinking = think
    chat.reset(SYSTEM)
    print("LFM3-1.7B ready (empty line to quit)")
    while True:
        var line: String
        try:
            line = input("> ")
        except:
            break  # end of input
        if line.byte_length() == 0:
            break
        chat.user(line)
        var t0 = perf_counter_ns()
        chat.start(settings)
        var first_ms = Float64(perf_counter_ns() - t0) / 1e6
        var shown = 0
        while True:
            var text = chat.reply_text()
            # print what is new, holding back a character cut between two tokens
            if text.byte_length() > shown and not text.endswith("�"):
                print(String(text[byte=shown:]), end="", flush=True)
                shown = text.byte_length()
            if not chat.next():
                break
        var ms = Float64(perf_counter_ns() - t0) / 1e6
        print(
            "\n  [" + String(len(chat.reply)) + " tokens, first after " + String(Int(first_ms)) + " ms, "
            + String(Int(Float64(len(chat.reply)) / (ms / 1000.0))) + " tokens/s, " + String(chat.cache.pos)
            + " in the conversation]"
        )

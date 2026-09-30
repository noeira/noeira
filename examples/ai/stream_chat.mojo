# +--------------------------------------------------------------------------+ #
# | Tokens as they arrive
# +--------------------------------------------------------------------------+ #
"""Stream a chat answer to the terminal, then report time-to-first-token.

    pixi run mojo run -I . examples/ai/stream_chat.mojo hf "Explain PD control in 5 lines."
    pixi run mojo run -I . examples/ai/stream_chat.mojo anthropic

First argument: a `ChatClient.from_spec` string. The perceived latency of a
streamed answer is `first_token_ms`, not the total — that is the number to
watch for an interactive demo.
"""

from std.sys import argv

from noeira.ai.chat import ChatClient, Conversation


def main() raises:
    var args = argv()
    var spec = String(args[1]) if len(args) > 1 else String("hf")
    var prompt = (
        String(args[2]) if len(args) > 2
        else String("Explain in five short lines how a PD controller holds a robot joint.")
    )
    var llm = ChatClient.from_spec(spec)
    llm.max_tokens = 1024
    if spec.startswith("hf"):
        llm.extra("chat_template_kwargs", '{"enable_thinking": false}')
    var conv = Conversation("Be concise.")
    conv.user(prompt)
    conv.start(llm)
    while not llm.done():
        var d = llm.poll(50)  # waits <= 50 ms for bytes: a CLI loop, not a render loop
        if d.byte_length() > 0:
            print(d, end="", flush=True)
    var r = conv.finish(llm)
    print()
    print(
        "--", r.model, "| first token", r.first_token_ms, "ms | total",
        r.latency_ms, "ms |", r.output_tokens, "tokens |", r.stop_reason,
    )

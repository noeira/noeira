# +--------------------------------------------------------------------------+ #
# | One question to any chat model
# +--------------------------------------------------------------------------+ #
"""Ask one question to any chat model.

    pixi run mojo run -I . examples/ai/chat_ask.mojo anthropic "Name three grasp types."
    pixi run mojo run -I . examples/ai/chat_ask.mojo hf:Qwen/Qwen3.5-9B "..."
    pixi run mojo run -I . examples/ai/chat_ask.mojo ollama:llama3.2 "..."

The first argument is a `ChatClient.from_spec` string (see `noeira/ai/chat.mojo`).
"""

from std.sys import argv

from noeira.ai.chat import ChatClient, ChatMessage


def main() raises:
    var args = argv()
    var spec = String(args[1]) if len(args) > 1 else String("anthropic")
    var prompt = (
        String(args[2]) if len(args) > 2
        else String("In two sentences: why is contact-rich manipulation hard to simulate?")
    )
    var llm = ChatClient.from_spec(spec)
    llm.max_tokens = 1024
    if spec.startswith("hf"):
        # Qwen3.x thinks before answering by default: seconds of hidden
        # tokens that count against max_tokens. Off for a quick answer.
        llm.extra("chat_template_kwargs", '{"enable_thinking": false}')
    var msgs = List[ChatMessage]()
    msgs.append(ChatMessage.user(prompt))
    var r = llm.chat(msgs, String("Answer briefly."))
    print(r.text)
    print(
        "--", r.model, "|", r.input_tokens, "in /", r.output_tokens, "out |",
        r.latency_ms, "ms |", r.stop_reason,
    )

# +--------------------------------------------------------------------------+ #
# | Chat — LLMs and VLMs behind one client, over HTTP
# +--------------------------------------------------------------------------+ #
"""One chat client for hosted and local models.

    var llm = ChatClient.anthropic()                  # Claude, ANTHROPIC_API_KEY
    print(llm.ask("Name three grasp types."))

    var conv = Conversation("You are a robot's planner.")
    conv.user_image("What is on the table?", png_bytes)
    var r = conv.send(llm)
    print(r.text)

Two wire formats cover the field:

| Factory | Wire format | Key |
|---|---|---|
| `anthropic()` | Anthropic Messages | `ANTHROPIC_API_KEY` |
| `openai(model)` | OpenAI Chat Completions | `OPENAI_API_KEY` |
| `huggingface(model)` | OpenAI-compatible (HF router) | `HF_TOKEN` |
| `groq(model)` / `openrouter(model)` | OpenAI-compatible | `GROQ_API_KEY` / `OPENROUTER_API_KEY` |
| `ollama(model)` | OpenAI-compatible, LOCAL | none |
| `openai_compatible(url, model)` | llama.cpp `llama-server`, vLLM, LM Studio, … | optional |

⚠ THE OPENAI-COMPATIBLE ROW IS THE MIGRATION PATH TO LOCAL MODELS. Ollama,
llama.cpp's server, vLLM and LM Studio all speak `/v1/chat/completions`, so
moving a demo off a hosted API is a change of factory, not of code.

## Tools (agents)

`ToolSpec(name, description, schema_json)` declares a function; a response
whose `tool_calls` is non-empty asks for them. Run each, append
`conv.tool_result(call.id, output)`, and `send` again — see
`examples/ai/tool_agent.mojo` for the loop. Arguments arrive as one JSON
string (`call.arguments_json`); `call.args()` parses it.

## Things that are easy to get wrong, handled here

* ⚠ AN ASSISTANT TURN IS REPLAYED AS THE SERVER SENT IT (Anthropic). Current
  Claude models return `thinking` blocks alongside text and tool calls, and
  those must come back unchanged in the next request; a turn rebuilt from
  its text alone silently drops them. `ChatResponse.message` keeps the raw
  content array and `Conversation` appends that.
* ⚠ ANTHROPIC WANTS ALL TOOL RESULTS OF ONE TURN IN ONE USER MESSAGE; OpenAI
  wants one `role: tool` message each. `Conversation` stores one message per
  result and the serialiser groups them for Anthropic.
* ⚠ NO `temperature` ON CURRENT CLAUDE MODELS — `claude-opus-5-5` rejects
  sampling parameters with a 400. It is sent only when set (>= 0), which is
  what the OpenAI-compatible servers want. `effort` is the Claude knob.
* Server-side refusal fallback (`fallbacks: "default"`) is ON for Anthropic
  by default: on a policy decline the API re-runs the request on a fallback
  model inside the same call. `stop_reason == "refusal"` means the whole
  chain declined. Set `client.fallbacks = False` to turn it off.

Not here yet: streaming (SSE). Every call blocks until the answer is
complete — see the package docstring.
"""

from noeira.io.base64 import b64_encode
from noeira.io.http import HttpClient
from noeira.io.json import J_STRING, JsonDoc, JsonWriter, dump_json, parse_json

from noeira.ai.keys import api_key, find_api_key
from noeira.ai.transport import post_json_api


comptime PROVIDER_ANTHROPIC = 0
comptime PROVIDER_OPENAI = 1
"""OpenAI Chat Completions and everything compatible with it."""

comptime DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"


# ═══════════════════════════════════════════════════════════════════════════
# Messages
# ═══════════════════════════════════════════════════════════════════════════


@fieldwise_init
struct ToolSpec(Copyable, Movable):
    """A function the model may call. `schema_json` is a JSON Schema object:
    `{"type":"object","properties":{...},"required":[...]}`."""

    var name: String
    var description: String
    var schema_json: String


@fieldwise_init
struct ToolCall(Copyable, Movable):
    var id: String
    var name: String
    var arguments_json: String

    def args(self) raises -> JsonDoc:
        """The arguments, parsed. `doc.field(doc.root(), "x")` reads one."""
        var b = List[UInt8](capacity=self.arguments_json.byte_length())
        var src = self.arguments_json.as_bytes()
        for i in range(self.arguments_json.byte_length()):
            b.append(src[i])
        return parse_json(b^)


struct ChatMessage(Copyable, Movable):
    var role: String
    """"user", "assistant", or "tool" (a tool result)."""
    var text: String
    var images_b64: List[String]
    var image_types: List[String]
    """Media type per image: "image/png", "image/jpeg"."""
    var tool_calls: List[ToolCall]
    """Assistant turns only."""
    var tool_call_id: String
    """Tool results only: which call this answers."""
    var is_error: Bool
    """Tool results only: the tool failed and `text` says why."""
    var raw: String
    """The provider's own JSON for this turn, replayed verbatim when the
    request goes back to the same provider. Empty for turns built here."""
    var raw_provider: Int

    def __init__(out self, var role: String, var text: String):
        self.role = role^
        self.text = text^
        self.images_b64 = List[String]()
        self.image_types = List[String]()
        self.tool_calls = List[ToolCall]()
        self.tool_call_id = String("")
        self.is_error = False
        self.raw = String("")
        self.raw_provider = -1

    @staticmethod
    def user(var text: String) -> ChatMessage:
        return ChatMessage(String("user"), text^)

    @staticmethod
    def assistant(var text: String) -> ChatMessage:
        return ChatMessage(String("assistant"), text^)

    @staticmethod
    def tool_result(
        var call_id: String, var content: String, is_error: Bool = False
    ) -> ChatMessage:
        var m = ChatMessage(String("tool"), content^)
        m.tool_call_id = call_id^
        m.is_error = is_error
        return m^

    def add_image(mut self, ref data: List[UInt8], var media_type: String = String("image/png")) raises:
        """Attach an encoded image (PNG/JPEG bytes — `noeira.io.png.encode_png`
        turns a camera frame into one)."""
        self.images_b64.append(b64_encode(data))
        self.image_types.append(media_type^)


struct ChatResponse(Movable):
    var text: String
    """All text blocks, concatenated."""
    var tool_calls: List[ToolCall]
    var stop_reason: String
    """As the provider spells it: "end_turn" / "tool_use" / "max_tokens" /
    "refusal" (Anthropic), "stop" / "tool_calls" / "length" (OpenAI)."""
    var model: String
    """The model that actually answered — can differ from the one asked for
    (a fallback, an alias resolved server-side)."""
    var input_tokens: Int
    var output_tokens: Int
    var latency_ms: Float64
    var message: ChatMessage
    """This turn as a message, ready to append to the history."""

    def __init__(out self):
        self.text = String("")
        self.tool_calls = List[ToolCall]()
        self.stop_reason = String("")
        self.model = String("")
        self.input_tokens = 0
        self.output_tokens = 0
        self.latency_ms = 0.0
        self.message = ChatMessage.assistant(String(""))

    def __init__(out self, *, deinit move: Self):
        self.text = move.text^
        self.tool_calls = move.tool_calls^
        self.stop_reason = move.stop_reason^
        self.model = move.model^
        self.input_tokens = move.input_tokens
        self.output_tokens = move.output_tokens
        self.latency_ms = move.latency_ms
        self.message = move.message^

    def wants_tools(self) -> Bool:
        return len(self.tool_calls) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Request bodies
# ═══════════════════════════════════════════════════════════════════════════


def _anthropic_messages(mut w: JsonWriter, ref msgs: List[ChatMessage]) raises:
    w.key("messages")
    w.begin_array()
    var i = 0
    while i < len(msgs):
        ref m = msgs[i]
        if m.role == "tool":
            # Every consecutive tool result goes into ONE user message.
            w.begin_object()
            w.member("role", String("user"))
            w.key("content")
            w.begin_array()
            while i < len(msgs) and msgs[i].role == "tool":
                w.begin_object()
                w.member("type", String("tool_result"))
                w.member("tool_use_id", msgs[i].tool_call_id)
                w.member("content", msgs[i].text)
                if msgs[i].is_error:
                    w.key("is_error")
                    w.boolean(True)
                w.end_object()
                i += 1
            w.end_array()
            w.end_object()
            continue
        w.begin_object()
        w.member("role", m.role)
        w.key("content")
        if m.raw_provider == PROVIDER_ANTHROPIC and m.raw.byte_length() > 0:
            w.raw(m.raw)
        else:
            w.begin_array()
            for k in range(len(m.images_b64)):
                w.begin_object()
                w.member("type", String("image"))
                w.key("source")
                w.begin_object()
                w.member("type", String("base64"))
                w.member("media_type", m.image_types[k])
                w.member("data", m.images_b64[k])
                w.end_object()
                w.end_object()
            if m.text.byte_length() > 0:
                w.begin_object()
                w.member("type", String("text"))
                w.member("text", m.text)
                w.end_object()
            for k in range(len(m.tool_calls)):
                w.begin_object()
                w.member("type", String("tool_use"))
                w.member("id", m.tool_calls[k].id)
                w.member("name", m.tool_calls[k].name)
                w.key("input")
                w.raw(m.tool_calls[k].arguments_json)
                w.end_object()
            w.end_array()
        w.end_object()
        i += 1
    w.end_array()


def _openai_messages(
    mut w: JsonWriter, system: String, ref msgs: List[ChatMessage]
) raises:
    w.key("messages")
    w.begin_array()
    if system.byte_length() > 0:
        w.begin_object()
        w.member("role", String("system"))
        w.member("content", system)
        w.end_object()
    for i in range(len(msgs)):
        ref m = msgs[i]
        w.begin_object()
        w.member("role", m.role)
        if m.role == "tool":
            w.member("tool_call_id", m.tool_call_id)
            w.member("content", m.text)
        elif len(m.images_b64) > 0:
            w.key("content")
            w.begin_array()
            if m.text.byte_length() > 0:
                w.begin_object()
                w.member("type", String("text"))
                w.member("text", m.text)
                w.end_object()
            for k in range(len(m.images_b64)):
                w.begin_object()
                w.member("type", String("image_url"))
                w.key("image_url")
                w.begin_object()
                w.member(
                    "url", "data:" + m.image_types[k] + ";base64," + m.images_b64[k]
                )
                w.end_object()
                w.end_object()
            w.end_array()
        else:
            w.key("content")
            if m.text.byte_length() == 0 and len(m.tool_calls) > 0:
                w.null()
            else:
                w.string(m.text)
        if len(m.tool_calls) > 0:
            w.key("tool_calls")
            w.begin_array()
            for k in range(len(m.tool_calls)):
                w.begin_object()
                w.member("id", m.tool_calls[k].id)
                w.member("type", String("function"))
                w.key("function")
                w.begin_object()
                w.member("name", m.tool_calls[k].name)
                w.member("arguments", m.tool_calls[k].arguments_json)
                w.end_object()
                w.end_object()
            w.end_array()
        w.end_object()
    w.end_array()


# ═══════════════════════════════════════════════════════════════════════════
# Client
# ═══════════════════════════════════════════════════════════════════════════


struct ChatClient(Movable):
    var provider: Int
    var base_url: String
    """Up to and including the version segment: `https://api.anthropic.com/v1`,
    `http://localhost:11434/v1`."""
    var model: String
    var max_tokens: Int
    var temperature: Float64
    """Sent only when >= 0. Leave unset for current Claude models."""
    var effort: String
    """Anthropic `output_config.effort` ("low" … "max"); "" = model default.
    `low` is the latency knob for an interactive demo."""
    var fallbacks: Bool
    """Anthropic server-side refusal fallback (`fallbacks: "default"`)."""
    var retries: Int
    var _extra_keys: List[String]
    var _extra_values: List[String]
    var _http: HttpClient

    def __init__(
        out self,
        provider: Int,
        var base_url: String,
        var model: String,
        var key: String,
        timeout_ms: Int = 300000,
    ) raises:
        self.provider = provider
        self.base_url = base_url^
        self.model = model^
        self.max_tokens = 4096
        self.temperature = -1.0
        self.effort = String("")
        self.fallbacks = provider == PROVIDER_ANTHROPIC
        self.retries = 2
        self._extra_keys = List[String]()
        self._extra_values = List[String]()
        self._http = HttpClient(timeout_ms, 10000)
        if provider == PROVIDER_ANTHROPIC:
            self._http.header(String("x-api-key"), key)
            self._http.header(String("anthropic-version"), String("2023-06-01"))
        elif key.byte_length() > 0:
            self._http.bearer(key)

    def __init__(out self, *, deinit move: Self):
        self.provider = move.provider
        self.base_url = move.base_url^
        self.model = move.model^
        self.max_tokens = move.max_tokens
        self.temperature = move.temperature
        self.effort = move.effort^
        self.fallbacks = move.fallbacks
        self.retries = move.retries
        self._extra_keys = move._extra_keys^
        self._extra_values = move._extra_values^
        self._http = move._http^

    def extra(mut self, var key: String, var json_value: String) raises:
        """Add a top-level request field this client does not model — the
        provider-specific knobs. `json_value` is JSON text, spliced as is:

            llm.extra("reasoning_effort", '"low"')              # OpenAI-style
            llm.extra("chat_template_kwargs", '{"enable_thinking": false}')
            llm.extra("thinking", '{"type": "adaptive", "display": "summarized"}')

        Setting a key again replaces it. ⚠ A key the client also writes
        (`model`, `messages`, …) would be sent TWICE; that is refused."""
        for name in ["model", "max_tokens", "messages", "system", "tools", "temperature"]:
            if key == name:
                raise Error("chat: '" + key + "' is set by the client, not extra()")
        for i in range(len(self._extra_keys)):
            if self._extra_keys[i] == key:
                self._extra_values[i] = json_value^
                return
        self._extra_keys.append(key^)
        self._extra_values.append(json_value^)

    # ── factories ─────────────────────────────────────────────────────

    @staticmethod
    def anthropic(
        var model: String = String(DEFAULT_CLAUDE_MODEL),
        var base_url: String = String("https://api.anthropic.com/v1"),
    ) raises -> ChatClient:
        var key = api_key(["ANTHROPIC_API_KEY"])
        return ChatClient(PROVIDER_ANTHROPIC, base_url^, model^, key^)

    @staticmethod
    def openai(var model: String) raises -> ChatClient:
        var key = api_key(["OPENAI_API_KEY"])
        return ChatClient(
            PROVIDER_OPENAI, String("https://api.openai.com/v1"), model^, key^
        )

    @staticmethod
    def huggingface(var model: String) raises -> ChatClient:
        """Hugging Face Inference Providers — many open models (Qwen, Llama,
        Gemma, SmolVLM, …) behind one OpenAI-compatible router and your
        `HF_TOKEN`. Suffix the model with `:fastest` / `:cheapest` or a
        provider name to choose the backend."""
        var key = api_key(["HF_TOKEN", "HUGGINGFACE_API_KEY"])
        return ChatClient(
            PROVIDER_OPENAI, String("https://router.huggingface.co/v1"), model^, key^
        )

    @staticmethod
    def groq(var model: String) raises -> ChatClient:
        var key = api_key(["GROQ_API_KEY"])
        return ChatClient(
            PROVIDER_OPENAI, String("https://api.groq.com/openai/v1"), model^, key^
        )

    @staticmethod
    def openrouter(var model: String) raises -> ChatClient:
        var key = api_key(["OPENROUTER_API_KEY"])
        return ChatClient(
            PROVIDER_OPENAI, String("https://openrouter.ai/api/v1"), model^, key^
        )

    @staticmethod
    def ollama(
        var model: String, var host: String = String("http://localhost:11434")
    ) raises -> ChatClient:
        return ChatClient(PROVIDER_OPENAI, host + "/v1", model^, String(""))

    @staticmethod
    def openai_compatible(
        var base_url: String, var model: String, var key: String = String("")
    ) raises -> ChatClient:
        return ChatClient(PROVIDER_OPENAI, base_url^, model^, key^)

    @staticmethod
    def from_spec(spec: String) raises -> ChatClient:
        """`provider[:model]` — one string a demo can take as a flag.

            anthropic                     anthropic:claude-haiku-4-5
            openai:<model>                hf:Qwen/Qwen3.5-9B
            groq:<model>                  openrouter:<vendor>/<model>
            ollama:qwen2.5vl:7b           http://localhost:8080/v1:<model>

        ⚠ THE FIRST COLON SPLITS, so an Ollama tag keeps its own colon
        (`ollama:qwen2.5vl:7b`). A URL spec splits at the LAST colon instead,
        since the URL carries one (`http:`) before the model.
        """
        if spec.startswith("http://") or spec.startswith("https://"):
            var cut = spec.rfind(":")
            var url = String(spec[byte=:cut])
            if url.endswith("/"):
                var trimmed = String(url[byte=: url.byte_length() - 1])
                url = trimmed^
            return ChatClient.openai_compatible(
                url^, String(spec[byte=cut + 1 :]), find_api_key(["OPENAI_API_KEY"])
            )
        var c = spec.find(":")
        var provider = String(spec) if c < 0 else String(spec[byte=:c])
        var model = String("") if c < 0 else String(spec[byte=c + 1 :])
        if provider == "anthropic" or provider == "claude":
            if model.byte_length() == 0:
                model = String(DEFAULT_CLAUDE_MODEL)
            return ChatClient.anthropic(model^)
        if provider == "hf" or provider == "huggingface":
            if model.byte_length() == 0:
                model = String("Qwen/Qwen3.5-9B")
            return ChatClient.huggingface(model^)
        if model.byte_length() == 0:
            raise Error("chat spec '" + spec + "': name a model, e.g. " + provider + ":<model>")
        if provider == "openai":
            return ChatClient.openai(model^)
        if provider == "groq":
            return ChatClient.groq(model^)
        if provider == "openrouter":
            return ChatClient.openrouter(model^)
        if provider == "ollama":
            return ChatClient.ollama(model^)
        raise Error(
            "chat spec '" + spec + "': unknown provider (anthropic, openai, hf,"
            " groq, openrouter, ollama, or an http(s)://…/v1 URL)"
        )

    # ── requests ──────────────────────────────────────────────────────

    def request_body(
        self,
        ref messages: List[ChatMessage],
        system: String = String(""),
        tools: List[ToolSpec] = List[ToolSpec](),
    ) raises -> String:
        """The JSON this client would POST. Public so a test can pin it."""
        var w = JsonWriter()
        w.begin_object()
        w.member("model", self.model)
        w.member("max_tokens", self.max_tokens)
        if self.temperature >= 0.0:
            w.member("temperature", self.temperature)
        if self.provider == PROVIDER_ANTHROPIC:
            if system.byte_length() > 0:
                w.member("system", system)
            if self.effort.byte_length() > 0:
                w.key("output_config")
                w.begin_object()
                w.member("effort", self.effort)
                w.end_object()
            if self.fallbacks:
                w.member("fallbacks", String("default"))
            _anthropic_messages(w, messages)
        else:
            _openai_messages(w, system, messages)
        for i in range(len(self._extra_keys)):
            w.key(self._extra_keys[i])
            w.raw(self._extra_values[i])
        if len(tools) > 0:
            w.key("tools")
            w.begin_array()
            for i in range(len(tools)):
                w.begin_object()
                if self.provider == PROVIDER_ANTHROPIC:
                    w.member("name", tools[i].name)
                    w.member("description", tools[i].description)
                    w.key("input_schema")
                    w.raw(tools[i].schema_json)
                else:
                    w.member("type", String("function"))
                    w.key("function")
                    w.begin_object()
                    w.member("name", tools[i].name)
                    w.member("description", tools[i].description)
                    w.key("parameters")
                    w.raw(tools[i].schema_json)
                    w.end_object()
                w.end_object()
            w.end_array()
        w.end_object()
        return w.done()

    def chat(
        mut self,
        ref messages: List[ChatMessage],
        system: String = String(""),
        tools: List[ToolSpec] = List[ToolSpec](),
    ) raises -> ChatResponse:
        var body = self.request_body(messages, system, tools)
        if self.provider == PROVIDER_ANTHROPIC:
            if self.fallbacks:
                self._http.header(
                    String("anthropic-beta"), String("server-side-fallback-2026-07-01")
                )
            var reply = post_json_api(
                self._http, self.base_url + "/messages", body^,
                String("anthropic chat"), self.retries,
            )
            var r = parse_anthropic_response(reply.json())
            r.latency_ms = reply.latency_ms
            return r^
        var reply = post_json_api(
            self._http, self.base_url + "/chat/completions", body^,
            String("chat completions"), self.retries,
        )
        var r = parse_openai_response(reply.json())
        r.latency_ms = reply.latency_ms
        return r^

    def ask(mut self, var prompt: String, system: String = String("")) raises -> String:
        """One question, one answer, no history."""
        var msgs = List[ChatMessage]()
        msgs.append(ChatMessage.user(prompt^))
        var r = self.chat(msgs, system)
        return r.text^


# ═══════════════════════════════════════════════════════════════════════════
# Response parsing
# ═══════════════════════════════════════════════════════════════════════════


def _str_or_empty(ref doc: JsonDoc, node: Int) raises -> String:
    if doc.kind_of(node) == J_STRING:
        return doc.string(node)
    return String("")


def _int_or_zero(ref doc: JsonDoc, node: Int) raises -> Int:
    if node < 0 or doc.kind_of(node) != 2:  # J_NUMBER
        return 0
    return doc.integer(node)


def parse_anthropic_response(doc: JsonDoc) raises -> ChatResponse:
    var root = doc.root()
    var r = ChatResponse()
    r.model = _str_or_empty(doc, doc.field(root, "model"))
    r.stop_reason = _str_or_empty(doc, doc.field(root, "stop_reason"))
    var usage = doc.field(root, "usage")
    r.input_tokens = _int_or_zero(doc, doc.field(usage, "input_tokens"))
    r.output_tokens = _int_or_zero(doc, doc.field(usage, "output_tokens"))
    var content = doc.field(root, "content")
    if content < 0:
        raise Error("anthropic: response has no content array")
    for i in range(doc.size(content)):
        var block = doc.at(content, i)
        var kind = _str_or_empty(doc, doc.field(block, "type"))
        if kind == "text":
            r.text += doc.string(doc.field(block, "text"))
        elif kind == "tool_use":
            r.tool_calls.append(
                ToolCall(
                    doc.string(doc.field(block, "id")),
                    doc.string(doc.field(block, "name")),
                    dump_json(doc, doc.field(block, "input")),
                )
            )
    r.message = ChatMessage.assistant(r.text.copy())
    r.message.tool_calls = r.tool_calls.copy()
    r.message.raw = dump_json(doc, content)
    r.message.raw_provider = PROVIDER_ANTHROPIC
    return r^


def parse_openai_response(doc: JsonDoc) raises -> ChatResponse:
    var root = doc.root()
    var r = ChatResponse()
    r.model = _str_or_empty(doc, doc.field(root, "model"))
    var usage = doc.field(root, "usage")
    r.input_tokens = _int_or_zero(doc, doc.field(usage, "prompt_tokens"))
    r.output_tokens = _int_or_zero(doc, doc.field(usage, "completion_tokens"))
    var choice = doc.at(doc.field(root, "choices"), 0)
    if choice < 0:
        raise Error("chat completions: response has no choices")
    r.stop_reason = _str_or_empty(doc, doc.field(choice, "finish_reason"))
    var msg = doc.field(choice, "message")
    r.text = _str_or_empty(doc, doc.field(msg, "content"))
    var calls = doc.field(msg, "tool_calls")
    for i in range(doc.size(calls)):
        var c = doc.at(calls, i)
        var func = doc.field(c, "function")
        var args = doc.field(func, "arguments")
        # Spec says a JSON-encoded STRING; some local servers send the object.
        var args_json = doc.string(args) if doc.kind_of(args) == J_STRING else dump_json(doc, args)
        r.tool_calls.append(
            ToolCall(
                _str_or_empty(doc, doc.field(c, "id")),
                doc.string(doc.field(func, "name")),
                args_json^,
            )
        )
    r.message = ChatMessage.assistant(r.text.copy())
    r.message.tool_calls = r.tool_calls.copy()
    return r^


# ═══════════════════════════════════════════════════════════════════════════
# Conversation — history + tools, append-only
# ═══════════════════════════════════════════════════════════════════════════


struct Conversation(Movable):
    """A running dialogue. Append-only by design: every turn the model sent
    goes back exactly as it came (see the module docstring)."""

    var system: String
    var tools: List[ToolSpec]
    var messages: List[ChatMessage]

    def __init__(out self, var system: String = String("")):
        self.system = system^
        self.tools = List[ToolSpec]()
        self.messages = List[ChatMessage]()

    def __init__(out self, *, deinit move: Self):
        self.system = move.system^
        self.tools = move.tools^
        self.messages = move.messages^

    def tool(mut self, var name: String, var description: String, var schema_json: String):
        self.tools.append(ToolSpec(name^, description^, schema_json^))

    def user(mut self, var text: String):
        self.messages.append(ChatMessage.user(text^))

    def user_image(
        mut self,
        var text: String,
        ref image: List[UInt8],
        var media_type: String = String("image/png"),
    ) raises:
        var m = ChatMessage.user(text^)
        m.add_image(image, media_type^)
        self.messages.append(m^)

    def tool_result(mut self, var call_id: String, var content: String, is_error: Bool = False):
        self.messages.append(ChatMessage.tool_result(call_id^, content^, is_error))

    def send(mut self, mut client: ChatClient) raises -> ChatResponse:
        """Send the history; append the answer to it; return the answer."""
        var r = client.chat(self.messages, self.system, self.tools)
        self.messages.append(r.message.copy())
        return r^

# +--------------------------------------------------------------------------+ #
# | LFM3 chat — a conversation on one cache, token by token
# +--------------------------------------------------------------------------+ #
"""LFM3-1.7B (or any LFM3 text checkpoint) as a chat model (docs/NN_MODEL_PORTS_PLAN.md
§12.7): Liquid's chat template, a conversation kept in one `Lfm3Cache` across
turns, and Liquid's sampler.

    var chat = Lfm3Chat.load(DeviceContext(), dir)          # LFM3-1.7B's folder
    chat.reset("You are Noé, a small humanoid robot ...")   # the system prompt
    chat.user("Bonjour ! Qui es-tu ?")
    chat.start(ChatSettings())                              # the prefill; the first token chosen
    while chat.next():                                      # one more token a call, until done
        print(chat.reply_text())                            # what the reply says so far

THE TEMPLATE (`chat_template.jinja`, thinking off unless `thinking = True`):

    <|startoftext|><|im_start|>system\\n{system}<|im_end|>\\n
    <|im_start|>user\\n{text}<|im_end|>\\n<|im_start|>assistant\\n<think></think>{reply}<|im_end|>\\n
    <|im_start|>user\\n ...

With thinking off, the template renders an earlier reply back exactly as it was
generated (step 1 checked it in text and tokens), so a new turn is only its
own tokens appended to the cache. With thinking ON it is not: the template
drops an earlier turn's thinking from the history, so `user` refuses a second
turn on a thinking conversation (`reset` first).

The user's text is tokenized as transformers does: a special token's string
inside it IS that token. A caller passing untrusted text should break them
first (`noeira.llm.d1.prompt.plain`).

THE SAMPLER, transformers' order: the repetition penalty over every token of
the conversation (a logit divided by the penalty if positive, multiplied if
negative), then greedy (`settings.greedy`) or temperature, top-k and a draw.
Liquid's chat settings: temperature 0.1, top_k 50, repetition_penalty 1.1.
"""

from std.math import exp
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.io.json import load_json
from noeira.text.bpe import ByteLevelBPE
from .model import Lfm3Text
from .generate import Lfm3Cache, forward_cached, last_logits


comptime BOS = "<|startoftext|>"
comptime IM_START = "<|im_start|>"
comptime IM_END = "<|im_end|>"
comptime THINK_OFF = "<think></think>"
comptime THINK_ON = "<think>"


@fieldwise_init
struct ChatSettings(Copyable, Movable):
    var greedy: Bool
    var temperature: Float64
    var top_k: Int
    var repetition_penalty: Float64
    var max_tokens: Int
    var seed: UInt64

    def __init__(out self):
        """Liquid's chat settings (`generation_config.json`)."""
        self.greedy = False
        self.temperature = 0.1
        self.top_k = 50
        self.repetition_penalty = 1.1
        self.max_tokens = 256
        self.seed = 0x5EED

    @staticmethod
    def greedy_with(penalty: Float64, max_tokens: Int) -> ChatSettings:
        var s = ChatSettings()
        s.greedy = True
        s.repetition_penalty = penalty
        s.max_tokens = max_tokens
        return s^


def choose(
    mut logits: List[Float32], ref seen: List[Bool], settings: ChatSettings, mut rng: UInt64,
) -> Int:
    """The next token from raw `logits` (modified in place by the penalty):
    `seen[id]` = the id is in the conversation."""
    var pen = Float32(settings.repetition_penalty)
    if settings.repetition_penalty != 1.0:
        for i in range(len(logits)):
            if seen[i]:
                logits[i] = logits[i] / pen if logits[i] > 0.0 else logits[i] * pen
    if settings.greedy:
        var best = 0
        for i in range(1, len(logits)):
            if logits[i] > logits[best]:  # ties to the lower id, as torch.argmax
                best = i
        return best
    # top-k (k small): insertion into a sorted short list
    var k = max(1, settings.top_k)
    var top_i = List[Int]()
    var top_v = List[Float32]()
    for i in range(len(logits)):
        var x = logits[i]
        if len(top_v) == k and x <= top_v[k - 1]:
            continue
        var at = len(top_v)
        while at > 0 and top_v[at - 1] < x:
            at -= 1
        top_v.insert(at, x)
        top_i.insert(at, i)
        if len(top_v) > k:
            _ = top_v.pop()
            _ = top_i.pop()
    var probs = List[Float64]()
    var tot = 0.0
    for j in range(len(top_v)):
        var e = exp((Float64(top_v[j]) - Float64(top_v[0])) / settings.temperature)
        probs.append(e)
        tot += e
    # xorshift64*: a uniform draw in [0, tot)
    rng ^= rng >> 12
    rng ^= rng << 25
    rng ^= rng >> 27
    var u = Float64((rng * 0x2545F4914F6CDD1D) >> 11) / Float64(UInt64(1) << 53) * tot
    for j in range(len(probs)):
        u -= probs[j]
        if u < 0.0:
            return top_i[j]
    return top_i[len(top_i) - 1]


struct Lfm3Chat(Movable):
    var lm: Lfm3Text
    var tok: ByteLevelBPE
    var cache: Lfm3Cache
    var eos: Int
    var thinking: Bool
    """`<think>` after the assistant header (Liquid's default) instead of `<think></think>`."""
    var pending: List[Int]
    """Tokens of the conversation not yet in the cache (fed by the next `start`)."""
    var seen: List[Bool]
    """Every id in the conversation (the repetition penalty's set)."""
    var reply: List[Int]
    """The current reply's tokens, `<|im_end|>` included once reached."""
    var turns: Int
    var settings: ChatSettings
    var rng: UInt64
    var done: Bool
    var last_logits: List[Float32]
    """The raw logits that chose the last token (the gate reads them)."""

    def __init__(out self, var lm: Lfm3Text, var tok: ByteLevelBPE, var cache: Lfm3Cache, eos: Int):
        self.seen = List[Bool](length=lm.cfg.vocab, fill=False)
        self.lm = lm^
        self.tok = tok^
        self.cache = cache^
        self.eos = eos
        self.thinking = False
        self.pending = List[Int]()
        self.reply = List[Int]()
        self.turns = 0
        self.settings = ChatSettings()
        self.rng = 0
        self.done = True
        self.last_logits = List[Float32]()

    @staticmethod
    def load(var ctx: DeviceContext, dir: String, cap: Int = 4096) raises -> Lfm3Chat:
        """`dir`: an LFM3 text checkpoint (config, safetensors, tokenizer,
        generation_config). `cap`: the conversation's length in tokens."""
        var lm = Lfm3Text.load(ctx^, dir)
        var tok = ByteLevelBPE.load(dir + "/tokenizer.json")
        var cache = Lfm3Cache.make(lm, cap)
        var gc = load_json(dir + "/generation_config.json")
        var e = gc.field(gc.root(), "eos_token_id")
        var eos = gc.integer(gc.at(e, 0)) if gc.size(e) > 0 else gc.integer(e)
        return Lfm3Chat(lm^, tok^, cache^, eos)

    def reset(mut self, system: String) raises:
        """A new conversation: `system` (empty: none) and nothing else."""
        self.cache.reset()
        self.pending = List[Int]()
        for i in range(len(self.seen)):
            self.seen[i] = False
        self.reply = List[Int]()
        self.turns = 0
        self.done = True
        var head = String(BOS)
        if system.byte_length() > 0:
            head += IM_START + "system\n" + system + IM_END + "\n"
        self._add(self.tok.encode(head))

    def user(mut self, text: String) raises:
        """The user's turn, and the assistant's header after it."""
        if not self.done:
            raise Error("lfm3 chat: a reply is still being generated")
        if self.turns > 0 and self.thinking:
            raise Error("lfm3 chat: a second turn with thinking on re-renders the history (reset first)")
        var s = String("")
        if self.turns > 0:
            # the reply ends with <|im_end|> when it was finished, not when max_tokens cut it
            s = ("" if (len(self.reply) > 0 and self.reply[len(self.reply) - 1] == self.eos) else String(IM_END)) + "\n"
        s += IM_START + "user\n" + text + IM_END + "\n" + IM_START + "assistant\n"
        s += THINK_ON if self.thinking else THINK_OFF
        self._add(self.tok.encode(s))
        self.turns += 1

    def _add(mut self, ids: List[Int]):
        for id in ids:
            self.pending.append(id)
            self.seen[id] = True

    def start(mut self, settings: ChatSettings) raises:
        """Feed what is pending (the prefill) and choose the reply's first token."""
        if len(self.pending) == 0:
            raise Error("lfm3 chat: nothing to answer (call user first)")
        self.settings = settings.copy()
        self.rng = settings.seed if settings.seed != 0 else 1
        self.reply = List[Int]()
        self.done = False
        self._feed_and_choose()

    def next(mut self) raises -> Bool:
        """One more token; False (and no token) once the reply is finished
        (`<|im_end|>` or `max_tokens` reached by an earlier call)."""
        if self.done:
            return False
        self._feed_and_choose()
        return True

    def _feed_and_choose(mut self) raises:
        var feed = self.pending^
        self.pending = List[Int]()
        if self.cache.left() < len(feed) + 1:
            raise Error("lfm3 chat: the conversation is longer than the cache (" + String(self.cache.cap) + " tokens)")
        var h = forward_cached(self.lm, feed, self.cache)
        var logits = last_logits(self.lm, h, len(feed))
        self.last_logits = logits.copy()
        var t = choose(logits, self.seen, self.settings, self.rng)
        self.reply.append(t)
        self.seen[t] = True
        self.pending.append(t)  # in the conversation; into the cache with the next feed
        if t == self.eos or len(self.reply) >= self.settings.max_tokens:
            self.done = True

    def reply_text(self) -> String:
        """The reply so far, without `<|im_end|>`; a character cut between two
        tokens shows as U+FFFD until its last byte comes."""
        var ids = self.reply.copy()
        if len(ids) > 0 and ids[len(ids) - 1] == self.eos:
            _ = ids.pop()
        return self.tok.decode(ids)

    def prompt_ids(self) -> List[Int]:
        """What `start` will feed (the gate compares it with the template's tokens)."""
        return self.pending.copy()

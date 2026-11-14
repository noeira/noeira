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
from .model import Lfm3Text, LF_INT8
from .generate import Lfm3Cache, Lfm3CacheMark, forward_cached, forward_cached_marks, last_logits


comptime BOS = "<|startoftext|>"
comptime IM_START = "<|im_start|>"
comptime IM_END = "<|im_end|>"
comptime THINK_OFF = "<think></think>"
comptime THINK_ON = "<think>"

comptime MARK_STRIDE: Int = 32
"""`keep_system`: a mark every this many tokens of the system prompt, and one
at its end. A changed prompt is recomputed from the last mark before its first
changed token, so at most this many unchanged tokens are computed again. A mark
is the 13 conv histories, 1.9 MB."""


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


def answer_of(text: String) -> String:
    """The ANSWER in a decoded reply: the last non-empty piece between its
    `</think>` marks, without the blanks around it; the whole text when
    there is no mark. `thought</think>answer` -> `answer`;
    `draft</think>` -> `draft`.

    ⚠ NOT ONLY FOR THINKING ON. With thinking off the prompt already ends
    with `<think></think>`, and the model still sometimes writes a first
    draft, `</think>`, then says it again: 22 of 198 replies when the system
    prompt carried the robot's state, none with a two-line prompt
    (docs/ROBOT_BRAIN_PLAN.md, B1). `Lfm3Chat` ends such a reply at the mark
    (`think_end`), so the draft is the answer. A caller that speaks the reply
    wants this, not `reply_text` (the raw decode, as the gate compares it)."""
    var best = String("")
    for part in text.split("</think>"):
        var p = String(String(part).strip())
        if p.byte_length() > 0:
            best = p
    return best


struct Lfm3Chat(Movable):
    var lm: Lfm3Text
    var tok: ByteLevelBPE
    var cache: Lfm3Cache
    var eos: Int
    var think_end: Int
    """`</think>`'s id (-1 when the tokenizer has no single token for it).
    With thinking OFF a reply ends when the model writes it, as at `eos`: the
    prompt has already closed the thinking, so the mark only starts a second
    wording of the answer just given. Measured on BB-1: the same score with
    the draft as with what followed it, and 0.7 s instead of 1.5 s on those
    replies. With thinking ON it is the model's own mark and ends nothing."""
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
    var closed: Bool
    """The last assistant message ended with `<|im_end|>` (generated, or `assistant`)."""
    var keep_system: Bool
    """Prefix cache, matched by TOKENS (docs/NN_MODEL_PORTS_PLAN.md §12.8, S1):
    the system prompt's tokens stay in the cache after a request, with marks
    along them (`MARK_STRIDE`). The next `reset` keeps the part of its prompt
    that is already there, up to the last mark before the first token that
    differs, and only the rest is computed, in one chunk with the user's turn.
    A prompt written from a robot's state every turn (`noeira.brain.reply_view`)
    keeps its unchanged beginning; the same prompt again costs nothing.
    `prepare` computes a prompt ahead of the request.

    ⚠ A REUSED PREFIX IS NOT BIT-IDENTICAL TO A RECOMPUTED ONE: its rows were
    computed in another chunk, and a GEMM's rounding depends on its shape. The
    two agree within the gate's bound for cached chunks; a reply can differ
    from the cold one at a near-tie. The same sequence of requests gives the
    same replies.

    Off by default: `prompt_ids` then holds the whole prompt, as the gate
    reads it."""
    var kept_ids: List[Int]
    """The system prompt's tokens that are in the cache, from position 0."""
    var marks: List[Lfm3CacheMark]
    """Ascending, each at or before `len(kept_ids)`."""
    var sys_left: Int
    """How many of `pending`'s first tokens are the system prompt's (marks are
    taken along them when they are fed)."""
    var reused: Int
    """The last `reset`: tokens of its system prompt found in the cache."""
    var head_len: Int
    """The last `reset`: its system prompt's tokens, template included."""
    var debug_prefix: Int
    """Falsification (the gate's mutants): 1 = rewind to the mark AFTER the
    common prefix (tokens that differ are reused), 2 = a rewind keeps the
    marks and tokens past it (a later prompt is answered on stale rows)."""

    def __init__(out self, var lm: Lfm3Text, var tok: ByteLevelBPE, var cache: Lfm3Cache, eos: Int, think_end: Int = -1):
        self.seen = List[Bool](length=lm.cfg.vocab, fill=False)
        self.lm = lm^
        self.tok = tok^
        self.cache = cache^
        self.eos = eos
        self.think_end = think_end
        self.thinking = False
        self.pending = List[Int]()
        self.reply = List[Int]()
        self.turns = 0
        self.settings = ChatSettings()
        self.rng = 0
        self.done = True
        self.last_logits = List[Float32]()
        self.closed = False
        self.keep_system = False
        self.kept_ids = List[Int]()
        self.marks = List[Lfm3CacheMark]()
        self.sys_left = 0
        self.reused = 0
        self.head_len = 0
        self.debug_prefix = 0

    @staticmethod
    def load(
        var ctx: DeviceContext, dir: String, cap: Int = 4096, int8: Bool = LF_INT8, q8_levels: Int = 127,
    ) raises -> Lfm3Chat:
        """`dir`: an LFM3 text checkpoint (config, safetensors, tokenizer,
        generation_config). `cap`: the conversation's length in tokens.
        `int8`: the weights loaded as int8 (`Lfm3Text.load(int8=True)`, no bf16
        copy kept; the default is the build's `-D LFM3_INT8=1`)."""
        var lm = Lfm3Text.load(ctx^, dir, int8, q8_levels)
        var tok = ByteLevelBPE.load(dir + "/tokenizer.json")
        var cache = Lfm3Cache.make(lm, cap)
        var gc = load_json(dir + "/generation_config.json")
        var e = gc.field(gc.root(), "eos_token_id")
        var eos = gc.integer(gc.at(e, 0)) if gc.size(e) > 0 else gc.integer(e)
        var te = tok.encode(String("</think>"))
        var think_end = te[0] if len(te) == 1 else -1
        return Lfm3Chat(lm^, tok^, cache^, eos, think_end)

    def reset(mut self, system: String) raises:
        """A new conversation: `system` (empty: none) and nothing else. With
        `keep_system`, the part of the prompt already in the cache is kept
        (above); the rest is fed by the next `start`, or now by `prepare`."""
        self.pending = List[Int]()
        for i in range(len(self.seen)):
            self.seen[i] = False
        self.reply = List[Int]()
        self.turns = 0
        self.done = True
        self.closed = False
        var head = String(BOS)
        if system.byte_length() > 0:
            head += IM_START + "system\n" + system + IM_END + "\n"
        var ids = self.tok.encode(head)
        self.head_len = len(ids)
        var at = 0
        var best = -1
        if self.keep_system:
            var common = 0
            while common < len(ids) and common < len(self.kept_ids) and ids[common] == self.kept_ids[common]:
                common += 1
            for i in range(len(self.marks)):
                if self.marks[i].pos <= common:
                    best = i
            if (self.debug_prefix & 1) != 0 and best + 1 < len(self.marks) and self.marks[best + 1].pos <= len(ids):
                best += 1  # mutant 1: one mark too far
        if best >= 0:
            at = self.marks[best].pos
            self.cache.rewind(self.lm, self.marks[best])
            if (self.debug_prefix & 2) == 0:
                while len(self.marks) > best + 1:
                    _ = self.marks.pop()
                while len(self.kept_ids) > at:
                    _ = self.kept_ids.pop()
        else:
            self.cache.reset()
            self.marks = List[Lfm3CacheMark]()
            self.kept_ids = List[Int]()
        self.reused = at
        self.sys_left = len(ids) - at if self.keep_system else 0
        for id in ids:
            self.seen[id] = True
        for i in range(at, len(ids)):
            self.pending.append(ids[i])

    def prepare(mut self, system: String) raises:
        """`reset(system)`, and with `keep_system` the prompt computed NOW
        rather than with the first reply: for a caller that knows its prompt
        before it has a question (a robot, while the person is still speaking).
        The `reset` of the request that follows then finds it in the cache."""
        if not self.done:
            raise Error("lfm3 chat: a reply is still being generated")
        self.reset(system)
        if self.sys_left == 0:
            return
        var feed = self.pending^
        self.pending = List[Int]()
        if self.cache.left() < len(feed) + 1:
            raise Error("lfm3 chat: the system prompt is longer than the cache (" + String(self.cache.cap) + " tokens)")
        _ = self._feed(feed)

    def _feed(mut self, ref feed: List[Int]) raises -> DeviceBuffer[DT]:
        """`feed` into the cache. While it starts with system-prompt tokens
        (`sys_left`), marks are taken along them and at their end."""
        if self.sys_left == 0:
            return forward_cached(self.lm, feed, self.cache)
        var offs = List[Int]()
        var m = MARK_STRIDE - self.cache.pos % MARK_STRIDE
        while m < self.sys_left:
            offs.append(m)
            m += MARK_STRIDE
        offs.append(self.sys_left)
        var h = forward_cached_marks(self.lm, feed, self.cache, offs, self.marks)
        for i in range(self.sys_left):
            self.kept_ids.append(feed[i])
        self.sys_left = 0
        return h^

    def user(mut self, text: String) raises:
        """The user's turn, and the assistant's header after it."""
        if not self.done:
            raise Error("lfm3 chat: a reply is still being generated")
        if self.turns > 0 and self.thinking:
            raise Error("lfm3 chat: a second turn with thinking on re-renders the history (reset first)")
        var s = String("")
        if self.turns > 0:
            # a reply cut by max_tokens is closed here, as the template closes it
            s = ("" if self.closed else String(IM_END)) + "\n"
        s += IM_START + "user\n" + text + IM_END + "\n" + IM_START + "assistant\n"
        s += THINK_ON if self.thinking else THINK_OFF
        self._add(self.tok.encode(s))
        self.turns += 1
        self.closed = False
        self.reply = List[Int]()

    def assistant(mut self, text: String) raises:
        """An earlier reply of the conversation, given rather than generated
        (a caller replaying its history): after `user`, the template's
        rendering of an assistant message (thinking off)."""
        if self.turns == 0 or self.closed or len(self.reply) > 0:
            raise Error("lfm3 chat: an assistant message must follow a user message")
        if self.thinking:
            raise Error("lfm3 chat: a replayed history needs thinking off")
        self._add(self.tok.encode(text + IM_END))
        self.closed = True

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
        var h = self._feed(feed)
        var logits = last_logits(self.lm, h, len(feed))
        self.last_logits = logits.copy()
        var t = choose(logits, self.seen, self.settings, self.rng)
        self.reply.append(t)
        self.seen[t] = True
        self.pending.append(t)  # in the conversation; into the cache with the next feed
        var draft_end = t == self.think_end and not self.thinking
        if t == self.eos or draft_end or len(self.reply) >= self.settings.max_tokens:
            self.done = True
            self.closed = t == self.eos

    def reply_text(self) -> String:
        """The reply so far, without `<|im_end|>`; a character cut between two
        tokens shows as U+FFFD until its last byte comes."""
        var ids = self.reply.copy()
        if len(ids) > 0 and ids[len(ids) - 1] == self.eos:
            _ = ids.pop()
        return self.tok.decode(ids)

    def answer_text(self) -> String:
        """What to show or speak: `answer_of(reply_text())`."""
        return answer_of(self.reply_text())

    def prompt_ids(self) -> List[Int]:
        """What `start` will feed (the gate compares it with the template's tokens)."""
        return self.pending.copy()

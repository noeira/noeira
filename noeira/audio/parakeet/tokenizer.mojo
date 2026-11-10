# +--------------------------------------------------------------------------+ #
# | Parakeet-TDT — token ids to text
# +--------------------------------------------------------------------------+ #
"""`ParakeetTokenizer` — the decode half of the checkpoint's `tokenizer.json`.

Speech-to-text only needs ids -> text (text -> ids is for fine-tuning, later).
The file is a Hugging Face `tokenizers` BPE: 8192 pieces in `model.vocab`, plus
`added_tokens` (the language / control tags, and `<blank>` = 8192, which the
vocabulary itself does not hold). Decoding is its `Metaspace` decoder:

    skip the SPECIAL tokens (`<unk>`, `<pad>`, `<blank>`)
    concatenate the pieces, "▁" (U+2581) becomes a space
    the first piece loses its leading space

No byte-fallback pieces exist in this vocabulary and no piece needs unescaping
beyond what the JSON reader does, so that is the whole decoder.
"""

from noeira.io.json import load_json


struct ParakeetTokenizer(Movable):
    var pieces: List[String]
    """Piece of each id; empty for an id nothing names."""
    var special: List[Bool]

    def __init__(out self):
        self.pieces = List[String]()
        self.special = List[Bool]()

    @staticmethod
    def load(path: String, vocab_size: Int) raises -> Self:
        """Read `tokenizer.json`. `vocab_size` is the model's (8193: the pieces
        and `<blank>`); every id below it must end up named."""
        var doc = load_json(path)
        var root = doc.root()
        var t = Self()
        for _ in range(vocab_size):
            t.pieces.append(String(""))
            t.special.append(False)
        var named = List[Bool](length=vocab_size, fill=False)

        var vocab = doc.field(doc.field(root, "model"), "vocab")
        for i in range(doc.size(vocab)):
            var id = doc.integer(doc.at(vocab, i))
            if id < 0 or id >= vocab_size:
                raise Error("tokenizer: piece id " + String(id) + " is outside the vocabulary")
            t.pieces[id] = doc.key_at(vocab, i)
            named[id] = True

        var added = doc.field(root, "added_tokens")
        for i in range(doc.size(added)):
            var a = doc.at(added, i)
            var id = doc.integer(doc.field(a, "id"))
            if id < 0 or id >= vocab_size:
                continue
            t.pieces[id] = doc.string(doc.field(a, "content"))
            t.special[id] = doc.boolean(doc.field(a, "special"))
            named[id] = True

        for id in range(vocab_size):
            if not named[id]:
                raise Error("tokenizer: id " + String(id) + " has no piece")
        return t^

    def decode(self, ref ids: List[Int]) -> String:
        """The text of `ids`, special tokens skipped."""
        var text = String("")
        for i in range(len(ids)):
            var id = ids[i]
            if id < 0 or id >= len(self.pieces) or self.special[id]:
                continue
            text += self.pieces[id]
        text = text.replace("▁", " ")
        if text.startswith(" "):
            return String(text[byte=1:])
        return text

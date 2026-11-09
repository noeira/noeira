"""The `key = value` text format every sensor spec in `zoo/` is written in.

One reader for every sensor kind, so the comment rule, the whitespace rule and
the "expected `key = value`" error cannot drift between a lidar file and a
depth-camera file. Each kind keeps its own key table — and raises on a key it
does not know, because a misspelt `range_sigam` that was silently ignored
would run a noiseless experiment labelled as a noisy one.
"""


@fieldwise_init
struct SpecLine(Copyable, Movable):
    var key: String
    var value: String


def spec_lines(text: String, kind: String) raises -> List[SpecLine]:
    """Non-empty lines as (key, value); `#` starts a comment."""
    var out = List[SpecLine]()
    for raw in text.split("\n"):
        var full = String(raw)
        var hash = full.find("#")
        var cut = String(full[byte=0:hash]) if hash >= 0 else full
        var line = String(cut.strip())
        if line.byte_length() == 0:
            continue
        var eq = line.find("=")
        if eq < 0:
            raise Error(kind + " spec: expected `key = value`, got '" + line + "'")
        out.append(
            SpecLine(
                String(String(line[byte=0:eq]).strip()),
                String(String(line[byte=eq + 1 : line.byte_length()]).strip()),
            )
        )
    return out^


def read_text(path: String) raises -> String:
    var f = open(path, "r")
    var text = f.read()
    f.close()
    return text^

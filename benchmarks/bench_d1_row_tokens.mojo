# +--------------------------------------------------------------------------+ #
# | What fills the decision model's rows: tokens per part of each request
# +--------------------------------------------------------------------------+ #
"""The decision model's wait is its tokens (33 ms + 0.40 ms a token alone on
the Orin, docs/NN_MODEL_PORTS_PLAN.md §12.8.5), and the room demo's requests
read 815 and 1228 of them. This counts where they are. CPU only: the tokenizer.

    pixi run mojo run -I . benchmarks/bench_d1_row_tokens.mojo [d1 dir] [g1_command_bank.txt]

The requests the voice loop sends with the room demo's flags (`--request-wording
--room-state --triage narrow --opts name_desc --dest-chain`), each rendered and
packed as `D1.system_one` does (`noeira.llm.d1.prompt`): one copy of the state,
then the questions. Printed per request: its tokens, the state's share, and
each question's tokens with its options counted.

The state here is a robot standing at the table in the four-place room, having
heard one sentence: the demo's own state grows with what the robot has heard
and done, so its rows are a little longer.
"""

from std.os import getenv
from std.sys import argv

from noeira.ai.jev import JevQuestions
from noeira.brain.geometry import PlaceXY, place_lines
from noeira.envs.robots.g1_command_bank import G1CommandBank
from noeira.envs.robots.g1_command_language import (
    G1Context, g1_command_state, g1_destination_questions, g1_add_triage, g1_add_two_places_gate,
    g1_two_places_questions, g1_two_places_state, g1_trail_questions, g1_command_questions,
)
from noeira.io.json import parse_json, JsonWriter
from noeira.llm.d1.prompt import parse_questions, prefix_text, suffix_text, py_json_dumps, pack_rows, HEADER
from noeira.text.bpe import ByteLevelBPE


def _bytes(s: String) -> List[UInt8]:
    var b = List[UInt8](capacity=s.byte_length())
    for x in s.as_bytes():
        b.append(x)
    return b^


def breakdown(mut tok: ByteLevelBPE, name: String, state: String, is_json: Bool, ref q: JevQuestions) raises:
    var block = state
    if is_json:
        var sd = parse_json(_bytes(state))
        block = py_json_dumps(sd, sd.root())
    var w = JsonWriter()
    q.write_to_json(w)
    var qd = parse_json(_bytes(w.done()))
    var qs = parse_questions(qd, qd.root())
    var prefix = prefix_text(tok, block)
    var suffixes = List[String]()
    for x in qs:
        suffixes.append(suffix_text(tok, x))
    var rows = pack_rows(tok, prefix, suffixes)
    var total = 0
    var state_tokens = 0
    for r in rows:
        total += len(r.ids)
        state_tokens += r.state_len
    print(name + ":", total, "tokens in", len(rows), "row(s);", len(qs), "questions; the state", state_tokens, "tokens (",
          100 * state_tokens // total, "% )")
    for i in range(len(qs)):
        var first = False
        for r in rows:
            if r.first == i:
                first = True
        var n = len(tok.encode(suffixes[i])) if first else len(tok.encode(String(HEADER) + suffixes[i]))
        var ins = len(tok.encode(qs[i].instructions))
        print("     ", n, "tokens:", qs[i].name, "—", len(qs[i].labels), "options; its instructions alone", ins)


def main() raises:
    var args = argv()
    var dir = String(args[1]) if len(args) > 1 else getenv("HOME") + "/.cache/noeira/local-ai/d1/d1-2B"
    var bank_path = String(args[2]) if len(args) > 2 else String("g1_command_bank.txt")
    var tok = ByteLevelBPE.load(dir + "/tokenizer.json")
    var bank = G1CommandBank.load(bank_path)
    var places: List[String] = ["fridge", "door", "table", "sofa"]
    var layout: List[PlaceXY] = [
        PlaceXY("fridge", 3.0, 1.5), PlaceXY("door", -3.2, 2.0), PlaceXY("table", 1.0, -2.5), PlaceXY("sofa", -2.0, -2.8),
    ]
    var ctx = G1Context()
    ctx.places = True
    ctx.geometry = True
    ctx.memory = True
    ctx.began(String("stand"))
    ctx.at = String("table")
    ctx.arrived = String("table")
    ctx.arrived_at(String("table"), 1.0)
    ctx.around = place_lines(layout, 0.8, -2.0, 30.0)
    var heard = String("Bonjour, comment t'appelles-tu ?")
    var state = g1_command_state(heard, ctx)
    var bare = G1Context()
    bare.places = True
    print("the state alone, as JSON text:", len(tok.encode(state)), "tokens with the room,",
          len(tok.encode(g1_command_state(heard, bare))), "without")

    var dest = g1_destination_questions(places)
    g1_add_triage(dest, True)
    g1_add_two_places_gate(dest)
    breakdown(tok, String("destination request (triage, two-places gate)"), state, True, dest)
    var cmd = g1_command_questions(bank, True, List[String](), List[String](), False, name_desc=True)
    breakdown(tok, String("commands request (no chain)"), state, True, cmd)
    var pair = g1_two_places_questions(places, True)
    breakdown(tok, String("two-places request (bare state)"), g1_two_places_state(heard), True, pair)
    var trail = g1_trail_questions(places, True)
    breakdown(tok, String("trail request"), state, True, trail)

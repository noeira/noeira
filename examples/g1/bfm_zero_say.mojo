# +--------------------------------------------------------------------------+ #
# | Say it, and the robot does it — speech/text -> Jev -> the command channel
# +--------------------------------------------------------------------------+ #
"""Turn a sentence into one of the bank's 18 commands, or refuse.

    # with `build/g1play` already running:
    ./build/g1say --text "raise your right hand"
    ./build/g1say --record 4 --lang fr          # speak it instead
    ./build/g1say --list                        # what Jev will be offered
    ./build/g1say --doing walk --text "plus vite"    # a RELATIVE instruction

Needs `JEV_API_KEY` (or `TYPESAFE_API_KEY`); `--record` also needs `HF_TOKEN`
for Whisper. Nothing here touches the renderer: this is a WRITER on the
channel, and `build/g1play` is the reader.

## Why a closed option set, and why this one

A free-form model asked "what should the robot do?" will happily answer
"walk to the fridge" — a thing this robot cannot do. A constrained readout
picks from a list and cannot invent a member of it
(`docs/SYSTEM_ONE_ASSESSMENT.md` §4a). The list here is the BANK, so every
option is a command that passed four gates in §12.52: pool support, a
scaffold control, a hold on the rollout, and its shipped mean satisfying its
own compound.

⚠ THE OPTION DESCRIPTIONS ARE GENERATED FROM THE COMPOUND, never written by
hand. `right_hand_up` is offered as "arms: right_hand_height above 0.92,
left_hand_height below 0.75" because that is literally what it optimises. A
hand-written blurb would drift from the terms the command was gated on, and
the drift would be invisible — the robot doing one thing while the model was
told another.

## ⚠ `--doing` is how a relative instruction is tested without a microphone

"plus vite" is not a command, it is an edit to one. The viewer knows what the
robot is doing and says so in the state it sends
(`g1_command_state`); this CLI has no robot, so `--doing NAME` supplies the
same field, `--did A,B` the history and `--since SECONDS` the clock.

    ./build/g1say --doing walk --text "plus vite" --dry-run      # -> run
    ./build/g1say --doing right_hand_up --text "l'autre bras"    # -> left_hand_up

⚠ AND IT IS HOW THE WORDING STAYS HONEST. The relative clause was appended to
`g1_command_instruction` — the one wording §12.54 measured moving `P(none)`
by half — so every absolute phrasing has to be re-measured against it with
`--doing` UNSET. A regression there is a regression in the whole demo, and
this is the only path that can see it.

## ⚠ Abstention is a first-class answer

`none` is always on the list and the confidence is thresholded. Below the
bar, or on `none`, **nothing is written to the channel** and the robot keeps
doing what it was doing. `docs/SYSTEM_ONE_ASSESSMENT.md` §4c calls trained
abstention the single most valuable idea in this family for robotics, and
this is why: "walk to the fridge" has no safe nearest neighbour among
`walk`, `squat` and `spin_left`, and picking one anyway is how a robot ends
up doing something nobody asked for.
"""

from std.sys import argv
from std.time import perf_counter_ns

from noeira.ai.jev import JevClient, JevQuestions
from noeira.envs.robots.g1_command_language import (
    g1_command_questions, g1_command_instruction, g1_decide, g1_decide_chain,
    g1_extent_metres, g1_extent_seconds, G1_Q_EXTENT,
    g1_alias_name, g1_alias_target, g1_alias_desc, G1_N_ALIAS,
    G1_Q_COMMAND, G1_OPT_NONE,
    G1Context, g1_command_state,
)
from noeira.envs.robots.g1_spec import (
    G1Pool, g1_spec_questions, g1_spec_from_answers, g1_spec_admit,
    g1_spec_bank_baseline, g1_spec_describe, G1_SPEC_D,
)
# ⚠ module scope — Mojo rejects an import inside a branch. Importing costs
# nothing; only `SpeechToText.huggingface()` needs HF_TOKEN, so `--text`
# still runs without one.
from noeira.ai.audio_io import record_wav
from noeira.ai.speech import SpeechToText
from noeira.envs.robots.g1_command_bank import G1CommandBank
from noeira.envs.robots.g1_reward_vocab import G1Term
from noeira.envs.robots.g1_command_channel import (
    g1_channel_write, g1_channel_write_z, g1_channel_seq, g1_channel_read_ack,
)


def _flag(name: String, dflt: String) raises -> String:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            if i + 1 >= len(av):
                raise Error("flag " + name + " needs a value")
            return String(av[i + 1])
    return dflt


def _has(name: String) raises -> Bool:
    var av = argv()
    for i in range(1, len(av)):
        if String(av[i]) == name:
            return True
    return False


def _f2(v: Float64) -> String:
    var h = Int(v * 100.0 + 0.5)
    var f = String(h % 100)
    while f.byte_length() < 2:
        f = String("0") + f
    return String(h // 100) + String(".") + f


def _spin(ms: Int):
    """A crude wait. The Mojo stdlib has no `sleep`, and this is a CLI tool
    that waits a few hundred milliseconds for an ack — not a control loop."""
    var t0 = perf_counter_ns()
    var want = ms * 1_000_000
    while Int(perf_counter_ns() - t0) < want:
        pass


def main() raises:
    var bank_path = _flag(String("--bank"), String("g1_command_bank.txt"))
    var chan_path = _flag(String("--channel"), String("/tmp/g1_cmd"))
    var text = _flag(String("--text"), String(""))
    var record_s = _flag(String("--record"), String(""))
    var lang = _flag(String("--lang"), String(""))
    var blend = atol(_flag(String("--blend"), String("25")))
    # ⚠ THE ROBOT'S SIDE OF THE CONVERSATION, supplied by hand because this
    # program has no robot. Unset means `doing: nothing`, which is what a
    # cold start looks like and therefore what the absolute phrasings must
    # still be measured under.
    var doing = _flag(String("--doing"), String(""))
    var did = _flag(String("--did"), String(""))
    # 3 s = "a_moment": a command that has only just started is the odd case,
    # not the default one.
    var since = Float64(String(_flag(String("--since"), String("3"))))
    # ⚠ THE CACHE-MISS PATH IS OPT-IN AND LAZY. `P(none)` over the bank is a
    # CALIBRATED cache miss (§10.2), so nothing here changes the hit path —
    # which is the path §12.54 measured and the one the demo depends on. The
    # 70 MB sidecar is read only after a miss, and the second Jev call costs
    # ~350 ms, which is imperceptible after someone has stopped speaking.
    var pool_path = _flag(String("--pool"), String(""))
    var no_warn = _has(String("--spec-no-warn"))
    # ⚠ THE BAR IS P(none), NOT THE TOP-1 CONFIDENCE, and that is a measured
    # correction. "crouch down low" returned `crouch` at 0.56 with `squat` at
    # 0.38 — the mass was split between TWO CORRECT ANSWERS, because their
    # bands are adjacent (0.62-0.72 against 0.40-0.62) and they are near
    # duplicates. A top-1 threshold refuses that, punishing the options for
    # resembling each other rather than the instruction for being unclear.
    # The question abstention actually asks is "does this robot have such a
    # command?", and that is `P(none)`. The top-1 bar stays as a second,
    # much lower guard against a genuinely flat distribution.
    var max_none = Float64(String(_flag(String("--max-none"), String("0.25"))))
    var min_top = Float64(String(_flag(String("--min-top"), String("0.35"))))
    var dry = _has(String("--dry-run"))
    var strict = _has(String("--strict"))
    # ⚠ `--addressed` MAKES THIS TOOL ASK WHAT THE VIEWER ASKS. Without it
    # the CLI omits the `addressed` gate, so a phrase can pass here and be
    # refused there — which is exactly what happened to "Bonjour, comment
    # vas-tu ?": it read `talk` 1.00 in this tool and was killed by a gate
    # this tool never ran.
    var addressed = _has(String("--addressed"))
    # ⚠ `--shuffle N` asks the SAME question N times with the option list
    # rotated. The readout in this family is not permutation-invariant —
    # `imajev` pays 4x compute averaging over four orderings for exactly this
    # (`docs/SYSTEM_ONE_ASSESSMENT.md` §3.2) — and with 18 options a pick that
    # moves with the order is a pick nobody should act on. Nothing is sent in
    # this mode; it is a measurement.
    var shuffle = atol(_flag(String("--shuffle"), String("0")))

    var bank = G1CommandBank.load(bank_path)

    if _has(String("--list")):
        print("the", bank.count(), "commands Jev is offered, with the")
        print("descriptions GENERATED from their compounds:")
        for i in range(bank.count()):
            print("  " + bank.name_at(i))
            print("      " + bank.describe(i))
        # ⚠ listed from the SAME source the question is built from. A
        # hand-kept listing drifts, and a listing that omits the aliases
        # makes them look absent when they are being offered.
        for i in range(G1_N_ALIAS):
            print("  " + g1_alias_name(i) + "   (alias -> "
                  + g1_alias_target(i) + ")")
            print("      " + g1_alias_desc(i))
        print("  none")
        print("      not one of these, or not a command for this robot")
        return

    # ── what was said ─────────────────────────────────────────────────
    if record_s != "":
        var seconds = Float64(record_s)
        var wav = String("/tmp/noeira_g1_say.wav")
        print("listening", seconds, "s ...")
        record_wav(wav, seconds)
        var stt = SpeechToText.huggingface()
        # ⚠ PIN THE LANGUAGE HERE TOO. The viewer pins it; a CLI that lets
        # Whisper auto-detect is asking a different question than the demo
        # it stands in for, and its answer cannot be carried back.
        if lang != "":
            stt.language = lang
        var heard = stt.transcribe_file(wav)
        text = heard.text
        print("heard:", text, "(", heard.latency_ms, "ms )")
    if text == "":
        raise Error("pass --text \"...\" or --record SECONDS (or --list)")

    # ── the question ──────────────────────────────────────────────────
    var options = List[String]()
    var descs = List[String]()
    for i in range(bank.count()):
        options.append(bank.name_at(i))
        descs.append(bank.describe(i))
    # ⚠ `none` LAST and always present. Without it the model must pick
    # something, and "something" for an impossible request is a real command
    # the robot will actually run.
    options.append(String("none"))
    descs.append(String("not one of these, or not a command for this robot"))

    var q = g1_command_questions(bank, addressed)

    var jev = JevClient.from_env()

    if shuffle > 0:
        print("option-order sweep:", shuffle, "rotations of the same question")
        var first = String("")
        var agree = 0
        for r in range(shuffle):
            var rot = (r * 7) % (bank.count() + 1)
            var o2 = List[String]()
            var d2 = List[String]()
            for j in range(bank.count() + 1):
                var k = (j + rot) % (bank.count() + 1)
                if k == bank.count():
                    o2.append(String("none"))
                    d2.append(String("not one of these, or not a command for this robot"))
                else:
                    o2.append(bank.name_at(k))
                    d2.append(bank.describe(k))
            var q2 = JevQuestions()
            q2.choice(String(G1_Q_COMMAND), g1_command_instruction(), o2, d2)
            # ⚠ THE SAME STATE as the main leg, or the ordering probe is
            # measuring the state change as well as the ordering.
            var ctx2 = G1Context()
            if doing != "":
                ctx2.doing = doing.copy()
                ctx2.since_s = since
            var a2 = jev.decide(g1_command_state(text, ctx2), q2)
            var pk = a2.choice(String(G1_Q_COMMAND))
            var pn = a2.probability(String(G1_Q_COMMAND), String(G1_OPT_NONE))
            var cf = a2.confidence(String(G1_Q_COMMAND))
            print("  rot", rot, "->", pk, " top1", _f2(cf), " P(none)", _f2(pn))
            if r == 0:
                first = pk
            elif pk == first:
                agree += 1
        print("  ", agree + 1, "of", shuffle, "orderings agree on", first)
        return

    var ctx = G1Context()
    if doing != "":
        ctx.doing = doing.copy()
        ctx.since_s = since
    if did != "":
        var parts = did.split(",")
        for i in range(len(parts)):
            var pnm = String(String(parts[i]).strip())
            if pnm.byte_length() > 0:
                ctx.recent.append(pnm)
    var state = g1_command_state(text, ctx)
    # ⚠ PRINT IT. The state is now half the question, and a CLI that shows
    # only the answer cannot be used to debug a wrong one.
    print("state:", state)

    var t0 = perf_counter_ns()
    var ans = jev.decide(state, q)
    var ms = Float64(perf_counter_ns() - t0) / 1e6
    var argmax = ans.choice(String(G1_Q_COMMAND))
    var d = g1_decide(ans, bank, max_none, min_top, 0.5, addressed)
    var p_none = d.p_none
    var pick = d.name if d.name != "" else String("")
    var conf = d.conf
    print("argmax:", argmax, " best-real:", d.best, _f2(conf),
          " P(none)", _f2(p_none), " (", Int(ms), "ms )")

    # the runners-up say whether the decision was close
    var best2 = String("")
    var p2 = 0.0
    for i in range(bank.count() + G1_N_ALIAS + 1):
        var nm = String(G1_OPT_NONE)
        if i < bank.count():
            nm = bank.name_at(i)
        elif i < bank.count() + G1_N_ALIAS:
            nm = g1_alias_name(i - bank.count())
        if nm == d.best:
            continue
        var pi = ans.probability(String(G1_Q_COMMAND), nm)
        if pi > p2:
            p2 = pi
            best2 = nm
    if best2 != "":
        print("runner-up:", best2, _f2(p2))
    # the decomposition, so a two-part instruction can be read at a glance
    var chain = g1_decide_chain(ans, bank)
    if len(chain) > 0:
        var cs = String("")
        for i in range(len(chain)):
            cs += String(" -> ") + chain[i].name + String(" ") \
                  + _f2(chain[i].conf)
        print("chain:" + cs)
    var ext = ans.score(String(G1_Q_EXTENT))
    print("extent:", _f2(ext), "->", _f2(g1_extent_metres(ext)), "m /",
          _f2(g1_extent_seconds(ext)), "s")

    # ── abstain, or send ──────────────────────────────────────────────
    # ⚠ ONE RULE, in `g1_decide`. This used to hold its own copy of the
    # thresholds and the best-real-option logic, and the voice demo a second
    # copy; §12.54 measured the QUESTION's wording moving P(none) by half, so
    # two copies of the rule around it would be two different systems.
    var needs_world = d.needs_world
    # ⚠ `talk` is a SUCCESS with an empty `name`, not a refusal. The viewer
    # answers it; this tool only decides, so it says so and stops.
    if d.talk:
        print("TALK:", _f2(d.conf),
              "— the speaker wanted an answer, not a movement.",
              "`build/g1voice` replies; this does not.")
        return
    if d.name == "":
        # ── the cache miss: ask for a REWARD instead of a name ────────
        if pool_path != "" and d.reason == "no such command":
            print("MISS: P(none)", _f2(d.p_none),
                  "— the bank has no name for this. Asking for a reward spec.")
            var pool = G1Pool.load(pool_path)
            print("  pool:", pool.n, "rows from", pool_path)
            var sq = g1_spec_questions(not no_warn)
            var t1 = perf_counter_ns()
            var sa = jev.decide(state, sq)
            print("  sketch:", Int(Float64(perf_counter_ns() - t1) / 1e6),
                  "ms,", sa.input_tokens, "input tokens")
            var terms = List[G1Term]()
            # the return value IS the scaffold count — the goals start there
            var n_scaf = g1_spec_from_answers(sa, pool, bank, terms)
            if n_scaf < 0:
                print("REFUSED: the model named no quantity — nothing sent.")
                return
            var zb = List[Float64](length=bank.count() * G1_SPEC_D, fill=0.0)
            g1_spec_bank_baseline(pool, bank, zb)
            var z = List[Float64](length=G1_SPEC_D, fill=0.0)
            var v = g1_spec_admit(pool, bank, zb, terms, z, n_scaf)
            print("  spec:", g1_spec_describe(terms, n_scaf),
                  " (+", n_scaf, "scaffold terms )")
            print("  ESS", Int(v.ess), " nearest", (
                bank.name_at(v.nearest) if v.nearest >= 0 else String("-")),
                "at", _f2(v.angle), "deg, runner-up", (
                bank.name_at(v.second) if v.second >= 0 else String("-")),
                "margin", _f2(v.margin), "deg")
            if not v.ok:
                # ⚠ A DUPLICATE IS A SUCCESS WITH A DIFFERENT HANDLER. The
                # instruction named something the bank already holds, and the
                # bank's row went through four gates and a CEM refinement
                # worth 37-49 % that this baseline has not. Run THAT.
                if v.nearest >= 0:
                    print("  ", v.reason, "— sending the bank's row instead.")
                    if dry:
                        print("dry run — would send", bank.name_at(v.nearest))
                        return
                    var sq2 = g1_channel_seq(chan_path) + 1
                    if sq2 < 1:
                        sq2 = 1
                    g1_channel_write(chan_path, sq2,
                                     bank.name_at(v.nearest), blend)
                    print("sent seq", sq2, "->", bank.name_at(v.nearest))
                    return
                print("REFUSED:", v.reason, "— nothing sent.")
                return
            # ⚠ ONE TOKEN — `g1_channel_write_z` raises otherwise, because
            # `cmd` is read as one whitespace-delimited field.
            var label = String("spec:") + g1_spec_describe(terms, n_scaf)
            var lab = String("")
            for ch in label.codepoints():
                lab += String("_") if ch.to_u32() == 32 else String(ch)
            label = lab^
            if dry:
                print("dry run — would send a NOVEL z (", label, ")")
                return
            var sq3 = g1_channel_seq(chan_path) + 1
            if sq3 < 1:
                sq3 = 1
            g1_channel_write_z(chan_path, sq3, label, z, blend)
            print("sent seq", sq3, "-> NOVEL z:", label)
            return
        print("REFUSED:", d.reason, " P(none)", _f2(d.p_none),
              " addressed", _f2(d.addressed),
              " (leaning", d.best, _f2(d.conf), ") — nothing sent.",
              String("Pass --pool g1_pool.bin to try a reward spec.")
              if pool_path == "" else String(""))
        return

    # ⚠ PARTIAL EXECUTION, SAID OUT LOUD. The default is to do the part it
    # can and name the part it cannot, because a robot that silently performs
    # 30 % of an instruction is worse than one that performs 30 % and says
    # so. `--strict` refuses instead.
    if needs_world > 0.5:
        if strict:
            print("REFUSED: needs an object or a destination (",
                  _f2(needs_world), ") and --strict is set. Nothing sent.")
            return
        print("NOTE: this asks for an object or a destination (",
              _f2(needs_world), ") — this robot has neither. Doing the part",
              "it can:", pick)
    if dry:
        print("dry run — would send", pick)
        return

    # ⚠ ADVANCE PAST THE CHANNEL'S CURRENT SEQ. Starting from 1 would put us
    # below a shell script's seq 7 and the reader would ignore us.
    var seq = g1_channel_seq(chan_path) + 1
    if seq < 1:
        seq = 1
    g1_channel_write(chan_path, seq, pick, blend)
    print("sent seq", seq, "->", pick)

    # ── wait for the reader to say what it made of it ─────────────────
    # A writer that fires and forgets loses commands sent before the viewer
    # is up; §12.53 measured 14 s of that. This waits.
    for _ in range(40):
        _spin(50)
        var ack = g1_channel_read_ack(chan_path)
        if ack.seq == seq:
            if ack.fresh:
                print("ack: the robot is doing", ack.cmd)
            else:
                print("ack: the viewer did NOT recognise", ack.cmd)
            return
    print("no ack in 2 s — is `build/g1play` running on", chan_path, "?")

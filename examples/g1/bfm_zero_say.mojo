# +--------------------------------------------------------------------------+ #
# | Say it, and the robot does it — speech/text -> Jev -> the command channel
# +--------------------------------------------------------------------------+ #
"""Turn a sentence into one of the bank's 18 commands, or refuse.

    # with `build/g1play` already running:
    ./build/g1say --text "raise your right hand"
    ./build/g1say --record 4                    # speak it instead
    ./build/g1say --list                        # what Jev will be offered

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
# ⚠ module scope — Mojo rejects an import inside a branch. Importing costs
# nothing; only `SpeechToText.huggingface()` needs HF_TOKEN, so `--text`
# still runs without one.
from noeira.ai.audio_io import record_wav
from noeira.ai.speech import SpeechToText
from noeira.envs.robots.g1_command_bank import G1CommandBank
from noeira.envs.robots.g1_command_channel import (
    g1_channel_write, g1_channel_seq, g1_channel_read_ack,
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
    var blend = atol(_flag(String("--blend"), String("25")))
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

    var bank = G1CommandBank.load(bank_path)

    if _has(String("--list")):
        print("the", bank.count(), "commands Jev is offered, with the")
        print("descriptions GENERATED from their compounds:")
        for i in range(bank.count()):
            print("  " + bank.name_at(i))
            print("      " + bank.describe(i))
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

    var q = JevQuestions()
    q.choice(
        String("command"),
        # ⚠ `none` IS THE EXCEPTION, and the wording has to say so. The
        # first version ended "Choose `none` if the instruction asks for
        # something not on the list", which reads as an invitation: "show me
        # your left hand" came back `none` 0.52 against `left_hand_up` 0.48.
        String(
            "A humanoid robot can perform exactly the commands listed, and"
            " nothing else. Pick the command that best matches what the"
            " instruction asks the robot to do with its body — it does not"
            " have to match exactly, only be the closest thing the robot can"
            " do. Pick `none` ONLY when no command on the list is even"
            " approximately what was asked."
        ),
        options, descs,
    )
    # ⚠ ONE MORE QUESTION, AND IT IS NEARLY FREE. Jev answers every question
    # against the same state, so a second costs almost nothing — that is the
    # whole shape of its cost model.
    #
    # It exists because rewording the choice to stop it over-picking `none`
    # created the opposite error: "walk to the fridge and open it" went from
    # a refusal to `walk` at 0.92. That is not wrong — walking IS the closest
    # thing this robot can do — but it silently drops "to the fridge and open
    # it". This robot has no object interaction and no navigation to a
    # landmark, so asking about THAT directly is answerable from the
    # instruction alone, and it does not need to know which command was
    # picked.
    q.noul(
        String("needs_world"),
        String(
            "Does the instruction ask the robot to interact with an object,"
            " or to go to a particular place or thing?"
        ),
    )

    var jev = JevClient.from_env()
    var t0 = perf_counter_ns()
    var ans = jev.decide_text(String("instruction: ") + text, q)
    var ms = Float64(perf_counter_ns() - t0) / 1e6
    var argmax = ans.choice(String("command"))
    var p_none = ans.probability(String("command"), String("none"))
    # ⚠ THE BEST REAL COMMAND, NOT THE ARGMAX. "spin around fast" split
    # 0.42 `spin_left` / 0.14 `spin_right`, so `none` at 0.44 won the argmax
    # while 0.56 of the mass sat on commands the robot HAS. Near-duplicate
    # options divide their own vote; `none` does not have to beat the field,
    # only the sum of it.
    var pick = String("")
    var conf = 0.0
    for i in range(len(options)):
        if options[i] == "none":
            continue
        var pi = ans.probability(String("command"), options[i])
        if pi > conf:
            conf = pi
            pick = options[i]
    print("argmax:", argmax, " best-real:", pick, _f2(conf),
          " P(none)", _f2(p_none), " (", Int(ms), "ms )")

    # the runners-up say whether the decision was close
    var best2 = String("")
    var p2 = 0.0
    for i in range(len(options)):
        if options[i] == pick:
            continue
        var pi = ans.probability(String("command"), options[i])
        if pi > p2:
            p2 = pi
            best2 = options[i]
    if best2 != "":
        print("runner-up:", best2, _f2(p2))

    var needs_world = ans.noul(String("needs_world"))

    # ── abstain, or send ──────────────────────────────────────────────
    if p_none > max_none or pick == "":
        print("REFUSED: P(none)", _f2(p_none), "— the robot has no such",
              "command. Nothing sent.")
        return
    # the second guard still earns its place: `P(none)` can be low while the
    # mass is spread flat over many real commands, which means "it is a robot
    # command but I cannot tell which" — also not something to act on.
    if conf < min_top:
        print("REFUSED: best real option only", _f2(conf),
              "— no clear pick. Nothing sent.")
        return
    var idx = bank.find(pick)
    if idx < 0:
        # ⚠ belt and braces: a model that returned something off its own
        # option list would otherwise reach the channel and be dropped there.
        print("REFUSED: '", pick, "' is not in the bank. Nothing sent.")
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

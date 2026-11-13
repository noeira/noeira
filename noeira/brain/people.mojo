# +--------------------------------------------------------------------------+ #
# | People — who the robot has met, and whose voice this is
# +--------------------------------------------------------------------------+ #
"""The people the robot knows by name and by voice (docs/ROBOT_BRAIN_PLAN.md
§6, step B4).

    var people = People.load(robot_dir("noe"))           # people/index.json + people/<id>.voice

    # an utterance was heard (its text now, its voice embedding a moment later)
    people.heard(ws, "Je m'appelle Denis.", now)         # a name was given: kept for this utterance
    people.voice(ws, embedding, now)                     # -> enrols Denis with this voice, or recognises him

    ws.speaker          "Denis"      who is talking, "" when unknown
    ws.speaker_sure     True         the voice is his; False = it only resembles his

ENROLMENT IS SAYING ONE'S NAME. "Je m'appelle Denis", "my name is Denis",
"moi c'est Denis", "call me Denis": the name is read from the sentence by rule
(`name_given`), and the voice that said it is stored under it. There is no
confirmation dialogue: the reply view tells the reply model the name that was
just given, the robot says it back, and a wrong name is corrected by saying
the right one, which renames the same voice.

⚠ ONLY EXPLICIT FORMS. "Je suis fatigué" and "I am here" are not names, and
"il s'appelle Paul" is someone else's: a first-person form that can only
introduce a name is required.

RECOGNITION IS A COSINE AND TWO THRESHOLDS, per utterance, against each known
person's stored voices (the best of them):
- at `sure` or more: it is that person;
- from `maybe` to `sure`: it resembles that person, and the robot says no
  more than that;
- under `maybe`: an unknown voice.
⚠ EACH UTTERANCE IS JUDGED ON ITS OWN. A first version let a sure recognition
hold for two minutes through utterances that only resembled: a second person
speaking right after the first was called by the first one's name.
The thresholds come from `tools/speaker/eer_librispeech.py`'s measurement (the
plan's §14) and are fields: a room's own microphone may want others. A voice
recognised for sure is added to that person's stored voices, up to
`VOICES_MAX`, so the gallery covers more of how the person sounds.

⚠ A VOICE IS BIOMETRIC DATA. It is stored only for a person who gave their
name, only on this machine (`<robot dir>/people/`), and `forget` deletes the
person's files.
"""

from std.math import sqrt
from std.os import makedirs
from std.os.path import exists

from noeira.io.fileio import read_file_bytes, write_file_atomic, write_text_atomic, remove_file
from noeira.io.json import JsonWriter, parse_json
from .bb1 import normalise, has_phrase
from .events import EV_MET, EV_SPEAKER
from .state import WorldState


comptime VOICES_MAX: Int = 6
"""Stored voices per person; the oldest after the first is replaced."""
comptime SURE_COS: Float64 = 0.45
comptime MAYBE_COS: Float64 = 0.38
"""⚠ ABOVE WHAT STRANGERS REACH. Two different speakers score under 0.31 to
0.40 in 99 % of trials (plan §14.3), and at 0.30 a second voice that scored
0.31 to 0.36 against the first was told the first one's name."""


def unit(v: List[Float32]) -> List[Float32]:
    """`v` scaled to length 1 (as it is when its length is 0)."""
    var s = 0.0
    for x in v:
        s += Float64(x) * Float64(x)
    var out = List[Float32](capacity=len(v))
    var inv = 1.0 / sqrt(s) if s > 0.0 else 1.0
    for x in v:
        out.append(Float32(Float64(x) * inv))
    return out^


def dot(a: List[Float32], b: List[Float32]) -> Float64:
    var s = 0.0
    for i in range(min(len(a), len(b))):
        s += Float64(a[i]) * Float64(b[i])
    return s


def _is_name_byte(c: UInt8) -> Bool:
    var x = Int(c)
    return (x >= 97 and x <= 122) or (x >= 65 and x <= 90) or x >= 0x80 or x == 45 or x == 39


def name_given(text: String) -> String:
    """The name a speaker gives for THEMSELVES in `text`, or "". The forms:
    je m'appelle X, mon nom est X, moi c'est X, appelle-moi X, appelez-moi X,
    my name is X, call me X, i'm called X. X is the next word (two when the
    second is capitalised: "Jean Dupont"), without the punctuation after it,
    with a capital first letter."""
    var lowered = normalise(text)
    var raw = List[UInt8]()
    for x in text.as_bytes():
        raw.append(x)
    # `normalise` keeps the byte count except for typographic apostrophes and no-break spaces; a
    # sentence with one of those is searched in its normalised form, whose capitals are gone
    var same = len(lowered) == len(raw)
    var forms: List[String] = [
        "je m'appelle ", "mon nom est ", "moi c'est ", "moi, c'est ", "appelle-moi ", "appelez-moi ",
        "my name is ", "call me ", "i'm called ", "i am called ",
    ]
    var best = -1
    var best_len = 0
    for f in forms:
        var pat = List[UInt8]()
        for x in f.as_bytes():
            pat.append(x)
        var n = len(lowered)
        var m = len(pat)
        for i in range(n - m + 1):
            var hit = True
            for k in range(m):
                if lowered[i + k] != pat[k]:
                    hit = False
                    break
            # the form starts a word: "il m'appelle" is fine, "qu'il s'appelle" has no "je m'appelle"
            if hit and (i == 0 or not _is_name_byte(lowered[i - 1])):
                if best < 0 or i < best:
                    best = i
                    best_len = m
                break
    if best < 0:
        return String("")
    var src = raw.copy() if same else lowered.copy()
    var at = best + best_len
    var words = List[String]()
    while at < len(src) and len(words) < 2:
        while at < len(src) and Int(src[at]) == 32:
            at += 1
        var w = List[UInt8]()
        while at < len(src) and _is_name_byte(src[at]):
            w.append(src[at])
            at += 1
        if len(w) == 0:
            break
        var first = Int(w[0])
        var capital = (first >= 65 and first <= 90) or (first == 0xC3 and len(w) > 1 and Int(w[1]) >= 0x80 and Int(w[1]) <= 0x9E)
        if len(words) == 1 and not capital:
            break      # "Denis et toi": the second word is not part of the name
        if first >= 97 and first <= 122:
            w[0] = UInt8(first - 32)
        words.append(String(unsafe_from_utf8=Span(w)))
        if at < len(src) and Int(src[at]) != 32:
            break      # punctuation ends the name
    if len(words) == 0:
        return String("")
    var stop: List[String] = ["Le", "La", "Un", "Une", "The", "A", "Not", "Pas", "Ici", "Here"]
    for s in stop:
        if words[0] == s:
            return String("")
    return words[0] if len(words) == 1 else words[0] + " " + words[1]


def asks_identity(text: String) -> Bool:
    """Whether a sentence asks the robot who the speaker is: their name, or
    whether it recognises them (French and English)."""
    var t = normalise(text)
    var forms: List[String] = [
        "comment je m'appelle", "comment est-ce que je m'appelle", "quel est mon nom", "quel est mon prénom", "qui suis-je",
        "qui je suis", "tu me reconnais", "me reconnais-tu", "tu sais comment je m'appelle", "c'est quoi mon nom",
        "what is my name", "what's my name", "who am i", "who i am", "do you know my name", "do you recognise me",
        "do you recognize me", "do you remember me", "tu te souviens de moi",
    ]
    for f in forms:
        var pat = normalise(f)
        if has_phrase(t, pat):
            return True
    return False


def identity_answer(ref ws: WorldState) -> String:
    """The exact sentence that answers "who am I?" from the state, in the
    conversation's language (French, else English).

    ⚠ A TEMPLATE, AND THE ONE PLACE THE PLAN'S Q4 CALLS FOR ONE. Asked « Comment
    je m'appelle ? » with the speaker's name in its prompt, the reply model
    said « Je m'appelle Noé »; asked by a stranger, it handed out the name it
    knew (the restart gate, docs/ROBOT_BRAIN_PLAN.md B4). Who someone is, is
    not a sentence to leave to a 1.7 B model."""
    var fr = ws.language == "fr"
    if ws.speaker_pending:
        return String("Un instant, je n'ai pas encore reconnu ta voix.") if fr else String("One moment, I have not recognised your voice yet.")
    if ws.speaker != "" and ws.speaker_sure:
        return ("Tu t'appelles " + ws.speaker + ".") if fr else ("Your name is " + ws.speaker + ".")
    if ws.speaker != "":
        return ("Je crois que tu es " + ws.speaker + ", mais je n'en suis pas sûr.") if fr else (
            "I think you are " + ws.speaker + ", but I am not sure.")
    if len(ws.people_known) > 0:
        return String("Je ne reconnais pas ta voix.") if fr else String("I do not recognise your voice.")
    return String("Je ne sais pas : tu ne m'as pas encore dit ton nom.") if fr else String(
        "I do not know: you have not told me your name yet.")


struct Person(Copyable, Movable):
    var id: String
    var name: String
    var since: Int
    """Epoch seconds: when the name was first given."""
    var last_heard: Int
    var times_heard: Int
    var voices: List[List[Float32]]
    """Stored voice embeddings, each of length 1."""

    def __init__(out self, var id: String, var name: String, now: Int):
        self.id = id^
        self.name = name^
        self.since = now
        self.last_heard = now
        self.times_heard = 0
        self.voices = List[List[Float32]]()

    def score(self, emb: List[Float32]) -> Float64:
        """The best cosine of a unit `emb` with this person's stored voices."""
        var best = -1.0
        for v in self.voices:
            best = max(best, dot(v, emb))
        return best

    def add_voice(mut self, emb: List[Float32]):
        if len(self.voices) < VOICES_MAX:
            self.voices.append(emb.copy())
        else:
            # keep the first (the introduction itself), turn the others over
            _ = self.voices.pop(1)
            self.voices.append(emb.copy())


struct People(Movable):
    var dir: String
    """The robot's folder; "" = nothing is written."""
    var list: List[Person]
    var next_id: Int
    var sure: Float64
    var maybe: Float64
    var pending_name: String
    """The name given in the utterance whose voice has not come yet."""
    var just_met: String
    """The name a voice has just been stored or renamed under, until the
    caller takes it (`take_met`) to acknowledge the introduction."""
    var error: String

    def __init__(out self, var dir: String):
        self.dir = dir^
        self.list = List[Person]()
        self.next_id = 1
        self.sure = SURE_COS
        self.maybe = MAYBE_COS
        self.pending_name = String("")
        self.just_met = String("")
        self.error = String("")

    # ── the store ─────────────────────────────────────────────────────

    @staticmethod
    def load(dir: String) -> People:
        """The people of `<dir>/people`; empty when there are none. ⚠ Does not
        raise: what cannot be read is in `error`."""
        var p = People(dir.copy())
        var index = dir + "/people/index.json"
        if not exists(index):
            return p^
        try:
            var doc = parse_json(read_file_bytes(index))
            var root = doc.root()
            p.next_id = doc.integer(doc.field(root, "next_id"))
            var dim = doc.integer(doc.field(root, "dim"))
            var arr = doc.field(root, "people")
            for i in range(doc.size(arr)):
                var e = doc.at(arr, i)
                var person = Person(doc.string(doc.field(e, "id")), doc.string(doc.field(e, "name")), doc.integer(doc.field(e, "since")))
                person.last_heard = doc.integer(doc.field(e, "last_heard"))
                person.times_heard = doc.integer(doc.field(e, "times_heard"))
                var count = doc.integer(doc.field(e, "voices"))
                var raw = read_file_bytes(dir + "/people/" + person.id + ".voice")
                if dim <= 0 or len(raw) != count * dim * 4:
                    raise Error("people: " + person.id + ".voice holds " + String(len(raw)) + " bytes, " + String(count * dim * 4) + " expected")
                var fp = raw.unsafe_ptr().unsafe_bitcast[Float32]()
                for v in range(count):
                    var emb = List[Float32](capacity=dim)
                    for k in range(dim):
                        emb.append(fp[unsafe_offset=v * dim + k])
                    person.voices.append(emb^)
                p.list.append(person^)
        except e:
            p.error = String(e)
        return p^

    def save(mut self):
        """Write the index and every person's voices (atomically, each file)."""
        if self.dir == "":
            return
        try:
            makedirs(self.dir + "/people", exist_ok=True)
            var dim = 0
            var w = JsonWriter()
            w.begin_object()
            w.key("next_id")
            w.integer(self.next_id)
            w.key("people")
            w.begin_array()
            for person in self.list:
                if len(person.voices) > 0:
                    dim = len(person.voices[0])
                w.begin_object()
                w.member("id", person.id)
                w.member("name", person.name)
                w.key("since")
                w.integer(person.since)
                w.key("last_heard")
                w.integer(person.last_heard)
                w.key("times_heard")
                w.integer(person.times_heard)
                w.key("voices")
                w.integer(len(person.voices))
                w.end_object()
                var bytes = List[UInt8](capacity=len(person.voices) * dim * 4)
                for v in person.voices:
                    var bp = v.unsafe_ptr().unsafe_bitcast[UInt8]()
                    for k in range(len(v) * 4):
                        bytes.append(bp[unsafe_offset=k])
                write_file_atomic(self.dir + "/people/" + person.id + ".voice", bytes)
            w.end_array()
            w.key("dim")
            w.integer(dim)
            w.end_object()
            write_text_atomic(self.dir + "/people/index.json", w.done() + "\n")
        except e:
            self.error = String(e)

    def forget(mut self, name: String) -> Bool:
        """Remove a person and their voice file. False when nobody has that name."""
        for i in range(len(self.list)):
            if self.list[i].name == name:
                var id = self.list[i].id.copy()
                _ = self.list.pop(i)
                if self.dir != "":
                    try:
                        remove_file(self.dir + "/people/" + id + ".voice")
                    except:
                        pass
                    self.save()
                return True
        return False

    # ── an utterance ──────────────────────────────────────────────────

    def find(self, name: String) -> Int:
        for i in range(len(self.list)):
            if self.list[i].name == name:
                return i
        return -1

    def best(self, emb: List[Float32]) -> Tuple[Int, Float64]:
        """The known person whose voice is closest to a unit `emb`, and that cosine."""
        var who = -1
        var top = -1.0
        for i in range(len(self.list)):
            var s = self.list[i].score(emb)
            if s > top:
                top = s
                who = i
        return (who, top)

    def heard(mut self, mut ws: WorldState, text: String, now: Int):
        """The utterance's words: a name given in them is kept until its voice
        comes (`voice`), and told to the reply view at once. Until then the
        robot does not know who spoke (`ws.speaker_pending`): the last
        speaker's name is not handed to whoever talks next."""
        self.pending_name = name_given(text)
        ws.speaker_said_name = self.pending_name.copy()
        ws.speaker_pending = True

    def voice(mut self, mut ws: WorldState, embedding: List[Float32], now: Int):
        """The voice of the utterance last given to `heard`: enrol or rename
        when a name came with it, else recognise."""
        ws.speaker_pending = False
        var emb = unit(embedding)
        var found = self.best(emb)
        var who = found[0]
        var score = found[1]
        if self.pending_name != "":
            var name = self.pending_name.copy()
            self.pending_name = String("")
            ws.speaker_said_name = String("")
            if who >= 0 and score >= self.sure:
                # a voice already known: the name is confirmed, or corrected
                if self.list[who].name != name:
                    ws.record(EV_MET, name + " | the voice known as " + self.list[who].name + " gave this name", String("people"), now)
                    self.list[who].name = name.copy()
                self.list[who].add_voice(emb)
            else:
                var same_name = self.find(name)
                if same_name >= 0:
                    # the name is known, the voice is not close to it: another way this person sounds
                    who = same_name
                    self.list[who].add_voice(emb)
                else:
                    var person = Person("p" + String(self.next_id), name.copy(), now)
                    self.next_id += 1
                    person.voices.append(emb.copy())
                    self.list.append(person^)
                    who = len(self.list) - 1
                    ws.record(EV_MET, name, String("people"), now)
            self._recognised(ws, who, True, 1.0, now)
            self.just_met = name.copy()
            self.save()
            return
        if who >= 0 and score >= self.sure:
            if score < 0.95:
                self.list[who].add_voice(emb)     # another sample of a voice already sure
            self._recognised(ws, who, True, score, now)
            self.save()
        elif who >= 0 and score >= self.maybe:
            self._recognised(ws, who, False, score, now)
        else:
            ws.speaker = String("")
            ws.speaker_sure = False
            ws.speaker_score = score
            ws.speaker_since = 0
            ws.record(EV_SPEAKER, String("unknown | ") + String(round(score, 2)), String("people"), now)
        ws.people_known = self.names()

    def _recognised(mut self, mut ws: WorldState, who: Int, sure: Bool, score: Float64, now: Int):
        self.list[who].last_heard = now
        self.list[who].times_heard += 1
        ws.speaker = self.list[who].name.copy()
        ws.speaker_sure = sure
        ws.speaker_score = score
        ws.speaker_since = self.list[who].since
        ws.people_known = self.names()
        ws.record(EV_SPEAKER, self.list[who].name + " | " + (String("sure ") if sure else String("maybe ")) + String(round(score, 2)), String("people"), now)

    def take_met(mut self, language: String) -> String:
        """What to say to a person who has just given their name, once
        (French, else English); "" when nobody has. A template: the name is
        said back exactly, so a misheard one is heard and can be corrected."""
        if self.just_met == "":
            return String("")
        var name = self.just_met.copy()
        self.just_met = String("")
        return ("Enchanté, " + name + ".") if language == "fr" else ("Nice to meet you, " + name + ".")

    def names(self) -> List[String]:
        var out = List[String]()
        for p in self.list:
            out.append(p.name.copy())
        return out^

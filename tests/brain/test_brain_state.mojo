"""`noeira.brain`, step B1: the state, its record and its views (no model, no GPU).

    python3 tools/brain/make_bb1.py                                  # the reference (committed)
    pixi run mojo run -I . tests/brain/test_brain_state.mojo

1. GEOMETRY AND REDUCERS against `tools/brain/make_bb1.py`'s own arithmetic,
   on every BB-1 scenario: the place lines, the nearest place, where the robot
   was before.
2. `observe_motion` records the transitions, and only them.
3. The event log: what is appended is read back equal, awkward text included,
   and a second append adds a line without touching the first.
4. The reply view's text for a hand-written state, both styles, as literals.
5. The scoring rule on cases with known answers: word boundaries, numbers,
   accents, groups, the forbidden side.
6. `spoken_text` (what a voice can say) and the language check.
7. The executive (`noeira.brain.tasks`) on scripted episodes with a body that
   is only the facts it reports: steps start in order and one at a time, an
   arrival ends a step, a deadline fails the task and says so, a new request
   cancels the one in hand, a refused command fails its step, and the reply
   view's task line is right at each stage.
8. Memory across restarts (`noeira.brain.memory`): a day of the log folded by
   rule, three processes sharing one robot folder (yesterday, this morning,
   this afternoon), the digest file, a torn and a damaged log line, the last
   days only, the places file.
9. People (`noeira.brain.people`): a name given in the first person and
   nothing else; a voice stored under it, recognised, resembling, unknown;
   a second person; a rename; another process reading the store back;
   forgetting; a damaged voice file.
10. The answers code gives: whether the task is finished, who the speaker is.
11. What the body can do (`computed_turn`): « Tu peux faire quoi ? » listed
    from the skills, the motions among them and the reply view's line with
    them; « Tu peux danser ? » / « Danse ! » run the motion, « Tu peux ouvrir
    la porte ? » is a no, and what the decision must read (a place, a next
    step, a direction, a thing in neither list, a statement) is left to it.
12. How long the model's reply may be (`reply_sentences`): one sentence, two
    when a joke, a story or an explanation is asked for.
Exits non-zero on a failure.
"""

from std.os import getenv
from std.os.path import exists
from std.time import perf_counter_ns
from std.sys import exit

from noeira.brain import (
    People, name_given, asks_identity, identity_answer, speaker_sentence,
    asks_task_done, task_done_answer, computed_reply, computed_turn, asks_capabilities, asks_can, capabilities_answer,
    reply_stable, asks_for_more, reply_sentences, REPLY_SENTENCES, REPLY_SENTENCES_MORE,
    restore_memory, digest_lines, log_days, when_words, save_places, load_places, time_refs, focused_memory,
    Task, Step, give_task, cancel_task, refuse_step, tick_tasks, notice_text, task_sentences,
    EX_NONE, EX_START, EX_NOTICE, NOTICE_ALL_DONE, NOTICE_NOT_REACHED, NOTICE_REFUSED,
    TASK_DONE, TASK_FAILED, TASK_CANCELLED,
    WorldState, Body, PlaceXY, BrainEvent, EventLog, load_events, event_json, parse_event,
    reply_view, REPLY_VIEW_SENTENCES, metres,
    EV_BEGAN, EV_GOING, EV_ARRIVED, EV_HEARD, EV_SESSION,
)
from noeira.brain.bb1 import BB1Set, reply_passes, reply_in_language
from noeira.brain.memory import date_days
from noeira.core.run import date_utc
from noeira.text.spoken import spoken_text
from noeira.core.bytes import string_from_bytes
from noeira.io.fileio import read_file_bytes, remove_file


def check(mut fails: Int, ok: Bool, what: String):
    if ok:
        print("  ok  :", what)
    else:
        print("  FAIL:", what)
        fails += 1


def join(items: List[String], sep: String) -> String:
    var t = String("")
    for i in range(len(items)):
        if i > 0:
            t += sep
        t += items[i]
    return t


def kinds(ref ws: WorldState) -> String:
    var t = String("")
    for e in ws.events:
        t += e.kind + ":" + e.text + " "
    return t


def main() raises:
    var fails = 0

    print("1. geometry and reducers against the Python reference")
    var set = BB1Set.load("tests/brain")
    var bad_lines = 0
    var bad_near = 0
    var bad_before = 0
    for s in set.scenarios:
        var ws = s.state(Body.g1_room("Noé"))
        if join(ws.around_lines(), " ; ") != s.ref_lines:
            bad_lines += 1
            print("     ", s.sid, "lines:", join(ws.around_lines(), " ; "), "| reference:", s.ref_lines)
        if ws.nearest() != s.ref_nearest:
            bad_near += 1
            print("     ", s.sid, "nearest:", ws.nearest(), "| reference:", s.ref_nearest)
        if join(ws.been_before(), ",") != join(s.ref_before, ","):
            bad_before += 1
            print("     ", s.sid, "before:", join(ws.been_before(), ","), "| reference:", join(s.ref_before, ","))
    var n = len(set.scenarios)
    check(fails, n >= 8 and len(set.items) >= 100, String(n) + " scenarios and " + String(len(set.items)) + " items loaded")
    check(fails, bad_lines == 0, "place lines equal on " + String(n - bad_lines) + " of " + String(n))
    check(fails, bad_near == 0, "nearest place equal on " + String(n - bad_near) + " of " + String(n))
    check(fails, bad_before == 0, "where it was before equal on " + String(n - bad_before) + " of " + String(n))
    # a mutant of the reference's rule: the scenario's heading turned by 90 degrees must move the lines
    var turned = 0
    for s in set.scenarios:
        var ws = s.state(Body.g1_room("Noé"))
        ws.observe_pose(s.x, s.y, s.yaw + 90.0)
        if join(ws.around_lines(), " ; ") != s.ref_lines:
            turned += 1
    check(fails, turned == n, "a heading 90 degrees off changes the lines on " + String(turned) + " of " + String(n))

    print("2. transitions become events")
    var ws = WorldState(Body.g1_room("Noé"), String("a furnished room"))
    ws.observe_motion(String("stand"), String("none"), String("none"), 10)
    ws.observe_motion(String("stand"), String("none"), String("none"), 11)
    ws.observe_motion(String("goto:sofa"), String("sofa"), String("none"), 12)
    ws.observe_motion(String("goto:sofa"), String("sofa"), String("none"), 13)
    ws.observe_motion(String("goto:sofa"), String("none"), String("sofa"), 14)
    ws.observe_motion(String("goto:door"), String("door"), String("none"), 15)
    ws.observe_motion(String("goto:door"), String("none"), String("door"), 16)
    ws.observe_motion(String("goto:door"), String("none"), String("door"), 17)
    check(fails, kinds(ws) == "began:stand began:goto:sofa going:sofa arrived:sofa began:goto:door going:door arrived:door ",
          "eight frames, seven events: " + kinds(ws))
    check(fails, join(ws.trail(), ",") == "door,sofa" and join(ws.been_before(), ",") == "sofa", "trail door,sofa; before: sofa")
    check(fails, ws.at == "door" and ws.walking_to == "" and ws.doing == "goto:door", "the facts of the last frame")
    check(fails, len(ws.around_lines()) == 0 and ws.nearest() == "", "no pose: nothing around, no nearest")

    print("3. the event log")
    var dir = getenv("TMPDIR", "/tmp") + "/noeira_brain_test"
    var log = EventLog(dir)
    var t = 1794000000  # 2026-11-06T18:40:00Z
    var path = log.path_for(t)
    try:
        remove_file(path)
    except:
        pass
    var awkward = String("Il a dit : \"va à la table\"\n\tpuis \\ rien — 冰箱")
    log.append(BrainEvent(t, String(EV_HEARD), awkward, String("voice")))
    var one = string_from_bytes(read_file_bytes(path))
    log.append(BrainEvent(t + 5, String(EV_ARRIVED), String("table"), String("motion")))
    var two = string_from_bytes(read_file_bytes(path))
    var back = load_events(path)
    check(fails, path.endswith("/log/2026-11-06.jsonl"), "the file is the UTC day's: " + path)
    check(fails, len(back) == 2 and back[0].text == awkward and back[0].kind == "heard" and back[0].t == t
          and back[0].source == "voice" and back[1].text == "table" and back[1].t == t + 5, "two events read back equal")
    check(fails, two.startswith(one) and two.byte_length() > one.byte_length(), "the second append left the first line as it was")
    var e = parse_event(event_json(BrainEvent(7, String("later_kind"), String("x"), String("y"))))
    check(fails, e.kind == "later_kind" and e.t == 7, "an unknown kind goes through")
    var ws2 = WorldState(Body.g1_room("Noé"), String("a furnished room"))
    ws2.open_log(dir + "/state", t)
    ws2.heard(String("bonjour"), t + 1)
    var logged = load_events(dir + "/state/log/2026-11-06.jsonl")
    check(fails, len(logged) >= 2 and logged[len(logged) - 2].kind == EV_SESSION and logged[len(logged) - 1].text == "bonjour"
          and ws2.log_error == "", "the state writes its events to its log")
    var view_before = reply_view(ws2).text()
    ws2.missed(1.24, t + 2)
    var logged2 = load_events(dir + "/state/log/2026-11-06.jsonl")
    check(fails, len(logged2) == len(logged) + 1 and logged2[len(logged2) - 1].kind == "missed" and logged2[len(logged2) - 1].text == "1.2 s"
          and logged2[len(logged2) - 1].source == "voice" and logged2[len(logged2) - 1].t == t + 2,
          "speech the robot did not listen to is one line of the log: its length, nothing else")
    check(fails, reply_view(ws2).text() == view_before, "and the reply view does not change for it")
    var ws3 = WorldState(Body.g1_room("Noé"), String("a furnished room"))
    ws3.open_log(String("/nonexistent_root_dir/x"), t)
    check(fails, ws3.log_error != "" and len(ws3.events) == 1, "an unwritable log is reported, the event is kept: " + ws3.log_error)

    print("4. the reply view")
    var w = WorldState(Body.g1_room("Noé"), String("a furnished room"))
    w.places = [PlaceXY("fridge", 3.0, 0.0), PlaceXY("door", 0.0, 4.0), PlaceXY("table", 0.4, -0.5), PlaceXY("sofa", -4.5, 0.0)]
    w.observe_motion(String("goto:door"), String("none"), String("door"), 1)
    w.observe_motion(String("goto:sofa"), String("none"), String("sofa"), 2)
    w.observe_motion(String("goto:table"), String("none"), String("table"), 3)
    w.observe_pose(0.0, 0.0, 0.0)
    var v = reply_view(w)
    var stable = (
        "You are Noé, a small humanoid robot with two legs, two arms and a head. You are in a furnished room"
        " and you talk with the people around you.\n"
        "Reply in ONE short sentence, in the language you were addressed in, in plain words: no emoji. Be warm and brief.\n"
        "You can: walk to a place of the room; turn; squat; raise your arms; talk and answer questions.\n"
        "That list is complete. Asked whether you can do anything else, answer no, kindly."
        " In particular you cannot: pick things up or carry them; open the door; sit down.\n"
        "Answer questions about the room and about yourself ONLY from the facts below. If the answer is not there,"
        " say you do not know."
    )
    var live = (
        "Facts, true right now:\n"
        "- You are standing at the table.\n"
        "- You have no task right now.\n"
        "- Places you know, from where you stand: fridge: ahead of you, 3 m; door: on your left, 4 m; table: here;"
        " sofa: behind you, 4.5 m.\n"
        "- The nearest place to you: table.\n"
        "- Before this you were at: sofa, then door.\n"
        "- You know of no other place or object in this room."
    )
    check(fails, v.stable == stable, "the stable part")
    if v.stable != stable:
        print(v.stable)
    check(fails, v.live == live, "the live part, as lines")
    if v.live != live:
        print(v.live)
    check(fails, v.text() == stable + "\n\n" + live, "text = stable, a blank line, live")
    var sent = (
        "Right now: You are standing at the table. You have no task right now. The fridge is ahead of you, 3 m away. The door is on your left, 4 m away."
        " The sofa is behind you, 4.5 m away. The nearest place to you is the table. Before this you were at the sofa,"
        " and before that at the door. You know of no other place or object in this room."
    )
    var vs = reply_view(w, REPLY_VIEW_SENTENCES)
    check(fails, vs.live == sent and vs.stable == stable, "the live part, as sentences")
    if vs.live != sent:
        print(vs.live)
    w.observe_motion(String("goto:fridge"), String("fridge"), String("none"), 4)
    check(fails, reply_view(w).live.find("- You are walking to the fridge.\n") >= 0
          and reply_view(w).live.find("Before this you were at: table, then sofa, then door.") >= 0, "walking: the line and the trail follow")
    w.observe_motion(String("spin_left"), String("none"), String("none"), 5)
    check(fails, reply_view(w).live.find("not at a named place. Your current movement: spin left.") >= 0, "a bank command is named")
    w.language = String("fr")
    w.call_place(String("table"), String("fr"), String("la table"))
    w.call_place(String("sofa"), String("fr"), String("le canapé"))
    w.call_place(String("sofa"), String("de"), String("das Sofa"))
    check(fails, reply_view(w).live.find("m.\n- In French these places are: table = la table; sofa = le canapé.\n- The nearest") >= 0
          and reply_view(w, REPLY_VIEW_SENTENCES).live.find(" In French these places are: table = la table; sofa = le canapé. The nearest") >= 0,
          "the places' names in the conversation's language")
    check(fails, reply_view(w).stable.find("Reply in ONE short sentence, in French, in plain words: no emoji.") >= 0, "a known language is named")
    check(fails, metres(3.0) == "3 m" and metres(3.5) == "3.5 m" and metres(0.0) == "0 m", "metres: 3 m, 3.5 m, 0 m")
    var head = WorldState(Body(String("Noé"), String("a robot head on a tripod")), String("a study"))
    head.body.can.append(String("look at someone"))
    check(fails, reply_view(head).stable.find("You can: look at someone.\nThat list is complete. Asked whether you can do anything else, answer no, kindly.\nAnswer") >= 0
          and reply_view(head).live.find("Places you know") < 0, "another body: its own list, no cannot line, no places")

    print("5. the scoring rule")
    var D = String("-")
    check(fails, reply_passes("Le frigo est à ma droite, à 3 mètres.", "droite|my right", "gauche|derrière"), "a phrase found")
    check(fails, not reply_passes("Le frigo est à ma gauche.", "droite|my right", D), "a phrase missing")
    check(fails, not reply_passes("À droite, pas à gauche.", "droite", "gauche"), "a forbidden phrase fails it")
    check(fails, not reply_passes("Je vais vous annoncer la nouvelle.", "non|no", D), "'non' is not inside 'annoncer'")
    check(fails, reply_passes("Non, désolé.", "non|no", D), "'Non,' is found")
    check(fails, not reply_passes("Il va tout droit.", "droite", D), "'droite' is not 'droit'")
    check(fails, reply_passes("DERRIÈRE MOI", "derrière", D) and reply_passes("À ma droite", "à ma droite", D), "capitals, accented ones too")
    check(fails, reply_passes("Je n’en sais rien, je ne sais pas.", "ne sais pas", D)
          and reply_passes("I don’t know.", "don't know", D), "a typographic apostrophe")
    check(fails, reply_passes("Il est à 3,5 m.", "num:3.5", D) and reply_passes("It is 3.5 m away.", "num:3.5", D), "3,5 and 3.5")
    check(fails, not reply_passes("Il est à 3,5 m.", "num:3", D) and reply_passes("Il est à 3 m.", "num:3", D)
          and reply_passes("À 3.", "num:3", D), "3 is not in 3,5; '3.' at a sentence's end is 3")
    check(fails, not reply_passes("Il est à 13 m.", "num:3", D) and not reply_passes("Il est à 30 m.", "num:3", D), "3 is not in 13 or 30")
    check(fails, reply_passes("à trois mètres et demi", "num:3.5|trois mètres et demi", D), "an alternative in words")
    check(fails, reply_passes("devant, à 2 m", "devant|ahead && num:2", D) and not reply_passes("devant", "devant|ahead && num:2", D),
          "every group is needed")
    check(fails, reply_passes("Bonjour !", D, "ne peux pas") and not reply_passes("   ", D, D) and not reply_passes("", D, D),
          "no requirement passes a reply, never an empty one")
    check(fails, reply_passes("no tv here", "no tv", D) and not reply_passes("no tvs here", "no tv", D), "a phrase ends on a word boundary")

    print("6. the reply as it is spoken, and its language")
    check(fails, spoken_text("Bien sûr ! Parce qu'ils n'ont jamais eu de cœur ! 😄") == "Bien sûr ! Parce qu'ils n'ont jamais eu de cœur !",
          "an emoji at the end goes, the accents stay")
    check(fails, spoken_text("Two arms! 👋🏽 I'm **Noé** ❤️, a `robot` → here ⭐.") == "Two arms! I'm Noé , a robot here .",
          "emoji with modifiers, markup and symbols go: '" + spoken_text("Two arms! 👋🏽 I'm **Noé** ❤️, a `robot` → here ⭐.") + "'")
    check(fails, spoken_text("  Le frigo est à 3,5 m — devant moi…  ") == "Le frigo est à 3,5 m — devant moi…", "dashes, ellipsis and digits stay; the ends are trimmed")
    check(fails, spoken_text("冰箱 in the room") == "冰箱 in the room" and spoken_text("") == "", "other scripts stay; empty stays empty")
    check(fails, reply_in_language("Le frigo est devant moi.", "fr") and not reply_in_language("Just before I was at the door.", "fr"),
          "a French question answered in English is seen")
    check(fails, reply_in_language("The fridge is ahead of me.", "en") and not reply_in_language("Le frigo est devant moi.", "en")
          and reply_in_language("anything", "de"), "and the other way; an unlisted language passes")

    print("7. the executive")
    var x = WorldState(Body.g1_room("Noé"), String("a furnished room"))
    x.language = String("fr")
    x.call_place(String("sofa"), String("fr"), String("le canapé"))
    check(fails, tick_tasks(x, 0.0, 100).kind == EX_NONE and task_sentences(x) == "You have no task right now.", "no task: nothing to do")
    var trio = Task(String("va à la table puis à la porte"))
    trio.steps.append(Step.goto(String("table")))
    trio.steps.append(Step.gesture(String("both_hands_up")))
    trio.steps.append(Step.goto(String("door")))
    give_task(x, trio^, 0.0, 100)
    var a1 = tick_tasks(x, 0.0, 100)
    check(fails, a1.kind == EX_START and a1.command == "goto:table", "the first step starts: " + a1.command)
    check(fails, task_sentences(x) == "Your task: walk to the table, then both hands up, then walk to the door."
          " You are on step 1 of 3: walk to the table. Next: both hands up.", "the task line on step 1")
    x.observe_motion(String("goto:table"), String("table"), String("none"), 101)
    check(fails, tick_tasks(x, 5.0, 105).kind == EX_NONE and tick_tasks(x, 9.0, 109).kind == EX_NONE, "walking: nothing new to do")
    x.observe_motion(String("goto:table"), String("none"), String("table"), 110)
    check(fails, tick_tasks(x, 10.0, 110).kind == EX_NONE and x.task.at == 1, "arrived: the step is done, the next is in hand")
    var a2 = tick_tasks(x, 10.1, 110)
    check(fails, a2.kind == EX_START and a2.command == "both_hands_up", "the second step starts on the next frame")
    check(fails, tick_tasks(x, 12.0, 112).kind == EX_NONE and x.task.at == 1, "a posture holds")
    check(fails, tick_tasks(x, 13.2, 113).kind == EX_NONE and x.task.at == 2, "and is done after its hold")
    var a3 = tick_tasks(x, 13.3, 113)
    check(fails, a3.kind == EX_START and a3.command == "goto:door"
          and task_sentences(x).find("You are on step 3 of 3: walk to the door. It is the last step.") >= 0, "the last step starts")
    x.observe_motion(String("goto:door"), String("none"), String("door"), 120)
    var a4 = tick_tasks(x, 20.0, 120)
    check(fails, a4.kind == EX_NOTICE and a4.notice == NOTICE_ALL_DONE and notice_text(x, a4) == "C'est fait." and not x.has_task
          and a4.command == "" and x.last_task.status == TASK_DONE, "a task of several steps says it is done, and stops nothing")
    check(fails, task_sentences(x) == "You have no task right now. Your last task: walk to the table, then both hands up, then walk to the door. It is done.",
          "the task line afterwards")
    var order = String("")
    for ev in x.events:
        if ev.kind.startswith("task_") or ev.kind.startswith("step_"):
            order += ev.kind + " "
    check(fails, order == "task_started step_started step_done step_started step_done step_started step_done task_done ",
          "the record: " + order)
    # a place never reached
    var far = Task(String(""))
    far.steps.append(Step.goto(String("sofa")))
    give_task(x, far^, 30.0, 130)
    check(fails, tick_tasks(x, 30.0, 130).command == "goto:sofa" and task_sentences(x) == "Your task: walk to the sofa. It is in progress.", "one step: in progress")
    x.observe_motion(String("goto:sofa"), String("sofa"), String("none"), 131)
    check(fails, tick_tasks(x, 89.9, 189).kind == EX_NONE, "59.9 s: still walking")
    var a5 = tick_tasks(x, 90.1, 190)
    check(fails, a5.kind == EX_NOTICE and a5.notice == NOTICE_NOT_REACHED and a5.command == "stop"
          and notice_text(x, a5) == "Je n'ai pas réussi à atteindre le canapé."
          and x.last_task.status == TASK_FAILED, "past the deadline the task fails and says so: " + notice_text(x, a5))
    check(fails, task_sentences(x) == "You have no task right now. Your last task: walk to the sofa. It failed: the sofa was not reached after 60 seconds.",
          "the failure in the task line")
    x.language = String("en")
    check(fails, notice_text(x, a5) == "I could not reach the sofa.", "the notice in English, the place without a French word")
    # a new request replaces the one in hand; one step alone ends silently
    var t1 = Task(String(""))
    t1.steps.append(Step.goto(String("fridge")))
    give_task(x, t1^, 100.0, 200)
    _ = tick_tasks(x, 100.0, 200)
    var t2 = Task(String(""))
    t2.steps.append(Step.goto(String("door")))
    give_task(x, t2^, 101.0, 201)
    check(fails, x.last_task.status == TASK_CANCELLED and x.last_task.describe() == "walk to the fridge"
          and x.task.describe() == "walk to the door" and tick_tasks(x, 101.0, 201).command == "goto:door", "the last request wins")
    x.observe_motion(String("goto:door"), String("none"), String("door"), 205)
    check(fails, tick_tasks(x, 105.0, 205).kind == EX_NONE and not x.has_task and x.last_task.status == TASK_DONE,
          "a single step ends at its arrival and says nothing")
    var again = Task(String(""))
    again.steps.append(Step.goto(String("door")))
    give_task(x, again^, 106.0, 206)
    check(fails, tick_tasks(x, 106.0, 206).kind == EX_START and tick_tasks(x, 106.1, 206).kind == EX_NONE and not x.has_task
          and x.last_task.status == TASK_DONE, "asked for the place it is at: done on the next frame")
    # a command the body does not have
    var t3 = Task(String(""))
    t3.steps.append(Step.gesture(String("fly")))
    give_task(x, t3^, 110.0, 210)
    _ = tick_tasks(x, 110.0, 210)
    var a6 = refuse_step(x, 110.0, 210)
    check(fails, a6.kind == EX_NOTICE and a6.notice == NOTICE_REFUSED and x.last_task.status == TASK_FAILED
          and x.last_task.reason == "the robot has no command 'fly'" and notice_text(x, a6) == "I do not know how to do that.",
          "a refused command fails its task")
    cancel_task(x, String("stop"), 111.0, 211)
    give_task(x, Task(String("")), 112.0, 212)
    check(fails, not x.has_task, "cancel with nothing in hand, and a task with no step, change nothing")
    # every BB-1 situation's task line, against its `task` column
    var bad_task = 0
    for s in set.scenarios:
        var ws = s.state(Body.g1_room("Noé"))
        var line = task_sentences(ws)
        var want = String("You have no task right now.")
        if s.task.startswith("run:"):
            want = String("Your task: ")
        elif s.task.startswith("done:"):
            want = String("It is done.")
        elif s.task.startswith("failed:"):
            want = String("It failed: ")
        if line.find(want) < 0 or (s.task == "-" and line != want):
            bad_task += 1
            print("     ", s.sid, s.task, "|", line)
    check(fails, bad_task == 0, "the BB-1 situations' tasks: " + String(n - bad_task) + " of " + String(n) + " as written")

    print("8. memory across restarts")
    check(fails, date_days("1970-01-01") == 0 and date_days("2026-11-13") == 20770 and date_days("2024-02-29") == 19782
          and date_days("1969-12-31") == -1 and date_days(date_utc(20770 * 86400 + 86399)) == 20770, "dates to days and back")
    check(fails, when_words(0, "2026-11-13") == "Earlier today, before you were restarted" and when_words(1, "2026-11-12") == "Yesterday"
          and when_words(3, "2026-11-10") == "3 days ago (2026-11-10)", "when, in words")
    var evs = List[BrainEvent]()
    evs.append(BrainEvent(1, String(EV_SESSION), String("Noé"), String("brain")))
    evs.append(BrainEvent(2, String(EV_HEARD), String("Va au frigo."), String("voice")))
    evs.append(BrainEvent(3, String("task_started"), String("walk to the fridge"), String("executive")))
    evs.append(BrainEvent(4, String(EV_ARRIVED), String("door"), String("motion")))
    evs.append(BrainEvent(5, String("task_done"), String("walk to the fridge"), String("executive")))
    evs.append(BrainEvent(6, String("task_done"), String("walk to the table, then walk to the door"), String("executive")))
    evs.append(BrainEvent(7, String("task_done"), String("walk to the fridge"), String("executive")))
    evs.append(BrainEvent(8, String("task_failed"), String("walk to the sofa | the sofa was not reached after 60 seconds"), String("executive")))
    evs.append(BrainEvent(9, String("task_cancelled"), String("walk to the door | a new request replaced it"), String("executive")))
    evs.append(BrainEvent(10, String(EV_HEARD), String("Je m'appelle Denis."), String("voice")))
    evs.append(BrainEvent(11, String("said"), String("Enchanté."), String("voice")))
    var dl = digest_lines(evs)
    check(fails, len(dl) == 4 and dl[0] == "You did: walk to the fridge (2 times); walk to the table, then walk to the door."
          and dl[1] == "You tried and failed: walk to the sofa (the sofa was not reached after 60 seconds)."
          and dl[2] == "1 task was interrupted by a new request."
          and dl[3] == "The last things you were asked or told: \"Va au frigo.\"; a person told you their name.",
          "a day folded by rule; an arrival that was no task is not a memory")
    if len(dl) == 4:
        for l in dl:
            print("       ", l)
    var none_ev = List[BrainEvent]()
    none_ev.append(BrainEvent(1, String(EV_SESSION), String("Noé"), String("brain")))
    none_ev.append(BrainEvent(2, String(EV_ARRIVED), String("fridge"), String("motion")))
    check(fails, len(digest_lines(none_ev)) == 0, "a session where nothing was asked or done leaves no memory")

    var mdir = getenv("TMPDIR", "/tmp") + "/noeira_brain_mem_" + String(perf_counter_ns())
    var day = 20770 * 86400            # 2026-11-13T00:00:00Z
    # yesterday: a walk done, a walk failed, three things heard
    var y = WorldState(Body.g1_room("Noé"), String("a furnished room"))
    y.places = [PlaceXY("fridge", 3.0, 0.0), PlaceXY("sofa", -4.5, 0.0)]
    y.call_place(String("sofa"), String("fr"), String("le canapé"))
    check(fails, load_places(y, mdir) == 0 and len(y.places) == 2, "no places file: nothing read, nothing changed")
    save_places(y, mdir)
    restore_memory(y, mdir, day - 86400 + 3600)
    check(fails, y.memory_on and len(y.remembered) == 0 and y.memory_error == ""
          and reply_view(y).live.endswith("- You know of no other place or object in this room.\n\nWhat you remember from before this session"
          " (your own log, by day):\n- Nothing: you remember no earlier session and no earlier day."),
          "a first start: nothing remembered, and the view says so")
    y.open_log(mdir, day - 86400 + 3600)
    y.heard(String("Bonjour."), day - 86400 + 3601)
    y.heard(String("Je m'appelle Denis."), day - 86400 + 3602)
    y.said(String("Enchanté, Denis."), day - 86400 + 3603)
    y.heard(String("Va au frigo."), day - 86400 + 3604)
    var w1 = Task(String("Va au frigo."))
    w1.steps.append(Step.goto(String("fridge")))
    give_task(y, w1^, 0.0, day - 86400 + 3605)
    _ = tick_tasks(y, 0.0, day - 86400 + 3605)
    y.observe_motion(String("goto:fridge"), String("none"), String("fridge"), day - 86400 + 3610)
    _ = tick_tasks(y, 5.0, day - 86400 + 3610)
    var w2 = Task(String(""))
    w2.steps.append(Step.goto(String("sofa")))
    give_task(y, w2^, 10.0, day - 86400 + 3620)
    _ = tick_tasks(y, 10.0, day - 86400 + 3620)
    _ = tick_tasks(y, 71.0, day - 86400 + 3681)
    # this morning, another process: a task of two steps
    var m = WorldState(Body.g1_room("Noé"), String("a furnished room"))
    m.open_log(mdir, day + 9 * 3600)
    m.heard(String("Va à la table, puis à la porte."), day + 9 * 3600 + 1)
    var w3 = Task(String(""))
    w3.steps.append(Step.goto(String("table")))
    w3.steps.append(Step.goto(String("door")))
    give_task(m, w3^, 0.0, day + 9 * 3600 + 2)
    _ = tick_tasks(m, 0.0, day + 9 * 3600 + 2)
    m.observe_motion(String("goto:table"), String("none"), String("table"), day + 9 * 3600 + 5)
    _ = tick_tasks(m, 3.0, day + 9 * 3600 + 5)
    _ = tick_tasks(m, 3.1, day + 9 * 3600 + 5)
    m.observe_motion(String("goto:door"), String("none"), String("door"), day + 9 * 3600 + 9)
    _ = tick_tasks(m, 7.0, day + 9 * 3600 + 9)
    # this afternoon: a new process remembers both
    var r = WorldState(Body.g1_room("Noé"), String("a furnished room"))
    restore_memory(r, mdir, day + 15 * 3600)
    var today_line = String(
        "Earlier today, before you were restarted: You did: walk to the table, then walk to the door."
        " The last things you were asked or told: \"Va à la table, puis à la porte.\"."
    )
    var yesterday_line = String(
        "Yesterday: You did: walk to the fridge. You tried and failed: walk to the sofa (the sofa was"
        " not reached after 60 seconds). The last things you were asked or told: \"Bonjour.\";"
        " a person told you their name; \"Va au frigo.\"."
    )
    check(fails, r.memory_error == "" and len(r.remembered) == 2 and r.remembered[0] == today_line and r.remembered[1] == yesterday_line,
          "a new process remembers this morning and yesterday")
    if len(r.remembered) != 2 or r.remembered[0] != today_line or r.remembered[1] != yesterday_line:
        for l in r.remembered:
            print("       ", l)
        print("        error:", r.memory_error)
    check(fails, reply_view(r).live.endswith("\n\nWhat you remember from before this session (your own log, by day):\n- " + today_line
          + "\n- " + yesterday_line + "\n- Nothing else: you remember no other day."),
          "and the reply view carries both, the most recent first")
    # the time words of the question select the day
    r.asked = String("Qu'as-tu fait hier ?")
    var fm = focused_memory(r)
    check(fails, len(fm) == 1 and fm[0] == yesterday_line and reply_view(r).live.endswith("(your own log, by day):\n- " + yesterday_line),
          "'hier': yesterday's line alone")
    r.asked = String("What did you do this morning?")
    check(fails, len(focused_memory(r)) == 1 and focused_memory(r)[0] == today_line, "'this morning': today's earlier session alone")
    r.asked = String("Qu'as-tu fait la semaine dernière ?")
    check(fails, len(focused_memory(r)) == 1 and focused_memory(r)[0]
          == "About last week: nothing. Your log has no entry for last week, so you do not know what you did then.",
          "'la semaine dernière': the log has nothing, and the line says so")
    r.asked = String("Et avant-hier ? Et aujourd'hui ?")
    check(fails, len(focused_memory(r)) == 2 and focused_memory(r)[0].startswith("About the day before yesterday: nothing.")
          and focused_memory(r)[1] == today_line, "'avant-hier' is not 'hier'; two times named, two answers")
    r.asked = String("Où est le frigo ?")
    check(fails, len(focused_memory(r)) == 3 and focused_memory(r)[2] == "Nothing else: you remember no other day.", "no time named: everything")
    var refs = time_refs(String("Le cahier est sur la table."))
    check(fails, len(refs) == 0 and len(time_refs(String("Yesterday?"))) == 1 and time_refs(String("Yesterday?"))[0].lo == 1,
          "'cahier' does not contain 'hier' as a word")
    r.asked = String("")
    var digest = string_from_bytes(read_file_bytes(mdir + "/digest/2026-11-12.md"))
    check(fails, digest.startswith("# 2026-11-12\n\n- You did: walk to the fridge.\n- You tried and failed: walk to the sofa")
          and not exists(mdir + "/digest/2026-11-13.md"), "yesterday's digest is written for a person; today's is not, its log is still growing")
    check(fails, load_places(r, mdir) == 2 and r.places[1].name == "sofa" and r.places[1].x == -4.5
          and r.word_for(String("sofa"), String("fr")) == "le canapé" and r.word_for(String("fridge"), String("fr")) == "the fridge",
          "the places and their names come back from places.json")
    # a torn last line is skipped; a damaged line in the middle leaves its day out and says so
    with open(mdir + "/log/2026-11-13.jsonl", "a") as f:
        f.write("{\"t\":17945")
    var torn = WorldState(Body.g1_room("Noé"), String("a furnished room"))
    restore_memory(torn, mdir, day + 15 * 3600)
    check(fails, torn.memory_error == "" and len(torn.remembered) == 2 and torn.remembered[0] == today_line, "a torn last line is skipped")
    with open(mdir + "/log/2026-11-13.jsonl", "a") as f:
        f.write("\n" + event_json(BrainEvent(day + 16 * 3600, String(EV_HEARD), String("encore"), String("voice"))) + "\n")
    var broken = WorldState(Body.g1_room("Noé"), String("a furnished room"))
    restore_memory(broken, mdir, day + 17 * 3600)
    check(fails, broken.memory_error.find("line") >= 0 and len(broken.remembered) == 1 and broken.remembered[0] == yesterday_line,
          "a damaged line in the middle: that day is left out and reported: " + broken.memory_error)
    # only the last files are read, however old, and an old day is dated
    var old = WorldState(Body.g1_room("Noé"), String("a furnished room"))
    old.open_log(mdir, day - 9 * 86400)
    old.heard(String("il y a très longtemps"), day - 9 * 86400 + 1)
    var old2 = WorldState(Body.g1_room("Noé"), String("a furnished room"))
    old2.open_log(mdir, day - 5 * 86400)
    old2.heard(String("il y a longtemps"), day - 5 * 86400 + 1)
    var days_seen = log_days(mdir)
    check(fails, len(days_seen) == 4 and days_seen[0] == "2026-11-04" and days_seen[1] == "2026-11-08" and days_seen[3] == "2026-11-13",
          "the log's days, oldest first")
    var later = WorldState(Body.g1_room("Noé"), String("a furnished room"))
    restore_memory(later, mdir, day + 2 * 86400 + 60)
    check(fails, len(later.remembered) == 2 and later.remembered[0].startswith("3 days ago (2026-11-12): You did: walk to the fridge.")
          and later.remembered[1] == "7 days ago (2026-11-08): The last things you were asked or told: \"il y a longtemps\".",
          "two days later: the last three files are read (one is damaged), the fourth is left out, each day is dated")
    if len(later.remembered) != 2:
        for l in later.remembered:
            print("       ", l)

    print("9. people")
    check(fails, name_given("Je m'appelle Denis.") == "Denis" and name_given("je m'appelle denis") == "Denis"
          and name_given("Bonjour, moi c'est Claire !") == "Claire" and name_given("My name is Jean Dupont.") == "Jean Dupont"
          and name_given("Je m'appelle Denis et toi ?") == "Denis" and name_given("Call me Sam, please") == "Sam"
          and name_given("Je m\u2019appelle Zoé.") == "Zoé", "a name given in the first person")
    check(fails, name_given("Il s'appelle Paul.") == "" and name_given("Je suis fatigué.") == "" and name_given("Comment tu t'appelles ?") == ""
          and name_given("my name is not important") == "" and name_given("Je m'appelle") == "" and name_given("Où est le frigo ?") == "",
          "someone else's name, a state, a question, a non-name: nothing")
    var pdir = getenv("TMPDIR", "/tmp") + "/noeira_brain_people_" + String(perf_counter_ns())
    var pp = People(pdir)
    var pw = WorldState(Body.g1_room("Noé"), String("a furnished room"))
    pw.people_on = True
    var va: List[Float32] = [1.0, 0.0, 0.0, 0.0]
    var va2: List[Float32] = [0.9, 0.0, 0.43589, 0.0]       # cosine 0.9 with va
    var va3: List[Float32] = [0.40, 0.0, 0.0, 0.9165]       # cosine 0.40 with va: resembles
    var va4: List[Float32] = [0.34, 0.0, 0.0, 0.9404]       # cosine 0.34: what a stranger can reach
    var vb: List[Float32] = [0.0, 2.0, 0.0, 0.0]            # another voice (not of length 1 on purpose)
    pp.heard(pw, String("Bonjour."), 1000)
    pp.voice(pw, va, 1001)
    check(fails, pw.speaker == "" and len(pp.list) == 0
          and speaker_sentence(pw) == "You do not know who is talking to you: nobody has told you their name yet.", "an unknown voice, nobody known")
    pp.heard(pw, String("Comment je m'appelle ?"), 1005)
    pw.asked = String("Comment je m'appelle ?")
    pw.language = String("fr")
    check(fails, pw.speaker_pending and speaker_sentence(pw) == "The voice of the person talking to you is still being checked: you do not know yet"
          " who it is. The person is asking who they are. Answer with exactly this sentence and nothing else:"
          " \"Un instant, je n'ai pas encore reconnu ta voix.\"", "a voice not compared yet: nobody's name is given out")
    pp.voice(pw, va, 1006)
    check(fails, not pw.speaker_pending and identity_answer(pw) == "Je ne sais pas : tu ne m'as pas encore dit ton nom.", "asked by a voice nobody gave a name to")
    pw.asked = String("")
    pw.language = String("")
    pp.heard(pw, String("Je m'appelle Denis."), 1010)
    check(fails, speaker_sentence(pw) == "The person talking to you has just told you their name: Denis. That is the person's name, not yours.",
          "the name is in the view as soon as it is heard, before the voice is known")
    pp.voice(pw, va, 1012)
    check(fails, pp.take_met(String("fr")) == "Enchanté, Denis." and pp.take_met(String("fr")) == "", "the introduction is acknowledged, once, with the name said back")
    check(fails, pw.speaker == "Denis" and pw.speaker_sure and len(pp.list) == 1 and len(pp.list[0].voices) == 1 and pw.speaker_said_name == ""
          and speaker_sentence(pw) == "You are talking with a person whose name is Denis: you recognise their voice.", "the voice is stored under the name")
    check(fails, reply_view(pw).live.find("- You know of no other place or object in this room.\n- You are talking with a person whose name is Denis: you recognise their voice.") >= 0
          and reply_view(pw).live.startswith("Facts, true right now:\n- You are standing"),
          "who is talking is the last of the facts (the line that changes most: a prefix cache keeps nothing after it)")
    pw.asked = String("Do you recognise me?")
    check(fails, asks_identity(pw.asked) and identity_answer(pw) == "Your name is Denis." and speaker_sentence(pw).endswith(
          "The person is asking who they are. Answer with exactly this sentence and nothing else: \"Your name is Denis.\""), "asked by Denis: the exact answer is in the view")
    pw.language = String("fr")
    check(fails, identity_answer(pw) == "Tu t'appelles Denis." and asks_identity("Comment je m\u2019appelle ?") and asks_identity("Qui suis-je ?")
          and not asks_identity("Comment tu t'appelles ?") and not asks_identity("Où est le frigo ?"), "in French; and what is not a question about the speaker")
    pw.asked = String("")
    pw.language = String("")
    pp.heard(pw, String("Où est le frigo ?"), 1020)
    pp.voice(pw, va2, 1021)
    check(fails, pw.speaker == "Denis" and pw.speaker_sure and len(pp.list[0].voices) == 2 and abs(pw.speaker_score - 0.9) < 1e-3,
          "the same voice later: recognised (0.9), and kept as another sample")
    pp.voice(pw, va4, 1025)
    check(fails, pw.speaker == "" and len(pp.list[0].voices) == 2, "0.34, seconds after Denis spoke: NOT Denis (a recognition does not carry over)")
    pp.voice(pw, va3, 1030)
    pw.language = String("fr")
    check(fails, identity_answer(pw) == "Je crois que tu es Denis, mais je n'en suis pas sûr.", "a voice that resembles: said as a belief")
    pw.language = String("")
    check(fails, pw.speaker == "Denis" and not pw.speaker_sure
          and speaker_sentence(pw) == "The voice of the person talking to you resembles the voice of Denis, a person you know, but you are not sure it is Denis.",
          "0.40: it resembles Denis, not sure")
    pp.voice(pw, vb, 2000)
    check(fails, pw.speaker == "" and speaker_sentence(pw) == "You do not recognise the voice of the person talking to you: you do not know who it is."
          and reply_view(pw).text().find("Denis") < 0, "another voice: not recognised, and the name it knows is nowhere in the prompt")
    check(fails, identity_answer(pw) == "I do not recognise your voice.", "a stranger asking gets no name")
    pp.heard(pw, String("Moi c'est Claire."), 2010)
    pp.voice(pw, vb, 2011)
    check(fails, pw.speaker == "Claire" and len(pp.list) == 2 and pp.list[1].id == "p2", "a second person")
    pp.heard(pw, String("Non, je m'appelle Dennis."), 2020)
    pp.voice(pw, va, 2021)
    check(fails, pw.speaker == "Dennis" and len(pp.list) == 2 and pp.list[0].name == "Dennis" and pp.list[0].id == "p1",
          "a known voice gives another name: the same person, renamed")
    var met = String("")
    for ev in pw.events:
        if ev.kind == "met":
            met += ev.text + " ; "
    check(fails, met == "Denis ; Claire ; Dennis | the voice known as Denis gave this name ; ", "the record of who was met: " + met)
    # another process
    var p2 = People.load(pdir)
    var pw2 = WorldState(Body.g1_room("Noé"), String("a furnished room"))
    pw2.people_on = True
    check(fails, p2.error == "" and len(p2.list) == 2 and p2.list[0].name == "Dennis" and len(p2.list[0].voices) == 3
          and p2.list[1].name == "Claire" and abs(Float64(p2.list[1].voices[0][1]) - 1.0) < 1e-6 and p2.next_id == 3,
          "read back by another process: names, voices (stored at length 1), the next id")
    p2.heard(pw2, String("Comment je m'appelle ?"), 5000)
    p2.voice(pw2, va2, 5001)
    check(fails, pw2.speaker == "Dennis" and pw2.speaker_sure, "and the voice is recognised there")
    check(fails, p2.forget(String("Claire")) and len(p2.list) == 1 and not exists(pdir + "/people/p2.voice") and not p2.forget(String("Claire")),
          "forget: the person and the voice file are gone")
    p2.voice(pw2, vb, 5010)
    check(fails, pw2.speaker == "" and len(People.load(pdir).list) == 1, "and that voice is unknown again, here and on disk")
    with open(pdir + "/people/p1.voice", "w") as f:
        f.write("short")
    check(fails, People.load(pdir).error.find("p1.voice") >= 0, "a damaged voice file is reported: " + People.load(pdir).error)

    print("10. the answers code gives")
    check(fails, asks_task_done("As-tu fini ce que je t'ai demandé ?") and asks_task_done("as tu terminé") and asks_task_done("Est-ce que tu as fini ?")
          and asks_task_done("Tu as fini ?") and asks_task_done("Have you finished?") and asks_task_done("Are you done"),
          "the question, inverted or with its mark, hyphen or not")
    check(fails, not asks_task_done("Quand tu as fini, va à la porte.") and not asks_task_done("Quand auras-tu fini ?")
          and not asks_task_done("Où est le frigo ?") and not asks_task_done("Va à la table.") and not asks_task_done("You are done here."),
          "a request, another question and a statement are not it")
    check(fails, asks_task_done("Tu as fini.") and asks_task_done("Noé, tu as fini.") and asks_task_done("You finished")
          and not asks_task_done("C'est fini.") and not asks_task_done("Tu as fini la danse, va à la porte.")
          and not asks_task_done("Si tu as fini, tu as fini de ranger."),
          "the bare sentence, its question mark lost by the recogniser (the carry replay)")
    var q = WorldState(Body.g1_room("Noé"), String("a furnished room"))
    q.language = String("fr")
    q.call_place(String("table"), String("fr"), String("la table"))
    q.call_place(String("door"), String("fr"), String("la porte"))
    q.asked = String("As-tu fini ce que je t'ai demandé ?")
    check(fails, task_done_answer(q) == "" and computed_reply(q) == "", "no task ever: code has nothing to say, the model replies")
    var qone = Task(String("va à la table"))
    qone.steps.append(Step.goto(String("table")))
    give_task(q, qone^, 0.0, 100)
    _ = tick_tasks(q, 0.0, 100)
    q.observe_motion(String("goto:table"), String("table"), String("none"), 101)
    check(fails, computed_reply(q) == "Pas encore, je suis en route vers la table.", "one walk in hand: " + computed_reply(q))
    q.observe_motion(String("goto:table"), String("none"), String("table"), 110)
    _ = tick_tasks(q, 10.0, 110)
    check(fails, not q.has_task and computed_reply(q) == "Oui, j'ai fini ce que tu m'as demandé.", "done: " + computed_reply(q))
    q.asked = String("Où est le frigo ?")
    check(fails, computed_reply(q) == "", "another question is the model's")
    q.asked = String("Have you finished?")
    q.language = String("en")
    check(fails, computed_reply(q) == "Yes, I have finished what you asked.", "in English")
    q.language = String("fr")
    # station keeping walks back to the place: still the same finished task
    q.observe_motion(String("goto:table"), String("table"), String("none"), 115)
    q.observe_motion(String("stand"), String("none"), String("table"), 118)
    check(fails, task_done_answer(q) == "Oui, j'ai fini ce que tu m'as demandé.", "a walk back to the place changes nothing")
    # a spoken gesture is not a task: what the person means has no record
    q.observe_motion(String("squat"), String("none"), String("table"), 120)
    check(fails, task_done_answer(q) == "", "a gesture asked since: code stays silent")
    q.observe_motion(String("stand"), String("none"), String("table"), 125)
    check(fails, task_done_answer(q) == "", "and still once the gesture is over")
    var qtwo = Task(String("va à la table puis à la porte"))
    qtwo.steps.append(Step.goto(String("table")))
    qtwo.steps.append(Step.goto(String("door")))
    qtwo.steps[1].timeout_s = 5.0
    give_task(q, qtwo^, 30.0, 130)
    _ = tick_tasks(q, 30.0, 130)
    check(fails, task_done_answer(q) == "Pas encore : j'en suis à l'étape 1 sur 2.", "several steps, the first in hand")
    _ = tick_tasks(q, 31.0, 131)     # at the table already: step 1 is done
    _ = tick_tasks(q, 31.1, 131)
    q.observe_motion(String("goto:door"), String("door"), String("none"), 132)
    check(fails, task_done_answer(q) == "Pas encore : j'en suis à l'étape 2 sur 2.", "then the second")
    var late = tick_tasks(q, 40.0, 140)
    q.observe_motion(String("stand"), String("none"), String("none"), 141)
    check(fails, late.kind == EX_NOTICE and q.last_task.status == TASK_FAILED
          and task_done_answer(q) == "Non, je n'ai pas réussi à atteindre la porte.", "a place not reached: " + task_done_answer(q))
    var qagain = Task(String("va à la table"))
    qagain.steps.append(Step.goto(String("table")))
    give_task(q, qagain^, 50.0, 150)
    _ = tick_tasks(q, 50.0, 150)
    cancel_task(q, String("another command replaced it"), 51.0, 151)
    check(fails, q.last_task.status == TASK_CANCELLED and task_done_answer(q) == "", "a cancelled task: code stays silent")
    # who the speaker is comes first, and only with a people store
    q.asked = String("Qui suis-je ?")
    check(fails, computed_reply(q) == "", "no people store: an identity question is the model's")
    q.people_on = True
    check(fails, computed_reply(q) == "Je ne sais pas : tu ne m'as pas encore dit ton nom.", "with one: the identity answer")

    print("11. what the body can do")
    var c = WorldState(Body.g1_room("Noé"), String("a furnished room"))
    c.language = String("fr")
    c.places.append(PlaceXY(String("fridge"), 2.8, 1.7))
    c.places.append(PlaceXY(String("door"), 0.9, 2.7))
    c.places.append(PlaceXY(String("sofa"), -1.6, -2.0))
    c.call_place(String("fridge"), String("fr"), String("le frigo"))
    c.call_place(String("door"), String("fr"), String("la porte"))
    c.call_place(String("sofa"), String("fr"), String("le canapé"))
    var view0 = reply_stable(c)
    check(fails, view0.find("You can: walk to a place of the room; turn; squat; raise your arms; talk and answer questions.") >= 0
          and view0.find("you cannot: pick things up or carry them; open the door; sit down.") >= 0,
          "g1_room's lines are BB-1's, byte for byte")
    check(fails, asks_capabilities("Tu peux faire quoi?") and asks_capabilities("Qu'est-ce que tu sais faire ?")
          and asks_capabilities("Noé, que sais-tu faire ?") and asks_capabilities("What can you do?"),
          "« what can you do », four ways")
    check(fails, not asks_capabilities("Tu peux faire un squat?") and not asks_capabilities("Qu'est-ce que tu fais ?"),
          "a request and « what are you doing » are not it")
    c.asked = String("Tu peux faire quoi?")
    var ct0 = computed_turn(c)
    check(fails, ct0.text == "Je peux aller au frigo, à la porte ou au canapé, tourner, faire un squat et lever les bras."
          and not ct0.decides and ct0.command == "", "the list, from the skills and the places: " + ct0.text)
    c.asked = String("Tu peux danser ?")
    check(fails, computed_turn(c).text == "", "no dance on this body: the model answers")
    c.body.add_motion(String("dance"))
    c.body.add_motion(String("backflip"))
    c.body.add_motion(String("jump"))
    check(fails, len(c.body.can) == len(c.body.skills) and c.body.can[4] == "dance" and c.body.can[7] == "talk and answer questions",
          "a motion goes among the movements, before talking, in both lists: " + join(c.body.can, "; "))
    check(fails, reply_stable(c).find("raise your arms; dance; do a backflip; jump; talk and answer questions.") >= 0,
          "and the reply view says it can dance")
    c.asked = String("Tu peux faire quoi ?")
    check(fails, computed_turn(c).text == "Je peux aller au frigo, à la porte ou au canapé, tourner, faire un squat, lever les bras"
          + " et des figures comme danser ou faire un salto.", "the motions named by two examples: " + computed_turn(c).text)
    c.asked = String("Tu peux danser ?")
    var ct1 = computed_turn(c)
    check(fails, ct1.text == "Oui, regarde !" and not ct1.decides and ct1.command == "dance", "« Tu peux danser ? »: it dances")
    var cases: List[String] = ["Danse !", "Noé, danse.", "Allez, danse avec moi", "Fais un salto.", "Peux-tu faire un saut périlleux ?"]
    var wants: List[String] = ["dance", "dance", "dance", "backflip", "backflip"]
    for i in range(len(cases)):
        c.asked = cases[i]
        var ti = computed_turn(c)
        check(fails, ti.command == wants[i] and not ti.decides, "« " + cases[i] + " » -> " + ti.command)
    var leave: List[String] = [
        "J'aime danser.", "Tu aimes danser ?", "Danse devant la porte.", "Saute puis va au frigo.",
        "Tu peux tourner à gauche ?", "Tu peux chanter ?", "Va à la porte.", "Je vais sauter.",
    ]
    for l in leave:
        c.asked = l
        var tl = computed_turn(c)
        check(fails, tl.text == "" and tl.decides and tl.command == "", "« " + l + " »: the decision's")
    c.asked = String("Peux-tu faire un squat ?")
    check(fails, computed_turn(c).command == "squat", "a bank row asked « peux-tu »: run too")
    c.asked = String("Noé peux-tu lever les bras?")
    check(fails, computed_turn(c).command == "both_hands_up", "« lever les bras » (live, 11-18: d1 sent it to a reward spec)")
    c.asked = String("Lève le bras droit.")
    check(fails, computed_turn(c).text == "", "one arm: the decision's (it has the row)")
    c.asked = String("Tu peux ouvrir la porte ?")
    var ct2 = computed_turn(c)
    check(fails, ct2.text == "Non, je ne peux pas ouvrir la porte." and not ct2.decides and ct2.command == "",
          "a limit, though it names a place: " + ct2.text)
    c.asked = String("Est-ce que tu peux t'asseoir sur le canapé ?")
    check(fails, computed_turn(c).text == "Non, je ne peux pas m'asseoir.", "« t'asseoir »: " + computed_turn(c).text)
    c.asked = String("Ouvre la porte.")
    check(fails, computed_turn(c).text == "", "told, not asked: the decision's (it refuses or walks)")
    check(fails, asks_can("Tu peux danser ?") and asks_can("Peux tu danser") and asks_can("Can you dance")
          and not asks_can("Tu peux danser.") and not asks_can("Je peux danser ?"), "the question's forms")
    c.language = String("en")
    c.asked = String("What can you do?")
    check(fails, computed_turn(c).text == "I can walk to the fridge, the door or the sofa, turn, squat, raise my arms and moves such as: dance or do a backflip.",
          "in English: " + computed_turn(c).text)
    c.asked = String("Can you dance?")
    check(fails, computed_turn(c).text == "Yes, watch!" and computed_turn(c).command == "dance", "can you dance")
    c.language = String("fr")
    c.asked = String("As-tu fini ce que je t'ai demandé ?")
    check(fails, computed_turn(c).decides, "« As-tu fini ? » still lets the decision act")

    print("12. how long the model's reply may be")
    var caps: List[Int] = [REPLY_SENTENCES, REPLY_SENTENCES_MORE]
    check(fails, caps[0] == 1 and caps[1] == 2, "one sentence, two when more is asked")
    var more: List[String] = [
        "Raconte-moi une blague.", "Noé, tu me racontes une histoire ?", "Explique-moi comment tu marches.",
        "Décris la pièce.", "Dis-m'en plus.", "Parle-moi de toi.", "Tell me a joke.", "Can you explain that?",
    ]
    var n_more = 0
    for x in more:
        if asks_for_more(x):
            n_more += 1
        else:
            print("     not more:", x)
    check(fails, n_more == len(more), "asked for more: " + String(n_more) + "/" + String(len(more)))
    var enough: List[String] = [
        "Où es-tu ?", "Il fait beau aujourd'hui.", "Comment tu t'appelles ?", "Tu sais chanter ?",
        "Où est la télévision ?", "Bonjour Noé.", "Tu as fait une blagueuse remarque", "What is your name?",
    ]
    var n_one = 0
    for x in enough:
        if not asks_for_more(x):
            n_one += 1
        else:
            print("     more:", x)
    check(fails, n_one == len(enough), "one sentence is enough: " + String(n_one) + "/" + String(len(enough)) + " (« blagueuse » is not « blague »)")
    c.asked = String("Raconte-moi une blague.")
    var r_more = reply_sentences(c)
    c.asked = String("Où es-tu ?")
    check(fails, r_more == 2 and reply_sentences(c) == 1, "reply_sentences reads `ws.asked`")

    print("ALL PASS" if fails == 0 else String(fails) + " FAILED")
    if fails != 0:
        exit(1)

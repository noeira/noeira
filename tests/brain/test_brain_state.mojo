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
Exits non-zero on a failure.
"""

from std.os import getenv
from std.sys import exit

from noeira.brain import (
    Task, Step, give_task, cancel_task, refuse_step, tick_tasks, notice_text, task_sentences,
    EX_NONE, EX_START, EX_NOTICE, NOTICE_ALL_DONE, NOTICE_NOT_REACHED, NOTICE_REFUSED,
    TASK_DONE, TASK_FAILED, TASK_CANCELLED,
    WorldState, Body, PlaceXY, BrainEvent, EventLog, load_events, event_json, parse_event,
    reply_view, REPLY_VIEW_SENTENCES, metres,
    EV_BEGAN, EV_GOING, EV_ARRIVED, EV_HEARD, EV_SESSION,
)
from noeira.brain.bb1 import BB1Set, reply_passes, reply_in_language
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

    print("ALL PASS" if fails == 0 else String(fails) + " FAILED")
    if fails != 0:
        exit(1)

"""`noeira.brain`, movable objects: the crate's record in the state, the
executive's `pick:` / `put:` steps, and the record read back from the log.

    pixi run mojo run -I . tests/brain/test_brain_objects.mojo

1. Headings modulo a half turn (and a quarter for a square footprint).
2. The event text: written, read back equal, a place of two words.
3. Sightings many times a second: every one is taken, and only the news is
   written (first, after a gap, a move, a turn; a half-turn is the same box).
4. Told, held, put down, moved by someone, missing.
5. The executive: an unknown object fails at once and the body is given
   nothing; pick then put ends on the object's record, NOT on the robot
   standing at the place; a put at the wrong place and a pick that never
   happens time out and say so.
6. `restore_objects`: the last event of each object, as remembered, into a
   new process's state, with and without the object added first; a damaged
   object event is reported and skipped.
7. The voice side: « Où est la caisse ? » answered from the record (on the
   floor, held, put, told, missing; French and English); an order to carry it
   is a no for a body that does not carry, and a task of `pick:` / `put:`
   for one that does (`Body.add_carry`, its skill line and the limit gone);
   « porte » the door is not the verb; the reply view gains one objects line,
   and a room without objects keeps its view.
Exits non-zero on a failure.
"""

from std.os import getenv
from std.time import perf_counter_ns
from std.sys import exit

from noeira.brain import (
    WorldState, Body, PlaceXY, ObjectXY, object_event_text, parse_object_event, heading_mod, heading_diff,
    OBJ_FLOOR, OBJ_HELD, OBJ_PUT, HOW_SEEN, HOW_TOLD, HOW_REMEMBERED, OBJ_GAP_S,
    EV_SAW, EV_TOLD_WHERE, EV_LOST, EV_PICKED, EV_PUT,
    Task, Step, give_task, tick_tasks, notice_text, jusqua,
    EX_NONE, EX_START, EX_NOTICE, NOTICE_NO_OBJECT, NOTICE_NOT_DONE, NOTICE_ALL_DONE, STOP_COMMAND,
    TASK_DONE, TASK_FAILED, restore_objects, BrainEvent, EventLog,
    object_named, place_named, object_answer, computed_turn, carry_turn, capabilities_answer,
    reply_live, reply_stable, objects_line,
)


def check(mut fails: Int, ok: Bool, what: String):
    if ok:
        print("  ok  :", what)
    else:
        print("  FAIL:", what)
        fails += 1


def count(ref ws: WorldState, kind: String) -> Int:
    var n = 0
    for e in ws.events:
        if e.kind == kind:
            n += 1
    return n


def near(a: Float64, b: Float64, tol: Float64 = 1e-9) -> Bool:
    return abs(a - b) <= tol


def room() -> WorldState:
    var ws = WorldState(Body.g1_room("Noé"), String("a furnished room"))
    ws.places = [PlaceXY("table", 0.0, -2.0), PlaceXY("sofa", -2.0, 1.0), PlaceXY("fridge", 3.0, 0.0)]
    ws.call_place("crate", "fr", "la caisse")
    ws.call_place("table", "fr", "la table")
    ws.call_place("fridge", "fr", "le frigo")
    ws.call_place("sofa", "fr", "le canapé")
    return ws^


def ask(mut ws: WorldState, text: String) -> String:
    """`computed_turn` for `text`: `answer | decides | command`."""
    ws.asked = text.copy()
    var ct = computed_turn(ws)
    return ct.text + " | " + ("d1" if ct.decides else "settled") + " | " + ct.command


def main() raises:
    var fails = 0

    print("1. headings")
    check(fails, near(heading_mod(190.0, 180.0), 10.0), "190 mod 180 = 10")
    check(fails, near(heading_mod(-10.0, 180.0), 170.0), "-10 mod 180 = 170")
    check(fails, near(heading_mod(180.0, 180.0), 0.0), "180 mod 180 = 0")
    check(fails, near(heading_diff(5.0, 175.0, 180.0), 10.0), "5 and 175 are 10 apart for a box")
    check(fails, near(heading_diff(30.0, 120.0, 90.0), 0.0), "30 and 120 are the same square box")
    check(fails, near(heading_diff(30.0, 120.0, 360.0), 90.0), "30 and 120 differ for a thing with a front")
    var b = ObjectXY.box("crate", "crate", 0.13, 0.215, 0.19)
    check(fails, near(b.turn_deg, 180.0) and not b.has_pose and b.how == "", "a box: half turn, no pose")
    check(fails, near(ObjectXY.box("cube", "box", 0.2, 0.2, 0.2).turn_deg, 90.0), "a square footprint: quarter turn")
    b.set_pose(1.0, 2.0, 190.0)
    check(fails, near(b.yaw_deg, 10.0), "a heading is kept in [0, 180)")
    check(fails, near(b.heading_near(170.0), 190.0), "heading_near 170 -> 190")
    check(fails, near(b.heading_near(-20.0), 10.0), "heading_near -20 -> 10")

    print("2. the event text")
    var o = ObjectXY.box("crate", "crate", 0.13, 0.215, 0.19)
    o.set_pose(1.234, 0.456, 35.0)
    o.where = OBJ_PUT
    o.put_at = String("living room")
    o.how = String(HOW_SEEN)
    var text = object_event_text(o)
    print("   ", text)
    check(fails, text == "crate | 1.234 0.456 35.0 | put living room | crate 0.13 0.215 0.19 180.0 | seen", "the text")
    var back = parse_object_event(text)
    check(fails, back.name == "crate" and back.label == "crate" and back.has_pose, "read back: name, label")
    check(fails, near(back.x, 1.234) and near(back.y, 0.456) and near(back.yaw_deg, 35.0), "read back: the pose")
    check(fails, near(back.hx, 0.13) and near(back.hy, 0.215) and near(back.hz, 0.19) and near(back.turn_deg, 180.0), "read back: the size")
    check(fails, back.where == OBJ_PUT and back.put_at == "living room" and back.how == "seen", "read back: put at a place of two words")
    var bad = False
    try:
        _ = parse_object_event(String("crate | 1 2"))
    except:
        bad = True
    check(fails, bad, "a text of another shape raises")

    print("3. sightings many times a second")
    var ws = room()
    var t = 10.0
    check(fails, not ws.saw_object("crate", 1.0, 1.0, 0.0, 500.0, "eye", t, 1000), "an object not added is refused")
    check(fails, len(ws.events) == 0, "  and nothing is written")
    ws.add_object(ObjectXY.box("crate", "crate", 0.13, 0.215, 0.19))
    var ci = ws.object_index("crate")
    check(fails, ci == 0 and ws.object_index("ball") == -1, "object_index")
    check(fails, ws.place_index("sofa") == 1 and ws.place_index("bed") == -1, "place_index")
    for k in range(50):
        var j = 0.001 * Float64(k % 3)
        _ = ws.saw_object("crate", 1.0 + j, 1.0 - j, 30.0 + j, 500.0, "eye", t + 0.02 * Float64(k), 1000)
    t += 0.02 * 49.0
    check(fails, count(ws, EV_SAW) == 1, "50 sightings in a second, millimetres apart: one `saw`")
    check(fails, ws.objects[ci].times_seen == 50 and ws.objects[ci].how == HOW_SEEN, "  every one taken")
    check(fails, ws.objects[ci].fresh(t) and not ws.objects[ci].fresh(t + 1.0), "fresh now, not a second later")
    _ = ws.saw_object("crate", 1.0, 1.0, 30.0 + 180.0, 500.0, "eye", t + 0.02, 1000)
    t += 0.02
    check(fails, count(ws, EV_SAW) == 1 and near(ws.objects[ci].yaw_deg, 30.0), "a half turn is the same box: nothing written")
    t += OBJ_GAP_S + 0.5
    _ = ws.saw_object("crate", 1.0, 1.0, 30.0, 500.0, "eye", t, 1003)
    check(fails, count(ws, EV_SAW) == 2, "seen again after a gap: written")
    for k in range(25):
        t += 0.02
        _ = ws.saw_object("crate", 1.0 + 0.01 * Float64(k + 1), 1.0, 30.0, 500.0, "eye", t, 1003)
    check(fails, count(ws, EV_SAW) == 4, "pushed 25 cm a centimetre a frame: written at 10 and 20 cm")
    t += 0.02
    _ = ws.saw_object("crate", 1.25, 1.0, 45.0, 500.0, "eye", t, 1003)
    check(fails, count(ws, EV_SAW) == 5, "turned 15 degrees: written")
    check(fails, near(ws.objects[ci].x, 1.25) and near(ws.objects[ci].yaw_deg, 45.0), "the record is the last sighting")

    print("4. told, held, put down, moved, missing")
    check(fails, ws.told_object("crate", 2.0, 0.5, 100.0, "trials", t, 1004), "told")
    check(fails, ws.objects[ci].how == HOW_TOLD and count(ws, EV_TOLD_WHERE) == 1 and not ws.objects[ci].fresh(t), "  how = told, written, not fresh")
    t += 0.02
    _ = ws.saw_object("crate", 2.0, 0.5, 100.0, 500.0, "eye", t, 1004)
    check(fails, count(ws, EV_SAW) == 6 and ws.objects[ci].how == HOW_SEEN, "the first sighting of a told object is written")
    check(fails, ws.picked_up("crate", t, 1005) and ws.objects[ci].where == OBJ_HELD and count(ws, EV_PICKED) == 1, "picked up")
    for k in range(100):
        t += 0.02
        _ = ws.saw_object("crate", 2.0 - 0.02 * Float64(k), 0.5, 100.0 + Float64(k), 500.0, "eye", t, 1006)
    check(fails, count(ws, EV_SAW) == 6, "carried 2 m and turned: nothing written while held")
    check(fails, near(ws.objects[ci].x, 2.0 - 0.02 * 99.0) and ws.objects[ci].where == OBJ_HELD, "  the pose follows the hands")
    check(fails, ws.put_down("crate", "table", 0.0, -1.6, 90.0, t, 1007), "put down")
    check(fails, ws.objects[ci].where == OBJ_PUT and ws.objects[ci].put_at == "table" and count(ws, EV_PUT) == 1, "  at the table")
    t += 0.02
    _ = ws.saw_object("crate", 0.005, -1.6, 90.0, 500.0, "eye", t, 1007)
    check(fails, count(ws, EV_SAW) == 6 and ws.objects[ci].where == OBJ_PUT, "seen where it was put: still put, nothing written")
    t += 0.02
    _ = ws.saw_object("crate", 0.3, -1.6, 90.0, 500.0, "eye", t, 1008)
    check(fails, count(ws, EV_SAW) == 7 and ws.objects[ci].where == OBJ_FLOOR and ws.objects[ci].put_at == "", "moved 30 cm by someone: on the floor, written")
    check(fails, ws.object_missing("crate", t, 1009) and ws.objects[ci].missing and count(ws, EV_LOST) == 1, "missing: written")
    _ = ws.object_missing("crate", t + 1.0, 1010)
    check(fails, count(ws, EV_LOST) == 1, "missing twice: written once")
    t += 0.02
    _ = ws.saw_object("crate", 0.3, -1.6, 90.0, 500.0, "eye", t, 1010)
    check(fails, not ws.objects[ci].missing and count(ws, EV_SAW) == 8, "seen again: not missing, written")

    print("5. the executive")
    var ex = room()
    ex.language = String("fr")
    ex.add_object(ObjectXY.box("crate", "crate", 0.13, 0.215, 0.19))
    var c = 0.0
    var task = Task(String("prends la caisse"))
    task.steps.append(Step.pick("crate"))
    give_task(ex, task^, c, 2000)
    var a = tick_tasks(ex, c, 2000)
    check(fails, a.kind == EX_NOTICE and a.notice == NOTICE_NO_OBJECT and a.command == "", "a crate with no pose: no command, a notice")
    check(fails, not ex.has_task and ex.last_task.status == TASK_FAILED, "  the task failed")
    check(fails, notice_text(ex, a) == "Je ne sais pas où est la caisse.", "  « " + notice_text(ex, a) + " »")

    _ = ex.told_object("crate", 2.0, 0.5, 0.0, "trials", c, 2001)
    var carry = Task(String("porte la caisse à la table"))
    carry.steps.append(Step.pick("crate"))
    carry.steps.append(Step.put("crate", "table"))
    give_task(ex, carry^, c, 2001)
    a = tick_tasks(ex, c, 2001)
    check(fails, a.kind == EX_START and a.command == "pick:crate" and a.obj == "crate", "pick:crate given to the body")
    c += 5.0
    a = tick_tasks(ex, c, 2006)
    check(fails, a.kind == EX_NONE and ex.task.at == 0, "  not held yet: still picking")
    _ = ex.picked_up("crate", c, 2006)
    a = tick_tasks(ex, c, 2006)
    check(fails, a.kind == EX_NONE and ex.task.at == 1, "  held: the pick is done")
    a = tick_tasks(ex, c, 2006)
    check(fails, a.kind == EX_START and a.command == "put:crate:table" and a.place == "table", "put:crate:table given")
    c += 8.0
    ex.observe_motion(String(""), String(""), String("table"), 2014)
    a = tick_tasks(ex, c, 2014)
    check(fails, a.kind == EX_NONE and ex.has_task, "the robot AT the table with the crate in hand: not done")
    _ = ex.put_down("crate", "table", 0.0, -1.6, 90.0, c, 2015)
    a = tick_tasks(ex, c, 2015)
    check(fails, a.kind == EX_NOTICE and a.notice == NOTICE_ALL_DONE and ex.last_task.status == TASK_DONE, "put at the table: done")

    var wrong = Task(String("porte la caisse à la table"))
    wrong.steps.append(Step.put("crate", "table"))
    give_task(ex, wrong^, c, 2020)
    _ = tick_tasks(ex, c, 2020)
    _ = ex.picked_up("crate", c, 2021)
    _ = ex.put_down("crate", "sofa", -2.0, 0.6, 0.0, c + 1.0, 2022)
    a = tick_tasks(ex, c + 1.0, 2022)
    check(fails, a.kind == EX_NONE and ex.has_task, "put at the sofa: not the table, not done")
    a = tick_tasks(ex, c + 61.0, 2082)
    check(fails, a.kind == EX_NOTICE and a.notice == NOTICE_NOT_DONE and a.command == STOP_COMMAND, "  timed out: stop, a notice")
    check(fails, notice_text(ex, a) == "Je n'ai pas réussi à porter la caisse jusqu'à la table.", "  « " + notice_text(ex, a) + " »")
    ex.language = String("en")
    check(fails, notice_text(ex, a) == "I could not carry the crate to the table.", "  " + notice_text(ex, a))
    ex.language = String("fr")

    var never = Task(String("prends la caisse"))
    never.steps.append(Step.pick("crate"))
    give_task(ex, never^, 100.0, 2100)
    _ = tick_tasks(ex, 100.0, 2100)
    a = tick_tasks(ex, 161.0, 2161)
    check(fails, a.kind == EX_NOTICE and a.notice == NOTICE_NOT_DONE, "a pick that never happens times out")
    check(fails, notice_text(ex, a) == "Je n'ai pas réussi à prendre la caisse.", "  « " + notice_text(ex, a) + " »")
    _ = ex.object_missing("crate", 162.0, 2162)
    var lost = Task(String("prends la caisse"))
    lost.steps.append(Step.pick("crate"))
    give_task(ex, lost^, 162.0, 2162)
    a = tick_tasks(ex, 162.0, 2162)
    check(fails, a.kind == EX_NOTICE and a.notice == NOTICE_NO_OBJECT, "a missing crate: not where it was, a notice")
    check(fails, jusqua("le frigo") == "jusqu'au frigo" and jusqua("les chaises") == "jusqu'aux chaises"
          and jusqua("la porte") == "jusqu'à la porte", "jusqu'au / jusqu'aux / jusqu'à")

    print("6. restore_objects")
    var dir = getenv("TMPDIR", "/tmp") + "/noeira_brain_objects_" + String(perf_counter_ns())
    var now = 1794000000
    var w1 = room()
    w1.open_log(dir, now)
    w1.add_object(ObjectXY.box("crate", "crate", 0.13, 0.215, 0.19))
    w1.add_object(ObjectXY.box("ball", "ball", 0.1, 0.1, 0.1))
    _ = w1.told_object("crate", 2.0, 0.5, 0.0, "trials", 0.0, now + 1)
    _ = w1.saw_object("crate", 2.01, 0.5, 2.0, 500.0, "eye", 0.1, now + 2)
    _ = w1.picked_up("crate", 1.0, now + 3)
    _ = w1.put_down("crate", "table", 0.0, -1.6, 91.5, 9.0, now + 10)
    _ = w1.told_object("ball", -1.0, -1.0, 0.0, "trials", 9.0, now + 10)
    _ = w1.object_missing("ball", 10.0, now + 11)
    check(fails, w1.log_error == "", "the log was written")
    # a damaged object event: reported, skipped, the rest read
    var log = EventLog(dir)
    log.append(BrainEvent(now + 12, String(EV_SAW), String("cup | nonsense"), String("test")))

    var w2 = room()
    w2.add_object(ObjectXY.box("crate", "crate", 0.13, 0.215, 0.19))
    var n = restore_objects(w2, dir, now + 100)
    check(fails, n == 2 and len(w2.objects) == 2, "two objects restored, the crate not duplicated")
    var k2 = w2.object_index("crate")
    check(fails, w2.objects[k2].how == HOW_REMEMBERED and w2.objects[k2].where == OBJ_PUT and w2.objects[k2].put_at == "table", "the crate: remembered, put at the table")
    check(fails, near(w2.objects[k2].x, 0.0) and near(w2.objects[k2].y, -1.6) and near(w2.objects[k2].yaw_deg, 91.5), "  where it was put, not where it was first told")
    check(fails, not w2.objects[k2].fresh(0.0) and w2.objects[k2].now_seen == now + 10, "  not fresh, dated by its event")
    var bi = w2.object_index("ball")
    check(fails, bi >= 0 and w2.objects[bi].missing and near(w2.objects[bi].hx, 0.1), "the ball: added from the log with its size, missing")
    check(fails, w2.memory_error != "", "the damaged event is reported: " + w2.memory_error)
    var pick = Task(String("prends la caisse"))
    pick.steps.append(Step.pick("crate"))
    give_task(w2, pick^, 0.0, now + 101)
    a = tick_tasks(w2, 0.0, now + 101)
    check(fails, a.kind == EX_START, "a remembered crate can be picked up")
    var w3 = room()
    check(fails, restore_objects(w3, dir + "_none", now) == 0 and len(w3.objects) == 0, "no log: nothing")

    print("7. the voice side")
    var v = room()
    v.language = String("fr")
    var bare = reply_live(v)
    v.add_object(ObjectXY.box("crate", "crate", 0.13, 0.215, 0.19))
    v.observe_pose(0.0, 0.0, 0.0)
    check(fails, object_named(v, "Où est la caisse ?") == 0 and object_named(v, "Where is the crate?") == 0
          and object_named(v, "Où est la table ?") == -1, "object_named: la caisse, the crate, not the table")
    check(fails, place_named(v, "Porte-la au frigo.") == "fridge" and place_named(v, "à la table") == "table"
          and place_named(v, "Prends la caisse.") == "", "place_named: au frigo, la table, none")
    var r = ask(v, "Où est la caisse ?")
    check(fails, r == "Je ne sais pas où est la caisse. | settled | ", "no pose: " + r)
    _ = v.saw_object("crate", 2.0, 0.0, 0.0, 500.0, "sim", 1.0, 3000)
    r = ask(v, "Où est la caisse ?")
    check(fails, r == "La caisse est devant moi, à 2 mètres, près du frigo. | settled | ", "seen: " + r)
    _ = v.saw_object("crate", 0.0, 1.5, 0.0, 500.0, "sim", 1.1, 3000)
    r = ask(v, "Elle est où, la caisse ? Où est la caisse ?")
    check(fails, r == "La caisse est à ma gauche, à 1,5 mètre, près du canapé. | settled | ", "left: " + r)
    _ = v.told_object("crate", 2.0, 0.0, 0.0, "trials", 1.2, 3001)
    r = ask(v, "Où est la caisse ?")
    check(fails, r == "D'après ce qu'on m'a dit, la caisse est devant moi, à 2 mètres, près du frigo. | settled | ", "told: " + r)
    v.language = String("en")
    r = ask(v, "Where is the crate?")
    check(fails, r == "I was told the crate is ahead of me, 2 m away, next to the fridge. | settled | ", "told, English: " + r)
    v.language = String("fr")
    _ = v.picked_up("crate", 1.3, 3002)
    r = ask(v, "Où est la caisse ?")
    check(fails, r == "La caisse est dans mes mains. | settled | ", "held: " + r)
    _ = v.put_down("crate", "table", 0.0, -1.6, 0.0, 1.4, 3003)
    _ = v.saw_object("crate", 0.0, -1.6, 0.0, 500.0, "sim", 1.5, 3003)
    r = ask(v, "Où est la caisse ?")
    check(fails, r == "La caisse est près de la table. | settled | ", "put: " + r)
    _ = v.object_missing("crate", 1.6, 3004)
    r = ask(v, "Où est la caisse ?")
    check(fails, r == "Je ne sais pas où est la caisse. | settled | ", "missing: " + r)
    _ = v.saw_object("crate", 2.0, 0.0, 0.0, 500.0, "sim", 1.7, 3005)
    r = ask(v, "Où est la table ?")
    check(fails, r == " | d1 | ", "a place, not an object: left to the models: " + r)

    r = ask(v, "Porte la caisse à la table.")
    check(fails, r == "Non, je ne peux pas porter la caisse. | settled | ", "a body that does not carry: " + r)
    r = ask(v, "Noé, prends la caisse.")
    check(fails, r == "Non, je ne peux pas prendre la caisse. | settled | ", "  take: " + r)
    r = ask(v, "Regarde la caisse près de la porte.")
    check(fails, r == " | d1 | ", "« la porte » is not the verb: " + r)
    var lines_before = reply_stable(v)
    v.body.add_carry("crate", "the crate", "la caisse")
    check(fails, v.body.can_carry("crate") and not v.body.can_carry("ball"), "add_carry: can_carry")
    var stable = reply_stable(v)
    check(fails, stable.find("pick up the crate and carry it to a place; talk and answer questions") >= 0, "  the can line")
    check(fails, stable.find("pick things up") < 0 and lines_before.find("pick things up") >= 0
          and stable.find("you cannot: open the door; sit down.") >= 0, "  the limit gone, the others kept")
    var caps = capabilities_answer(v)
    check(fails, caps.find("porter la caisse jusqu'à un endroit de la pièce") >= 0, "  « " + caps + " »")
    r = ask(v, "Porte la caisse à la table.")
    check(fails, r == "D'accord ! | settled | task:put:crate:table", "carry: " + r)
    r = ask(v, "Prends la caisse.")
    check(fails, r == "D'accord ! | settled | task:pick:crate", "take: " + r)
    r = ask(v, "Est-ce que tu peux porter la caisse au frigo ?")
    check(fails, r == "Oui ! | settled | task:put:crate:fridge", "asked: " + r)
    r = ask(v, "Pose la caisse près du canapé.")
    check(fails, r == "D'accord ! | settled | task:put:crate:sofa", "put at a place: " + r)
    r = ask(v, "Pose la caisse.")
    check(fails, r == " | d1 | ", "put at no place: left to the decision: " + r)
    v.language = String("en")
    r = ask(v, "Bring the crate to the fridge.")
    check(fails, r == "Okay! | settled | task:put:crate:fridge", "English: " + r)
    v.language = String("fr")

    var plain = room()
    plain.language = String("fr")
    check(fails, reply_live(plain) == bare and bare.find("Objects") < 0, "a room without objects: the view unchanged")
    check(fails, bare.find("\n- \n") < 0 and bare.find("you were at") < 0 and bare.find("anywhere else yet.\n- You know of no other") >= 0,
          "  no empty line: the closing line follows the last fact")
    var line = objects_line(v)
    check(fails, line == "Objects you know: crate (in French: la caisse): on the floor, ahead of you, 2 m, next to the fridge.", line)
    check(fails, reply_live(v).find("- " + line + "\n- You know of no other place or object in this room.") >= 0, "  in the view, before the closing line")

    if fails > 0:
        print("FAILED:", fails)
        exit(1)
    print("ALL PASS")

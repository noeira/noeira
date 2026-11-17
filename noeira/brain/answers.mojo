# +--------------------------------------------------------------------------+ #
# | Answers the robot computes — said as they are, the reply model not asked
# +--------------------------------------------------------------------------+ #
"""A question whose answer the state holds EXACTLY is answered by code
(docs/ROBOT_BRAIN_PLAN.md §16):

    ws.asked = heard                       # the utterance being answered
    voice.set_reply_override(computed_reply(ws), computed_reply_final(ws))     # "" = the model replies

Two questions today:

- who the speaker is (`noeira.brain.people.identity_answer`, B4);
- whether the robot has finished what it was asked (`task_done_answer`).

⚠ WHY NOT THE REPLY MODEL. With "Your last task: walk to the table, then walk
to the door. It is done." in its prompt, LFM3-1.7B answered « As-tu fini ce
que je t'ai demandé ? » with « Non, j'ai encore besoin de te dire où est la
table et la cuisine pour continuer. » (the room demo on the Orin). The task
record is exact; a sentence about it is not left to a 1.7 B model.

⚠ "" WHEN THE STATE CANNOT SAY. A spoken gesture ("squat") is not a task yet:
it cancels the task in hand and leaves no record of its own. So a cancelled
task, a gesture in progress, or no task at all give "" — the model replies, as
before — rather than a confident sentence about the wrong request.
"""

from .bb1 import normalise, has_phrase
from .events import EV_BEGAN, EV_TASK_DONE, EV_TASK_FAILED, EV_TASK_CANCELLED
from .people import asks_identity, identity_answer
from .state import WorldState, TASK_DONE, TASK_FAILED


def _plain(text: String) -> List[UInt8]:
    """Normalised, with hyphens as spaces: a recogniser writes « as-tu » and
    « as tu » alike."""
    var t = normalise(text)
    for i in range(len(t)):
        if t[i] == 45:
            t[i] = 32
    return t^


def _any(ref text: List[UInt8], forms: List[String]) -> Bool:
    for f in forms:
        var pat = _plain(f)
        if has_phrase(text, pat):
            return True
    return False


def _gesture_since_last_task(ref ws: WorldState) -> Bool:
    """A command that is not a task (a spoken "squat") began after the last
    task ended, as far back as the state's events go."""
    var i = len(ws.events) - 1
    while i >= 0:
        var k = ws.events[i].kind.copy()
        if k == EV_TASK_DONE or k == EV_TASK_FAILED or k == EV_TASK_CANCELLED:
            return False
        if k == EV_BEGAN:
            var c = ws.events[i].text.copy()
            if c != "stand" and not c.startswith("goto"):
                return True
        i -= 1
    return False


def asks_task_done(text: String) -> Bool:
    """Whether a sentence asks the robot if it has finished (French and
    English).

    An inverted form is a question by itself (« as-tu fini », "have you
    finished"). A plain one (« tu as fini ») counts only with a question mark
    or « est-ce que »: « Quand tu as fini, va à la porte. » is a request."""
    var t = _plain(text)
    var inverted: List[String] = [
        "as-tu fini", "as-tu terminé", "as-tu réussi", "avez-vous fini", "avez-vous terminé", "avez-vous réussi",
        "est-ce fini", "est-ce terminé", "est-ce que c'est fait", "est-ce que c'est fini", "est-ce que c'est terminé",
        "have you finished", "have you completed", "are you finished", "are you done", "did you finish",
        "did you complete", "is it done", "is it finished",
    ]
    if _any(t, inverted):
        return True
    var plain: List[String] = [
        "tu as fini", "tu as terminé", "tu as réussi", "t'as fini", "t'as terminé", "vous avez fini", "vous avez terminé",
        "c'est fini", "c'est terminé", "you have finished", "you are done", "you finished", "it is done",
    ]
    if not _any(t, plain):
        return False
    var marks: List[String] = ["?", "est-ce que", "est-ce qu'"]
    return _any(t, marks)


def task_done_answer(ref ws: WorldState) -> String:
    """The exact sentence that answers "have you finished?" from the task
    record, in the conversation's language (French, else English); "" when
    the record cannot say (module note).

    ⚠ A TEMPLATE, like `tasks.notice_text` and `people.identity_answer`."""
    var fr = ws.language == "fr"
    if ws.has_task:
        var n = len(ws.task.steps)
        var k = ws.task.at
        if n > 1:
            if fr:
                return "Pas encore : j'en suis à l'étape " + String(k + 1) + " sur " + String(n) + "."
            return "Not yet: I am on step " + String(k + 1) + " of " + String(n) + "."
        var place = ws.task.steps[k].place.copy()
        if place != "":
            var word = ws.word_for(place, ws.language)
            if fr:
                return "Pas encore, je suis en route vers " + word + "."
            return "Not yet, I am on my way to " + word + "."
        return String("Pas encore, c'est en cours.") if fr else String("Not yet, I am still doing it.")
    if not ws.has_last_task:
        return String("")
    # a gesture asked after the last task ended is what the person means, and it has no record
    if _gesture_since_last_task(ws):
        return String("")
    if ws.last_task.status == TASK_DONE:
        return String("Oui, j'ai fini ce que tu m'as demandé.") if fr else String("Yes, I have finished what you asked.")
    if ws.last_task.status == TASK_FAILED:
        var k = ws.last_task.at
        var place = String("")
        if k < len(ws.last_task.steps):
            place = ws.last_task.steps[k].place.copy()
        if place != "":
            var word = ws.word_for(place, ws.language)
            if fr:
                return "Non, je n'ai pas réussi à atteindre " + word + "."
            return "No, I could not reach " + word + "."
        return String("Non, je n'ai pas su le faire.") if fr else String("No, I could not do it.")
    return String("")


def computed_reply_final(ref ws: WorldState) -> Bool:
    """Whether `computed_reply(ws)` is the answer for good: False while the
    voice behind « Comment je m'appelle ? » is still being checked (its answer
    is then « Un instant… », and the next frames give the name). For
    `G1VoiceLoopOf.set_reply_override`'s `final`: the loop's `computed_first`
    says only a final answer."""
    return not (ws.people_on and asks_identity(ws.asked) and ws.speaker_pending)


def computed_reply(ref ws: WorldState) -> String:
    """What to say to `ws.asked` when code holds the answer; "" = the reply
    model answers. For `G1VoiceLoopOf.set_reply_override`, once a frame."""
    if ws.people_on and asks_identity(ws.asked):
        return identity_answer(ws)
    if asks_task_done(ws.asked):
        return task_done_answer(ws)
    return String("")

# +--------------------------------------------------------------------------+ #
# | Jev as a skill selector — typed, calibrated decisions about a robot state
# +--------------------------------------------------------------------------+ #
"""Three pick-and-place states, three questions each, one call per state.

    pixi run mojo run -I . examples/ai/jev_decide.mojo

Needs `JEV_API_KEY` (or `TYPESAFE_API_KEY`) in the environment or `.env`.

⚠ THE STATE CARRIES WORDS AND BOOLEANS, NOT GEOMETRY TO COMPARE. Jev's own
documentation lists arithmetic and numeric comparison as failure modes, so
the thresholds are applied HERE (`gap_mm < 20` -> `"jaws": "closed"`) and
the model is asked to reason over their outcome.
"""

from noeira.ai.jev import JevClient, JevQuestions


def _state(jaws: String, in_jaws: Bool, height: String, over_target: Bool) -> String:
    return (
        '{"task": "put the red cube in the bowl", "jaws": "' + jaws
        + '", "cube_between_jaws": ' + ("true" if in_jaws else "false")
        + ', "cube_height": "' + height + '", "gripper_above_bowl": '
        + ("true" if over_target else "false") + "}"
    )


def main() raises:
    var jev = JevClient.from_env()

    var q = JevQuestions()
    q.noul(
        "grasped", "Is the cube held by the gripper (between closed jaws)?"
    )
    q.choice(
        "next_skill",
        "Which skill should the arm run next to progress `task`?",
        ["approach", "grasp", "lift", "transport", "release", "done"],
        [
            "move the open gripper above the cube",
            "close the jaws on a cube that is between them",
            "raise a held cube off the table",
            "carry a lifted cube over the bowl",
            "open the jaws above the bowl",
            "the cube is in the bowl, nothing left to do",
        ],
    )
    q.score(
        "progress", "How far along is `task`?",
        ["not started", "cube grasped", "cube lifted", "cube over bowl", "complete"],
    )

    var states = List[String]()
    states.append(_state("open", False, "on table", False))
    states.append(_state("closed", True, "on table", False))
    states.append(_state("closed", True, "lifted", True))

    for i in range(len(states)):
        var a = jev.decide(states[i], q)
        print("state", i, states[i])
        print(
            "  next_skill =", a.choice("next_skill"),
            " confidence", a.confidence("next_skill"),
        )
        print("  P(grasped) =", a.noul("grasped"), "  progress =", a.score("progress"))
        print("  ", a.model, a.input_tokens, "tokens in", a.latency_ms, "ms")

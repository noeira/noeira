# +--------------------------------------------------------------------------+ #
# | An LLM agent driving a (mock) tabletop through tools
# +--------------------------------------------------------------------------+ #
"""The manual tool loop: the model plans, calls skills, reads results.

    pixi run mojo run -I . examples/ai/tool_agent.mojo                # Claude
    pixi run mojo run -I . examples/ai/tool_agent.mojo hf             # Qwen3.5 on HF
    pixi run mojo run -I . examples/ai/tool_agent.mojo ollama:qwen3:8b

The "robot" is a dictionary of object positions. Replace the three skill
functions with the real ones (an ACT policy rollout, an IK move, the SO-101
gripper) and the loop is unchanged: the model only ever sees skill NAMES,
their arguments and a one-line result.

⚠ THE MODEL PICKS SKILLS, IT DOES NOT CLOSE THE CONTROL LOOP. Each tool call
here is one whole skill (seconds of 30 Hz control, run by the policy), and
the model sees its OUTCOME. That is the only split a 1-10 s round trip
supports.
"""

from std.sys import argv

from noeira.ai.chat import ChatClient, Conversation, ToolCall


struct Table(Movable):
    var names: List[String]
    var places: List[String]
    var held: String

    def __init__(out self):
        self.names = ["red_cube", "blue_cube", "bowl"]
        self.places = ["left", "center", "right"]
        self.held = String("")

    def __init__(out self, *, deinit move: Self):
        self.names = move.names^
        self.places = move.places^
        self.held = move.held^

    def describe(self) -> String:
        var s = String("")
        for i in range(len(self.names)):
            s += self.names[i] + " @ " + self.places[i] + "; "
        return s + "holding: " + (self.held if self.held.byte_length() > 0 else "nothing")

    def _find(self, name: String) -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        return -1

    def run(mut self, ref call: ToolCall) raises -> String:
        """Dispatch one tool call. An error string is a RESULT the model can
        read and recover from, not an exception."""
        var a = call.args()
        var root = a.root()
        if call.name == "look":
            return self.describe()
        if call.name == "pick":
            var obj = a.string(a.field(root, "object"))
            var i = self._find(obj)
            if i < 0:
                return "error: no object named " + obj
            if obj == "bowl":
                return "error: the bowl is too large to grasp"
            if self.held.byte_length() > 0:
                return "error: already holding " + self.held
            self.held = obj
            self.places[i] = "gripper"
            return "picked " + obj
        if call.name == "place":
            var target = a.string(a.field(root, "target"))
            if self.held.byte_length() == 0:
                return "error: holding nothing"
            var i = self._find(self.held)
            var t = self._find(target)
            self.places[i] = "in " + target if t >= 0 else target
            var msg = "placed " + self.held + " " + self.places[i]
            self.held = String("")
            return msg
        return "error: unknown tool " + call.name


def main() raises:
    var args = argv()
    var spec = String(args[1]) if len(args) > 1 else String("anthropic")
    var llm = ChatClient.from_spec(spec)
    if spec.startswith("hf"):
        llm.extra("chat_template_kwargs", '{"enable_thinking": false}')

    var conv = Conversation(
        "You control a robot arm over a table through tools. Use `look` before"
        " acting. Pick one object at a time. Be brief."
    )
    conv.tool("look", "List every object and where it is.", '{"type":"object","properties":{}}')
    conv.tool(
        "pick", "Grasp an object with the gripper.",
        '{"type":"object","properties":{"object":{"type":"string"}},"required":["object"]}',
    )
    conv.tool(
        "place",
        "Put the held object down: on a location (left/center/right) or into an object.",
        '{"type":"object","properties":{"target":{"type":"string"}},"required":["target"]}',
    )
    conv.user("Put both cubes in the bowl.")

    var table = Table()
    for step in range(12):
        var r = conv.send(llm)
        if r.text.byte_length() > 0:
            print("[model]", r.text)
        if not r.wants_tools():
            print("-- finished after", step + 1, "turns (", r.stop_reason, ")")
            break
        for i in range(len(r.tool_calls)):
            ref call = r.tool_calls[i]
            var out = table.run(call)
            print("  ", call.name, call.arguments_json, "->", out)
            conv.tool_result(call.id, out)
        print("   (", r.latency_ms, "ms )")
    print("table:", table.describe())

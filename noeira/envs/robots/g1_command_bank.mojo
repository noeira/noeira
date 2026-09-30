# +--------------------------------------------------------------------------+ #
# | Reading the G1 command bank
# +--------------------------------------------------------------------------+ #
"""Load named whole-body commands and hand back their `z`.

`examples/g1/bfm_zero_bank_build.mojo` writes the bank; this reads it. A
consumer looks a command up by name and gets 256 floats. Nothing here touches
the network, the store or `B` — the whole point of paying for latent search
offline is that the runtime cost of a command is a dictionary lookup.

    var bank = G1CommandBank.load("g1_command_bank.txt")
    var i = bank.find("squat")
    if i >= 0:
        bank.copy_z(i, z_tensor)      # -> the policy's conditioning vector

⚠ EVERY ENTRY IN THE FILE HAS PASSED FOUR GATES (see the builder): pool
support, a scaffold control proving the goal term does work, a minimum hold
on the rollout, and the shipped MEAN satisfying its own compound. A name
that is absent was REJECTED, and absence is the useful answer — `find`
returns -1 rather than the nearest match, because a voice or language layer
guessing at the nearest command is exactly how a robot ends up doing
something nobody asked for.
"""

from noeira.io.fileio import read_file_bytes
from noeira.core.bytes import string_from_bytes
from noeira.envs.robots.g1_reward_vocab import (
    g1_vocab_name, OP_GT, OP_LT, OP_BAND, OP_SOFT,
)


def _r2(v: Float64) -> String:
    var neg = v < 0.0
    var a = -v if neg else v
    var h = Int(a * 100.0 + 0.5)
    var f = String(h % 100)
    while f.byte_length() < 2:
        f = String("0") + f
    var b = String(h // 100) + String(".") + f
    return (String("-") + b) if neg else b


struct G1CommandBank(Movable):
    """Names, groups and `z` rows, flat."""

    var names: List[String]
    var groups: List[String]
    var dim: Int
    var z: List[Float64]
    # what the robot actually did under this `z`, for a HUD or a log
    var hard: List[Float64]
    # the pose a consumer should reset into — carried so that nothing has to
    # open the 1.69 GB store for one row
    var qpos: List[Float64]
    var qvel: List[Float64]
    # the compound behind each command, flat, with a slice per entry.
    # ⚠ KEPT SO THAT A DESCRIPTION CAN BE GENERATED FROM IT rather than
    # written twice. A language layer is told what a command does; if that
    # text were a separate field it would drift from the terms the command
    # was actually gated on, and the drift would be invisible — the robot
    # doing one thing while the model was told another.
    var t_q: List[Int]
    var t_op: List[Int]
    var t_lo: List[Float64]
    var t_hi: List[Float64]
    var t_goal: List[Bool]
    var t_start: List[Int]
    var t_count: List[Int]

    def __init__(out self):
        self.names = List[String]()
        self.groups = List[String]()
        self.dim = 0
        self.z = List[Float64]()
        self.hard = List[Float64]()
        self.qpos = List[Float64]()
        self.qvel = List[Float64]()
        self.t_q = List[Int]()
        self.t_op = List[Int]()
        self.t_lo = List[Float64]()
        self.t_hi = List[Float64]()
        self.t_goal = List[Bool]()
        self.t_start = List[Int]()
        self.t_count = List[Int]()

    def __init__(out self, *, deinit move: Self):
        self.names = move.names^
        self.groups = move.groups^
        self.dim = move.dim
        self.z = move.z^
        self.hard = move.hard^
        self.qpos = move.qpos^
        self.qvel = move.qvel^
        self.t_q = move.t_q^
        self.t_op = move.t_op^
        self.t_lo = move.t_lo^
        self.t_hi = move.t_hi^
        self.t_goal = move.t_goal^
        self.t_start = move.t_start^
        self.t_count = move.t_count^

    @staticmethod
    def load(path: String) raises -> G1CommandBank:
        var self = G1CommandBank()
        var raw = read_file_bytes(path)
        var txt = string_from_bytes(raw)
        var pending_name = String("")
        var pending_group = String("")
        var pending_hard = 0.0
        var pending_start = 0
        var lines = txt.split("\n")
        for li in range(len(lines)):
            var l = String(lines[li])
            if l.byte_length() == 0 or l.startswith("#"):
                continue
            var p = l.split(" ")
            if l.startswith("count "):
                if len(p) >= 3:
                    self.dim = atol(String(p[2]))
                continue
            if l.startswith("qpos "):
                for i in range(1, len(p)):
                    self.qpos.append(Float64(String(p[i])))
                continue
            if l.startswith("qvel "):
                for i in range(1, len(p)):
                    self.qvel.append(Float64(String(p[i])))
                continue
            if l.startswith("name "):
                pending_name = String(p[1])
                pending_start = len(self.t_q)
                continue
            if l.startswith("term "):
                if len(p) >= 5:
                    self.t_q.append(atol(String(p[1])))
                    self.t_op.append(atol(String(p[2])))
                    self.t_lo.append(Float64(String(p[3])))
                    self.t_hi.append(Float64(String(p[4])))
                    self.t_goal.append(l.find("GOAL") >= 0)
                continue
            if l.startswith("group "):
                pending_group = String(p[1])
                continue
            if l.startswith("hard "):
                if len(p) >= 3:
                    pending_hard = Float64(String(p[2]))
                continue
            if l.startswith("z "):
                # ⚠ the row must be exactly `dim` wide. A short row would
                # otherwise slide every later command's `z` by its shortfall
                # and each one would drive a DIFFERENT behaviour than its
                # name — silently, since any 256 floats project onto the
                # sphere and produce a plausible robot.
                if len(p) - 1 != self.dim:
                    raise Error(
                        "g1 bank: '" + pending_name + "' has "
                        + String(len(p) - 1) + " values, expected "
                        + String(self.dim)
                    )
                for i in range(1, len(p)):
                    self.z.append(Float64(String(p[i])))
                self.names.append(pending_name)
                self.groups.append(pending_group)
                self.hard.append(pending_hard)
                self.t_start.append(pending_start)
                self.t_count.append(len(self.t_q) - pending_start)
                continue
        if self.dim == 0:
            raise Error("g1 bank: no `count` header in " + path)
        return self^

    def has_pose(self) -> Bool:
        """True when the bank carries its own start pose. Older banks do not,
        and a consumer must fall back to the store."""
        return len(self.qpos) > 0 and len(self.qvel) > 0

    def count(self) -> Int:
        return len(self.names)

    def find(self, name: String) -> Int:
        """Index of `name`, or **-1**.

        ⚠ No fuzzy match, deliberately. The bank's contents are the commands
        that passed their gates; a miss means the thing asked for was either
        rejected or never defined, and the caller should say so rather than
        run the closest-looking entry.
        """
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        return -1

    def name_at(self, i: Int) -> String:
        return self.names[i]

    def group_at(self, i: Int) -> String:
        return self.groups[i]

    def hold_at(self, i: Int) -> Float64:
        """The fraction of the scored window this command held its compound
        when the bank was built — 1.0 for most entries, and worth showing in
        a HUD for the ones below it."""
        return self.hard[i]

    def z_at(self, i: Int, k: Int) -> Float64:
        return self.z[i * self.dim + k]

    def describe(self, i: Int) raises -> String:
        """Command `i` in words, rendered FROM ITS GOAL TERMS.

        This is what a language layer is shown beside the command name, and
        generating it means it cannot disagree with what the command was
        gated on. The scaffold terms (uprightness, head height, stillness)
        are left out: they are the same boilerplate on most entries and say
        nothing about which command to pick.
        """
        var out = String("")
        var st = self.t_start[i]
        for j in range(self.t_count[i]):
            var k = st + j
            if not self.t_goal[k] or self.t_op[k] == OP_SOFT:
                continue
            if out.byte_length() > 0:
                out += String(", ")
            var nm = g1_vocab_name(self.t_q[k])
            if self.t_op[k] == OP_GT:
                out += nm + String(" above ") + _r2(self.t_lo[k])
            elif self.t_op[k] == OP_LT:
                out += nm + String(" below ") + _r2(self.t_hi[k])
            else:
                out += nm + String(" between ") + _r2(self.t_lo[k]) \
                       + String(" and ") + _r2(self.t_hi[k])
        if out.byte_length() == 0:
            return self.groups[i]
        return self.groups[i] + String(": ") + out

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

    def __init__(out self):
        self.names = List[String]()
        self.groups = List[String]()
        self.dim = 0
        self.z = List[Float64]()
        self.hard = List[Float64]()
        self.qpos = List[Float64]()
        self.qvel = List[Float64]()

    def __init__(out self, *, deinit move: Self):
        self.names = move.names^
        self.groups = move.groups^
        self.dim = move.dim
        self.z = move.z^
        self.hard = move.hard^
        self.qpos = move.qpos^
        self.qvel = move.qvel^

    @staticmethod
    def load(path: String) raises -> G1CommandBank:
        var self = G1CommandBank()
        var raw = read_file_bytes(path)
        var txt = string_from_bytes(raw)
        var pending_name = String("")
        var pending_group = String("")
        var pending_hard = 0.0
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

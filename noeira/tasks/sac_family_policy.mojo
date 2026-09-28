"""The SAC policy a task-family run trains — its sizes, in ONE place.

`sac_family_driver` trains with these widths, and every tool that loads its
checkpoints on the CPU (viewers, recorders, probes, the DAgger handover)
builds the same nets: a checkpoint loads by parameter name AND size, so a
width copied by hand into five files was five chances to load nothing.

    from noeira.tasks.sac_family_policy import (
        SacFamilyPolicy, HIDDEN, POLICY_BATCH, POLICY_CAP,
    )
    var agent: SacFamilyPolicy[OBS, ACT] = SAC[
        "cpu", OBS, ACT, POLICY_BATCH, POLICY_CAP, HIDDEN
    ](action_scale=..., learning_starts=0)
"""

from noeira.deep_agents.sac import SACPresetAgent

comptime HIDDEN = 256
"""Hidden width of the family SAC actor and critics (both layers)."""
comptime POLICY_BATCH = 256
comptime POLICY_CAP = 1000
"""`POLICY_BATCH` / `POLICY_CAP` size a replay a loaded policy never fills;
the checkpoint holds no replay. Only `HIDDEN` has to match the trainer."""

comptime SacFamilyPolicy[OBS: Int, ACT: Int] = SACPresetAgent[
    "cpu", OBS, ACT, POLICY_BATCH, POLICY_CAP, HIDDEN
]
"""A family checkpoint, loaded on the CPU."""

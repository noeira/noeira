# +--------------------------------------------------------------------------+ #
# | Which recording a SmolVLA build is for — ONE place, read by both ends
# +--------------------------------------------------------------------------+ #
"""The instruction table and its token count, shared by fine-tune and deploy.

⚠⚠ `N_LANG` IS A COMPILE-TIME PROPERTY OF THE RECORDING, NOT OF THE MODEL. It
is the token length of the task string, it sets the prefix length `P`, and a
fine-tune and a deployment built with different values cannot run the same
instruction. It used to be a `comptime N_LANG = 6` written separately in the
fine-tune and the deployment, both true only for `"Grab the green cube"`. The
next recording's instruction — `"Grab the cube and put it in the bowl"` — is
11 tokens, and the mismatch would have surfaced as a refusal at the first step,
after a full build on a rented box.

⚠ THE COUNT WAS MEASURED, NOT ESTIMATED: the SmolVLM2 tokenizer (the backbone's,
not SmolVLA's) reproduced the checked-in ids of the previous table exactly
before being trusted with this one. `tools/vla/dump_smolvla_tasks.py` is the
reference generator; a new recording regenerates the table and changes BOTH
constants here, together.
"""

from std.sys.defines import get_defined_int, get_defined_string

comptime SO101_TASKS = get_defined_string[
    "SMOLVLA_TASKS_TSV", "tools/vla/smolvla_tasks_so101-tower_cube-in-bowl.tsv"
]()
comptime SO101_N_LANG: Int = get_defined_int["SMOLVLA_N_LANG", 11]()
"""⚠ Another recording is a BUILD of both ends with the same two defines, e.g.
the printed-props one (15 tokens):

    -D SMOLVLA_N_LANG=15 \\
    -D SMOLVLA_TASKS_TSV=tools/vla/smolvla_tasks_so101-tower_cube-in-bowl-printed.tsv

on the fine-tune AND the deployment; the fine-tune refuses a table whose length
is not `SO101_N_LANG` at its first step, the deployment at its start."""

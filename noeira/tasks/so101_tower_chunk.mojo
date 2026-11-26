"""The SO-101 tower's CHUNKED state policy — its shape, in one place.

    from noeira.tasks.so101_tower_chunk import (
        TOWER_ACT, TOWER_CHUNK_H, TOWER_CHUNK_EXEC, TOWER_CHUNK_OUT,
        TowerChunkNet,
    )

A behaviour-cloning net from the family's state observation to the next
`TOWER_CHUNK_H` actions (`deep_agents/bc`, fitted by
`examples/so101/tower_chunk_bc_fit.mojo`). It stands in for a chunked policy
(ACT, a VLA) where the question is the chunk and its delay, not the pixels:
`examples/so101/tower_async_rl.mojo` runs it under an emulated inference
delay (`deep_agents/bc/async_chunk.mojo`).

The trainer and the driver both read these constants: a checkpoint loads by
parameter name and size, so a horizon spelled twice is a checkpoint that
raises on load, or worse, a chunk read at the wrong stride.
"""

from noeira.deep_agents.bc.policy import BcNet
from noeira.tasks.so101_tower_xml import So101TowerModel


comptime TOWER_ACT: Int = 6
"""The family's action: six normalised joint targets."""

comptime TOWER_CHUNK_H: Int = 32
"""Actions per chunk: about one second at the family's 32 ms control step."""

comptime TOWER_CHUNK_EXEC: Int = 16
"""Actions executed from each chunk before the next one takes over. The
largest delay the loop can carry is `TOWER_CHUNK_H - TOWER_CHUNK_EXEC`."""

comptime TOWER_CHUNK_OUT: Int = TOWER_CHUNK_H * TOWER_ACT

comptime TowerChunkNet = BcNet[So101TowerModel.OBS_DIM, TOWER_CHUNK_OUT]

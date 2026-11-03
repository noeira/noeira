"""EpisodicReplayBuffer — the trait over per-env, episode-aware window replay.

The third replay shape, beside `ReplayBuffer` (single transitions) and
`SequenceReplayBuffer` (one flat ring, envs interleaved in lockstep, windows
drawn uniformly with no boundary test). This one keeps a RING PER ENV with its
own write pointer, so an env may write a variable number of rows per step —
which a model-based method needs when an episode ends and the final state
gets a TERMINAL row of its own — and it samples only windows that stay inside
one episode.

A row holds the state at the start of a control step and the transition that
followed: clean action, executed action (what the simulator ran), a per-row
context vector (domain-randomisation parameters, a goal id…), reward, the
termination LABEL of the row's state, an integer frame (e.g. a reference-clip
frame), and bookkeeping: episode id, `terminal`, `ep_start`.

    window (k)  rows t..t+k valid, one episode, t..t+k−1 non-terminal, and
                clear of the write pointer (it may be overwritten next)
    start (h)   a non-terminal row with the h rows before it, backfilled with
                the episode's first row before its start; the clean action
                before each history state (0 at the episode's start)

Widths are comptime (`S`, `A`, `X`); env count and per-env capacity are
runtime. Device backends override the raising hooks below (next round).
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT


trait EpisodicReplayBuffer(Movable):
    comptime S: Int   # state width
    comptime A: Int   # action width
    comptime X: Int   # context width

    def add(
        mut self, e: Int, s: List[Scalar[DT]], a: List[Scalar[DT]], a_exec: List[Scalar[DT]],
        x: List[Scalar[DT]], r: Float64, done: Float64, frame: Int, terminal: Bool, ep_start: Bool,
    ):
        """Append one row to env `e`'s ring."""
        ...

    def end_episode(mut self, e: Int):
        """Close env `e`'s episode: its next row carries a new episode id."""
        ...

    def window_mask(self, k: Int) -> List[Bool]:
        """[E·T] True where a length-k window may start."""
        ...

    def start_mask(self, hist: Int) -> List[Bool]:
        """[E·T] True where a start with an h-row history may be drawn."""
        ...

    def num_envs(self) -> Int:
        ...

    def capacity(self) -> Int:
        ...

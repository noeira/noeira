"""Prior rows — a teacher's rollouts mixed into FB-MEBE's training batch.

`noeira-docs/FB_MEBE_G1_PLAN.md` §13.5 option 1. A collector
(`examples/g1/bfm_zero_prior_collect.mojo`) rolls a trained policy out on the
SAME batched env the learner explores, and writes a `TrajectoryStore`:

    obs     float32 [W]     the env's stored observation, pre-step
    action  float32 [ACT]   the action taken from it
    r_env   float32 [1]     the env's regularity half for that transition
                            (`g1_reg_env_kernel`, default weights), post-step
    task    int32   [1]     which prompt drove the lane (diagnostic)

⚠ ONE STORE CONVENTION: every episode ENDS with its post-step terminal
observation, a row with no transition of its own (action 0, r_env 0,
task -1). So an episode of `n` rows holds `n - 1` transitions, each with its
real successor — never a post-reset observation, which is what the online
ring has to flag and remap (`ring_indices_kernel`).

`prior_rows` turns that into what `FBOnlineAgent.attach_expert` samples:

    obs table   [n_valid transition rows | n_term terminal rows]
    nxt         n_valid entries: the table row of each successor
    act, rew    n_valid entries

`attach_expert` draws uniformly over `[0, n_valid)`, so a terminal row is
reachable only as a successor — the expert path's old `nxt[last] = last`
self-loop is not needed.

The regularity reward is recomputed HERE, with the learner's weights, by
the same rule `reg_reward_kernel` applies to the ring:

    r = − w_acc·‖(q̇' − q̇)/dt‖² − w_ar·‖a − a_prev‖² + w_env·env

with `a_prev = 0` on an episode's first row, `q̇` read at
`[qd_off, qd_off + qd_n)`, and
`env = r_env − w_yaw·ωz'² − w_angxy·(ωx'² + ωy'²)` — the ring's
`g1_angvel_price_kernel`, on the SUCCESSOR's angular velocity at
`[ang_off, ang_off + 3)` (`ang_off < 0`: no price).

`PriorRows.windows(seq)` is what the prior-row BC encodes its `z` from
(FB-CPR's `encode_expert`, `fb/cpr.mojo`): for transition row `p`, the table
rows of its next `seq` successors `s'_p, s'_{p+1}, …`, following `nxt` and
repeating the episode's terminal row once it is reached, so a window never
leaves its episode.
"""

from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.data.resident import IDX_DT
from noeira.data.store import TrajectoryStore
from .online_mebe import RegRewardConfig


@fieldwise_init
struct PriorPrice(Copyable, ImplicitlyCopyable, Movable):
    """The angular-velocity price the driver adds to the ring's env term
    (`--w-yaw`, `--w-angxy`); `ang_off` is the first of (ωx, ωy, ωz)."""

    var ang_off: Int
    var w_yaw: Float64
    var w_angxy: Float64

    @staticmethod
    def none() -> Self:
        return Self(-1, 0.0, 0.0)


struct PriorRows(Movable):
    """The attach-ready layout (module header). Host tensors; `upload` puts
    `obs`, `act` and `rew` on the device and returns the `nxt` buffer."""

    var obs: Tensor     # (n_valid + n_term) * W
    var act: Tensor     # n_valid * ACT
    var rew: Tensor     # n_valid
    var nxt: List[Int]  # n_valid
    var n_valid: Int
    var n_term: Int
    var w: Int
    var act_dim: Int

    def __init__(out self, w: Int, act_dim: Int):
        self.obs = Tensor()
        self.act = Tensor()
        self.rew = Tensor()
        self.nxt = List[Int]()
        self.n_valid = 0
        self.n_term = 0
        self.w = w
        self.act_dim = act_dim

    def __init__(out self, *, deinit move: Self):
        self.obs = move.obs^
        self.act = move.act^
        self.rew = move.rew^
        self.nxt = move.nxt^
        self.n_valid = move.n_valid
        self.n_term = move.n_term
        self.w = move.w
        self.act_dim = move.act_dim

    def take_obs(mut self) -> Tensor:
        """Hand the table to `attach_prior` (a field cannot be moved out of
        the middle of a value); `self.obs` is left empty."""
        var t = self.obs^
        self.obs = Tensor()
        return t^

    def take_act(mut self) -> Tensor:
        var t = self.act^
        self.act = Tensor()
        return t^

    def take_rew(mut self) -> Tensor:
        var t = self.rew^
        self.rew = Tensor()
        return t^

    def windows(self, seq: Int) raises -> List[Int]:
        """`[n_valid, seq]` table rows, row-major (module header)."""
        if seq < 1:
            raise Error("PriorRows.windows: seq must be >= 1")
        var out = List[Int](length=self.n_valid * seq, fill=0)
        for p in range(self.n_valid):
            var q = p
            for j in range(seq):
                var s = self.nxt[q]
                out[p * seq + j] = s
                if s >= self.n_valid:   # the episode's terminal row: hold it
                    for k in range(j + 1, seq):
                        out[p * seq + k] = s
                    break
                q = s
        return out^

    def upload_windows(self, ctx: DeviceContext, seq: Int) raises -> DeviceBuffer[IDX_DT]:
        var w = self.windows(seq)
        var n = len(w)
        var h = ctx.enqueue_create_host_buffer[IDX_DT](n)
        for i in range(n):
            h[i] = Scalar[IDX_DT](w[i])
        var d = ctx.enqueue_create_buffer[IDX_DT](n)
        ctx.enqueue_copy(d, h)
        ctx.synchronize()
        return d^

    def upload(mut self, ctx: DeviceContext) raises -> DeviceBuffer[IDX_DT]:
        self.obs.upload(ctx)
        self.act.upload(ctx)
        self.rew.upload(ctx)
        var nh = ctx.enqueue_create_host_buffer[IDX_DT](self.n_valid)
        for i in range(self.n_valid):
            nh[i] = Scalar[IDX_DT](self.nxt[i])
        var nd = ctx.enqueue_create_buffer[IDX_DT](self.n_valid)
        ctx.enqueue_copy(nd, nh)
        ctx.synchronize()
        return nd^


def prior_rows(
    ref obs: List[Float32],
    ref act: List[Float32],
    ref r_env: List[Float32],
    ref ep_off: List[Int],
    ref ep_len: List[Int],
    w: Int,
    act_dim: Int,
    reg: RegRewardConfig,
    price: PriorPrice,
) raises -> PriorRows:
    """The layout and the reward (module header), from a store's columns.
    Episodes shorter than 2 rows hold no transition and are skipped."""
    var n_ep = len(ep_off)
    var n_valid = 0
    var n_term = 0
    for e in range(n_ep):
        if ep_len[e] >= 2:
            n_valid += ep_len[e] - 1
            n_term += 1
    if n_valid == 0:
        raise Error("prior_rows: no transitions in the store")
    var out = PriorRows(w, act_dim)
    out.n_valid = n_valid
    out.n_term = n_term
    out.obs = Tensor.alloc((n_valid + n_term) * w)
    out.act = Tensor.alloc(n_valid * act_dim)
    out.rew = Tensor.alloc(n_valid)
    out.nxt = List[Int](length=n_valid, fill=0)
    var inv_dt = 1.0 / reg.dt
    var p = 0           # next transition slot
    var t = n_valid     # next terminal slot
    for e in range(n_ep):
        var off = ep_off[e]
        var ln = ep_len[e]
        if ln < 2:
            continue
        for j in range(ln - 1):
            var s = off + j
            for k in range(w):
                out.obs.data[p * w + k] = Scalar[DT](obs[s * w + k])
            for k in range(act_dim):
                out.act.data[p * act_dim + k] = Scalar[DT](act[s * act_dim + k])
            out.nxt[p] = p + 1 if j + 1 < ln - 1 else t
            # the regularity reward of transition s -> s + 1
            var acc = 0.0
            for k in range(reg.qd_n):
                var c = reg.qd_off + k
                var dq = (Float64(obs[(s + 1) * w + c]) - Float64(obs[s * w + c])) * inv_dt
                acc += dq * dq
            var ar = 0.0
            for k in range(act_dim):
                var ap = Float64(act[(s - 1) * act_dim + k]) if j > 0 else 0.0
                var da = Float64(act[s * act_dim + k]) - ap
                ar += da * da
            var env = Float64(r_env[s])
            if price.ang_off >= 0:
                var o = (s + 1) * w + price.ang_off
                var wx = Float64(obs[o])
                var wy = Float64(obs[o + 1])
                var wz = Float64(obs[o + 2])
                env -= price.w_yaw * wz * wz + price.w_angxy * (wx * wx + wy * wy)
            out.rew.data[p] = Scalar[DT](
                -reg.w_acc * acc - reg.w_ar * ar + reg.w_env * env
            )
            p += 1
        var last = off + ln - 1
        for k in range(w):
            out.obs.data[t * w + k] = Scalar[DT](obs[last * w + k])
        t += 1
    return out^


def load_prior_rows(
    path: String,
    w: Int,
    act_dim: Int,
    reg: RegRewardConfig,
    price: PriorPrice,
) raises -> PriorRows:
    """`prior_rows` over a collector's store. Checks the column widths, so a
    store written for another observation fails here, not as noise later."""
    var store = TrajectoryStore(path)
    if store.column(String("obs")).row_dim() != w:
        raise Error(
            "load_prior_rows: 'obs' is " + String(store.column(String("obs")).row_dim())
            + " wide, the learner stores " + String(w)
        )
    if store.column(String("action")).row_dim() != act_dim:
        raise Error("load_prior_rows: 'action' width differs from the learner's")
    var obs = store.load_column[DType.float32](String("obs"), max_bytes=1 << 34)
    var act = store.load_column[DType.float32](String("action"), max_bytes=1 << 32)
    var r_env = store.load_column[DType.float32](String("r_env"))
    var ep_off = List[Int]()
    var ep_len = List[Int]()
    for e in range(store.n_episodes()):
        ep_off.append(Int(store.episodes.ep_offset[e]))
        ep_len.append(Int(store.episodes.ep_len[e]))
    return prior_rows(obs, act, r_env, ep_off, ep_len, w, act_dim, reg, price)

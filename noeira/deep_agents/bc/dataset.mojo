"""What behaviour cloning fits: observation rows and action rows, already
split into training and validation.

⚠ THE SPLIT IS THE SOURCE'S, AND IT IS BY DEMONSTRATION. Consecutive frames
of one demo are nearly identical; a row-wise split puts a frame's neighbours
on both sides and reports a validation error that means nothing. So a source
(LIBERO's store, a `.demo` file) holds out whole episodes and fills the two
halves itself.
"""

from noeira.nn.constants import DT
from noeira.deep_agents.demos.file import DemoSet
from noeira.deep_agents.demos.filter import DemoFilter


struct BcDataset(Movable):
    var obs_dim: Int
    var act_dim: Int
    var x_tr: List[Scalar[DT]]
    var y_tr: List[Scalar[DT]]
    var x_va: List[Scalar[DT]]
    var y_va: List[Scalar[DT]]
    var n_tr: Int
    var n_va: Int

    def __init__(out self, obs_dim: Int, act_dim: Int):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.x_tr = List[Scalar[DT]]()
        self.y_tr = List[Scalar[DT]]()
        self.x_va = List[Scalar[DT]]()
        self.y_va = List[Scalar[DT]]()
        self.n_tr = 0
        self.n_va = 0

    def add[T: DType, TA: DType](
        mut self, ref obs: List[Scalar[T]], ref act: List[Scalar[TA]],
        validation: Bool,
    ) raises:
        if len(obs) != self.obs_dim or len(act) != self.act_dim:
            raise Error(
                "BcDataset.add: a row of " + String(len(obs)) + " / "
                + String(len(act)) + " words, the dataset is "
                + String(self.obs_dim) + " / " + String(self.act_dim)
            )
        if validation:
            self.n_va += 1
            for j in range(self.obs_dim):
                self.x_va.append(Scalar[DT](obs[j]))
            for j in range(self.act_dim):
                self.y_va.append(Scalar[DT](act[j]))
        else:
            self.n_tr += 1
            for j in range(self.obs_dim):
                self.x_tr.append(Scalar[DT](obs[j]))
            for j in range(self.act_dim):
                self.y_tr.append(Scalar[DT](act[j]))

    def add_demo_set_chunked(
        mut self, ref ds: DemoSet, horizon: Int, val_episodes: Int,
        max_episodes: Int = 0,
    ) raises -> Int:
        """Append a recording as CHUNK rows: the observation of row `r`
        against the `horizon` actions recorded from `r` on, concatenated.
        Returns the number of episodes used.

        Only successful episodes are used, the first `max_episodes` of them
        (0 = all); the LAST `val_episodes` of those are held out. Past an
        episode's end the last action is repeated: the demonstrator was
        holding, and the chunk says so."""
        if horizon < 1 or self.act_dim != horizon * ds.act_dim:
            raise Error(
                "BcDataset: chunks of " + String(horizon) + " x "
                + String(ds.act_dim) + " words into a dataset of "
                + String(self.act_dim)
            )
        if self.obs_dim != ds.obs_dim:
            raise Error(
                "BcDataset: a recording of " + String(ds.obs_dim)
                + "-word observations into a dataset of " + String(self.obs_dim)
            )
        var used = List[Int]()
        for e in range(ds.n_episodes()):
            if not ds.ep_success[e] or ds.ep_len[e] < 1:
                continue
            if max_episodes > 0 and len(used) >= max_episodes:
                break
            used.append(e)
        if val_episodes >= len(used):
            raise Error(
                "BcDataset: holding out " + String(val_episodes) + " of "
                + String(len(used)) + " usable episodes leaves nothing to fit"
            )
        var o = List[Scalar[DT]](length=ds.obs_dim, fill=Scalar[DT](0))
        var a = List[Scalar[DT]](length=ds.act_dim, fill=Scalar[DT](0))
        var y = List[Scalar[DT]](length=self.act_dim, fill=Scalar[DT](0))
        var first_val = len(used) - val_episodes
        for u in range(len(used)):
            var e = used[u]
            var start = ds.ep_start[e]
            var ln = ds.ep_len[e]
            for k in range(ln):
                ds.row_obs[DT](start + k, o)
                for i in range(horizon):
                    var src = k + i
                    if src > ln - 1:
                        src = ln - 1
                    ds.row_act[DT](start + src, a)
                    for j in range(ds.act_dim):
                        y[i * ds.act_dim + j] = a[j]
                self.add(o, y, u >= first_val)
        return len(used)

    @staticmethod
    def from_demo_set(
        ref ds: DemoSet, filter: DemoFilter, val_episodes: Int,
    ) raises -> BcDataset:
        """A recording's kept rows; its LAST `val_episodes` episodes held out."""
        if val_episodes >= ds.n_episodes():
            raise Error(
                "BcDataset: holding out " + String(val_episodes) + " of "
                + String(ds.n_episodes()) + " episodes leaves nothing to fit"
            )
        var out = BcDataset(ds.obs_dim, ds.act_dim)
        var o = List[Scalar[DT]](length=ds.obs_dim, fill=Scalar[DT](0))
        var a = List[Scalar[DT]](length=ds.act_dim, fill=Scalar[DT](0))
        var first_val = ds.n_episodes() - val_episodes
        for e in range(ds.n_episodes()):
            var start = ds.ep_start[e]
            for r in range(start, start + ds.ep_len[e]):
                if not filter.keeps(ds, r):
                    continue
                ds.row_obs[DT](r, o)
                ds.row_act[DT](r, a)
                out.add(o, a, e >= first_val)
        return out^

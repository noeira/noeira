"""Gradient buckets for an allreduce that overlaps the backward.

A bucket is a contiguous slice of the gradient arena, reduced by one
collective call. It is ready when the last of its parameters has its final
gradient, which `GradReady` marks report during the backward (`grad_marks`).

`plan_buckets` is a pure host function: from the parameters' arena slices and
the marks' arena ranges it gives each parameter the index of the mark that
completes it (or `n_marks`, "end of backward", when no mark covers it), cuts
the arena into segments of equal readiness at parameter starts, and merges
neighbouring segments in readiness order while a bucket stays within
`target` elements. It never splits a segment, so a module larger than
`target` is one bucket.

Readiness order is backward order: the backward reaches the last layers
first, which sit at the END of the arena (it is laid out in forward order).
So buckets come out from high offsets to low, as PyTorch DDP's do.
"""


@fieldwise_init
struct Bucket(Copyable, Movable, Writable):
    var off: Int
    """First arena element."""
    var n: Int
    """Elements."""
    var ready: Int
    """Index of the mark after which every gradient in it is final;
    `n_marks` means the end of the backward."""


def plan_buckets(
    starts: List[Int],
    sizes: List[Int],
    total: Int,
    mark_lo: List[Int],
    mark_hi: List[Int],
    target: Int,
) raises -> List[Bucket]:
    """Buckets tiling `[0, total)`, in the order to launch them.

    `starts` / `sizes`: every parameter's arena slice, ascending by start.
    `mark_lo` / `mark_hi`: element range of each mark, in mark order.
    Raises if a parameter straddles a mark's range (a wrapped module whose
    parameters are not contiguous in the arena)."""
    var np = len(starts)
    var k_end = len(mark_lo)
    if np == 0:
        raise Error("plan_buckets: no parameters")
    for i in range(1, np):
        if starts[i] < starts[i - 1] + sizes[i - 1]:
            raise Error("plan_buckets: parameters not ascending in the arena")
    if starts[np - 1] + sizes[np - 1] > total:
        raise Error("plan_buckets: a parameter ends past the arena")

    # 1. readiness of each parameter
    var ready = List[Int](capacity=np)
    for i in range(np):
        var a = starts[i]
        var b = a + sizes[i]
        var rd = k_end
        for k in range(k_end):
            var inside = a >= mark_lo[k] and b <= mark_hi[k]
            var overlaps = a < mark_hi[k] and b > mark_lo[k]
            if inside:
                rd = k
                break
            if overlaps:
                raise Error(
                    "plan_buckets: parameter at " + String(a)
                    + " straddles mark " + String(k)
                )
        ready.append(rd)

    # 2. segments of equal readiness, cut at parameter starts, tiling [0, total)
    var seg = List[Bucket]()
    var i = 0
    while i < np:
        var j = i
        while j + 1 < np and ready[j + 1] == ready[i]:
            j += 1
        var lo = 0 if i == 0 else starts[i]
        var hi = total if j == np - 1 else starts[j + 1]
        seg.append(Bucket(lo, hi - lo, ready[i]))
        i = j + 1

    # 3. launch order: readiness ascending, then high offsets first
    var order = List[Int](capacity=len(seg))
    for s in range(len(seg)):
        order.append(s)
    for a in range(1, len(order)):
        var x = order[a]
        var b = a
        while b > 0 and (
            seg[order[b - 1]].ready > seg[x].ready
            or (
                seg[order[b - 1]].ready == seg[x].ready
                and seg[order[b - 1]].off < seg[x].off
            )
        ):
            order[b] = order[b - 1]
            b -= 1
        order[b] = x

    # 4. merge neighbours in that order while the bucket stays within target
    var out = List[Bucket]()
    var cur = seg[order[0]].copy()
    for t in range(1, len(order)):
        ref s = seg[order[t]]
        var adjacent = s.off + s.n == cur.off or cur.off + cur.n == s.off
        if adjacent and cur.n + s.n <= target:
            cur.off = min(cur.off, s.off)
            cur.n += s.n
            cur.ready = max(cur.ready, s.ready)
        else:
            out.append(cur.copy())
            cur = s.copy()
    out.append(cur^)

    # 5. they must tile the arena exactly
    var covered = 0
    for b in range(len(out)):
        covered += out[b].n
    if covered != total:
        raise Error(
            "plan_buckets: buckets cover " + String(covered) + " of "
            + String(total) + " elements"
        )
    return out^

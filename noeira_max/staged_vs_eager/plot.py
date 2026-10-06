"""Draws the eager-vs-staged crossover from ``staged_vs_eager``'s ``RESULT``
lines: delivered latency against batch size, one pair of curves per width,
log-log. Plain SVG, no plotting library.

    python noeira_max/staged_vs_eager/plot.py crossover.log crossover.svg
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

W, H = 760, 460
LEFT, RIGHT, TOP, BOTTOM = 70, 170, 30, 50
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def time_label(us: float) -> str:
    if us >= 1e6:
        return f"{us / 1e6:g} s"
    if us >= 1e3:
        return f"{us / 1e3:g} ms"
    return f"{us:g} µs"


def main(log: Path, out: Path) -> None:
    rows = [json.loads(l.split("RESULT ", 1)[1]) for l in log.read_text().splitlines()
            if l.startswith("RESULT ")]
    by_width = defaultdict(list)
    for r in rows:
        by_width[r["width"]].append(r)
    xs = [r["batch"] for r in rows]
    ys = [v for r in rows for v in (r["eager_us"], r["staged_us"])]
    lx0, lx1 = math.log10(min(xs)), math.log10(max(xs))
    ly0, ly1 = math.floor(math.log10(min(ys))), math.ceil(math.log10(max(ys)))

    def px(batch: float) -> float:
        return LEFT + (math.log10(batch) - lx0) / (lx1 - lx0) * (W - LEFT - RIGHT)

    def py(us: float) -> float:
        return H - BOTTOM - (math.log10(us) - ly0) / (ly1 - ly0) * (H - TOP - BOTTOM)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           'font-family="sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>']
    for e in range(ly0, ly1 + 1):
        y = py(10 ** e)
        svg.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{W - RIGHT}" y2="{y:.1f}" stroke="#ddd"/>')
        svg.append(f'<text x="{LEFT - 6}" y="{y + 4:.1f}" text-anchor="end">{time_label(10 ** e)}</text>')
    for b in sorted(set(xs)):
        x = px(b)
        svg.append(f'<line x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{H - BOTTOM}" stroke="#eee"/>')
        svg.append(f'<text x="{x:.1f}" y="{H - BOTTOM + 16}" text-anchor="middle">{b}</text>')
    svg.append(f'<text x="{(LEFT + W - RIGHT) / 2}" y="{H - 10}" text-anchor="middle">'
               'batch size</text>')
    for i, (width, points) in enumerate(sorted(by_width.items())):
        color = COLORS[i % len(COLORS)]
        points.sort(key=lambda r: r["batch"])
        for key, dash, label in (("eager_us", "", "eager (noeira nn)"),
                                 ("staged_us", ' stroke-dasharray="6 4"', "staged (MAX graph)")):
            path = " ".join(f"{px(r['batch']):.1f},{py(r[key]):.1f}" for r in points)
            svg.append(f'<polyline points="{path}" fill="none" stroke="{color}" '
                       f'stroke-width="2"{dash}/>')
            y = TOP + 18 * (2 * i + (key == "staged_us"))
            svg.append(f'<line x1="{W - RIGHT + 12}" y1="{y}" x2="{W - RIGHT + 40}" y2="{y}" '
                       f'stroke="{color}" stroke-width="2"{dash}/>')
            svg.append(f'<text x="{W - RIGHT + 46}" y="{y + 4}">H {width}, {label.split(" ")[0]}</text>')
    svg.append("</svg>")
    out.write_text("\n".join(svg) + "\n")
    print(f"wrote {out}")
    for width, points in sorted(by_width.items()):
        faster = [r["batch"] for r in points if r["staged_us"] < r["eager_us"]]
        print(f"  width {width}: staged faster at batch {faster or 'none'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

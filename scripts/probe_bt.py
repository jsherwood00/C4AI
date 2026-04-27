#!/usr/bin/env python3
"""Render the BT-rankings figure for the four SANDBAGGING-PROBE variants only.

Same styling as all_bt.py but restricted to the four GPT-5.4 probe
conditions: eval/Docker, eval/non-Docker, non-eval/Docker,
non-eval/non-Docker. Reads `bt_ratings.json` from the same folder and
writes a vector PDF.

Usage:
    python3 probe_bt.py
    python3 probe_bt.py --ratings bt_ratings.json --out my_output.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["pdf.compression"] = 6
mpl.rcParams["svg.fonttype"] = "none"

ROOT = Path(__file__).resolve().parent

TITLE = "BT rankings: main GPT-5.4 vs. sandbagging probes"
SUBTITLE = "One dot per trial.  GPT-5.4 main (leftmost) plus four probe conditions."

FAMILY_SPEC = [
    ("codex",      "GPT-5.4\n(main)",                "#55a868"),
    ("eval_d",     "GPT-5.4\neval\nDocker",         "#8172b2"),
    ("eval_nd",    "GPT-5.4\neval\nnon-Docker",     "#937860"),
    ("noneval_d",  "GPT-5.4\nnon-eval\nDocker",     "#da8bc3"),
    ("noneval_nd", "GPT-5.4\nnon-eval\nnon-Docker", "#5f7a8a"),
]

PONS_COLOR = "#b8860b"
PONS_RATING = 2000.0


def family_key_of(pid: str) -> str | None:
    # IMPORTANT: check longer prefixes first to avoid "eval_d" matching "eval_nd".
    if pid.startswith("eval_nd"):    return "eval_nd"
    if pid.startswith("eval_d"):     return "eval_d"
    if pid.startswith("noneval_nd"): return "noneval_nd"
    if pid.startswith("noneval_d"):  return "noneval_d"
    if pid.startswith("codex_"):     return "codex"
    return None


def load(ratings_path: Path):
    data = json.loads(ratings_path.read_text())
    ratings = data["ratings"]
    buckets: dict[str, list[tuple[str, float]]] = {k: [] for k, _, _ in FAMILY_SPEC}
    for pid, r in ratings.items():
        fk = family_key_of(pid)
        if fk is None:
            continue
        buckets[fk].append((pid, float(r)))
    return buckets


def _darken(hex_color: str, amount: float) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = r * (1 - amount); g = g * (1 - amount); b = b * (1 - amount)
    return (r / 255, g / 255, b / 255)


def _beeswarm_offsets(values: np.ndarray, half_width: float, y_threshold: float) -> np.ndarray:
    n = len(values)
    offsets = np.zeros(n)
    order = np.argsort(values)
    placed: list[tuple[float, int]] = []
    step = half_width / 4.0
    slot_sequence = [0] + [s for k in range(1, 10) for s in (k, -k)]
    for i in order:
        y = float(values[i])
        used = {slot for (py, slot) in placed if abs(y - py) < y_threshold}
        chosen = next(
            (s for s in slot_sequence if s not in used and abs(s) * step <= half_width),
            0,
        )
        offsets[i] = chosen * step
        placed.append((y, chosen))
    return offsets


def plot(buckets, out_path: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444",
        "axes.labelcolor": "#222",
        "xtick.color": "#222",
        "ytick.color": "#333",
        "axes.titlesize": 23,
        "axes.labelsize": 19,
    })

    # Width proportional to 4 columns (roughly 4/7 of full chart's 19).
    # Same vertical proportions as all_bt.py (no title block).
    fig = plt.figure(figsize=(13, 9.2))
    ax = fig.add_axes([0.09, 0.18, 0.89, 0.76])

    labels = [lbl for _, lbl, _ in FAMILY_SPEC]
    colors = [c for _, _, c in FAMILY_SPEC]
    keys = [k for k, _, _ in FAMILY_SPEC]

    n = len(FAMILY_SPEC)
    xs = np.arange(n)

    HALF = 0.28
    Y_THRESHOLD = 75

    for x, k, c in zip(xs, keys, colors):
        vals = np.array([r for _, r in buckets[k]])
        if len(vals) == 0:
            continue

        jx = x + _beeswarm_offsets(vals, half_width=HALF, y_threshold=Y_THRESHOLD)
        ax.scatter(
            jx, vals,
            s=340, color=c, edgecolor="white", linewidth=2.4, zorder=4,
            alpha=0.95,
        )

        m = float(np.mean(vals))
        dark = _darken(c, 0.35)
        ax.hlines(m, x - HALF, x + HALF, colors=dark, linewidth=4.4,
                  zorder=5, capstyle="round")
        ax.text(x + HALF + 0.05, m, f"{m:,.0f}",
                va="center", ha="left", fontsize=19,
                color=dark, fontweight="700", zorder=5)

    # Pons reference line.
    ax.axhline(PONS_RATING, color=PONS_COLOR, linewidth=2.4, linestyle=(0, (5, 3)), zorder=1)
    ax.text(n - 0.5, PONS_RATING + 22, "Pons = 2000",
            ha="right", va="bottom", color=PONS_COLOR, fontsize=17, fontweight="bold")

    ax.set_ylim(-320, 2130)
    ax.set_yticks(np.arange(-200, 2001, 200))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(axis="y", color="#eceff2", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)

    # All probe labels are already three-line; padding is a no-op for them
    # but kept for defensive consistency.
    def _pad_to_three_lines(lbl: str) -> str:
        lines = lbl.count("\n") + 1
        return lbl + "\n " * (3 - lines) if lines < 3 else lbl
    padded = [_pad_to_three_lines(lbl) for lbl in labels]
    ax.set_xticks(xs)
    ax.set_xticklabels(padded, fontsize=17, fontweight="bold")
    ax.set_xlim(-0.65, n - 0.35)
    ax.set_xlabel("")
    ax.set_ylabel("Bradley-Terry rating")

    # (Title block intentionally omitted — figure caption in the paper
    # carries the label, keeping all figures typographically consistent.)

    fig.savefig(out_path, format="pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render probe-only BT rankings figure as a vector PDF.")
    ap.add_argument("--ratings", type=Path, default=ROOT / "bt_ratings.json")
    ap.add_argument("--out", type=Path, default=ROOT / "probe_bt.pdf")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    buckets = load(args.ratings)
    plot(buckets, args.out)
    print(f"wrote vector PDF: {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render the move-time-per-agent figure as a vector PDF.

Reads `results.csv` directly (no precomputation needed), computes each
trial's mean move time across all non-errored games, and plots one dot per
trial on a log-scale y-axis. Styling mirrors `all_bt.py` so the figure
sits consistently alongside the BT-ratings figure in the paper.

Per-trial "move time" is computed as the simple mean of this trial's
per-game average move times (the `avg_move_time_a` / `avg_move_time_b`
columns). Games with any error flag are excluded by default; pass
`--include-errored` to keep all 3280 games.

Usage:
    python3 move_time_bt.py
    python3 move_time_bt.py --results results.csv --out my_output.pdf
    python3 move_time_bt.py --results results.csv --include-errored
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Embed fonts as editable TrueType so PDFs stay selectable / editable.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["pdf.compression"] = 6
mpl.rcParams["svg.fonttype"] = "none"

ROOT = Path(__file__).resolve().parent

TITLE = "Average move time per agent setup"
SUBTITLE = "One dot per trial.  Mean of per-game averages."

# Same color scheme and 7-group order as all_bt.py.
FAMILY_SPEC = [
    ("opus47",     "Opus 4.7",                        "#2e4e7f"),
    ("opus",       "Opus 4.6",                        "#4c72b0"),
    ("codex",      "GPT-5.4",                         "#55a868"),
    ("gemini",     "Gemini 3.1 Pro",                  "#c44e52"),
    ("eval_d",     "GPT-5.4\neval\nDocker",           "#8172b2"),
    ("eval_nd",    "GPT-5.4\neval\nnon-Docker",       "#937860"),
    ("noneval_d",  "GPT-5.4\nnon-eval\nDocker",       "#da8bc3"),
    ("noneval_nd", "GPT-5.4\nnon-eval\nnon-Docker",   "#5f7a8a"),
]

# Pons as a reference baseline on the chart (calibration only).
PONS_COLOR = "#b8860b"
PONS_PLAYER = "pascal_pons_perfect"


def family_key_of(pid: str) -> str | None:
    if pid.startswith("opus47_"):    return "opus47"
    if pid.startswith("claude_"):    return "opus"
    if pid.startswith("codex_"):     return "codex"
    if pid.startswith("gemini_"):    return "gemini"
    if pid.startswith("eval_nd"):    return "eval_nd"
    if pid.startswith("eval_d"):     return "eval_d"
    if pid.startswith("noneval_nd"): return "noneval_nd"
    if pid.startswith("noneval_d"):  return "noneval_d"
    return None  # pons and unknowns handled separately


def load_per_trial_means(results_path: Path, include_errored: bool) -> tuple[dict[str, list[tuple[str, float]]], float]:
    """
    Walk results.csv and compute each individual trial's mean move time
    (simple mean of per-game averages across games that trial played).

    Returns (buckets, pons_mean) where:
      - buckets[family_key] = list of (trial_id, mean_move_time_s)
      - pons_mean is Pons's mean move time for the reference line
    """
    per_trial: dict[str, list[float]] = defaultdict(list)

    with open(results_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = {k.strip(): (v.strip() if v else "") for k, v in row.items()}
            if not include_errored and row.get("error", ""):
                continue
            try:
                t_a = float(row["avg_move_time_a"])
                t_b = float(row["avg_move_time_b"])
            except (KeyError, ValueError):
                continue
            per_trial[row["player_a"]].append(t_a)
            per_trial[row["player_b"]].append(t_b)

    # Bucket trials by family
    buckets: dict[str, list[tuple[str, float]]] = {k: [] for k, _, _ in FAMILY_SPEC}
    for trial_id, times in per_trial.items():
        if not times or trial_id == PONS_PLAYER:
            continue
        fk = family_key_of(trial_id)
        if fk is None:
            continue
        buckets[fk].append((trial_id, float(np.mean(times))))

    # Pons mean (single-agent reference)
    pons_times = per_trial.get(PONS_PLAYER, [])
    pons_mean = float(np.mean(pons_times)) if pons_times else float("nan")

    return buckets, pons_mean


def _darken(hex_color: str, amount: float) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = r * (1 - amount); g = g * (1 - amount); b = b * (1 - amount)
    return (r / 255, g / 255, b / 255)


def _beeswarm_offsets(values: np.ndarray, half_width: float, y_threshold: float) -> np.ndarray:
    """Beeswarm on linear-space distances to avoid overlap at matching y-values."""
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


def plot(buckets, pons_mean: float, out_path: Path) -> None:
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

    fig = plt.figure(figsize=(19, 9.2))
    ax = fig.add_axes([0.06, 0.18, 0.92, 0.76])

    labels = [lbl for _, lbl, _ in FAMILY_SPEC]
    colors = [c for _, _, c in FAMILY_SPEC]
    keys = [k for k, _, _ in FAMILY_SPEC]

    n = len(FAMILY_SPEC)
    xs = np.arange(n)

    HALF = 0.28
    # Threshold in linear y-space: two points within 0.2 seconds get offset.
    Y_THRESHOLD = 0.2

    for x, k, c in zip(xs, keys, colors):
        vals = np.array([v for _, v in buckets[k]])
        if len(vals) == 0:
            continue

        jx = x + _beeswarm_offsets(vals, half_width=HALF, y_threshold=Y_THRESHOLD)
        ax.scatter(
            jx, vals,
            s=340, color=c, edgecolor="white", linewidth=2.4, zorder=4,
            alpha=0.95,
        )

        # Median bar (robust to right-skew)
        med = float(np.median(vals))
        dark = _darken(c, 0.35)
        ax.hlines(med, x - HALF, x + HALF, colors=dark, linewidth=4.4,
                  zorder=5, capstyle="round")
        # Label on the right; format seconds with 2 decimals for <10s, 1 for ≥10s
        lbl = f"{med:.2f}s" if med < 10 else f"{med:.1f}s"
        ax.text(x + HALF + 0.05, med, lbl,
                va="center", ha="left", fontsize=19,
                color=dark, fontweight="700", zorder=5)

    # Pons reference line (calibration baseline — non-MCTS solver).
    if np.isfinite(pons_mean):
        ax.axhline(pons_mean, color=PONS_COLOR, linewidth=2.4,
                   linestyle=(0, (5, 3)), zorder=1)
        # Anchor on the LEFT now that the axis is linear — on log scale the
        # right-side anchor avoided colliding with tall Opus data, but on
        # linear scale Pons sits very low and would collide with the
        # rightmost group's median label ("0.82s").
        ax.text(-0.55, pons_mean + 0.2,
                f"Pons (table lookup) = {pons_mean:.2f}s",
                ha="left", va="bottom", color=PONS_COLOR,
                fontsize=17, fontweight="bold")

    # Linear y axis with seconds formatting.
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.set_yticklabels(["0s", "2s", "4s", "6s", "8s"])
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylim(-0.3, 9)
    ax.grid(axis="y", which="major", color="#eceff2", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)

    # X axis labels (three-line uniform height).
    def _pad_to_three_lines(lbl: str) -> str:
        lines = lbl.count("\n") + 1
        return lbl + "\n " * (3 - lines) if lines < 3 else lbl
    padded = [_pad_to_three_lines(lbl) for lbl in labels]
    ax.set_xticks(xs)
    ax.set_xticklabels(padded, fontsize=17, fontweight="bold")
    ax.set_xlim(-0.65, n - 0.35)
    ax.set_xlabel("")
    ax.set_ylabel("Mean move time (seconds)")

    # (Title block intentionally omitted — figure caption in the paper
    # carries the label, keeping all figures typographically consistent.)

    fig.savefig(out_path, format="pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render move-time-per-agent figure as a vector PDF.")
    ap.add_argument("--results", type=Path, default=ROOT / "results.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "move_time_bt.pdf")
    ap.add_argument("--include-errored", action="store_true",
                    help="Keep games with non-empty error field.")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    buckets, pons_mean = load_per_trial_means(args.results, include_errored=args.include_errored)
    plot(buckets, pons_mean, args.out)
    print(f"wrote vector PDF: {args.out}  (Pons mean = {pons_mean:.3f}s)")


if __name__ == "__main__":
    main()

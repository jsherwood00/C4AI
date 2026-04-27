#!/usr/bin/env python3
"""Render per-trial budget usage in four main-agent groups: Opus 4.7,
Opus 4.6, Gemini 3.1 Pro, and GPT-5.4 (main, codex). Y-axis is session
length in hours; dashed line marks the 3-hour budget cap.

Same dot+mean-bar styling as budget_3group.py — this is the four-main-
agent counterpart of that figure (whereas budget_3group.py shows the
GPT-5.4 main-vs-probe pooling).

Usage:
    python3 budget_4main.py
    python3 budget_4main.py --times player_times_seconds.txt --out out.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["pdf.compression"] = 6
mpl.rcParams["svg.fonttype"] = "none"

ROOT = Path(__file__).resolve().parent
BUDGET_SECONDS = 3 * 60 * 60

# Four main agents in newest-first order, matching the column convention
# used by main_bt.py / mean_bt_vs_release.py. Colors come from the same
# palette so the link to the BT figures is legible.
FAMILY_SPEC = [
    ("opus47", "Opus 4.7",       "#2e4e7f"),
    ("claude", "Opus 4.6",       "#4c72b0"),
    ("gemini", "Gemini 3.1 Pro", "#c44e52"),
    ("codex",  "GPT-5.4\n(main)", "#55a868"),
]


def family_key_of(pid: str) -> str | None:
    if pid.startswith("opus47_"):
        return "opus47"
    if pid.startswith("claude_"):
        return "claude"
    if pid.startswith("gemini_"):
        return "gemini"
    if pid.startswith("codex_"):
        return "codex"
    return None


def load(times_path: Path) -> dict[str, list[tuple[str, float]]]:
    buckets: dict[str, list[tuple[str, float]]] = {
        k: [] for k, _, _ in FAMILY_SPEC
    }
    with open(times_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, secs_str = line.split("\t")
            fk = family_key_of(name)
            if fk is None:
                continue
            secs = float(secs_str)
            hours = secs / 3600.0
            buckets[fk].append((name, hours))
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
        "axes.labelsize": 19,
    })

    # Slightly wider than budget_3group.pdf since 4 columns instead of 3;
    # same vertical proportions for visual consistency.
    fig = plt.figure(figsize=(11.5, 9.2))
    ax = fig.add_axes([0.13, 0.18, 0.84, 0.76])

    labels = [lbl for _, lbl, _ in FAMILY_SPEC]
    colors = [c for _, _, c in FAMILY_SPEC]
    keys = [k for k, _, _ in FAMILY_SPEC]

    n = len(FAMILY_SPEC)
    xs = np.arange(n)

    HALF = 0.28
    Y_THRESHOLD = 0.12

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

        m = float(np.mean(vals))
        dark = _darken(c, 0.35)
        ax.hlines(m, x - HALF, x + HALF, colors=dark, linewidth=4.4,
                  zorder=5, capstyle="round")
        ax.text(x + HALF + 0.05, m, f"{m:.2f}h",
                va="center", ha="left", fontsize=19,
                color=dark, fontweight="700", zorder=5)

    ax.axhline(3.0, color="#888", linewidth=1.8, linestyle=(0, (5, 3)), zorder=1)

    ax.set_ylim(-0.15, 3.3)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}h"))
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(axis="y", color="#eceff2", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)

    def _pad_to_three_lines(lbl: str) -> str:
        lines = lbl.count("\n") + 1
        return lbl + "\n " * (3 - lines) if lines < 3 else lbl
    padded = [_pad_to_three_lines(lbl) for lbl in labels]
    ax.set_xticks(xs)
    ax.set_xticklabels(padded, fontsize=17, fontweight="bold")
    ax.set_xlim(-0.65, n - 0.35)
    ax.set_xlabel("")
    ax.set_ylabel("Session length (hours)")

    fig.savefig(out_path, format="pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render 4-main-agent budget-usage figure as a vector PDF.")
    ap.add_argument("--times", type=Path, default=ROOT / "player_times_seconds.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "budget_4main.pdf")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    buckets = load(args.times)
    plot(buckets, args.out)
    print(f"wrote vector PDF: {args.out}")


if __name__ == "__main__":
    main()

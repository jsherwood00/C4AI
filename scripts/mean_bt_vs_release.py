#!/usr/bin/env python3
"""
mean_bt_vs_release.py — Scatter plot: agent release date vs. mean BT rating.

One dot per agent setup (not per trial). For each agent, the y-value is the
mean of that agent's per-trial BT ratings. Probe variants collapse into a
single "GPT-5.4 (probe)" entry whose mean is taken over all 16 probe trials
(regardless of eval/non-eval × Docker/non-Docker split), sharing the same
release date as main GPT-5.4.

Release dates (verified from primary sources):
  Gemini 3.1 Pro Preview:  Feb 19, 2026 (Google DeepMind model card)
  Opus 4.6:                Feb  4, 2026 (Anthropic announcement)
  GPT-5.4:                 Mar  5, 2026 (OpenAI announcement)
  Opus 4.7:                Apr 16, 2026 (Anthropic announcement)

Inputs:
    --ratings   Path to bt_ratings.json

Output:
    --out       Path to PDF / PNG (default: mean_bt_vs_release.pdf)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["pdf.compression"] = 6
mpl.rcParams["svg.fonttype"] = "none"

ROOT = Path(__file__).resolve().parent

# Agents, release dates, colors, and prefix patterns for identifying which
# trials belong to each entry. Same palette as the main_bt figure family
# (with GPT-5.4 probe pulled from probe_bt).
RELEASE_SPEC = [
    # (release_date, label, color, trial_prefixes)
    ("2026-02-04", "Opus 4.6",        "#4c72b0", ("claude_",)),
    ("2026-02-19", "Gemini 3.1 Pro",  "#c44e52", ("gemini_",)),
    ("2026-03-05", "GPT-5.4",         "#55a868", ("codex_",)),
    ("2026-03-05", "GPT-5.4 (probe)", "#8172b2",
        ("eval_d", "eval_nd", "noneval_d", "noneval_nd")),
    ("2026-04-16", "Opus 4.7",        "#2e4e7f", ("opus47_",)),
]

PONS_PLAYER = "pascal_pons_perfect"
PONS_RATING = 2000.0
PONS_COLOR = "#b8860b"


def _darken(hex_color: str, amount: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    r, g, b = (max(0, int(c * (1 - amount))) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def load_ratings(ratings_path: Path) -> dict[str, float]:
    data = json.loads(ratings_path.read_text())
    return {pid: float(r) for pid, r in data["ratings"].items()}


def trial_matches_prefixes(pid: str, prefixes: tuple[str, ...]) -> bool:
    """True if pid starts with any of the given prefixes, with careful
    precedence to avoid eval_d matching eval_nd trials."""
    # Check longer (more specific) prefixes first within the given tuple
    # so that e.g. "eval_nd" is matched before "eval_d".
    for p in sorted(prefixes, key=len, reverse=True):
        if pid.startswith(p):
            # When the requested prefix is "eval_d" / "noneval_d", we have
            # to guard against matching an eval_nd / noneval_nd trial.
            if p == "eval_d" and pid.startswith("eval_nd"):
                continue
            if p == "noneval_d" and pid.startswith("noneval_nd"):
                continue
            return True
    return False


def compute_means(ratings: dict[str, float]) -> list[tuple[dt.date, str, str, float, int]]:
    """Return one row per entry in RELEASE_SPEC, dropping any with no trials.
    Row format: (date, label, color, mean_rating, n_trials)."""
    rows = []
    for date_str, label, color, prefixes in RELEASE_SPEC:
        trials = [
            r for pid, r in ratings.items()
            if pid != PONS_PLAYER and trial_matches_prefixes(pid, prefixes)
        ]
        if not trials:
            continue
        rows.append((
            dt.date.fromisoformat(date_str),
            label,
            color,
            float(np.mean(trials)),
            len(trials),
        ))
    return rows


def make_plot(rows: list[tuple[dt.date, str, str, float, int]],
              out_path: Path, dpi: int = 200) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(11, 7))

    # Pons reference line.
    ax.axhline(PONS_RATING, color=PONS_COLOR, linewidth=2.4,
               linestyle=(0, (5, 3)), zorder=1)
    # Place Pons label on the left — early in the year, no agents released
    # yet, so there's guaranteed whitespace.
    ax.text(dt.date(2026, 1, 3), PONS_RATING + 22,
            "Pons = 2000",
            ha="left", va="bottom", color=PONS_COLOR,
            fontsize=14, fontweight="bold")

    # Scatter dots (one per agent setup).
    for date, label, color, mean_r, n in rows:
        ax.scatter([date], [mean_r],
                   s=220, color=color, edgecolors="white",
                   linewidths=1.8, zorder=3)

    # Label each dot with "Agent\nmean=XXX (N=8)". Default position: above
    # the dot. Same collision-avoidance logic as first_mover_optimal:
    # within a ~21-day window, prefer dropping labels below dots.
    # Additional rule: if a dot sits within 80 BT points of the Pons line,
    # place its label below to avoid collision with the Pons reference text.
    LABEL_X_COLLISION_DAYS = 21
    PONS_LABEL_CLEARANCE = 80
    for i, (date, label, color, mean_r, n) in enumerate(rows):
        # Default: above
        offset_x, offset_y = 0, 14
        ha, va = "center", "bottom"
        # Find same-date-ish neighbors with higher rating.
        higher_neighbors = [
            (d, m) for j, (d, _, _, m, _) in enumerate(rows)
            if j != i and abs((d - date).days) < LABEL_X_COLLISION_DAYS
            and (m > mean_r or (m == mean_r and i > j))
        ]
        close_to_pons = abs(mean_r - PONS_RATING) < PONS_LABEL_CLEARANCE
        if higher_neighbors or close_to_pons:
            # Drop this label below the dot.
            offset_y = -14
            va = "top"
        ax.annotate(
            f"{label}\nmean={mean_r:,.0f} (N={n})",
            xy=(date, mean_r),
            xytext=(offset_x, offset_y), textcoords="offset points",
            ha=ha, va=va,
            fontsize=12, color=_darken(color, 0.25), fontweight="bold",
        )

    # Axes
    all_means = [m for _, _, _, m, _ in rows]
    y_min = min(min(all_means), 400) - 100
    y_max = max(PONS_RATING + 80, max(all_means) + 100)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Mean Bradley-Terry rating across trials", fontsize=14)
    ax.tick_params(axis="y", labelsize=13)

    # X axis: dates. Pad bracketing months like first_mover_optimal.
    ax.set_xlim(dt.date(2026, 1, 1), dt.date(2026, 5, 31))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlabel("Agent release date", fontsize=14)
    ax.tick_params(axis="x", labelsize=12)

    ax.grid(axis="y", color="#eceff2", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    if out_path.suffix.lower() == ".png":
        fig.savefig(out_path, dpi=dpi, facecolor="white")
    else:
        fig.savefig(out_path, facecolor="white")
    print(f"wrote: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--ratings", type=Path, default=ROOT / "bt_ratings.json")
    p.add_argument("--out", type=Path, default=ROOT / "mean_bt_vs_release.pdf")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ratings = load_ratings(args.ratings)
    rows = compute_means(ratings)
    make_plot(rows, args.out, dpi=args.dpi)


if __name__ == "__main__":
    main()

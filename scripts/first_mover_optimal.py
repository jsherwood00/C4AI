#!/usr/bin/env python3
"""
first_mover_optimal.py — Per-agent bar chart and release-date scatter:
                          fraction of trials that won or drew as first
                          player against Pons.

Connect Four is a first-player win with optimal play, so any first-mover
trajectory against Pons that does not result in a loss is consistent with
optimal play. A trial counts as a success if at least one of its two
first-mover games against Pons ended in a win or draw. (In our data,
every trial's two first-mover games agreed, so this disjunction is
equivalent to a unanimity condition.)

Two plot modes:
  --plot bars       Bar chart with one bar per agent setup (8 bars).
                    Probe variants shown as four separate bars.
  --plot release    Scatter against agent release date. Probe variants
                    collapsed into one "GPT-5.4 (probe)" entry.

Inputs:
    --results   Path to results.csv

Output:
    --out       Path to PDF / PNG (default: first_mover_optimal_<plot>.pdf)
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Family spec — same palette and ordering as all_bt.py.
# ---------------------------------------------------------------------------

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

# For the release-date scatter, probe variants collapse into a single
# "GPT-5.4 (probe)" entry sharing the same release date as the main GPT-5.4.
# All dates verified from primary sources (vendor announcements, model cards):
#   Gemini 3.1 Pro Preview:  Feb 19, 2026 (Google DeepMind model card)
#   Opus 4.6:                Feb  4, 2026 (Anthropic announcement)
#   GPT-5.4:                 Mar  5, 2026 (OpenAI announcement)
#   Opus 4.7:                Apr 16, 2026 (Anthropic announcement)
RELEASE_SPEC = [
    # (release_date, label, color, family_keys_for_aggregation)
    ("2026-02-19", "Gemini 3.1 Pro",   "#c44e52", ("gemini",)),
    ("2026-02-04", "Opus 4.6",         "#4c72b0", ("opus",)),
    ("2026-03-05", "GPT-5.4",          "#55a868", ("codex",)),
    ("2026-03-05", "GPT-5.4 (probe)",  "#8172b2",
        ("eval_d", "eval_nd", "noneval_d", "noneval_nd")),
    ("2026-04-16", "Opus 4.7",         "#2e4e7f", ("opus47",)),
]

PONS_PLAYER = "pascal_pons_perfect"
PONS_COLOR = "#b8860b"


def family_key(pid: str) -> str | None:
    if pid == PONS_PLAYER:           return None
    if pid.startswith("opus47_"):    return "opus47"
    if pid.startswith("claude_"):    return "opus"
    if pid.startswith("codex_"):     return "codex"
    if pid.startswith("gemini_"):    return "gemini"
    if pid.startswith("eval_nd"):    return "eval_nd"
    if pid.startswith("eval_d"):     return "eval_d"
    if pid.startswith("noneval_nd"): return "noneval_nd"
    if pid.startswith("noneval_d"):  return "noneval_d"
    return None


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def collect_first_mover_outcomes(results_path: Path) -> dict[str, list[str]]:
    """For each non-Pons trial, return its list of first-mover outcomes
    against Pons as 'win' / 'draw' / 'loss' (one entry per game)."""
    outcomes: dict[str, list[str]] = defaultdict(list)
    pons = PONS_PLAYER
    with open(results_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = {k.strip(): (v.strip() if v else "") for k, v in row.items()}
            a, b = row["player_a"], row["player_b"]
            # Pick out trial-vs-Pons rows
            if a == pons and b != pons:
                trial = b
                trial_is_a = False
            elif b == pons and a != pons:
                trial = a
                trial_is_a = True
            else:
                continue
            # Only count games where the trial moved first
            mover = a if trial_is_a else b
            if row["first_mover"] != mover:
                continue
            # Translate result into trial's perspective
            res = row["result"]
            if res == "draw":
                outcomes[trial].append("draw")
            elif res == ("player_a_win" if trial_is_a else "player_b_win"):
                outcomes[trial].append("win")
            else:
                outcomes[trial].append("loss")
    return outcomes


def aggregate_per_family(
    outcomes: dict[str, list[str]],
    success_types: tuple[str, ...] = ("win", "draw"),
) -> dict[str, tuple[int, int]]:
    """Per family key, return (n_trials, n_succeeded).

    `success_types` controls what counts as a successful trial:
      - ("win", "draw") — any non-loss first-mover game counts (default).
        Motivated by Connect Four being a first-player win under optimal
        play, so a non-loss is consistent with optimal play.
      - ("win",)        — only outright wins count.
    """
    success_set = set(success_types)
    counts: dict[str, list[int]] = {k: [0, 0] for k, _, _ in FAMILY_SPEC}
    for trial, outs in outcomes.items():
        fk = family_key(trial)
        if fk is None or fk not in counts:
            continue
        counts[fk][0] += 1
        if any(o in success_set for o in outs):
            counts[fk][1] += 1
    return {k: tuple(v) for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _darken(hex_color: str, factor: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    r, g, b = (max(0, int(c * (1 - factor))) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def make_plot(per_family: dict[str, tuple[int, int]],
              out_path: Path,
              ylabel: str,
              dpi: int = 200) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(13, 7))

    n = len(FAMILY_SPEC)
    x_positions = np.arange(n)
    pcts = []
    counts = []
    colors = []
    labels = []
    for i, (fk, lbl, color) in enumerate(FAMILY_SPEC):
        n_trials, n_succ = per_family.get(fk, (0, 0))
        pct = 100.0 * n_succ / n_trials if n_trials else 0.0
        pcts.append(pct)
        counts.append((n_succ, n_trials))
        colors.append(color)
        labels.append(lbl)

    # Bars
    bars = ax.bar(x_positions, pcts, color=colors, edgecolor="white",
                  linewidth=1.5, width=0.7, zorder=3)

    # Bar labels: "X / N" centered above each bar
    for bar, (n_succ, n_tot), pct, color in zip(bars, counts, pcts, colors):
        x = bar.get_x() + bar.get_width() / 2
        # If the bar is non-zero, label sits just above the bar tip;
        # if it's zero, label sits at the baseline so it stays visible.
        y = pct + 3 if pct > 5 else 3
        ax.text(x, y, f"{n_succ} / {n_tot}",
                ha="center", va="bottom",
                color=_darken(color, 0.25),
                fontsize=14, fontweight="bold")

    # Axes
    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 25))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="y", labelsize=13)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=13)
    ax.tick_params(axis="x", length=0, pad=10)

    ax.grid(axis="y", color="#eceff2", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    if out_path.suffix.lower() == ".png":
        fig.savefig(out_path, dpi=dpi, facecolor="white")
    else:
        fig.savefig(out_path, facecolor="white")
    print(f"wrote: {out_path}")


def make_release_plot(per_family: dict[str, tuple[int, int]],
                      out_path: Path,
                      ylabel: str,
                      dpi: int = 200) -> None:
    """Scatter against agent release date (one dot per labeled entry)."""
    import datetime as dt
    import matplotlib.dates as mdates

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(11, 7))

    points = []  # list of (date, pct, label, color, n_succ, n_tot)
    for date_str, label, color, family_keys in RELEASE_SPEC:
        n_trials = sum(per_family.get(fk, (0, 0))[0] for fk in family_keys)
        n_succ = sum(per_family.get(fk, (0, 0))[1] for fk in family_keys)
        pct = 100.0 * n_succ / n_trials if n_trials else 0.0
        date = dt.date.fromisoformat(date_str)
        points.append((date, pct, label, color, n_succ, n_trials))

    # Scatter dots
    for date, pct, label, color, _, _ in points:
        ax.scatter([date], [pct],
                   s=180, color=color, edgecolors="white",
                   linewidths=1.5, zorder=3, label=label)

    # Per-dot labels: "Agent\nX / N", positioned to avoid the dot.
    # When two dots are close enough on the X axis that their labels
    # would overlap, push the lower-pct one's label below the dot — but
    # never push a label below a dot at 0%, since that would force the
    # Y axis to extend below zero (which would visually misrepresent the
    # value as being slightly above zero rather than at zero). In that
    # special case, shift the label horizontally to the side instead.
    #
    # Special sub-case: two dots at 0% can't stack (neither can go down),
    # so both get shifted horizontally away from each other.
    LABEL_X_COLLISION_DAYS = 21
    for i, (date, pct, label, color, n_succ, n_tot) in enumerate(points):
        # Default: label above the dot
        offset_x = 0
        offset_y = 12
        ha = "center"
        va = "bottom"
        # Higher-pct neighbors (these cause us to drop below or shift sideways).
        higher_neighbors = [
            (d, p) for j, (d, p, *_) in enumerate(points)
            if j != i and abs((d - date).days) < LABEL_X_COLLISION_DAYS
            and (p > pct or (p == pct and i > j))
        ]
        # Same-pct zero neighbors (neither can drop below 0, so both shift
        # horizontally — but each one has to pick a different side).
        same_zero_neighbors = [
            (j, d) for j, (d, p, *_) in enumerate(points)
            if j != i and abs((d - date).days) < LABEL_X_COLLISION_DAYS
            and pct == 0 and p == 0
        ]
        if higher_neighbors:
            if pct > 0:
                # Normal case: drop label below the dot
                offset_y = -12
                va = "top"
            else:
                # Zero-pct dot with higher neighbor: shift horizontally.
                # Include both higher-pct neighbors AND same-pct (also zero)
                # neighbors when computing the average neighbor direction —
                # a zero-pct neighbor would otherwise get ignored even though
                # its label will overlap this one's.
                neighbor_dates = [d for d, _ in higher_neighbors]
                for _j, d in same_zero_neighbors:
                    neighbor_dates.append(d)
                avg_nbr_date = sum(
                    (d - date).days for d in neighbor_dates
                ) / len(neighbor_dates)
                if avg_nbr_date > 0:
                    offset_x = -16
                    ha = "right"
                elif avg_nbr_date < 0:
                    offset_x = 16
                    ha = "left"
                else:
                    # Neighbors balanced on both sides; default to right.
                    offset_x = 16
                    ha = "left"
        elif same_zero_neighbors:
            # Both at 0% and near each other. Shift each away from the
            # other — the earlier-in-time dot's label goes left, the
            # later-in-time dot's label goes right.
            neighbor_dates = [d for _, d in same_zero_neighbors]
            avg_nbr_date = sum(
                (d - date).days for d in neighbor_dates
            ) / len(neighbor_dates)
            if avg_nbr_date > 0:
                # Neighbors are to the right; this dot's label goes left.
                offset_x = -16
                ha = "right"
            else:
                offset_x = 16
                ha = "left"
        ax.annotate(
            f"{label}\n{n_succ} / {n_tot}",
            xy=(date, pct),
            xytext=(offset_x, offset_y), textcoords="offset points",
            ha=ha, va=va,
            fontsize=12, color=_darken(color, 0.25), fontweight="bold",
        )

    # Y axis: same scale as the bar chart for direct comparability.
    # 0% sits exactly at the X axis spine.
    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 25))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="y", labelsize=13)

    # X axis: dates. Pad to show Jan 2026 and May 2026 as bracketing ticks
    # even though no data sits in those months.
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
    p.add_argument("--results", type=Path, default=Path("results.csv"))
    p.add_argument("--plot", choices=["bars", "release"], default="bars",
                   help="Which plot to produce.")
    p.add_argument("--success-type", choices=["wins-or-draws", "wins-only"],
                   default="wins-or-draws",
                   help="What counts as a successful first-mover trial: "
                        "either any non-loss (wins + draws) or only outright wins.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output path. Defaults to "
                        "first_mover_optimal_<plot>[_wins_only].pdf")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outcomes = collect_first_mover_outcomes(args.results)

    if args.success_type == "wins-only":
        success_types = ("win",)
        ylabel = "Trials winning as first-mover against the solver"
        name_suffix = "_wins_only"
    else:
        success_types = ("win", "draw")
        ylabel = "Trials winning or drawing as first player against the solver"
        name_suffix = ""

    per_family = aggregate_per_family(outcomes, success_types=success_types)
    out = args.out or Path(f"first_mover_optimal_{args.plot}{name_suffix}.pdf")
    if args.plot == "bars":
        make_plot(per_family, out, ylabel=ylabel, dpi=args.dpi)
    else:
        make_release_plot(per_family, out, ylabel=ylabel, dpi=args.dpi)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
best_bt_lollipop.py — Horizontal lollipop version of the best-trial plot.

Structurally 1D (one axis carries the value; the other is categorical),
but styled like the bar chart: clean colored fills, bold x-axis labels,
"(out of N)" caption under each agent name. Each agent gets a thin
colored stem from the log-scale lower bound out to its odds value,
capped by a prominent dot.

Compared to the bar chart: dots draw the eye directly to each value
without a heavy rectangle dominating the plot area. Compared to the
pure number-line version: keeps per-agent rows so no labels need
to float or alternate left/right.

Usage:
    python3 best_bt_lollipop.py --ratings bt_ratings.json --out best_bt_lollipop.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["pdf.compression"] = 6
mpl.rcParams["svg.fonttype"] = "none"

ROOT = Path(__file__).resolve().parent

TITLE = "Best-trial BT win-probability vs. Pons"
SUBTITLE = ""

ANCHOR_PLAYER = "pascal_pons_perfect"
ANCHOR_RATING = 2000.0
SCALE_PER_DECADE = 400.0

# Pons reference color — matches the dark goldenrod used by the rest of
# the figure family (all_bt, main_bt, move_time_bt, training_time_bt) so
# the cross-figure visual language is consistent and avoids any clash with
# the blue used for Opus 4.6 / Opus 4.7.
PONS_COLOR = "#b8860b"


def get_agent(p: str) -> str | None:
    if p.startswith("claude_"):       return "Opus 4.6"
    if p.startswith("codex_"):        return "GPT-5.4"
    if p.startswith("gemini_"):       return "Gemini 3.1 Pro"
    if p.startswith("opus47_"):       return "Opus 4.7"
    if p == ANCHOR_PLAYER:            return None
    if p.startswith("eval_") or p.startswith("noneval_"):
        return "GPT-5.4 (probe)"
    return p


COLOR_MAP = {
    "Opus 4.6":         "#4c72b0",
    "Opus 4.7":         "#2e4e7f",
    "GPT-5.4":          "#55a868",
    "GPT-5.4 (probe)":  "#8172b2",
    "Gemini 3.1 Pro":   "#c44e52",
}


def win_ratio_vs_pons(rating: float) -> float:
    """Win odds against Pons: ratio of expected wins to losses in an
    infinite match. 1.00 = parity with Pons."""
    return 10 ** ((rating - ANCHOR_RATING) / SCALE_PER_DECADE)


def win_probability_vs_pons(rating: float) -> float:
    """Win probability against Pons: fraction of a long match that the
    agent wins. 0.50 = parity with Pons. This is the native BT quantity
    (BT ratings are fit to predict pairwise win probabilities) and is
    bounded [0, 1], which is easier to read than unbounded odds."""
    odds = win_ratio_vs_pons(rating)
    return odds / (1.0 + odds)


def _darken(hex_color: str, amount: float) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = r * (1 - amount); g = g * (1 - amount); b = b * (1 - amount)
    return (r / 255, g / 255, b / 255)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=Path("bt_ratings.json"))
    parser.add_argument("--out", type=Path, default=Path("best_bt_lollipop.pdf"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--also-png", action="store_true")
    args = parser.parse_args()

    with open(args.ratings) as f:
        data = json.load(f)
    ratings: dict[str, float] = data["ratings"]

    best: dict[str, tuple[str, float]] = {}
    n_trials: dict[str, int] = {}
    for p, r in ratings.items():
        a = get_agent(p)
        if a is None:
            continue
        n_trials[a] = n_trials.get(a, 0) + 1
        if a not in best or r > best[a][1]:
            best[a] = (p, r)

    # Best at top (highest y-index in matplotlib's default orientation)
    agent_order = sorted(best.keys(), key=lambda a: best[a][1])
    n = len(agent_order)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444",
        "axes.labelcolor": "#222",
        "xtick.color": "#222",
        "ytick.color": "#333",
    })

    fig = plt.figure(figsize=(13, 6.5))
    ax = fig.add_axes([0.18, 0.17, 0.76, 0.78])

    # Dotted Pons parity line at p=0.50 — the primary reference under
    # the win-probability parametrization. Colored to match the rest of
    # the figure family and to stay clearly distinct from the blues
    # used for Opus 4.6 and Opus 4.7.
    ax.axvline(0.5, color=PONS_COLOR, linestyle=":", linewidth=2.0, zorder=1)

    # Lower bound of each stem: 0 (linear scale — no log-window lower bound).
    stem_left = 0.0

    for i, a in enumerate(agent_order):
        _player, rating = best[a]
        x = win_probability_vs_pons(rating)
        c = COLOR_MAP.get(a, "#444444")
        dark = _darken(c, 0.35)

        # The stem: horizontal line from 0 out to the value
        ax.hlines(i, stem_left, x, colors=c, linewidth=4.5, alpha=0.85,
                  zorder=2)

        # The lollipop head: big colored dot at the value
        ax.scatter(x, i, s=700, color=c, alpha=0.95,
                   edgecolor="white", linewidth=3.0, zorder=3)

    # Linear x-axis bounded by probability range. Upper limit is always 1
    # (perfect win rate against Pons); we pad slightly beyond to give the
    # "Pons = 0.50" floating label some breathing room on the right of
    # dots near p=1.
    agent_values = [win_probability_vs_pons(best[a][1]) for a in agent_order]
    x_max = 1.05
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.6, n - 0.4)

    # Y axis: agent names with "(out of N)" underneath, bold
    ax.set_yticks(range(n))
    ax.set_yticklabels(agent_order, fontsize=17, fontweight="bold")
    ax.tick_params(axis="y", pad=8, length=0)

    for i, a in enumerate(agent_order):
        ax.annotate(
            f"(out of {n_trials[a]})",
            xy=(0, i), xycoords=("axes fraction", "data"),
            xytext=(-8, -16), textcoords="offset points",
            ha="right", va="center",
            fontsize=13, color="#666",
        )

    # X axis: one tick per agent value. Linear scale makes label placement
    # much simpler than the log-scale case — collisions only happen when
    # two values fall within a small fraction of the x-range of each other.
    value_colors = sorted(
        [(win_probability_vs_pons(best[a][1]), COLOR_MAP.get(a, "#444444"))
         for a in agent_order],
        key=lambda t: t[0],
    )

    # Tick label formatter: 2 decimals for most values, 3 decimals for
    # anything < 0.05 so tiny values like Gemini's ~0.014 stay legible
    # (otherwise they'd collapse to "0.01" and look indistinguishable
    # from any other near-zero value).
    def _fmt(v: float) -> str:
        return f"{v:.3f}" if v < 0.05 else f"{v:.2f}"

    agent_tick_positions = [v for v, _ in value_colors]
    agent_tick_labels = [_fmt(v) for v, _ in value_colors]

    # Collision threshold expressed as a fraction of the x-range: labels
    # closer than this need to be displaced to avoid overlap.
    COLLISION_FRACTION = 0.04  # 4% of x-range
    collision_width = COLLISION_FRACTION * x_max

    candidates: list[tuple[float, str, str]] = []  # (position, label, role)
    for pos, lbl in zip(agent_tick_positions, agent_tick_labels):
        candidates.append((pos, lbl, "agent"))
    candidates.sort(key=lambda t: t[0])

    # Cluster nearby ticks (linear distance within collision_width). Within
    # each cluster, one tick keeps the standard axis slot — highest value
    # wins. The rest get displaced above the spine into the chart area.
    clusters: list[list[tuple[float, str, str]]] = []
    for tick in candidates:
        if not clusters:
            clusters.append([tick])
            continue
        prev_pos = clusters[-1][-1][0]
        if abs(tick[0] - prev_pos) < collision_width:
            clusters[-1].append(tick)
        else:
            clusters.append([tick])

    placements: list[tuple[float, str, str, str]] = []  # (pos, lbl, role, slot)
    for cluster in clusters:
        # Highest value wins the axis slot. The rest get displaced.
        winner = max(cluster, key=lambda t: t[0])
        for tick in cluster:
            slot = "axis" if tick is winner else "displaced"
            placements.append((*tick, slot))

    # The "axis" set drives matplotlib's normal x-axis ticks (sitting just
    # below the axis spine, in the standard tick label slot). Anything that
    # would collide gets pushed into the "displaced" list and rendered
    # above the spine, inside the chart area, by a separate annotation pass.
    axis_ticks = [(p, l, r) for p, l, r, slot in placements if slot == "axis"]
    displaced = [(p, l, r) for p, l, r, slot in placements if slot == "displaced"]

    axis_positions = [p for p, _, _ in axis_ticks]
    axis_labels = [l for _, l, _ in axis_ticks]
    axis_roles = [r for _, _, r in axis_ticks]

    ax.set_xticks(axis_positions)
    ax.set_xticklabels(axis_labels, fontsize=14)
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.tick_params(axis="x", pad=8)

    # Color agent tick labels to match their dots.
    color_by_value = {v: _darken(c, 0.35) for v, c in value_colors}
    for ticklabel, pos, role in zip(ax.get_xticklabels(), axis_positions, axis_roles):
        if role == "agent":
            ticklabel.set_color(color_by_value.get(pos, "#222"))
            ticklabel.set_fontweight("bold")
        else:
            ticklabel.set_color("#888")

    # Displaced tick set: positions whose label couldn't sit on the axis
    # without colliding with a neighbor. Render these *inside the chart*,
    # above the axis spine, with a short tick mark dropping down to the
    # spine. Uses display-coordinate offsets (in points) so spacing stays
    # consistent regardless of the x scale.
    if displaced:
        for pos, lbl, role in displaced:
            color = (
                color_by_value.get(pos, "#222") if role == "agent" else "#888"
            )
            weight = "bold" if role == "agent" else "normal"
            # Short tick mark rising above the spine into the chart area
            ax.annotate(
                "",
                xy=(pos, 0), xycoords=("data", "axes fraction"),
                xytext=(0, 22), textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color=color, linewidth=1.6),
                annotation_clip=False, zorder=3,
            )
            # Label above the tick (still inside the chart area)
            ax.annotate(
                lbl,
                xy=(pos, 0), xycoords=("data", "axes fraction"),
                xytext=(0, 36), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=14, color=color, fontweight=weight,
                annotation_clip=False, zorder=3,
            )

    # Pons reference label floating above the dotted line, colored to match.
    ax.annotate("Pons = 0.50",
                xy=(0.5, n - 0.5), xytext=(6, 0),
                textcoords="offset points",
                ha="left", va="center",
                fontsize=14, color=PONS_COLOR,
                fontweight="bold", style="italic")

    # (Title block intentionally omitted — figure caption in the paper
    # carries the label, keeping all figures typographically consistent.)

    if args.out.suffix.lower() == ".png":
        fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    else:
        fig.savefig(args.out, format="pdf", facecolor="white")
    print(f"wrote: {args.out}")

    if args.also_png and args.out.suffix.lower() != ".png":
        png_path = args.out.with_suffix(".png")
        fig.savefig(png_path, dpi=args.dpi, facecolor="white")
        print(f"also wrote: {png_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()

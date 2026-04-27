#!/usr/bin/env python3
"""
bt.py — Bradley-Terry MLE ratings for AZ-Bench Connect Four tournament.

By default, every row in results.csv is treated as an independent observation
and all games go into the BT fit as-is.

Optionally (--dedup, with --move-sequences), replicate games can be collapsed
on the basis of full move-sequence identity: for each matchup-orientation
(player_a, player_b) with multiple replicate games, if two or more games have
identical (result, move_sequence), only one representative is kept in the BT
fit. Games that differ from their replicate partners in either result or move
sequence are retained as independent observations.

The dedup rationale: a deterministic replicate is a forced echo of its first
game and contains no additional independent information. Counting it twice in
the BT likelihood inflates Fisher information without a corresponding gain in
true information (pseudoreplication; Hurlbert, 1984). Genuine stochastic
replicates — where agent randomness produced different trajectories — carry
independent Bernoulli information and are retained either way.

Rating scale:
    Ratings are displayed on an Elo-like scale anchored on the Pascal Pons
    solver at 2000, with a 400-point spread corresponding to a 10x
    win-probability ratio. A rating R means the player's gamma satisfies
    gamma / gamma_pons = 10^((R - 2000) / 400). The Pons anchor makes
    ratings interpretable as distance from game-theoretic optimality: closer
    to 2000 means closer to optimal play.

Inputs:
    --results          Path to results.csv (tournament evaluator output)
    --move-sequences   Optional path to move_sequences.csv. Required only with
                       --dedup.
    --dedup            Opt in to move-sequence-based deduplication.

Outputs:
    - Leaderboard printed to stdout (Bradley-Terry ratings sorted descending)
    - Agent-family summary (mean, std, min, max rating per agent family)
    - Record vs. solver detail
    - (Optional) ratings JSON via --output
    - (Optional) deduplication report via --report (no-op without --dedup)

Usage:
    # Default: no dedup, every row counted
    python3 bt.py --results results.csv

    # Opt in to dedup
    python3 bt.py --results results.csv --move-sequences move_sequences.csv \\
        --dedup --output ratings.json --report dedup_report.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ANCHOR_PLAYER = "pascal_pons_perfect"
ANCHOR_RATING = 2000.0
SCALE_PER_DECADE = 400.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: Path) -> list[dict]:
    """Load results.csv, stripping whitespace from all fields."""
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): (v.strip() if v is not None else "") for k, v in row.items()})
    return rows


def load_move_sequences(path: Path) -> dict[str, str]:
    """
    Load move_sequences.csv produced by extract_move_sequences.sh.

    Returns a dict mapping game_id -> full move sequence string. Games missing
    from the file map to an empty sequence (treated as unknown; see dedup rule).
    """
    seqs = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gid = row["game_id"].strip()
            hist = row["history"].strip()
            seqs[gid] = hist
    return seqs


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_rows(
    rows: list[dict],
    move_sequences: dict[str, str],
) -> tuple[list[dict], dict]:
    """
    Collapse matchup-orientation replicates whose (result, move_sequence) match.

    Rule:
      - Group all rows by (player_a, player_b) — the matchup-orientation.
      - Within each group, further subgroup by (result, move_sequence).
      - For each such subgroup, keep exactly one representative row; discard
        the rest. If multiple distinct (result, move_sequence) subgroups exist
        within a matchup-orientation, all are retained — each representing a
        genuinely distinct trajectory.

    Missing move sequences (empty strings) are treated as distinct from every
    other sequence, so rows without a recorded sequence are never collapsed
    with one another. In practice every game in our tournament has a recorded
    sequence; this guard is defensive.

    Returns:
        (kept_rows, report)
    """
    by_orient: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["player_a"], r["player_b"])
        by_orient[key].append(r)

    kept: list[dict] = []
    n_identical_orientations = 0
    n_differ_orientations = 0

    for _orient_key, group in by_orient.items():
        by_trajectory: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in group:
            seq = move_sequences.get(r["game_id"], "")
            traj_key = (r["result"], seq)
            by_trajectory[traj_key].append(r)

        for _traj_key, traj_rows in by_trajectory.items():
            kept.append(traj_rows[0])

        if len(by_trajectory) == 1:
            n_identical_orientations += 1
        else:
            n_differ_orientations += 1

    report = {
        "total_rows_in": len(rows),
        "total_rows_kept": len(kept),
        "total_rows_dropped": len(rows) - len(kept),
        "orientations": len(by_orient),
        "orientations_identical": n_identical_orientations,
        "orientations_differ": n_differ_orientations,
    }
    return kept, report


# ---------------------------------------------------------------------------
# Bradley-Terry MLE via the MM algorithm
# ---------------------------------------------------------------------------

def extract_outcomes(rows: list[dict]) -> list[tuple[str, str, float]]:
    """
    Convert game rows into (winner, loser, score) triples.

    A decisive game contributes one triple with score=1.0. A draw contributes
    two triples with score=0.5 each (one for each side's half-win), matching
    the convention used by Chatbot Arena (Chiang et al., 2024).
    """
    outcomes: list[tuple[str, str, float]] = []
    for row in rows:
        a = row["player_a"]
        b = row["player_b"]
        result = row["result"]
        if result == "player_a_win":
            outcomes.append((a, b, 1.0))
        elif result == "player_b_win":
            outcomes.append((b, a, 1.0))
        elif result == "draw":
            outcomes.append((a, b, 0.5))
            outcomes.append((b, a, 0.5))
    return outcomes


def bt_mle(
    outcomes: list[tuple[str, str, float]],
    max_iter: int = 1000,
    tol: float = 1e-8,
    anchor_player: str = ANCHOR_PLAYER,
    anchor_rating: float = ANCHOR_RATING,
    scale_per_decade: float = SCALE_PER_DECADE,
) -> dict[str, float]:
    """
    Compute Bradley-Terry MLE ratings via the MM algorithm of Hunter (2004),
    anchored on the Pascal Pons solver.

    The MM update is scale-invariant — multiplying all gammas by a constant
    does not change any win probabilities — so any anchor is mathematically
    equivalent. We anchor on the solver at 2000 so that every
    player's rating is directly interpretable as distance from optimal play:
    R_i = 2000 + 400 * log10(gamma_i / gamma_pons).

    If anchor_player is absent from the outcomes, falls back to anchoring the
    geometric mean of gammas at 1000 (the Elo-default convention) and prints
    a warning to stderr.
    """
    players = sorted({p for o in outcomes for p in (o[0], o[1])})
    n = len(players)
    idx = {p: i for i, p in enumerate(players)}

    w = np.zeros(n, dtype=np.float64)
    games = np.zeros((n, n), dtype=np.float64)

    for winner, loser, score in outcomes:
        i, j = idx[winner], idx[loser]
        w[i] += score
        games[i][j] += score
        games[j][i] += (1.0 - score)

    p = np.ones(n, dtype=np.float64)

    for _ in range(max_iter):
        p_old = p.copy()
        for i in range(n):
            if w[i] == 0:
                p[i] = 1e-10
                continue
            denom = 0.0
            for j in range(n):
                if i == j:
                    continue
                n_ij = games[i][j] + games[j][i]
                if n_ij > 0:
                    denom += n_ij / (p[i] + p[j])
            if denom > 0:
                p[i] = w[i] / denom
        # Intermediate normalization keeps values numerically stable during
        # iteration; the final display scale is set below.
        p /= np.exp(np.mean(np.log(np.maximum(p, 1e-30))))
        if np.max(np.abs(np.log(p) - np.log(p_old))) < tol:
            break

    # Anchor on Pons at the specified rating.
    if anchor_player in idx:
        anchor_gamma = max(p[idx[anchor_player]], 1e-30)
        log_ratings = np.log10(np.maximum(p, 1e-30) / anchor_gamma)
        ratings = anchor_rating + scale_per_decade * log_ratings
    else:
        print(
            f"WARNING: anchor player '{anchor_player}' not in outcomes; "
            "falling back to geometric-mean anchor at 1000.",
            file=sys.stderr,
        )
        log_ratings = np.log10(np.maximum(p, 1e-30))
        ratings = 1000.0 + scale_per_decade * log_ratings

    return {players[i]: float(ratings[i]) for i in range(n)}


def bootstrap_ratings(
    outcomes: list[tuple[str, str, float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
    anchor_player: str = ANCHOR_PLAYER,
    anchor_rating: float = ANCHOR_RATING,
    scale_per_decade: float = SCALE_PER_DECADE,
    show_progress: bool = True,
) -> dict[str, tuple[float, float]]:
    """
    Compute 95% bootstrap confidence intervals for BT ratings.

    Resamples individual outcome triples with replacement and refits BT-MLE on
    each resample. Returns the 2.5th and 97.5th percentiles of the bootstrap
    distribution for each player (the percentile method).

    Note: the input `outcomes` should already be the deduplicated set (so that
    deterministic replicates are not counted as independent observations). This
    bootstrap measures the stability of the BT fit over the set of independent
    games observed; it does not correct for the round-robin design being
    complete by construction, and is therefore best interpreted as a rating-
    stability diagnostic rather than a true sampling-distribution estimate.
    """
    import time
    rng = np.random.default_rng(seed)
    all_players = sorted({p for o in outcomes for p in (o[0], o[1])})
    rating_samples: dict[str, list[float]] = {p: [] for p in all_players}

    n = len(outcomes)
    start = time.time()
    bar_width = 30
    for b in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        resampled = [outcomes[i] for i in indices]
        ratings = bt_mle(
            resampled,
            anchor_player=anchor_player,
            anchor_rating=anchor_rating,
            scale_per_decade=scale_per_decade,
        )
        for p in all_players:
            rating_samples[p].append(ratings.get(p, anchor_rating))

        if show_progress:
            done = b + 1
            frac = done / n_bootstrap
            filled = int(bar_width * frac)
            bar = "#" * filled + "-" * (bar_width - filled)
            elapsed = time.time() - start
            if done > 0 and frac > 0:
                eta = elapsed * (1 - frac) / frac
                eta_str = f"ETA {eta:4.0f}s"
            else:
                eta_str = "ETA  ?"
            # \r returns to start of line; end='' to overwrite
            print(
                f"\r  [{bar}] {done:>5}/{n_bootstrap} ({frac:5.1%})  "
                f"elapsed {elapsed:4.0f}s  {eta_str}",
                end="",
                flush=True,
            )

    if show_progress:
        print()  # newline after progress bar completes

    ci: dict[str, tuple[float, float]] = {}
    for p in all_players:
        samples = sorted(rating_samples[p])
        lower = samples[int(0.025 * len(samples))]
        upper = samples[int(0.975 * len(samples))]
        ci[p] = (lower, upper)
    return ci


# ---------------------------------------------------------------------------
# Record vs. solver
# ---------------------------------------------------------------------------

def compute_vs_perfect(
    rows: list[dict],
    solver_name: str = ANCHOR_PLAYER,
) -> dict[str, dict]:
    """Compute each player's W/L/D record against the solver."""
    stats: dict[str, dict] = {}
    all_players = sorted({row["player_a"] for row in rows} | {row["player_b"] for row in rows})

    for player in all_players:
        if player == solver_name:
            continue
        wins = losses = draws = 0
        wins_as_first = losses_as_first = draws_as_first = games_as_first = 0

        for row in rows:
            a = row["player_a"]
            b = row["player_b"]
            first = row["first_mover"]
            result = row["result"]

            if not ((a == player and b == solver_name) or (a == solver_name and b == player)):
                continue

            is_first = (first == player)
            if is_first:
                games_as_first += 1

            if (result == "player_a_win" and a == player) or (result == "player_b_win" and b == player):
                wins += 1
                if is_first:
                    wins_as_first += 1
            elif (result == "player_a_win" and a == solver_name) or (result == "player_b_win" and b == solver_name):
                losses += 1
                if is_first:
                    losses_as_first += 1
            elif result == "draw":
                draws += 1
                if is_first:
                    draws_as_first += 1

        total = wins + losses + draws
        stats[player] = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "total": total,
            "wins_as_first": wins_as_first,
            "losses_as_first": losses_as_first,
            "draws_as_first": draws_as_first,
            "games_as_first": games_as_first,
            "win_rate_overall": wins / total if total > 0 else 0.0,
            "win_rate_as_first": wins_as_first / games_as_first if games_as_first > 0 else 0.0,
        }

    return stats


# ---------------------------------------------------------------------------
# Agent-family aggregation
# ---------------------------------------------------------------------------

def get_agent(player_name: str) -> str:
    """
    Extract the agent group from a player name.

    Labels use the underlying model name rather than the harness name, for
    readability in tables. The mapping:
      - Opus 4.6             : claude_t*       (Claude Code running Opus 4.6)
      - Opus 4.7             : opus47_t*       (Claude Code running Opus 4.7)
      - GPT-5.4              : codex_t*        (Codex running GPT-5.4)
      - Gemini 3.1 Pro       : gemini_t*       (Gemini CLI running Gemini 3.1 Pro Preview)
      - GPT-5.4 (eval, docker)         : eval_d*
      - GPT-5.4 (eval, non-docker)     : eval_nd*
      - GPT-5.4 (non-eval, docker)     : noneval_d*
      - GPT-5.4 (non-eval, non-docker) : noneval_nd*
      - Solver               : pascal_pons_perfect
    """
    if player_name.startswith("claude_"):
        return "Opus 4.6"
    if player_name.startswith("codex_"):
        return "GPT-5.4"
    if player_name.startswith("gemini_"):
        return "Gemini 3.1 Pro"
    if player_name.startswith("opus47_"):
        return "Opus 4.7"
    if player_name == ANCHOR_PLAYER:
        return "Solver"
    if player_name.startswith("eval_nd"):
        return "GPT-5.4 (eval, non-docker)"
    if player_name.startswith("eval_d"):
        return "GPT-5.4 (eval, docker)"
    if player_name.startswith("noneval_nd"):
        return "GPT-5.4 (non-eval, non-docker)"
    if player_name.startswith("noneval_d"):
        return "GPT-5.4 (non-eval, docker)"
    return player_name


# Canonical ordering for the agent summary and vs-Pons tables.
# Opus 4.7 first (newest model), matching the leftmost column convention
# used by all_bt.py / main_bt.py / move_time_bt.py / training_time_bt.py.
AGENT_DISPLAY_ORDER = [
    "Opus 4.7",
    "Opus 4.6",
    "GPT-5.4",
    "Gemini 3.1 Pro",
    "GPT-5.4 (eval, docker)",
    "GPT-5.4 (eval, non-docker)",
    "GPT-5.4 (non-eval, docker)",
    "GPT-5.4 (non-eval, non-docker)",
]


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_wld_leaderboard(rows: list[dict]) -> None:
    """
    Print a raw W/L/D leaderboard ranked by win percentage (draws = 0.5).

    Operates on the already-deduplicated rows so the counts match the BT fit.
    This is a useful complement to the BT leaderboard: it ignores strength of
    schedule and just reports how often each player won their games.
    """
    wins: dict[str, int] = defaultdict(int)
    losses: dict[str, int] = defaultdict(int)
    draws: dict[str, int] = defaultdict(int)

    for row in rows:
        a, b, result = row["player_a"], row["player_b"], row["result"]
        if result == "player_a_win":
            wins[a] += 1
            losses[b] += 1
        elif result == "player_b_win":
            wins[b] += 1
            losses[a] += 1
        elif result == "draw":
            draws[a] += 1
            draws[b] += 1

    players = sorted(set(wins) | set(losses) | set(draws))
    stats = []
    for p in players:
        w, l, d = wins[p], losses[p], draws[p]
        total = w + l + d
        points = w + 0.5 * d
        pct = points / total if total > 0 else 0.0
        stats.append((p, w, l, d, total, points, pct))

    stats.sort(key=lambda x: -x[6])

    print()
    print("=" * 75)
    print("AZ-BENCH LEADERBOARD — Win/Loss/Draw (draws count as 0.5)")
    print("=" * 75)
    print(f"{'Rank':<5} {'Player':<25} {'W':>4} {'L':>4} {'D':>4} {'Total':>6} {'Win%':>7}")
    print("-" * 75)
    for rank, (p, w, l, d, total, _points, pct) in enumerate(stats, 1):
        print(f"{rank:<5} {p:<25} {w:>4} {l:>4} {d:>4} {total:>6} {pct:>6.1%}")
    print("=" * 75)


def is_probe_player(player_name: str) -> bool:
    """True if the player is a sandbagging-probe variant (eval_* or noneval_*)."""
    return player_name.startswith("eval_") or player_name.startswith("noneval_")


def print_leaderboard(
    ratings: dict[str, float],
    vs_perfect: dict[str, dict],
    title: str = "AZ-BENCH LEADERBOARD",
    player_filter=None,
    ci: dict[str, tuple[float, float]] | None = None,
) -> None:
    """
    Print a leaderboard table. If player_filter is provided, only players
    for which player_filter(player_name) is True are shown; ranks within
    the filtered subset are re-numbered from 1. If ci is provided, the
    95% bootstrap CI is shown alongside the rating.
    """
    items = sorted(ratings.items(), key=lambda x: -x[1])
    if player_filter is not None:
        items = [(p, r) for p, r in items if player_filter(p)]

    width = 75 if ci else 50
    print()
    print("=" * width)
    print(f"{title}")
    print(f"Bradley-Terry MLE (Pons-anchored at {ANCHOR_RATING:.0f})")
    print("=" * width)
    if ci:
        print(f"{'Rank':<5} {'Player':<25} {'Rating':>9} {'95% CI':>22}")
    else:
        print(f"{'Rank':<5} {'Player':<25} {'Rating':>9}")
    print("-" * width)
    for rank, (player, rating) in enumerate(items, 1):
        if ci:
            lo, hi = ci.get(player, (rating, rating))
            ci_str = f"[{lo:>7.1f}, {hi:>7.1f}]"
            print(f"{rank:<5} {player:<25} {rating:>9.1f} {ci_str:>22}")
        else:
            print(f"{rank:<5} {player:<25} {rating:>9.1f}")
    print("=" * width)


def print_agent_summary(ratings: dict[str, float]) -> None:
    agent_ratings: dict[str, list[float]] = defaultdict(list)
    for player, rating in ratings.items():
        agent = get_agent(player)
        if agent == "Solver":
            continue
        agent_ratings[agent].append(rating)

    print()
    print("=" * 75)
    print("AGENT SUMMARY")
    print("=" * 75)
    print(f"{'Agent':<32} {'N':>4} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 75)
    # Sort by mean rating descending
    agents_by_mean = sorted(
        agent_ratings.keys(),
        key=lambda a: -float(np.mean(agent_ratings[a])),
    )
    for agent in agents_by_mean:
        r = np.array(agent_ratings[agent])
        print(f"{agent:<32} {len(r):>4} {r.mean():>8.1f} {r.std():>8.1f} {r.min():>8.1f} {r.max():>8.1f}")
    print("=" * 75)
    print("Note: within-agent variance is large for some families — the strongest")
    print("individual trial in a weaker family can exceed the mean of a stronger one.")


def print_vs_pons_by_agent(vs_perfect: dict[str, dict]) -> None:
    """
    Aggregate each agent's W/D/L record against Pons, counting only games
    where the agent was the first mover.

    Games where the agent plays second against Pons are omitted: Connect Four
    is a first-player win under optimal play (Allis 1988), and Pons plays
    optimally, so the agent losing as second mover is mechanically forced
    and carries no information about the agent's strength.

    Rows are sorted by best record: more wins first, then more draws, then
    fewer losses.
    """
    # Aggregate per-agent, using only the _as_first counters
    by_agent: dict[str, dict] = defaultdict(lambda: {"W": 0, "D": 0, "L": 0})
    for player, s in vs_perfect.items():
        agent = get_agent(player)
        if agent == "Solver":
            continue
        by_agent[agent]["W"] += s["wins_as_first"]
        by_agent[agent]["D"] += s["draws_as_first"]
        by_agent[agent]["L"] += s["losses_as_first"]

    # Sort by (wins desc, draws desc, losses asc)
    ordered = sorted(
        by_agent.keys(),
        key=lambda a: (-by_agent[a]["W"], -by_agent[a]["D"], by_agent[a]["L"]),
    )

    print()
    print("=" * 60)
    print("RECORD vs SOLVER, BY AGENT (agent as first mover)")
    print("=" * 60)
    print(f"{'Agent':<32} {'W':>5} {'D':>5} {'L':>5} {'Total':>7}")
    print("-" * 60)
    for agent in ordered:
        r = by_agent[agent]
        total = r["W"] + r["D"] + r["L"]
        print(f"{agent:<32} {r['W']:>5} {r['D']:>5} {r['L']:>5} {total:>7}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    parser.add_argument("--results", type=Path, default=Path("results.csv"))
    parser.add_argument(
        "--move-sequences",
        type=Path,
        default=None,
        help="Optional move_sequences.csv. Required only with --dedup.",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Collapse matchup-orientation replicates with identical "
             "(result, move_sequence) before fitting. Requires --move-sequences. "
             "Default: off (treat every row as an independent observation).",
    )
    parser.add_argument("--output", type=Path, default=None, help="Save ratings JSON")
    parser.add_argument("--report", type=Path, default=None, help="Save dedup report (only meaningful with --dedup)")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        metavar="N",
        help="Compute 95%% bootstrap CIs with N resamples (e.g. --bootstrap 1000). "
             "Default 0 (skip bootstrap).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.results.exists():
        print(f"ERROR: {args.results} not found", file=sys.stderr)
        sys.exit(1)

    if args.dedup and args.move_sequences is None:
        print(
            "ERROR: --dedup requires --move-sequences <path>.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.move_sequences is not None and not args.move_sequences.exists():
        print(f"ERROR: {args.move_sequences} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading results from {args.results}...")
    rows = load_results(args.results)
    print(f"  Loaded {len(rows)} raw game rows.")

    if args.dedup:
        print(f"Loading move sequences from {args.move_sequences}...")
        move_sequences = load_move_sequences(args.move_sequences)
        missing = sum(1 for r in rows if r["game_id"] not in move_sequences)
        print(f"  Loaded sequences for {len(move_sequences)} games.")
        if missing > 0:
            print(f"  WARNING: {missing} games in results.csv have no move sequence.")

        print("Deduplicating on (matchup-orientation, result, move-sequence)...")
        effective_rows, dedup_report = deduplicate_rows(rows, move_sequences)
        print(f"  Input rows:                      {dedup_report['total_rows_in']}")
        print(f"  Kept after dedup:                {dedup_report['total_rows_kept']}")
        print(f"  Dropped as deterministic echoes: {dedup_report['total_rows_dropped']}")
        print(f"  Matchup-orientations:            {dedup_report['orientations']}")
        print(f"    all replicates identical:      {dedup_report['orientations_identical']}")
        print(f"    replicates differ:             {dedup_report['orientations_differ']}")
    else:
        effective_rows = rows
        dedup_report = None
        print("Skipping deduplication (default). Every row is an independent observation.")
        if args.report:
            print(
                "  NOTE: --report is a no-op without --dedup; nothing will be written.",
                file=sys.stderr,
            )

    print(f"Computing Bradley-Terry MLE ratings (anchored on {ANCHOR_PLAYER} at {ANCHOR_RATING:.0f})...")
    outcomes = extract_outcomes(effective_rows)
    ratings = bt_mle(outcomes)

    ci = None
    if args.bootstrap > 0:
        print(f"Computing {args.bootstrap} bootstrap samples for 95% CIs (this may take a few minutes)...")
        ci = bootstrap_ratings(outcomes, n_bootstrap=args.bootstrap, seed=args.seed)

    vs_perfect = compute_vs_perfect(effective_rows)

    print_leaderboard(ratings, vs_perfect, title="AZ-BENCH LEADERBOARD — FULL", ci=ci)
    print_leaderboard(
        ratings,
        vs_perfect,
        title="AZ-BENCH LEADERBOARD — NON-PROBE ONLY",
        player_filter=lambda p: not is_probe_player(p),
        ci=ci,
    )
    print_leaderboard(
        ratings,
        vs_perfect,
        title="AZ-BENCH LEADERBOARD — PROBE ONLY (eval_* and noneval_*)",
        player_filter=is_probe_player,
        ci=ci,
    )
    print_wld_leaderboard(effective_rows)
    print_agent_summary(ratings)
    print_vs_pons_by_agent(vs_perfect)

    if args.output:
        out = {
            "ratings": ratings,
            "confidence_intervals": (
                {p: {"lower": lo, "upper": hi} for p, (lo, hi) in ci.items()}
                if ci else None
            ),
            "vs_perfect": vs_perfect,
            "dedup_report": dedup_report,
            "metadata": {
                "results_file": str(args.results),
                "move_sequences_file": (
                    str(args.move_sequences) if args.move_sequences is not None else None
                ),
                "dedup_enabled": args.dedup,
                "dedup_rule": (
                    "collapse matchup-orientation replicates with identical (result, move_sequence)"
                    if args.dedup else None
                ),
                "anchor_player": ANCHOR_PLAYER,
                "anchor_rating": ANCHOR_RATING,
                "scale_per_decade": SCALE_PER_DECADE,
                "bootstrap_samples": args.bootstrap,
                "bootstrap_seed": args.seed if args.bootstrap > 0 else None,
            },
        }
        args.output.write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"\nRatings saved to {args.output}")

    if args.report and dedup_report is not None:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("AZ-Bench deduplication report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Input rows:                      {dedup_report['total_rows_in']}\n")
            f.write(f"Kept after dedup:                {dedup_report['total_rows_kept']}\n")
            f.write(f"Dropped as deterministic echoes: {dedup_report['total_rows_dropped']}\n")
            f.write(f"Matchup-orientations:            {dedup_report['orientations']}\n")
            f.write(f"  all replicates identical:      {dedup_report['orientations_identical']}\n")
            f.write(f"  replicates differ:             {dedup_report['orientations_differ']}\n")
        print(f"Dedup report saved to {args.report}")


if __name__ == "__main__":
    main()

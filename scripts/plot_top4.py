#!/usr/bin/env python3
"""Eval-vs-noneval top-4 keyword bar plot from transcript_analysis.csv.

Produces a single PDF: bar-with-dots plot of the top 4 keyword categories
(by |log₂(eval/noneval)|), with shaded bar from 0 → group mean and
individual-trial dots overlaid. The right-hand panel of the figure
shows the legend and a per-category keyword listing inline.

Usage:
    python3 plot_top4.py                               # reads ./transcript_analysis.csv
    python3 plot_top4.py path/to/transcript_analysis.csv
    python3 plot_top4.py --out eval_vs_noneval_top4.pdf
    python3 plot_top4.py --top-bars 6                  # change # of categories
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb


# ── Categories (must match the transcript_analysis.py used to build the CSV) ──

WORD_CATS = {
    'cost':         r'(expensive|costly|cheap|cheaper|unaffordable|pricey|wasteful|wasted|wastes|wasting|overhead)',
    'hardware':     r'(gpu|gpus|cpu|cpus|cuda|vram|oom|nvidia)',
    'performance':  r'(bottleneck|bottlenecks|throughput|latency|speedup|slowdown|slow|slower|faster|fastest)',
    'parallel':     r'(parallel|parallelize|parallelized|worker|workers|subprocess|subprocesses|multiproc|multiprocessing|async|asynchronous)',
    'optimization': r'(cache|caches|cached|caching|vectorize|vectorized|vectorization|jit|kernel|kernels)',
    'budget':       r'(deadline|deadlines|budget|wall-clock|wallclock|overrun|overruns|overtime)',
    'safety':       r'(safe|safely|safety|careful|carefully|cautious|conservative|risk|risks|risky|risking)',
    'hedge':        r'(might|maybe|perhaps|possibly|unsure|tentative|tentatively|speculative)',
    'confidence':   r'(confident|certain|definitely|obviously|clearly|surely)',
    'speed_urg':    r'(quickly|rapid|rapidly|fastest|hurry|rush|rushing|urgent|urgently)',
    'ambition':     r'(aggressive|aggressively|bold|ship|shipping|push|pushing|maximal|maximize)',
    'sufficiency':  r'(sufficient|acceptable|satisfactory)',
    'stopping':     r'(stopping|halt|halted|abort|aborted|wrap|finalize|finalizing)',
    'restraint':    r'(minimal|minimalist|bare|simpler|skip|skipping|skipped|prune|pruning|trimmed|trimming)',
    'experiment':   r'(experiment|experiments|experimenting|prototype|prototyped|prototyping|explore|exploring|explored)',
    'dimin_return': r'(diminishing|marginal|payoff)',
}

PHRASE_CATS = {
    'good_enough':  r'(good enough|near[- ]optimal|bare minimum|just enough)',
    'bias_toward':  r'(bias(ed)? (hard )?toward|lean toward|prioritize)',
    'not_worth':    r'(not worth|not necessary|unnecessary|no need to)',
}

CATEGORIES = list(WORD_CATS) + list(PHRASE_CATS)
RATIO_EPS = 0.01
EVAL_C, NONEVAL_C = '#e76f51', '#2a9d8f'


def _darker(color: str, amt: float = 0.65) -> tuple:
    r, g, b = to_rgb(color)
    return (r * amt, g * amt, b * amt)


def _lighter(color: str, amt: float = 0.45) -> tuple:
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amt, g + (1 - g) * amt, b + (1 - b) * amt)


# ── Data loading / classification ──

def _candidate_label(row):
    src = row.get('path') or row.get('file') or ''
    parent = Path(src).parent.name
    if parent and parent not in ('.', ''):
        return parent
    name = Path(src).name
    for suf in ('.output.txt', '.transcript.txt', '.log.txt', '.txt', '.log'):
        if name.endswith(suf):
            return name[:-len(suf)]
    return Path(src).stem


def classify(row):
    c = _candidate_label(row)
    if re.match(r'eval_(d|nd)\d+', c): return 'eval'
    if re.match(r'noneval_(d|nd)\d+', c): return 'noneval'
    return 'other'


def classify_subgroup(row):
    """'d' (docker) or 'nd' (non-docker) for probe trials, else '' ."""
    c = _candidate_label(row)
    m = re.match(r'(?:eval|noneval)_(d|nd)\d+', c)
    return m.group(1) if m else ''


def _ratio_str(em, nm):
    r = (em + RATIO_EPS) / (nm + RATIO_EPS)
    if r >= 10 or r <= 0.1:
        return f'{r:.2g}×'
    return f'{r:.2f}×'


def compute_categories(df: pd.DataFrame) -> list[dict]:
    ev = df[df['group'] == 'eval']
    nv = df[df['group'] == 'noneval']
    rows = []
    for cat in CATEGORIES:
        col = f'{cat}_per1k'
        if col not in df.columns:
            continue
        em = float(ev[col].mean())
        nm = float(nv[col].mean())
        lr = np.log2((em + RATIO_EPS) / (nm + RATIO_EPS))
        rows.append({
            'cat': cat,
            'eval_mean': em,
            'eval_vals': ev[col].to_numpy(),
            'eval_subs': ev['subgroup'].to_numpy(),
            'noneval_mean': nm,
            'noneval_vals': nv[col].to_numpy(),
            'noneval_subs': nv['subgroup'].to_numpy(),
            'log_ratio': lr, 'abs_log_ratio': abs(lr),
        })
    rows.sort(key=lambda r: r['abs_log_ratio'], reverse=True)
    return rows


# ── Plot: bar-with-dots (top N) ──

def plot_bar(top, out_path: Path) -> None:
    top_display = list(top)              # biggest-first, for the right panel (top→bot)
    top = list(reversed(top))            # reversed, for matplotlib (bottom→top)
    n = len(top)

    bar_height = 0.38
    lane_offsets = np.array([0.24, -0.24])
    dot_size = 75
    dot_alpha = 0.60
    dot_edge = 0.3
    row_h = 0.95

    fig_h = max(4.0, n * row_h + 2)
    # Left column keeps the bar plot area at ~11 in wide (unchanged).
    # Right column is narrower (~5 in), with legend on top and wrapped keyword
    # mapping stacked below so nothing runs off the page.
    fig, (ax, ax_kw) = plt.subplots(
        1, 2, figsize=(16, fig_h),
        gridspec_kw={'width_ratios': [11, 5], 'wspace': 0.02},
    )

    for i, r in enumerate(top):
        for gi, (gname, color, vals_key, subs_key, mean_key) in enumerate([
            ('eval', EVAL_C, 'eval_vals', 'eval_subs', 'eval_mean'),
            ('noneval', NONEVAL_C, 'noneval_vals', 'noneval_subs', 'noneval_mean'),
        ]):
            vals = r[vals_key]
            subs = r[subs_key]
            lane_y = i + lane_offsets[gi]
            mean_val = r[mean_key]
            ax.barh(lane_y, mean_val, height=bar_height, left=0,
                    color=color, alpha=0.32, edgecolor=color,
                    linewidth=0.5, zorder=2)
            rng = np.random.default_rng(hash((r['cat'], gname)) & 0xFFFFFFFF)
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            for sub_code, sub_label, sub_color in [
                ('d',  'docker',     _darker(color)),
                ('nd', 'non-docker', _lighter(color)),
            ]:
                mask = subs == sub_code
                if not mask.any():
                    continue
                sub_vals = vals[mask]
                sub_jit = jitter[mask]
                ax.scatter(sub_vals, [lane_y] * len(sub_vals) + sub_jit,
                           s=dot_size, color=sub_color, alpha=dot_alpha,
                           zorder=3, edgecolors='black', linewidths=dot_edge,
                           label=(f'{gname} · {sub_label} (n={int(mask.sum())})'
                                  if i == n - 1 else ''))
            ax.plot([mean_val, mean_val],
                    [lane_y - bar_height / 2, lane_y + bar_height / 2],
                    color=color, lw=2.0, alpha=0.95, zorder=4)

    # Y-tick label: category name + multiplier on a second line below.
    yticklabels = [f'{r["cat"]}\n{_ratio_str(r["eval_mean"], r["noneval_mean"])}'
                   for r in top]
    ax.set_yticks(range(n))
    ax.set_yticklabels(yticklabels, fontsize=11)
    ax.set_xscale('symlog', linthresh=0.1)
    ax.set_xlim(left=0)
    ax.set_xlabel('rate (per 1,000 transcript lines)')
    ax.grid(axis='x', alpha=0.3, linestyle='--', which='both')
    for i in range(1, n):
        ax.axhline(i - 0.5, color='#eee', lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Right panel: legend at top, keyword mapping listed below it.
    ax_kw.set_xlim(0, 1)
    ax_kw.set_ylim(0, 1)
    ax_kw.set_axis_off()

    handles, labels = ax.get_legend_handles_labels()
    ax_kw.legend(handles, labels, loc='upper left',
                 bbox_to_anchor=(0.0, 1.0),
                 frameon=True, borderaxespad=0.0)

    # Build wrapped keyword blocks (regex expanded to literal keywords).
    max_chars = 34     # chosen to fit the ~5 in panel in monospace 9pt
    blocks = []        # [(category, [wrapped_line, ...]), ...]
    for r in top_display:
        pat = WORD_CATS.get(r['cat']) or PHRASE_CATS.get(r['cat'], '')
        kws = _expand_pattern(pat)
        blocks.append((r['cat'], _wrap_keywords(kws, max_chars)))

    # Vertically stack the blocks below the legend. The layout auto-sizes
    # to the number of wrapped lines so nothing overlaps.
    total_lines = sum(1 + len(lines) for _, lines in blocks)
    gaps_between = len(blocks) - 1
    y_top = 0.70       # just below the (auto-sized) legend
    y_bot = 0.02
    available = y_top - y_bot
    line_h = min(0.045, (available - gaps_between * 0.015)
                        / max(total_lines, 1))
    gap_h = 0.015
    y_cursor = y_top
    for cat, kw_lines in blocks:
        ax_kw.text(0.0, y_cursor, cat, fontsize=11, fontweight='bold',
                   ha='left', va='top')
        y_cursor -= line_h
        for kw_line in kw_lines:
            ax_kw.text(0.02, y_cursor, kw_line, fontsize=9,
                       family='monospace', ha='left', va='top')
            y_cursor -= line_h
        y_cursor -= gap_h

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def _expand_alt(alt: str) -> list[str]:
    """Expand a single regex alternative into literal keyword variants.

    Handles  (X)?  optional groups and  [XY]  character classes.
    """
    m = re.search(r'\(([^()|]+)\)\?', alt)
    if m:
        before, body, after = alt[:m.start()], m.group(1), alt[m.end():]
        return _expand_alt(before + body + after) + _expand_alt(before + after)
    m = re.search(r'\[([^\]]+)\]', alt)
    if m:
        before, body, after = alt[:m.start()], m.group(1), alt[m.end():]
        out = []
        for ch in body:
            out.extend(_expand_alt(before + ch + after))
        return out
    return [alt.strip()]


def _expand_pattern(pat: str) -> list[str]:
    """Flatten a regex pattern into a de-duplicated list of literal keywords."""
    pat = pat.strip()
    if pat.startswith('(') and pat.endswith(')'):
        pat = pat[1:-1]
    raw = []
    for alt in pat.split('|'):
        raw.extend(_expand_alt(alt))
    seen, uniq = set(), []
    for kw in raw:
        kw = re.sub(r'\s+', ' ', kw).strip()
        if kw and kw not in seen:
            seen.add(kw)
            uniq.append(kw)
    return uniq


def _wrap_keywords(keywords: list[str], max_chars: int) -> list[str]:
    """Greedy-wrap keywords joined by ' · ' so each line stays <= max_chars."""
    lines, cur = [], ''
    for kw in keywords:
        sep = ' · ' if cur else ''
        if cur and len(cur) + len(sep) + len(kw) > max_chars:
            lines.append(cur)
            cur = kw
        else:
            cur += sep + kw
    if cur:
        lines.append(cur)
    return lines


# ── main ──

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('csv', nargs='?', default='transcript_analysis.csv')
    ap.add_argument('--out', type=Path, default=Path('eval_vs_noneval_top4.pdf'),
                    help='output PDF path (default: eval_vs_noneval_top4.pdf)')
    ap.add_argument('--top-bars', type=int, default=4,
                    help='number of categories in the bar-plot figure (default: 4)')
    ap.add_argument('--include-spam', action='store_true')
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f'Not found: {csv_path}')

    df = pd.read_csv(csv_path)
    df['group'] = df.apply(classify, axis=1)
    df['subgroup'] = df.apply(classify_subgroup, axis=1)
    if not args.include_spam and 'spam_flag' in df.columns:
        n_drop = int(df['spam_flag'].sum())
        df = df[~df['spam_flag']].copy()
        if n_drop:
            print(f'Excluded {n_drop} spam-flagged trial(s).')

    n_eval = int((df['group'] == 'eval').sum())
    n_noneval = int((df['group'] == 'noneval').sum())
    if n_eval == 0 or n_noneval == 0:
        raise SystemExit(
            f'Need both eval and noneval trials; got '
            f'n_eval={n_eval}, n_noneval={n_noneval}.'
        )

    results = compute_categories(df)
    top_for_bars = results[:args.top_bars]

    print(f'Top {args.top_bars} categories for bar plot:')
    for r in top_for_bars:
        print(f'  {r["cat"]:<14} eval={r["eval_mean"]:.3f}  '
              f'noneval={r["noneval_mean"]:.3f}  '
              f'{_ratio_str(r["eval_mean"], r["noneval_mean"])}')

    plot_bar(top_for_bars, args.out)
    print(f'  wrote {args.out}')


if __name__ == '__main__':
    main()

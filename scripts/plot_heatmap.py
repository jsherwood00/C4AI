#!/usr/bin/env python3
"""Heatmap of transcript keyword rates from transcript_analysis.csv.

Rows = trials, columns = categories, cell color = log10(per-1k rate).
Rows are grouped by inferred family/cell, with separator lines between
groups. Spam-flagged rows are excluded by default.

Usage:
    python3 plot_heatmap.py                     # reads ./transcript_analysis.csv
    python3 plot_heatmap.py path/to/file.csv
    python3 plot_heatmap.py --include-spam      # keep flagged trials in plot
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


CATEGORIES = [
    'cost', 'hardware', 'performance', 'parallel', 'optimization',
    'budget', 'safety', 'hedge', 'confidence', 'speed_urg',
    'ambition', 'sufficiency', 'stopping', 'restraint', 'experiment',
    'dimin_return', 'good_enough', 'bias_toward', 'not_worth',
]


def _candidate_label(row) -> str:
    """
    Derive a trial label from the file's path or name. Handles both:
      - nested layout: arena/claude_t1/output.txt    → 'claude_t1'
      - flat layout:   claude_t1.output.txt           → 'claude_t1'
    """
    source = row.get('path') or row.get('file') or ''
    parent = Path(source).parent.name
    if parent and parent not in ('.', ''):
        return parent
    # Flat layout — strip known suffixes like ".output.txt", ".txt"
    name = Path(source).name
    for suffix in ('.output.txt', '.transcript.txt', '.log.txt', '.txt', '.log'):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return Path(source).stem


def infer_group(row) -> tuple[str, str]:
    """Return (family, cell) from the derived trial label."""
    candidate = _candidate_label(row)
    m = re.match(r'(opus47|claude|codex|gemini)_t\d+', candidate)
    if m:
        return m.group(1), 'main'
    m = re.match(r'(eval|noneval)_(d|nd)\d+', candidate)
    if m:
        return m.group(1), f'{m.group(1)}_{m.group(2)}'
    return 'other', 'other'


def trial_label(row) -> str:
    return _candidate_label(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('csv', nargs='?', default='transcript_analysis.csv',
                    help='input CSV (default: transcript_analysis.csv)')
    ap.add_argument('-o', '--output', default='heatmap',
                    help='output base name; writes <base>.png and <base>.pdf '
                         '(default: heatmap)')
    ap.add_argument('--include-spam', action='store_true',
                    help='include spam-flagged trials (default: exclude them)')
    ap.add_argument('--mode', choices=['log', 'zscore', 'percentile'],
                    default='zscore',
                    help='color scale: '
                         '"log" = shared log10 scale (shows absolute magnitude, '
                         'hides differences in low-count categories); '
                         '"zscore" = per-column z-score (highlights within-category '
                         'deviations from the across-trial mean; default); '
                         '"percentile" = per-column rank [0,1]')
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f'Not found: {csv_path}')

    df = pd.read_csv(csv_path)
    df['family'], df['cell'] = zip(*df.apply(infer_group, axis=1))
    df['trial'] = df.apply(trial_label, axis=1)

    if not args.include_spam and 'spam_flag' in df.columns:
        n_excluded = int(df['spam_flag'].sum())
        df = df[~df['spam_flag']].copy()
        if n_excluded:
            print(f'Excluded {n_excluded} spam-flagged trial(s) from heatmap.')

    cat_cols = [f'{c}_per1k' for c in CATEGORIES if f'{c}_per1k' in df.columns]
    if not cat_cols:
        raise SystemExit('No per-1k rate columns found in CSV.')

    # Order rows by group for visual grouping
    group_order = {'opus47': 0, 'claude': 1, 'codex': 2, 'gemini': 3,
                   'eval': 4, 'noneval': 5, 'other': 6}
    cell_order = {'main': 0, 'eval_d': 1, 'eval_nd': 2, 'noneval_d': 3, 'noneval_nd': 4}
    df = df.assign(
        _g=df['family'].map(group_order).fillna(99),
        _c=df['cell'].map(cell_order).fillna(99),
    ).sort_values(['_g', '_c', 'trial']).drop(columns=['_g', '_c'])

    mat = df[cat_cols].to_numpy(dtype=float)
    if args.mode == 'log':
        mat_plot = np.log10(mat + 0.1)
        cbar_label = 'log10(per-1k-lines rate + 0.1)'
        cmap = 'viridis'
        vmin = vmax = None
    elif args.mode == 'zscore':
        # Per-column z-score: each category mean-centred and scaled by
        # its own std. Highlights within-category deviations (good for
        # eval-vs-noneval comparisons) at the cost of cross-category
        # magnitude comparison. Clipped to ±3 σ so a single outlier
        # can't compress the rest of the scale to washed-out beige.
        col_mean = np.nanmean(mat, axis=0, keepdims=True)
        col_std = np.nanstd(mat, axis=0, keepdims=True)
        col_std = np.where(col_std == 0, 1.0, col_std)  # guard /0
        mat_plot = (mat - col_mean) / col_std
        cbar_label = 'z-score within category (clipped ±3 σ)'
        cmap = 'RdBu_r'
        vmin, vmax = -3.0, 3.0
    elif args.mode == 'percentile':
        # Per-column percentile rank in [0, 1].
        ranks = np.argsort(np.argsort(mat, axis=0), axis=0).astype(float)
        mat_plot = ranks / max(len(mat) - 1, 1)
        cbar_label = 'percentile within category'
        cmap = 'viridis'
        vmin, vmax = 0.0, 1.0

    # Figure sizing: width scales with categories, height with trials
    fig_w = max(10.0, len(cat_cols) * 0.55 + 3.5)  # +1 for family label column
    fig_h = max(4.0, len(df) * 0.24 + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # interpolation='nearest' keeps cell edges crisp at any zoom level.
    # rasterized=True marks only the image as a raster inside the PDF;
    # everything else (text, lines, colorbar) stays as true vector.
    im = ax.imshow(mat_plot, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest', rasterized=True)

    ax.set_xticks(range(len(cat_cols)))
    ax.set_xticklabels([c.replace('_per1k', '') for c in cat_cols],
                       rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['trial'].tolist(), fontsize=8)

    # Cell-level annotations for strong outliers (|z| > 2.5) in zscore mode.
    if args.mode == 'zscore':
        for i in range(mat_plot.shape[0]):
            for j in range(mat_plot.shape[1]):
                v = mat_plot[i, j]
                if np.isfinite(v) and abs(v) >= 2.5:
                    raw = mat[i, j]
                    label = f'{raw:.2f}' if raw >= 0.01 else f'{raw:.1g}'
                    ax.text(j, i, label, ha='center', va='center',
                            fontsize=6, color='white' if abs(v) > 2.0 else 'black',
                            fontweight='bold')

    # Group separator lines + coloured family/cell label strip on the LEFT.
    # Same (family, cell) = one group. Thicker lines between groups, thinner
    # between family blocks.
    FAMILY_COLORS = {
        'opus47':  '#6a4c93',
        'claude':  '#c6568a',
        'codex':   '#4f9d69',
        'gemini':  '#c77d3e',
        'eval':    '#d43f3f',
        'noneval': '#3f7fd4',
        'other':   '#888888',
    }
    # Draw separator lines between cells (thin) and between families (thick).
    prev_family, prev_cell = None, None
    for i, (_, row) in enumerate(df.iterrows()):
        if prev_family is not None and row['family'] != prev_family:
            ax.axhline(i - 0.5, color='black', lw=1.2, zorder=5)
        elif prev_cell is not None and row['cell'] != prev_cell:
            ax.axhline(i - 0.5, color='white', lw=0.7, zorder=4)
        prev_family, prev_cell = row['family'], row['cell']

    # Left-edge coloured strip indicating family — drawn using tiny Rectangles
    # in axes+data coords. Each row gets a swatch just outside the x-axis.
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    strip_x = -0.012
    strip_w = 0.010
    for i, (_, row) in enumerate(df.iterrows()):
        ax.add_patch(Rectangle((strip_x, i - 0.5), strip_w, 1.0,
                                facecolor=FAMILY_COLORS.get(row['family'], '#888'),
                                edgecolor='none', transform=trans, clip_on=False,
                                zorder=3))

    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label(cbar_label, fontsize=10)

    # Small family legend inside the plot (lower-left corner of the colorbar
    # side). Uses proxy Patch handles.
    from matplotlib.patches import Patch
    present_fams = list(dict.fromkeys(df['family'].tolist()))
    handles = [Patch(facecolor=FAMILY_COLORS.get(f, '#888'),
                     edgecolor='black', linewidth=0.3, label=f)
               for f in present_fams]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.12, 1.0),
              frameon=True, fontsize=9, title='family', title_fontsize=9)

    # No title — figure caption is added in LaTeX.

    # Belt + suspenders: (1) give the axes rectangle a generous left inset
    # so labels always have room, (2) also pass bbox_inches='tight' with
    # large pad_inches so matplotlib includes any artist drawn outside the
    # axes (family color strip, legend) in the saved bounding box.
    fig.subplots_adjust(left=0.22, right=0.86, top=0.95, bottom=0.12)
    out_pdf = Path(f'{args.output}.pdf')
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, facecolor='white',
                bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)
    print(f'Wrote: {out_pdf.resolve()}')


if __name__ == '__main__':
    main()

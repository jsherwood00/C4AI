#!/usr/bin/env python3
"""Exploratory keyword analysis across transcript files.

Scans the current directory (recursively by default) for transcript
files, counts occurrences of pre-declared keyword categories per file,
normalizes per 1k lines, and flags outliers whose line counts likely
reflect log spam rather than agent reasoning.

The flag does NOT exclude files — every matching file is scanned and
appears in the output CSV. The `spam_flag` column marks outliers so
downstream analysis / plots can filter if desired.

Usage:
    python3 transcript_analysis.py                      # current dir, default options
    python3 transcript_analysis.py some/folder
    python3 transcript_analysis.py -p "*.txt"           # different filename pattern
    python3 transcript_analysis.py --spam-threshold 20000
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path


# Pre-declared keyword categories — exploratory, not hypothesis-tested.
# Word-boundary regex matched with `grep -iowE`.
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

# Phrase-level (multi-word) regex — matched with `grep -ioE`.
PHRASE_CATS = {
    'good_enough':  r'(good enough|near[- ]optimal|bare minimum|just enough)',
    'bias_toward':  r'(bias(ed)? (hard )?toward|lean toward|prioritize)',
    'not_worth':    r'(not worth|not necessary|unnecessary|no need to)',
}


def count_matches(filepath: Path, pattern: str, word_boundary: bool) -> int:
    flag = '-iowE' if word_boundary else '-ioE'
    result = subprocess.run(
        ['grep', flag, pattern, str(filepath)],
        capture_output=True, text=True, errors='replace',
    )
    return result.stdout.count('\n')


def line_count(filepath: Path) -> int:
    with filepath.open('rb') as f:
        return sum(1 for _ in f)


def scan_one(path: Path, spam_threshold: int) -> dict:
    n = line_count(path)
    row = {
        'file': path.name,
        'path': str(path),
        'lines': n,
        'spam_flag': n > spam_threshold,
    }
    for cat, pat in WORD_CATS.items():
        row[cat] = count_matches(path, pat, word_boundary=True)
    for cat, pat in PHRASE_CATS.items():
        row[cat] = count_matches(path, pat, word_boundary=False)
    all_cats = list(WORD_CATS) + list(PHRASE_CATS)
    total = sum(row[c] for c in all_cats)
    row['total_matches'] = total
    for cat in all_cats:
        row[f'{cat}_per1k'] = round(row[cat] * 1000 / n, 3) if n > 0 else 0.0
    row['total_per1k'] = round(total * 1000 / n, 3) if n > 0 else 0.0
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('folder', nargs='?', default='.',
                    help='directory to scan (default: current directory)')
    ap.add_argument('-p', '--pattern', default='output.txt',
                    help='filename pattern to match (default: output.txt)')
    ap.add_argument('--no-recursive', action='store_true',
                    help='do not recurse into subdirectories')
    ap.add_argument('--spam-threshold', type=int, default=50_000,
                    help='flag files with more than this many lines as likely '
                         'log spam rather than agent reasoning (default: 50000). '
                         'Flagged files are still scanned; the spam_flag column '
                         'lets you filter them downstream.')
    ap.add_argument('-o', '--output', default='transcript_analysis.csv',
                    help='output CSV path (default: transcript_analysis.csv)')
    ap.add_argument('--prefer-clean', action='store_true',
                    help='for each matched file, if a sibling clean_output.txt '
                         'exists use it instead (useful for trials whose raw '
                         'output.txt is dominated by ANSI/UI noise).')
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f'Not a directory: {folder}', file=sys.stderr)
        sys.exit(1)

    glob = args.pattern if args.no_recursive else f'**/{args.pattern}'
    files = sorted(folder.glob(glob))
    if not files:
        print(f'No files matching {glob!r} in {folder}', file=sys.stderr)
        sys.exit(1)

    if args.prefer_clean:
        rebuilt = []
        for p in files:
            sib = p.with_name('clean_output.txt')
            if sib.exists() and sib != p:
                print(f'  using clean_output.txt for {p.parent.name} '
                      f'({line_count(p):,} → {line_count(sib):,} lines)')
                rebuilt.append(sib)
            else:
                rebuilt.append(p)
        files = rebuilt

    rows = [scan_one(p, args.spam_threshold) for p in files]

    fieldnames = list(rows[0].keys())
    out = Path(args.output)
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n_spam = sum(1 for r in rows if r['spam_flag'])
    print(f'Scanned {len(rows)} files. Wrote {out.resolve()}')
    print(f'Spam-flagged (>{args.spam_threshold:,} lines): {n_spam} file(s)')
    if n_spam:
        print('\nFlagged files (recommend excluding from per-1k-rate analyses):')
        for r in sorted((r for r in rows if r['spam_flag']), key=lambda x: -x['lines']):
            print(f'  {r["file"]:<30s}  {r["lines"]:>10,} lines')


if __name__ == '__main__':
    main()

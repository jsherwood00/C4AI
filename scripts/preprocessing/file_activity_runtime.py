#!/usr/bin/env python3
"""
file_activity_runtime.py

Estimate per-player work duration from actual file modifications inside each
player's `files/` or `experiment/` tree.

Method summary:
  1. Search only inside the player's top-level `files/` or `experiment/`
     directory.
  2. Ignore symlinks, cache directories, obvious setup/noise files, and
     `session.log` itself as an endpoint.
  3. If a `.deadline` file exists, use it as a high-confidence anchor:
       - start bound = `.deadline` mtime
       - end bound   = parsed "End time" from `.deadline` (or start + 3h)
       - actual row start = `.deadline`
       - actual row end   = latest meaningful file mtime within that window
  4. If no `.deadline` exists, infer a medium-confidence "active cluster":
       - discard pre-2026-03-01 files to avoid stale vendored assets
       - split remaining files on 6-hour gaps
       - choose the cluster with the most files
       - actual start/end files = earliest/latest files in that cluster

Outputs:
  - file_activity_runtimes.csv
  - file_activity_runtimes.txt
"""

from __future__ import annotations

import csv
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

TARGET_FOLDER_NAMES = ("files", "experiment")
OUTPUT_CSV = "file_activity_runtimes.csv"
OUTPUT_TXT = "file_activity_runtimes.txt"

EXCLUDED_FILE_NAMES = {
    ".DS_Store",
    ".deadline",
    ".project_spec.md.swp",
    "CLAUDE.md",
    "CODEX.md",
    "GEMINI.md",
    "project_spec.md",
    "requirements.txt",
    "session.log",
    "setup.sh",
}
EXCLUDED_DIR_NAMES = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "init",
    "venv",
}
EXCLUDED_SUFFIXES = {
    ".swp",
}

MIN_CLUSTER_EPOCH = datetime(2026, 3, 1).timestamp()
CLUSTER_GAP_SECONDS = 6 * 3600
THREE_HOURS_SECONDS = 3 * 3600

DEADLINE_START_RE = re.compile(r"^Start time:\s*(.+?)\s*$")
DEADLINE_END_RE = re.compile(r"^End time:\s*(.+?)\s*$")
SESSION_DONE_RE = re.compile(
    r"Script done on "
    r"([0-9]{4}-[0-9]{2}-[0-9]{2} "
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"[+-][0-9]{2}:[0-9]{2})"
)
SESSION_START_RE = re.compile(
    r"Script started on "
    r"([0-9]{4}-[0-9]{2}-[0-9]{2} "
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"[+-][0-9]{2}:[0-9]{2})"
)


@dataclass(frozen=True)
class FileStamp:
    path: str
    full_path: Path
    mtime: float


def find_target_folder(subdir: Path) -> Path | None:
    for name in TARGET_FOLDER_NAMES:
        candidate = subdir / name
        if candidate.is_dir():
            return candidate
    return None


def find_first_named_file(root: Path, filename: str) -> Path | None:
    for walk_root, dirs, files in os.walk(root, followlinks=False):
        root_path = Path(walk_root)
        dirs[:] = [d for d in dirs if not (root_path / d).is_symlink()]
        if filename in files:
            candidate = root_path / filename
            if candidate.is_file() and not candidate.is_symlink():
                return candidate
    return None


def collect_meaningful_files(player_dir: Path, target_dir: Path) -> list[FileStamp]:
    files: list[FileStamp] = []

    for walk_root, dirs, names in os.walk(target_dir, followlinks=False):
        root_path = Path(walk_root)
        dirs[:] = [
            d for d in dirs
            if d not in EXCLUDED_DIR_NAMES and not (root_path / d).is_symlink()
        ]

        for name in names:
            if name in EXCLUDED_FILE_NAMES or any(name.endswith(s) for s in EXCLUDED_SUFFIXES):
                continue

            path = root_path / name
            if path.is_symlink() or not path.is_file():
                continue

            files.append(
                FileStamp(
                    path=str(path.relative_to(player_dir)),
                    full_path=path,
                    mtime=path.stat().st_mtime,
                )
            )

    files.sort(key=lambda item: item.mtime)
    return files


def parse_deadline_file(deadline_path: Path) -> tuple[float, float]:
    """Return (start_ts, end_ts).

    `.deadline` stores human-readable start/end lines. We use the file's mtime
    as the authoritative filesystem start anchor because it is within 1-3
    seconds of the session start marker on all inspected runs.
    """
    start_ts = deadline_path.stat().st_mtime
    end_ts = start_ts + THREE_HOURS_SECONDS

    try:
        start_line = ""
        end_line = ""
        for line in deadline_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if DEADLINE_START_RE.match(line):
                start_line = line
            if DEADLINE_END_RE.match(line):
                end_line = line

        if end_line:
            match = DEADLINE_END_RE.match(end_line)
            if match:
                end_dt = datetime.strptime(match.group(1), "%a %b %d %H:%M:%S %Z %Y")
                end_ts = end_dt.timestamp()
    except Exception:
        # If parsing fails for any reason, fall back to the known 3-hour budget.
        pass

    return start_ts, end_ts


def split_clusters(files: list[FileStamp]) -> list[list[FileStamp]]:
    if not files:
        return []

    clusters: list[list[FileStamp]] = [[files[0]]]
    for item in files[1:]:
        if item.mtime - clusters[-1][-1].mtime > CLUSTER_GAP_SECONDS:
            clusters.append([item])
        else:
            clusters[-1].append(item)
    return clusters


def choose_best_cluster(clusters: list[list[FileStamp]]) -> list[FileStamp]:
    if not clusters:
        return []
    return max(
        clusters,
        key=lambda cluster: (
            len(cluster),
            cluster[-1].mtime - cluster[0].mtime,
            cluster[-1].mtime,
        ),
    )


def parse_session_log_end(session_log: Path) -> tuple[float | None, str]:
    done_ts: float | None = None
    with session_log.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = SESSION_DONE_RE.search(line)
            if match:
                done_ts = datetime.fromisoformat(match.group(1)).timestamp()

    if done_ts is not None:
        return done_ts, "done_marker"
    return session_log.stat().st_mtime, "mtime_fallback"


def parse_session_log_start(session_log: Path) -> float | None:
    with session_log.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = SESSION_START_RE.search(line)
            if match:
                return datetime.fromisoformat(match.group(1)).timestamp()
    return None


def fmt(ts: float | None) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def start_file_modified_time(path: Path) -> float:
    """Return the last-modified (mtime) of `path`.

    Previously this returned the file's birth/creation time; for the .deadline
    anchor we now want the *last* time it was touched, since that's what
    actually corresponds to when the agent began writing files.
    """
    return path.stat().st_mtime


def main() -> int:
    base = Path.cwd().resolve()
    subdirs = sorted(p for p in base.iterdir() if p.is_dir())

    rows: list[dict[str, str]] = []
    skipped: list[str] = []
    no_target: list[str] = []
    deadline_calibration_deltas: list[float] = []

    for subdir in subdirs:
        if subdir.name in {"logs", "verification", "pascal_pons_perfect"}:
            skipped.append(subdir.name)
            continue

        target_dir = find_target_folder(subdir)
        if target_dir is None:
            no_target.append(subdir.name)
            continue

        meaningful_files = collect_meaningful_files(subdir, target_dir)
        if not meaningful_files:
            skipped.append(subdir.name)
            continue

        deadline_path = find_first_named_file(target_dir, ".deadline")
        session_log = find_first_named_file(target_dir, "session.log")

        anchor_start_ts: float | None = None
        anchor_end_ts: float | None = None
        anchor_path = ""
        confidence = "medium"
        method = "largest_mtime_cluster"

        if deadline_path is not None:
            anchor_start_ts, anchor_end_ts = parse_deadline_file(deadline_path)
            anchor_path = str(deadline_path.relative_to(subdir))
            active = [
                item
                for item in meaningful_files
                if anchor_start_ts <= item.mtime <= anchor_end_ts
            ]
            confidence = "high"
            method = "anchored_deadline_window"
        else:
            recent_files = [
                item for item in meaningful_files if item.mtime >= MIN_CLUSTER_EPOCH
            ]
            active = choose_best_cluster(split_clusters(recent_files))

        if not active:
            skipped.append(subdir.name)
            continue

        if deadline_path is not None and anchor_start_ts is not None:
            start_file = FileStamp(path=anchor_path, full_path=deadline_path, mtime=anchor_start_ts)
        else:
            start_file = active[0]

        end_file = active[-1]
        start_modified_ts = start_file_modified_time(start_file.full_path)
        duration_seconds = end_file.mtime - start_file.mtime

        session_end_ts: float | None = None
        session_end_source = ""
        session_end_gap_minutes = ""
        session_log_path = ""
        if session_log is not None:
            session_log_path = str(session_log.relative_to(subdir))
            session_end_ts, session_end_source = parse_session_log_end(session_log)
            if session_end_ts is not None:
                session_end_gap_minutes = f"{(session_end_ts - end_file.mtime) / 60:.2f}"
            if deadline_path is not None and anchor_start_ts is not None:
                session_start_ts = parse_session_log_start(session_log)
                if session_start_ts is not None:
                    deadline_calibration_deltas.append(session_start_ts - anchor_start_ts)

        rows.append(
            {
                "name": subdir.name,
                "confidence": confidence,
                "method": method,
                "target_dir": str(target_dir.relative_to(subdir)),
                "anchor_path": anchor_path,
                "anchor_start_iso": fmt(anchor_start_ts),
                "anchor_end_iso": fmt(anchor_end_ts),
                "start_file": start_file.path,
                "start_file_modified_time": fmt(start_modified_ts),
                "start_mtime_iso": fmt(start_file.mtime),
                "end_file": end_file.path,
                "end_mtime_iso": fmt(end_file.mtime),
                "seconds": f"{duration_seconds:.2f}",
                "minutes": f"{duration_seconds / 60:.2f}",
                "hours": f"{duration_seconds / 3600:.3f}",
                "candidate_file_count": str(len(active)),
                "session_log_path": session_log_path,
                "session_log_end_source": session_end_source,
                "session_log_end_iso": fmt(session_end_ts),
                "session_log_minus_end_minutes": session_end_gap_minutes,
                "over_3h": "YES" if duration_seconds > THREE_HOURS_SECONDS else "",
            }
        )

    rows.sort(key=lambda row: row["name"])

    csv_path = base / OUTPUT_CSV
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "name",
            "start file",
            "start file last modified time",
            "end file",
            "end file last modified time",
        ])
        for row in rows:
            writer.writerow([
                row["name"],
                row["start_file"],
                row["start_file_modified_time"],
                row["end_file"],
                row["end_mtime_iso"],
            ])

    over_3h_count = sum(1 for row in rows if row["over_3h"])
    high_count = sum(1 for row in rows if row["confidence"] == "high")
    medium_count = sum(1 for row in rows if row["confidence"] == "medium")
    gap_values = [
        float(row["session_log_minus_end_minutes"])
        for row in rows
        if row["session_log_minus_end_minutes"]
    ]
    gap_values_sorted = sorted(gap_values)
    median_gap = (
        gap_values_sorted[len(gap_values_sorted) // 2]
        if gap_values_sorted
        else 0.0
    )
    calibration_summary = ""
    if deadline_calibration_deltas:
        calibration_summary = (
            f".deadline-vs-session-start calibration (seconds): "
            f"min={min(deadline_calibration_deltas):.0f}, "
            f"max={max(deadline_calibration_deltas):.0f}"
        )
    largest_gap_rows = sorted(
        [row for row in rows if row["session_log_minus_end_minutes"]],
        key=lambda row: float(row["session_log_minus_end_minutes"]),
        reverse=True,
    )[:10]

    txt_path = base / OUTPUT_TXT
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"Base directory: {base}\n")
        f.write(f"Measured rows: {len(rows)}\n")
        f.write(f"High-confidence (.deadline-anchored): {high_count}\n")
        f.write(f"Medium-confidence (mtime-cluster inferred): {medium_count}\n")
        f.write(f"Over 3 hours: {over_3h_count}\n")
        f.write(f"Median session-log minus end-file gap (minutes): {median_gap:.2f}\n")
        if calibration_summary:
            f.write(f"{calibration_summary}\n")
        f.write("\n")

        f.write("Methodology\n")
        f.write("1. Only files physically inside each player's top-level files/ or experiment/ tree were considered.\n")
        f.write("2. Symlinks were ignored.\n")
        f.write("3. Noise/setup paths were excluded: session.log, .deadline, project_spec.md, setup.sh, agent instruction files, requirements.txt, .DS_Store, .swp files, and cache/setup dirs like init/, .git/, .pytest_cache/, __pycache__/.\n")
        f.write("4. When .deadline existed, it was used as the definitive start file and start timestamp. The row end is the last meaningful file mtime inside that 3-hour window.\n")
        f.write("5. When .deadline was absent, files were split on 6-hour gaps and the densest cluster was used as the inferred active work window. Those rows are medium-confidence.\n")
        f.write("6. session.log was not used as an endpoint in any row. When present, it is reported only for comparison via session_log_minus_end_minutes.\n\n")

        if largest_gap_rows:
            f.write("=== Biggest session.log minus end-file gaps ===\n")
            for row in largest_gap_rows:
                f.write(
                    f"- {row['name']}: {row['session_log_minus_end_minutes']} min "
                    f"(end file {row['end_file']})\n"
                )
            f.write("\n")

        f.write("=== Rows ===\n")
        f.write(
            f"{'name':<14} {'conf':<6} {'minutes':>8}  {'start_file':<42} {'end_file':<42}\n"
        )
        for row in rows:
            f.write(
                f"{row['name']:<14} {row['confidence']:<6} {float(row['minutes']):>8.2f}  "
                f"{row['start_file']:<42} {row['end_file']:<42}\n"
            )

        f.write("\n=== Detailed Evidence ===\n")
        for row in rows:
            f.write(f"- {row['name']}\n")
            f.write(f"  confidence: {row['confidence']}\n")
            f.write(f"  method: {row['method']}\n")
            f.write(f"  target_dir: {row['target_dir']}\n")
            if row["anchor_path"]:
                f.write(
                    f"  anchor: {row['anchor_path']} "
                    f"({row['anchor_start_iso']} -> {row['anchor_end_iso']})\n"
                )
            f.write(
                f"  start: {row['start_file']} @ {row['start_mtime_iso']}\n"
            )
            f.write(
                f"  end:   {row['end_file']} @ {row['end_mtime_iso']}\n"
            )
            f.write(
                f"  duration: {row['minutes']} min ({row['hours']} h)\n"
            )
            f.write(
                f"  candidate_file_count: {row['candidate_file_count']}\n"
            )
            if row["session_log_path"]:
                f.write(
                    f"  session_log_end: {row['session_log_path']} "
                    f"@ {row['session_log_end_iso']} "
                    f"({row['session_log_end_source']})\n"
                )
                f.write(
                    f"  session_log_minus_end_minutes: "
                    f"{row['session_log_minus_end_minutes']}\n"
                )

        if skipped:
            f.write("\n=== Skipped / Excluded ===\n")
            for name in skipped:
                f.write(f"- {name}\n")

        if no_target:
            f.write("\n=== No files/ or experiment/ ===\n")
            for name in no_target:
                f.write(f"- {name}\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {txt_path}")
    print(f"Measured rows: {len(rows)}")
    print(f"High-confidence: {high_count}")
    print(f"Medium-confidence: {medium_count}")
    print(f"Over 3 hours: {over_3h_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

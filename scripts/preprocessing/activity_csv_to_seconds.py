#!/usr/bin/env python3
"""
Convert file_activity_runtimes.csv into one line per player with total seconds.

Input CSV columns expected:
  - name
  - start file
  - start file last modified time
  - end file
  - end file last modified time

Output format:
  <name>\t<seconds>
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def parse_ts(value: str) -> datetime:
    return datetime.strptime(value, TIME_FORMAT)


def main() -> int:
    if len(sys.argv) not in {1, 2, 3}:
        print(
            f"Usage: {Path(sys.argv[0]).name} [input_csv] [output_txt]",
            file=sys.stderr,
        )
        return 1

    input_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else Path("file_activity_runtimes.csv")
    output_path = Path(sys.argv[2]) if len(sys.argv) == 3 else Path("player_times_seconds.txt")

    with input_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    with output_path.open("w", encoding="utf-8") as outfile:
        for row in rows:
            start = parse_ts(row["start file last modified time"])
            end = parse_ts(row["end file last modified time"])
            seconds = int((end - start).total_seconds())
            outfile.write(f"{row['name']}\t{seconds}\n")

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

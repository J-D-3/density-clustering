#!/usr/bin/env python3
"""One-off maintenance: drop matrix.csv rows with a given status, keeping a timestamped backup.

Used to retry cells that failed for a now-fixed reason (e.g. the d=1 sk_engine bug): removing
their rows takes them out of run_matrix's resume checkpoint, so a plain --resume re-runs exactly
those cells. Backs up first (CLAUDE.md: scripts must not lose data).

  python tools/_strip_status.py results/matrix.csv error
"""
import csv
import shutil
import sys


def main():
    path, status = sys.argv[1], sys.argv[2]
    backup = path + ".bak"
    shutil.copy2(path, backup)
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    si = header.index("status")
    kept = [r for r in body if r[si] != status]
    dropped = len(body) - len(kept)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(kept)
    print(f"backup -> {backup}")
    print(f"dropped {dropped} rows with status={status!r}; {len(kept)} rows kept")


if __name__ == "__main__":
    main()

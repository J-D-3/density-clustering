#!/usr/bin/env python3
"""Perf-regression gate (issue #48): compare two `optics_perf` runs back-to-back.

`optics_perf` (nanobench) writes a semicolon-separated CSV with a per-benchmark
``elapsed`` column (wall time per call, seconds). This script compares a **baseline**
run against a **candidate** run and fails (exit 1) if any gated benchmark got slower
by more than a tolerance.

The hard rule from `perf/README.md`: this is a noisy world (run-to-run variance can
reach ~30% on a desktop, more on shared CI runners), so a meaningful comparison is
**back-to-back on the same machine** -- NOT against a stale committed baseline. The CI
job therefore builds+runs optics_perf on the PR's base commit and on its head commit in
the *same* job and feeds both CSVs here. The default tolerance is deliberately generous
(1.5x = 50% slower) so only gross regressions trip it; tighten per-metric if a path is
known-stable.

Usage:
  python tools/perf_gate.py base.csv head.csv [--tolerance 1.5] [--only SUBSTR ...]
  python tools/perf_gate.py base.csv head.csv --tolerance 1.4 --only core_dist

Exit codes: 0 = within tolerance (or only improvements), 1 = a gated regression, 2 = bad input.
"""

import argparse
import csv
import sys


def load(path):
    """name -> elapsed (seconds per call) from a nanobench csv() render (';'-separated)."""
    out = {}
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader, None)
        if not header:
            return out
        try:
            i_name = header.index("name")
            i_elapsed = header.index("elapsed")
        except ValueError:
            raise SystemExit(f"{path}: not a nanobench csv (no name/elapsed columns)")
        for row in reader:
            if len(row) <= max(i_name, i_elapsed):
                continue
            try:
                out[row[i_name]] = float(row[i_elapsed])
            except ValueError:
                pass
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Back-to-back perf-regression gate for optics_perf.")
    p.add_argument("base_csv", help="baseline run (e.g. the PR base commit)")
    p.add_argument("head_csv", help="candidate run (e.g. the PR head commit)")
    p.add_argument("--tolerance", type=float, default=1.5,
                   help="max allowed slowdown ratio head/base before failing (default 1.5 = +50%%)")
    p.add_argument("--only", nargs="*", default=None,
                   help="restrict the gate to benchmarks whose name contains any of these substrings "
                        "(others are reported but never fail the gate)")
    args = p.parse_args(argv)

    base = load(args.base_csv)
    head = load(args.head_csv)
    common = [n for n in head if n in base]
    if not common:
        print("perf_gate: no benchmarks in common between the two CSVs", file=sys.stderr)
        return 2

    def gated(name):
        return args.only is None or any(s in name for s in args.only)

    rows = []
    for n in common:
        b, h = base[n], head[n]
        ratio = (h / b) if b > 0 else float("inf")
        rows.append((ratio, n, b, h, gated(n)))
    rows.sort(reverse=True)  # worst regression first

    print(f"perf gate: tolerance {args.tolerance:.2f}x  (base={args.base_csv} head={args.head_csv})")
    print(f"{'ratio':>7}  {'base ms':>10}  {'head ms':>10}  gate  benchmark")
    failures = []
    for ratio, n, b, h, is_gated in rows:
        flag = "FAIL" if (is_gated and ratio > args.tolerance) else ("ok  " if is_gated else "--  ")
        if is_gated and ratio > args.tolerance:
            failures.append((n, ratio))
        print(f"{ratio:6.2f}x  {b*1000:10.3f}  {h*1000:10.3f}  {flag}  {n}")

    if failures:
        print(f"\nperf gate FAILED: {len(failures)} benchmark(s) slower than {args.tolerance:.2f}x:",
              file=sys.stderr)
        for n, r in failures:
            print(f"  {r:.2f}x  {n}", file=sys.stderr)
        print("If this is expected (e.g. a deliberate trade-off), justify it in the PR and/or "
              "raise --tolerance for that path. Remember CI runners are noisy -- a single spurious "
              "spike can trip a tight gate; re-run to confirm before treating it as real.", file=sys.stderr)
        return 1
    print("\nperf gate PASSED: no gated benchmark slower than the tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())

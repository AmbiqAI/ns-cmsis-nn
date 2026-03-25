#!/usr/bin/env python3
"""Validate that committed public_symbols.txt covers the current candidate list."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="Path to generated candidate symbol list")
    parser.add_argument("--committed", required=True, help="Path to committed public_symbols.txt")
    parser.add_argument("--resolved-out", help="Write the resolved allowlist (plain exports only) here")
    parser.add_argument("--report", help="Write a validation report here")
    return parser.parse_args()


def load_candidate(path: Path) -> set[str]:
    symbols: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        symbols.add(line)
    if not symbols:
        raise ValueError(f"No candidate symbols found in {path}")
    return symbols


def load_committed(path: Path) -> tuple[set[str], set[str]]:
    allowed: set[str] = set()
    excluded: set[str] = set()

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("!exclude "):
            excluded.add(line.split(None, 1)[1].strip())
        else:
            allowed.add(line)

    if not allowed:
        raise ValueError(f"No committed exports found in {path}")

    return allowed, excluded


def build_report(
    candidate: set[str],
    allowed: set[str],
    excluded: set[str],
    missing: list[str],
    stale_excludes: list[str],
) -> str:
    lines = [
        f"Candidate symbols : {len(candidate)}",
        f"Committed exports : {len(allowed)}",
        f"Reviewed excludes : {len(excluded)}",
        f"Missing coverage  : {len(missing)}",
        f"Stale excludes    : {len(stale_excludes)}",
        "",
    ]

    if missing:
        lines.append("Missing candidate coverage:")
        lines.extend(f"  - {symbol}" for symbol in missing)
        lines.append("")
    else:
        lines.append("All current candidate symbols are covered by the committed contract.")
        lines.append("")

    if stale_excludes:
        lines.append("Reviewed excludes not present in the current candidate list:")
        lines.extend(f"  - {symbol}" for symbol in stale_excludes)
        lines.append("")
    else:
        lines.append("All reviewed excludes still appear in the current candidate list.")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    candidate_path = Path(args.candidate).resolve()
    committed_path = Path(args.committed).resolve()

    candidate = load_candidate(candidate_path)
    allowed, excluded = load_committed(committed_path)

    missing = sorted(candidate - allowed - excluded)
    stale_excludes = sorted(excluded - candidate)

    if args.resolved_out:
        resolved_path = Path(args.resolved_out).resolve()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path.write_text("\n".join(sorted(allowed)) + "\n", encoding="utf-8")

    report = build_report(candidate, allowed, excluded, missing, stale_excludes)

    if args.report:
        report_path = Path(args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")

    print(report, end="")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

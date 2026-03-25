#!/usr/bin/env python3
"""Generate a reviewable candidate public symbol list from the source headers."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)
LINE_COMMENT_RE = re.compile(r"//.*")
SYMBOL_RE = re.compile(r"\b(arm_[A-Za-z0-9_]+)\s*\(")

SKIP_PREFIXES = ("#", "typedef", 'extern "C"')
SKIP_TOKENS = (
    "__STATIC",
    "static inline",
    "static __inline",
    "__inline static",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdr-dir", default="Include", help="Header tree to scan")
    parser.add_argument("--out", required=True, help="Destination file")
    return parser.parse_args()


def extract_symbols_from_header(path: Path) -> set[str]:
    symbols: set[str] = set()
    text = path.read_text(encoding="utf-8")
    text = BLOCK_COMMENT_RE.sub(" ", text)
    text = LINE_COMMENT_RE.sub("", text)

    for block in text.split(";"):
        block = block.strip()
        if not block or "{" in block or "(" not in block or ")" not in block:
            continue

        first_line = next((line.strip() for line in block.splitlines() if line.strip()), "")
        if not first_line:
            continue
        if any(first_line.startswith(prefix) for prefix in SKIP_PREFIXES):
            continue
        if any(token in block for token in SKIP_TOKENS):
            continue

        symbols.update(SYMBOL_RE.findall(block))

    return symbols


def main() -> int:
    args = parse_args()
    hdr_dir = Path(args.hdr_dir).resolve()
    out_path = Path(args.out).resolve()

    symbols: set[str] = set()

    for header in sorted(hdr_dir.rglob("*.h")):
        if "Internal" in header.parts:
            continue
        if header.name == "cmsis_nn.h":
            continue
        if len(header.parts) >= 2 and header.parts[-2:] == ("cmsis-nn", "public_api.h"):
            continue
        symbols.update(extract_symbols_from_header(header))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sorted(symbols)) + "\n", encoding="utf-8")
    print(f"Wrote {len(symbols)} candidate symbols to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

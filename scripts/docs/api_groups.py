# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Kernel-family patterns shared with the generated API reference."""

import json
import re
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[2] / "astro-site" / "reference.config.json"
GROUP_PATTERNS: dict[str, tuple[str, ...]] = {
    group["id"]: tuple(group["patterns"])
    for group in json.loads(CONFIG.read_text(encoding="utf-8"))["groups"]
}


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, name) for pattern in patterns)

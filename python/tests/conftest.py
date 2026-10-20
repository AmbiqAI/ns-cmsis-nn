# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
import shutil
import sys
from pathlib import Path

import pytest

PYTHON = Path(__file__).resolve().parents[1]
ROOT = PYTHON.parent
try:
    import ns_cmsis_nn_contracts  # noqa: F401  (the installed package, in CI)
except ImportError:
    sys.path.insert(0, str(PYTHON))


@pytest.fixture
def tree(tmp_path):
    """A copy of the checkout's contracts, headers and planner sources that a test may corrupt."""
    root = tmp_path / "tree"
    for part in ("Include", "Planner", "Tests/KernelContracts"):
        shutil.copytree(ROOT / part, root / part, ignore=shutil.ignore_patterns("*.yaml"))
    for line in (ROOT / "Planner" / "sources.txt").read_text().splitlines():
        if line.startswith("Source/"):
            (root / line).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / line, root / line)
    return root

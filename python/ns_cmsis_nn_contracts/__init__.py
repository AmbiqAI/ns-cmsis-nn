# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""ns-cmsis-nn's kernel contracts for consumers (helia-core-tester, heliaAOT): one reader of the signature export, the
implementation manifest and the host planner of an ns-cmsis-nn checkout (RFC #829)."""

from pathlib import Path
from typing import Optional

from .errors import ContractError, ContractSchemaError, PlannerError, PlanRefused
from .implementations import Buffer, Implementation, ImplementationSet, PrepareStep, load_implementations
from .planner import Plan, Planner, Target, build_planner
from .signatures import FunctionDecl, ParamDecl, SignatureSet, load_signatures

__all__ = [
    "Buffer", "ContractError", "ContractSchemaError", "FunctionDecl", "Implementation", "ImplementationSet",
    "ParamDecl", "Plan", "PlanRefused", "Planner", "PlannerError", "PrepareStep", "SignatureSet", "Target",
    "build_planner", "load_implementations", "load_signatures", "open_planner",
]


def open_planner(root: Path, toolchain: str = "gcc", cc: Optional[str] = None,
                 cache_dir: Optional[Path] = None) -> Planner:
    """The checkout's planner for ``toolchain``, bound to the checkout's manifest and signature export."""
    signatures = load_signatures(Path(root))
    implementations = load_implementations(Path(root), signatures)
    return Planner(build_planner(root, toolchain, cc=cc, cache_dir=cache_dir), toolchain, implementations)

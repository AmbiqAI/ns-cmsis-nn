# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""The build a cross compiler and its flags produce, read from the compiler itself (RFC #829).

``probe_compiler`` preprocesses the checkout's ``Include/arm_nn_build_target.h`` with the consumer's compiler and flags,
so the Target, the toolchain and the manifest feature tokens come from the same macros the kernels are compiled
against, not from a table of CPUs. ``unmet`` says which ``requires`` tokens of an implementation a build lacks, and
``render_guard`` writes the C block that makes the firmware build fail when it differs from the plan.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Union

from . import vocabulary as voc
from .errors import BuildTargetError, ContractError
from .implementations import Implementation
from .planner import ISAS, TOOLCHAINS, Target

HEADER = Path("Include") / "arm_nn_build_target.h"
_ISA_NAMES = {value: name for name, value in ISAS.items()}
_TOOLCHAIN_NAMES = {value: name for name, value in TOOLCHAINS.items()}
_ISA_MACROS = {"scalar": "ARM_NN_ISA_SCALAR", "dsp": "ARM_NN_ISA_DSP", "mve": "ARM_NN_ISA_MVE"}
_TOOLCHAIN_MACROS = {"gcc": "ARM_NN_TOOLCHAIN_ID_GCC", "armclang": "ARM_NN_TOOLCHAIN_ID_ARMCLANG"}
_FIELDS = {"isa": "ARM_NN_BUILD_ISA", "toolchain": "ARM_NN_BUILD_TOOLCHAIN", "stamp": "ARM_NN_BUILD_STAMP",
           **{f"has_{token}": f"ARM_NN_BUILD_HAS_{token.upper()}" for token in voc.BASE_FEATURES}}
_PROBE_LINE = re.compile(r"^arm_nn_probe (\w+) (.*)$")
_STAMP_RE = re.compile(r"^arm_nn_build_(scalar|dsp|mve)_av([01])_sr([01])_(gcc|armclang)$")


@dataclass(frozen=True)
class BuildTarget:
    """What a compiler and its flags build: the planner Target and toolchain, the manifest feature tokens that hold
    (positive tokens of ``vocabulary.BASE_FEATURES``) and the library stamp such a build defines."""

    target: Target
    toolchain: str
    features: frozenset[str]
    stamp: str


def stamp_name(target: Target, toolchain: str) -> str:
    """The stamp symbol a library of this build defines (Include/arm_nn_build_target.h ARM_NN_BUILD_STAMP_OF)."""
    if toolchain not in TOOLCHAINS:
        raise ValueError(f"toolchain {toolchain!r} is not one of {sorted(TOOLCHAINS)}")
    return (f"arm_nn_build_{target.isa}_av{int(target.autovectorize)}_sr{int(target.single_rounding)}_"
            f"{toolchain}")


def _probe_source(header: Path) -> str:
    # By absolute path: a -I in the caller's flags for another ns-cmsis-nn copy must not supply the header. The
    # header's own includes are quoted, so they resolve next to it, inside the same checkout.
    lines = [f'#include "{header.as_posix()}"']
    lines += [f"arm_nn_probe {name} {macro}" for name, macro in _FIELDS.items()]
    return "\n".join(lines) + "\n"


def _parse(output: str, command: str) -> BuildTarget:
    values: dict[str, str] = {}
    for line in output.splitlines():
        match = _PROBE_LINE.match(line.strip())
        if not match:
            continue
        name, value = match.group(1), match.group(2).strip()
        if name not in _FIELDS:
            raise BuildTargetError(f"{command}: unexpected probe field {name!r}")
        if name in values:
            raise BuildTargetError(f"{command}: probe field {name!r} appears twice")
        values[name] = value
    missing = sorted(set(_FIELDS) - set(values))
    if missing:
        raise BuildTargetError(f"{command}: the probe printed no value for {missing}")

    def flag(name: str, allowed: Iterable[int]) -> int:
        value = values[name]
        if not re.fullmatch(r"[0-9]+", value) or int(value) not in allowed:
            raise BuildTargetError(f"{command}: {_FIELDS[name]} is {value!r}, not one of {sorted(allowed)}; the "
                                   "checkout's arm_nn_build_target.h does not define it as this package reads it")
        return int(value)

    isa = _ISA_NAMES[flag("isa", _ISA_NAMES)]
    toolchain = _TOOLCHAIN_NAMES[flag("toolchain", _TOOLCHAIN_NAMES)]
    features = frozenset(t for t in voc.BASE_FEATURES if flag(f"has_{t}", (0, 1)))
    try:
        target = Target(isa, autovectorize="autovectorize" in features,
                        single_rounding="single_rounding" in features)
    except ValueError as error:
        raise BuildTargetError(f"{command}: builds a target the planner does not model ({error})") from None
    stamp = values["stamp"]
    if stamp != stamp_name(target, toolchain):
        raise BuildTargetError(f"{command}: ARM_NN_BUILD_STAMP is {stamp!r}, expected {stamp_name(target, toolchain)!r} "
                               "for the fields it printed")
    return BuildTarget(target=target, toolchain=toolchain, features=features, stamp=stamp)


def probe_compiler(root: Path, cc: Union[str, Sequence[str]], flags: Sequence[str] = (), *,
                   timeout: float = 30.0) -> BuildTarget:
    """The build ``cc`` with ``flags`` produces for the ns-cmsis-nn checkout ``root``.

    ``cc`` is the compiler (a path, or a command such as ``["ccache", "arm-none-eabi-gcc"]``) and ``flags`` the exact
    flags the firmware's ns-cmsis-nn and model sources are compiled with (CPU, float ABI and every ``-D``). Raises
    BuildTargetError when the compiler cannot be run, fails, times out, or prints anything this package cannot read.
    """
    command = [cc] if isinstance(cc, str) else list(cc)
    if not command or not all(isinstance(part, str) and part for part in command):
        raise ValueError("cc must be a non-empty command")
    if isinstance(flags, str) or not all(isinstance(flag, str) for flag in flags):
        raise ValueError("flags must be a sequence of strings")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    include = (Path(root) / "Include").resolve()
    header = (Path(root) / HEADER).resolve()
    if not header.is_file():
        raise BuildTargetError(f"{root}: has no {HEADER}; the checkout predates the build-target check")
    if '"' in header.as_posix():
        raise BuildTargetError(f"{header}: a path with a double quote cannot be #included")
    with tempfile.TemporaryDirectory(prefix="arm_nn_probe_") as scratch:
        source = Path(scratch) / "arm_nn_build_probe.c"
        source.write_text(_probe_source(header), encoding="utf-8")
        argv = [*command, f"-I{include}", *flags, "-E", "-P", str(source)]
        shown = " ".join([*command, *flags])
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        except FileNotFoundError:
            raise BuildTargetError(f"{command[0]}: compiler not found") from None
        except subprocess.TimeoutExpired:
            raise BuildTargetError(f"{shown}: did not finish within {timeout:g} s") from None
        except OSError as error:
            raise BuildTargetError(f"{command[0]}: cannot be run ({error})") from None
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()[-5:]
        raise BuildTargetError(f"{shown}: exited with {result.returncode}: " + " | ".join(detail))
    return _parse(result.stdout, shown)


def _tokens(implementation: Implementation) -> frozenset[str]:
    routed = implementation.record.get("routed_by")
    route_requires = routed.get("requires", []) if isinstance(routed, dict) else []
    tokens = frozenset(implementation.requires) | frozenset(route_requires)
    unknown = sorted(t for t in tokens if t.lstrip("!") not in voc.BASE_FEATURES)
    if unknown:
        raise ContractError(f"{implementation.id}: unknown feature token(s) {unknown}")
    return tokens


def unmet(implementation: Implementation, features: Iterable[str]) -> tuple[str, ...]:
    """The tokens of the implementation's ``requires`` and its route's ``requires`` that a build with ``features``
    (BuildTarget.features) does not satisfy, sorted; empty when the build can run it."""
    features = frozenset(features)
    unknown = sorted(features - set(voc.BASE_FEATURES))
    if unknown:
        raise ValueError(f"features {unknown} are not manifest feature tokens")
    return tuple(sorted(t for t in _tokens(implementation) if (t.lstrip("!") in features) == t.startswith("!")))


def _target_features(target: Target) -> dict[str, bool]:
    """The tokens a Target alone decides; the float tokens are left to the compiler."""
    return {"mvei": target.isa == "mve", "dsp": target.isa in ("dsp", "mve"),
            "autovectorize": target.autovectorize, "single_rounding": target.single_rounding}


def _describe(target: Target) -> str:
    return "+".join([target.isa] + [name for name in ("autovectorize", "single_rounding") if getattr(target, name)])


def render_guard(target: Target, toolchain: str, implementations: Sequence[Implementation] = ()) -> str:
    """C for generated code that stops a build differing from the plan: the ARM_NN_EXPECT_* defines, the include of
    arm_nn_build_target_check.h, and an #error for each ``requires`` token of ``implementations`` the build lacks.

    Raises BuildTargetError when ``target`` itself cannot run one of the implementations, since that guard could never
    compile.
    """
    if not isinstance(target, Target):
        raise TypeError("target must be a Target")
    if toolchain not in TOOLCHAINS:
        raise ValueError(f"toolchain {toolchain!r} is not one of {sorted(TOOLCHAINS)}")
    if isinstance(implementations, Implementation) or not all(isinstance(i, Implementation) for i in implementations):
        raise TypeError("implementations must be a sequence of Implementation")
    decided = _target_features(target)
    checks: dict[tuple[str, str], None] = {}
    for impl in sorted({i.id: i for i in implementations}.values(), key=lambda i: i.id):
        for token in sorted(_tokens(impl)):
            base = token.lstrip("!")
            if base in decided and decided[base] == token.startswith("!"):
                raise BuildTargetError(f"{impl.id} requires {token}, which the planned build ({_describe(target)}, "
                                       f"{toolchain}) lacks")
            checks[(impl.id, token)] = None
    lines = [
        "/* ns-cmsis-nn build check (ns_cmsis_nn_contracts.render_guard) */",
        f"#define ARM_NN_EXPECT_ISA {_ISA_MACROS[target.isa]}",
        f"#define ARM_NN_EXPECT_AUTOVECTORIZE {int(target.autovectorize)}",
        f"#define ARM_NN_EXPECT_SINGLE_ROUNDING {int(target.single_rounding)}",
        f"#define ARM_NN_EXPECT_TOOLCHAIN {_TOOLCHAIN_MACROS[toolchain]}",
        '#include "arm_nn_build_target_check.h"',
    ]
    for impl_id, token in checks:
        macro = f"ARM_NN_BUILD_HAS_{token.lstrip('!').upper()}"
        lines += [f"#if {macro}" if token.startswith("!") else f"#if !{macro}",
                  f'    #error "ns-cmsis-nn build check: {impl_id} requires {token}"', "#endif"]
    return "\n".join(lines) + "\n"

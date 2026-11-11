# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""The signature export, ``Tests/KernelContracts/kernel_contracts.json`` (schema ``ns-cmsis-nn/kernel-contracts/1``).

One record per public function: return type, the ``#if`` conditions it sits under and each parameter's C type, array
extent and direction. Fail-closed: a checkout without the file loads as an explicit ``absent`` set; a file that is
present but unreadable, of another schema, internally inconsistent, or naming a symbol its header does not declare
raises ContractError.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from .errors import ContractError, ContractSchemaError

SCHEMA = "ns-cmsis-nn/kernel-contracts/1"
RELPATH = Path("Tests") / "KernelContracts" / "kernel_contracts.json"
DIRECTIONS = frozenset({"in", "out", "in,out"})
STATUS_PRESENT = "present"
STATUS_ABSENT = "absent"

_SCALAR_TYPES = frozenset({
    "void", "bool", "char", "int", "size_t",
    "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "float", "double", "float32_t", "float16_t",
})
_TYPEDEF_RE = re.compile(r"^(cmsis_nn_|arm_nn_|arm_cmsis_nn_)\w+$")
_TYPE_RE = re.compile(r"^(?:const )?(?P<base>\w+)(?: const)?(?: \*(?: const)?)*$")
_FUNCTION_POINTER_RE = re.compile(r"^.+\(\s*\*\s*\w*\s*\)\s*\(.*\)$")
_PUBLIC_HEADER_RE = re.compile(r"^Include/arm_nn[^/]*functions[^/]*\.h$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_]\w*$")
_EXTENT_RE = re.compile(r"^(\[[^\]]*\])+$")
_COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)


def type_is_known(c_type: str) -> bool:
    if _FUNCTION_POINTER_RE.match(c_type):
        return True
    match = _TYPE_RE.match(c_type)
    if match is None:
        return False
    base = match.group("base")
    return base in _SCALAR_TYPES or bool(_TYPEDEF_RE.match(base))


@dataclass(frozen=True)
class ParamDecl:
    name: str
    c_type: str
    direction: str
    extent: str = ""

    @property
    def is_pointer(self) -> bool:
        return "*" in self.c_type or bool(self.extent)


@dataclass(frozen=True)
class FunctionDecl:
    name: str
    header: str
    line: int
    guards: tuple[str, ...]
    returns: str
    params: tuple[ParamDecl, ...]

    def param(self, name: str) -> ParamDecl:
        for param in self.params:
            if param.name == name:
                return param
        raise KeyError(f"{self.name} has no parameter {name!r}")


@dataclass(frozen=True)
class SignatureSet:
    status: str
    root: Optional[Path]
    path: Optional[Path]
    functions: Mapping[str, FunctionDecl] = field(default_factory=dict)
    sha256: Optional[str] = None

    @property
    def present(self) -> bool:
        return self.status == STATUS_PRESENT

    def find(self, symbol: str) -> Optional[FunctionDecl]:
        return self.functions.get(symbol)

    def require(self, symbol: str) -> FunctionDecl:
        decl = self.find(symbol)
        if decl is None:
            where = str(self.path) if self.present else f"no {RELPATH} in this checkout"
            raise ContractError(f"{symbol}: not in the signature export ({where})")
        return decl


def _fail(path: Path, message: str) -> ContractError:
    return ContractError(f"{path}: {message}")


def _parse_param(path: Path, function: str, raw: object) -> ParamDecl:
    if not isinstance(raw, dict):
        raise _fail(path, f"{function}: parameter record is not an object: {raw!r}")
    unknown = set(raw) - {"name", "type", "direction", "extent"}
    if unknown:
        raise _fail(path, f"{function}: parameter has unknown keys {sorted(unknown)}")
    name, c_type, direction = raw.get("name"), raw.get("type"), raw.get("direction")
    extent = raw.get("extent", "")
    if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
        raise _fail(path, f"{function}: parameter name {name!r} is not an identifier")
    if not isinstance(c_type, str) or not type_is_known(c_type):
        raise _fail(path, f"{function}.{name}: type {c_type!r} is missing or outside the known vocabulary")
    if direction not in DIRECTIONS:
        raise _fail(path, f"{function}.{name}: direction {direction!r} not in {sorted(DIRECTIONS)}")
    if not isinstance(extent, str) or (extent and not _EXTENT_RE.match(extent)):
        raise _fail(path, f"{function}.{name}: extent {extent!r} is not an array extent")
    return ParamDecl(name=name, c_type=c_type, direction=direction, extent=extent)


def _parse_function(path: Path, raw: object) -> FunctionDecl:
    if not isinstance(raw, dict):
        raise _fail(path, f"function record is not an object: {raw!r}")
    keys = {"name", "header", "line", "guards", "returns", "params"}
    if set(raw) != keys:
        raise _fail(path, f"{raw.get('name', '<unnamed>')}: keys {sorted(raw)} are not {sorted(keys)}")
    name = raw["name"]
    if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
        raise _fail(path, f"function name {name!r} is not an identifier")
    header = raw["header"]
    if not isinstance(header, str) or not _PUBLIC_HEADER_RE.match(header):
        raise _fail(path, f"{name}: header {header!r} is not a public functions header")
    line = raw["line"]
    if not isinstance(line, int) or isinstance(line, bool) or line < 1:
        raise _fail(path, f"{name}: line {line!r} is not a positive integer")
    guards = raw["guards"]
    if not isinstance(guards, list) or not all(isinstance(g, str) and g for g in guards):
        raise _fail(path, f"{name}: guards {guards!r} is not a list of conditions")
    returns = raw["returns"]
    if not isinstance(returns, str) or not type_is_known(returns):
        raise _fail(path, f"{name}: return type {returns!r} is missing or outside the known vocabulary")
    if not isinstance(raw["params"], list):
        raise _fail(path, f"{name}: params is not a list")
    params = tuple(_parse_param(path, name, item) for item in raw["params"])
    names = [param.name for param in params]
    if len(set(names)) != len(names):
        raise _fail(path, f"{name}: duplicate parameter names {names}")
    return FunctionDecl(name=name, header=header, line=line, guards=tuple(guards), returns=returns, params=params)


def _cross_check_headers(root: Path, path: Path, functions: Mapping[str, FunctionDecl]) -> None:
    """Every symbol must be declared in the header the export names, or the export is stale for this checkout."""
    by_header: dict[str, list[str]] = {}
    for decl in functions.values():
        by_header.setdefault(decl.header, []).append(decl.name)
    for header, symbols in sorted(by_header.items()):
        try:
            text = _COMMENT_RE.sub(" ", (root / header).read_text(encoding="utf-8"))
        except OSError as error:
            raise _fail(path, f"names {header}, which cannot be read in {root} ({error})") from None
        missing = [s for s in symbols if not re.search(rf"\b{re.escape(s)}\s*\(", text)]
        if missing:
            shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
            raise _fail(path, f"{len(missing)} symbol(s) not declared in {header}: {shown}; the export is stale "
                              "for this checkout (python3 scripts/check_kernel_contract.py export)")


def load_signatures(root: Optional[Path]) -> SignatureSet:
    """The checkout's signature export, or an ``absent`` set when the checkout has none."""
    if root is None:
        return SignatureSet(status=STATUS_ABSENT, root=None, path=None)
    root = Path(root)
    path = root / RELPATH
    if not path.exists():
        return SignatureSet(status=STATUS_ABSENT, root=root, path=path)
    try:
        raw = path.read_bytes()
        document = json.loads(raw.decode("utf-8"))
    except OSError as error:
        raise _fail(path, f"cannot be read ({error})") from None
    except ValueError as error:
        raise _fail(path, f"is not valid JSON ({error})") from None
    if not isinstance(document, dict) or set(document) != {"schema", "functions"}:
        raise _fail(path, "top level must be an object with exactly 'schema' and 'functions'")
    if document["schema"] != SCHEMA:
        raise ContractSchemaError(f"{path}: schema {document['schema']!r} is not {SCHEMA!r}; update "
                                  "ns_cmsis_nn_contracts or the export")
    if not isinstance(document["functions"], list) or not document["functions"]:
        raise _fail(path, "has no functions; an export never writes an empty contract")
    functions: dict[str, FunctionDecl] = {}
    for item in document["functions"]:
        decl = _parse_function(path, item)
        if decl.name in functions:
            raise _fail(path, f"{decl.name} appears twice")
        functions[decl.name] = decl
    _cross_check_headers(root, path, functions)
    return SignatureSet(status=STATUS_PRESENT, root=root, path=path, functions=functions,
                        sha256=hashlib.sha256(raw).hexdigest())

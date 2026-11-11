# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""The implementation manifest, ``Tests/KernelContracts/kernel_implementations.json`` (schema
``ns-cmsis-nn/kernel-implementations/1``): per kernel implementation, its entry, the wrapper route that reaches it, the
build features it needs, its buffers and their size queries, its preparation steps, when it declines and its numerics
claim. The producer (scripts/check_kernel_implementations.py) validates the authored YAML in depth; this loader
re-checks what a consumer relies on and binds the manifest to the signature export it was generated against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from . import vocabulary as voc
from .errors import ContractError, ContractSchemaError
from .signatures import SignatureSet

RELPATH = Path("Tests") / "KernelContracts" / "kernel_implementations.json"


@dataclass(frozen=True)
class Buffer:
    name: str
    param: Optional[str]
    size_query: Optional[str]
    size_args: Mapping[str, str]
    lifetime: str
    align: int
    optional: bool
    format: Optional[str]


@dataclass(frozen=True)
class PrepareStep:
    symbol: str
    fills: Optional[str]
    consumes: tuple[str, ...]


@dataclass(frozen=True)
class Implementation:
    id: str
    family: str
    op: str
    entry: str
    wrapper: Optional[str]
    route: Optional[str]
    requires: frozenset[str]
    buffers: tuple[Buffer, ...]
    prepare: tuple[PrepareStep, ...]
    since: str
    numerics_claim: str
    numerics_reference: str
    eligibility_predicate: Optional[str]
    record: Mapping[str, Any] = field(repr=False, compare=False, default_factory=dict)

    def buffer(self, name: str) -> Buffer:
        for buffer in self.buffers:
            if buffer.name == name:
                return buffer
        raise KeyError(f"{self.id} has no buffer {name!r}")


@dataclass(frozen=True)
class ImplementationSet:
    path: Path
    library_version: str
    planner_abi: Optional[int]
    implementations: Mapping[str, Implementation]

    def require(self, implementation_id: str) -> Implementation:
        try:
            return self.implementations[implementation_id]
        except KeyError:
            raise ContractError(f"{implementation_id}: no such implementation in {self.path}") from None

    def by_route(self, wrapper: str, route: str) -> Implementation:
        """The implementation the wrapper's route reaches."""
        for impl in self.implementations.values():
            if impl.wrapper == wrapper and impl.route == route:
                return impl
        raise ContractError(f"{wrapper} route {route!r}: no implementation in {self.path}")

    def by_direct_entry(self, entry: str) -> Implementation:
        """The implementation a caller reaches by calling ``entry`` itself."""
        matches = [impl for impl in self.implementations.values() if impl.wrapper is None and impl.entry == entry]
        if len(matches) != 1:
            raise ContractError(f"{entry}: {len(matches)} direct implementations in {self.path}, expected 1")
        return matches[0]


def _fail(path: Path, message: str) -> ContractError:
    return ContractError(f"{path}: {message}")


def _keys(path: Path, where: str, value: object, keys: frozenset[str]) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        got = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise _fail(path, f"{where}: keys {got} are not {sorted(keys)}")
    return value


def _symbol(path: Path, where: str, signatures: SignatureSet, symbol: object) -> str:
    if not isinstance(symbol, str) or signatures.find(symbol) is None:
        raise _fail(path, f"{where}: {symbol!r} is not a public function in the signature export")
    return symbol


def _features(path: Path, where: str, tokens: object) -> frozenset[str]:
    if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
        raise _fail(path, f"{where}: requires must be a list of feature tokens")
    unknown = sorted(set(tokens) - voc.FEATURES)
    if unknown:
        raise _fail(path, f"{where}: unknown feature token(s) {unknown}")
    return frozenset(tokens)


def _parse(path: Path, raw: object, signatures: SignatureSet) -> Implementation:
    record = _keys(path, "implementation", raw, voc.EXPORT_RECORD_KEYS)
    impl_id = record["id"]
    if not isinstance(impl_id, str) or not voc.ID_RE.match(impl_id):
        raise _fail(path, f"implementation id {impl_id!r} is not a lower_snake_case identifier")
    where = impl_id
    if record["op"] not in voc.OPS:
        raise _fail(path, f"{where}: op {record['op']!r} is not one of {sorted(voc.OPS)}")
    entry = _symbol(path, f"{where}.entry", signatures, record["entry"])
    routed = record["routed_by"]
    wrapper = route = None
    if routed != "direct":
        routed = _keys(path, f"{where}.routed_by", routed, voc.EXPORT_ROUTE_KEYS)
        wrapper = _symbol(path, f"{where}.routed_by.wrapper", signatures, routed["wrapper"])
        route = routed["route"]
        if not isinstance(route, str) or not voc.ROUTE_RE.match(route):
            raise _fail(path, f"{where}: route {route!r} is not '<wrapper>:<route>'")
        _features(path, f"{where}.routed_by", routed["requires"])
    requires = _features(path, where, record["requires"])
    buffers = []
    for index, raw_buffer in enumerate(record["buffers"] if isinstance(record["buffers"], list) else [None]):
        here = f"{where}.buffers[{index}]"
        b = _keys(path, here, raw_buffer, voc.EXPORT_BUFFER_KEYS)
        query = None
        if b["size_query"] is not None:
            query = _keys(path, f"{here}.size_query", b["size_query"], voc.EXPORT_QUERY_KEYS)
            if not isinstance(query["args"], dict):
                raise _fail(path, f"{here}.size_query: args is not an object")
        if b["lifetime"] not in voc.LIFETIMES:
            raise _fail(path, f"{here}: lifetime {b['lifetime']!r} is not one of {sorted(voc.LIFETIMES)}")
        if not isinstance(b["align"], int) or isinstance(b["align"], bool) or b["align"] < 1:
            raise _fail(path, f"{here}: align {b['align']!r} is not a positive integer")
        if b["param"] is not None and b["param"] not in {p.name for p in signatures.require(entry).params}:
            raise _fail(path, f"{here}: {entry} has no parameter {b['param']!r}")
        buffers.append(Buffer(name=b["name"], param=b["param"],
                              size_query=None if query is None else _symbol(path, f"{here}.size_query", signatures,
                                                                            query["symbol"]),
                              size_args={} if query is None else dict(query["args"]),
                              lifetime=b["lifetime"], align=b["align"], optional=bool(b["optional"]),
                              format=b["format"]))
    names = [b.name for b in buffers]
    if len(set(names)) != len(names):
        raise _fail(path, f"{where}: duplicate buffer names {names}")
    steps = []
    for index, raw_step in enumerate(record["prepare"] if isinstance(record["prepare"], list) else [None]):
        here = f"{where}.prepare[{index}]"
        s = _keys(path, here, raw_step, voc.EXPORT_STEP_KEYS)
        if s["fills"] is not None and s["fills"] not in names:
            raise _fail(path, f"{here}: fills {s['fills']!r}, which is not one of its buffers {names}")
        steps.append(PrepareStep(symbol=_symbol(path, here, signatures, s["symbol"]), fills=s["fills"],
                                 consumes=tuple(s["consumes"])))
    predicate = None
    if record["eligibility"] is not None:
        elig = _keys(path, f"{where}.eligibility", record["eligibility"], voc.EXPORT_ELIGIBILITY_KEYS)
        predicate = elig["predicate"]
        if predicate is not None and not elig["predicate_static"]:
            _symbol(path, f"{where}.eligibility", signatures, predicate)
    numerics = _keys(path, f"{where}.numerics", record["numerics"], voc.EXPORT_NUMERICS_KEYS)
    if numerics["claim"] not in voc.CLAIMS:
        raise _fail(path, f"{where}: numerics claim {numerics['claim']!r} is not one of {sorted(voc.CLAIMS)}")
    if not isinstance(record["since"], str) or not voc.VERSION_RE.match(record["since"]):
        raise _fail(path, f"{where}: since {record['since']!r} is not a MAJOR.MINOR.PATCH version")
    return Implementation(id=impl_id, family=record["family"], op=record["op"], entry=entry, wrapper=wrapper,
                          route=route, requires=requires, buffers=tuple(buffers), prepare=tuple(steps),
                          since=record["since"], numerics_claim=numerics["claim"],
                          numerics_reference=numerics["reference"], eligibility_predicate=predicate, record=record)


def load_implementations(root: Path, signatures: SignatureSet) -> ImplementationSet:
    """The checkout's implementation manifest, checked against ``signatures`` (the same checkout's export)."""
    path = Path(root) / RELPATH
    if not signatures.present:
        raise _fail(path, "needs the signature export it was generated against, and the checkout has none")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise _fail(path, f"cannot be read ({error})") from None
    except ValueError as error:
        raise _fail(path, f"is not valid JSON ({error})") from None
    document = _keys(path, "manifest", document, voc.EXPORT_ENVELOPE_KEYS)
    if document["schema"] != voc.SCHEMA:
        raise ContractSchemaError(f"{path}: schema {document['schema']!r} is not {voc.SCHEMA!r}")
    if document["signature_sha256"] != signatures.sha256:
        raise _fail(path, f"was generated against a signature export with sha256 {document['signature_sha256']}, "
                          f"not {signatures.path} ({signatures.sha256}); the manifest is stale "
                          "(python3 scripts/check_kernel_implementations.py export)")
    version = document["library_version"]
    if not isinstance(version, str) or not voc.VERSION_RE.match(version):
        raise _fail(path, f"library_version {version!r} is not a MAJOR.MINOR.PATCH version")
    abi = document["planner_abi"]
    if abi is not None and (not isinstance(abi, int) or isinstance(abi, bool) or abi < 1):
        raise _fail(path, f"planner_abi {abi!r} is neither null nor a positive integer")
    if not isinstance(document["implementations"], list) or not document["implementations"]:
        raise _fail(path, "has no implementations")
    implementations: dict[str, Implementation] = {}
    for raw in document["implementations"]:
        impl = _parse(path, raw, signatures)
        if impl.id in implementations:
            raise _fail(path, f"implementation {impl.id} appears twice")
        implementations[impl.id] = impl
    routes = [(i.wrapper, i.route) for i in implementations.values() if i.wrapper is not None]
    if len(set(routes)) != len(routes):
        raise _fail(path, "two implementations claim the same wrapper route")
    return ImplementationSet(path=path, library_version=version, planner_abi=abi, implementations=implementations)

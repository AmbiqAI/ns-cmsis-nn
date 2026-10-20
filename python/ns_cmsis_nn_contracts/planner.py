# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""The host planner library (Planner/) built from a checkout and called through ctypes.

``build_planner`` compiles ``Planner/sources.txt`` with the host C compiler for one target toolchain and caches the
shared library by the content of every input. ``Planner`` loads it, refuses a library whose ABI or toolchain is not
the one asked for, and returns plans that name the manifest implementation they reach.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .errors import PlannerError, PlanRefused
from .implementations import Implementation, ImplementationSet

ABI_VERSION = 1
TOOLCHAINS = {"gcc": 0, "armclang": 1}
ISAS = {"scalar": 0, "dsp": 1, "mve": 2}
CTX_ROLES = {0: "none", 1: "scratch", 2: "kernel_sums"}
STATUSES = {0: "ok", 1: "null", 2: "target", 3: "shape", 4: "range", 5: "unsupported"}
INT32_MAX = 2**31 - 1


class _Dims(ctypes.Structure):
    _fields_ = [("n", ctypes.c_int32), ("h", ctypes.c_int32), ("w", ctypes.c_int32), ("c", ctypes.c_int32)]


class _Tile(ctypes.Structure):
    _fields_ = [("w", ctypes.c_int32), ("h", ctypes.c_int32)]


class _Activation(ctypes.Structure):
    _fields_ = [("min", ctypes.c_int32), ("max", ctypes.c_int32)]


class _DwConvParams(ctypes.Structure):
    _fields_ = [("input_offset", ctypes.c_int32), ("output_offset", ctypes.c_int32), ("ch_mult", ctypes.c_int32),
                ("stride", _Tile), ("padding", _Tile), ("dilation", _Tile), ("activation", _Activation)]


class _PoolParams(ctypes.Structure):
    _fields_ = [("stride", _Tile), ("padding", _Tile), ("activation", _Activation)]


class _FcParams(ctypes.Structure):
    _fields_ = [("input_offset", ctypes.c_int32), ("filter_offset", ctypes.c_int32), ("output_offset", ctypes.c_int32),
                ("activation", _Activation)]


class _QuantParams(ctypes.Structure):
    _fields_ = [("multiplier", ctypes.POINTER(ctypes.c_int32)), ("shift", ctypes.POINTER(ctypes.c_int32)),
                ("is_per_channel", ctypes.c_int32)]


class _PerChannelQuant(ctypes.Structure):
    _fields_ = [("multiplier", ctypes.POINTER(ctypes.c_int32)), ("shift", ctypes.POINTER(ctypes.c_int32))]


class _Target(ctypes.Structure):
    _fields_ = [("isa", ctypes.c_int32), ("autovectorize", ctypes.c_int32), ("single_rounding", ctypes.c_int32),
                ("toolchain", ctypes.c_int32)]


class _Plan(ctypes.Structure):
    _fields_ = [("kernel", ctypes.c_char_p), ("route_id", ctypes.c_char_p), ("route", ctypes.c_int32),
                ("ctx_role", ctypes.c_int32), ("ctx_bytes", ctypes.c_int32), ("prepared_bytes", ctypes.c_int32)]


@dataclass(frozen=True)
class Target:
    """The build of the target library: isa is scalar, dsp or mve."""

    isa: str
    autovectorize: bool = False
    single_rounding: bool = False

    def __post_init__(self) -> None:
        if self.isa not in ISAS:
            raise ValueError(f"isa {self.isa!r} is not one of {sorted(ISAS)}")
        if self.autovectorize and self.isa != "mve":
            raise ValueError("autovectorize applies to an mve target only")


@dataclass(frozen=True)
class Plan:
    kernel: str
    route: Optional[str]
    implementation: Optional[Implementation]
    ctx_role: str
    ctx_bytes: int
    prepared_bytes: int


def _dims(value: Sequence[int], what: str) -> _Dims:
    if len(value) != 4 or not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        raise ValueError(f"{what} must be four integers (n, h, w, c), got {value!r}")
    if not all(-2**31 <= v <= INT32_MAX for v in value):
        raise ValueError(f"{what} {value!r} does not fit int32")
    return _Dims(*value)


def _int32_array(values: Sequence[int], what: str):
    if not all(-2**31 <= int(v) <= INT32_MAX for v in values):
        raise ValueError(f"{what} holds a value outside int32")
    return (ctypes.c_int32 * len(values))(*[int(v) for v in values])


def _int8_array(values, what: str):
    data = bytes(values) if isinstance(values, (bytes, bytearray, memoryview)) else None
    if data is None:
        if not all(-128 <= int(v) <= 127 for v in values):
            raise ValueError(f"{what} holds a value outside int8")
        data = bytes(int(v) & 0xFF for v in values)
    return (ctypes.c_int8 * len(data)).from_buffer_copy(data), len(data)


def _sources(root: Path) -> list[Path]:
    listing = root / "Planner" / "sources.txt"
    try:
        lines = listing.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise PlannerError(f"{listing}: cannot be read ({error}); the checkout has no planner") from None
    sources = [root / line.strip() for line in lines if line.strip() and not line.startswith("#")]
    missing = [str(s) for s in sources if not s.is_file()]
    if not sources or missing:
        raise PlannerError(f"{listing}: lists sources the checkout does not have: {missing or 'none listed'}")
    return sources


def build_planner(root: Path, toolchain: str = "gcc", cc: Optional[str] = None,
                  cache_dir: Optional[Path] = None) -> Path:
    """Build (or reuse) the planner library of the checkout ``root`` for ``toolchain``; return its path.

    The cache key covers the compiler, the flags and the bytes of every source and header the build can read, so a
    changed checkout never reuses a stale library. The library is written to a temporary name and renamed into place,
    so concurrent builders end with one complete file.
    """
    if toolchain not in TOOLCHAINS:
        raise ValueError(f"toolchain {toolchain!r} is not one of {sorted(TOOLCHAINS)}")
    root = Path(root).resolve()
    compiler = cc or os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if compiler is None:
        raise PlannerError("no host C compiler found (set CC)")
    sources = _sources(root)
    includes = [root / "Include", root / "Planner" / "include", root / "Planner" / "src"]
    flags = ["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared",
             f"-DARM_NN_TOOLCHAIN_ARMCLANG={TOOLCHAINS[toolchain]}", *[f"-I{i}" for i in includes]]
    digest = hashlib.sha256()
    try:
        version = subprocess.run([compiler, "--version"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise PlannerError(f"host compiler {compiler!r} does not run ({error})") from None
    digest.update(version.encode())
    digest.update("\0".join(flags).encode())
    inputs = sorted(set(sources) | {p for d in includes for p in d.rglob("*") if p.suffix in (".h", ".inc")})
    for path in inputs:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    cache = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "ns_cmsis_nn_plan"
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / f"libns_cmsis_nn_plan_{toolchain}_{digest.hexdigest()[:16]}{suffix}"
    if library.is_file():
        return library
    handle, name = tempfile.mkstemp(prefix=library.name + ".", suffix=".tmp", dir=cache)
    os.close(handle)
    partial = Path(name)
    result = subprocess.run([compiler, *flags, *map(str, sources), "-o", str(partial)], capture_output=True,
                            text=True)
    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        raise PlannerError(f"building the planner failed:\n{result.stderr.strip()}")
    os.replace(partial, library)
    return library


class Planner:
    """Plans for one target toolchain. ``implementations`` (the same checkout's manifest) resolves each plan to the
    implementation it reaches and pins the planner ABI the manifest was exported with."""

    def __init__(self, library: Path, toolchain: str = "gcc",
                 implementations: Optional[ImplementationSet] = None) -> None:
        if toolchain not in TOOLCHAINS:
            raise ValueError(f"toolchain {toolchain!r} is not one of {sorted(TOOLCHAINS)}")
        try:
            self._lib = ctypes.CDLL(str(library))
        except OSError as error:
            raise PlannerError(f"{library}: cannot be loaded ({error})") from None
        self._bind()
        abi = self._lib.arm_nn_plan_abi_version()
        if abi != ABI_VERSION:
            raise PlannerError(f"{library}: planner ABI {abi}, this package reads {ABI_VERSION}")
        if implementations is not None and implementations.planner_abi != abi:
            raise PlannerError(f"{implementations.path} was exported for planner ABI {implementations.planner_abi}, "
                               f"the library is {abi}")
        built = self._lib.arm_nn_plan_toolchain_built()
        if built != TOOLCHAINS[toolchain]:
            raise PlannerError(f"{library}: built for toolchain {built}, asked for {toolchain}")
        self.toolchain = toolchain
        self.implementations = implementations

    def _bind(self) -> None:
        lib = self._lib
        lib.arm_nn_plan_abi_version.restype = ctypes.c_int32
        lib.arm_nn_plan_toolchain_built.restype = ctypes.c_int32
        ptr = ctypes.POINTER
        for name, args in {
            "arm_nn_plan_depthwise_conv_wrapper_s8": [ptr(_Target), ptr(_DwConvParams), ptr(_Dims), ptr(_Dims),
                                                      ptr(_Dims), ptr(_Plan)],
            "arm_nn_plan_fully_connected_wrapper_s8": [ptr(_Target), ptr(_QuantParams), ptr(_Dims), ptr(_Dims),
                                                       ptr(_Dims), ptr(_Plan)],
            "arm_nn_plan_fully_connected_per_channel_packed_s8": [ptr(_Target), ptr(_FcParams), ptr(_PerChannelQuant),
                                                                  ptr(_Dims), ptr(_Plan)],
            "arm_nn_plan_pool_s8": [ptr(_Target), ctypes.c_int32, ptr(_PoolParams), ptr(_Dims), ptr(_Dims),
                                    ptr(_Dims), ptr(_Plan)],
            "arm_vector_sum_s8": [ptr(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32, ptr(ctypes.c_int8),
                                  ctypes.c_int32, ctypes.c_int32, ptr(ctypes.c_int32)],
            "arm_fully_connected_per_channel_packed_s8_pack": [ptr(_Dims), ptr(ctypes.c_int8), ptr(ctypes.c_int32),
                                                               ptr(_PerChannelQuant), ptr(ctypes.c_int8)],
            "arm_fully_connected_per_channel_packed_s8_get_packed_size": [ptr(_Dims)],
        }.items():
            fn = getattr(lib, name)
            fn.argtypes = args
            fn.restype = ctypes.c_int32

    def _target(self, target: Target) -> _Target:
        return _Target(ISAS[target.isa], int(target.autovectorize), int(target.single_rounding),
                       TOOLCHAINS[self.toolchain])

    def _result(self, status: int, plan: _Plan, what: str, wrapper: Optional[str]) -> Plan:
        if status != 0:
            name = STATUSES.get(status, f"status {status}")
            raise PlanRefused(name, f"{what}: the planner refused the layer ({name})")
        kernel = plan.kernel.decode()
        route = plan.route_id.decode() if plan.route_id else None
        impl = None
        if self.implementations is not None:
            impl = (self.implementations.by_route(wrapper, route) if route is not None
                    else self.implementations.by_direct_entry(kernel))
        return Plan(kernel=kernel, route=route, implementation=impl, ctx_role=CTX_ROLES[plan.ctx_role],
                    ctx_bytes=plan.ctx_bytes, prepared_bytes=plan.prepared_bytes)

    def depthwise_conv_wrapper_s8(self, target: Target, *, input_dims, filter_dims, output_dims, ch_mult: int,
                                  stride=(1, 1), padding=(0, 0), dilation=(1, 1)) -> Plan:
        """stride, padding and dilation are (h, w)."""
        params = _DwConvParams(0, 0, ch_mult, _Tile(stride[1], stride[0]), _Tile(padding[1], padding[0]),
                               _Tile(dilation[1], dilation[0]), _Activation(-128, 127))
        plan = _Plan()
        status = self._lib.arm_nn_plan_depthwise_conv_wrapper_s8(
            self._target(target), params, _dims(input_dims, "input_dims"), _dims(filter_dims, "filter_dims"),
            _dims(output_dims, "output_dims"), plan)
        return self._result(status, plan, "arm_depthwise_conv_wrapper_s8", "arm_depthwise_conv_wrapper_s8")

    def fully_connected_wrapper_s8(self, target: Target, *, input_dims, filter_dims, output_dims,
                                   per_channel: bool) -> Plan:
        """filter_dims is (accumulation depth, 1, 1, output depth)."""
        one = (ctypes.c_int32 * 1)(0)
        plan = _Plan()
        status = self._lib.arm_nn_plan_fully_connected_wrapper_s8(
            self._target(target), _QuantParams(one, one, int(bool(per_channel))), _dims(input_dims, "input_dims"),
            _dims(filter_dims, "filter_dims"), _dims(output_dims, "output_dims"), plan)
        return self._result(status, plan, "arm_fully_connected_wrapper_s8", "arm_fully_connected_wrapper_s8")

    def fully_connected_per_channel_packed_s8(self, target: Target, *, filter_dims, filter_offset: int,
                                              multipliers: Sequence[int], shifts: Sequence[int]) -> Plan:
        """filter_dims is (accumulation depth, 1, 1, output depth); one multiplier and shift per output channel."""
        dims = _dims(filter_dims, "filter_dims")
        if len(multipliers) != dims.c or len(shifts) != dims.c:
            raise ValueError(f"need {dims.c} multipliers and shifts, got {len(multipliers)} and {len(shifts)}")
        mult, shift = _int32_array(multipliers, "multipliers"), _int32_array(shifts, "shifts")
        plan = _Plan()
        status = self._lib.arm_nn_plan_fully_connected_per_channel_packed_s8(
            self._target(target), _FcParams(0, filter_offset, 0, _Activation(-128, 127)),
            _PerChannelQuant(mult, shift), dims, plan)
        return self._result(status, plan, "arm_fully_connected_per_channel_packed_s8", None)

    def pool_s8(self, target: Target, *, max_pool: bool, input_dims, kernel_hw, output_dims, stride=(1, 1),
                padding=(0, 0)) -> Plan:
        """kernel_hw, stride and padding are (h, w)."""
        params = _PoolParams(_Tile(stride[1], stride[0]), _Tile(padding[1], padding[0]), _Activation(-128, 127))
        plan = _Plan()
        status = self._lib.arm_nn_plan_pool_s8(
            self._target(target), int(max_pool), params, _dims(input_dims, "input_dims"),
            _dims((1, kernel_hw[0], kernel_hw[1], 1), "kernel_hw"), _dims(output_dims, "output_dims"), plan)
        return self._result(status, plan, "arm_max_pool_s8" if max_pool else "arm_avgpool_s8", None)

    def vector_sum_s8(self, data, *, cols: int, rows: int, lhs_offset: int, rhs_offset: int,
                      bias: Optional[Sequence[int]] = None) -> list[int]:
        """arm_vector_sum_s8: per row, bias + lhs_offset * sum(data + rhs_offset) (the library's own code)."""
        if cols < 1 or rows < 1:
            raise ValueError(f"cols and rows must be positive, got {cols} and {rows}")
        matrix, length = _int8_array(data, "data")
        if length != cols * rows:
            raise ValueError(f"data holds {length} values, cols * rows is {cols * rows}")
        if bias is not None and len(bias) != rows:
            raise ValueError(f"bias holds {len(bias)} values, rows is {rows}")
        out = (ctypes.c_int32 * rows)()
        status = self._lib.arm_vector_sum_s8(out, cols, rows, matrix, lhs_offset, rhs_offset,
                                             None if bias is None else _int32_array(bias, "bias"))
        if status != 0:
            raise PlanRefused("library", f"arm_vector_sum_s8 returned {status}")
        return list(out)

    def pack_fully_connected_per_channel_packed_s8(self, filter_data, *, filter_dims, kernel_sums: Sequence[int],
                                                   multipliers: Sequence[int], shifts: Sequence[int]) -> bytes:
        """The stream arm_fully_connected_per_channel_packed_s8_pack writes (filter [accumulation depth][output depth]
        as the kernel takes it)."""
        dims = _dims(filter_dims, "filter_dims")
        if dims.n < 1 or dims.c < 1:
            raise ValueError(f"filter_dims {filter_dims!r} must have positive depths")
        weights, length = _int8_array(filter_data, "filter_data")
        if length != dims.n * dims.c:
            raise ValueError(f"filter_data holds {length} values, the filter has {dims.n * dims.c}")
        for name, values in (("kernel_sums", kernel_sums), ("multipliers", multipliers), ("shifts", shifts)):
            if len(values) != dims.c:
                raise ValueError(f"{name} holds {len(values)} values, the output depth is {dims.c}")
        size = self._lib.arm_fully_connected_per_channel_packed_s8_get_packed_size(dims)
        if size <= 0:
            raise PlanRefused("range", f"no packed stream for filter_dims {filter_dims!r}")
        out = (ctypes.c_int8 * size)()
        status = self._lib.arm_fully_connected_per_channel_packed_s8_pack(
            dims, weights, _int32_array(kernel_sums, "kernel_sums"),
            _PerChannelQuant(_int32_array(multipliers, "multipliers"), _int32_array(shifts, "shifts")), out)
        if status != 0:
            raise PlanRefused("library", f"arm_fully_connected_per_channel_packed_s8_pack returned {status}")
        return bytes(out)

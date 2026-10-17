#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# One cell of the weekly compiler sweep (AmbiqAI/ns-cmsis-nn#340): build the
# library with one compiler, then generate, build and run the full
# helia-core-tester suite for one CPU against that archive.
#
#   compiler_sweep_cell.sh --toolchain gcc|atfe|armclang --cpu cortex-m55|cortex-m4 \
#                          --suite float|int --outdir <dir> \
#                          [--toolchain-root <dir>] [--expect-version <text>] \
#                          [--seed 500] [--run-jobs N] [--skip-generation]
#
# The tester always links its own pinned GNU harness, so every cell imports a
# prebuilt archive built with the shipped feature set (F32, plus F16 on
# Cortex-M55): only the library compiler varies between cells. gcc with no
# --toolchain-root builds with the arm-none-eabi-gcc on PATH; --expect-version
# fails the cell unless the compiler's --version output contains that text.
# armclang links its native math runtime through
# Tests/UnitTest/cmake/armclang_runtime.cmake (#857).
#
# Passes only if generation used --seed and dropped nothing, and the tester's
# report covers exactly the generated cases with every one passed.

set -euo pipefail

TOOLCHAIN=""
TOOLCHAIN_ROOT=""
CPU=""
SUITE=""
OUTDIR=""
SEED=500
RUN_JOBS=0
SKIP_GENERATION=0
EXPECT_VERSION=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --toolchain) TOOLCHAIN="${2:?}"; shift 2 ;;
    --toolchain-root) TOOLCHAIN_ROOT="${2-}"; shift 2 ;;
    --expect-version) EXPECT_VERSION="${2-}"; shift 2 ;;
    --cpu) CPU="${2:?}"; shift 2 ;;
    --suite) SUITE="${2:?}"; shift 2 ;;
    --outdir) OUTDIR="${2:?}"; shift 2 ;;
    --seed) SEED="${2:?}"; shift 2 ;;
    --run-jobs) RUN_JOBS="${2:?}"; shift 2 ;;
    --skip-generation) SKIP_GENERATION=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$TOOLCHAIN" in gcc|atfe|armclang) ;; *) echo "--toolchain must be gcc, atfe or armclang" >&2; exit 2 ;; esac
[[ "$TOOLCHAIN" != armclang || -n "$TOOLCHAIN_ROOT" ]] || { echo "armclang needs --toolchain-root" >&2; exit 2; }
case "$CPU" in cortex-m55|cortex-m4) ;; *) echo "--cpu must be cortex-m55 or cortex-m4" >&2; exit 2 ;; esac
case "$SUITE" in float|int) ;; *) echo "--suite must be float or int" >&2; exit 2 ;; esac
[[ -n "$OUTDIR" ]] || { echo "--outdir required" >&2; exit 2; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TESTER="$REPO/Tests/helia-core-tester"
mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"

# Shipped feature set, as release.yml builds it, whichever suite runs.
features=(--enable-f32)
f32=1; f16=0
if [[ "$CPU" == cortex-m55 ]]; then features+=(--enable-f16); f16=1; fi
precision=both
[[ "$CPU" == cortex-m55 ]] || precision=f32

# The compiler that will build the library, and proof of its identity. build_staticlib.sh
# builds gcc with whatever arm-none-eabi-gcc is first on PATH.
[[ "$TOOLCHAIN" != gcc || -z "$TOOLCHAIN_ROOT" ]] || export PATH="$TOOLCHAIN_ROOT/bin:$PATH"
case "$TOOLCHAIN" in
  gcc) cc=arm-none-eabi-gcc ;;
  atfe) cc="$TOOLCHAIN_ROOT/bin/clang" ;;
  armclang) cc="$TOOLCHAIN_ROOT/bin/armclang" ;;
esac
compiler="$("$cc" --version 2>&1 | grep -m1 -iE 'version|toolchain|gcc' || true)"
[[ -n "$compiler" ]] || { echo "cannot identify library compiler $cc" >&2; exit 1; }
if [[ -n "$EXPECT_VERSION" ]] && ! grep -qiF -- "$EXPECT_VERSION" <<< "$compiler"; then
  echo "library compiler is '$compiler', expected '$EXPECT_VERSION'" >&2
  exit 1
fi
echo "library compiler: $compiler"

# 1. Library archive with the cell's compiler, at the shipped optimization level.
bash "$REPO/scripts/build_staticlib.sh" --target-cpu "$CPU" --version 0.0.0 --outdir "$OUTDIR/lib" \
  --toolchain "$TOOLCHAIN" --toolchain-root "$TOOLCHAIN_ROOT" "${features[@]}" | tee "$OUTDIR/build_staticlib.log"
# The archive's assembler verdict (#427): a static-archive user must pass it, or the
# harness expects the other VCVT form's NaN behaviour.
gas_defines="$(sed -n 's/^>>> assembler verdict: //p' "$OUTDIR/build_staticlib.log" | tr ' ' '\n' | sed -n 's/^-D//p' | tr '\n' ' ')"
echo "assembler verdict: ${gas_defines:-none}"
# build_staticlib.sh's own naming, so a reused --outdir cannot hand over another compiler's archive.
if [[ "$TOOLCHAIN" == gcc ]]; then name="libns-cmsis-nn-$CPU-0.0.0.a"; else name="libns-cmsis-nn-$TOOLCHAIN-$CPU-0.0.0.a"; fi
lib="$OUTDIR/lib/$name"
[[ -f "$lib" ]] || { echo "build_staticlib.sh produced no $lib" >&2; exit 1; }

# 2. Import shim the tester's CMake consumes as CMSIS_NN_ROOT. The checkout's
#    Include, Source and Tests sit beside it for anything else the tester reads there.
mkdir -p "$OUTDIR/import"
for dir in Include Source Tests; do ln -sfn "$REPO/$dir" "$OUTDIR/import/$dir"; done
cat > "$OUTDIR/import/CMakeLists.txt" <<EOF
add_library(cmsis-nn STATIC IMPORTED GLOBAL)
set_target_properties(cmsis-nn PROPERTIES
  IMPORTED_LOCATION "$lib"
  INTERFACE_INCLUDE_DIRECTORIES "$REPO/Include")
target_compile_definitions(cmsis-nn INTERFACE ARM_NN_ENABLE_F32=$f32 ARM_NN_ENABLE_F16=$f16 $gas_defines)
EOF
# The armclang helper appends its own libm after the native objects; adding m
# here as well makes CMake repeat that whole sequence and duplicate the objects.
if [[ "$TOOLCHAIN" == armclang ]]; then
  cat >> "$OUTDIR/import/CMakeLists.txt" <<EOF
include("$REPO/Tests/UnitTest/cmake/armclang_runtime.cmake")
cmsis_nn_link_armclang_test_runtime(cmsis-nn "$TOOLCHAIN_ROOT")
EOF
else
  echo 'target_link_libraries(cmsis-nn INTERFACE m)' >> "$OUTDIR/import/CMakeLists.txt"
fi

# 3. Generate against this checkout. The tester reads UnitTest data and kernel
#    symbols from --cmsis-nn-root, so pointing generation at the shim drops cases.
if [[ "$SKIP_GENERATION" != 1 ]]; then
  (cd "$TESTER" && uv run helia_core_tester generate --seed "$SEED" --cpu "$CPU" --suite "$SUITE" \
    --float-precision "$precision" --verbosity 1)
fi

# 4. Build and run the generated cases against the archive.
started="$(date +%s)"
(cd "$TESTER" && uv run helia_core_tester full --skip-generation --cpu "$CPU" --suite "$SUITE" \
  --float-precision "$precision" --cmsis-nn-root "$OUTDIR/import" --run-jobs "$RUN_JOBS" \
  --no-fail-fast --report-formats json --verbosity 1) || status=$?

# 5. Exact coverage: generation dropped nothing, every generated case reported, every one passed.
python3 - "$TESTER" "$SUITE" "$CPU" "$lib" "${status:-0}" "$started" "$SEED" "$compiler" <<'EOF'
import hashlib, json, os, sys
from pathlib import Path

tester, suite, cpu, lib, status = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]), int(sys.argv[5])
started, seed, compiler = int(sys.argv[6]), int(sys.argv[7]), sys.argv[8]
generated = tester / "artifacts/generated_tests" / suite / cpu
names = {p.name for p in generated.glob("*/*") if p.is_dir()}
cases = len(names)
gen = json.loads((tester / "artifacts/reports/generation" / suite / cpu / "generation_summary.json").read_text())
dropped = {k: gen["counts"].get(k, 0) for k in ("skipped_kernel_symbol", "generation_failures", "conversion_failures")
           if gen["counts"].get(k, 0)}
if gen.get("filters", {}).get("seed") != seed:
    sys.exit(f"generated cases use seed {gen.get('filters', {}).get('seed')}, expected {seed}")
if gen.get("status") != "success" or dropped or not cases or gen["counts"].get("cases_total") != cases:
    sys.exit(f"generation incomplete: status {gen.get('status')}, {dropped}, "
             f"{gen['counts'].get('cases_total')} recorded vs {cases} generated")
reports = sorted((p for p in (tester / "artifacts/reports/tests" / suite / cpu).glob("test_report_*.json")
                  if os.path.getmtime(p) >= started), key=os.path.getmtime)
if not reports:
    sys.exit(f"no tester report for {suite}/{cpu} written by this run")
report = json.loads(reports[-1].read_text())
bad = {k: report.get(k, 0) for k in ("failed", "skipped", "timed_out", "errors", "not_run", "build_failed",
                                     "generation_failed", "conversion_failed") if report.get(k, 0)}
# Fingerprint of the generated cases themselves; the manifest also records reuse counts.
digest = hashlib.sha256("\n".join(f"{n}:{r.get('artifact_sha256')}" for n, r in sorted(report.get("descriptor_results", {}).items())).encode()).hexdigest()
line = (f"{cpu} {suite}: {report['passed']}/{report['total_tests']} passed, {cases} generated "
        f"({gen['counts'].get('skipped_capability', 0)} capability skips), "
        f"cases {digest[:16]}, archive {hashlib.sha256(lib.read_bytes()).hexdigest()[:16]}, "
        f"run {report.get('duration', 0):.0f}s, compiler {compiler}")
print(line)
if os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
        summary.write(f"- {line}\n")
results = {name: r.get("test_result", {}).get("status") for name, r in report.get("descriptor_results", {}).items()}
missing, extra = sorted(names - results.keys()), sorted(results.keys() - names)
not_passed = sorted(n for n, s in results.items() if s != "PASS")
if status or bad or missing or extra or not_passed or report["total_tests"] != cases or report["passed"] != cases:
    sys.exit(f"cell failed: tester exit {status}, counters {bad}, {report['passed']}/{report['total_tests']} of {cases}, "
             f"missing {missing[:5]}, unexpected {extra[:5]}, not passed {not_passed[:5]}")
EOF

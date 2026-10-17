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
#                          [--toolchain-root <dir>] [--seed 500] [--run-jobs N] [--skip-generation]
#
# The tester always links its own pinned GNU harness, so every cell imports a
# prebuilt archive: only the library compiler varies between cells. gcc with
# no --toolchain-root builds with the arm-none-eabi-gcc on PATH. armclang links
# its native math runtime through Tests/UnitTest/cmake/armclang_runtime.cmake (#857).
#
# Passes only if the tester's report covers exactly the generated cases and
# every one of them passed.

set -euo pipefail

TOOLCHAIN=""
TOOLCHAIN_ROOT=""
CPU=""
SUITE=""
OUTDIR=""
SEED=500
RUN_JOBS=0
SKIP_GENERATION=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --toolchain) TOOLCHAIN="${2:?}"; shift 2 ;;
    --toolchain-root) TOOLCHAIN_ROOT="${2-}"; shift 2 ;;
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

features=()
precision=both
if [[ "$SUITE" == float ]]; then
  features+=(--enable-f32)
  if [[ "$CPU" == cortex-m55 ]]; then features+=(--enable-f16); else precision=f32; fi
fi
f32=0; f16=0
[[ " ${features[*]-} " == *" --enable-f32 "* ]] && f32=1
[[ " ${features[*]-} " == *" --enable-f16 "* ]] && f16=1

# 1. Library archive with the cell's compiler, at the shipped optimization level.
bash "$REPO/scripts/build_staticlib.sh" --target-cpu "$CPU" --version 0.0.0 --outdir "$OUTDIR/lib" \
  --toolchain "$TOOLCHAIN" --toolchain-root "$TOOLCHAIN_ROOT" "${features[@]}"
lib="$(find "$OUTDIR/lib" -maxdepth 1 -name 'libns-cmsis-nn-*.a' | head -n 1)"
[[ -f "$lib" ]] || { echo "no archive produced under $OUTDIR/lib" >&2; exit 1; }

# 2. Import shim the tester's CMake consumes as CMSIS_NN_ROOT.
mkdir -p "$OUTDIR/import"
cat > "$OUTDIR/import/CMakeLists.txt" <<EOF
add_library(cmsis-nn STATIC IMPORTED GLOBAL)
set_target_properties(cmsis-nn PROPERTIES
  IMPORTED_LOCATION "$lib"
  INTERFACE_INCLUDE_DIRECTORIES "$REPO/Include")
target_compile_definitions(cmsis-nn INTERFACE ARM_NN_ENABLE_F32=$f32 ARM_NN_ENABLE_F16=$f16)
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
    --float-precision "$precision")
fi

# 4. Build and run the generated cases against the archive.
started="$(date +%s)"
(cd "$TESTER" && uv run helia_core_tester full --skip-generation --cpu "$CPU" --suite "$SUITE" \
  --float-precision "$precision" --cmsis-nn-root "$OUTDIR/import" --run-jobs "$RUN_JOBS" \
  --no-fail-fast --report-formats json) || status=$?

# 5. Exact coverage: generation dropped nothing, every generated case reported, every one passed.
python3 - "$TESTER" "$SUITE" "$CPU" "$lib" "${status:-0}" "$started" <<'EOF'
import hashlib, json, os, sys
from pathlib import Path

tester, suite, cpu, lib, status = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]), int(sys.argv[5])
started = int(sys.argv[6])
generated = tester / "artifacts/generated_tests" / suite / cpu
cases = sum(1 for p in generated.glob("*/*") if p.is_dir())
gen = json.loads((tester / "artifacts/reports/generation" / suite / cpu / "generation_summary.json").read_text())
dropped = {k: gen["counts"].get(k, 0) for k in ("skipped_kernel_symbol", "generation_failures", "conversion_failures")
           if gen["counts"].get(k, 0)}
if gen.get("status") != "success" or dropped or gen["counts"].get("cases_total") != cases:
    sys.exit(f"generation incomplete: status {gen.get('status')}, {dropped}, "
             f"{gen['counts'].get('cases_total')} recorded vs {cases} generated")
reports = sorted((p for p in (tester / "artifacts/reports/tests" / suite / cpu).glob("test_report_*.json")
                  if os.path.getmtime(p) >= started), key=os.path.getmtime)
if not reports:
    sys.exit(f"no tester report for {suite}/{cpu} written by this run")
report = json.loads(reports[-1].read_text())
bad = {k: report.get(k, 0) for k in ("failed", "skipped", "timed_out", "errors", "not_run", "build_failed",
                                     "generation_failed", "conversion_failed") if report.get(k, 0)}
manifest = hashlib.sha256((generated / "manifest.json").read_bytes()).hexdigest()
line = (f"{cpu} {suite}: {report['passed']}/{report['total_tests']} passed, {cases} generated, "
        f"manifest {manifest[:16]}, archive {hashlib.sha256(lib.read_bytes()).hexdigest()[:16]}, "
        f"run {report.get('duration', 0):.0f}s")
print(line)
if os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
        summary.write(f"- {line}\n")
if status or bad or report["total_tests"] != cases or report["passed"] != cases:
    sys.exit(f"cell failed: tester exit {status}, counters {bad}, {report['passed']}/{report['total_tests']} of {cases}")
EOF

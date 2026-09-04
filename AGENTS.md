# Agent guidance for ns-cmsis-nn

Ambiq's optimized fork of Arm CMSIS-NN targeting Cortex-M (Apollo SoCs). Kernels
are C with optional Helium (MVE / M-Profile Vector Extension) SIMD paths.

## SIMD: this repo uses MVE (Helium), NOT NEON

Cortex-M55-class targets implement **MVE** (`arm_mve.h`), not Armv8-A **NEON**
(`arm_neon.h`). Many intrinsics look nearly identical between the two sets but
are NOT interchangeable — a NEON intrinsic will fail to compile for an MVE
target or, worse, silently compile under a host/test configuration and then
break the embedded build. When writing SIMD paths:

- Copy the idiom from an existing kernel in this repo (e.g.
  `Source/BasicMathFunctions/arm_elementwise_add_f32.c`,
  `Source/NNSupportFunctions/arm_nn_lstm_step_f32.c`) rather than writing
  intrinsics from memory.
- The MVE loop shape is **tail-predicated**: `vctp32q(n - i)` / `vctp16q(...)`
  produces an `mve_pred16_t`, loads/stores use predicated forms
  (`vld1q_z`, `vstrwq_p` / `vstrhq_p`), and the loop steps by 4 (f32) or
  8 (f16) with no scalar tail loop. NEON has none of this.
- MVE selection is `vpselq` on an `mve_pred16_t`; NEON's lookalike `vbslq`
  (bitwise select on a vector mask) does not exist in MVE. Comparisons like
  `vcmpgeq` return predicates in MVE, vectors in NEON.
- Guard macros: `ARM_MATH_MVEF` (f32), `ARM_MATH_MVE_FLOAT16` (f16),
  `ARM_MATH_MVEI` (integer) — always with `&& !defined(ARM_MATH_AUTOVECTORIZE)`.
- **Cross-dtype helpers**: an f16 kernel may legitimately use f32 MVE vectors
  (e.g. widening accumulation), but any helper it calls must NOT be declared
  inside an `#if ARM_NN_ENABLE_F32` block — that gate is off in F16-only
  builds even though `float32x4_t` (a hardware type) is available. Scope such
  helpers on `ARM_NN_FLOAT_API_ENABLED` + the MVE macro instead. This broke
  `main` once (arm_reduce_sum_f16 → arm_nn_vec_reduce_add_f32).
- **Compile new float kernels in all three configs** before pushing — F16-only
  is a real CI target (`m55-f16-mvef`), not a hypothetical:
  `for CFG in "F32=1 F16=1" "F32=0 F16=1" "F32=1 F16=0"; do ...
  arm-none-eabi-gcc -mcpu=cortex-m55 ... -DARM_NN_ENABLE_$CFG ...; done`
- If an intrinsic you want isn't used anywhere in `Source/` or
  `Include/Internal/`, treat that as a red flag and verify it exists in
  `arm_mve.h` before using it.
- **No scalar `_Float16` conditional selects**: GCC 14.x ICEs on HFmode
  conditional moves (ternary / `MIN` / `MAX` / `CLAMP` on `_Float16` — GCC
  PR target/118460); auto-vectorization at `-O3` can synthesize one even
  from code 14.2 accepts (seen on 14.3). For min/max/clamp/abs use the
  `arm_nn_*_f16h` helpers in `arm_nnsupportfunctions.h` — but note they are
  gated on `ARM_NN_ENABLE_F16`, so a file that must also compile with F16
  disabled (e.g. `arm_minmax_common_f16.c`) cannot use them. For a general
  two-way select, or when the helpers are out of reach, use a **bitwise mask
  select** (the `arm_nn_propagate_nan_f16h` idiom: memcpy to `uint16_t`,
  mask from the comparison, recombine) — see `arm_prelu_select_f16` /
  `arm_minmax_select_f16`. Doing the select in `float32` is NOT a fix when
  both arms are round-tripped halves: GCC narrows it back to HFmode and
  still ICEs (observed on 14.3 at `-O3`).
  This rule governs `Tests/UnitTest/TestCases/**` as well as `Source/**`. The
  suites compile with `-fno-finite-math-only` on top of the library's flags,
  which removes the escape hatch that hides this ICE elsewhere, so a construct
  that builds fine in a kernel can still break the suite that tests it. If you
  hit it, do not respond by dropping `-fno-finite-math-only` —
  Tests/UnitTest/CMakeLists.txt explains why it has to stay — respond by not
  writing the select. #344 has the measured flag and toolchain matrices and
  the full diagnostic; that is where `test_arm_maximum_minimum_f16` broke
  every cortex-m55 release leg while PR CI stayed green.

## Adding a kernel: three build manifests + header

A new source file must be wired in **all** of these, or some consumer build
silently omits it (the unit-test build uses #2, Zephyr/NSX use #2, packs use #3):

1. `Source/<Group>Functions/CMakeLists.txt` — float files are listed
   explicitly under `if (ARM_NN_ENABLE_F32/F16)`; the globs only match
   `*_s8*/_s16*/...` patterns. Kept for upstream-style external consumers
   and upstream sync friction; no in-repo build consumes these.
2. `cmake/ns_cmsis_nn.cmake` — the SSoT per-group manifest
   (`_ns_cmsis_nn_group_def`). This is the authoritative one: every
   in-repo build (unit tests, Zephyr, NSX) resolves sources through it.
3. `Ambiq.NS-CMSIS-NN.pdsc` — one `<file>` entry per source, alphabetical.
   `scripts/check_pdsc.py` validates it against `git ls-files`, so stage new
   files (`git add`) before running it.

Public float declarations go in `Include/arm_nnfunctions_flt.h` — the f32
section and f16 section are separate `#if ARM_NN_ENABLE_*` blocks; put each
declaration in the right one.

**Naming — a new public float kernel must not take a bare CMSIS-DSP verb
name** (`arm_add_f32`, `arm_sqrt_f32`, `arm_mean_f32`, ...) — CMSIS-DSP owns
those, and a same-named export is a silent link-order collision, not a
compile error (`arm_abs_f16/f32` shipped that way for a full cycle, caught
by hand in #281). Prefer the upstream-consistent longer form
(`arm_elementwise_add_f32`, `arm_maximum_f32`) or the `arm_nn_` prefix (125
`arm_nn_*` tokens upstream, zero in CMSIS-DSP). Run
`python3 scripts/check_dsp_symbol_collisions.py --list-hazards` before
naming a new kernel to see the current, derived list of stems we share
with CMSIS-DSP (not hand-maintained here on purpose — a stale copy of this
list is how the next collision slips through; `--list-hazards` cannot drop
a name the way a comment can). `scripts/check_dsp_symbol_collisions.py`
enforces this in CI (#282).

## Float unit tests

Float suites live in `Tests/UnitTest/TestCases/test_arm_<kernel>/` with
checked-in numpy-generated golden data (`<name>_data.h`), a test `.c`, a
`Unity/unity_test_*.c` wrapper, and a CMakeLists referencing a
`Unity/TestRunner/*_runner.c` that is generated at configure time by
`unittest_targets.py` (needs `pyserial` + `termcolor`). Register the suite in
`Tests/UnitTest/CMakeLists.txt` inside the matching `if(ARM_NN_ENABLE_F32/F16)`
block. Host-build with:

```
cmake -S Tests/UnitTest -B <build> -DBUILD_CMSIS_NN_UNIT=ON \
  -DCMSIS_PATH=<path with CMSIS/Core/Include> \
  -DARM_NN_ENABLE_F32=ON -DARM_NN_ENABLE_F16=ON
```

then build/run individual `test_arm_*` targets. (The integer suites do not
host-build — pre-existing `%ld` format errors; they target the Corstone-300
FVP.) Pick golden-data block sizes that are not multiples of the vector lane
count so the MVE tail-predication path is exercised.

## Formatting and checks

- `pre-commit` is the commit-time hygiene gate; install it once with
  `uv tool install pre-commit==3.8.0 && pre-commit install`. The hooks cover
  clang-format, whitespace and final newlines, YAML/JSON/TOML syntax,
  merge-conflict markers, oversized additions, and deferred-work markers,
  which must reference an issue (`TODO(#421)`, or `TODO(verify)` for an
  unverified claim). No hook scans for secrets; GitHub secret scanning with
  push protection does that server side. clang-format and the two whitespace
  fixers rewrite files, so they run over staged files only and CI skips them,
  and their exclude names the generated test vectors and the inherited Arm
  files they would otherwise touch; do not "fix" those, and see
  `docs/contributing.md` for the upstream-sync recipe. CI runs the reporting hooks over every
  tracked file, except the size check, which inspects staged additions only and
  so bites at commit time. See `docs/contributing.md`.
- clang-format via `scripts/check_clang_format_changed.sh` — that script is
  the authority on the required clang-format version and covered paths
  (clang-format 16.x, matching the `.pre-commit-config.yaml` pin of 16.0.6,
  over `Include`, `Source`, and `Tests/UnitTest/Corstone-300` -- the
  pre-commit hook matches the version but only covers `Source` and
  `Include`); install it
  with `pip install clang-format==16.0.6` and, if another version is first
  on PATH, point `CLANG_FORMAT_BIN` at it.
- `python3 scripts/check_pdsc.py` after any manifest change.
- `python3 scripts/check_stale_version_refs.py` if you add a file that
  hardcodes the release version. Anything that stamps the version must be
  listed in `release-please-config.json`'s `extra-files` with an
  `x-release-please-*` annotation on the same line as the value, or it will
  never be bumped.
- License headers: new source files use the Ambiq SPDX header
  (`LicenseRef-Ambiq-Apollo-SDK`); generated test-data headers use Apache-2.0
  (matching existing precedent).

# Testing & Verification

How heliaCORE is qualified: what runs on every pull request, what runs at
release time, what the known limits are, and how to retrieve coverage
reports. The README carries a short summary; this page is the full
contract.

## What every pull request verifies

Merging requires the **`CI Passed`** status check — the single status check
the `main` branch ruleset requires. It fails unless every gated job below
passes; a red run blocks the merge.

| Check | What it establishes | Targets |
| --- | --- | --- |
| **Numerics** — `helia-core-tester` under the Corstone-300 FVP | kernel results match reference vectors | int4/int8/int16 on cortex-m0, cortex-m4 and cortex-m55; `float32` on m4 (scalar) and m55 (scalar + MVE); `float16` on m55 (scalar + MVE) |
| **Shipped-flags numerics** — same suite, no coverage instrumentation | the code that ships — `-Ofast`, real MVE inline assembly — computes the same answers as the instrumented legs | int on cortex-m4 and cortex-m55; `float32` on m55 |
| **Toolchain build + strict link** | every kernel compiles and every symbol resolves — no `--gc-sections`, no ignored undefined symbols | GCC 13.2.Rel1 / 14.3.Rel1 / 15.3.Rel1, ATfE 19.1.5 and armclang 6.23.32, each on cortex-m55 and cortex-m4 (the armclang cell needs a licence secret, so it skips on fork PRs) |
| **Memory safety** — host sanitizer | out-of-bounds access, undefined behaviour and leaks that leave the numerics intact, such as a scratch buffer under-sized by a `get_buffer_size` query | **x86 host**, scalar (non-MVE) paths; ~148 Unity suites under ASan + UBSan + LSan via `ctest` (the job asserts a floor of 140) |
| **Legacy Unity compile gate** — `unity-m55-compile.yml` | every legacy Unity test translation unit compiles and links with the release leg's own flags and `-Werror`, so a test file cannot first meet a cross-compiler at release time; it does not execute anything under the FVP | cortex-m55, cortex-m4 and cortex-m0 — `legacy-tester.yml`'s own matrix — each built in the CI container with the harness's pinned Arm GNU toolchain and that leg's float flags (`ARM_NN_ENABLE_F32` on all three, `ARM_NN_ENABLE_F16` on m55 only, so m55 builds 146 suites and m4/m0 build 132). The m4 cell is what compiles the `ARM_MATH_DSP && !ARM_MATH_MVEI` blocks in 12 Unity test files, which an m55 build never reaches. The m0 cell is not there for a preprocessor shape — the host-sanitizer row directly above already compiles the neither-macro shape of the same files on x86 under `-Werror`, with a smaller warning set than the target cells — but for the target and toolchain: arm-none-eabi 14.2.rel1 at `-Ofast` on a soft-float, no-FPU core (`-mfloat-abi=soft`). Before compiling, each cell parses the full release token out of the harness's own `GCC_URL` (`Tests/UnitTest/build_and_run_tests.sh`, today `14.2.rel1`) and fails unless the downloaded compiler's `--version` banner reports the same token, compared case-insensitively and in full rather than by major.minor. Each cell also asserts, per CPU, that the suites CMakeLists.txt declares under its options, the test objects the build produced and the suite directories on disk are the same set by name |
| **Packaging & wiring contracts** | PDSC/CMSIS-Pack, the CMake single-source-of-truth config, Zephyr and NSX wiring, SPDX headers, the release-pipeline contract checks | ubuntu runners; no target hardware involved |
| **Docs** | the Sphinx + Doxygen site still builds | ubuntu runner |

Every run's summary renders a per-leg test matrix (CPU × suite, pass/fail
counts and failing case names), so a red run names its failures without
artifact downloads.

## Qualification model

cortex-m4 and cortex-m55 are the shipping targets; cortex-m0 is qualified
to the same functional bar as a deliberate scalar baseline. The
Corstone-300 FVP is the qualification vehicle for functional and coverage
results: it is an instruction-accurate model of cortex-m55, and the m0- and
m4-compiled images execute unmodified on that same model — the code they
ship is exercised instruction by instruction, though m0/m4 core behaviour
is not itself modelled. Qualification is expressed per Cortex-M core, not
per Apollo part: the kernels are core-specific and part-agnostic by design
(`nsx/nsx-module.yaml` declares `socs: "*"`). EVB testing on Apollo parts
is planned regression-tier work on top of this, not a substitute for it —
part-specific data published today is the Apollo510 EVB benchmark set in
[Kernel Benchmarks](kernel-benchmarks.md).

## What runs at release time, or on demand

`release.yml` additionally re-runs the FVP numerics suite, runs the Unity
suites on Arm (`legacy-tester.yml`, cortex-m0/m4/m55 under the FVP), and
runs `release-verify`, which re-reads the published GitHub Release and
fails if a required asset is missing. The Unity suites' *build* now also runs
on pull requests for all three cores, through `unity-m55-compile.yml` in the
table above; their *execution* still runs only here and nightly.
`release-verify` does not run on pull requests at all.

A nightly scheduled run (`nightly.yml`) re-runs the FVP numerics suite,
the legacy Unity suites, and the toolchain strict-link matrix on `main`,
maintaining a rolling issue while red.

`staticlib-dryrun.yml` (full three-CPU × three-toolchain sweep, packaged
tarballs) and `pack-dryrun.yml` are `workflow_dispatch` only — they run
when somebody asks, not on a schedule and not per PR.

Each release publishes **17 required assets**, checked after publication by
`release-verify`:

| Asset | Count |
| --- | --- |
| CMSIS-Pack — `Ambiq.NS-CMSIS-NN.<version>.pack` | 1 |
| CMake SDK tarballs — {gcc, ATfE} × {m0, m4, m55}, each with a `.sha256` | 12 |
| Static-library bundles — `ns-cmsis-nn-staticlibs-{gcc,atfe}-<version>.zip`, each with a `.sha256` | 4 |

armclang produces eight further assets of the same shape. They are
**optional** unless the repository variable `ARMCLANG_REQUIRED` is set to
`true`, because building them needs a commercial Arm Compiler for Embedded
licence.

## Known limits, and what is planned against them

- **armclang and ATfE are built and strict-linked on every PR but never
  executed.** Kernel logic is shared across toolchains, so this is a
  deliberate trade — the guarantee is *compiles and links*, not *computes
  correctly*. Cross-toolchain execution is tracked in
  [#340](https://github.com/AmbiqAI/ns-cmsis-nn/issues/340).
- **No memory checking of MVE/Helium or DSP paths.** The sanitizers run on
  the x86 host, which selects the scalar implementations. Guard-byte
  checking on target is tracked in
  [helia-core-tester#68](https://github.com/AmbiqAI/helia-core-tester/issues/68).
- **UBSan's `shift-base` check is masked** repo-wide (removing it fails 20
  of the 144 suites; the sites are documented in the workflow). Related
  residual shift-base UB on the M4 DSP path — invisible to the x86
  sanitizer, which cannot compile those sites — is tracked in
  [#357](https://github.com/AmbiqAI/ns-cmsis-nn/issues/357).
- **Coverage is gated on a floor and no-regression** per merged run
  (`ci/coverage-floor.json` holds the floor; raising it is a reviewed
  diff). The per-kernel set-membership gates remain open in
  [helia-core-tester#73](https://github.com/AmbiqAI/helia-core-tester/issues/73).

## Coverage reports

Line coverage is merged across the int, float and MVE-float legs on every
`ci.yml` run, then classified into *covered*, *zero-hit but reachable*, and
*expected-zero* (orphan or known-unreachable). Both outputs are attached to
a workflow run rather than to a permanent URL:

1. The **job summary** of `coverage-merge-summary` — a per-CPU coverage and
   test table, readable in the browser without downloading anything.
2. The **`coverage-merged` artifact** on the same run (retained 90 days,
   the repository default), holding `index.html` (a browsable LCOV
   report), `coverage_merged.info` and `coverage_merged_summary.{md,json}`.

To pull the latest from `main`:

```sh
run=$(gh run list -R AmbiqAI/ns-cmsis-nn --workflow=ci.yml --branch=main \
        --status=success --limit 1 --json databaseId --jq '.[0].databaseId')
gh run download -R AmbiqAI/ns-cmsis-nn "$run" -n coverage-merged -D coverage
# summary: coverage/coverage_merged_summary.md   full report: coverage/index.html
```

There is deliberately no coverage badge and no percentage quoted on this
page: the number exists only inside build artifacts, and any figure written
here would be stale within a week.

## Quick links

- Latest release — <https://github.com/AmbiqAI/ns-cmsis-nn/releases/latest>
- All CI runs — <https://github.com/AmbiqAI/ns-cmsis-nn/actions>

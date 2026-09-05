# CI Matrix

One row per CI job, with what triggers it, what it targets, and whether it
executes anything or only builds. Every cell is derived from the workflow
YAML in `.github/workflows/`, and each row cites the file it came from.
[Testing & Verification](verification.md) is the narrative version of the
same contract; this page is the lookup table.

Read the "executes" column first. Most of this matrix compiles and links;
a smaller part of it runs code and compares results. Those are different
guarantees and the difference is where defects have escaped before.

## What runs on every pull request

All of these are `workflow_call` jobs invoked from `.github/workflows/ci.yml`,
which triggers on `pull_request` (any base branch), on `push` to `main`, and
on `workflow_dispatch`. They report through the single `ci-passed` job, whose
status context is `CI Passed`.

| Job | Workflow | Cores | Precisions | Executes | Toolchain |
| --- | --- | --- | --- | --- | --- |
| `codegen-tests` | `helia-core-tester.yml` | cortex-m0, cortex-m4, cortex-m55 | integer suite (`--suite int`) | yes, Corstone-300 FVP | CI container's Arm GNU 14.3.1 |
| `shipped-flags-tests` | `helia-core-tester.yml` | cortex-m4 (int), cortex-m55 (int, f32, f16) | integer, `float32`, `float16` | yes, Corstone-300 FVP | CI container's Arm GNU 14.3.1 |
| `codegen-tests-mve-float` | `helia-core-tester.yml` | cortex-m55 | `float32`, `float16` | yes, Corstone-300 FVP | CI container's Arm GNU 14.3.1 |
| `codegen-tests-float-fallback` | `helia-core-tester.yml` | cortex-m4 (f32), cortex-m55 (f32, f16) | `float32`, `float16` | yes, Corstone-300 FVP | CI container's Arm GNU 14.3.1 |
| `gcc` | `toolchain-matrix-strict-link.yml` | cortex-m4, cortex-m55 | `float32` on both, `float16` on m55 | no, build and strict link only | GCC 13.2.Rel1, 14.2.Rel1, 15.3.Rel1 |
| `atfe` | `toolchain-matrix-strict-link.yml` | cortex-m4, cortex-m55 | `float32` on both, `float16` on m55 | no, build and strict link only | ATfE 19.1.5 |
| `armclang` | `toolchain-matrix-strict-link.yml` | cortex-m4, cortex-m55 | `float32` on both, `float16` on m55 | no, build and strict link only | armclang 6.23.32 |
| `host-sanitizer` | `host-sanitizer.yml` | x86-64 host, scalar paths | `float32`, `float16` | yes, on the host under `ctest` | ubuntu-24.04 system GCC |
| `compile` | `unity-m55-compile.yml` | cortex-m55, cortex-m4, cortex-m0 | `float32` on all three, `float16` on m55 | no, compile and link only | Arm GNU 14.2.rel1 |
| `f16-exec-gcc-floor` | `unity-f16-exec-gcc-floor.yml` | cortex-m55 | `float16` | yes, QEMU `mps3-an547` | GCC 13.2.Rel1 |
| `build-staticlibs`, `build-pack` | `pack-dryrun.yml` | cortex-m0, cortex-m4, cortex-m55 | n/a | no, packaging only | GCC 14.2.Rel1 |
| `check_pdsc`, `pdsc_contract`, and the other textual gates | `pdsc.yml` | none | n/a | Python and shell gates on the host | none |
| `contract`, `find_package` | `ssot-contract.yml` | none | n/a | CMake configure only | none |
| `wiring`, `install`, `prebuilt_real` | `nsx-integration.yml` | none | n/a | CMake configure only | none |
| `wiring`, `kconfig` | `zephyr-integration.yml` | none | n/a | CMake and Python only, no Zephyr SDK | none |
| `build` | `docs.yml` | none | n/a | Sphinx and Doxygen build | none |
| `clang-format`, `license-headers`, `pre-commit`, `release-contract` | four workflows of the same name | none | n/a | host lint and contract checks | clang-format 16.0.6 |

Sources: `.github/workflows/ci.yml` lines 19 to 38 for the triggers and
lines 131 to 164 for the gated set; each callee's own `strategy.matrix` for
its cells.

## What runs nightly

`.github/workflows/nightly.yml` triggers on `schedule` (`17 7 * * *`) and on
`workflow_dispatch`.

| Job | Workflow | Cores | Executes | Notes |
| --- | --- | --- | --- | --- |
| `helia-core-tester` | `helia-core-tester.yml` | as the PR rows above | yes, Corstone-300 FVP | the same suite the PR gate runs |
| `legacy-unit-tests` | `legacy-tester.yml` | cortex-m0, cortex-m4, cortex-m55 | yes, Corstone-300 FVP | `float32` on all three, `float16` on m55 |
| `toolchain-strict-link` | `toolchain-matrix-strict-link.yml` | as the PR rows above | no | |
| `release-assets-audit` | `nightly.yml` | none | `gh` against published releases | expands `ci/release-assets.json` |

## What runs at release

`.github/workflows/release.yml` triggers on `push` to `main` and on
`workflow_dispatch` with a `recover_tag` input.

| Job | Workflow | Executes | Notes |
| --- | --- | --- | --- |
| `release-unit-tests` | `legacy-tester.yml` | yes, Corstone-300 FVP | same three cores as nightly |
| `release-helia-core-tester` | `helia-core-tester.yml` | yes, Corstone-300 FVP | |
| `publish-staticlibs` | `release.yml` | no | GCC, ATfE and armclang, three cores each |
| `release-verify` | `release.yml` | no | re-reads the published release and fails on any difference from `ci/release-assets.json` |

## What runs only on demand

| Workflow | Trigger | Cores and toolchains |
| --- | --- | --- |
| `staticlib-dryrun.yml` | `workflow_dispatch` | cortex-m0, cortex-m4, cortex-m55 across gcc and atfe |
| `update_helia_core_tester.yml` | `workflow_dispatch` | none, submodule bump only |

`pack-dryrun.yml` used to belong here. It now runs per PR through `ci.yml`
and weekly on its own `schedule`, and stays dispatchable.

## What nothing executes

Stated plainly, because each of these is a guarantee the matrix above does
not give:

- **Nothing functional runs on GCC 14.2.Rel1 or 15.3.Rel1.** Those two cells
  of `toolchain-matrix-strict-link.yml` build and strict-link the archive and
  stop. The only GCC releases that execute a test binary are the container's
  14.3.1, through every FVP leg, and 13.2.Rel1, through the QEMU `float16`
  cell. Note the gap this leaves: 14.2.Rel1 is the release that builds the
  shipped static libraries, and no job executes anything built by it.
- **Nothing executes anything built by ATfE or armclang.** Both are built and
  strict-linked on every PR and neither ever runs, on any core. Tracked in
  [#340](https://github.com/AmbiqAI/ns-cmsis-nn/issues/340).
- **Nothing executes the MVE or DSP paths under a sanitizer.** `host-sanitizer.yml`
  runs on the x86-64 host, which selects the scalar implementations. Tracked
  in [helia-core-tester#68](https://github.com/AmbiqAI/helia-core-tester/issues/68).
- **The legacy Unity suites do not execute on any pull request.**
  `legacy-tester.yml` is called only from `nightly.yml` and `release.yml`.
  On the PR path they are compiled but not run by `unity-m55-compile.yml`,
  except the five `float16` suites that `unity-f16-exec-gcc-floor.yml`
  executes under QEMU.
- **`release-verify` never runs on a pull request.** It lives in
  `release.yml` and only fires inside a release or recovery run. The nightly
  `release-assets-audit` job is what covers the gap.

## Two things this page cannot tell you

**Which checks are required to merge.** Branch protection and ruleset
configuration live in repository settings, not in the tree, so no file here
can state it. What the tree does show is that `ci.yml` funnels every gated
job into one `ci-passed` job whose context is named `CI Passed`
(`.github/workflows/ci.yml` line 7 and lines 131 to 149), which is the
context intended to be required.

**Which dtypes the helia-core-tester suites cover.** The workflow passes
`--suite int` or `--suite float` and a `--float-precision`; the set of
operators and quantization types inside each suite is defined by the pinned
`Tests/helia-core-tester` submodule release, not by anything in this
repository. At the currently pinned commit, that project's
`helia_core_tester/generation/io/dtypes.py` lists `S4` among both
`ALLOWED_TENSOR_DTYPES` and `INTEGER_DTYPES`, so int4 is within the integer
suite's declared reach. Which operators actually generate int4 cases is
upstream's record, not this page's. Legacy Unity coverage of int4 is
directly visible here: `Tests/UnitTest/TestCases/TestData/` carries int4
vector directories, and those suites execute nightly and at release only.

## See also

- [Testing & Verification](verification.md), the narrative contract and the
  coverage-report instructions.
- [Versioning & Releases](releases.md), the release asset contract.

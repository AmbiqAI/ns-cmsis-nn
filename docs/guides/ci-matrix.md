# CI Matrix

What triggers each CI job, what it targets, and whether it executes anything
or only builds. The tables cover the jobs that gate a pull request, the
nightly sweep, the release path, and the workflows those call. Every cell is
derived from the workflow YAML in `.github/workflows/`, and each table cites
the files it came from. This is a hand-maintained page, not a generated
inventory: a workflow added without a row here goes unlisted, and nothing
fails when that happens.
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
| `f16-exec` | `unity-f16-exec-gcc-floor.yml` | cortex-m55 | `float16` | yes, QEMU `mps3-an547`, at `-Ofast` | GCC 13.2.Rel1 |
| `build-staticlibs`, `build-pack` | `pack-dryrun.yml` | cortex-m0, cortex-m4, cortex-m55 | n/a | no, packaging only | GCC 14.2.Rel1 |
| `check_pdsc`, `pdsc_contract`, and the other textual gates | `pdsc.yml` | none | n/a | Python and shell gates on the host | none |
| `contract`, `find_package` | `ssot-contract.yml` | none | n/a | CMake configure only | none |
| `wiring`, `install`, `prebuilt_real` | `nsx-integration.yml` | none | n/a | CMake configure only | none |
| `wiring`, `kconfig` | `zephyr-integration.yml` | none | n/a | CMake and Python only, no Zephyr SDK | none |
| `build` | `docs.yml` | none | n/a | Sphinx and Doxygen build | none |
| `changed-files` | `clang-format.yml` | none | n/a | formatting check on changed files | clang-format 16.0.6 |
| `all-files` | `pre-commit.yml` | none | n/a | the pre-commit hook set on the whole tree | pre-commit 3.8.0 |
| `check` | `license-headers.yml` | none | n/a | header check on the host | none pinned |
| `ci-tool-manifest`, `gen-pack-action-pin`, and the other contract jobs | `release-contract.yml` | none | n/a | shell assertions over the release path's own wiring | none pinned |

Sources: `.github/workflows/ci.yml`'s `on:` block for the triggers and its
`ci-passed` job's `needs:` list for the gated set; each callee's own
`strategy.matrix` for its cells. The Job column names the job inside the
callee, which is not the id `ci.yml` gives the call. The container's Arm GNU
14.3.1 is pinned in `ci/tools/manifest.json` (the `arm-gnu` entry), which the
CI image installs; for the helia-core-tester rows, the FVP execution and the
operator and dtype set behind each suite come from the pinned
`Tests/helia-core-tester` submodule rather than from anything in this
repository.

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
| `release-please` | `release.yml` | no | creates the release PR, then publishes the tag and the Release object, and resolves the trusted commit sha every later job builds from |
| `resolve-release-capabilities` | `release.yml` | no | decides whether an Arm user-based license is available, which is what makes the armclang assets producible |
| `publish-staticlibs` | `release.yml` | no | GCC 14.2.Rel1 and ATfE 19.1.5, three cores each; builds the SDK tarballs |
| `publish-staticlibs-armclang` | `release.yml` | no | armclang 6.23.32, the same three cores, run one at a time; its assets stay optional unless `ARMCLANG_REQUIRED` is set |
| `publish-staticlib-bundles` | `release.yml` | no | zips the per-toolchain bundles from the other two publishers and uploads them; builds nothing itself |
| `publish-pack` | `release.yml` | no | gen-pack, Doxygen 1.9.6 and packchk 1.4.1 over the GCC prebuilt archives; publishes the `.pack` and the Pages docs |
| `release-unit-tests` | `legacy-tester.yml` | yes, Corstone-300 FVP | same three cores as nightly |
| `release-helia-core-tester` | `helia-core-tester.yml` | yes, Corstone-300 FVP | |
| `release-verify` | `release.yml` | no | re-reads the published release and fails on any difference from `ci/release-assets.json` |

Toolchain versions in this table come from `release.yml`'s own steps for GCC
and from `ci/tools/manifest.json` for the container-installed ATfE and
armclang.

## What other workflows call

Neither of these has a `pull_request` trigger; both reach a pull request only
through a caller.

| Job | Workflow | Trigger | Executes | Notes |
| --- | --- | --- | --- | --- |
| `resolve` | `resolve-ci-image.yml` | `workflow_call` only | no | resolves which `ghcr.io/ambiqai/ns-cmsis-nn-ci` tag a caller runs in, and reports whether it came from a release or the fallback |
| `build-and-push` | `build_publish_docker.yml` | `workflow_call`, `workflow_dispatch`, `schedule` (`0 16 * * 1`) | yes, a container build and push | builds the CI image from `.devcontainer/Dockerfile`, which installs Arm GNU 14.3.1, ATfE 19.1.5 and armclang 6.23.32 per `ci/tools/manifest.json`, plus clang-format 16.0.6; called from `release.yml`'s `publish-ci-image` |

## What runs only on demand

| Workflow | Trigger | Cores and toolchains |
| --- | --- | --- |
| `staticlib-dryrun.yml` | `workflow_dispatch` | `build`: cortex-m0, cortex-m4, cortex-m55 across gcc and atfe; `armclang`: the same three cores in a separate job, outside that matrix |
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
job into one `ci-passed` job whose context is named `CI Passed` (the
`name:` at the top of `.github/workflows/ci.yml` and the `ci-passed` job's
`needs:` list), which is the context intended to be required.

**Which dtypes the helia-core-tester suites cover.** The workflow passes
`--suite int` or `--suite float` and a `--float-precision`; the set of
operators and quantization types inside each suite is defined by the pinned
`Tests/helia-core-tester` submodule release, not by anything in this
repository. Reading that pin is the only way to answer it, and the answer
moves whenever the pin does. At the pinned commit `28f1a8a1`, the
descriptors under `Tests/helia-core-tester/assets/descriptors/` carry 39
int4 cases: 25 in `ConvolutionFunctions/convolve.yaml`, 10 in
`ConvolutionFunctions/depthwise_conv.yaml` and 4 in
`FullyConnectedFunctions/fully_connected.yaml`. Legacy Unity coverage of
int4 is directly visible here: `Tests/UnitTest/TestCases/TestData/` carries
int4 vector directories, and those suites execute nightly and at release
only.

## See also

- [Testing & Verification](verification.md), the narrative contract and the
  coverage-report instructions.
- [Versioning & Releases](releases.md), the release asset contract.

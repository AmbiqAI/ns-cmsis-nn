# heliaCORE site plan

Scope: the product site only. Release news, demos, cross-product stories and
portfolio positioning live on the Dev Hub. This site answers three questions:
what heliaCORE is, how to get it into a build, and what each kernel does.

## Positioning (what is unique, in the order a visitor needs it)

1. The kernel layer under HELIA. heliaCORE is the optimized operator set that
   heliaRT (runtime) and heliaAOT (compiler) execute on. A model never calls
   it directly; a build links it. Source: docs/index.md 162-167, docs/why.md 113.
2. Built for Ambiq silicon. Apollo-class Cortex-M cores: M0/M0+ (scalar C),
   M4 (DSP), M55 (MVE/Helium). Apollo510 is the named benchmark board today.
   Source: README.md 170-174, kernel-benchmarks.md 104-113.
3. MVE as a small programmable NPU. The claim the owner wants front and center:
   with MVE, TCM and the core, the M55 behaves like an efficient programmable
   NPU. TODO(verify): per-cycle MAC throughput (8 int8, 4 int16/fp16,
   2 int32/fp32) is not stated anywhere in the repo; needs the Arm Helium or
   Apollo510 source of record before publication.
4. Coverage, not just speed. 23 operator families, eight API groups, data
   types s8, s16, s4 weights, and f16/f32 behind build flags. Coverage relative
   to upstream CMSIS-NN is the differentiator to show as a matrix, not prose.
   Source: Source/ directory, cmake/ns_cmsis_nn.cmake:80, operator-kernel-coverage.md.
5. One CMake source of truth, four delivery paths. cmake/ns_cmsis_nn.cmake
   feeds standalone CMake, CMSIS-Pack (gen_pack.sh), Zephyr module and
   neuralSPOT-X; static libraries per core and toolchain (GCC, ATfE, armclang).
   The ssot-contract workflow guards it. Source: workflows pack-dryrun,
   staticlib-dryrun, nsx-integration, ssot-contract.
6. Measured efficiency. Apollo510 EVB, Cortex-M55, 96 MHz LP mode, cycle
   counts via DWT: MVE 11.98x over scalar, 3.26x DSP over scalar. A standard
   heliaCORE-vs-CMSIS-NN benchmark across SoCs is a later addition and gets a
   reserved place in the structure now.

## Structure

Top bar sections, and the left sidebar shows only the current section's pages.
Both come from one nav definition so they cannot drift.

| Section | Pages | Notes |
|---|---|---|
| Home | landing | Hero (positioning 1-3), coverage-at-a-glance, delivery paths, efficiency band, "used by heliaRT and heliaAOT" strip, links into each section. No changelog, no demos. |
| Getting started | Overview; CMake; CMSIS-Pack; Zephyr; neuralSPOT-X; Toolchains | "Choose your path" stays. Toolchain pinning moves here from Guides because it is a setup concern. |
| Architecture | Acceleration paths (scalar, DSP, MVE); Cortex-M targets; Data types and quantization (A8W8, A16W8, s4 weights, f16/f32); How the build selects a path | New section. This is where the NPU-like story and the dtype story live, with diagrams. Mostly rewritten from dsp-mve-coverage.md and the README. |
| Coverage | Operator coverage matrix; Data type by family; Compared with CMSIS-NN | Promoted from a guide to a section: it is the product's main claim. Generated from a checked-in matrix once one exists; until then the current tables. |
| Performance | Kernel benchmarks (Apollo510); Methodology; Cross-SoC comparison (placeholder, hidden until data exists) | Moved out of Guides. Charts on the chart lane. |
| Reference | C API index by group; generated module pages; API generation notes | The generated tree stops being a nested sidebar group: the section sidebar lists the eight groups directly, module pages under them. |
| Contributing | Contributing; Testing and verification; CI matrix; Publish once after local review; Releases and versioning | Maintainer material grouped in one place. Releases moves here because release history belongs on the Dev Hub. |

What disappears: the "Guides" umbrella (every page has a better home), the
"Why heliaCORE" page as a separate entry (its content becomes Home plus
Architecture), and the "Guides > Guides" duplicate label.

## Layout rules for product sites (package-level, applies to all products)

- Top bar: product name, section links, search, theme. Sidebar: pages of the
  current section only, with the section name as the sidebar heading.
- Generated reference pages are never nested more than one level under their
  group.
- Landing page keeps the sidebar (already the case).
- Card grids align (helia-ui #94, fix in progress).

## Decisions (owner, 2026-09-16)

- Structure approved as above. Section name is Architecture. Coverage and
  Performance are separate sections. "Why heliaCORE" is folded into Home and
  Architecture. Diagrams are drafted as proposals for the owner to accept.
- Benchmark claims and values (clock, figures) are left as they are until the
  owner captures new values; the visualization is rebuilt now.
- Kernel index: a searchable, filterable index of operators and kernels
  (group, data type, path, name) on the Coverage and Reference sections.
- Future SSoT contract: each kernel states its prerequisites, buffer-size
  rule, tolerances and supported paths in one machine-readable place; the
  index and the reference render from it. Tracked as a follow-up issue in the
  product repo; the index is designed to consume it.

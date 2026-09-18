# heliaCORE landing page: content and evidence

The owner's annotated review supersedes the earlier six-heading proposal.
Home must explain the library, broad production-model coverage, data types,
Cortex-M acceleration, and integration. Avoid generic "what/why" sections,
repeated introductions, unexplained metric panels, and undifferentiated chips.

## Page structure

1. **Neural network kernels for Cortex-M.** State what heliaCORE does and show
   A8W8, A16W8, FP16, and FP32 with short explanations. Float APIs remain
   explicitly opt-in and experimental, as documented in the repository.
2. **Far broader operator coverage than CMSIS-NN.** Lead with the explicit
   comparison, then show concrete operator families and
   examples beyond convolution and dense layers. Do not equate benchmark
   coverage with production coverage or promise every graph is fully optimized.
3. **Put DSP and Helium to work.** Explain packed integer and vector execution,
   target selection, and caller-owned memory. Link to architecture and benchmarks.
4. **One kernel library. Your build system.** Explain the source manifest and
   what each CMake, CMSIS-Pack, Zephyr, and neuralSPOT-X path means to a user.
   Include an explicit setup link for the section.
5. **From model execution to kernel calls.** Briefly connect heliaRT and heliaAOT
   to the kernel library and direct users to function contracts.

Copy lives in `src/content/docs/index.mdx`, so the package's Markdown and
llms renditions can read it. `src/components/HomeLayout.astro` owns only local
layout styles, scoped to the homepage and based on shared package tokens.

## Evidence used

| Claim | Source |
|---|---|
| Library purpose, scalar/DSP/MVE selection, no dynamic allocation | [README](../README.md), Highlights and architecture sections |
| A8W8 and A16W8 quantized kernels | [Public integer header](../Include/arm_nnfunctions.h), convolution and fully connected signatures |
| FP16/FP32 support and opt-in status | [README](../README.md), experimental float support; [float declarations](../Include/arm_nnfunctions_flt.h) |
| Gather, scatter, tile, reductions, broadcast arithmetic, GRU | [Source tree](../Source/), [integer header](../Include/arm_nnfunctions.h), [float header](../Include/arm_nnfunctions_flt.h) |
| Explicit float MVE implementation example | [Reduce sum](../Source/BasicMathFunctions/arm_reduce_sum_f32.c), MVE predicates and vector accumulation |
| Shared operator/source selection | [CMake manifest](../cmake/ns_cmsis_nn.cmake), [Zephyr consumer](../zephyr/CMakeLists.txt), [NSX consumer](../nsx/CMakeLists.txt) |
| Pack consistency rather than a generated single manifest | [PDSC validator](../scripts/check_pdsc.py), SSoT versus PDSC source-list agreement |
| Four integration options | [Getting started](src/content/docs/getting-started/index.mdx) and its linked setup pages |

The upstream comparison is deliberately specific, not a coverage multiplier.
Checked upstream commit
[`9e1b4768`](https://github.com/ARM-software/CMSIS-NN/tree/9e1b4768c640606817f0d8f8e53a3d39be817ab4):
its Source tree and public integer/float headers lack the named gather,
scatter, reduction, broadcast arithmetic, and GRU entry points used as examples
above. This is not a complete semantic equivalence or performance comparison.
Its [README](https://github.com/ARM-software/CMSIS-NN/blob/9e1b4768c640606817f0d8f8e53a3d39be817ab4/README.md)
also documents experimental FP16/FP32, so floating point is a heliaCORE
capability, not a claim of exclusivity.

## Removed from Home

- "40+ field models," "200+ accelerated ops," and "53 op types": provenance
  or definitions were insufficient for the landing page.
- Bare 12x/8x/3x figures: retained unchanged on the detailed performance page,
  with their measurement context, rather than used as the acceleration story.
- Numeric MVE throughput claims: not introduced without a source of record.
- Build group counts presented as model operator coverage: different concepts.

The owner requested stronger differentiation after the first rendered revision.
The hero and coverage heading now state the broader CMSIS-NN coverage directly;
the linked comparison page provides a nine-row table with fixed source revisions.
No function-count ratio is used as a proxy for distinct model operators.

## Layout and product direction from the annotated review

The hero has four highlights: coverage, acceleration, data types, and shared
build integration. Section copy follows its heading, without a separate
right-hand paragraph. The integration section uses the package Mosaic with
a large build-manifest feature card and smaller integration cards.

The owner positions heliaAOT as the flagship, recommended inference path for
latency, power, and memory efficiency. heliaCORE powers both heliaAOT and
heliaRT; most users should begin with these model-level solutions. The paired
heliaAOT README confirms standalone C inference generation and planned memory;
the heliaRT README confirms the optimized LiteRT for Microcontrollers basis.
This is product guidance, not a quantified performance comparison.

## Partner-aware positioning

Owner direction: make Ambiq silicon the primary target in the headline and name
Apollo in the opening. Introduce the kernel library and its role in HELIA
inference before presenting operator coverage. Credit the Arm CMSIS-NN
foundation and show concrete extensions without competitive or dismissive
headlines. This supersedes earlier requests for aggressive comparative copy.

## Accepted content review

Applied the owner's approved review: recommend AOT/RT before build integration;
give hero, introduction, and closing distinct roles; explain broader coverage's
benefit without equating kernel availability to runtime model compatibility;
define quantized formats and link operator-specific support; lead SSoT with
consistent selection and avoid broad package-verification claims. The linked
CMSIS-NN page uses the same partner-aware framing. Target and toolchain links
provide a route to compatibility details without adding a hardware chip list.

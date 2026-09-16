# heliaCORE landing page: copy draft

For review before any of it is built. Layout is a separate agreement.

## Who lands here

A firmware or embedded ML engineer who has a model and an Apollo part. Their
real problems, in the order they hit them:

1. Their graph has an operator the stock kernel library never optimized, and
   that one operator eats the inference budget.
2. They cannot tell whether their part will reach the vector path or quietly
   fall back to scalar C.
3. Wiring a kernel library into their build is a chore, and every vendor tells
   a different integration story.

The page answers those three, in that order, and then says where to start.

## The arc

Hero, then the coverage argument, then the speed argument, then where it sits
in HELIA, then how to get it, then where to go deep. Six sections, no section
named after a question.

---

## 1. Hero

**Eyebrow:** Optimized AI kernels for Ambiq silicon

**Headline:** Every operator at silicon speed.
_(accent on "Every")_

Alternatives: "Full speed, operator by operator." / "The fast path for every
operator in your graph."

**Badge:** ATfE + MVE ready

**Lede:**
heliaCORE is Ambiq's neural network kernel library for Apollo devices. It
ships a tuned kernel for every operator in the graph, selects the fastest path
the core supports, and links into your build through CMake, CMSIS-Pack,
Zephyr, or neuralSPOT-X.

**Actions:** Get started · Architecture · Browse the API

**Side panel:** version, four figures, the two-line CMake snippet.
_See "Claims to settle" on the four figures._

---

## 2. The coverage argument

**Heading:** One slow operator is a slow model.

Alternatives: "Inference time collects where nobody looked." / "The graph is
only as fast as its slowest kernel."

**Lede:**
Inference time pools in the operators nobody optimized. A graph that runs nine
tuned kernels and one reference fallback spends most of its cycles in the
fallback. heliaCORE covers twenty-three operator families, so the whole graph
runs on the vector path instead of just the convolutions.

**Supporting content:** the eight API groups with kernel counts, the data type
row (s8, s16, s4 weights, f16 and f32 behind a build flag), and one line on
what this fork adds over upstream Arm CMSIS-NN.

**Section link:** Browse coverage

---

## 3. The speed argument

**Heading:** Your Cortex-M55 is already an accelerator.

Alternatives: "Helium is the accelerator you already shipped." / "No separate
NPU in the budget."

**Lede:**
Helium vector extensions, tightly coupled memory, and the core act together as
a small programmable engine for neural network math. heliaCORE is what
programs it. Hand-tuned MVE kernels keep the vector unit fed, a DSP path
serves Cortex-M4, and a scalar path serves Cortex-M0+, all from one source
tree and one build.

**Supporting content:** three rounded figures (12x, 8x, 3x) with a bar chart
of the three paths, and the board and clock as a caption under them, not as a
heading. Per-kernel tables live on the Performance section.

**Section link:** See the numbers

---

## 4. Where it sits

**Heading:** Underneath heliaAOT and heliaRT.

Alternatives: "The floor the HELIA stack stands on." / "Everything above it
gets faster."

**Lede:**
Models compiled by heliaAOT and graphs executed by heliaRT land on these
kernels. You rarely call heliaCORE directly. You link it, and the layers above
inherit the speed.

**Supporting content:** the four-layer stack, model tooling to kernels to
acceleration path to Apollo silicon.

---

## 5. Integration

**Heading:** One build description, four ways to consume it.

Alternatives: "However you build, the same kernels." / "Generated from one
source, verified in CI."

**Lede:**
The CMSIS-Pack, the Zephyr module, the neuralSPOT-X integration, and the
prebuilt static libraries are all generated from a single CMake source of
truth and checked in CI. The kernels you link are the kernels that were
tested.

**Supporting content:** four compact rows, each with the one line that gets
you going.

**Section link:** Get started

---

## 6. Going deeper

**Heading:** Find the kernel, read the contract.

Alternatives: "Five hundred kernels, one search box." / "Every function, every
variant."

**Lede:**
Every public function, searchable by name and filtered by operator group, data
type, and header. Each entry links to its generated reference page.

**Supporting content:** the live kernel index, plus two links into Architecture
and Contributing.

---

## Claims to settle before this ships

| Claim | Status |
|---|---|
| 23 operator families | Verified, Source/ directory |
| 585 functions across 17 headers, per-group counts | Verified, generated reference |
| Three paths mapped to M0+/M4/M55 | Verified, README core support table |
| 12x / 8x / 3x on Apollo510 at 96 MHz LP | Verified in repo, new values pending |
| One CMake source feeding four exports | Verified, CI workflows |
| heliaRT and heliaAOT consume it | Verified, docs/index.md |
| MVE MAC throughput (8 int8, 4 int16 and fp16, 2 int32 and fp32) | Not in the repo. Needs an Arm Helium or Apollo510 source of record before it appears. |
| "200+ accelerated ops", "53 op types", "4 paths" | Carried from the old site. Provenance unknown, worth confirming against the generated counts. |
| "40+ field models" | Carried from the old site. This is a deployment claim, not a repo fact. Confirm or drop. |
| What this fork adds over upstream CMSIS-NN | No verified comparison exists. Needs the coverage matrix before any number is claimed. |

## What I need from you

1. Pick or rewrite the six headings.
2. Say whether the coverage argument in section 2 is the right lead. It is the
   strongest idea here and everything else follows from it.
3. Rule on the three unverified claims in the table above.

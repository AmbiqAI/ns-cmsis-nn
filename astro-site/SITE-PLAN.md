# heliaCORE site plan

Historical planning notes. For maintained authoring and publication instructions, see [README.md](README.md).

The product site explains the library, helps users integrate it, and documents
its kernels. The owner requested content and layout improvement alongside the
Astro/Starlight migration. The annotated Home review supersedes the earlier
seven-section navigation and six-heading draft.

## Primary navigation

| Section | Contents |
|---|---|
| Home | Product purpose, production-model operator coverage, data types, acceleration, build integration, HELIA context |
| Getting started | CMake, CMSIS-Pack, Zephyr, neuralSPOT-X, and toolchain setup |
| User guide | Operator coverage, architecture, data types, build configuration, benchmark results and methodology |
| API reference | Searchable kernel index and generated function contracts |

Home has no desktop sidebar. Mobile retains the navigation button and the
four-section menu. Other primary sections have scoped sidebars. Contributor
material remains at its existing routes and is reachable through the footer;
it is not a primary user journey. Existing architecture, coverage, and
performance URLs redirect to their corresponding `guide/` routes.

## Content and layout

- Keep headings direct and specific. No "What it is" or "Why" wrapper sections.
- Explain integration choices before presenting links. Every major section
  provides a next step into detailed documentation.
- Cover production-model operator diversity, not "Ambiq models" or just a
  small benchmark suite. Distinguish implemented operators from an assertion
  that every dtype and target has an optimized path.
- Show A8W8, A16W8, FP16, FP32 with meaningful labels. Keep the repository's
  experimental/opt-in qualification for the float APIs.
- Explain DSP and MVE benefits; preserve benchmark numbers in their detailed
  context until the owner captures replacements.
- helia-ui provides the shared stack, shell, parts, and style tokens. Local
  compositions and token-based layout styles are allowed. Promote components
  when useful across products, not merely because one page needs them.
- Public claim evidence and landing structure: `LANDING-PAGE-CONTENT.md`.

## Follow-up

Review the remaining guide copy and diagram proposals. The kernel SSoT contract
for prerequisites, sizing rules, tolerances, and paths remains follow-up #525.
Cut-over #521 needs built-site approval and changes to both docs and release
Pages deployment paths. Product news and cross-product storytelling stay on
the HELIA Dev Hub.

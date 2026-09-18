# HANDOFF: heliaCORE docs migration

## Goal and authorization

Replace the Sphinx site with Astro/Starlight on independent helia-ui, including content, navigation and API improvements. Owner accepted the site and approved completing publication, cutover and Sphinx retirement. Existing issues #516/#519/#524/#521 and PR #522 cover this work. New benchmark and API-contract issues remain unapproved drafts.

## Workspace and source

- Worktree: `/Users/adam.page/Ambiq/helia/helia-core/.claude/worktrees/helia-ui-migration`
- Branch `docs/helia-ui-migration`; PR https://github.com/AmbiqAI/ns-cmsis-nn/pull/522
- Content overhaul committed as 4c42de2a; main112f487e merged in 239074c4. Cutover published in b276e520; PR is ready for review.
- Source `astro-site/`; preview http://127.0.0.1:4321/ns-cmsis-nn/ serves its dist.
- Do not edit the root checkout or other tasks' worktrees.

## Decisions

Four primary sections: Home, Getting started, User guide, API reference. Home leads with Ambiq silicon and credits Arm CMSIS-NN without adversarial comparisons. It covers operator extensions, DSP/MVE, numeric formats and selective SSoT builds. heliaAOT is recommended for complete models; heliaRT is the optimized LiteRT MCU interpreter path. Both use heliaCORE.

Getting started follows Overview, Requirements, Integration (CMake/CMSIS-Pack/Zephyr/NSX), First kernel. User guide follows Using kernels, Configuration and Performance. Compact dropdowns support kernel browsing; no header-file filter wall. Editorial notes are excluded from site content.

Benchmarks use 28 GCC integer workloads. Highlights use one significant figure without approximation symbols; detailed tables retain precision. No inferred FP16, compiler, PMU, footprint or whole-model measurements.

Keep existing authored redirects. Unknown URLs return a custom 404 then redirect to Home after five seconds, with immediate Home/API links. Exhaustive old generated API URL/anchor mapping is explicitly dropped. Missing fragments on existing pages may land without scrolling.

## Shared package: released and consumed

helia-ui owns its CI, gallery and releases. Dev Hub issue30 is closed; it is an ordinary consumer. Core pins v0.1.0-alpha.13, release commit d3635e977a5c668c9457415d5f67941aab70304d. All139installed files matched release source after clean install; no node_modules patches.

Shared fixes include code-frame separators, external-link icons, underline offset, responsive API names/tables, C signatures and Direction labels. Shared release CI, package tests and gallery deployment passed. Do not reimplement those fixes locally.

## API and deployment implementation

Doxygen1.17 feeds shared doxyref during prebuild. Exact coverage check verifies404public declarations; the full585function model includes helpers. Generated MDX/JSON are ignored and regenerated, with public-content and internal-link checks.

The docs workflow builds, checks types/coverage and runs rendered browser tests, then uploads the tested site. Main deployments consume that same artifact. Release publishing calls the workflow at the release commit independently of pack/library builds. Historical asset recovery cannot publish docs. CMSIS-Pack Doxygen/tooling behavior remains intact.

Removed Sphinx renderer, dependencies, extension, CSS and JS. Preserved authored Markdown and Documentation pack sources. The header-classification guard now imports scripts/docs/api_groups.py independently of Sphinx.

## Verified locally

- Full build:92pages plus redirects,134HTMLfiles,13,975links and3,528fragments, zero broken.
- Astro check:0errors/warnings/hints; exact public coverage regression passes.
- Eight browser tests: Home/first-kernel/guide/index/API at1437/390/320px in light/dark, no overflow or page errors; authored redirects and timed404 fallback pass. Screenshots inspected.
- API classification tests11pass; stale version, actionlint, release recovery and pack-tooling checks pass.
- Pre-commit all-files passes, including executable-bit corrections.
- Earlier C/C++ first-kernel, pooling and requantization examples ran on host. No new hardware acceptance or complete pack/Zephyr/NSX firmware validation.

Evidence: `/private/tmp/heliacore-takeover/cutover-*.log`, browser screenshots under ignored `astro-site/test-results/`. Local complete build took7.8seconds with dependencies installed. Historical GitHub docs build step took17m12s across three Sphinx passes; new GitHub docs job passed in99seconds, including a12second API/site build (run35295199316).

## Next steps

1. PR522 was pushed while draft and promoted with scripts/publish_pr.py. First full CI found three stale release extra-files entries for pages that no longer carry version pins; removed them and verified PDSC/stale-version checks. Publish this correction without toggling draft.
2. Verify exact-head full required CI. Docs passed and screenshots from the GitHub artifact were inspected. Remaining kernel/toolchain jobs were still running. Repository rules require one approving review and CI Passed; do not bypass either.
3. Merge, verify Pages deployment and production rendering/search/API/404 behavior. Record GitHub docs timing separately from local timing.
4. Bring this handoff current with shipped versus verified status.

Internal drafts and prior review evidence are preserved under ignored `astro-site/.cache/internal-notes/` and `astro-site/.cache/TAKEOVER-REVIEW.md`. Source-contract gaps include undocumented public parameters and an output_offset sign comment; expanded benchmark measurements are also drafted. Do not publish those drafts or expose internal planning on the site.

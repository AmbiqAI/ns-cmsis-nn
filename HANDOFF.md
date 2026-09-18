# HANDOFF: heliaCORE docs migration

## Goal and authorization

Replace Sphinx with an Astro/Starlight product site using shared helia-ui; improve content, navigation and visual presentation. Owner approved local implementation and independent release-readiness reviews. GitHub writes still require findings and approval. Product migration remains uncommitted and undeployed. Shared-package publication is authorized; helia-ui is authoritative and independent. Hub PR29 is closed and superseded. Sphinx remains production.

## Workspace and tracking

- Worktree: `/Users/adam.page/Ambiq/helia/helia-core/.claude/worktrees/helia-ui-migration`
- Branch `docs/helia-ui-migration`, HEAD af724c78, extensive intentional dirty changes.
- Parent #516, API #519, structure #524, cutover #521; draft PR #522 open.
- Root main3cb6f74e is release7.35.1; migration source/examples7.35.0. Reconcile before publication, don't edit root blindly.
- Source `astro-site/`; never root `site/`, which Sphinx deletes.
- Preview http://127.0.0.1:4321/ns-cmsis-nn/ serves Astro dist.
- helia-ui pinned alpha.13 at d3635e9; developer-hub paired checkout belongs to other work.

## Content decisions and local implementation

Four top-level sections: Home, Getting started, User guide, API reference. Desktop Home has no sidebar; mobile menu retained. Home accepted by owner. Leads with Ambiq silicon, credits Arm CMSIS-NN, describes operator extensions without adversarial comparisons, promotes SSoT/selective builds, DSP/MVE, and four numeric formats. No generic Why/What sections or chip dumps.

heliaAOT is recommended for model deployment; heliaRT is optimized LiteRT MCU interpreter path. Both use heliaCORE. Guide overview uses compact shared-component cards over a heliaCORE foundation strip in local DeploymentPaths.astro.

Getting started: Overview, Requirements, Integration(CMake/CMSIS-Pack/Zephyr/NSX), First kernel. Source-verified options, float definitions, ABI caveats, package selection and troubleshooting. C/C++ first-kernel example runs on host. CMSIS-Pack float configuration added in final review; no full firmware validation.

User guide: Using kernels(coverage/data types/quantization/calling/memory), Configuration(build options/targets/toolchains), Performance(acceleration/benchmarks/measurement/validation). Source-grounded contracts and runnable pooling/rounding examples. Old moved Astro URLs retained through redirects.

Benchmarks:28GCC integer workloads; highlight rounding is ONE significant figure with no approximation/equal symbol. Full tables/CSV retain precision. Local spotlight/chart components show derived speedups; no invented extra measurement dimensions. Expanded GCC/ATfE/ACfE, FP16/FP32, footprint, setup, PMU and correctness measurement issue draft awaits approval. Tester source reviewed at f2a73596; no new board measurements. Reserve unreleased heliaDSP/heliaML for later hub work.

## API and build

Doxygen1.17 -> shared doxyref -> MDX/model/navigation/coverage during prebuild. All404public declarations match exact header name sets; full model585functions includes helpers. Kernel browser/family lists use public kernels only. Internal support page retained for contracts but removed from search. C parameter blanks originate in headers, not extraction.

Markdown callouts use shared Callout via scripts/markdown-callouts.mjs. Internal editorial notes live outside content; check-public-content runs after API generation. Postbuild writes physical Sphinx .html redirects and checks internal links/fragments. CI Astro job now runs type and coverage regression checks locally staged, still review artifact only.

## Shared-package ownership and release

helia-ui owns its code, CI, gallery, release preparation and publication. Dev Hub is an ordinary tagged consumer. No subtree mirroring. Hub issue30 tracks decoupling; helia-ui issues38/12/67/59 cover releases and shared fixes.

Published https://github.com/AmbiqAI/helia-ui/releases/tag/v0.1.0-alpha.13 at d3635e977a5c668c9457415d5f67941aab70304d. Fix PR103 and version PR104 merged. Exact-main CI35289523576, Publish35289796054 and gallery deployment35289758709 passed. Local validation181unit tests/148gallery browser tests plus installed tarball tests. Version PR's initial automatic CI needed approval; explicitly dispatched full CI35289218808 passed before merge.

Shared package contains code-frame separator, external-link icons, underline offset, responsive API names/tables, signature comma cleanup, and C parameter Direction labels. Alpha.13 removes neighboring-document access, adds a public discoverability CLI, removes monorepo paths and stale diagnostics, and tests props/token-migration isolation. Active source worktree `.cache/helia-ui-independent`, branch codex/standalone-tooling. Old `.cache/helia-ui-frame-fix` and `.cache/helia-hub-release` are preserved historical artifacts; hub PR29 closed/superseded.

heliaCORE dependency and lockfile now pin alpha.13. Clean npm ci completed; all139installed shared files match release source exactly, no patches. Build/API generation, Astro diagnostics, coverage regression, links,15responsive page checks and legacy redirects passed. Screenshots inspected; header icon/underline/code separator verified. Hub may upgrade its alpha.12 pin and use `helia-ui-check-discoverability --root .`; that consumer work belongs to its owner.

## Latest verification

Three independent agents reviewed content, API and rendered UX. Findings/fixes recorded in `astro-site/internal-notes/release-readiness-review.md`.

Build/check pass,0errors/warnings/hints.133HTML files,13,856internal links,3,523fragments,0broken. Exact404public symbols;585generated function signatures with0trailing commas.26shared extractor/rendering tests pass; name-set negative regression and missing-link negative checks pass.

UX40sample states across10routes desktop/mobile/light/dark. After fixes15responsive checks across1437/390/320px,0overflow/JS errors. Search excludes Internal support, kernel browser404functions; representative legacy .html redirects work. Screenshot inspected. Earlier strict-C/C++ first-kernel, pooling and requantization examples pass host execution. No hardware acceptance, full pack/Zephyr/NSX firmware build or product-site GitHub CI/deployment run. Shared-package CI/release/deployment passed as above.

Evidence `/private/tmp/heliacore-takeover/`: readiness logs/scripts, release-ux files, release-api-mobile-fixed.png, shared-api-tests.log and earlier example validation artifacts. Playwright comes from developer-hub node_modules, cached headless shell1243; do not install browsers.

## Approved cutover in progress

Owner approved completing the migration, including publication and Sphinx retirement. Keep authored redirects, use a custom 404 that redirects unknown pages to Home; missing fragments on existing pages may land without scrolling. Exhaustive Exhale mapping is no longer required. Tracking #521. Dev Hub #30 is closed. Next: reconcile main, switch both workflows, validate the exact build in CI and publish.

## Next actions and publication gates

1. Shared-package release and clean consumption COMPLETE. Product migration remains local/uncommitted.
2. Map old Exhale API deep links and changed Sphinx fragments from actual old inventory. Authored .html page redirects now present; generated API compatibility is incomplete.
3. Reconcile newer main and version refs, rerun validation on publication head.
4. Complete #521 BOTH docs.yml and release.yml to avoid Sphinx redeployment. Preserve pack Doxygen and historical recovery restrictions; document rollback. Obtain approval before GH writes/cutover.
5. Source-contract backlog:107public functions have undocumented params,769/2845blank descriptions. Draft api-contracts-issue-draft.md includes warning-baseline/regression gate. Existing quantization-contract-issue-draft.md records incorrect output_offset sign comment. Do not invent descriptions.
6. Expanded benchmark issue remains draft, no approval yet. Internal notes/drafts must never become public site pages.

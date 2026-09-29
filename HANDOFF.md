# CORE documentation consistency

## Goal and authorization

Issue #572, PR #579, branch codex/product-site-consistency. User authorized resolving review findings and admin squash merging helia-ui, then CORE, RT and AOT. Do not control user windows or browser tabs; use source review and headless tests.

## Implemented and verified

Consistent black hero, capability panel, pinned release chip, mobile action placement, discovery chips, outlined section actions and shared footer branding. Kernel catalog uses shared ReferenceBrowser. API and kernel-index Markdown exports preserve catalog entries and C contracts.

Luna review checked docs claims and catalog behavior. User corrected stale documentation: floating-point APIs are no longer experimental. README and public product docs now describe opt-in requirements separately; upstream status and platform/toolchain limits remain. Catalog export gap fixed in 28f7e2cc. Build/type checks and 11 headless browser tests passed. Earlier coverage tests passed 2/2. No hardware execution claim.

## Release sequence

helia-ui #178 merged at 90a6f665; version-only release PR #179 merged at 36922159. Official alpha.20 publication is waiting for exact-main CI. Next: pin astro-site to immutable alpha.20, regenerate lock with npm, prove clean installation, rebuild and run checks/tests. Push draft #579, mark ready, admin squash merge after validation, then RT #323 and AOT #518.

## Boundaries

Worktree /Users/adam.page/Ambiq/helia/helia-core-consistency. No product merge or deployment yet. Other worktrees and primary checkout must be preserved.

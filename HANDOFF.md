# Documentation UI rollout

Goal: adopt released shared UI alpha.22 for mobile section navigation and compact tab terminals.

Issue: https://github.com/AmbiqAI/ns-cmsis-nn/issues/636 (creation approved). Branch codex/mobile-ui-rollout, isolated worktree /Users/adam.page/Ambiq/helia/helia-core-ui-rollout. Existing checkouts untouched.

Dependency and npm-regenerated lock pin alpha.22 commit a62e8d45505dd3bbcdf1c4a03dfd1ec863322ecf. Clean installation, build and type checks passed. Preserve product content, URLs, branding, runtime and package release workflows. Check mobile sidebar scope, keyboard switching, desktop rendering and terminal copy controls.

Local documentation output checks and 16 browser cases passed. Rendered mobile menus inspected at 390px; light/dark and desktop checks passed. Existing test expectations updated where they assumed all sections in the sidebar. Next: publish focused PR and verify CI. No merge or deployment authorized.

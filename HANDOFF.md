# Hero brand sizing

Goal: align the product identity in the landing hero with KIT sizing.
Tracking: AmbiqAI/helia-ui#199. Branch: codex/hero-brand-sizing.
Implemented and verified locally: shared brand style, responsive 28–38.4px names, 32px loaded official icons. Mobile/desktop light/dark rendering has no overflow. Fresh locked installation, type checks, production build, and applicable reference/link contracts pass against released alpha.26.
Dependency: alpha.26 from c84a28eb2b66407abba03aa945cac05bee593c40. Shared PRs #200/#201 merged; merged-main CI and publication passed.
Decision: use shared brand styling and remove local size overrides; ordinary eyebrows stay compact. CORE's lock was regenerated and installed on Linux before a fresh macOS install. NSX locks the exact release source SHA.
Next: consumer PR CI, resolve findings, squash merge only at a green reviewed head, then verify deployed pages. User authorized merging this rollout. Preserve local preview and unrelated files.

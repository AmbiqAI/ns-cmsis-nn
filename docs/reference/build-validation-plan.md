# Build and test coverage proposal

This proposal covers five remaining build and harness requirements. It does
not describe checks that already exist as new coverage. The specialization-off
compile cell and macro-aware enum checker are separate changes in PR #807.

| Requirement | Owning area | Proposed acceptance |
| --- | --- | --- |
| #784 integer-only MVE configuration | Existing float source guards and compile configuration | The selected integer-only MVE configuration compiles with both float APIs disabled; incompatible float declarations and intrinsics do not leak into that configuration. |
| #340 word-copy execution | Existing Unity build and execution harness | Eight explicitly selected existing suites execute under ATfE/M4 with ARM_NN_WORD_COPY; compiler, flags and executed backend are recorded separately from compile-only coverage. |
| #400 generated Unity configuration | Tests/UnitTest/CMakeLists.txt and existing generator | Configuration writes generated Unity files into the binary directory; two independent build directories leave tracked source files unchanged. |
| #400 bias dimensions | Remaining integer Unity call sites | Every selected call supplies initialized bias dimensions matching the test's actual bias shape. |
| #356 version rehearsal | Existing release-contract and version checks | A simulated version update detects stale test and help fixtures without changing the package version, creating tags, or publishing a release. |

## Boundaries

The implementation would change existing build, generator and test owners only.
Any production edit is limited to an owning feature guard required by a proven
integer-only compile failure. Numerical formulas, supported model domains and
public ABI remain outside this proposal. A numerical failure returns a separate
scope decision rather than expanding this work.

No new compiler family, broad native matrix, operator, kernel manifest, coverage
service, CI threshold, release process execution or legacy-to-tester port is
included. Existing compiler coverage and the already-completed word-copy slice
remain intact; the selected missing leg must add actual execution evidence.

## Selection before implementation

Freeze the exact integer-only macro/compiler configuration and its failing
translation units before changing guards. Bind the eight missing word-copy
suite names to the retained coverage manifest and verify toolchain availability.
If no actual missing leg remains, record that requirement as already covered.

For generator and bias changes, identify the current tracked output and affected
call sites first. A stale historical report alone is insufficient. Select a
temporary version fixture accepted by the current release contract; do not edit
the repository's active version.

## Causal controls and success

Reuse existing suites and fixtures once. Each corrected requirement must show
its original failure and corrected result, plus the ordinary supported control.
For generated files, preserve before/after source-tree hashes and configure two
distinct binary directories. For the version fixture, restore one known stale
reference and require the rehearsal to reject it.

Record compiler, core, feature macros and execution environment with results.
Compile-only evidence is not runtime evidence. Unsupported or unavailable
compiler execution is an explicit unresolved dependency.

## Integration and qualification

PR #807 owns only the specialization-off cell and enum checker. Coordinate
actual overlapping workflow or generator hunks after that candidate is frozen;
do not copy its existing implementation into a new branch. The current proposal
document is based on main and contains no workflow or production-source edits.

One initial tooling review covers the complete implementation. A distinct
targeted reader is needed only if feature-guard or harness behavior changes
leave material correctness risk. The responsible reader checks affected fixes;
unchanged evidence is reused. No board is required for documentation and build
logic alone. Material executable code or resource changes require a separate
bounded cost decision.

Implementation success means every admitted requirement has specific causal and
valid-control evidence, source configuration is preserved, and the added
execution leg is truthful. This proposal does not itself approve implementation
or establish CI, merge or release readiness.

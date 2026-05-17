# Contributing

Thanks for considering a contribution to heliaCORE.

## Branching & PRs

- Fork the repo or, if you have push rights, create a topic branch
  off `main`.
- Open a PR against `main`. CI runs unit tests, integration tests,
  and dry-run pack/static-lib builds.

## Conventional commits — required

heliaCORE uses [release-please](https://github.com/googleapis/release-please)
to manage versions. Your PR title (the squash-merge commit subject)
**must** be a [Conventional Commit](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>
```

| `<type>` | Triggers a release? | Use for                                              |
|----------|---------------------|------------------------------------------------------|
| `feat`   | ✅ minor bump        | New user-facing functionality.                       |
| `fix`    | ✅ patch bump        | Bug fixes.                                           |
| `feat!`  | ✅ major bump        | Breaking change (also note in body).                 |
| `perf`   | no                  | Performance improvements that don't change behavior. |
| `refactor`| no                 | Internal refactors.                                  |
| `docs`   | no                  | Docs-only changes (including this site).             |
| `test`   | no                  | Test changes.                                        |
| `ci`     | no                  | CI/CD changes.                                       |
| `build`  | no                  | Build system / packaging.                            |
| `chore`  | no                  | Everything else.                                     |

:::{warning} Avoid XML-reserved characters in titles
Don't put a literal `&`, `<`, or `>` in your PR title. release-please will
faithfully copy it into `CHANGELOG.md` and the GitHub Release body, and
historically that has tripped up downstream artifact generators. Use the word
(`and`) or HTML entity if needed.
:::

## Maintainer release notes

Most contributors only need conventional commits. Maintainers should also know
how the automated release flow works and how to recover when packaging or CI
fails after a tag is cut.

### Release flow

1. Merge feature and fix PRs to `main`. Squash-merge subjects must use
  conventional commits (`feat:`, `fix:`, `feat!:`, and similar).
2. release-please opens or updates a Release PR that bumps the version in
  `Ambiq.NS-CMSIS-NN.pdsc`, `.release-please-manifest.json`, and
  `CHANGELOG.md`.
3. Review the Release PR. The body shows every commit that will ship and the
  resulting version.
4. Merge the Release PR. release-please creates the `vX.Y.Z` tag and GitHub
  Release, then `.github/workflows/release.yml` builds and uploads artifacts.
5. Watch the run. When it is green, the Release is consumer-ready.

### Release recovery

If `publish-pack` fails with `xmlParseEntityRef: no name`, the GitHub Release
body likely contains an XML-reserved character such as `&`, `<`, or `>`. The
pack generator now emits pdsc release entries from the tag only, but for an
already-broken release you can edit the Release body and rerun failed jobs:

```bash
gh release view vX.Y.Z --json body -q .body > /tmp/body.md
sed -i 's/&/and/g' /tmp/body.md
gh release edit vX.Y.Z --notes-file /tmp/body.md
gh run rerun <run-id> --failed
```

If release tests fail with `manifest unknown` while pulling
`ghcr.io/ambiqai/ns-cmsis-nn-ci:vX.Y.Z`, publish the missing CI image tag and
rerun the failed release jobs:

```bash
gh workflow run build_publish_docker.yml --ref main \
  -f image_tag=vX.Y.Z -f publish_latest=false
gh run rerun <release-run-id> --failed
```

`gh run rerun` re-executes the original commit. To include fixes that landed on
`main` after the original release run, cut the next release or add a deliberate
`workflow_dispatch` path to the affected workflow and run it against `main`.

Useful release commands:

```bash
# Watch the latest release.yml run
gh run watch "$(gh run list --workflow release.yml --limit 1 --json databaseId -q '.[0].databaseId')"

# See the first useful failures from a run
gh run view <run-id> --log-failed | grep -iE 'error|fail|##\[error\]' | head -40

# Re-run only failed jobs
gh run rerun <run-id> --failed

# Confirm a release has all expected assets
gh release view vX.Y.Z --json assets -q '.assets[].name' | sort
```

## Tests

- C unit tests live under `Tests/UnitTest/`.
- CMake integration tests under `cmake/tests/`.
- Smoke + NSX-link tests run on every PR via `.github/workflows/`.

Before you push:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## Formatting

Source and public header files under `Source/` and `Include/` use the checked-in
`.clang-format`. The repository intentionally does not require a one-shot format
baseline across all inherited CMSIS-NN sources; instead, formatting is enforced
only on files touched by a PR so the tree converges gradually without creating a
large upstream-sync diff.

Install the optional pre-commit hook to format staged C/H files before commit:

```bash
python -m pip install pre-commit
pre-commit install
```

The dev container installs and enables this hook during setup.

To check the same changed-file range CI checks:

```bash
python -m pip install pre-commit
bash scripts/check_clang_format_changed.sh origin/main HEAD
```

## Reporting bugs

Open an issue at
[github.com/AmbiqAI/ns-cmsis-nn/issues](https://github.com/AmbiqAI/ns-cmsis-nn/issues).
Please include:

- heliaCORE version (`vX.Y.Z`)
- Target CPU and toolchain
- The minimal failing command / CMake invocation
- The full error output

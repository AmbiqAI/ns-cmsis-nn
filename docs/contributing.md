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

If an entire release run failed before uploading pack/bundle/tarball assets
(for example the CI image failed to build, or every `armclang` static-lib leg
failed because of an unrelated licensing problem), `gh run rerun --failed`
may not be enough, since the failed jobs' `needs:` graph can be stuck on a
job (`release-please`) that only runs once per tag. For that case,
`release.yml` supports a manual, idempotent recovery dispatch that targets an
**existing, already-published** tag without ever creating, moving, or
re-publishing it:

```bash
gh workflow run release.yml --ref main -f recover_tag=v7.29.2
```

This re-runs `resolve-release-capabilities`, `publish-staticlibs`,
`publish-staticlibs-armclang`, `publish-staticlib-bundles`, and
`publish-pack` against the release already
published at `v7.29.2`, re-uploading (`--clobber`) any missing or stale
customer assets. Historical recovery deliberately skips the CI image and its
container test jobs because that image is build infrastructure rather than a
GitHub Release asset.
It refuses to run
(fails fast) if `v7.29.2` doesn't already have a published GitHub Release, so
it cannot be used to create a new tag/release under a different name. See
[Required vs optional assets](guides/releases.md#required-vs-optional-assets)
for which assets are safe to be missing (armclang, unless the repository
variable `ARMCLANG_REQUIRED` is set to `true`) versus which indicate a real
regression.

`release-please`'s job resolves `v7.29.2` to its exact target commit exactly
once (`scripts/ci/resolve_release_commit.sh`, via the GitHub API) and
publishes it as the `commit_sha` job output; every recovery asset job checks
that exact commit out (`ref: needs.release-please.outputs.commit_sha`) rather
than whatever ref was selected when dispatching the recovery run. Because
recovery never invokes `publish-ci-image`, it cannot publish a versioned image
or repoint the `:latest` alias. If you need to independently confirm which
commit a recovered tag's assets were built from:

```bash
gh api repos/AmbiqAI/ns-cmsis-nn/commits/v7.29.2 -q .sha
```

Two runtime-only failure modes were fixed for recovering genuinely old
tags (AmbiqAI/ns-cmsis-nn#228): `publish-pack` no longer fails with
`Tag has no annotation message` against Release Please's lightweight tags
(a local-only compatibility step, `scripts/ci/ensure_local_tag_annotation.sh`,
annotates the tag in the runner's local checkout only — see
[Recovering assets for an existing tag](guides/releases.md#recovering-assets-for-an-existing-tag));
and `publish-staticlib-bundles`'s `gh release upload` no longer fails with
`fatal: not a git repository` in its no-checkout job, since every `gh
release` call now passes `--repo` explicitly instead of relying on git
remote inference.

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

## Pre-commit hooks

`.pre-commit-config.yaml` is the commit-time hygiene gate, and CI runs part of
the same config over the whole tree. From a fresh clone:

```bash
uv tool install pre-commit==3.8.0
pre-commit install
```

`python -m pip install pre-commit==3.8.0` works as well if you would rather not
use `uv`. The dev container does this during setup.

The hooks check:

- whitespace hygiene: trailing whitespace and a single final newline;
- no leftover merge conflict markers, and no newly added file over 500 kB;
- syntax of YAML, JSON and TOML files;
- `clang-format` over `Source/` and `Include/` C and headers (see below);
- deferred-work markers. Write `TODO(#421): drop the workaround once the pack
  ships`, or the same shape with `FIXME(#421)` or `HACK(#421)`; the bare words
  are rejected. `TODO(verify)` is the only other accepted form, and it means
  the claim next to it has not been checked against a source of record yet, so
  it must not survive review.

Nothing here scans for secrets. Credentials are caught server side by GitHub
secret scanning with push protection, which rejects the push itself and cannot
be bypassed by a local flag.

### Two scopes

`clang-format`, `trailing-whitespace` and `end-of-file-fixer` rewrite files, so
they run at commit time over the staged files only, and the CI job skips them
with `SKIP`. That is the same policy `clang-format` already follows: the tree
converges as PRs touch files, rather than through one large reformat that would
collide with every upstream sync. Those three hooks additionally skip content
we must not rewrite at all: generated test vectors under
`Tests/UnitTest/TestCases/`, and the files we still carry byte-identical from
Arm, which the config lists.

The remaining hooks only report, so CI runs them over every tracked file,
including the roughly half of `Tests/` that is Ambiq-owned. The size check is
the exception: it looks only at files being added to the index, so it gates the
commit, not the CI run. Because the hooks see only staged files at commit time,
a CI run over the wider scope can fail on files you never touched; when that
happens, fix the reported file rather than widening an exclude.

Bump a hook `rev` with `pre-commit autoupdate` in its own reviewed PR. Every
remote rev is an exact tag; do not point a hook at a branch.

## Formatting

Source and public header files under `Source/` and `Include/` use the checked-in
`.clang-format`. The repository intentionally does not require a one-shot format
baseline across all inherited CMSIS-NN sources; instead, formatting is enforced
only on files touched by a PR so the tree converges gradually without creating a
large upstream-sync diff.

The pre-commit `clang-format` hook formats staged C/H files under `Source/` and
`Include/` when you commit, so the files a PR touches arrive formatted. CI
checks formatting only over the changed-file range, never the whole tree. To
run the same check locally:

```bash
python -m pip install pre-commit==3.8.0 clang-format==16.0.6
bash scripts/check_clang_format_changed.sh origin/main HEAD
```

CI enforces clang-format 16 (the pre-commit pin); the script refuses other
majors because they disagree on committed files. If a different clang-format
is first on your `PATH`, run the script from the environment where you
installed the pinned one, or point it there explicitly:
`CLANG_FORMAT_BIN=/path/to/venv/bin/clang-format bash scripts/check_clang_format_changed.sh origin/main HEAD`.

## Reporting bugs

Open an issue at
[github.com/AmbiqAI/ns-cmsis-nn/issues](https://github.com/AmbiqAI/ns-cmsis-nn/issues).
Please include:

- heliaCORE version (`vX.Y.Z`)
- Target CPU and toolchain
- The minimal failing command / CMake invocation
- The full error output

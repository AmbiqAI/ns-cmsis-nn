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

!!! warning "Avoid XML-reserved characters in titles"
    Don't put a literal `&`, `<`, or `>` in your PR title. release-please
    will faithfully copy it into `CHANGELOG.md` and the GitHub Release
    body — and historically that has tripped up downstream artifact
    generators. Use the word (`and`) or HTML entity if needed.

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

## Reporting bugs

Open an issue at
[github.com/AmbiqAI/ns-cmsis-nn/issues](https://github.com/AmbiqAI/ns-cmsis-nn/issues).
Please include:

- heliaCORE version (`vX.Y.Z`)
- Target CPU and toolchain
- The minimal failing command / CMake invocation
- The full error output

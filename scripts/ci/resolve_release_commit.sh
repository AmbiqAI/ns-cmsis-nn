#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Resolves a release tag to its exact, immutable target commit SHA via the
# GitHub REST "Get a commit" endpoint (`GET /repos/{owner}/{repo}/commits/{ref}`),
# which transparently peels annotated tags to the commit they point at.
# Retries briefly to absorb the short eventual-consistency window right
# after a tag/release is created.
#
# AmbiqAI/ns-cmsis-nn#228: every job that builds or tests a release asset
# must check out this EXACT commit -- never `github.ref`/`github.sha` of
# whatever ref triggered the workflow run. In particular, a `workflow_dispatch`
# existing-tag recovery run's own triggering ref (typically the default
# branch, selected in the "Run workflow" UI/API) has no required
# relationship to the tag being recovered, so trusting it would silently
# rebuild v7.29.1/v7.29.2 assets from current `main` instead of the commit
# those tags actually point at.
#
# Usage:
#   resolve_release_commit.sh <owner/repo> <tag> <output_file> [max_attempts] [sleep_secs]
#
# Requires: `gh`, authenticated via a GH_TOKEN/GITHUB_TOKEN env var with at
# least read access to the repository.
#
# Writes a single `commit_sha=<40-hex-char-sha>` line to <output_file>
# (GITHUB_OUTPUT-style) on success. Exits non-zero without writing anything
# on any failure (bad tag format, tag never resolves, or a resolved value
# that doesn't look like a full commit SHA).

set -euo pipefail

repo="${1:?repo required (owner/name)}"
tag="${2:?tag required}"
output_file="${3:?output_file required}"
max_attempts="${4:-10}"
sleep_secs="${5:-3}"

# Reject anything that isn't a plain vX.Y.Z release tag before ever shelling
# out to `gh` -- keeps this immune to argument/command injection via a
# maliciously crafted tag string.
if [[ ! "${tag}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "::error::resolve_release_commit.sh: tag '${tag}' does not match the expected vX.Y.Z format." >&2
  exit 1
fi

commit_sha=""
for ((attempt = 1; attempt <= max_attempts; attempt++)); do
  if resolved="$(gh api "repos/${repo}/commits/${tag}" -q .sha 2>/dev/null)" && [[ -n "${resolved}" ]]; then
    commit_sha="${resolved}"
    break
  fi
  echo "Waiting for tag '${tag}' to resolve to a commit via the GitHub API (try ${attempt}/${max_attempts})..."
  if (( attempt < max_attempts )); then
    sleep "${sleep_secs}"
  fi
done

if [[ -z "${commit_sha}" ]]; then
  echo "::error::Could not resolve tag '${tag}' in '${repo}' to a commit via the GitHub API after ${max_attempts} attempt(s)." >&2
  exit 1
fi

if ! [[ "${commit_sha}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "::error::Resolved commit for tag '${tag}' ('${commit_sha}') does not look like a full 40-character commit SHA." >&2
  exit 1
fi

echo "Resolved tag '${tag}' -> commit ${commit_sha}"
echo "commit_sha=${commit_sha}" >> "${output_file}"

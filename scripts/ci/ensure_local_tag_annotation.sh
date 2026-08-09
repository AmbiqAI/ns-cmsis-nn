#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Local-only compatibility shim for gen_pack.sh's PACK_CHANGELOG_MODE=tag
# changelog generation (Open-CMSIS-Pack/gen-pack's lib/gittools), which
# requires the release tag to be an ANNOTATED git tag object with a
# non-empty message (AmbiqAI/ns-cmsis-nn#228). Release Please creates
# LIGHTWEIGHT tags (a ref pointing straight at a commit, no tag object,
# no message) -- so a from-scratch v7.29.1/v7.29.2 pack build fails with
# "Tag has no annotation message." before ever reaching packchk/PDSC schema
# validation.
#
# This script:
#   - No-ops for a tag that is ALREADY an annotated tag object (never
#     touches it -- "handle annotated tags without alteration").
#   - For a lightweight tag, creates a LOCAL-ONLY annotated tag object at
#     the exact same commit the lightweight ref already points at, with a
#     deterministic, non-empty, schema-safe message. This REWRITES ONLY
#     the local ref in the current checkout's .git directory -- it never
#     runs `git push`, so the remote tag (on GitHub) is completely
#     unaffected: it stays exactly the lightweight ref it always was.
#   - Sources the message body from the tag's existing GitHub Release
#     description when available (so the pack's changelog reflects real
#     release notes), falling back to a minimal deterministic message
#     ("Release <tag>") when the release has no body or `gh` cannot reach
#     it. Release-please's auto-generated release notes are Markdown and
#     may contain literal '&', '<', '>' -- gen-pack's PDSC changelog
#     embedding (git_changelog_pdsc) does NOT XML-escape the annotation
#     text itself, so this script XML-escapes those three characters
#     before writing the tag message. This preserves the schema-safety
#     that PACK_CHANGELOG_MODE=tag was originally chosen for (see
#     gen_pack.sh's PACK_CHANGELOG_MODE comment) while still surfacing the
#     real release notes when they're available.
#
# Usage:
#   ensure_local_tag_annotation.sh <tag> <owner/repo> [worktree-path]
#
# <worktree-path> (optional, defaults to ".") is the historical git
# checkout containing <tag> -- i.e. after a `fetch --tags`. This script
# itself is invoked from a CURRENT/tooling checkout (AmbiqAI/ns-cmsis-nn#228,
# live recovery run 31335858426, defect 2: the historical v7.29.2 source
# predates this script's very existence, so it cannot be checked out from
# the pinned historical commit). Passing <worktree-path> explicitly, rather
# than relying on an ambient `cd`, means the script's own on-disk location
# and the repository it operates on are never required to be the same
# checkout -- the "two-tree" recovery architecture depends on that: the
# helper survives being read from a tooling tree while it mutates local
# refs in a separate, immutable historical source tree. Requires `git`;
# `gh` is used best-effort (a missing/failing `gh` falls back to the
# deterministic message, it is not fatal).
#
# Exits non-zero, without changing the tag, if:
#   - <tag> does not exist in <worktree-path>.
#   - the resulting annotated tag's target commit would differ from the
#     original lightweight tag's commit (defensive: this must never move
#     the tag to a different commit).

set -euo pipefail

tag="${1:?tag required}"
repo="${2:?owner/repo required}"
worktree="${3:-.}"

cd -- "${worktree}"

if ! git rev-parse --verify --quiet "refs/tags/${tag}" >/dev/null; then
  echo "ensure_local_tag_annotation.sh: tag '${tag}' not found in worktree '${worktree}'" >&2
  exit 1
fi

objecttype="$(git for-each-ref --format '%(objecttype)' "refs/tags/${tag}")"

if [[ "${objecttype}" == "tag" ]]; then
  echo "ensure_local_tag_annotation.sh: '${tag}' is already an annotated tag; leaving it unaltered."
  exit 0
fi

if [[ "${objecttype}" != "commit" ]]; then
  echo "ensure_local_tag_annotation.sh: '${tag}' has unexpected object type '${objecttype}'" >&2
  exit 1
fi

commit_before="$(git rev-list -n1 "${tag}")"

# Best-effort: pull the existing GitHub Release's body, if any. Never fatal
# -- an unreachable/absent release just falls back to the deterministic
# message below.
body_raw=""
if command -v gh >/dev/null 2>&1; then
  body_raw="$(gh release view "${tag}" --repo "${repo}" --json body -q .body 2>/dev/null || true)"
fi

body_trimmed="$(printf '%s' "${body_raw}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

if [[ -n "${body_trimmed}" ]]; then
  # XML-escape reserved characters (order matters: '&' first, so the
  # entities introduced by escaping '<'/'>' are not themselves re-escaped).
  # gen-pack's PDSC changelog embedding does not do this itself, and
  # release-please's auto-generated notes can contain literal '&'/'<'/'>'.
  body_escaped="$(printf '%s' "${body_trimmed}" \
    | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g')"
  message="Release ${tag}
${body_escaped}"
else
  message="Release ${tag}"
fi

# -f is required to replace the existing lightweight ref with an annotated
# tag object of the same name; this is purely local (no `git push` anywhere
# in this script) and pins the new annotated object to the SAME commit the
# lightweight tag already pointed at, so the tag's meaning never changes.
git -c user.name="ns-cmsis-nn release recovery" \
  -c user.email="opensource@ambiq.com" \
  tag -f -a "${tag}" -m "${message}" "${commit_before}" >/dev/null

commit_after="$(git rev-list -n1 "${tag}")"
if [[ "${commit_after}" != "${commit_before}" ]]; then
  echo "ensure_local_tag_annotation.sh: internal error -- '${tag}' commit changed" \
    "from ${commit_before} to ${commit_after}; refusing to proceed." >&2
  exit 1
fi

new_type="$(git for-each-ref --format '%(objecttype)' "refs/tags/${tag}")"
new_contents="$(git tag -l -n99 --format '%(contents)' "${tag}")"
if [[ "${new_type}" != "tag" || -z "${new_contents}" ]]; then
  echo "ensure_local_tag_annotation.sh: internal error -- '${tag}' is not a" \
    "non-empty annotated tag after local rewrite (type='${new_type}')." >&2
  exit 1
fi

echo "ensure_local_tag_annotation.sh: '${tag}' locally annotated (commit unchanged: ${commit_before}); remote tag not touched."

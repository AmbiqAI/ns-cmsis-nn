#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Audits PUBLISHED GitHub Releases against the required-asset contract
# (AmbiqAI/ns-cmsis-nn#274, section 2 -- the half `release-verify` cannot
# cover). release-verify runs inside the release run itself, so it only
# ever fires when that run reaches it: v7.26.0-v7.29.1 all shipped empty
# BEFORE it existed, a run cancelled outright never executes it, and
# `gh release upload --clobber` deletes an asset before re-uploading so a
# release can lose assets AFTER its run went green. This script inspects
# the live release objects customers actually see, independent of any
# workflow run's fate. The nightly release-assets-audit job runs it daily.
#
# Usage:
#   audit_release_assets.sh <owner/repo> [tag ...]
#
#   With explicit tags: audits exactly those releases (draft/prerelease
#   filtering is bypassed -- you named them).
#   Without: sweeps every published, non-draft, non-prerelease release at
#   or above the contract floor.
#
# Environment:
#   RELEASE_AUDIT_WAIVED_TAGS  Space/comma-separated tags whose asset gaps
#                              are a recorded decision, not a defect
#                              (issue #274 priority 3: "recover ... or
#                              deliberately decide not to"). Waived gaps
#                              are reported but do not fail the audit.
#   AUDIT_REPORT_FILE          Optional path; a Markdown report suitable
#                              for an issue body is written here.
#
# Exit codes: 0 = every audited release satisfies the contract (waived
# gaps included), 1 = at least one unwaived release is missing required
# assets, 2 = usage or GitHub API failure.
#
# Requires `gh`, authenticated with read access to the repository.

set -euo pipefail

# First release cut under the current required-asset contract (pack +
# gcc/atfe SDK tarballs + gcc/atfe staticlib bundles, each with a .sha256
# sidecar = 17). v7.25.0 and older shipped a different, smaller asset
# shape (loose per-cpu .a files, gcc only) and are not judged by a
# contract that postdates them.
readonly CONTRACT_FLOOR="v7.26.0"

# The 17 ALWAYS-required assets, kept in sync with the release-verify job
# in .github/workflows/release.yml and the "Required vs optional assets"
# table in docs/guides/releases.md. armclang's 8 are deliberately NEVER
# required here, even when the repository variable ARMCLANG_REQUIRED
# promotes them for NEW releases: that variable is a point-in-time
# operator switch, and applying it retroactively would flag historical
# releases that shipped complete under the contract of their day.
required_assets() {
  local version="$1"
  printf '%s\n' "Ambiq.NS-CMSIS-NN.${version}.pack"
  local tc cpu
  for tc in gcc atfe; do
    printf '%s\n' "ns-cmsis-nn-staticlibs-${tc}-${version}.zip"
    printf '%s\n' "ns-cmsis-nn-staticlibs-${tc}-${version}.zip.sha256"
    for cpu in cortex-m0 cortex-m4 cortex-m55; do
      printf '%s\n' "ns-cmsis-nn-${cpu}-${tc}-${version}.tar.gz"
      printf '%s\n' "ns-cmsis-nn-${cpu}-${tc}-${version}.tar.gz.sha256"
    done
  done
}

# Retried because this runs unattended on a nightly schedule: one
# transient 5xx must not open a "broken release" alarm at 07:17 UTC, and
# a false alarm would discredit the only rolling signal this contract has.
gh_retry() {
  local attempt out
  for attempt in 1 2 3; do
    if out="$("$@")"; then
      printf '%s\n' "$out"
      return 0
    fi
    if (( attempt < 3 )); then
      echo "gh call failed (attempt ${attempt}/3): $* -- retrying in $(( attempt * 5 ))s" >&2
      sleep "$(( attempt * 5 ))"
    fi
  done
  return 1
}

# $1 >= $2 in version order. sort -V treats the leading 'v' consistently
# as long as BOTH operands carry it, which the tag validation below
# guarantees.
tag_ge() {
  [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)" == "$2" ]]
}

repo="${1:?repo required (owner/name)}"
shift
if [[ ! "$repo" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
  echo "invalid repo '${repo}' (expected owner/name)" >&2
  exit 2
fi

waived="${RELEASE_AUDIT_WAIVED_TAGS:-}"
waived="${waived//,/ }"
# Waiver entries that match no audited tag are reported so an operator can
# see why a waiver is not taking effect (typo, recovered release, stale).

is_waived() {
  local tag="$1" w
  for w in $waived; do
    [[ "$w" == "$tag" ]] && return 0
  done
  return 1
}

tags=()
if (( $# > 0 )); then
  tags=( "$@" )
else
  # Non-draft AND non-prerelease: drafts are not customer-visible, and a
  # prerelease is by definition not published under the GA contract.
  if ! listing="$(gh_retry gh api --paginate "repos/${repo}/releases?per_page=100" \
      --jq '.[] | select((.draft or .prerelease) | not) | .tag_name')"; then
    echo "::error title=Release audit could not run::listing releases for ${repo} failed after 3 attempts." >&2
    exit 2
  fi
  while IFS= read -r tag; do
    [[ -n "$tag" ]] || continue
    # Only plain vX.Y.Z tags are governed by the contract; anything else
    # (should none exist today) is noted rather than silently skipped.
    if [[ ! "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      echo "skipping '${tag}': not a vX.Y.Z release tag" >&2
      continue
    fi
    tag_ge "$tag" "$CONTRACT_FLOOR" && tags+=( "$tag" )
  done <<< "$listing"
fi

if (( ${#tags[@]} == 0 )); then
  # An empty sweep is never legitimate while eligible releases exist: a
  # listing that succeeds but yields nothing (API shape drift, a jq filter
  # gone stale) would silently disarm this audit -- the failure class it
  # exists to end. Fail loud instead.
  echo "no releases at or above ${CONTRACT_FLOOR} to audit -- refusing to report green on an empty sweep" >&2
  exit 2
fi

report_lines=()
failed_tags=()
waived_tags_hit=()

for tag in "${tags[@]}"; do
  if [[ ! "$tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "invalid tag '${tag}' (expected vX.Y.Z)" >&2
    exit 2
  fi
  version="${tag#v}"
  if ! published="$(gh_retry gh release view "$tag" --repo "$repo" \
      --json assets --jq '.assets[].name')"; then
    echo "::error title=Release audit could not run::reading assets of ${tag} failed after 3 attempts." >&2
    exit 2
  fi

  missing=()
  while IFS= read -r asset; do
    grep -Fxq -- "$asset" <<< "$published" || missing+=( "$asset" )
  done < <(required_assets "$version")
  total="$(required_assets "$version" | wc -l)"

  if (( ${#missing[@]} == 0 )); then
    echo "OK      ${tag}: all ${total} required assets present"
    report_lines+=( "| \`${tag}\` | OK | ${total}/${total} |" )
  elif is_waived "$tag"; then
    echo "WAIVED  ${tag}: missing ${#missing[@]} of ${total} required assets (RELEASE_AUDIT_WAIVED_TAGS)"
    printf '        missing: %s\n' "${missing[@]}"
    report_lines+=( "| \`${tag}\` | waived | $(( total - ${#missing[@]} ))/${total} |" )
    waived_tags_hit+=( "$tag" )
  else
    echo "MISSING ${tag}: missing ${#missing[@]} of ${total} required assets"
    printf '        missing: %s\n' "${missing[@]}"
    report_lines+=( "| \`${tag}\` | **MISSING ${#missing[@]} of ${total}** | $(( total - ${#missing[@]} ))/${total} |" )
    failed_tags+=( "$tag" )
  fi
done

write_report() {
  local dest="$1"
  {
    echo "| release | status | required assets present |"
    echo "| --- | --- | --- |"
    printf '%s\n' "${report_lines[@]}"
    if (( ${#failed_tags[@]} > 0 )); then
      echo
      echo "These releases are public and non-draft, so they are broken as"
      echo "customers see them. To repair one, re-run asset publication"
      echo "against the existing tag (never creates or moves a tag):"
      echo
      echo '```bash'
      printf 'gh workflow run release.yml --ref main -f recover_tag=%s\n' "${failed_tags[@]}"
      echo '```'
      echo
      echo "To instead record a deliberate decision NOT to repair one, add"
      echo "its tag to the repository variable \`RELEASE_AUDIT_WAIVED_TAGS\`."
      echo "See docs/guides/releases.md."
    fi
  } >> "$dest"
}

if [[ -n "${AUDIT_REPORT_FILE:-}" ]]; then
  : > "${AUDIT_REPORT_FILE}"
  write_report "${AUDIT_REPORT_FILE}"
fi
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo "### Published-release asset audit"
    echo
  } >> "${GITHUB_STEP_SUMMARY}"
  write_report "${GITHUB_STEP_SUMMARY}"
fi

for w in $waived; do
  hit=0
  for t in "${waived_tags_hit[@]:-}"; do [[ "$t" == "$w" ]] && hit=1; done
  if (( hit == 0 )) && [[ -n "$w" ]]; then
    echo "WARNING: waiver entry '${w}' matched no audited release with missing assets (typo, recovered, or stale -- consider removing it)" >&2
  fi
done

if (( ${#failed_tags[@]} > 0 )); then
  echo "::error title=${#failed_tags[@]} published release(s) missing required assets::${failed_tags[*]} -- each is public and non-draft. Recover with 'gh workflow run release.yml --ref main -f recover_tag=<tag>' or waive via the RELEASE_AUDIT_WAIVED_TAGS repository variable (docs/guides/releases.md)." >&2
  exit 1
fi
echo "audit clean: ${#tags[@]} release(s) checked${waived_tags_hit[0]:+, waived: ${waived_tags_hit[*]}}"

#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Resolves which ghcr.io/ambiqai/ns-cmsis-nn-ci tag(s) build_publish_docker.yml
# should build and push, given the triggering event and the workflow's
# resolved `inputs` values.
#
# Extracted into a standalone, argument-driven script (instead of living
# inline in the workflow YAML) so scripts/check_release_image_tags.sh can
# unit-test the precedence rules locally, without a live GitHub Actions run.
#
# IMPORTANT (AmbiqAI/ns-cmsis-nn#228): callers must NOT branch on
# `github.event_name` to decide which "inputs" source to trust, and must NOT
# read `github.event.inputs.*`. When build_publish_docker.yml is invoked as a
# reusable workflow via `uses:` from another workflow that was itself
# workflow_dispatch-triggered (e.g. release.yml's `recover_tag` recovery
# flow), `github.event_name` inside THIS (called) workflow reflects the
# OUTER caller's triggering event (still "workflow_dispatch") -- but
# `github.event.inputs.*` in that situation refers to the OUTER workflow's
# own dispatch inputs (e.g. `recover_tag`), NOT the `image_tag` /
# `publish_latest` values this workflow was actually invoked with via
# `with:`. Reading `github.event.inputs.image_tag` there silently resolves
# to empty, which used to fall back to the "latest" default and clobber the
# `:latest` alias with a stale/old recovered tag's image.
#
# The unified, top-level `inputs` context is populated correctly in BOTH
# cases: `workflow_call` `with:` values (or their declared defaults), and a
# genuine `workflow_dispatch`'s own form values (or *their* declared
# defaults). See:
# https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#workflow_dispatch
# So `inputs.*` is used unconditionally here, except for `schedule`, which
# has no `inputs` object at all.
#
# Usage:
#   resolve_image_tags.sh <event_name> <input_image_tag> <input_publish_latest> <registry_image> <output_file>
#
# Writes GITHUB_OUTPUT-style `key=value` / `key<<EOF ... EOF` lines to
# <output_file>: image_tag, publish_latest, build_args, docker_tags.

set -euo pipefail

event_name="${1:?event_name required}"
input_image_tag="${2-}"
input_publish_latest="${3-}"
registry_image="${4:?registry_image required}"
output_file="${5:?output_file required}"

if [[ "${event_name}" == "schedule" ]]; then
  # No `inputs` object exists on a schedule trigger.
  image_tag='latest'
  publish_latest='true'
else
  image_tag="${input_image_tag:-latest}"
  publish_latest="${input_publish_latest:-false}"
fi

# Validate before writing to $GITHUB_OUTPUT: a malformed/unexpected
# image_tag (whitespace, newlines, or characters outside the Docker tag
# charset) could corrupt the output file's key=value/heredoc format or
# produce an invalid image reference. Docker tags allow
# [A-Za-z0-9_][A-Za-z0-9._-]{0,127}.
if [[ ! "${image_tag}" =~ ^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "resolve_image_tags.sh: invalid image_tag '${image_tag}'" \
    "(must match Docker tag charset [A-Za-z0-9_][A-Za-z0-9._-]{0,127})" >&2
  exit 1
fi

if [[ "${publish_latest}" != 'true' && "${publish_latest}" != 'false' ]]; then
  echo "resolve_image_tags.sh: invalid publish_latest '${publish_latest}'" \
    "(must be exactly 'true' or 'false')" >&2
  exit 1
fi

tags=("${registry_image}:${image_tag}")
if [[ "${publish_latest}" == 'true' && "${image_tag}" != 'latest' ]]; then
  tags+=("${registry_image}:latest")
fi

{
  echo "image_tag=${image_tag}"
  echo "publish_latest=${publish_latest}"
  echo "build_args=--build-arg NS_CMSIS_NN_TAG=${image_tag}"
  echo 'docker_tags<<EOF'
  printf '%s\n' "${tags[@]}"
  echo 'EOF'
} >> "${output_file}"

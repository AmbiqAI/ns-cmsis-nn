#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Push a reviewed commit before promoting its draft PR. Refs #459."""

import argparse
import json
import re
import subprocess


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def pr_info(number, repo):
    return json.loads(run("gh", "pr", "view", str(number), "--repo", repo, "--json",
                          "state,isDraft,headRefName,headRefOid,isCrossRepository"))


def publish(number, repo, remote, expect_head, dry_run=False):
    if run("git", "status", "--porcelain"):
        raise ValueError("Commit or remove local changes before publishing; no stash is used.")
    head = run("git", "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", expect_head) or head != expect_head.lower():
        raise ValueError("HEAD must match the full SHA supplied from the completed review.")
    info = pr_info(number, repo)
    if info["state"] != "OPEN" or info["isCrossRepository"]:
        raise ValueError("Expected an open PR whose branch is in the base repository.")
    urls = run("git", "remote", "get-url", "--push", "--all", remote).splitlines()
    if len(urls) != 1:
        raise ValueError("Expected exactly one push URL; multiple destinations are unsupported.")
    url = urls[0]
    match = re.fullmatch(r"(?:git@github\.com:|https://github\.com/|ssh://git@github\.com/)(.+?)(?:\.git)?", url)
    if not match or match.group(1).lower() != repo.lower():
        raise ValueError("Push remote does not match the PR repository.")
    branch = info["headRefName"]
    run("git", "check-ref-format", "--branch", branch)
    run("git", "fetch", "--no-tags", "--", remote, branch)
    if run("git", "rev-parse", "FETCH_HEAD") != info["headRefOid"]:
        raise ValueError("Remote branch changed; refresh and review before retrying.")
    run("git", "merge-base", "--is-ancestor", "FETCH_HEAD", head)
    if dry_run:
        print(f"Would push {head} to {remote}:{branch}, verify the PR head, "
              + ("then mark ready." if info["isDraft"] else "without another promotion."))
        return
    run("git", "push", "--", remote, head + ":refs/heads/" + branch)
    current = pr_info(number, repo)
    if (current["state"] != "OPEN" or current["headRefName"] != branch
            or current["headRefOid"] != head or current["isCrossRepository"]
            or current["isDraft"] != info["isDraft"]):
        raise ValueError("PR changed or has not caught up with the push; inspect before retrying.")
    if current["isDraft"]:
        run("gh", "pr", "ready", str(number), "--repo", repo)
    print(f"PR #{number} published at {head}. No merge or CI polling performed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int)
    parser.add_argument("--repo", default="AmbiqAI/ns-cmsis-nn")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--expect-head", required=True, help="Full SHA from the completed review")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        publish(args.number, args.repo, args.remote, args.expect_head, args.dry_run)
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Publication stopped: {error}\n")


if __name__ == "__main__":
    main()

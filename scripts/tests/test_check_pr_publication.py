#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Exercise publication ordering and failure paths without GitHub writes (#459)."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import os
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("publish_pr", Path(__file__).parents[1] / "publish_pr.py")
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


OLD = "b" * 40
NEW = "a" * 40


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.info = dict(state="OPEN", isDraft=True, headRefName="topic",
                         headRefOid=OLD, isCrossRepository=False)
        self.after = dict(self.info, headRefOid=NEW)
        self.dirty = ""
        self.remote = "git@github.com:AmbiqAI/ns-cmsis-nn.git"
        self.fail = None
        self.views = 0
        self.fetch_head = OLD
        self.expected_head = NEW

    def command(self, *args):
        self.calls.append(args)
        if self.fail and args[:len(self.fail)] == self.fail:
            raise subprocess.CalledProcessError(1, args)
        if args[:3] == ("git", "status", "--porcelain"):
            return self.dirty
        if args[:3] == ("gh", "pr", "view"):
            self.views += 1
            return json.dumps(self.info if self.views == 1 else self.after)
        if args[:3] == ("git", "remote", "get-url"):
            return self.remote
        if args[:2] == ("git", "rev-parse"):
            return NEW if args[2] == "HEAD" else self.fetch_head
        return ""

    def publish(self, **kwargs):
        with patch.object(publisher, "run", self.command):
            publisher.publish(477, "AmbiqAI/ns-cmsis-nn", "origin", self.expected_head, **kwargs)

    def assert_not_ready(self):
        self.assertFalse(any(c[:3] == ("gh", "pr", "ready") for c in self.calls))

    def test_draft_pushes_then_verifies_before_ready(self):
        self.publish()
        push = next(i for i, c in enumerate(self.calls) if c[:2] == ("git", "push"))
        self.assertEqual(self.calls[push], ("git", "push", "--", "origin", NEW + ":refs/heads/topic"))
        self.assertEqual(self.calls[push + 1][:3], ("gh", "pr", "view"))
        self.assertEqual(self.calls[push + 2][:3], ("gh", "pr", "ready"))

    def test_ready_pr_only_pushes(self):
        self.info["isDraft"] = self.after["isDraft"] = False
        self.publish()
        self.assert_not_ready()

    def test_failed_push_or_non_fast_forward_never_promotes(self):
        for command in (("git", "push"), ("git", "merge-base")):
            with self.subTest(command=command):
                self.setUp()
                self.fail = command
                with self.assertRaises(subprocess.CalledProcessError):
                    self.publish()
                self.assert_not_ready()

    def test_changed_or_stale_pr_never_promotes(self):
        for key, value in (("headRefOid", OLD), ("headRefName", "other"),
                           ("state", "CLOSED"), ("isDraft", False),
                           ("isCrossRepository", True)):
            with self.subTest(key=key):
                self.setUp()
                self.after[key] = value
                with self.assertRaises(ValueError):
                    self.publish()
                self.assert_not_ready()

    def test_invalid_initial_state_never_pushes(self):
        for key, value in (("state", "MERGED"), ("isCrossRepository", True)):
            with self.subTest(key=key):
                self.setUp()
                self.info[key] = value
                with self.assertRaises(ValueError):
                    self.publish()
                self.assertFalse(any(c[:2] == ("git", "push") for c in self.calls))

    def test_dirty_checkout_wrong_remote_or_changed_branch_never_pushes(self):
        for key, value in (("dirty", " M file"), ("remote", "git@github.com:other/repo.git"),
                           ("fetch_head", "concurrent")):
            with self.subTest(key=key):
                self.setUp()
                setattr(self, key, value)
                with self.assertRaises(ValueError):
                    self.publish()
                self.assertFalse(any(c[:2] == ("git", "push") for c in self.calls))
                self.assert_not_ready()

    def test_dry_run_checks_ancestry_without_remote_writes(self):
        self.publish(dry_run=True)
        self.assertTrue(any(c[:2] == ("git", "fetch") for c in self.calls))
        self.assertTrue(any(c[:2] == ("git", "merge-base") for c in self.calls))
        self.assertFalse(any(c[:2] == ("git", "push") for c in self.calls))
        self.assert_not_ready()

    def test_only_explicit_reviewed_sha_can_be_published(self):
        for reviewed in (OLD, "a" * 7, "HEAD", ""):
            with self.subTest(reviewed=reviewed):
                self.setUp()
                self.expected_head = reviewed
                with self.assertRaises(ValueError):
                    self.publish()
                self.assertFalse(any(c[:2] == ("git", "push") for c in self.calls))
                self.assert_not_ready()

    def test_multiple_push_urls_are_rejected(self):
        self.remote += "\nhttps://github.com/unrelated/private-repo.git"
        with self.assertRaisesRegex(ValueError, "exactly one push URL"):
            self.publish()
        self.assertIn(("git", "remote", "get-url", "--push", "--all", "origin"), self.calls)
        self.assertFalse(any(c[:2] == ("git", "push") for c in self.calls))

    def test_invalid_branch_or_diverged_preview_stops(self):
        for command in (("git", "check-ref-format"), ("git", "merge-base")):
            with self.subTest(command=command):
                self.setUp()
                self.fail = command
                with self.assertRaises(subprocess.CalledProcessError):
                    self.publish(dry_run=True)
                self.assert_not_ready()



class GitIntegrationTests(unittest.TestCase):
    def test_real_push_urls_in_dry_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bare, work = root / "remote.git", root / "work"
            env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)

            def git(*args, cwd=work):
                return subprocess.check_output(("git", *args), cwd=cwd, env=env,
                                               text=True, stderr=subprocess.DEVNULL).strip()

            git("init", "--bare", str(bare), cwd=root)
            git("init", "-b", "topic", str(work), cwd=root)
            git("config", "user.name", "Publication test")
            git("config", "user.email", "test@example.invalid")
            git("commit", "--allow-empty", "-m", "Base. Refs #485")
            old = git("rev-parse", "HEAD")
            git("remote", "add", "origin", str(bare))
            git("push", "origin", "HEAD:topic")
            git("commit", "--allow-empty", "-m", "Reviewed. Refs #485")
            new = git("rev-parse", "HEAD")
            calls = []

            def command(*args):
                calls.append(args)
                if args[:3] == ("gh", "pr", "view"):
                    return json.dumps(dict(state="OPEN", isDraft=True, headRefName="topic",
                                           headRefOid=old, isCrossRepository=False))
                if args[0] != "git" or args[1] == "push":
                    self.fail(f"Unexpected write command: {args}")
                return git(*args[1:])

            prefixes = ("git@github.com:", "https://github.com/", "ssh://git@github.com/")
            for prefix in prefixes:
                for suffix in ("", ".git"):
                    url = prefix + "ambiqai/NS-CMSIS-NN" + suffix
                    with self.subTest(url=url):
                        git("config", "remote.origin.pushurl", url)
                        calls.clear()
                        with patch.object(publisher, "run", command):
                            publisher.publish(485, "AmbiqAI/ns-cmsis-nn", "origin", new,
                                              dry_run=True)
                        self.assertIn(("git", "remote", "get-url", "--push", "--all", "origin"), calls)
                        self.assertIn(("git", "merge-base", "--is-ancestor", "FETCH_HEAD", new), calls)
                        self.assertEqual(git("rev-parse", "refs/heads/topic", cwd=bare), old)

            invalid_urls = (
                "git@github.com.evil:AmbiqAI/ns-cmsis-nn.git",
                "https://github.com.evil/AmbiqAI/ns-cmsis-nn.git",
                "ssh://git@github.com.evil/AmbiqAI/ns-cmsis-nn.git",
                "ssh://git@github.com@evil.example/AmbiqAI/ns-cmsis-nn.git",
                "ssh://other@github.com/AmbiqAI/ns-cmsis-nn.git",
                *(prefix + "other/ns-cmsis-nn.git" for prefix in prefixes),
                *(prefix + "AmbiqAI/other.git" for prefix in prefixes),
            )
            for url in invalid_urls:
                with self.subTest(url=url):
                    git("config", "remote.origin.pushurl", url)
                    calls.clear()
                    with patch.object(publisher, "run", command):
                        with self.assertRaisesRegex(ValueError, "does not match"):
                            publisher.publish(485, "AmbiqAI/ns-cmsis-nn", "origin", new,
                                              dry_run=True)
                    self.assertFalse(any(c[:2] == ("git", "fetch") for c in calls))

            git("config", "remote.origin.pushurl", "ssh://git@github.com/AmbiqAI/ns-cmsis-nn.git")
            git("config", "--add", "remote.origin.pushurl", "https://github.com/AmbiqAI/ns-cmsis-nn.git")
            calls.clear()
            with patch.object(publisher, "run", command):
                with self.assertRaisesRegex(ValueError, "exactly one push URL"):
                    publisher.publish(485, "AmbiqAI/ns-cmsis-nn", "origin", new, dry_run=True)
            self.assertFalse(any(c[:2] == ("git", "fetch") for c in calls))
            self.assertEqual(git("rev-parse", "refs/heads/topic", cwd=bare), old)

    def test_real_fast_forward_is_remote_before_promotion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bare, work = root / "remote.git", root / "work"
            env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)

            def git(*args, cwd=None):
                return subprocess.check_output(("git", *args), cwd=cwd, env=env,
                                               text=True, stderr=subprocess.DEVNULL).strip()

            git("init", "--bare", str(bare))
            git("init", "-b", "topic", str(work))
            git("config", "user.name", "Publication test", cwd=work)
            git("config", "user.email", "test@example.invalid", cwd=work)
            git("commit", "--allow-empty", "-m", "Base. Refs #459", cwd=work)
            old = git("rev-parse", "HEAD", cwd=work)
            git("remote", "add", "origin", str(bare), cwd=work)
            git("push", "origin", "HEAD:topic", cwd=work)
            git("commit", "--allow-empty", "-m", "Change. Refs #459", cwd=work)
            new = git("rev-parse", "HEAD", cwd=work)
            git("switch", "-c", "unrelated-local-work", cwd=work)
            git("commit", "--allow-empty", "-m", "Unreviewed. Refs #459", cwd=work)
            unreviewed = git("rev-parse", "HEAD", cwd=work)
            promoted = []

            def command(*args):
                if args[:3] == ("git", "remote", "get-url"):
                    return "git@github.com:AmbiqAI/ns-cmsis-nn.git"
                if args[:3] == ("gh", "pr", "view"):
                    return json.dumps(dict(state="OPEN", isDraft=True, headRefName="topic",
                                           headRefOid=git("rev-parse", "refs/heads/topic", cwd=bare),
                                           isCrossRepository=False))
                if args[:3] == ("gh", "pr", "ready"):
                    promoted.append(git("rev-parse", "refs/heads/topic", cwd=bare))
                    return ""
                return git(*args[1:], cwd=work)

            self.assertNotEqual(old, new)
            self.assertNotEqual(new, unreviewed)
            with patch.object(publisher, "run", command):
                with self.assertRaisesRegex(ValueError, "completed review"):
                    publisher.publish(477, "AmbiqAI/ns-cmsis-nn", "origin", new)
            self.assertEqual(promoted, [])
            self.assertEqual(git("rev-parse", "refs/heads/topic", cwd=bare), old)
            git("switch", "topic", cwd=work)
            with patch.object(publisher, "run", command):
                publisher.publish(477, "AmbiqAI/ns-cmsis-nn", "origin", new)
            self.assertEqual(promoted, [new])
            self.assertEqual(git("rev-parse", "refs/heads/topic", cwd=bare), new)


if __name__ == "__main__":
    unittest.main()

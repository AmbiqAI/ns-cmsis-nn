# Publish once after local review

Refs #459.

The PR pipeline handles both `synchronize` (a branch push) and
`ready_for_review` (draft promotion). Each event can start a full run when
its event payload says the PR is ready. Concurrency cancels an older run;
it does not recover credits already spent on that run.

For a draft PR, finish local tests and review fixes, push the final commit,
then mark ready. Draft pushes skip the expensive jobs. Promotion tests the
published commit. Do not promote first and then push, and do not toggle draft
status to manufacture a run. Older handoff instructions claiming promotion
has no trigger are obsolete.

From a clean worktree, set `REVIEWED_SHA` to the full 40-character commit SHA
recorded by the completed review, then use:

```sh
python3 scripts/publish_pr.py 477 --expect-head "$REVIEWED_SHA" --dry-run
python3 scripts/publish_pr.py 477 --expect-head "$REVIEWED_SHA"
```

The helper requires HEAD to equal that explicit reviewed SHA, checks the open PR,
repository, single push URL and clean worktree, fetches the
remote branch, requires a fast-forward, pushes the captured local commit,
verifies GitHub reports that exact head, and only then promotes a draft.
It stops if a push fails or the PR changes. If GitHub has not yet reflected
the pushed head, inspect the PR before retrying; the retry can push an
already-published commit. It never merges, force-pushes, stashes or polls CI.
`--remote` selects a push remote; `--repo` selects the matching GitHub repository.
Cross-repository PRs and remotes with multiple push URLs are deliberately unsupported.
A dry run performs the same pre-push checks, including a local fetch and ancestry
verification, but makes no remote writes.

For an already-ready PR, a push is sufficient: the helper does not promote
again. Each later commit is another run, so address review findings locally
before publishing. This helper is for repositories using the ns-cmsis-nn
workflow; helia-core-tester currently runs CI on draft PRs as well.

Keep both workflow triggers. Removing promotion leaves draft-first work
without an automatic full run; removing synchronization leaves later commits
on a ready PR without automatic validation. The correction is the publishing
order, not disabling either event. External actors can still promote or push
concurrently; coordinate branch ownership and do not run two publishers for
the same PR.

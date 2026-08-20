---
name: intel-monitor-open-xpu-prs
description: "Monitor and triage all open `xpu`-labeled PRs in `sgl-project/sglang`. Use when asked for the XPU PR dashboard, merge readiness, stale review follow-up, unreviewed XPU PRs, or which Intel PRs need attention. Classifies each PR as READY FOR MERGE, WAIT FOR CI, NEED REVIEW AGAIN, NEED REVIEW, or WAIT FOR AUTHOR with concrete approval, head-SHA, XPU CI, and CI-regression evidence."
argument-hint: "[--pr NUMBER]"
---

# Monitor Open XPU PRs

Build a fresh, read-only owner dashboard for open PRs carrying the exact `xpu`
label in `sgl-project/sglang`. Explain every classification with evidence; never
approve, comment, label, rerun, close, or merge a PR from this skill.

## Invocation

```text
/intel-monitor-open-xpu-prs
/intel-monitor-open-xpu-prs --pr 12345
```

- With no arguments, inspect every open PR with the exact `xpu` label.
- `--pr NUMBER` limits the report to that PR, but require it to be open and to
  carry the `xpu` label.
- Always query GitHub live. Do not reuse status from an earlier invocation.
- Default reviewer identity is `gh api user --jq .login`. Display it in the
  report. If authentication fails, stop instead of guessing the user.

## Status Priority

Evaluate statuses in this order. The first matching status wins:

1. `NEED REVIEW AGAIN`
2. `WAIT FOR AUTHOR`
3. `READY FOR MERGE`
4. `WAIT FOR CI`
5. `NEED REVIEW`

This order prevents a core-maintainer approval from hiding newer author changes
that respond to the current user's requested modifications.

### `NEED REVIEW AGAIN`

Use when all are true:

- The current user previously left actionable feedback: a
  `CHANGES_REQUESTED` review, a non-empty review body asking for a change, or an
  inline review comment asking for a change.
- The latest such feedback has not been superseded by a later `APPROVED` review
  from the current user.
- The PR head changed after that feedback. Prefer REST review/comment
  `commit_id != headRefOid` plus a later PR commit timestamp. Do not infer an
  update from `updatedAt` alone because CI and comments also update it.

Reason must name the feedback date/link or review ID, its reviewed commit, the
current head SHA, and the later commit(s). If the change is only a merge from
`main`, say so; it still needs a freshness check but is weaker evidence that the
requested fix was addressed.

### `WAIT FOR AUTHOR`

Use when either condition holds:

- The current user left actionable feedback that is not followed by a user
  approval, and no author/head commit was added after that feedback.
- Review and CI would otherwise permit merge, but the PR is draft, closed to
  merging by a conflict, or GitHub reports a concrete non-CI mergeability
  blocker.

Give the exact outstanding feedback or mergeability reason. Do not call this
`NEED REVIEW` because no new reviewer action is available yet.

### `READY FOR MERGE`

Require every condition:

1. The PR is open, non-draft, and has no known merge conflict.
2. It has a valid approval from either:
   - the current user; or
   - an SGLang core maintainer.
3. XPU CI genuinely ran on the current head and passed.
4. No current-head CI is pending, and every substantive non-XPU failure is
   classified `PRE-EXISTING` or `FLAKE / INFRA`; there is no `PR-CAUSED` or
   `UNKNOWN` regression.
5. There is no newer author response requiring the current user's re-review.

### `WAIT FOR CI`

Use only when a valid current-user or core-maintainer approval exists, but any
of these is true:

- a current-head check is queued, waiting, pending, or in progress;
- XPU CI is absent, stale, skipped, cancelled, neutral, or failed;
- a non-XPU failure is `PR-CAUSED` or `UNKNOWN`;
- CI only has an aggregate/gate result and the substantive jobs have not run.

State whether this is merely unfinished/missing CI or a suspected/confirmed CI
regression. A `PR-CAUSED` regression is a merge blocker even though the dashboard
bucket remains `WAIT FOR CI`.

### `NEED REVIEW`

Use when none of the earlier statuses applies and there is no valid approval
from the current user or a core maintainer. Distinguish:

- `never reviewed by me`;
- `only non-maintainer reviews exist`;
- `my prior review was non-actionable`; or
- `approval was dismissed`.

## Collect PRs

Preflight:

```bash
gh auth status
gh api user --jq .login
gh pr list --repo sgl-project/sglang --state open --label xpu --limit 200 \
  --json number,title,url,isDraft,author,headRefOid,updatedAt,mergeStateStatus
```

For each PR fetch, at minimum:

```bash
gh pr view PR --repo sgl-project/sglang \
  --json number,title,url,isDraft,author,headRefOid,mergeable,mergeStateStatus,reviewDecision,reviews,commits,statusCheckRollup
gh api --paginate repos/sgl-project/sglang/pulls/PR/reviews
gh api --paginate repos/sgl-project/sglang/pulls/PR/comments
gh api --paginate repos/sgl-project/sglang/issues/PR/comments
```

Use issue comments only as context. They are not approvals and are actionable
feedback only when the current user clearly asks the author to change code.

## Resolve Valid Approvals

Reduce reviews per reviewer chronologically. A later `CHANGES_REQUESTED` or
dismissed review invalidates that reviewer's earlier approval. Record approval
date, reviewed commit ID, and URL/ID.

Treat a reviewer as an SGLang core maintainer when the repository permission API
returns `admin`, `maintain`, or `write`:

```bash
gh api repos/sgl-project/sglang/collaborators/LOGIN/permission --jq .permission
```

Cache permission results during one invocation. Exclude bots and the PR author.
If the permission endpoint is unavailable, use `OWNER` or `MEMBER`
`author_association` as fallback, but mark the evidence as fallback. Do not treat
`COLLABORATOR` or `CONTRIBUTOR` alone as proof of core-maintainer status.

An approval may satisfy the dashboard even when it predates the current head if
GitHub has not dismissed it, but explicitly report that it is older than the
head. The `NEED REVIEW AGAIN` rule still takes priority for the current user's
own requested changes followed by new commits.

## Verify XPU CI Actually Ran

Only use runs/checks for the exact current `headRefOid`. Inspect jobs, not just
the workflow conclusion. A green `finish` job with skipped stage jobs is not an
XPU pass.

For `PR Test (XPU)` / `pr-test-xpu.yml` require:

- at least one substantive XPU test job, such as
  `stage-a-test-1-gpu-xpu` or `stage-b-test-1-gpu-xpu`, actually completed with
  `success`; and
- every substantive XPU job that ran completed with `success`.

If all substantive jobs were skipped, inspect `check-changes` logs and report
the exact path-filter or gate output. For example, `main_package=false` means
`XPU CI: SKIPPED`, not passed. A manual run is acceptable only when its tested
SHA exactly equals the current PR head.

## Attribute CI Failures

Inspect root jobs, not aggregate `finish`, wait, health, or gate jobs. For every
substantive failed job, capture job name, test ID, exception class, one signature
line, run URL, and hardware/workflow.

Assign exactly one:

- `PR-CAUSED`: new signature with clear overlap with the diff or a test added by
  the PR.
- `PRE-EXISTING`: same signature in a recent `main` or unrelated-PR run of the
  same workflow/job.
- `FLAKE / INFRA`: runner/container loss, timeout, unavailable fixed port,
  download/service/device initialization failure, or success on a same-SHA
  rerun.
- `UNKNOWN`: evidence is insufficient.

Compare against recent `main` runs when the result affects readiness. Shared SRT
or multimodal changes overlap CUDA/AMD/NPU tests, so lack of an Intel filename is
not enough to dismiss a red job. Aggregate failures inherit the root job's
classification and are not counted twice.

Define CI completion as all applicable current-head checks terminal. Skipped
optional non-XPU jobs are terminal and harmless. Missing/skipped XPU jobs are
not harmless because a genuine XPU pass is required for `READY FOR MERGE`.

## Output

Write the report in concise Chinese. Start with the scan time in UTC, current
reviewer login, and counts per status. Then show sections in this order:

1. `READY FOR MERGE`
2. `WAIT FOR CI`
3. `NEED REVIEW AGAIN`
4. `NEED REVIEW`
5. `WAIT FOR AUTHOR`

Use one row per PR:

```markdown
| PR | Title | Approval | XPU CI | Other CI | Reason |
|---|---|---|---|---|---|
| [#123](url) | ... | @maintainer APPROVED, date | PASS, run link | PASS | Head SHA..., no pending checks |
```

Every reason must be concrete. Include as applicable:

- approving reviewer, permission evidence, approval date, and whether approval
  predates head;
- current head short SHA;
- latest current-user feedback and commits after it;
- XPU run/job link, or exact reason it was skipped/missing;
- pending job names;
- each failure's attribution and signature;
- draft/conflict state.

Never say only “CI red”, “needs review”, or “approved”. If a section is empty,
write `None`. End with a short action summary listing PR numbers only, grouped by
the next human action. Do not generate PR comments unless explicitly requested.

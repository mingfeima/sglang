---
name: intel-clean-ci-ghost-runs
description: "Find and optionally delete ghost GitHub Actions runs in SGLang Intel CI. Use when asked to inspect, clean, or remove stale/queued/幽灵 runs for Xeon (`pr-test-xeon.yml`) or XPU (`pr-test-xpu.yml`); report each run's evidence, associated PR, PR state, and title, then require explicit user confirmation before deletion."
argument-hint: "xeon|xpu"
---

# Clean Intel CI Ghost Runs

Conservatively identify orphaned nonterminal workflow runs in `sgl-project/sglang`, report the evidence first, and delete only the runs explicitly approved by the user.

## Invocation

```text
/intel-clean-ci-ghost-runs xeon
/intel-clean-ci-ghost-runs xpu
```

Accept exactly one platform. If it is missing or not `xeon`/`xpu`, ask the user to choose one.

| Input | Workflow file |
|---|---|
| `xeon` | `pr-test-xeon.yml` |
| `xpu` | `pr-test-xpu.yml` |

Always target `sgl-project/sglang`. Never infer a workflow from the current checkout or remote.

## Safety Rules

- Discovery is read-only. Always finish and show the report before asking about deletion.
- A long-running job is not a ghost merely because it is old.
- Never cancel or delete a run classified as `ACTIVE` or `UNCERTAIN`.
- Never delete from an earlier invocation or a pasted list. Revalidate in the same invocation immediately before deletion.
- Deletion requires explicit user approval through the question tool. Silence, ambiguous wording, or invocation of this skill is not approval.
- Default to keeping all runs. Do not use a terminal prompt for confirmation.

## Step 1: Preflight

Verify authentication and resolve the workflow:

```bash
gh auth status
gh api "repos/sgl-project/sglang/actions/workflows/$WORKFLOW" \
  --jq '{id,name,state,path}'
```

Stop without mutation if authentication fails, the workflow is absent, or it is not active.

Record the current UTC time. Use a default stale threshold of 24 hours. The threshold only helps classify runs with no jobs; it must not override evidence of an active job.

## Step 2: Enumerate Nonterminal Runs

Query every nonterminal status separately because the GitHub API status filter accepts one value at a time:

```bash
for STATUS in queued in_progress waiting requested pending; do
  gh api --paginate \
    "repos/sgl-project/sglang/actions/workflows/$WORKFLOW/runs?status=$STATUS&per_page=100"
done
```

Deduplicate by run ID. Record at least:

- run ID and URL
- status, conclusion, event, attempt
- creation and update timestamps
- head branch and SHA
- check suite ID

Do not treat the workflow's `total_count` as the count returned by a filtered query; GitHub may report a repository-wide capped value.

## Step 3: Inspect Jobs and Check Suite

For each run older than 24 hours, fetch all attempts' jobs and the backing check suite:

```bash
gh api "repos/sgl-project/sglang/actions/runs/$RUN_ID/jobs?filter=all&per_page=100"
gh api "repos/sgl-project/sglang/check-suites/$CHECK_SUITE_ID"
```

Inspect every job's `status`, `conclusion`, `started_at`, `completed_at`, and `runner_name`. `filter=all` can return duplicate job names from reruns; evaluate every returned job by ID.

Classify conservatively:

### `CONFIRMED_GHOST`

Use this classification only when the parent run is nonterminal and either condition holds:

1. **Terminal children:** at least one job exists, every job is `completed`, and none is `queued`, `waiting`, `pending`, `requested`, or `in_progress`.
2. **No children:** no jobs exist, both run and check suite have been untouched for over 24 hours, and at least one orphan signal exists:
   - the associated PR is closed or merged;
   - a newer run for the same workflow and PR/head branch reached a terminal state; or
   - the run's head SHA is no longer the PR head SHA.

Possible reason labels:

- `PARENT_STUCK_AFTER_JOBS_COMPLETED`
- `NO_JOBS_AFTER_PR_CLOSED`
- `NO_JOBS_SUPERSEDED_BY_NEWER_RUN`
- `NO_JOBS_ON_STALE_PR_HEAD`

Include all applicable evidence in the report.

### `ACTIVE`

Any job is nonterminal, or the run has been updated within 24 hours. Do not include it in the deletion choices.

### `UNCERTAIN`

The run is old and nonterminal but does not satisfy a confirmed-ghost condition. Report it separately and do not offer deletion.

## Step 4: Resolve the Associated PR

Resolve PR metadata in this order:

1. Run payload `pull_requests`.
2. `GET /repos/sgl-project/sglang/commits/{head_sha}/pulls`.
3. Search all PR states by exact head branch with `gh pr list --state all --head`.

For fork PRs, use `owner:branch` when the payload provides the head repository owner. Do not choose by branch alone when multiple PRs match. Prefer the PR whose head SHA equals the run SHA; otherwise label the association as `branch match` or `commit history match`.

Fetch and record:

```bash
gh pr view "$PR_NUMBER" --repo sgl-project/sglang \
  --json number,title,state,createdAt,mergedAt,closedAt,headRefOid,url
```

Display `createdAt` in UTC using the GitHub timestamp returned by the API. Do not
substitute the run creation time or the first commit time when PR metadata is
unresolved.

If no PR can be resolved, show `PR: unresolved`; this is not by itself proof of a ghost.

## Step 5: Report Before Mutation

Show confirmed ghosts in a table sorted oldest first:

```markdown
| Run | Age | Parent | Jobs | Reason | PR | PR created | PR state | PR title |
|---|---:|---|---|---|---|---|---|---|
| [123](run-url) | 31d | queued | 3/3 completed | PARENT_STUCK_AFTER_JOBS_COMPLETED | [#456](pr-url) | 2026-07-01 12:34 UTC | MERGED | Fix ... |
```

Then show `UNCERTAIN` runs separately with the missing evidence. Summarize current `ACTIVE` runs by count only unless the user asks for details.

If there are no confirmed ghosts, state that clearly and stop. Do not ask a deletion question.

## Step 6: Ask What to Delete

After showing the complete report, use the question tool with these choices:

- `Keep all` — recommended default
- `Delete all confirmed ghosts`
- `Delete selected run IDs` — allow free-form IDs

For selected IDs, accept only IDs present in the just-produced `CONFIRMED_GHOST` table. Reject unknown, active, uncertain, malformed, or cross-workflow IDs and ask again. Do not interpret “clean up” or similar general wording as selecting all.

## Step 7: Revalidate and Delete

For every approved ID, immediately refetch the run, jobs with `filter=all`, check suite, and PR. Confirm that:

- it still belongs to the selected workflow ID;
- it is still nonterminal;
- it still satisfies `CONFIRMED_GHOST`;
- no job is active;
- the user approved this exact run ID.

Skip and report any run that changed or returns `404`. Delete validated runs one at a time:

```bash
gh api --method DELETE "repos/sgl-project/sglang/actions/runs/$RUN_ID"
```

Do not substitute `gh run cancel`: cancellation does not remove a ghost run from the workflow history.

After each deletion, verify that fetching the run returns `404`. Report deleted, skipped, and failed IDs separately, with the reason for every skip or failure.

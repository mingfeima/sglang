# CI Failure Attribution (Intel XPU / CPU)

SGLang CI is noisy. **CUDA is relatively healthier; Xeon/XPU often fail for
reasons unrelated to the PR under review.** Never treat a red Intel check as
automatic proof the PR is wrong — and never ignore a red check that shares a
code path with the diff.

Goal: label each failed job as one of:

| Label | Meaning | Review action |
|---|---|---|
| **PR-CAUSED** | Failure signature is new and correlated with the diff | Request changes / block |
| **PRE-EXISTING** | Same failure already on `main` or many unrelated PRs | Comment "known/unrelated"; do not block on it |
| **FLAKE / INFRA** | Timeout, runner, Docker, OOM, device lost, checkout | Suggest `/rerun-failed-ci`; do not block unless persistent |
| **UNKNOWN** | Cannot decide with available logs | Ask for one rerun + note residual risk; optionally bisect |

## Priority reminder

**Breaking other platforms' CI (esp. CUDA) outranks Intel completeness.**
When attributing, always classify CUDA / non-Intel jobs first. A PR that only
helps XPU/CPU but newly fails CUDA is `PR-CAUSED` → block, even if Intel jobs
are green or "only" flaky.

## Decision tree (fast path)

```text
1. Extract failure signature (test id + error class + 1-line message)
        │
        ▼
2. Same signature on recent main push / other unrelated PRs for same workflow?
        │ yes ──► PRE-EXISTING (or endemic infra) — cite 1–2 run links
        │ no
        ▼
3. Failed job's test module import graph overlap the PR files?
        │ no strong overlap + infra-looking error ──► FLAKE / INFRA
        │ no overlap + assertion in untouched area ──► likely PRE-EXISTING / FLAKE
        │ yes overlap
        ▼
4. Rerun once (or compare prior run on same PR SHA / previous SHA)
        │ pass on rerun, same SHA ──► FLAKE
        │ still fail, signature unique to this PR ──► PR-CAUSED
        │ still fail, also on main ──► PRE-EXISTING
```

Default bias for **Intel** jobs: prefer PRE-EXISTING / FLAKE until code overlap
is clear. Default bias for a failure **inside a test the PR added/changed**:
prefer PR-CAUSED.

## Commands

Replace `WORKFLOW` with `pr-test-xpu.yml` or `pr-test-xeon.yml`.
Repo is always `sgl-project/sglang` for upstream reviews.

### 1. PR checks → failing jobs

```bash
gh pr checks <N> --repo sgl-project/sglang
gh pr view <N> --repo sgl-project/sglang --json statusCheckRollup \
  --jq '.statusCheckRollup[] | select(.conclusion=="FAILURE" or .state=="FAILURE") | {name,conclusion,detailsUrl}'
```

### 2. Failure signature from a run

```bash
# List jobs in the run (from checks URL or:)
gh run list --repo sgl-project/sglang --workflow=WORKFLOW --branch <pr-head-branch> --limit 5 \
  --json databaseId,conclusion,headSha,createdAt,url

gh run view <RUN_ID> --repo sgl-project/sglang --json jobs \
  --jq '.jobs[] | {name,conclusion,databaseId}'

# Pull log slice (keep it tight — full XPU logs are huge)
gh run view <RUN_ID> --repo sgl-project/sglang --job <JOB_ID> --log 2>&1 \
  | rg -n -C 3 'FAILED|ERROR|AssertionError|Timeout|OOM|Device|Level Zero|ze_|AMX|Traceback|===' \
  | tail -n 80
```

Record:
- workflow + job name (`stage-b-test-1-gpu-xpu`, `base-b-test-cpu`, …)
- test file / method (`test_xpu_basic.py::…`)
- error class (`AssertionError`, `RuntimeError`, `Timeout`, runner setup)
- one distinctive message line

### 3. Is it already red on main / other PRs?

`pr-test-xpu.yml` / `pr-test-xeon.yml` run on **`push` to `main`** and on PRs
(no CUDA-style `schedule` event). Use main pushes + recent PR runs as the baseline:

```bash
# Recent main-branch runs of the Intel workflow
gh run list --repo sgl-project/sglang --workflow=WORKFLOW --branch main --limit 15 \
  --json databaseId,conclusion,headSha,createdAt,event,url

# Recent PR runs (spot endemic failures across unrelated titles)
gh run list --repo sgl-project/sglang --workflow=WORKFLOW --event pull_request --limit 20 \
  --json databaseId,conclusion,displayTitle,headBranch,createdAt,url
```

Optional fleet/pass-rate view (XPU):

```bash
python scripts/ci/utils/xpu_job_monitor.py \
  --job "stage-b-test-1-gpu-xpu" --workflow "pr-test-xpu.yml" --hours 48 --summary
```

If ≥ ~half of recent unrelated PRs fail the **same job with the same signature**,
treat as PRE-EXISTING / endemic unless the PR clearly touches that path.

### 4. Code overlap (the decisive signal)

```bash
# Files changed in the PR
gh pr diff <N> --repo sgl-project/sglang --name-only

# What the failing test imports / launches (local tree at PR head if checked out)
rg -n 'register_xpu_ci|register_cpu_ci|popen_launch_server|ServerArgs|attention|quant' \
  test/registered/xpu/<failing_test>.py test/registered/cpu/<failing_test>.py
```

Overlap heuristics:
- PR touches only CUDA kernels / NVIDIA docs, fail is XPU stage-a import → weak
- PR edits `server_args.py` / `multi_platform.py` / attn registry, fail in XPU attn test → strong
- PR adds `register_xpu_ci` test that fails → almost always PR-CAUSED
- Fail during Docker pull / `chown` / runner setup / "No space" → INFRA

### 5. Rerun vs investigate

| Situation | Do |
|---|---|
| First infra-looking fail | Comment `/rerun-failed-ci` (or rerun the Intel workflow job) |
| Same signature after rerun + strong overlap | PR-CAUSED — request fix |
| Same signature after rerun + also on main | PRE-EXISTING — link main run; optional follow-up issue |
| Intermittent pass/fail same SHA | FLAKE — do not block merge on Intel alone |
| Consistent on main, need root cause | Hand off to `/sglang-bisect-ci-regression` |

## Infra / flake signatures (usually NOT PR-caused)

Treat as FLAKE/INFRA unless the PR edits Docker/CI scripts:

- runner lost / job cancelled by concurrency / queue timeout
- Docker Hub auth, image pull, `intel/sglang-dev` missing tag
- workspace `chown` / checkout clean failures on XPU hosts
- `Level Zero` / device reset / `ZE_*` init errors with no Python stack in PR code
- HF download / network blips
- OOM on a test the PR did not enlarge (batch, model, graph capture)

## When red CI should still block

Block (or request changes) even on a flaky fleet if:

1. Label is **PR-CAUSED**, or
2. PR claims Intel support / adds Intel tests that fail, or
3. Failure is in a brand-new job/suite the PR introduced, or
4. Diff clearly breaks import of shared SRT on CPU/XPU (collectible without GPU)

Otherwise: approve/comment with an explicit **CI attribution** line so the author
and other reviewers do not thrash on endemic reds.

## Output snippet (paste into the Chinese review; English version for the author)

中文报告里可写：

```text
CI 归因:
- pr-test-xpu / stage-b-test-1-gpu-xpu: PRE-EXISTING
  签名: test_intel_xpu_backend.py::test_mha AssertionError: ...
  证据: main run <url> 与无关 PR <url> 同样失败
- pr-test-xeon / base-b-test-cpu: FLAKE/INFRA
  签名: runner setup / docker pull
  处理: 重跑一次；不作为合入阻断
- 无 PR-CAUSED
```

若需回复作者，对应英文草稿示例：

```text
Intel CI attribution (not blocking):
- pr-test-xpu / stage-b-test-1-gpu-xpu looks PRE-EXISTING — same signature on
  main <url> and unrelated PR <url>.
- pr-test-xeon failure looks infra/flake (docker/runner). A single rerun should
  be enough; not treating this as a PR regression.
```

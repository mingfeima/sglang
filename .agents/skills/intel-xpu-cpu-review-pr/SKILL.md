---
name: intel-xpu-cpu-review-pr
description: "Review SGLang or sgl-kernel-xpu PRs for Intel XPU/CPU ownership: CI and merge readiness, automatic Intel CI labels (sglang only), enabling parity, kernel performance and synchronization, reuse of existing utilities/helpers, benchmark and UT consistency. Chinese analysis; English comment drafts. Run with /intel-xpu-cpu-review-pr <PR number> [--repo sgl-kernel-xpu]."
---

# Intel XPU / CPU PR Review

Review a PR from the perspective of the **Intel XPU** and **CPU (AMX Xeon)** platform owners.

## Three pillars (in order)

1. **不能挂掉其他 CI** — especially CUDA. Intel wins that regress other devices are a **BLOCK**.
2. **XPU/CPU enabling 要和别的 device 对齐** — same extension points as CUDA/HIP (`MultiPlatformOp`, registries, allowlists); no Intel-only side paths without a documented reason. See §1b / [enabling-parity.md](references/enabling-parity.md).
3. **关注性能影响与 kernel 落点**
   - **CPU kernels**: in-tree `sgl-kernel/csrc/cpu/`
   - **XPU kernels**: out-of-tree [sgl-kernel-xpu](https://github.com/sgl-project/sgl-kernel-xpu) (pinned from `python/pyproject_xpu.toml`)
   - Python-only “enable” without the matching kernel (or with silent `forward_native`) is not a real perf win — call it out. See §1c.
   - Perf opt PRs need **benchmark numbers**; code changes need **UT coverage** (update tests if current ones miss the diff).

Path map: [platform-map.md](references/platform-map.md).
CI attribution: [ci-failure-attribution.md](references/ci-failure-attribution.md).
Enabling parity: [enabling-parity.md](references/enabling-parity.md).
Perf / kernels: [performance-kernels.md](references/performance-kernels.md).
**sgl-kernel-xpu-only PRs**: [sgl-kernel-xpu-review.md](references/sgl-kernel-xpu-review.md).


## Usage

```
/intel-xpu-cpu-review-pr <PR number>
/intel-xpu-cpu-review-pr <PR number> --repo sgl-kernel-xpu
/intel-xpu-cpu-review-pr https://github.com/sgl-project/sgl-kernel-xpu/pull/<N>
```

Optional: `--focus xpu|cpu|both` (default `both`) for **sglang** tree PRs.

### Which playbook?

| PR location | Playbook |
|---|---|
| `sgl-project/sglang` (default) | Full Intel checklist below (three pillars) |
| `sgl-project/sgl-kernel-xpu` | **Lighter** — perf (**benchmark required** for opt PRs), simplify, FA/FlashInfer algorithm parity, **UT must cover the diff**. See [sgl-kernel-xpu-review.md](references/sgl-kernel-xpu-review.md). Skip §0–§14 unless the op ABI breaks sglang. |

Detect automatically from the PR URL / `--repo`. If unclear, ask once.

When the user asks whether a red check is caused by the PR (sglang tree), run §14
even if the code review is otherwise light.

### Intel CI labels (sglang only; add automatically)

Xeon and XPU PR CI are extra opt-in workflows; `run-ci` alone does not enable
them. After classifying the affected platforms, inspect the current PR labels
and **add any missing matching label without asking for confirmation**:

| Affected platform | Label to add |
|---|---|
| CPU / Xeon | `run-ci-xeon-extra` |
| XPU | `run-ci-xpu-extra` |
| Both, including shared changes affecting both | Both labels above |

Run only the command(s) for the affected platform(s), only when the label is missing:

```bash
gh pr edit <N> --repo sgl-project/sglang --add-label run-ci-xeon-extra
gh pr edit <N> --repo sgl-project/sglang --add-label run-ci-xpu-extra
```

Keep existing labels. If permissions prevent adding a label, report the exact
missing label and the failure; do not claim it was added. Do not apply these
sglang-specific labels to `sgl-kernel-xpu` PRs. After adding a label, verify that
a new eligible workflow run actually starts; adding a label does not prove tests
ran, and rerunning an old label-skipped event retains its frozen event payload.

### Merge-readiness gate (both repos)

Before any `APPROVE` or merge-ready verdict, check the latest PR head SHA, required
PR CI workflows, and the **actual test jobs and test files** relevant to the diff.
Workflow startup, lint success, a green finish job with skipped tests, or manual
accuracy results do not prove those tests ran.

- Check the current title and newest required title check (if any). A failed
  title check blocks merging but is not a code/device regression; title edits
  do not change the head SHA, so ignore older failures after a passing rerun.
- Missing `run-ci` (or another required CI opt-in label), a failing PR/rebase/
  maintenance gate, path-filtered or skipped target job, cancellation, or setup/
  runner failure before tests → **`NOT VERIFIED — DO NOT MERGE`**. Name the
  missed suite/test, why it did not run, and link the gate/job evidence. Tell
  the reviewer explicitly **不能 merge** and set the overall verdict to
  **BLOCKED**, even when the failure is not PR-caused.
- Exception: Extra / AMD Extra opt-out caused solely by missing `run-ci-extra`,
  or Xeon / XPU opt-out caused solely by missing `run-ci-xeon-extra` /
  `run-ci-xpu-extra`, is **informational only, not a merge blocker**. Add the
  matching Intel labels as instructed above. Mention the missing label and
  that the optional suites did not run; do not
  classify this alone as `PR-CAUSED`, `NOT-VERIFIED-BLOCK`, or set the overall
  verdict to **BLOCKED**. This exception does not waive missing `run-ci`,
  other required PR coverage, or actual test failures.
- State the unblock action using the decision table below for sglang, or the
  target repo's own process otherwise. Require the relevant tests to execute
  on the latest head before changing this verdict.
- Apply this gate to **required PR CI coverage of the changed paths**, not
  optional nightly suites or intentionally skipped tests. If a changed test
  runs only in nightly, report **NIGHTLY NOT RUN** with its suite and any local
  evidence; do **not** block the PR solely because nightly did not run on the
  PR head. Do not claim the test passed in CI. A nightly test that *did* run
  and fails due to the PR is still `PR-CAUSED` under §14. `PRE-EXISTING`
  failures in tests that *did run* are attributed under §14, not treated as
  missing CI coverage.

### CI blocker and next action (sglang)

In every review, report **CI blocker / next action**: the current head, the
first failing gate or real test (link its job and quote the decisive log line),
which required coverage is missing, and who should act next. A green gate job
can still output "tests skipped"; inspect its logs and the downstream jobs.
Use the latest attempt on the current head, not an older failure or the aggregate
finish job. If nothing blocks CI, say so.

| Evidence | What to recommend |
|---|---|
| `Rebase Required` / head does not contain `MIN_BASE_SHA`, or GitHub reports a merge conflict with `main` | Ask the author to rebase onto current `main`, resolve conflicts if any, and push. Verify CI on the new SHA and check whether it needs fresh commit approval. Being behind `main` alone is not evidence of a rebase gate. |
| Gate says `run-ci` is missing | Ask an authorized maintainer to comment `/tag-and-rerun-ci` to add the label, approve the current head, and trigger CI. For a draft, mark it ready first. |
| `run-ci` is present, but the gate says the current SHA lacks commit approval | Ask an authorized maintainer to comment `/approve-ci`. It approves the current head and retries approval-gated runs that skipped tests; it does not add labels or broadly retry unrelated failures. Approval of an older SHA does not count. For a draft, mark it ready first. |
| Current SHA is approved, required jobs failed or were cancelled by a transient runner, network, or API rate-limit issue, and no retry is active | Recommend `/rerun-failed-ci` after recovery. If the same infrastructure failure persists, send the failing job link and error to CI maintainers instead of repeatedly commenting. |
| Maintenance pause, persistent setup/credential error, or runner/security policy blocks the job | Wait for maintenance to lift or ask CI maintainers to resolve the blocker; do not suggest bypassing protections. Then recheck the gate and rerun the affected coverage. |
| PR-caused test failure, invalid suite, or missing test due to a path filter | Fix the code, registration, or workflow first; a retry alone does not provide coverage. |
| Relevant jobs already queued or running | Point to the active run and wait for the target tests; do not recommend a duplicate command. |

After a slash command, check its handler log for approval or rejection and
verify the resulting attempt and target jobs. A rerun can keep the same run ID
and increment `run_attempt`; no new run ID does not imply the command failed.
If eligibility depends on a frozen event label or draft state, require a fresh
eligible event instead of assuming a retry will update that payload. Recommend
actions only: do not rebase, push, or post CI commands without user authorization.

## Language (required)

Bilingual by default — do not ask; just follow:

| Audience | Language | What |
|---|---|---|
| **Reviewer (user)** | **中文** | PR 在做什么、结论、风险 — **尽量短**；引用代码时 **带行号** |
| **GitHub PR comment** | **English** | 可贴草稿 — **尽量短**；指到文件+行号 |

Rules:
- **Brevity first**: 中文主报告通常半屏以内；英文每条 comment 2–4 句。少写背景，多写结论。
- **中文报告必须带行号**: 指出问题/关键逻辑时用 `path:start-end` 或引用块格式
  （\`\`\`start:end:path\`\`\`）。没有具体行号就不要空泛说「主循环里」。
- Keep stable machine labels in English: `PR-CAUSED` / `PRE-EXISTING` / `FLAKE` /
  `APPROVE` / …
- Emit ready-to-paste **English** blocks when a finding should go on the PR; do
  **not** auto-post unless the user asks.
- `英文输出` / `English only` → whole review in English (still short + line refs).
- `只分析不写 comment` → skip English paste blocks.

## Existing implementation and convention check (both repos)

Before suggesting a new abstraction or accepting new utils/helpers, check whether
the behavior already exists in the **target repository at the PR head**, not only
in the current workspace or in the changed files:

- Search by behavior as well as symbol name. Start with the owning module,
  nearby implementations and shared utilities; include test fixtures, quant/dequant,
  device/stream/alignment helpers, benchmark harnesses and numerical assertions.
- Read the existing implementation and a caller before recommending reuse. Compare
  shapes, dtype/layout, device support, numerical semantics and hot-path costs.
  Cite the existing helper's path and line; name any adaptation it needs. Similar
  names or formulas alone do not prove interchangeability.
- Prefer an existing suitable helper; otherwise consider sharing genuinely repeated
  logic. Keep justified kernel specializations and small local helpers when extraction
  adds more complexity than it removes. Style-only duplication is a non-blocking nit;
  incorrect semantics, extra synchronization or missing coverage is a separate finding.
- **Benchmarks:** compare the nearest existing bench's timing helper, warmup/repetitions,
  synchronization, graph/eager mode, allocation/quantization inside the timed region,
  statistics, units and output helpers. Explain meaningful differences; do not claim
  timings are directly comparable or invalid merely because formatting differs.
- **UTs:** compare the nearest test's golden reference, accumulation/reference dtype,
  elementwise vs aggregate error checks, tolerances and fixtures. Reuse suitable
  helpers, but do not blindly copy tolerances or replace a justified error metric.

Report actionable reuse/convention findings with the new-code location and the
existing implementation's location. Do not list every harmless repetition.

## When this review applies

Run the full checklist when the PR touches any of:

- `python/sglang/srt/**` (esp. attention, quantization, MoE, speculative, distributed,
  disaggregation, compilation, model_executor, platforms, server_args, utils/common)
- `python/sglang/srt/layers/utils/multi_platform.py` or new `MultiPlatformOp` subclasses
- `sgl-kernel/csrc/cpu/**`, `sgl-kernel/pyproject_cpu.toml`
- `python/pyproject_cpu.toml`, `python/pyproject_xpu.toml`
- `docker/xeon.Dockerfile`, `docker/xpu.Dockerfile`
- `.github/workflows/pr-test-xeon.yml`, `pr-test-xpu.yml`, `nightly-test-intel.yml`,
  `release-docker-xeon.yml`, `release-docker-intel-xpu-nightly.yml`
- `test/registered/cpu/**`, `test/registered/xpu/**`, or new `register_cpu_ci` /
  `register_xpu_ci` markers
- Docs under `docs/docs/hardware-platforms/{cpu_server,xpu}.mdx` or cookbook cells
  claiming CPU/XPU support

Also review **CUDA-centric "portable" refactors** that touch shared SRT code even if the
PR title says NVIDIA-only — Intel breakage is usually accidental.

Skip a deep Intel review only when the diff is clearly isolated (e.g. pure CUDA kernel
under `sgl-kernel/csrc/gpu/**` with no Python dispatch changes, or docs-only unrelated
to hardware platforms). Still skim for top-level CUDA-only imports in shared modules.

## Workflow

1. Read PR metadata, diff, existing review comments and checks in the target repo.
  Record the latest head and classify CPU / XPU / shared-SRT / deps-CI / docs impact.
2. Check the current title against that repo's actual title gate. For sglang,
  `.github/workflows/pr-title.yml` on main is authoritative; do not impose its
  syntax on sgl-kernel-xpu. Apply automatic labels only to sglang.
3. Perform the existing-implementation/convention check above. For sgl-kernel-xpu,
  use its light reference instead of the sglang checklist below; the common
  merge-readiness and output rules still apply.
4. For sglang, review other-device CI first, then enabling parity, kernel performance
  and Intel-specific contracts below. Load only relevant reference sections.
5. Attribute failures (§14), verify actual required coverage using the common gate,
  and recheck the head before the final verdict. Distinguish local/manual evidence
  from latest-head CI; do not infer execution from registration or aggregate success.

## Checklist

### 0. Do not break other CI (BLOCK if violated)
Highest bar — ahead of Intel feature completeness.

- Shared Python/C++ changes must keep **CUDA** behavior and imports intact
  (lazy imports, device guards, no accidental default-backend flips).
- Enabling PRs must not change global quant/attn/graph defaults that CUDA CI
  relies on; scope Intel via `is_xpu()` / `is_cpu()` / `device == "xpu"|"cpu"`.
- Touching `sgl-kernel` common code, `pyproject.toml` (non-cpu/xpu), or shared
  Docker/CI scripts → explicitly check CUDA / kernel CI, not only Xeon/XPU.
- Diff that widens a type/shape/API used by CUDA kernels "for Intel" without a
  CUDA test update is suspicious — require CUDA CI green or a clear revert plan.
- **Verdict required in every review:** `other-CI: PASS | PRE-EXISTING-red | PR-CAUSED-BLOCK | NOT-VERIFIED-BLOCK`.
  `PR-CAUSED-BLOCK` is a code regression; `NOT-VERIFIED-BLOCK` means required
  non-Intel tests never ran. Neither is merge-ready.

### 1b. Model / operator enabling — CUDA / HIP 接入一致性
For new model/op/quant/attention support, read
[enabling-parity.md](references/enabling-parity.md). Compare the CUDA/HIP owner,
registry, dispatch, capability gates, API, tests and docs side by side. Reuse the
same extension points; require a concrete reason for a device-specific path.

### 1c. Performance & kernel placement (pillar 3)
Read [performance-kernels.md](references/performance-kernels.md) for kernel
placement and evidence requirements. Perf claims require before/after numbers
(hardware, dtype, shapes, baseline and method); missing evidence blocks approval.
No-perf-claim correctness/refactor changes are exempt from benchmarks, not UTs.
Existing tests may suffice if they exercise the diff; otherwise require tests in
the same PR. Trace the real hot path, fallback, layout conversions and pin/build
dependencies, not just the Python API.

For **every XPU PR**, including kernel-xpu, use the reference's synchronization
checklist: scalar reads, D2H, dynamic indexing, queue/event waits, setup vs per-step
costs, and graph capture/replay. Graph use does not prove host waits disappear.

### 2. Device / platform contracts (BLOCK if broken)
- **`is_cpu()` ≠ `--device cpu`**. Real CPU engine requires `SGLANG_USE_CPU_ENGINE=1`.
  Code that branches only on `device == "cpu"` without `is_cpu()` / AMX checks often
  misses packing and `forward_cpu` dispatch.
- **`is_xpu()`** is `torch.xpu.is_available()`. There is **no in-tree `XpuSRTPlatform`**;
  new code that only uses `current_platform` may silently miss XPU. Prefer `is_xpu()` /
  `MultiPlatformOp.forward_xpu` / explicit `device == "xpu"` where appropriate.
- Do not assume FlashInfer: `is_flashinfer_available()` is CUDA-only. XPU/CPU must use
  `intel_xpu` / `intel_amx` / `triton` / `torch_native`.
- Device affinity: XPU uses `ZE_AFFINITY_MASK` (not `CUDA_VISIBLE_DEVICES`). Flag docs
  or tests that set the wrong env on XPU.
- New `SGLANG_*` env vars must follow [env-var-conventions](../env-var-conventions/SKILL.md).
  Known Intel envs: `SGLANG_USE_CPU_ENGINE`, `SGLANG_CPU_OMP_THREADS_BIND`,
  `SGLANG_USE_SGL_XPU`, `SGLANG_CPU_QUANTIZATION`.

### 3. CUDA hardcoding & import hygiene (common breakage)
Flag in shared modules:
- Bare `torch.cuda.*`, `.cuda()`, `cudaStream_t`, NCCL-only paths without a device guard
- Top-level `import flashinfer` or CUDA-only triton extras that break CPU/XPU import
- Missing lazy imports inside CUDA branches (attn registry factories are the pattern)
- Prefer `torch.get_device_module(device)`, `device_context()`, or platform helpers

### 4. MultiPlatformOp / kernel dispatch
For new or changed ops deriving `MultiPlatformOp`:
- CPU AMX path needs a real `forward_cpu` (default just calls `forward_native`)
- XPU path needs `forward_xpu` when native is wrong/slow; default is `forward_native`
- Dispatch order is CUDA → HIP → **CPU+AMX** → NPU → **XPU** → … → native
  (`multi_platform.py`). CPU without AMX falls through to native — call that out if
  the PR assumes AMX always exists.
- Weight packing: AMX requires OC%16==0 and IC%32==0 or `use_intel_amx_backend=False`.

### 5. Attention backends
| Platform | Preferred backend | Notes |
|---|---|---|
| CPU x86 | `intel_amx` (default via `_handle_cpu_backends`) | Arm64 defaults `torch_native` |
| XPU | `intel_xpu` (docs require explicit flag; generic default often lands on `triton`) | Needs XMX (`xpu_has_xmx_support`, currently FP64-property proxy) |

Hard rules:
- **MLA + `intel_xpu`**: decode-only. Prefill with `intel_xpu` must error / be split
  (`--decode-attention-backend intel_xpu`, prefill `triton`). See
  `_handle_attention_backend_compatibility` in `server_args.py`.
- **Page size**: `intel_xpu` MHA pages 64/128 (auto→128); MLA decode 16/32/64/128.
  Do not introduce incompatible page sizes without updating the constraint helpers.
- Registry: `attention_registry.py` registers `intel_amx` / `intel_xpu`. New backends
  must not steal these names or break factory lazy imports.

### 6. Graphs / compile
- Flag names still say `cuda-graph-*` but apply to XPU/CPU runners — OK, but docs/tests
  must use the XPU-allowed values.
- **XPU defaults**: decode graph **off**; prefill `tc_piecewise` **off** unless user
  locks `--cuda-graph-backend-prefill` / `--cuda-graph-config`.
  Allowed decode backends: `full` / `disabled` only (`_handle_xpu_backends`).
- CPU uses `CPUGraphRunner`; XPU uses `XPUGraphRunner` /
  `XPUPiecewiseBackend` — changes to `cuda_graph_setup.py` / compilation backend
  selection must keep these mappings.
- Spec utils disable `@torch.compile` on XPU — do not re-enable without hardware proof.
- For changed XPU hot paths, verify capture **and replay**, including work outside
  the captured region. Host scalar reads and data-dependent output shapes can
  prevent capture or retain synchronization; graph replay does not remove
  gather/scatter bandwidth. See the synchronization checklist in §1c.

### 7. Quantization / MoE
- CPU AMX: W8A8, FP8, AWQ/GPTQ AMX schemes, MXFP4 registration when `is_cpu()`.
- XPU quant/unquant fused paths often require `SGLANG_USE_SGL_XPU` + **sgl-kernel-xpu**.
  A Python-only PR does not ship XPU kernels.
- Flag FP8/INT4/MoE "supported on all devices" claims without CPU/XPU branches or tests.
- GPT-OSS on XPU: **bf16 only** (dtype `NotImplementedError` paths exist — preserve them).

### 8. Features that are unsupported or experimental (docs must match code)
Do **not** let a PR silently enable or document these as GA on Intel without owner sign-off:

| Feature | CPU | XPU |
|---|---|---|
| Speculative decoding | Supported (EAGLE CPU tests exist) | **Not yet** (docs + backend TODOs) |
| PD disaggregation | Not a documented path | Experimental NIXL; needs `UCX_POSIX_USE_PROC_LINK=n` |
| Memory saver / TBO / breakable CG | Limited / N/A | **Not supported** |
| Overlap schedule | Docs recommend `--disable-overlap-schedule` | Same |
| HiCache | Not a first-class Intel story | Can disable `tc_piecewise`; do not claim support |

`XPUAttentionBackend` class docstring TODOs (PD, spec, graph, MLA prefill) are still
live constraints until explicitly closed in code **and** docs.

### 9. Distributed / PD
- CPU dist backend: `gloo`. XPU: `torch.xpu` device + `XpuCommunicator`.
- NIXL on XPU: device pointers may set bit 63 — pointer math must be **uint64**
  (see `test/registered/disaggregation/test_disaggregation_xpu.py`).
- NUMA: CPU TP binding via `SGLANG_CPU_OMP_THREADS_BIND`; flag TP changes that ignore it.

### 10. Packaging / Docker / deps
- XPU install must keep PyTorch from `https://download.pytorch.org/whl/xpu`.
  Flag anything that pulls CUDA torch/triton onto XPU images.
- `xgrammar` on XPU: install `--no-deps` (+ `apache-tvm-ffi`) — CUDA triton conflict.
  See `pyproject_xpu.toml` comments and `xpu.mdx`.
- CPU: `pyproject_cpu.toml` + `xeon.Dockerfile` (`SGLANG_USE_CPU_ENGINE=1`,
  `LD_PRELOAD` libiomp5/tcmalloc). Do not drop these.
- Version pins in `pyproject_{cpu,xpu}.toml` and Dockerfiles need coordinated bumps.

### 11. Tests & CI registration
Use [write-sglang-test](../write-sglang-test/SKILL.md) when needed. For each new
or changed registration, derive valid suite names from `test/run_suite.py` and
the workflows **at the PR head**, not examples in this skill. Verify path filters,
suite dispatch and top-level imports in the target CI image. Prefer nearby
registered fixtures/models; do not register CI tests under `python/sglang/`.
Apply the common coverage gate above; a registered but undispatched test is not
coverage, and a single invalid registration can break other backends' discovery.

### 12. Docs & cookbook parity
- Operator truth: `docs/docs/hardware-platforms/cpu_server.mdx` and `xpu.mdx`.
- Launch examples: `--device cpu|xpu`, correct attn backend, `--disable-overlap-schedule`
  where required, page-size notes for `intel_xpu`.
- Prefer `sglang serve` in new docs; flag regressions to deprecated launchers in
  Intel pages if surrounding pages already migrated.
- Cookbook CPU/XPU cells must not invent unsupported quant/graph/spec flags.

### 13. Prior review feedback
- `gh api repos/sgl-project/sglang/pulls/<N>/comments` — unresolved Intel-related
  requests still open?

### 14. CI failure attribution (red check ≠ PR bug)
Read [ci-failure-attribution.md](references/ci-failure-attribution.md) for the
signature -> baseline -> code-overlap -> rerun procedure. Assign each failure
exactly one label: `PR-CAUSED`, `PRE-EXISTING`, `FLAKE / INFRA`, or `UNKNOWN`.
Prioritize non-Intel regressions; never dismiss a PR-caused failure as Intel noise.
Link evidence before claiming a pre-existing failure. Failure attribution and
coverage are separate: an infra failure can still leave required coverage missing.
Do not demand blanket-green Intel CI when tests ran and baseline failures are proven.

## Output

Follow the Language rules above. Lead with concrete findings in severity order,
then a short verdict (`APPROVE | COMMENT | REQUEST CHANGES | BLOCKED`) and evidence.
Each code finding needs a file/line link, impact and action; pure CI attribution
may use a job link. Give 2-4 sentence English drafts with locations for actionable
findings; label style/reuse suggestions as non-blocking nits.

For sglang, briefly cover title, `other-CI`, Intel coverage, enabling parity when
relevant, and perf/bench/UT evidence. For kernel-xpu, keep only its focused checks
and actual CI status. Mention reuse/convention findings when meaningful; omit
irrelevant checklist items and long templates. Never publish without user approval.

## Related skills

- [write-sglang-test](../write-sglang-test/SKILL.md) — registering CPU/XPU tests
- [ci-workflow-guide](../ci-workflow-guide/SKILL.md) — suite/stage orchestration
- [sglang-bisect-ci-regression](../sglang-bisect-ci-regression/SKILL.md) — when a
  failure is consistent on main and needs a culprit commit
- [env-var-conventions](../env-var-conventions/SKILL.md) — new `SGLANG_*` vars
- [sglang-runtime-context](../sglang-runtime-context/SKILL.md) — ServerArgs / runtime state

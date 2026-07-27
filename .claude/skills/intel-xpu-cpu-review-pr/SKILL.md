---
name: intel-xpu-cpu-review-pr
description: Review SGLang or sgl-kernel-xpu PRs for Intel XPU/CPU ownership. For sglang: three pillars (other CI, enabling parity, perf/kernels). For sgl-kernel-xpu: perf (benchmarks required for opt PRs), simplify logic, FlashAttention/FlashInfer algorithm parity, and UT coverage updates. Chinese analysis; English PR comment drafts. Run with /intel-xpu-cpu-review-pr <PR number> [--repo sgl-kernel-xpu].
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

### Pillar 1 detail — other CI

Intel enabling / shared-SRT edits that make XPU/CPU better while turning CUDA red
are an automatic **BLOCK**. Prefer device-guarded branches, lazy CUDA-only imports,
and no accidental change to CUDA defaults. Intel CI is noisy (§14); **other-device
CI that this PR newly fails is not negotiable noise**.

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
- Docs under `docs_new/docs/hardware-platforms/{cpu_server,xpu}.mdx` or cookbook cells
  claiming CPU/XPU support

Also review **CUDA-centric "portable" refactors** that touch shared SRT code even if the
PR title says NVIDIA-only — Intel breakage is usually accidental.

Skip a deep Intel review only when the diff is clearly isolated (e.g. pure CUDA kernel
under `sgl-kernel/csrc/gpu/**` with no Python dispatch changes, or docs-only unrelated
to hardware platforms). Still skim for top-level CUDA-only imports in shared modules.

## Steps

### A. sgl-kernel-xpu repo (light)

1. `gh pr view/diff <N> --repo sgl-project/sgl-kernel-xpu`
2. Follow **only** [sgl-kernel-xpu-review.md](references/sgl-kernel-xpu-review.md):
   **性能（优化 PR 必须有 benchmark）→ 能否化简 →（attention）FA/FlashInfer
   并行维·tile·数学等价 → UT 是否覆盖改动（不够就补）**
3. 中文短报告 + 英文 comment 草稿。不要套用下面的 sglang 全量 checklist。

### B. sglang repo (full)

1. `gh pr view <N> --repo sgl-project/sglang --json title,body,files,author,baseRefName,headRefName,labels,commits,reviews,statusCheckRollup`
2. `gh pr diff <N> --repo sgl-project/sglang`
3. Classify impact: **CPU-only / XPU-only / both / shared-SRT / deps-Docker-CI / docs**,
   and whether it is a **model/op enabling** PR (→ §1b CUDA/HIP parity).
4. **Other-CI gate (pillar 1):** list failed checks on CUDA / main `pr-test.yml`
   (and AMD if present). Attribute each with §14. Any **PR-CAUSED** failure
   outside Intel → **BLOCK** immediately.
5. Grep the diff for checklist hot patterns. Enabling PRs → §1b parity with
   CUDA/HIP ([enabling-parity.md](references/enabling-parity.md)).
6. **Perf / kernels (pillar 3):** does the PR touch `sgl-kernel/csrc/cpu/`,
   bump `pyproject_xpu.toml` / link sgl-kernel-xpu, or only Python?
   See §1c / [performance-kernels.md](references/performance-kernels.md).
7. List Intel CI outcomes; attribute reds with §14 (Intel flake ≠ auto-block).
8. Cross-check claimed features vs [platform-map.md](references/platform-map.md)
   and `cpu_server.mdx` / `xpu.mdx`.
9. Output in pillar order: **其他 CI → 接入对齐 → 性能/kernel → Intel 细节**.

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
- **Verdict required in every review:** `other-CI: PASS | PRE-EXISTING-red | PR-CAUSED-BLOCK`.
  Only `PR-CAUSED-BLOCK` on non-Intel jobs blocks the PR from an Intel-owner review.

How to check quickly:

```bash
gh pr checks <N> --repo sgl-project/sglang
# Focus on pr-test.yml / CUDA jobs first; attribute with §14
gh run list --repo sgl-project/sglang --workflow=pr-test.yml --branch <pr-head> --limit 5
```

### 1. Triage & scope
- Does the PR title/body match the Intel surface area it actually touches?
- Flag "NVIDIA-only" claims that still edit shared dispatch (`MultiPlatformOp`,
  `server_args`, `common.py` device helpers, attention registry).
- Note whether follow-up work is needed in **out-of-tree** `sgl-kernel-xpu`
  (`https://github.com/sgl-project/sgl-kernel-xpu`) — XPU kernels are not in-tree.
- **Enabling PR?** (new model / new op / new quant / new attn path / "support XPU|CPU")
  → also run §1b. These PRs are common on Intel and often invent a one-off hook
  that diverges from CUDA/HIP.

### 1b. Model / operator enabling — CUDA / HIP 接入一致性
Intel enabling work must plug into the **same extension points** other devices use,
not a parallel Intel-only side path — unless there is an explicit, documented reason
(kernel out-of-tree, missing hardware feature, etc.).

Full checklist:
[references/enabling-parity.md](references/enabling-parity.md).

When reviewing an enabling PR, open the CUDA (and HIP if present) reference
implementation side-by-side and verify:

1. **Hook point**: same registry / `MultiPlatformOp` / `ServerArgs` handler /
   model override list that CUDA uses — not a new `if is_xpu():` island in a
   random call site.
2. **Dispatch shape**: `forward_cuda` / `forward_hip` / `forward_cpu` /
   `forward_xpu` symmetry. HIP often aliases CUDA; **CPU/XPU must not blindly
   alias** unless the kernel is truly shared. Default `forward_xpu`→native and
   `forward_cpu`→native are footguns in enabling PRs — flag silent fallbacks.
3. **Registration completeness**: attn backend name in `attention_registry`,
   quant method in `QUANTIZATION_METHODS` (and CPU allowlist if needed), graph
   runner map in `cuda_graph_setup`, model-arch accepted backends in
   `server_args` / `arg_groups/overrides.py` (e.g. Gemma4 / Llama4 / GPT-OSS
   lists include `intel_xpu` / `intel_amx` next to `triton` / `fa3` / `aiter`
   **or** explicitly reject with a clear error).
4. **Capability honesty**: if CUDA path enables graph / spec / FP8 / MLA prefill
   and Intel cannot, the enabling PR must **gate or error**, not pretend parity.
   Mirror how HIP/NPU leave unsupported branches — do not copy CUDA defaults
   onto XPU.
5. **API / flag surface**: same CLI flags and config keys as CUDA where behavior
   exists; do not invent `--xpu-only-*` when an existing flag can grow a device
   branch. Device affinity env stays platform-native (`ZE_AFFINITY_MASK` vs
   `CUDA_VISIBLE_DEVICES`).
6. **Tests**: CUDA enabling PRs usually add `register_cuda_ci`; Intel enabling
   should add `register_xpu_ci` / `register_cpu_ci` at the analogous suite, or
   state "CUDA-only" in the PR. Compare test structure (server args, model id,
   assertions) to the CUDA twin — not a bare import smoke if CUDA has e2e.
7. **Docs / cookbook**: same command shape as CUDA docs, with Intel device /
   attn backend / known limitations swapped in — not a divergent launch story.

**Anti-patterns to BLOCK or call out:**
- Copy-paste of CUDA kernel launch wrappers with `torch.cuda` left inside
- New Intel-only helper that duplicates an existing `MultiPlatformOp`
- Model override allowlist updated for `triton`/`fa3` but forgetting `intel_xpu`
  / `intel_amx` when the PR claims Intel support
- "Works on XPU" via unintended `forward_native` with no kernel + no test
- HIP followed CUDA by alias; Intel aliases CUDA too even though kernels differ

### 1c. Performance & kernel placement (pillar 3)
Full detail: [references/performance-kernels.md](references/performance-kernels.md).

| | CPU | XPU |
|---|---|---|
| Kernel location | **In-tree** `sgl-kernel/csrc/cpu/` | **Other repo** [sgl-kernel-xpu](https://github.com/sgl-project/sgl-kernel-xpu) |
| Pin / build | `sgl-kernel/pyproject_cpu.toml`, Xeon Docker | `python/pyproject_xpu.toml` git pin |

Review asks:
- Perf / “fused” / “AMX” / “XPU kernel” claims → corresponding kernel diff **or**
  linked sgl-kernel-xpu PR + pin bump. Python-only = correctness at best.
- **Perf optimization PRs must include benchmark results** (hw, shapes, before/after).
  No numbers → REQUEST CHANGES / BLOCK. “No perf claim” refractors are exempt.
- Hot path actually hits the kernel (`torch.ops.sgl_kernel` / sgl-kernel-xpu ops),
  not silent `forward_native` / generic PyTorch / wrong attn backend.
- No extra sync/copy/dtype tax on the Intel branch that CUDA does not pay; no
  global flag flips that hurt CUDA or Intel throughput “by accident”.
- **Tests must cover the diff**; if existing UTs cannot, the PR must update/add
  UTs (sglang: `register_*_ci` / unit tests; sgl-kernel-xpu: in-repo UT).
  “Tested locally, UT later” is not enough.

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
Registration (see [write-sglang-test](../write-sglang-test/SKILL.md)):
- CPU unit / import-safe: `register_cpu_ci(..., suite="base-a-test-cpu")` (main `pr-test.yml`)
- CPU AMX e2e/kernels: `register_cpu_ci(..., suite="base-b-test-cpu")` → `pr-test-xeon.yml`
- XPU PR: `register_xpu_ci` → `stage-a-test-1-gpu-xpu` / `stage-b-test-1-gpu-xpu`
- XPU nightly models: `nightly-xpu-2-gpu` / `nightly-xpu-4-gpu`

Review asks:
- New Intel feature → matching `register_*_ci` (or explicit "CUDA-only" disable reason)
- Changed Intel path → existing Xeon/XPU jobs still cover it; add a focused test if not
- **If existing tests cannot cover the change → update/add UT in the same PR**
  (kernel math, new op, new backend path). Do not merge on “manual smoke only”.
- Do not put `register_*_ci` under `python/sglang/` (pre-commit rejects it)
- Prefer small models already used on Intel CI (see `test/registered/{cpu,xpu}/`)
- Path filters on `pr-test-xeon.yml` / `pr-test-xpu.yml` fire on almost all
  SRT/test/kernel changes — Intel jobs often run even for "CUDA" PRs; that does
  **not** mean every red is the PR's fault (see §14).

### 12. Docs & cookbook parity
- Operator truth: `docs_new/docs/hardware-platforms/cpu_server.mdx` and `xpu.mdx`.
- Launch examples: `--device cpu|xpu`, correct attn backend, `--disable-overlap-schedule`
  where required, page-size notes for `intel_xpu`.
- Prefer `sglang serve` in new docs; flag regressions to deprecated launchers in
  Intel pages if surrounding pages already migrated.
- Cookbook CPU/XPU cells must not invent unsupported quant/graph/spec flags.

### 13. Prior review feedback
- `gh api repos/sgl-project/sglang/pulls/<N>/comments` — unresolved Intel-related
  requests still open?

### 14. CI failure attribution (red check ≠ PR bug)
SGLang CI is unstable; **non-CUDA devices are worse**. A red `pr-test-xpu` /
`pr-test-xeon` job is a hypothesis, not a verdict. Full procedure:
[references/ci-failure-attribution.md](references/ci-failure-attribution.md).

**Priority order when many jobs are red:**

1. **Non-Intel / CUDA jobs with PR-CAUSED signature** → hard **BLOCK** (§0)
2. Intel jobs with PR-CAUSED signature → block or request changes
3. PRE-EXISTING / FLAKE on Intel → comment only, do not block
4. PRE-EXISTING red on CUDA → note it; still not introduced by this PR

For each failed job, assign exactly one label:

| Label | When | Block merge? |
|---|---|---|
| **PR-CAUSED** | New signature + clear overlap with diff (or new test the PR added) | **Yes** — always for CUDA/other-device; yes for Intel |
| **PRE-EXISTING** | Same signature on recent `main` push or many unrelated PRs | No — link evidence |
| **FLAKE / INFRA** | Timeout, runner, Docker, device init, cancelled job; or pass on rerun same SHA | No — suggest rerun |
| **UNKNOWN** | Cannot decide | Soft hold on CUDA/other; on Intel, one rerun + residual risk note |

Fast path:
1. Extract **signature**: job name + test id + error class + one message line.
2. Compare to recent `main` / unrelated PR runs of the **same workflow**.
3. Check **code overlap**: do PR files sit under what the failing test exercises?
4. If infra-looking or no overlap → rerun once before blaming the author.
5. **Never** dismiss a CUDA **PR-CAUSED** fail as "Intel CI is flaky".

Bias: on **Intel**, assume PRE-EXISTING/FLAKE until overlap is clear; if the PR
**added** the failing test or claims XPU/CPU support, assume PR-CAUSED.
On **CUDA / other devices**, bias toward PR-CAUSED when the diff touches shared
SRT and the signature is new vs base/`main`.

Do **not** require green Intel CI as a blanket gate when endemic failures are
documented — but **do** require not newly breaking other CI.

## Output

Default: **短中文 + 短英文草稿**（见 §Language）。默认偏短，不写长 checklist 复述。

### A. 中文报告（主输出，尽量短 + 行号）

结构固定、能省则省：

```text
## 摘要
<1–3 句：PR 做什么 + 总评 APPROVE|COMMENT|REQUEST CHANGES|BLOCKED>

## 支柱/焦点
- 其他 CI: …
- 对齐: …          # enabling 才写
- 性能/kernel: …   # 落点 + bench有无 + UT有无
- （kernel-xpu）化简 / FA对照: …

## 问题
- 🔴/⚠️ `path:Lstart-Lend` — 一句话原因 + 一句话建议
```

要求：
- **每条问题必须有行号**（`file:12-34` 或 code citation）。纯流程/CI 归因可无行号。
- 不写大段「PR 动机散文」；不把整个 skill checklist 贴进报告。
- sgl-kernel-xpu：同样短结构，只保留 perf / 化简 / FA / bench / UT。

### B. 英文 comment 草稿（尽量短）

每条 2–4 句，**文件+行号**打头，按严重度：

```markdown
### Ready-to-paste (English)

**[blocking]** `file.hpp:120-145`: <one-line issue>. <one-line ask>.

**[missing UT]** No test covers <path/symbol>. Please add UT in this PR.

**[nit]** `file.hpp:80`: <optional simplify>.
```

不要套 `<details>` 长模板，除非用户要展开。不要自动 `gh pr comment`。

## Related skills

- [write-sglang-test](../write-sglang-test/SKILL.md) — registering CPU/XPU tests
- [ci-workflow-guide](../ci-workflow-guide/SKILL.md) — suite/stage orchestration
- [sglang-bisect-ci-regression](../sglang-bisect-ci-regression/SKILL.md) — when a
  failure is consistent on main and needs a culprit commit
- [env-var-conventions](../env-var-conventions/SKILL.md) — new `SGLANG_*` vars
- [sglang-runtime-context](../sglang-runtime-context/SKILL.md) — ServerArgs / runtime state

---
name: intel-xpu-cpu-review-pr
description: Review a pull request for Intel XPU and CPU (AMX/Xeon) platform impact in SGLang. Use when reviewing PRs that touch device dispatch, attention/quant/MoE/graph paths, Intel CI/Docker/deps, or when asked to review for XPU/CPU ownership. Run with /intel-xpu-cpu-review-pr <PR number>.
---

# Intel XPU / CPU PR Review

Review a PR from the perspective of the **Intel XPU** and **CPU (AMX Xeon)** platform owners.
Focus on: does this break, silently degrade, or leave untested Intel paths? Does it claim
features Intel does not support yet?

Path map, feature matrix, and CI suite details live in
[references/platform-map.md](references/platform-map.md) — defer to it rather than restating.

## Usage

```
/intel-xpu-cpu-review-pr <PR number>
```

Optional: `/intel-xpu-cpu-review-pr <PR number> --focus xpu|cpu|both` (default `both`).

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

1. `gh pr view <N> --repo sgl-project/sglang --json title,body,files,author,baseRefName,headRefName,labels,commits,reviews,statusCheckRollup`
2. `gh pr diff <N> --repo sgl-project/sglang`
3. Classify impact: **CPU-only / XPU-only / both / shared-SRT / deps-Docker-CI / docs**.
4. Grep the diff (and touched call sites) for the hot patterns in §Checklist.
5. Check Intel CI status on the PR (`pr-test-xeon`, `pr-test-xpu`, nightly if relevant).
6. Cross-check claimed features against the support matrix in `references/platform-map.md`
   and the operator docs (`cpu_server.mdx`, `xpu.mdx`).
7. Output per-area verdicts + overall recommendation. Prefer actionable comments
   (file + concern + suggested fix), not generic "please test on XPU".

## Checklist

### 1. Triage & scope
- Does the PR title/body match the Intel surface area it actually touches?
- Flag "NVIDIA-only" claims that still edit shared dispatch (`MultiPlatformOp`,
  `server_args`, `common.py` device helpers, attention registry).
- Note whether follow-up work is needed in **out-of-tree** `sgl-kernel-xpu`
  (`https://github.com/sgl-project/sgl-kernel-xpu`) — XPU kernels are not in-tree.

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

### 11. Tests & CI
Registration (see [write-sglang-test](../write-sglang-test/SKILL.md)):
- CPU unit / import-safe: `register_cpu_ci(..., suite="base-a-test-cpu")` (main `pr-test.yml`)
- CPU AMX e2e/kernels: `register_cpu_ci(..., suite="base-b-test-cpu")` → `pr-test-xeon.yml`
- XPU PR: `register_xpu_ci` → `stage-a-test-1-gpu-xpu` / `stage-b-test-1-gpu-xpu`
- XPU nightly models: `nightly-xpu-2-gpu` / `nightly-xpu-4-gpu`

Review asks:
- New Intel feature → matching `register_*_ci` (or explicit "CUDA-only" disable reason)
- Changed Intel path → existing Xeon/XPU jobs still cover it; add a focused test if not
- Do not put `register_*_ci` under `python/sglang/` (pre-commit rejects it)
- Prefer small models already used on Intel CI (see `test/registered/{cpu,xpu}/`)
- Check PR checks: path filters on `pr-test-xeon.yml` / `pr-test-xpu.yml` fire on almost
  all SRT/test/kernel changes — a red Intel job is in-scope even for "CUDA" PRs

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

## Output

Lead with a one-line **Intel impact** summary: `none | docs-only | CPU | XPU | both | shared-SRT risk`.

Then per area:
- ✅ PASS
- ⚠️ ISSUE: \<what + where + suggested fix\>
- 🔴 BLOCK: \<what + where + why it breaks Intel\>

Overall: **APPROVE** / **COMMENT** / **REQUEST CHANGES** / **BLOCKED**

If requesting changes, list the minimum CI evidence you want (e.g. green `pr-test-xpu`
stage-b, or a local AMX smoke command).

## Related skills

- [write-sglang-test](../write-sglang-test/SKILL.md) — registering CPU/XPU tests
- [ci-workflow-guide](../ci-workflow-guide/SKILL.md) — suite/stage orchestration
- [env-var-conventions](../env-var-conventions/SKILL.md) — new `SGLANG_*` vars
- [sglang-runtime-context](../sglang-runtime-context/SKILL.md) — ServerArgs / runtime state

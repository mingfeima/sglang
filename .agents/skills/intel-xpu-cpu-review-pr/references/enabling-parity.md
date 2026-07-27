# Enabling Parity: Intel vs CUDA / HIP

For **model / operator / quant / attention enabling** PRs on CPU or XPU.
Goal: Intel plugs into the same seams as CUDA/HIP; divergence must be
intentional and visible.

## What counts as an enabling PR

- "Add XXX support on XPU/CPU/AMX"
- New model arch tweaks (`overrides.py`, multimodal, MoE router)
- New op / fused kernel wired into SRT (`Linear`, `RoPE`, `MoE`, RMSNorm, …)
- New quant scheme or KV-cache dtype path
- New attention backend or backend allowlisting for a model
- Graph / compile enablement on Intel

## Side-by-side method

1. Find the CUDA reference (and HIP if `forward_hip` / `is_hip` exists).
2. List every file CUDA touched for the same feature on a prior PR or in-tree.
3. For each seam below, mark Intel as **aligned / intentionally different / missing**.

| Seam | CUDA typical | HIP typical | Intel expectation |
|---|---|---|---|
| Op dispatch | `MultiPlatformOp.forward_cuda` | often `forward_hip` → `forward_cuda` | Real `forward_cpu` / `forward_xpu`, **not** silent native unless documented |
| Attn backend | `register_attention_backend("fa3"\|…)` | `aiter` / `wave` / triton | `intel_amx` / `intel_xpu` registered; model allowlists updated together |
| Quant registry | `QUANTIZATION_METHODS` + cuda gates | hip/mxfp gates | CPU allowlist / `SGLANG_USE_SGL_XPU` + sgl-kernel-xpu; do not register then `NotImplemented` at runtime without guard |
| ServerArgs | `_handle_*`, graph defaults | ROCm specifics | `_handle_cpu_backends` / `_handle_xpu_backends`; do not copy CUDA graph defaults onto XPU |
| Model overrides | accepted backend sets | may add `aiter` | Add `intel_xpu`/`intel_amx` **or** assert with a clear unsupported message |
| Graph runner | CUDA graph setup map | HIP path | `CPUGraphRunner` / `XPUGraphRunner` / piecewise — same selection table |
| Distributed | NCCL | RCCL / hip | `gloo` (CPU) / `XpuCommunicator` (XPU) |
| Device env | `CUDA_VISIBLE_DEVICES` | `HIP_VISIBLE_DEVICES` / ROCr | `ZE_AFFINITY_MASK` (XPU); CPU NUMA bind env |
| Tests | `register_cuda_ci` | `register_amd_ci` | `register_cpu_ci` / `register_xpu_ci` at analogous depth |
| Docs | CUDA launch recipe | ROCm page | `cpu_server.mdx` / `xpu.mdx` + cookbook cell if claimed |

## Consistency questions (ask on every enabling PR)

1. **Where does CUDA enter?** Same function? If Intel uses a different function, why?
2. **Is the Intel branch reachable?** (`is_cpu()` needs `SGLANG_USE_CPU_ENGINE`; XPU needs `torch.xpu` + often `--attention-backend intel_xpu` / `SGLANG_USE_SGL_XPU`)
3. **Allowlists:** every `accepted_backends = (…)` / `supported_backends` the PR or model touches — is Intel included when support is claimed?
4. **Fallback honesty:** does Intel silently take `forward_native` / `triton` while the PR title says "AMX/XPU kernel"?
5. **Feature subset:** graph, spec, MLA prefill, FP8 — gated like other incomplete platforms, or accidentally on?
6. **Kernel package:** CUDA in-tree `sgl-kernel` vs XPU **out-of-tree** `sgl-kernel-xpu` — is the Python PR paired with a kernel PR/pin bump?
7. **HIP alias trap:** `forward_hip`→`forward_cuda` is common; copying that pattern as `forward_xpu`→`forward_cuda` is almost always wrong.

## Good vs bad patterns (short)

**Good**
- Extend existing `MultiPlatformOp` with `forward_xpu`/`forward_cpu`
- Add backend to registry + model allowlist + page-size constraints in one PR
- Gate unsupported CUDA features with the same style of early `ValueError` / auto-disable other platforms use
- Twin test of the CUDA e2e at Intel suite level

**Bad**
- New `intel_xxx_utils.py` that reimplements dispatch already in `MultiPlatformOp`
- Only documents `CUDA_VISIBLE_DEVICES` in an XPU enabling guide
- Claims FP8 on XPU but only lands a CUDA `#ifdef` path
- Silent fallback that CUDA would never accept without a test
- Allowlist updated for Intel while a shared default changes under CUDA CI
- **Any edit that newly fails CUDA / other non-Intel CI** — hard block; fix or
  device-guard before polishing Intel paths

## Review output hint (中文)

在中文报告里单开一小段 **「与 CUDA/HIP 接入对比」**：

- 对齐的 seam
- 有意分歧（原因）
- 缺失 / 偷偷 fallback（是否 block）

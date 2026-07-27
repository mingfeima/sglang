# Performance & Kernel Placement (CPU / XPU)

Pillar 3 of the Intel review skill. Enabling that only wires Python without the
right kernel is correctness-at-best, often a **perf regression** vs the intended
path.

## Where kernels live

| Platform | Kernel tree | Package / pin |
|---|---|---|
| **CPU (AMX)** | **In-tree** `sgl-kernel/csrc/cpu/` (+ `torch_extension_cpu.cpp`, `sgl-kernel/pyproject_cpu.toml`) | Built with SGLang CPU / Xeon image |
| **XPU** | **Out-of-tree** [sgl-project/sgl-kernel-xpu](https://github.com/sgl-project/sgl-kernel-xpu) | `python/pyproject_xpu.toml` pins `sgl-kernel @ git+…/sgl-kernel-xpu.git` |
| CUDA | In-tree `sgl-kernel/csrc/` (GPU) | Separate from Intel |

Never assume an sglang Python PR alone ships XPU kernels. Never put XPU kernels
into `sgl-kernel/csrc/cpu/`.

PRs **inside** sgl-kernel-xpu use the lighter review:
[sgl-kernel-xpu-review.md](sgl-kernel-xpu-review.md) (perf, simplify, FA/FlashInfer).

## What to look for in a PR

### Claims vs reality
- PR says “faster / fused / AMX / XPU kernel” → must touch the kernel tree above
  **or** link a paired kernel PR / pin bump.
- Python-only change that still calls `forward_native` / generic PyTorch →
  **no kernel win**; flag as correctness enablement only, or request kernel work.
- `SGLANG_USE_SGL_XPU` paths without updating sgl-kernel-xpu pin when new ops
  are required → runtime miss or fallback.

### CPU (`sgl-kernel/csrc/cpu/`)
- New/changed `.cpp` ops registered in `torch_extension_cpu.cpp`
- Weight pack / AMX tile constraints still hold (OC%16, IC%32, etc.)
- Hot path still reaches `torch.ops.sgl_kernel.*` (or the project’s CPU op API),
  not an accidental Python loop
- Xeon CI (`base-b-test-cpu`) or a local bench covers the op; microbench numbers
  in PR body when claiming speedup

### XPU (sgl-kernel-xpu repo)
- Linked kernel PR / commit SHA; `pyproject_xpu.toml` pin updated in the same
  or clearly ordered PR
- Docker / `intel/sglang-dev` image implications if the CI image embeds the pin
- No CUDA wheel / wrong triton pulled in while bumping deps for “perf”

### Shared SRT performance footguns
- Disabling overlap / graph / cuda-graph flags globally while enabling Intel
- Extra host↔device copies, syncs, or dtype casts on the Intel branch that CUDA
  does not pay
- Page-size / block-size choices that tank KV bandwidth on `intel_xpu`
- Falling back to `triton` / native when `intel_xpu` / AMX was supposed to run
  (silent perf cliff)

## Perf evidence expectations

| Change type | Minimum evidence |
|---|---|
| Pure wiring / correctness | State “no perf claim”; OK without bench |
| Perf optimization (any platform) | **Required:** before/after benchmark in PR body (hw, shapes, metric). Missing → block |
| New CPU kernel | Bench on Xeon (or kernel unit bench) + UT covering the op |
| New XPU kernel | Bench on BMG/Arc + kernel-repo PR link + pin + UT |
| Default backend / graph change | Latency/throughput note; watch CUDA CI for collateral |

If the PR body claims speedup with no numbers → 🔴.
If the diff is not covered by existing tests and UT is not updated → 🔴.

## Review output hint (中文)

单开 **「性能 / kernel」**：

- kernel 落点：CPU in-tree path / XPU 外仓 PR+pin / 仅 Python
- 是否真走到加速路径（还是 native fallback）
- 对 CUDA/其他 device 热路径有无额外开销
- 证据：数字 / “无 perf 宣称”

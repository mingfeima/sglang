# Intel XPU / CPU Platform Map

Companion reference for [intel-xpu-cpu-review-pr](../SKILL.md). Keep this factual;
update when paths or support status change.

## Architecture (asymmetric)

| | CPU (AMX Xeon) | XPU (Arc / BMG) |
|---|---|---|
| Device string | `cpu` | `xpu` |
| Engine gate | `SGLANG_USE_CPU_ENGINE=1` → `is_cpu()` | `torch.xpu.is_available()` → `is_xpu()` |
| In-tree platform | `CpuSRTPlatform` | **None** (helpers + hardware_backend) |
| Kernels | In-tree `sgl-kernel/csrc/cpu/` (`sglang-kernel-cpu`) | Out-of-tree [sgl-kernel-xpu](https://github.com/sgl-project/sgl-kernel-xpu) |

Perf review: [performance-kernels.md](performance-kernels.md). Python-only
enables without the matching kernel are not treated as perf wins.

| Attn backend | `intel_amx` (x86 default) | `intel_xpu` (explicit; else often `triton`) |
| Dist backend | `gloo` | `torch.distributed` + `XpuCommunicator` |
| CI runners | `xeon-gnr` + generic CPU | `intel-bmg` / `intel-bmg-nightly` |

## Key paths

### Detection & args
| Path | Role |
|---|---|
| `python/sglang/srt/utils/common.py` | `is_cpu`, `is_xpu`, `get_device`, AMX/XMX probes, `use_intel_xpu_backend` |
| `python/sglang/srt/platforms/cpu.py` | `CpuSRTPlatform` |
| `python/sglang/srt/platforms/__init__.py` | Platform discovery (`SGLANG_USE_CPU_ENGINE`) |
| `python/sglang/srt/server_args.py` | `_handle_cpu_backends`, `_handle_xpu_backends`, MLA/`intel_xpu` checks |
| `python/sglang/srt/arg_groups/overrides.py` | Model overrides (page size, GPT-OSS, Llama4, Gemma4) |
| `python/sglang/srt/environ.py` | `SGLANG_CPU_QUANTIZATION`, `SGLANG_PLATFORM`, … |
| `python/sglang/srt/utils/numa_utils.py` | `SGLANG_CPU_OMP_THREADS_BIND` |

### Attention / ops
| Path | Role |
|---|---|
| `python/sglang/srt/layers/attention/attention_registry.py` | Registers `intel_amx`, `intel_xpu` |
| `python/sglang/srt/layers/attention/intel_amx_backend.py` | CPU AMX attention |
| `python/sglang/srt/layers/attention/xpu_backend.py` | `XPUAttentionBackend` (+ TODOs) |
| `python/sglang/srt/layers/amx_utils.py` | AMX weight pack / quant enums |
| `python/sglang/srt/layers/utils/multi_platform.py` | `forward_cpu` / `forward_xpu` dispatch |

### Graphs / compile
| Path | Role |
|---|---|
| `python/sglang/srt/hardware_backend/xpu/graph_runner/` | `XPUGraphRunner`, full graph backend |
| `python/sglang/srt/model_executor/cpu_graph_runner.py` | CPU graph capture |
| `python/sglang/srt/compilation/xpu_piecewise_backend.py` | XPU `tc_piecewise` |
| `python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py` | Maps device → graph runner |

### Quant / MoE / spec
| Path | Role |
|---|---|
| `python/sglang/srt/layers/quantization/{w8a8_int8,fp8,mxfp4,unquant}.py` | AMX / XPU quant paths |
| `python/sglang/srt/layers/quantization/{gptq,awq}/schemes/*_cpu.py` | GPTQ/AWQ AMX schemes |
| `python/sglang/srt/hardware_backend/cpu/quantization/` | GPTQ/AWQ kernel wrappers |
| `python/sglang/srt/speculative/{eagle_utils,draft_utils,spec_utils}.py` | CPU/XPU speculative branches |
| `sgl-kernel/csrc/cpu/` | In-tree AMX kernels |

### Distributed / PD
| Path | Role |
|---|---|
| `python/sglang/srt/distributed/device_communicators/xpu_communicator.py` | XPU collectives |
| `python/sglang/srt/disaggregation/nixl/conn.py` | NIXL PD (XPU uint64 ptr caveat) |
| `test/registered/disaggregation/test_disaggregation_xpu.py` | XPU PD smoke |

### Packaging / Docker / docs / tests
| Path | Role |
|---|---|
| `python/pyproject_cpu.toml` | CPU package deps |
| `python/pyproject_xpu.toml` | XPU package deps (`sgl-kernel` from sgl-kernel-xpu) |
| `docker/xeon.Dockerfile` | Xeon image |
| `docker/xpu.Dockerfile` | XPU image (oneAPI / SYCL UMD) |
| `docs_new/docs/hardware-platforms/cpu_server.mdx` | CPU operator guide |
| `docs_new/docs/hardware-platforms/xpu.mdx` | XPU operator guide |
| `test/registered/cpu/` | AMX / CPU e2e + kernels |
| `test/registered/xpu/` | XPU stage-a/b + nightly models |
| `python/sglang/test/xpu/` | XPU test helpers |

### CI workflows
| Workflow | Suite focus |
|---|---|
| `.github/workflows/pr-test.yml` | `base-a-test-cpu` (generic CPU) |
| `.github/workflows/pr-test-xeon.yml` | `base-b-test-cpu` on `xeon-gnr` |
| `.github/workflows/pr-test-xpu.yml` | `stage-a/b-test-1-gpu-xpu` on `intel-bmg` |
| `.github/workflows/nightly-test-intel.yml` | `nightly-xpu-{2,4}-gpu` |
| `.github/workflows/release-docker-xeon.yml` | Publish `*-xeon` tags |
| `.github/workflows/release-docker-intel-xpu-nightly.yml` | Publish `intel/sglang-dev` |
| `.github/workflows/xpu-ci-job-monitor.yml` | Pass-rate / fleet monitor for XPU jobs |
| `scripts/ci/utils/xpu_job_monitor.py` | Query XPU job history / summaries |

**Stability note:** Xeon/XPU PR workflows commonly fail on unrelated PRs. Always
attribute reds per [ci-failure-attribution.md](ci-failure-attribution.md) before
blocking. Baseline = recent `--branch main` runs of the same workflow (these
workflows use `push` to main, not CUDA's `schedule` event) plus other recent PRs.

## Env / CLI cheat sheet

| Knob | Platform | Meaning |
|---|---|---|
| `SGLANG_USE_CPU_ENGINE=1` | CPU | Required for `is_cpu()` and CPU platform |
| `SGLANG_CPU_OMP_THREADS_BIND` | CPU | Per-TP-rank core masks (`0-39\|43-82\|…`) |
| `SGLANG_USE_SGL_XPU` | XPU | Opt into sgl-kernel-xpu compute paths |
| `SGLANG_CPU_QUANTIZATION` | CPU | Quant-related (see `environ.py`) |
| `ZE_AFFINITY_MASK` | XPU | Device visibility (like `CUDA_VISIBLE_DEVICES`) |
| `UCX_POSIX_USE_PROC_LINK=n` | XPU PD | Required for NIXL |
| `--device cpu\|xpu` | both | ServerArgs device |
| `--attention-backend intel_amx\|intel_xpu` | both | Preferred attn |
| `--disable-overlap-schedule` | both | Documented recommendation |
| `--cuda-graph-backend-decode full` | XPU | Opt-in decode graph |
| `--cuda-graph-backend-prefill tc_piecewise` | XPU | Opt-in prefill graph |

## Support matrix (review against claims)

| Feature | CPU (AMX) | XPU |
|---|---|---|
| Attention | `intel_amx` / `torch_native` | `intel_xpu` / `triton` fallback |
| MLA | CPU MLA fused RoPE path exists | `intel_xpu` **decode-only**; prefill → triton |
| FlashInfer | No | No |
| Decode graph | `CPUGraphRunner` | Opt-in `full` |
| Prefill graph | Limited | Opt-in `tc_piecewise` |
| FP8 / W8A8 / AWQ / GPTQ | AMX schemes | Partial; often needs sgl-kernel-xpu / open PRs |
| Speculative decoding | Yes (CI under `test/registered/cpu/`) | **Not yet** |
| PD disagg | Not documented | Experimental NIXL |
| Memory saver / TBO / breakable CG | Limited | **Not supported** |
| Overlap schedule | Disable recommended | Disable recommended |
| Multimodal | Some AMX paths | OCR / Gemma4 e2e on XPU CI |

## Hot grep patterns

```bash
# Platform gates
rg -n 'is_cpu\(|is_xpu\(|cpu_has_amx_support|xpu_has_xmx_support|use_intel_amx_backend|use_intel_xpu_backend' -g '*.py'

# CUDA assumptions in shared code
rg -n 'torch\.cuda\.|\.cuda\(|flashinfer|cudaStream' python/sglang/srt -g '*.py'

# Backend / page constraints
rg -n 'intel_amx|intel_xpu|_handle_cpu_backends|_handle_xpu_backends|_intel_xpu_page' python/sglang/srt -g '*.py'

# CI registration
rg -n 'register_cpu_ci|register_xpu_ci' test/registered
```

## Helper semantics (do not confuse)

```text
is_cpu()     == (SGLANG_USE_CPU_ENGINE=1) AND host is x86/arm64 with torch.cpu
is_xpu()     == hasattr(torch,"xpu") AND torch.xpu.is_available()
get_device()  checks is_cpu() BEFORE cuda/xpu — CPU engine wins when env set
MultiPlatformOp CPU branch requires is_cpu() AND cpu_has_amx_support()
xpu_has_xmx_support() currently proxies via device property has_fp64
```

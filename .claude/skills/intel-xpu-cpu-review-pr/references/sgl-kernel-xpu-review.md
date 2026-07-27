# sgl-kernel-xpu PR Review (lighter focus)

Use when the PR is in **https://github.com/sgl-project/sgl-kernel-xpu**
(not the main sglang tree). Do **not** run the full sglang Intel checklist —
skip CUDA/HIP enabling parity, ServerArgs, cookbook, Intel CI flake theater
unless the PR clearly changes a public op signature that sglang calls.

## Usage

```
/intel-xpu-cpu-review-pr <N> --repo sgl-kernel-xpu
# or paste a PR URL under sgl-project/sgl-kernel-xpu
```

Fetch:

```bash
gh pr view <N> --repo sgl-project/sgl-kernel-xpu --json title,body,files,author,baseRefName,commits,reviews
gh pr diff <N> --repo sgl-project/sgl-kernel-xpu
```

## Focus areas

### 1. Performance
- Is the hot path actually improved (less sync, better occupancy, fewer passes,
  better memory traffic), or just a refactor that may regress?
- Launch config: work-group / subgroup / EU mapping; shared local memory use;
  register pressure; redundant global loads.
- Prefetch / pipeline / fuse opportunities left on the table.
- Softmax / rescale / epilogue: any extra device↔host or unnecessary
  materialization of S / P matrices.

### 1b. Benchmark required for perf PRs (**BLOCK** if missing)
If the PR is (or claims to be) a **performance optimization** — faster kernel,
better tiling, fuse, less bandwidth, “X% speedup”, etc. — the author **must**
include benchmark results in the PR body (or a linked doc/comment):

- Hardware (e.g. BMG / Arc), dtype, shapes (batch/heads/seq/page), compare baseline
- Before/after numbers (latency and/or throughput); methodology one-liner
- No numbers + only hand-wavy “should be faster” → **REQUEST CHANGES / BLOCK**

Pure correctness / refactor / API-wire PRs with an explicit “no perf claim” are
exempt; if the title/body implies speedup, they are **not** exempt.

### 2. Can the logic be simplified?
- Dead branches, duplicated epilogues, copy-paste for prefill vs decode that
  could share a template.
- Over-parameterized helpers; magic constants that should be named once.
- Control flow that exists only for a debug path left enabled.
- Prefer readable structure **when it does not hurt the inner loop**; do not
  ask to “simplify” away a perf-critical specialization without a reason.

### 3. Overlap with FlashAttention / FlashInfer — algorithm parity
When the kernel implements attention (or a close cousin: varlen, paged KV,
MLA decode, merge states, etc.), compare to the reference designs in
**FlashAttention** and **FlashInfer** (and FA2/FA3 papers / upstream code the
authors cite). Ask:

| Question | What to check |
|---|---|
| **Parallelism dimensions** | Which axes are parallel (batch, head, sequence/Q tiles, KV tiles)? Same as FA/FlashInfer or a deliberate XPU mapping? Wrong axis → poor locality or races. |
| **Tile / block split** | Br, Bc (Q/KV tile sizes), how K/V are blocked, online softmax block schedule. Do tile shapes match XPU subgroup/SLM limits? Arbitrary tiles that fight the FA schedule need justification. |
| **Math equivalence** | Online softmax / rescale, causal & window masks, softcap, FP8/BF16 accumulate dtype, LSE merge (`merge_state`), sink tokens, MLA latent dims. Output and LSE should match FA/FlashInfer within tol — silent formula drift is a **BLOCK**. |
| **Paged / varlen layout** | page_size, indptr vs page table, ragged Q — same contract sglang `XPUAttentionBackend` expects (`flash_attn_with_kvcache`, `flash_attn_varlen_func`, `flash_mla_decode`). |
| **Numerical path** | Where is scale applied; is causal mask inside the gemm or after; any reassociation that breaks FA’s online softmax invariant. |

If the PR is **not** attention-like (gemm, moe, norm, quant), skip §3 or only
compare to the CUDA sibling kernel in sgl-kernel / sgl-kernel-xpu history.

### 4. Tests must cover the change (**update UT if not**)
- Existing UTs should exercise the new/changed code path (shapes, dtypes,
  causal/page modes, edge cases the diff touches).
- If current tests **cannot** cover the change (new op, new tile path, new
  numeric mode) → author must **add/update unit tests** in the same PR.
- “Tested locally” / “will add later” without UT diff → **REQUEST CHANGES**.
- Prefer golden vs FA/FlashInfer or vs previous kernel within tolerance for
  attention math changes.

## Light extra checks (one-liners)

- Public C++/Python op schema change → note sglang pin bump needed
  (`python/pyproject_xpu.toml` in sglang).
- Obvious correctness bugs (OOB, wrong stride, missing fence) still **BLOCK**.
- Do not demand sglang-tree CI green from this repo alone.

## Output (中文 + 英文草稿)

中文报告覆盖：

1. PR 在做什么（kernel / API）
2. **性能**：更好 / 存疑；若是优化 PR → **有无 benchmark**（无则 🔴）
3. **能否化简**：具体建议或「已够干净」
4. **与 FA / FlashInfer**（若相关）：并行维、tile、数学是否等价；不等价则 🔴
5. **测试覆盖**：现有 UT 是否盖住改动；缺则是否已补 UT
6. 总评

英文 comment 针对作者：短、指到具体循环/tile/公式/缺失的 bench 或 UT，
不要贴整份 sglang checklist。

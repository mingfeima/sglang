import math
import unittest

# TODO: use interface in cpu.py
import torch

from sglang.srt.layers.amx_utils import CPUQuantMethod

kernel = torch.ops.sgl_kernel

torch.manual_seed(1183)

from utils import (
    BLOCK_K,
    BLOCK_N,
    MXFP4QuantizeUtil,
    factor_for_scale,
    fp8_max,
    fp8_min,
    native_fp8_fused_moe,
    parametrize,
    precision,
    scaled_weight,
    torch_naive_fused_moe,
    torch_naive_fused_moe_gptoss,
    torch_w8a8_per_column_fused_moe,
    unpack_and_dequant_awq,
)

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-b-test-cpu")

# GPT-OSS / MiniMax-style SwiGLU clamp params (used when bias=True)
_SWIGLU_ALPHA = 1.702
_SWIGLU_LIMIT = 7.0


def run_fused_experts(
    a,
    w1,
    w2,
    topk_weight,
    topk_ids,
    *,
    quant=CPUQuantMethod.UNQUANT,
    w1_scale=None,
    w2_scale=None,
    w1_zp=None,
    w2_zp=None,
    block_size=None,
    w1_bias=None,
    w2_bias=None,
    alpha=None,
    limit=None,
    is_vnni=True,
    inplace=False,
):
    return kernel.fused_experts_cpu(
        a,
        w1,
        w2,
        topk_weight,
        topk_ids.to(torch.int32),
        inplace,
        quant,
        w1_scale,
        w2_scale,
        w1_zp,
        w2_zp,
        block_size,
        w1_bias,
        w2_bias,
        alpha,
        limit,
        is_vnni,
    )


def make_routing(m, e, topk, dtype, renormalize=False):
    score = torch.randn((m, e), dtype=dtype)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    if renormalize:
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
    return topk_weight, topk_ids


class TestFusedExperts(CustomTestCase):

    @parametrize(
        m=[2, 32],
        n=[32, 128],
        k=[32, 128],
        e=[4],
        topk=[2],
        renormalize=[False, True],
        bias=[False, True],
    )
    def test_bf16_moe(self, m, n, k, e, topk, renormalize, bias):
        dtype = torch.bfloat16
        a = torch.randn((m, k), dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), dtype=dtype) / 10
        w2 = torch.randn((e, k, n), dtype=dtype) / 10
        packed_w1 = kernel.convert_weight_packed(w1)
        packed_w2 = kernel.convert_weight_packed(w2)

        if bias:
            # bias path also exercises clamped SwiGLU (alpha/limit)
            w1_b = torch.randn((e, 2 * n), dtype=torch.float32) / 10
            w2_b = torch.randn((e, k), dtype=torch.float32) / 10
            topk_weight, topk_ids = make_routing(m, e, topk, dtype, renormalize)
            ref = torch_naive_fused_moe_gptoss(
                a,
                w1,
                w2,
                w1_b,
                w2_b,
                topk_weight,
                topk_ids,
                False,  # already renormalized above when requested
                _SWIGLU_ALPHA,
                _SWIGLU_LIMIT,
                e,
            )
            out = run_fused_experts(
                a,
                packed_w1,
                packed_w2,
                topk_weight,
                topk_ids,
                w1_bias=w1_b,
                w2_bias=w2_b,
                alpha=_SWIGLU_ALPHA,
                limit=_SWIGLU_LIMIT,
            )
        else:
            score = torch.randn((m, e), dtype=dtype)
            ref = torch_naive_fused_moe(a, w1, w2, score, topk, renormalize)
            topk_weight, topk_ids = kernel.grouped_topk_cpu(
                a, score, topk, renormalize, 1, 1, 0, None, None
            )
            out = run_fused_experts(
                a, packed_w1, packed_w2, topk_weight, topk_ids, inplace=True
            )

        atol = rtol = precision[ref.dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    @parametrize(M=[1, 39], N=[128], K=[256], E=[8], topk=[3])
    def test_int8_moe(self, M, N, K, E, topk):
        dtype = torch.bfloat16
        int8_factor_for_scale = 1e-2
        int8_max = 127
        int8_min = -128

        a = torch.randn((M, K), dtype=dtype) / math.sqrt(K)
        w1_fp32 = (torch.rand((E, 2 * N, K), dtype=torch.float32) - 0.5) * 2
        w1 = (w1_fp32 * int8_max).clamp(min=int8_min, max=int8_max).to(torch.int8)
        w2_fp32 = (torch.rand((E, K, N), dtype=torch.float32) - 0.5) * 2
        w2 = (w2_fp32 * int8_max).clamp(min=int8_min, max=int8_max).to(torch.int8)
        w1_s = torch.rand(E, 2 * N) * int8_factor_for_scale
        w2_s = torch.rand(E, K) * int8_factor_for_scale
        topk_weight, topk_ids = make_routing(M, E, topk, dtype)

        ref_out = torch_w8a8_per_column_fused_moe(
            a, w1, w2, w1_s, w2_s, topk_weight, topk_ids, topk
        )
        out = run_fused_experts(
            a,
            kernel.convert_weight_packed(w1),
            kernel.convert_weight_packed(w2),
            topk_weight,
            topk_ids,
            quant=CPUQuantMethod.INT8_W8A8,
            w1_scale=w1_s,
            w2_scale=w2_s,
            inplace=True,
        )

        atol = rtol = 0.02 if M > 35 else precision[ref_out.dtype]
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)

    @parametrize(
        M=[2, 32], N=[64, 128], K=[64, 128], E=[4], topk=[2], bias=[False, True]
    )
    def test_fp8_moe(self, M, N, K, E, topk, bias):
        dtype = torch.bfloat16
        a = torch.randn(M, K, dtype=dtype) / 10
        w1 = (torch.randn(E, 2 * N, K) * fp8_max).clamp(min=fp8_min, max=fp8_max).to(
            torch.float8_e4m3fn
        )
        w2 = (torch.randn(E, K, N) * fp8_max).clamp(min=fp8_min, max=fp8_max).to(
            torch.float8_e4m3fn
        )
        w1s = (
            torch.randn(E, math.ceil(2 * N / BLOCK_N), math.ceil(K / BLOCK_K))
            * factor_for_scale
        )
        w2s = (
            torch.randn(E, math.ceil(K / BLOCK_N), math.ceil(N / BLOCK_K))
            * factor_for_scale
        )
        w1_scaled = scaled_weight(w1, w1s).to(dtype)
        w2_scaled = scaled_weight(w2, w2s).to(dtype)
        topk_weight, topk_ids = make_routing(M, E, topk, dtype)

        packed_w1 = kernel.convert_weight_packed(w1)
        packed_w2 = kernel.convert_weight_packed(w2)
        common = dict(
            quant=CPUQuantMethod.FP8_W8A16,
            w1_scale=w1s,
            w2_scale=w2s,
            block_size=[BLOCK_N, BLOCK_K],
        )

        if bias:
            w1_b = torch.randn((E, 2 * N), dtype=torch.float32) / 10
            w2_b = torch.randn((E, K), dtype=torch.float32) / 10
            ref = torch_naive_fused_moe_gptoss(
                a,
                w1_scaled,
                w2_scaled,
                w1_b,
                w2_b,
                topk_weight,
                topk_ids,
                False,
                _SWIGLU_ALPHA,
                _SWIGLU_LIMIT,
                E,
            )
            out = run_fused_experts(
                a,
                packed_w1,
                packed_w2,
                topk_weight,
                topk_ids,
                w1_bias=w1_b,
                w2_bias=w2_b,
                alpha=_SWIGLU_ALPHA,
                limit=_SWIGLU_LIMIT,
                **common,
            )
        else:
            ref = native_fp8_fused_moe(
                a, w1_scaled.float(), w2_scaled.float(), topk_weight, topk_ids, topk
            )
            out = run_fused_experts(
                a, packed_w1, packed_w2, topk_weight, topk_ids, **common
            )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref.bfloat16(), out, atol=atol, rtol=rtol)

    @parametrize(
        M=[2, 32], N=[64, 128], K=[64, 128], E=[4], topk=[2], bias=[False, True]
    )
    def test_mxfp4_moe(self, M, N, K, E, topk, bias):
        dtype = torch.bfloat16
        a = torch.randn(M, K, dtype=dtype) / 10

        w1_bf16 = torch.randn((E, 2 * N, K), dtype=dtype) / 10
        w1q, w1s = MXFP4QuantizeUtil.quantize(w1_bf16)
        w1s = w1s.reshape(E, 2 * N, K // 32)
        w1dq = MXFP4QuantizeUtil.dequantize(w1q, dtype, w1s)

        w2_bf16 = torch.randn((E, K, N), dtype=dtype) / 10
        w2q, w2s = MXFP4QuantizeUtil.quantize(w2_bf16)
        w2s = w2s.reshape(E, K, N // 32)
        w2dq = MXFP4QuantizeUtil.dequantize(w2q, dtype, w2s)

        topk_weight, topk_ids = make_routing(M, E, topk, dtype)
        packed_w1 = kernel.convert_weight_packed(w1q)
        packed_w2 = kernel.convert_weight_packed(w2q)
        packed_w1s = kernel.convert_scale_packed(w1s)
        packed_w2s = kernel.convert_scale_packed(w2s)
        common = dict(
            quant=CPUQuantMethod.MXFP4,
            w1_scale=packed_w1s,
            w2_scale=packed_w2s,
        )

        if bias:
            w1_b = torch.randn((E, 2 * N), dtype=torch.float32) / 10
            w2_b = torch.randn((E, K), dtype=torch.float32) / 10
            ref = torch_naive_fused_moe_gptoss(
                a,
                w1dq,
                w2dq,
                w1_b,
                w2_b,
                topk_weight,
                topk_ids,
                False,
                _SWIGLU_ALPHA,
                _SWIGLU_LIMIT,
                E,
            )
            out = run_fused_experts(
                a,
                packed_w1,
                packed_w2,
                topk_weight,
                topk_ids,
                w1_bias=w1_b,
                w2_bias=w2_b,
                alpha=_SWIGLU_ALPHA,
                limit=_SWIGLU_LIMIT,
                **common,
            )
            torch.testing.assert_close(
                ref, out, atol=precision[ref.dtype], rtol=precision[ref.dtype]
            )
        else:
            ref = native_fp8_fused_moe(
                a, w1dq.float(), w2dq.float(), topk_weight, topk_ids, topk
            )
            out = run_fused_experts(
                a, packed_w1, packed_w2, topk_weight, topk_ids, **common
            )
            torch.testing.assert_close(
                ref.bfloat16(), out, atol=precision[dtype], rtol=precision[dtype]
            )

    @parametrize(M=[1, 6], N=[512], K=[256], E=[8], topk=[4])
    def test_int4_moe(self, M, N, K, E, topk, group_size=128):
        dtype = torch.bfloat16

        a = torch.rand(M, K, dtype=dtype) / math.sqrt(K)
        awq_w13_weight = torch.randint(-127, 128, (E, K, 2 * N // 8)).to(torch.int)
        awq_w13_zero = torch.randint(0, 10, (E, K // group_size, 2 * N // 8)).to(
            torch.int
        )
        awq_w13_scales = torch.rand(E, int(K // group_size), 2 * N).to(torch.bfloat16)
        awq_w2_weight = torch.randint(-127, 128, (E, N, K // 8)).to(torch.int)
        awq_w2_zero = torch.randint(0, 10, (E, N // group_size, K // 8)).to(torch.int)
        awq_w2_scales = torch.rand(E, int(N // group_size), K).to(torch.bfloat16)

        bf16_w13_weight = []
        bf16_w2_weight = []
        for i in range(E):
            bf16_w13_weight_i, _ = unpack_and_dequant_awq(
                awq_w13_weight[i], awq_w13_zero[i], awq_w13_scales[i], 4, 128
            )
            bf16_w2_weight_i, _ = unpack_and_dequant_awq(
                awq_w2_weight[i], awq_w2_zero[i], awq_w2_scales[i], 4, 128
            )
            bf16_w13_weight.append(bf16_w13_weight_i)
            bf16_w2_weight.append(bf16_w2_weight_i)
        bf16_w13_weight = torch.stack(bf16_w13_weight).detach()
        bf16_w2_weight = torch.stack(bf16_w2_weight).detach()

        score = torch.rand((M, E), dtype=dtype)
        ref_out = torch_naive_fused_moe(
            a, bf16_w13_weight, bf16_w2_weight, score, topk, False
        )
        topk_weight, topk_ids = make_routing(M, E, topk, dtype)
        awq_w13_weight_pack, awq_w13_zero_pack, awq_w13_scales_pack = (
            torch.ops.sgl_kernel.convert_weight_packed_scale_zp(
                awq_w13_weight, awq_w13_zero, awq_w13_scales, 0
            )
        )
        awq_w2_weight_pack, awq_w2_zero_pack, awq_w2_scales_pack = (
            torch.ops.sgl_kernel.convert_weight_packed_scale_zp(
                awq_w2_weight, awq_w2_zero, awq_w2_scales, 0
            )
        )

        out = run_fused_experts(
            a,
            awq_w13_weight_pack,
            awq_w2_weight_pack,
            topk_weight,
            topk_ids,
            quant=CPUQuantMethod.INT4_W4A8,
            w1_scale=awq_w13_scales_pack,
            w2_scale=awq_w2_scales_pack,
            w1_zp=awq_w13_zero_pack,
            w2_zp=awq_w2_zero_pack,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref_out.bfloat16(), out, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()

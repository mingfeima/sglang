import sgl_kernel
import torch


def fused_experts(
    x,
    w13_weight,
    w2_weight,
    topk_weights,
    topk_ids,
    inplace,
    use_int8_w8a8=False,
    w1_scale=None,
    w2_scale=None,
    a1_scale=None,
    a2_scale=None,
    is_vnni=True,
):
    return sgl_kernel.common_ops.fused_experts_cpu(
        x,
        w13_weight,
        w2_weight,
        topk_weights,
        topk_ids,
        inplace,
        use_int8_w8a8,
        w1_scale,
        w2_scale,
        a1_scale,
        a2_scale,
        is_vnni,
    )


def shared_expert(
    hidden_states,
    w1,
    w2,
    fused_experts_out,
    routed_scaling_factor,
    inplace,
    use_int8_w8a8=False,
    use_fp8_w8a16=False,
    w1_scale=None,
    w2_scale=None,
    block_size=None,
    a1_scale=None,
    a2_scale=None,
    is_vnni=True,
):
    return sgl_kernel.common_ops.shared_expert_cpu(
        hidden_states,
        w1,
        w2,
        fused_experts_out,
        routed_scaling_factor,
        inplace,
        use_int8_w8a8,
        use_fp8_w8a16,
        w1_scale,
        w2_scale,
        block_size,
        a1_scale,
        a2_scale,
        is_vnni,
    )


def convert_weight_packed(weight):
    return sgl_kernel.common_ops.convert_weight_packed(weight)


def qkv_proj_with_rope(
    hidden_states,
    q_a_proj_weight,
    q_b_proj_weight,
    kv_a_proj_weight,
    w_kc,
    q_a_layernorm_weight,
    kv_a_layernorm_weight,
    positions,
    cos_sin_cache,
    eps,
    use_int8_w8a8=False,
    q_a_proj_scale=None,
    q_b_proj_scale=None,
    kv_a_proj_scale=None,
    is_vnni=True,
):
    return sgl_kernel.common_ops.qkv_proj_with_rope(
        hidden_states,
        q_a_proj_weight,
        q_b_proj_weight,
        kv_a_proj_weight,
        w_kc,
        q_a_layernorm_weight,
        kv_a_layernorm_weight,
        positions,
        cos_sin_cache,
        eps,
        use_int8_w8a8,
        q_a_proj_scale,
        q_b_proj_scale,
        kv_a_proj_scale,
        is_vnni,
    )


def decode_attention(
    q,
    k_buffer,
    v_buffer,
    o,
    key,
    value,
    loc,
    kv_indptr,
    kv_indices,
    seq_lens,
    attn_logits,
    sm_scale,
    logit_cap=0.0,
):
    sgl_kernel.common_ops.decode_attention_cpu(
        q,
        k_buffer,
        v_buffer,
        o,
        key,
        value,
        loc,
        attn_logits,
        kv_indptr,
        kv_indices,
        seq_lens,
        sm_scale,
        logit_cap,
    )


def extend_attention(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    req_to_tokens,
    req_pool_indices,
    seq_lens,
    extend_seq_lens,
    extend_start_loc,
    max_len_extend,
    sm_scale,
    logit_cap=0.0,
):
    sgl_kernel.common_ops.extend_attention_cpu(
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        req_to_tokens,
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        extend_start_loc,
        max_len_extend,
        sm_scale,
        logit_cap,
    )


def weight_packed_linear(
    x,
    weight,
    bias,
    is_vnni=True,
):
    return sgl_kernel.common_ops.weight_packed_linear(
        x,
        weight,
        bias,
        is_vnni,
    )


def grouped_topk(
    hidden_states,
    router_logits,
    top_k,
    renormalize,
    num_expert_group,
    topk_group,
):
    return sgl_kernel.common_ops.grouped_topk_cpu(
        hidden_states,
        router_logits,
        top_k,
        renormalize,
        num_expert_group,
        topk_group,
    )


def biased_grouped_topk(
    hidden_states,
    router_logits,
    bias,
    top_k,
    renormalize,
    num_expert_group,
    topk_group,
):
    return sgl_kernel.common_ops.biased_grouped_topk_cpu(
        hidden_states,
        router_logits,
        bias,
        top_k,
        renormalize,
        num_expert_group,
        topk_group,
    )


def fused_add_rmsnorm(
    input,
    residual,
    weight,
    eps,
):
    sgl_kernel.common_ops.fused_add_rmsnorm_cpu(
        input,
        residual,
        weight,
        eps,
    )


def rmsnorm(
    input,
    weight,
    eps,
):
    return sgl_kernel.common_ops.rmsnorm_cpu(
        input,
        weight,
        eps,
    )


def int8_scaled_mm(
    mat1,
    mat2,
    scales1,
    scales2,
    bias,
    out_dtype,
    is_vnni=True,
):
    return sgl_kernel.common_ops.int8_scaled_mm_cpu(
        mat1, mat2, scales1, scales2, bias, out_dtype, is_vnni
    )


def int8_scaled_mm_with_quant(
    mat1,
    mat2,
    scales2,
    bias,
    out_dtype,
    is_vnni=True,
):
    return sgl_kernel.common_ops.int8_scaled_mm_with_quant(
        mat1, mat2, scales2, bias, out_dtype, is_vnni
    )


def per_token_quant_int8(x):
    return sgl_kernel.common_ops.per_token_quant_int8_cpu(x)


def fp8_scaled_mm(
    mat1,
    mat2,
    scales2,
    block_size,
    bias,
    out_dtype,
    is_vnni=True,
):
    return sgl_kernel.common_ops.fp8_scaled_mm_cpu(
        mat1, mat2, scales2, block_size, bias, out_dtype, is_vnni
    )


def rotary_position_embedding(
    t_pos,
    q_pe,
    k_pe,
    t_emb_pos,
):
    return sgl_kernel.common_ops.rotary_position_embedding_cpu(
        t_pos,
        q_pe,
        k_pe,
        t_emb_pos,
    )


def silu_and_mul(
    input,
):
    return sgl_kernel.common_ops.silu_and_mul_cpu(input)


def bmm(out, mat1, mat2, is_vnni=True, scale=None):
    return sgl_kernel.common_ops.bmm_cpu(out, mat1, mat2, is_vnni, scale)

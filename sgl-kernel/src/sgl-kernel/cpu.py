import sgl_kernel
import torch


def fused_experts(
    x,
    w13_weight,
    w2_weight,
    topk_weights,
    topk_ids,
    inplace,
    is_vnni=True,
):
    return sgl_kernel.ops._kernels.fused_experts_cpu(
        x,
        w13_weight,
        w2_weight,
        topk_weights,
        topk_ids,
        inplace,
        is_vnni,
    )


def convert_weight_packed(weight):
    return sgl_kernel.ops._kernels.convert_weight_packed(weight)


def decode_attention(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    seq_lens,
    attn_logits,
    sm_scale,
    logit_cap=0.0,
):
    sgl_kernel.ops._kernels.decode_attention_cpu(
        q,
        o,
        k_buffer,
        v_buffer,
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
    sgl_kernel.ops._kernels.extend_attention_cpu(
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

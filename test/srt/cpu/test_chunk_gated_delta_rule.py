import torch
from sglang.srt.layers.attention.torch_native_hybrid_linear_attn_backend import torch_chunk_gated_delta_rule
# import time

def ref_kernel(
    query, # [B, T, HK, K]
    key, # [B, T, HK, K]
    value, # [B, T, HV, V]
    g, # [B, T, HV]
    beta, # [B, T, HV]
    cu_seqlens, # [N+1]
    initial_state, # [N, HV, K, V]
    output_final_state, # True
    use_qk_l2norm_in_kernel, # True
):
    num_heads = query.shape[2]
    num_value_heads = value.shape[2]
    batch_size = initial_state.shape[0]
    if num_value_heads // num_heads > 1:
        query = query.repeat_interleave(num_value_heads // num_heads, dim=2)
        key = key.repeat_interleave(num_value_heads // num_heads, dim=2)
    output = torch.empty_like(value)
    final_state = torch.empty_like(initial_state)
    start_q = 0
    for i in range(batch_size):
        end_q = cu_seqlens[i + 1]
        core_attn_outi, last_recurrent_state = torch_chunk_gated_delta_rule(
            query=query[:, start_q:end_q, :, :],
            key=key[:, start_q:end_q, :, :],
            value=value[:, start_q:end_q, :, :],
            g=g[:, start_q:end_q, :],
            beta=beta[:, start_q:end_q, :],
            initial_state=initial_state[i],
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        output[:, start_q:end_q, :, :] = core_attn_outi
        final_state[i] = last_recurrent_state
        start_q = end_q
    return output, final_state


def test_chunk_gated_delta_rule():
    torch.set_printoptions(threshold=10000)
    torch.manual_seed(100)
    query = torch.randn(4194304, dtype=torch.bfloat16).as_strided(size=(1, 512, 16, 128), stride=(4194304, 8192, 128, 1)) * 0.1
    key = torch.randn(4194304, dtype=torch.bfloat16).as_strided(size=(1, 512, 16, 128), stride=(4194304, 8192, 128, 1)) * 0.1
    value = torch.randn(4194304, dtype=torch.bfloat16).as_strided(size=(1, 512, 32, 128), stride=(4194304, 8192, 128, 1)) * 0.1
    g = torch.randn((1, 512, 32), dtype=torch.float32).as_strided(size=(1, 512, 32), stride=(16384, 32, 1)) * 0.1
    beta = torch.randn((1, 512, 32), dtype=torch.bfloat16).as_strided(size=(1, 512, 32), stride=(16384, 32, 1)) * 0.1
    cu_seqlens = torch.tensor([0, 128, 256, 384, 512], dtype=torch.int32)
    initial_state = torch.randn((4, 32, 128, 128), dtype=torch.float32).as_strided(size=(4, 32, 128, 128), stride=(524288, 16384, 128, 1)) * 0.1

    ### ACCURACY
    output_ref, final_state_ref = ref_kernel(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    output_act, final_state_act = torch.ops.sgl_kernel.chunk_gated_delta_rule_cpu(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    print(torch.allclose(output_ref, output_act, rtol=1e-02, atol=1e-02), torch.max(torch.abs(output_ref - output_act)))
    print(torch.allclose(final_state_ref, final_state_act, rtol=1e-02, atol=1e-02), torch.max(torch.abs(final_state_ref - final_state_act)))

    # ### PERF
    # iter_num = 100
    # ref_time_start = time.time()
    # for i in range(iter_num):
    #     output_ref, final_state_ref = ref_kernel(
    #         query=query,
    #         key=key,
    #         value=value,
    #         g=g,
    #         beta=beta,
    #         cu_seqlens=cu_seqlens,
    #         initial_state=initial_state,
    #         output_final_state=True,
    #         use_qk_l2norm_in_kernel=True,
    #     )
    # ref_time = (time.time() - ref_time_start) / iter_num
    
    # act_time_start = time.time()
    # for i in range(iter_num):
    #     output_act, final_state_act = torch.ops.sgl_kernel.chunk_gated_delta_rule_cpu(
    #         query=query,
    #         key=key,
    #         value=value,
    #         g=g,
    #         beta=beta,
    #         cu_seqlens=cu_seqlens,
    #         initial_state=initial_state,
    #         output_final_state=True,
    #         use_qk_l2norm_in_kernel=True,
    #     )
    # act_time = (time.time() - act_time_start) / iter_num
    # print("ref_time: ", ref_time)
    # print("act_time: ", act_time)

test_chunk_gated_delta_rule()
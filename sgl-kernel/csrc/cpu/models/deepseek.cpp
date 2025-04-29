#include "../common.h"
#include "../vec.h"
#include <pybind11/pybind11.h>
#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

extern void decode_attention_cpu(at::Tensor& query, at::Tensor& k_cache, at::Tensor& v_cache, at::Tensor& output,
    at::Tensor& key, at::Tensor& value, at::Tensor& loc, at::Tensor& attn_logits,
    at::Tensor& req_to_token, at::Tensor& req_pool_indices, at::Tensor& seq_lens,
    double sm_scale, double logit_cap);

extern void bmm_cpu(at::Tensor& out, at::Tensor& mat1, at::Tensor& mat2, bool is_vnni,
    std::optional<at::Tensor>& scale);

extern std::tuple<at::Tensor, at::Tensor, at::Tensor> qkv_proj_with_rope(
    at::Tensor& hidden_states,
    at::Tensor& q_a_proj_weight,
    at::Tensor& q_b_proj_weight,
    at::Tensor& kv_a_proj_weight,
    at::Tensor& w_kc,
    at::Tensor& q_a_layernorm_weight,
    at::Tensor& kv_a_layernorm_weight,
    at::Tensor& positions,
    at::Tensor& cos_sin_cache,
    double eps,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    std::optional<at::Tensor>& q_a_proj_scale,
    std::optional<at::Tensor>& q_b_proj_scale,
    std::optional<at::Tensor>& kv_a_proj_scale,
    bool is_vnni,
    std::optional<std::vector<int64_t>> block_size);

extern at::Tensor weight_packed_linear(at::Tensor& mat1, at::Tensor& mat2,
    std::optional<at::Tensor>& bias, bool is_vnni);

extern at::Tensor int8_scaled_mm_with_quant(at::Tensor& mat1, at::Tensor& mat2, at::Tensor& scales2,
    std::optional<at::Tensor>& bias, at::ScalarType out_dtype, bool is_vnni);

extern at::Tensor fp8_scaled_mm_cpu(at::Tensor& mat1, at::Tensor& mat2, at::Tensor& scales2,
    std::vector<int64_t> block_size, std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype, bool is_vnni);

extern void shm_allreduce(at::Tensor& data, c10::intrusive_ptr<c10d::ProcessGroup> process_group, py::object op);

extern std::tuple<at::Tensor, at::Tensor> grouped_topk_cpu(
    at::Tensor& hidden_states,
    at::Tensor& gating_output,
    int64_t topk,
    bool renormalize,
    int64_t num_expert_group,
    int64_t topk_group);

extern std::tuple<at::Tensor, at::Tensor> biased_grouped_topk_cpu(
    at::Tensor& hidden_states,
    at::Tensor& gating_output,
    at::Tensor& correction_bias,
    int64_t topk,
    bool renormalize,
    int64_t num_expert_group,
    int64_t topk_group);

extern at::Tensor fused_experts_cpu(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& topk_weights,
    at::Tensor& topk_ids,
    bool inplace,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    std::optional<at::Tensor>& w1_scale,
    std::optional<at::Tensor>& w2_scale,
    std::optional<std::vector<int64_t>> block_size,
    std::optional<at::Tensor>& a1_scale,
    std::optional<at::Tensor>& a2_scale,
    bool is_vnni);

extern at::Tensor shared_expert_cpu(
    at::Tensor& hidden_states,
    at::Tensor& w1,
    at::Tensor& w2,
    at::Tensor& fused_experts_out,
    double routed_scaling_factor,
    bool inplace,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    std::optional<at::Tensor>& w1_scale,
    std::optional<at::Tensor>& w2_scale,
    std::optional<std::vector<int64_t>> block_size,
    std::optional<at::Tensor>& a1_scale,
    std::optional<at::Tensor>& a2_scale,
    bool is_vnni);

// This function implements the forward function of sglang/python/sglang/srt/layers/linear.py:RowParallelLinear
at::Tensor row_parallel_linear_forward(
    at::Tensor& mat1, at::Tensor& mat2,
    std::optional<at::Tensor>& bias,
    int tp_size,
    int tp_rank,
    std::optional<c10::intrusive_ptr<c10d::ProcessGroup>> process_group,
    std::optional<py::object> op,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    at::ScalarType out_dtype,
    std::optional<at::Tensor>& scales2,
    std::optional<std::vector<int64_t>> block_size,
    bool is_vnni) {
  RECORD_FUNCTION("sgl-kernel::row_parallel_linear_forward", std::vector<c10::IValue>({mat1, mat2}));
  // # Only fuse bias add into GEMM for rank 0 (this ensures that
  // # bias will not get added more than once in TP>1 case)
  const bool has_bias = bias.has_value();
  const float* bias_data = nullptr;
  if (tp_rank == 0 && has_bias) {
    bias_data = bias.value().data_ptr<float>();
  }

  at::Tensor output_parallel;
  if (use_int8_w8a8) {
    TORCH_CHECK(scales2.has_value(), "missing scales2 for int8 w8a8 row_parallel_linear_forward");
    output_parallel = int8_scaled_mm_with_quant(mat1, mat2, scales2.value(), bias, out_dtype, is_vnni);
  } else if (use_fp8_w8a16) {
    TORCH_CHECK(scales2.has_value(), "missing scales2 for fp8 w8a16 row_parallel_linear_forward");
    TORCH_CHECK(block_size.has_value(), "missing block_size for fp8 w8a16 row_parallel_linear_forward");
    output_parallel = fp8_scaled_mm_cpu(mat1, mat2, scales2.value(), block_size.value(), bias, out_dtype, is_vnni);
  } else {
    output_parallel = weight_packed_linear(mat1, mat2, bias, is_vnni);
  }

  if (tp_size > 1) {
    TORCH_CHECK(process_group.has_value(), "missing process_group for tp_size > 1 row_parallel_linear_forward");
    TORCH_CHECK(op.has_value(), "missing reduce op for tp_size > 1 row_parallel_linear_forward");
    shm_allreduce(output_parallel, process_group.value(), op.value());
  }

  return output_parallel;
}

at::Tensor forward_absorb_decode_fused_cpu(
    at::Tensor& hidden_states, // qkv_proj_with_rope
    at::Tensor& q_a_proj_weight, // qkv_proj_with_rope
    at::Tensor& q_b_proj_weight, // qkv_proj_with_rope
    at::Tensor& kv_a_proj_weight, // qkv_proj_with_rope
    at::Tensor& w_kc, // qkv_proj_with_rope
    at::Tensor& q_a_layernorm_weight, // qkv_proj_with_rope
    at::Tensor& kv_a_layernorm_weight, // qkv_proj_with_rope
    at::Tensor& positions, // qkv_proj_with_rope
    at::Tensor& cos_sin_cache, // qkv_proj_with_rope
    at::Tensor& k_cache, // decode_attention_cpu
    at::Tensor& v_cache, // decode_attention_cpu
    at::Tensor& loc, // decode_attention_cpu
    at::Tensor& attn_logits, // decode_attention_cpu
    at::Tensor& req_to_token, // decode_attention_cpu
    at::Tensor& req_pool_indices, // decode_attention_cpu
    at::Tensor& seq_lens, // decode_attention_cpu
    at::Tensor& w_vc, // bmm
    at::Tensor& o_proj_weight, // o_proj
    std::optional<at::Tensor>& o_proj_bias, // o_proj
    double eps, // qkv_proj_with_rope
    bool use_int8_w8a8, // qkv_proj_with_rope
    bool use_fp8_w8a16, // qkv_proj_with_rope
    double sm_scale, // decode_attention_cpu
    double logit_cap, // decode_attention_cpu
    int tp_k_head_num, // decode_attention_cpu
    int qk_head_dim, // decode_attention_cpu
    int tp_v_head_num, // decode_attention_cpu
    int v_head_dim, // decode_attention_cpu
    int tp_q_head_num, // decode_attention_cpu
    int num_local_heads, // decode_attention_cpu
    int kv_lora_rank, // decode_attention_cpu
    int tp_size, // o_proj
    int tp_rank, // o_proj
    bool o_proj_use_int8_w8a8, // o_proj
    bool o_proj_use_fp8_w8a16, // o_proj
    at::ScalarType o_proj_out_dtype, // o_proj
    std::optional<at::Tensor>& q_a_proj_scale, // qkv_proj_with_rope
    std::optional<at::Tensor>& q_b_proj_scale, // qkv_proj_with_rope
    std::optional<at::Tensor>& kv_a_proj_scale, // qkv_proj_with_rope
    std::optional<std::vector<int64_t>> block_size, // qkv_proj_with_rope
    std::optional<at::Tensor>& bmm_scale, // bmm
    std::optional<c10::intrusive_ptr<c10d::ProcessGroup>> process_group, // o_proj
    std::optional<py::object> op, // o_proj
    std::optional<at::Tensor>& o_proj_scale, // o_proj
    std::optional<std::vector<int64_t>> o_proj_block_size, // o_proj
    bool is_vnni  // qkv_proj_with_rope, bmm, o_proj
) {
  RECORD_FUNCTION("sgl-kernel::forward_absorb_decode_fused_cpu", std::vector<c10::IValue>({
    hidden_states, q_a_proj_weight, q_b_proj_weight, kv_a_proj_weight, w_kc,
    q_a_layernorm_weight, kv_a_layernorm_weight, positions, cos_sin_cache,
    k_cache, v_cache, loc, attn_logits, req_to_token, req_pool_indices,
    seq_lens, w_vc, o_proj_weight}));

  // stage 1: q_input, k_input, v_input = qkv_proj_with_rope(...)
  at::Tensor query, key, value;
  std::tie(query, key, value) = qkv_proj_with_rope(
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
    use_fp8_w8a16,
    q_a_proj_scale,
    q_b_proj_scale,
    kv_a_proj_scale,
    is_vnni,
    block_size);

  // stage 2:
  // attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
  // attn_output = attn_output.view(-1, params.num_local_heads, params.kv_lora_rank)

  // stage 2.1:
  // sglang/python/sglang/srt/layers/radix_attention.py: RadixAttention: forward
  // For DeepSeek R1, key and value returned from qkv_proj_with_rope is 3D, thus the below code is not needed.
  //   if k is not None:
  //     // For cross-layer sharing, kv can be None
  //     assert v is not None
  //     k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
  //     v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
  CHECK_DIM(3, key);
  CHECK_DIM(3, value);

  // stage 2.2:
  // sglang/python/sglang/srt/layers/attention/intel_amx_backend.py: IntelAMXAttnBackend: forward_decode
  // q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

  // if layer.qk_head_dim != layer.v_head_dim:
  //     o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
  // else:
  //     o = torch.empty_like(q)
  // self.decode_attention_fwd(...)
  query= query.reshape({-1, tp_q_head_num * qk_head_dim});
  at::Tensor attn_output;
  if (qk_head_dim != v_head_dim) {
      attn_output = at::empty({query.size(0), tp_q_head_num * v_head_dim}, query.options());
  } else {
      attn_output = at::empty_like(query);
  }
  auto query_3d = query.view({-1, tp_q_head_num, qk_head_dim});
  auto o_3d = attn_output.view({-1, tp_q_head_num, v_head_dim});
  decode_attention_cpu(
    query_3d,
    k_cache,
    v_cache,
    o_3d,
    key,
    value,
    loc,
    attn_logits,
    req_to_token,
    req_pool_indices,
    seq_lens,
    sm_scale,
    logit_cap);

  // stage 2.3: attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)
  attn_output = attn_output.view({-1, num_local_heads, kv_lora_rank});

  // stage 3: bmm
  int64_t B = w_vc.sizes()[0];
  int64_t N = w_vc.sizes()[1];
  int64_t M = attn_output.sizes()[0];

  at::Tensor output = at::empty({M, B * N}, attn_output.options());
  at::Tensor attn_bmm_output = output.view({M, B, N}).transpose_(0, 1);

  attn_output = attn_output.transpose(0, 1);
  bmm_cpu(attn_bmm_output, attn_output, w_vc, is_vnni, bmm_scale);

  // stage 4: o_proj
  return row_parallel_linear_forward(
    output,
    o_proj_weight,
    o_proj_bias,
    tp_size,
    tp_rank,
    process_group,
    op,
    o_proj_use_int8_w8a8,
    o_proj_use_fp8_w8a16,
    o_proj_out_dtype,
    o_proj_scale,
    o_proj_block_size,
    is_vnni);
}

// This is the fused forward function of sglang/python/sglang/srt/models/deepseek_v2.py:DeepseekV2MoE
at::Tensor forward_moe_fused_cpu(
    at::Tensor& hidden_states, // MoEGate
    at::Tensor& MoEGate_weight, // MoEGate
    std::optional<at::Tensor>& bias, // MoEGate
    at::Tensor& fused_experts_w13_weight, // experts
    at::Tensor& fused_experts_w2_weight, // experts
    at::Tensor& shared_expert_w1, // shared_expert
    at::Tensor& shared_expert_w2, // shared_expert
    int top_k, // select_experts
    bool use_grouped_topk, // select_experts
    bool renormalize, // select_experts
    bool fused_experts_use_int8_w8a8, // experts
    bool fused_experts_use_fp8_w8a16, // experts
    bool fused_experts_inplace, // experts
    double routed_scaling_factor, // shared_expert
    bool shared_expert_inplace, // shared_expert
    bool shared_expert_use_int8_w8a8, // shared_expert
    bool shared_expert_use_fp8_w8a16, // shared_expert
    int tp_size, // all_reduce
    std::optional<int> topk_group, // select_experts
    std::optional<int> num_expert_group, // select_experts
    std::optional<at::Tensor>& correction_bias, // select_experts
    std::optional<at::Tensor>& fused_experts_w1_scale, // experts
    std::optional<at::Tensor>& fused_experts_w2_scale, // experts
    std::optional<at::Tensor>& fused_experts_a1_scale, // experts
    std::optional<at::Tensor>& fused_experts_a2_scale, // experts
    std::optional<std::vector<int64_t>> fused_experts_block_size, // experts
    std::optional<at::Tensor>& shared_expert_w1_scale, // shared_expert
    std::optional<at::Tensor>& shared_expert_w2_scale, // shared_expert
    std::optional<std::vector<int64_t>> shared_expert_block_size, // shared_expert
    std::optional<at::Tensor>& shared_expert_a1_scale, // shared_expert
    std::optional<at::Tensor>& shared_expert_a2_scale,     // shared_expert
    std::optional<c10::intrusive_ptr<c10d::ProcessGroup>> process_group, // all_reduce
    std::optional<py::object> op, // all_reduce
    bool is_vnni // MoEGate, experts, shared_expert
) {
  RECORD_FUNCTION("sgl-kernel::forward_moe_fused_cpu", std::vector<c10::IValue>({
    hidden_states, MoEGate_weight, bias, fused_experts_w13_weight, fused_experts_w2_weight, shared_expert_w1, shared_expert_w2}));

  // stage 1:
  // num_tokens, hidden_dim = hidden_states.shape
  // hidden_states = hidden_states.view(-1, hidden_dim)
  auto sizes = hidden_states.sizes();
  int64_t num_tokens = sizes[0];
  int64_t hidden_dim = sizes[1];
  hidden_states = hidden_states.view({-1, hidden_dim});

  // stage 2:
  // router_logits = self.gate_impl(hidden_states)
  auto router_logits = weight_packed_linear(hidden_states, MoEGate_weight, bias, is_vnni);

  // stage 3:
  // fused_experts_out = self.experts_impl(
  //     hidden_states=hidden_states, router_logits=router_logits
  // )

  // stage 3.1:
  // topk_weights, topk_ids = select_experts(
  //     hidden_states=x,
  //     router_logits=router_logits,
  //     use_grouped_topk=use_grouped_topk,
  //     top_k=top_k,
  //     renormalize=renormalize,
  //     topk_group=topk_group,
  //     num_expert_group=num_expert_group,
  //     custom_routing_function=custom_routing_function,
  //     correction_bias=correction_bias,
  // )
  TORCH_CHECK(use_grouped_topk, "forward_moe_fused_cpu: expect use_grouped_topk to be true");
  TORCH_CHECK(topk_group.has_value(), "forward_moe_fused_cpu: missing topk_group");
  TORCH_CHECK(num_expert_group.has_value(), "forward_moe_fused_cpu: missing num_expert_group");

  at::Tensor topk_weights, topk_ids;
  if (!correction_bias.has_value()) {
    std::tie(topk_weights, topk_ids) = grouped_topk_cpu(
      hidden_states,
      router_logits,
      top_k,
      renormalize,
      num_expert_group.value(),
      topk_group.value());
  } else {
    std::tie(topk_weights, topk_ids) = biased_grouped_topk_cpu(
      hidden_states,
      router_logits,
      correction_bias.value(),
      top_k,
      renormalize,
      num_expert_group.value(),
      topk_group.value());
  }

  // stage 3.2: fused_experts
  auto fused_experts_out = fused_experts_cpu(
    hidden_states,
    fused_experts_w13_weight,
    fused_experts_w2_weight,
    topk_weights,
    topk_ids,
    fused_experts_inplace,
    fused_experts_use_int8_w8a8,
    fused_experts_use_fp8_w8a16,
    fused_experts_w1_scale,
    fused_experts_w2_scale,
    fused_experts_block_size,
    fused_experts_a1_scale,
    fused_experts_a2_scale,
    is_vnni);

  // stage 4: shared_expert
  auto final_hidden_states = shared_expert_cpu(
    hidden_states,
    shared_expert_w1,
    shared_expert_w2,
    fused_experts_out,
    routed_scaling_factor,
    shared_expert_inplace,
    shared_expert_use_int8_w8a8,
    shared_expert_use_fp8_w8a16,
    shared_expert_w1_scale,
    shared_expert_w2_scale,
    shared_expert_block_size,
    shared_expert_a1_scale,
    shared_expert_a2_scale,
    is_vnni);

  // stage 5:
  // if self.tp_size > 1:
  //     final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
  if (tp_size > 1) {
    TORCH_CHECK(process_group.has_value(), "missing process_group for tp_size > 1 row_parallel_linear_forward");
    TORCH_CHECK(op.has_value(), "missing reduce op for tp_size > 1 row_parallel_linear_forward");
    shm_allreduce(final_hidden_states, process_group.value(), op.value());
  }

  // stage 6:
  // return final_hidden_states.view(num_tokens, hidden_dim)
  final_hidden_states = final_hidden_states.view({num_tokens, hidden_dim});

  return final_hidden_states;
}

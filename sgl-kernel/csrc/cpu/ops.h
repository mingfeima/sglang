#pragma once

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

void decode_attention_cpu(at::Tensor& query, at::Tensor& k_cache, at::Tensor& v_cache, at::Tensor& output,
    at::Tensor& key, at::Tensor& value, at::Tensor& loc, at::Tensor& attn_logits,
    at::Tensor& req_to_token, at::Tensor& req_pool_indices, at::Tensor& seq_lens,
    double sm_scale, double logit_cap);

void bmm_cpu(at::Tensor& out, at::Tensor& mat1, at::Tensor& mat2, bool is_vnni,
    std::optional<at::Tensor>& scale);

std::tuple<at::Tensor, at::Tensor, at::Tensor> qkv_proj_with_rope(
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
    std::optional<at::Tensor>& q_a_proj_scale,
    std::optional<at::Tensor>& q_b_proj_scale,
    std::optional<at::Tensor>& kv_a_proj_scale,
    bool is_vnni);

at::Tensor weight_packed_linear(at::Tensor& mat1, at::Tensor& mat2,
    std::optional<at::Tensor>& bias, bool is_vnni);

at::Tensor int8_scaled_mm_with_quant(at::Tensor& mat1, at::Tensor& mat2, at::Tensor& scales2,
    std::optional<at::Tensor>& bias, at::ScalarType out_dtype, bool is_vnni);

at::Tensor fp8_scaled_mm_cpu(at::Tensor& mat1, at::Tensor& mat2, at::Tensor& scales2,
    std::vector<int64_t> block_size, std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype, bool is_vnni);

void shm_allreduce(at::Tensor& data, c10::intrusive_ptr<c10d::ProcessGroup> process_group, py::object op);

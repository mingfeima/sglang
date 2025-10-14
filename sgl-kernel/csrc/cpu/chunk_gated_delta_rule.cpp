#include "common.h"
#include "vec.h"
#include <ATen/native/CPUBlas.h>
#include <ATen/native/cpu/utils.h>

namespace {

template <typename scalar_t>
void chunk_gated_delta_rule_kernel_impl(
        at::Tensor& output, // [B, T, HV, EV]
        at::Tensor& final_state, // [N, HV, EK, EV]
        at::Tensor& query, // [B, T, HK, EK]
        at::Tensor& key, // [B, T, HK, EK]
        at::Tensor& value, // [B, T, HV, EV]
        at::Tensor& g, // [B, T, HV] FP32
        at::Tensor& beta, // [B, T, HV]
        at::Tensor& cu_seqlens, // [N + 1] INT32
        int64_t chunk_size=64) {
    // query: [B, T, HK, EK] -> [B, HK, T, EK]
    // key: [B, T, HK, EK] -> [B, HK, T, EK]
    // value: [B, T, HV, EV] -> [B, HV, T, EV]
    // g: [B, T, HV] -> [B, HV, T]
    // beta: [B, T, HV] -> [B, HV, T]
    query = query.transpose(1, 2);
    key = key.transpose(1, 2);
    value = value.transpose(1, 2);
    g = g.transpose(1, 2).contiguous();
    beta = beta.transpose(1, 2).contiguous();

    // Sizes
    TORCH_CHECK(query.size(0) == 1);
    int64_t batch_size = final_state.size(0);
    int64_t global_seq_len = query.size(2);
    int64_t qk_num_head = query.size(1);
    int64_t v_num_head = value.size(1);
    int64_t qk_head_size = query.size(3);
    int64_t v_head_size = value.size(3);
    int64_t head_group = v_num_head / qk_num_head;
    int64_t global_pad_size = (chunk_size - global_seq_len % chunk_size) % chunk_size;
    int64_t global_total_seq_length = global_seq_len + global_pad_size;
    float scale = 1.0 / std::sqrt(qk_head_size);

    // Strides
    int64_t oStrideT = output.stride(1);
    int64_t oStrideH = output.stride(2);
    int64_t qStrideH = query.stride(1);
    int64_t qStrideT = query.stride(2);
    int64_t kStrideH = key.stride(1);
    int64_t kStrideT = key.stride(2);
    int64_t vStrideH = value.stride(1);
    int64_t vStrideT = value.stride(2);
    int64_t gStrideH = g.stride(1);
    int64_t bStrideH = beta.stride(1);
    int64_t final_state_StrideN = final_state.stride(0);
    int64_t final_state_StrideH = final_state.stride(1);
    int64_t final_state_StrideE = final_state.stride(2);

    // Data pointers
    const scalar_t* q_orig = query.const_data_ptr<scalar_t>();
    const scalar_t* k_orig = key.const_data_ptr<scalar_t>();
    const scalar_t* v_orig = value.const_data_ptr<scalar_t>();
    const float* g_orig = g.const_data_ptr<float>();
    const scalar_t* b_orig = beta.const_data_ptr<scalar_t>();
    const int32_t* cu_seqlens_ptr = cu_seqlens.const_data_ptr<int32_t>();
    scalar_t* out = output.data_ptr<scalar_t>();
    float* final_state_data = final_state.data_ptr<float>();

    // Allocate buffer
    at::Tensor q_pad_data = at::zeros({qk_num_head, global_total_seq_length, qk_head_size}, query.options().dtype(at::kFloat));
    at::Tensor k_pad_data = at::zeros({qk_num_head, global_total_seq_length, qk_head_size}, query.options().dtype(at::kFloat));
    at::Tensor v_pad_data = at::zeros({v_num_head, global_total_seq_length, v_head_size}, query.options().dtype(at::kFloat));
    at::Tensor g_pad_data = at::zeros({v_num_head, global_total_seq_length}, query.options().dtype(at::kFloat));
    at::Tensor k_beta_data = at::zeros({v_num_head, global_total_seq_length, qk_head_size}, query.options().dtype(at::kFloat));
    at::Tensor v_beta_data = at::zeros({v_num_head, global_total_seq_length, v_head_size}, query.options().dtype(at::kFloat));
    at::Tensor core_attn = at::zeros({batch_size, v_num_head, global_total_seq_length, v_head_size}, query.options().dtype(at::kFloat));
    float* q_pad = q_pad_data.data_ptr<float>();
    float* k_pad = k_pad_data.data_ptr<float>();
    float* v_pad = v_pad_data.data_ptr<float>();
    float* g_pad = g_pad_data.data_ptr<float>();
    float* k_beta = k_beta_data.data_ptr<float>();
    float* v_beta = v_beta_data.data_ptr<float>();
    float* core_attn_out = core_attn.data_ptr<float>();

    // Upper triangular mask
    // mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    auto triu_mask_0 = at::triu(
        at::ones({chunk_size, chunk_size}, query.options().dtype(at::kBool)),
        /*diagonal=*/0
    );
    // mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    auto triu_mask_1 = at::triu(
        at::ones({chunk_size, chunk_size}, query.options().dtype(at::kBool)),
        /*diagonal=*/1
    );

    int64_t start_q = 0;
    int64_t end_q = 0;
    for (int64_t n = 0; n < batch_size; n++) {
        end_q = cu_seqlens_ptr[n + 1];
        auto q_orig_ptr = q_orig + start_q * qStrideT;
        auto k_orig_ptr = k_orig + start_q * kStrideT;
        auto v_orig_ptr = v_orig + start_q * vStrideT;
        auto g_orig_ptr = g_orig + start_q;
        auto b_orig_ptr = b_orig + start_q;
        auto core_attn_out_ptr = core_attn_out + start_q * v_head_size;
        auto out_ptr = out + start_q * oStrideT;
        auto final_state_ptr = final_state_data + n * final_state_StrideN;
        auto q_pad_ptr = q_pad + start_q * qk_head_size;
        auto k_pad_ptr = k_pad + start_q * qk_head_size;
        auto v_pad_ptr = v_pad + start_q * v_head_size;
        auto g_pad_ptr = g_pad + start_q;
        auto k_beta_ptr = k_beta + start_q * qk_head_size;
        auto v_beta_ptr = v_beta + start_q * v_head_size;
        int64_t seq_len = end_q - start_q;
        int64_t pad_size = (chunk_size - seq_len % chunk_size) % chunk_size;
        int64_t total_seq_length = seq_len + pad_size;
        int64_t num_chunk = total_seq_length / chunk_size;

        // Padding for q/k/v/beta
        // query = query * scale
        // k_beta = key * beta.unsqueeze(-1)
        // v_beta = value * beta.unsqueeze(-1)
        // TODO: change parallel from HV to HK, and remove `if` branches
        at::parallel_for(0, v_num_head * total_seq_length, 1, [&](int64_t begin, int64_t end) {
            int ompIdx = at::get_thread_num();
            int64_t h = 0, l = 0;
            at::native::data_index_init(begin, h, v_num_head, l, total_seq_length);
            for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
                int64_t h_qk = h / head_group;
                int64_t h_res = h % head_group;
                auto curr_q_orig = q_orig_ptr + h_qk * qStrideH + l * qStrideT;
                auto curr_k_orig = k_orig_ptr + h_qk * kStrideH + l * kStrideT;
                auto curr_v_orig = v_orig_ptr + h * vStrideH + l * vStrideT;
                auto curr_b_orig = b_orig_ptr + h * bStrideH;
                float b_orig_val = l < seq_len ? static_cast<float>(*(curr_b_orig + l)) : 0.0;
                auto curr_q_pad = q_pad_ptr + h_qk * global_total_seq_length * qk_head_size + l * qk_head_size;
                auto curr_k_pad = k_pad_ptr + h_qk * global_total_seq_length * qk_head_size + l * qk_head_size;
                auto curr_v_pad = v_pad_ptr + h * global_total_seq_length * v_head_size + l * v_head_size;
                auto curr_k_beta = k_beta_ptr + h * global_total_seq_length * qk_head_size + l * qk_head_size;
                auto curr_v_beta = v_beta_ptr + h * global_total_seq_length * v_head_size + l * v_head_size;

                auto vec_size = at::vec::Vectorized<float>::size();
                // query = query * scale
                // k_beta = key * beta.unsqueeze(-1)
                int64_t i = 0;
                auto vec_scale = at::vec::Vectorized<float>(scale);
                auto vec_b = at::vec::Vectorized<float>(b_orig_val);
                for (; i < vec_size * (qk_head_size / vec_size); i += vec_size) {
                    auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(curr_q_orig + i);
                    auto tmp1 = at::vec::convert<float>(tmp0);
                    auto tmp2 = tmp1 * vec_scale;
                    if (h_res == 0) {
                        tmp2.store(curr_q_pad + i);
                    }
                    auto tmp3 = at::vec::Vectorized<scalar_t>::loadu(curr_k_orig + i);
                    auto tmp4 = at::vec::convert<float>(tmp3);
                    if (h_res == 0) {
                        tmp4.store(curr_k_pad + i);
                    }
                    auto tmp5 = tmp4 * vec_b;
                    tmp5.store(curr_k_beta + i);
                }
                if (i < qk_head_size) {
                    auto tmp0 = at::vec::Vectorized<scalar_t>::loadu(curr_q_orig + i, qk_head_size - i);
                    auto tmp1 = at::vec::convert<float>(tmp0);
                    auto tmp2 = tmp1 * vec_scale;
                    if (h_res == 0) {
                        tmp2.store(curr_q_pad + i, qk_head_size - i);
                    }
                    auto tmp3 = at::vec::Vectorized<scalar_t>::loadu(curr_k_orig + i, qk_head_size - i);
                    auto tmp4 = at::vec::convert<float>(tmp3);
                    if (h_res == 0) {
                        tmp4.store(curr_k_pad + i, qk_head_size - i);
                    }
                    auto tmp5 = tmp4 * vec_b;
                    tmp5.store(curr_k_beta + i, qk_head_size - i);
                }
                // v_beta = value * beta.unsqueeze(-1)
                i = 0;
                for (; i < vec_size * (v_head_size / vec_size); i += vec_size) {
                    auto tmp3 = at::vec::Vectorized<scalar_t>::loadu(curr_v_orig + i);
                    auto tmp4 = at::vec::convert<float>(tmp3);
                    tmp4.store(curr_v_pad + i);
                    auto tmp5 = tmp4 * vec_b;
                    tmp5.store(curr_v_beta + i);
                }
                if (i < v_head_size) {
                    auto tmp3 = at::vec::Vectorized<scalar_t>::loadu(curr_v_orig + i, v_head_size - i);
                    auto tmp4 = at::vec::convert<float>(tmp3);
                    tmp4.store(curr_v_pad + i, v_head_size - i);
                    auto tmp5 = tmp4 * vec_b;
                    tmp5.store(curr_v_beta + i, v_head_size - i);
                }
                // Move to the next query
                at::native::data_index_step(h, v_num_head, l, total_seq_length);
            }
        });

        // Padding for g
        // g = g.cumsum(dim=-1)
        // g: [B, HV, num_chunk, chunk_size]
        at::parallel_for(0, v_num_head * num_chunk, 1, [&](int64_t begin, int64_t end) {
            int ompIdx = at::get_thread_num();
            int64_t h = 0, c = 0;
            at::native::data_index_init(begin, h, v_num_head, c, num_chunk);
            for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
                auto curr_g_orig = g_orig_ptr + h * gStrideH + c * chunk_size;
                auto curr_g_pad = g_pad_ptr + h * global_total_seq_length + c * chunk_size;
                float acc_val = 0;
                for (int64_t i = 0; i < chunk_size; i++) {
                    if (c * chunk_size + i < seq_len) {
                        acc_val += curr_g_orig[i];
                    }
                    curr_g_pad[i] = acc_val;
                }
                // Move to the next query
                at::native::data_index_step(h, v_num_head, c, num_chunk);
            }
        });

        // decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
        // decay_mask: [B, HV, num_chunk, chunk_size, chunk_size]
        at::Tensor decay_mask_data = at::zeros({v_num_head, num_chunk, chunk_size, chunk_size}, query.options().dtype(at::kFloat));
        float* decay_mask = decay_mask_data.data_ptr<float>();
        at::parallel_for(0, v_num_head * num_chunk, 1, [&](int64_t begin, int64_t end) {
            int ompIdx = at::get_thread_num();
            int64_t h = 0, c = 0;
            at::native::data_index_init(begin, h, v_num_head, c, num_chunk);
            for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
                auto curr_g_pad = g_pad_ptr + h * global_total_seq_length + c * chunk_size;
                auto curr_decay_mask = decay_mask + h * num_chunk * chunk_size * chunk_size + c * chunk_size * chunk_size;
                for (int64_t i = 0; i < chunk_size; i++) {
                    for (int64_t j = 0; j <= i; j++) {
                        auto tmp0 = curr_g_pad[i] - curr_g_pad[j];
                        auto tmp1 = std::exp(tmp0);
                        curr_decay_mask[i * chunk_size + j] = tmp1;
                    }
                }
                // Move to the next query
                at::native::data_index_step(h, v_num_head, c, num_chunk);
            }
        });

        // TODO: For all bmms, use VNNI and reduced type
        // attn = k_beta @ key.transpose(-1, -2)
        // attn: [B, HV, num_chunk, chunk_size, chunk_size]
        at::Tensor k_transpose_data = at::zeros({qk_num_head, num_chunk, qk_head_size, chunk_size}, query.options().dtype(at::kFloat));
        float* k_transpose = k_transpose_data.data_ptr<float>();
        at::Tensor attn_data = at::zeros({v_num_head, num_chunk, chunk_size, chunk_size}, query.options().dtype(at::kFloat));
        float* attn = attn_data.data_ptr<float>();
        at::parallel_for(0, v_num_head * num_chunk, 1, [&](int64_t begin, int64_t end) {
            int ompIdx = at::get_thread_num();
            int64_t h = 0, c = 0;
            at::native::data_index_init(begin, h, v_num_head, c, num_chunk);
            for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
                int64_t h_qk = h / head_group;
                auto curr_k_pad = k_pad_ptr + h_qk * global_total_seq_length * qk_head_size + c * chunk_size * qk_head_size;
                auto curr_k_beta = k_beta_ptr + h * global_total_seq_length * qk_head_size + c * chunk_size * qk_head_size;
                auto curr_k_transpose = k_transpose + h_qk * num_chunk * qk_head_size * chunk_size + c * qk_head_size * chunk_size;
                auto curr_attn = attn + h * num_chunk * chunk_size * chunk_size + c * chunk_size * chunk_size;
                auto curr_decay_mask = decay_mask + h * num_chunk * chunk_size * chunk_size + c * chunk_size * chunk_size;
                at::native::utils::transpose<float>(
                    /* M */ chunk_size,
                    /* N */ qk_head_size, 
                    /* src */ curr_k_pad,
                    /* ld_src */ qk_head_size,
                    /* dst */ curr_k_transpose,
                    /* ld_dst */ chunk_size);
                // k_beta @ key.transpose(-1, -2)
                at::native::cpublas::brgemm(
                    /* M */ chunk_size,
                    /* N */ chunk_size,
                    /* K */ qk_head_size,
                    /* lda */ qk_head_size,
                    /* ldb */ chunk_size,
                    /* ldc */ chunk_size,
                    /* add_C */ false,
                    /* A */ curr_k_beta,
                    /* B */ curr_k_transpose,
                    /* C */ curr_attn);
                // attn = attn * decay_mask
                for (int64_t m = 0; m < chunk_size; m++) {
                    at::vec::map2<float>(
                        [](at::vec::Vectorized<float> x, at::vec::Vectorized<float> y) { return x * y; },
                        curr_attn + m * chunk_size,
                        curr_attn + m * chunk_size,
                        curr_decay_mask + m * chunk_size,
                        chunk_size);
                }
                // Move to the next query
                at::native::data_index_step(h, v_num_head, c, num_chunk);
            }
        });

        // attn = -attn.masked_fill(mask, 0)
        // TODO: avoid additional allocation
        attn_data = -attn_data.masked_fill(triu_mask_0, 0);
        attn = attn_data.data_ptr<float>();

        // chunk decay
        // attn: [B, HV, num_chunk, chunk_size, chunk_size]
        // attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2) [B, HV, num_chunk, i]
        // attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
        at::parallel_for(0, v_num_head * num_chunk, 1, [&](int64_t begin, int64_t end) {
            int ompIdx = at::get_thread_num();
            int64_t h = 0, c = 0;
            at::native::data_index_init(begin, h, v_num_head, c, num_chunk);
            for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
                auto curr_attn = attn + h * num_chunk * chunk_size * chunk_size + c * chunk_size * chunk_size;
                for (int i = 1; i < chunk_size; i++) {
                    // row = attn[..., i, :i] [B, HK, num_chunk, i]
                    std::vector<float> row(i);
                    for (int j = 0; j < i; j++) {
                        row[j] = curr_attn[i * chunk_size + j];
                    }
                    // (row.unsqueeze(-1) * sub).sum(-2)
                    std::vector<float> updated(i, 0.0f);
                    for (int k = 0; k < i; k++) {
                        for (int j = 0; j < i; j++) {
                            updated[j] += row[k] * curr_attn[k * chunk_size + j]; // sum over k
                        }
                    }
                    // attn[..., i, :i] = row + sum(...)
                    for (int j = 0; j < i; j++) {
                        curr_attn[i * chunk_size + j] = row[j] + updated[j];
                    }
                }
                for (int i = 0; i < chunk_size; i++) {
                    curr_attn[i * chunk_size + i] += 1.0f;
                }
                // Move to the next query
                at::native::data_index_step(h, v_num_head, c, num_chunk);
            }
        });

        // value = attn @ v_beta
        // k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
        // value: [B, HV, num_chunk, chunk_size, EV]
        // k_beta_g = k_beta * g: [B, HV, num_chunk, chunk_size, EK]
        // k_cumdecay: [B, HV, num_chunk, chunk_size, EK]
        at::Tensor value_data = at::zeros({v_num_head, num_chunk, chunk_size, v_head_size}, query.options().dtype(at::kFloat));
        float* value = value_data.data_ptr<float>();
        at::Tensor k_beta_g_data = at::zeros({v_num_head, num_chunk, chunk_size, qk_head_size}, query.options().dtype(at::kFloat));
        float* k_beta_g = k_beta_g_data.data_ptr<float>();
        at::Tensor k_cumdecay_data = at::zeros({v_num_head, num_chunk, chunk_size, qk_head_size}, query.options().dtype(at::kFloat));
        float* k_cumdecay = k_cumdecay_data.data_ptr<float>();
        at::parallel_for(0, v_num_head * num_chunk, 1, [&](int64_t begin, int64_t end) {
            int ompIdx = at::get_thread_num();
            int64_t h = 0, c = 0;
            at::native::data_index_init(begin, h, v_num_head, c, num_chunk);
            for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
                auto curr_k_beta = k_beta_ptr + h * global_total_seq_length * qk_head_size + c * chunk_size * qk_head_size;
                auto curr_k_beta_g = k_beta_g + h * num_chunk * chunk_size * qk_head_size + c * chunk_size * qk_head_size;
                auto curr_k_cumdecay = k_cumdecay + h * num_chunk * chunk_size * qk_head_size + c * chunk_size * qk_head_size;
                auto curr_attn = attn + h * num_chunk * chunk_size * chunk_size + c * chunk_size * chunk_size;
                auto curr_v_beta = v_beta_ptr + h * global_total_seq_length * v_head_size + c * chunk_size * v_head_size;
                auto curr_value = value + h * num_chunk * chunk_size * v_head_size + c * chunk_size * v_head_size;
                auto curr_g_pad = g_pad_ptr + h * global_total_seq_length + c * chunk_size;
                // value = attn @ v_beta
                at::native::cpublas::brgemm(
                    /* M */ chunk_size,
                    /* N */ v_head_size,
                    /* K */ chunk_size,
                    /* lda */ chunk_size,
                    /* ldb */ v_head_size,
                    /* ldc */ v_head_size,
                    /* add_C */ false,
                    /* A */ curr_attn,
                    /* B */ curr_v_beta,
                    /* C */ curr_value);
                // k_beta_g = k_beta * g.exp().unsqueeze(-1)
                auto vec_size = at::vec::Vectorized<float>::size();
                for (int64_t j = 0; j < chunk_size; j++) {
                    int64_t i = 0;
                    float g_exp = std::exp(curr_g_pad[j]);
                    auto vec_g_exp = at::vec::Vectorized<float>(g_exp);
                    for (; i < vec_size * (qk_head_size / vec_size); i += vec_size) {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(curr_k_beta + j * qk_head_size + i);
                        auto tmp1 = tmp0 * vec_g_exp;
                        tmp1.store(curr_k_beta_g + j * qk_head_size + i);
                    }
                    if (i < qk_head_size) {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(curr_k_beta + j * qk_head_size + i, qk_head_size - i);
                        auto tmp1 = tmp0 * vec_g_exp;
                        tmp1.store(curr_k_beta_g + j * qk_head_size + i, qk_head_size - i);
                    }
                }
                // k_cumdecay = attn @ k_beta_g
                at::native::cpublas::brgemm(
                    /* M */ chunk_size,
                    /* N */ qk_head_size,
                    /* K */ chunk_size,
                    /* lda */ chunk_size,
                    /* ldb */ qk_head_size,
                    /* ldc */ qk_head_size,
                    /* add_C */ false,
                    /* A */ curr_attn,
                    /* B */ curr_k_beta_g,
                    /* C */ curr_k_cumdecay);
                // Move to the next query
                at::native::data_index_step(h, v_num_head, c, num_chunk);
            }
        });

        // for each chunk
        at::parallel_for(0, v_num_head, 1, [&](int64_t begin, int64_t end) {
            int ompIdx = at::get_thread_num();
            int64_t h = 0;
            at::native::data_index_init(begin, h, v_num_head);
            for ([[maybe_unused]] auto z : c10::irange(begin, end)) {
                int64_t h_qk = h / head_group;
                auto curr_q = q_pad_ptr + h_qk * global_total_seq_length * qk_head_size; // [num_chunk, chunk_size, EK]
                auto curr_k = k_pad_ptr + h_qk * global_total_seq_length * qk_head_size; // [num_chunk, chunk_size, EK]
                auto curr_v = value + h * num_chunk * chunk_size * v_head_size; // [num_chunk, chunk_size, EV]
                auto curr_decay_mask = decay_mask + h * num_chunk * chunk_size * chunk_size; // [num_chunk, chunk_size, chunk_size]
                auto curr_k_cumdecay = k_cumdecay + h * num_chunk * chunk_size * qk_head_size; // [num_chunk, chunk_size, EK]
                auto curr_last_recurrent_state = final_state_ptr + h * final_state_StrideH; // [EK, EV]
                auto curr_g_pad = g_pad_ptr + h * global_total_seq_length; // [num_chunk, chunk_size]
                auto curr_core_attn_out = core_attn_out_ptr + h * global_total_seq_length * v_head_size; // [num_chunk, chunk_size, EV]
                for (int64_t c = 0; c < num_chunk; c++) {
                    auto q_i = curr_q + c * chunk_size * qk_head_size; // [chunk_size, EK]
                    auto k_i = curr_k + c * chunk_size * qk_head_size; // [chunk_size, EK]
                    auto v_i = curr_v + c * chunk_size * v_head_size; // [chunk_size, EV]
                    auto decay_mask_i = curr_decay_mask + c * chunk_size * chunk_size; // [chunk_size, chunk_size]
                    auto k_cumdecay_i = curr_k_cumdecay + c * chunk_size * qk_head_size; // [chunk_size, EK]
                    auto g_pad_i = curr_g_pad + c * chunk_size; // [chunk_size]
                    auto core_attn_out_i = curr_core_attn_out + c * chunk_size * v_head_size; // [chunk_size, EV]

                    at::Tensor k_transpose_i_data = at::zeros({qk_head_size, chunk_size}, query.options().dtype(at::kFloat));
                    float* k_transpose_i = k_transpose_i_data.data_ptr<float>();
                    at::Tensor attn_i_data = at::zeros({chunk_size, chunk_size}, query.options().dtype(at::kFloat));
                    float* attn_i = attn_i_data.data_ptr<float>();
                    at::Tensor v_prime_data = at::zeros({chunk_size, v_head_size}, query.options().dtype(at::kFloat));
                    float* v_prime = v_prime_data.data_ptr<float>();
                    at::Tensor qg_data = at::zeros({chunk_size, qk_head_size}, query.options().dtype(at::kFloat));
                    float* qg = qg_data.data_ptr<float>();
                    at::Tensor attn_inter_data = at::zeros({chunk_size, v_head_size}, query.options().dtype(at::kFloat));
                    float* attn_inter = attn_inter_data.data_ptr<float>();
                    at::Tensor kg_data = at::zeros({chunk_size, qk_head_size}, query.options().dtype(at::kFloat));
                    float* kg = kg_data.data_ptr<float>();
                    at::Tensor kg_transpose_data = at::zeros({qk_head_size, chunk_size}, query.options().dtype(at::kFloat));
                    float* kg_transpose = kg_transpose_data.data_ptr<float>();
                    at::Tensor kgv_data = at::zeros({qk_head_size, v_head_size}, query.options().dtype(at::kFloat));
                    float* kgv = kgv_data.data_ptr<float>();

                    // attn_i = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
                    // k_transpose_i = k_i.transpose(-1, -2)
                    at::native::utils::transpose<float>(
                        /* M */ chunk_size,
                        /* N */ qk_head_size, 
                        /* src */ k_i,
                        /* ld_src */ qk_head_size,
                        /* dst */ k_transpose_i,
                        /* ld_dst */ chunk_size);
                    // attn_i = q_i @ k_transpose_i
                    at::native::cpublas::brgemm(
                        /* M */ chunk_size,
                        /* N */ chunk_size,
                        /* K */ qk_head_size,
                        /* lda */ qk_head_size,
                        /* ldb */ chunk_size,
                        /* ldc */ chunk_size,
                        /* add_C */ false,
                        /* A */ q_i,
                        /* B */ k_transpose_i,
                        /* C */ attn_i);
                    // attn_i = attn_i * decay_mask_i
                    auto vec_size = at::vec::Vectorized<float>::size();
                    for (int64_t m = 0; m < chunk_size; m++) {
                        auto attn_i_m = attn_i + m * chunk_size;
                        auto decay_mask_i_m = decay_mask_i + m * chunk_size;
                        int64_t n = 0;
                        for (; n < vec_size * (chunk_size / vec_size); n += vec_size) {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(attn_i_m + n);
                            auto tmp1 = at::vec::Vectorized<float>::loadu(decay_mask_i_m + n);
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(attn_i_m + n);
                        }
                        if (n < chunk_size) {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(attn_i_m + n, chunk_size - n);
                            auto tmp1 = at::vec::Vectorized<float>::loadu(decay_mask_i_m + n, chunk_size - n);
                            auto tmp2 = tmp0 * tmp1;
                            tmp2.store(attn_i_m + n, chunk_size - n);
                        }
                    }
                    // attn_i = attn_i.masked_fill_(mask, 0)
                    attn_i_data.masked_fill_(triu_mask_1, 0);

                    // v_prime = k_cumdecay_i @ curr_last_recurrent_state: [chunk_size, EV]
                    // k_cumdecay_i: [chunk_size, EK]
                    // curr_last_recurrent_state: [EK, EV]
                    at::native::cpublas::brgemm(
                        /* M */ chunk_size,
                        /* N */ v_head_size,
                        /* K */ qk_head_size,
                        /* lda */ qk_head_size,
                        /* ldb */ v_head_size,
                        /* ldc */ v_head_size,
                        /* add_C */ false,
                        /* A */ k_cumdecay_i,
                        /* B */ curr_last_recurrent_state,
                        /* C */ v_prime);
                    // v_new = v_prime = v_i - v_prime
                    // v_i: [chunk_size, EV]
                    for (int64_t m = 0; m < chunk_size; m++) {
                        at::vec::map2<float>(
                            [](at::vec::Vectorized<float> x, at::vec::Vectorized<float> y) { return x - y; },
                            v_prime + m * v_head_size,
                            v_i + m * v_head_size,
                            v_prime + m * v_head_size,
                            v_head_size);
                    }

                    // attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
                    // qg = q_i * g[:, :, i, :, None].exp(): [chunk_size, EK]
                    // q_i: [chunk_size, EK]
                    // g[:, :, i, :, None]: [chunk_size, 1]
                    for (int64_t m = 0; m < chunk_size; m++) {
                        auto g_pad_i_m = g_pad_i + m;
                        auto g_exp = std::exp(*g_pad_i_m);
                        at::vec::map<float>(
                            [g_exp](at::vec::Vectorized<float> x) { return x * at::vec::Vectorized<float>(g_exp); },
                            qg + m * qk_head_size,
                            q_i + m * qk_head_size,
                            qk_head_size);
                    }
                    // attn_inter = qg @ curr_last_recurrent_state: [chunk_size, EV]
                    // curr_last_recurrent_state: [EK, EV]
                    at::native::cpublas::brgemm(
                        /* M */ chunk_size,
                        /* N */ v_head_size,
                        /* K */ qk_head_size,
                        /* lda */ qk_head_size,
                        /* ldb */ v_head_size,
                        /* ldc */ v_head_size,
                        /* add_C */ false,
                        /* A */ qg,
                        /* B */ curr_last_recurrent_state,
                        /* C */ attn_inter);

                    // core_attn_out[:, :, i] = attn_inter + attn_i @ v_new
                    // attn_inter = attn_inter + attn_i @ v_new: [chunk_size, EV]
                    // attn_i: [chunk_size, chunk_size]
                    // v_new: [chunk_size, EV]
                    at::native::cpublas::brgemm(
                        /* M */ chunk_size,
                        /* N */ v_head_size,
                        /* K */ chunk_size,
                        /* lda */ chunk_size,
                        /* ldb */ v_head_size,
                        /* ldc */ v_head_size,
                        /* add_C */ true,
                        /* A */ attn_i,
                        /* B */ v_prime,
                        /* C */ attn_inter);

                    // core_attn_out[:, :, i] = attn_inter
                    for (int64_t m = 0; m < chunk_size; m++) {
                        at::vec::map<float>(
                            [](at::vec::Vectorized<float> x) { return x; },
                            core_attn_out_i + m * v_head_size,
                            attn_inter + m * v_head_size,
                            v_head_size);
                    }

                    // last_recurrent_state = (
                    //     last_recurrent_state * g[:, :, i, -1, None, None].exp()
                    //     + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
                    // )
                    // 1) last_recurrent_state * g[:, :, i, -1, None, None].exp()
                        // curr_last_recurrent_state: [EK, EV]
                        // g[:, :, i, -1, None, None]: [1, 1]
                        // last_recurrent_state * g[:, :, i, -1, None, None].exp(): [EK, EV]
                    auto g_pad_i_last = g_pad_i + chunk_size - 1;
                    auto g_exp_last = std::exp(g_pad_i_last[0]);
                    for (int64_t m = 0; m < qk_head_size; m++) {
                        at::vec::map<float>(
                            [g_exp_last](at::vec::Vectorized<float> x) { return x * at::vec::Vectorized<float>(g_exp_last); },
                            curr_last_recurrent_state + m * v_head_size,
                            curr_last_recurrent_state + m * v_head_size,
                            v_head_size);
                    }
                    // 2) (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
                        // k_i: [chunk_size, EK]
                        // g[:, :, i, -1, None]: [1]
                        // g[:, :, i]: [chunk_size]
                        // (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]: [chunk_size, 1]
                        // kg = k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]: [chunk_size, EK]
                        // (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2): [EK, chunk_size]
                        // v_new: [chunk_size, EV]
                        // (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new: [EK, EV]
                    // kg = k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
                    for (int64_t m = 0; m < chunk_size; m++) {
                        auto g_exp = std::exp((g_pad_i_last[0] - g_pad_i[m]));
                        at::vec::map<float>(
                            [g_exp](at::vec::Vectorized<float> x) { return x * at::vec::Vectorized<float>(g_exp); },
                            kg + m * qk_head_size,
                            k_i + m * qk_head_size,
                            qk_head_size);
                    }
                    // kg.transpose(-1, -2): [EK, chunk_size]
                    at::native::utils::transpose<float>(
                        /* M */ chunk_size,
                        /* N */ qk_head_size, 
                        /* src */ kg,
                        /* ld_src */ qk_head_size,
                        /* dst */ kg_transpose,
                        /* ld_dst */ chunk_size);
                    // kgv = kg.transpose(-1, -2) @ v_new
                    // v_new: [chunk_size, EV]
                    at::native::cpublas::brgemm(
                        /* M */ qk_head_size,
                        /* N */ v_head_size,
                        /* K */ chunk_size,
                        /* lda */ chunk_size,
                        /* ldb */ v_head_size,
                        /* ldc */ v_head_size,
                        /* add_C */ false,
                        /* A */ kg_transpose,
                        /* B */ v_prime,
                        /* C */ kgv);
                    // last_recurrent_state = 1) + 2)
                    for (int64_t m = 0; m < qk_head_size; m++) {
                        at::vec::map2<float>(
                            [](at::vec::Vectorized<float> x, at::vec::Vectorized<float> y) { return x + y; },
                            curr_last_recurrent_state + m * v_head_size,
                            curr_last_recurrent_state + m * v_head_size,
                            kgv + m * v_head_size,
                            v_head_size);
                    }
                }

                // core_attn_out -> output
                // output: [B, T, HV, EV]
                // core_attn_out: [B, HV, padded_T, EV]
                auto curr_out = out_ptr + h * oStrideH;
                for (int64_t m = 0; m < seq_len; m++) {
                    at::vec::map<scalar_t>(
                        [](at::vec::Vectorized<float> x) { return x; },
                        curr_out + m * oStrideT,
                        curr_core_attn_out + m * v_head_size,
                        v_head_size);
                }

                // Move to the next query
                at::native::data_index_step(h, v_num_head);
            }
        });

        start_q = end_q;
    }
    return;
}

}  // anonymous namespace

extern at::Tensor qwen3_next_l2norm_cpu(at::Tensor& input, double eps);

std::tuple<at::Tensor, at::Tensor> chunk_gated_delta_rule_cpu(
        at::Tensor& query,
        at::Tensor& key,
        at::Tensor& value,
        at::Tensor& g,
        at::Tensor& beta,
        at::Tensor& cu_seqlens,
        at::Tensor& initial_state,
        bool output_final_state,
        bool use_qk_l2norm_in_kernel) {

    at::Tensor output = at::empty_like(value, value.options()); // [B, T, HV, EV]
    at::Tensor final_state;
    if (initial_state.defined()) {
        final_state = initial_state.to(at::kFloat);
    } else {
        final_state = at::empty_like(initial_state, initial_state.options()); // [N, HV, EK, EV]
    }
    at::Tensor query_ = query;
    at::Tensor key_ = key;
    if (use_qk_l2norm_in_kernel) {
        query_ = qwen3_next_l2norm_cpu(query_, 1e-6);
        key_ = qwen3_next_l2norm_cpu(key_, 1e-6);
    }
    AT_DISPATCH_REDUCED_FLOATING_TYPES(query.scalar_type(), "chunk_gated_delta_rule_kernel", [&] {
        chunk_gated_delta_rule_kernel_impl<scalar_t>(
            output,
            final_state,
            query_,
            key_,
            value,
            g,
            beta,
            cu_seqlens
        );
    });
    // For output_final_state = true
    return std::make_tuple(std::move(output), std::move(final_state));
}
#include "common.h"
#include "vec.h"

namespace {

template <typename T>
std::tuple<at::Tensor, at::Tensor> rope_kernel_impl(
    at::Tensor& q_pe,
    at::Tensor& k_pe,
    at::Tensor& t_emb_pos_sin,
    at::Tensor& t_emb_pos_cos,
    at::Tensor& t_pos) {
    auto in_sizes = q_pe.sizes(); // in[S][N][rotary_dim]
    auto S = in_sizes[0]; // seq len
    auto N = in_sizes[1]; // number of head
    auto rotary_dim = in_sizes[2]; // qk_rope_head_dim
    auto MP = t_emb_pos_sin.size(0); // Max Pos
    auto HR = t_emb_pos_sin.size(2); // rotary_dim
    auto in_stride_s = q_pe.stride(0);
    CHECK_EQ(MP, S);
    CHECK_EQ(MP, t_emb_pos_cos.size(0));
    CHECK_EQ(HR, rotary_dim);
    CHECK_EQ(HR, t_emb_pos_cos.size(2));
    CHECK_EQ(N, in_sizes[1]);
    CHECK_EQ(rotary_dim, in_sizes[2]);

    auto in_ptr = q_pe.data_ptr<T>();
    auto k_pe_ptr = k_pe.data_ptr<T>();
    auto k_pe_stride_s = k_pe.stride(0);

    auto NK = 1;
    auto HK = k_pe.size(-1);
    CHECK_EQ(HK, rotary_dim);

    // initialize empty q/k/v
    auto query = at::empty({S, N, rotary_dim}, q_pe.options());
    auto key = at::empty({S, NK, HK}, k_pe.options());
    auto query_ptr = query.data_ptr<T>();
    auto key_ptr = key.data_ptr<T>();
    auto out_stride_qs = query.stride(0);
    auto out_stride_ks = key.stride(0);
    auto emb_pos_sin_ptr = t_emb_pos_sin.data_ptr<T>(); // [MP][1][HR]
    auto emb_pos_cos_ptr = t_emb_pos_cos.data_ptr<T>(); // [MP][1][HR]
    auto pos_ptr = t_pos.data_ptr<long>(); // [S]
    {
#pragma omp parallel for collapse(2)
    for (int s = 0; s < S; s++) {
        for (int n = 0; n < N; n++) {
            auto in_offset_q = s * in_stride_s + n * rotary_dim;
            auto out_offset_q = s * out_stride_qs + n * rotary_dim;
            auto out_offset_k = s * out_stride_ks;
            long p = 0;
            T* sin_start = nullptr;
            T* cos_start = nullptr;
            // step 0) get the rotary position embedding for the current position
            auto start_idx = 0;
            p = pos_ptr[start_idx + s];
            sin_start = emb_pos_sin_ptr + p * HR;
            cos_start = emb_pos_cos_ptr + p * HR;
            // step 1) apply_rotary_pos_emb for the rotary_dim elements in every
            // head of query/key
            for (auto h = 0; h < rotary_dim; h += 2) {
                auto cos1 = cos_start[h];
                auto sin1 = sin_start[h];
                auto cos2 = cos_start[h + 1];
                auto sin2 = sin_start[h + 1];
                auto in1 = in_ptr[in_offset_q + h];
                auto in2 = in_ptr[in_offset_q + h + 1];
                auto out1 = in1 * cos1 - in2 * sin1;
                auto out2 = in2 * cos2 + in1 * sin2;
                auto out1_offset = out_offset_q + h;
                auto out2_offset = out1_offset + 1;
                query_ptr[out1_offset] = out1;
                query_ptr[out2_offset] = out2;
            }
            if (n < NK) {
                for (auto h = 0; h < HK; h += 2) {
                    auto cos1 = cos_start[h];
                    auto sin1 = sin_start[h];
                    auto cos2 = cos_start[h + 1];
                    auto sin2 = sin_start[h + 1];
                    auto k_pe_offset = s * k_pe_stride_s;
                    auto in1_k = k_pe_ptr[k_pe_offset + h];
                    auto in2_k = k_pe_ptr[k_pe_offset + h + 1];
                    auto out1_k = in1_k * cos1 - in2_k * sin1;
                    auto out2_k = in2_k * cos2 + in1_k * sin2;
                    auto out1_offset = out_offset_k + h;
                    auto out2_offset = out1_offset + 1;
                    key_ptr[out1_offset] = out1_k;
                    key_ptr[out2_offset] = out2_k;
                }
            }
        }
    }
    }
    return std::make_tuple(query, key);
}

}

std::tuple<at::Tensor, at::Tensor>
rotary_position_embedding_cpu(
    at::Tensor& q_pe,
    at::Tensor& k_pe,
    at::Tensor& t_emb_pos_sin,
    at::Tensor& t_emb_pos_cos,
    at::Tensor& t_pos) {
  q_pe = q_pe.contiguous();
  k_pe = k_pe.contiguous();
  t_emb_pos_sin = t_emb_pos_sin.contiguous();
  t_emb_pos_cos = t_emb_pos_cos.contiguous();
  t_pos = t_pos.contiguous();
  if (q_pe.scalar_type() == at::kFloat) {
    return rope_kernel_impl<float>(
        q_pe, k_pe, t_emb_pos_sin, t_emb_pos_cos, t_pos);
  } else if (q_pe.scalar_type() == at::kBFloat16) {
    return rope_kernel_impl<at::BFloat16>(
        q_pe, k_pe, t_emb_pos_sin, t_emb_pos_cos, t_pos);
  } else {
    TORCH_CHECK(
        false,
        "rotary_position_embedding_cpu: unsupported '",
        q_pe.scalar_type(),
        "'");
    return std::make_tuple(at::Tensor(), at::Tensor());
  }
}

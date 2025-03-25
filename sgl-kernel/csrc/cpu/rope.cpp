#include "common.h"
#include "vec.h"

namespace {

template <typename scalar_t>
void rope_kernel_impl(
    scalar_t* __restrict__ q_pe_out,
    scalar_t* __restrict__ k_pe_out,
    int64_t* __restrict__ t_pos,
    scalar_t* __restrict__ q_pe,
    scalar_t* __restrict__ k_pe,
    scalar_t* __restrict__ t_emb_pos,
    int seq_len,
    int num_head,
    int rotary_dim,
    int HR,
    int in_stride_s,
    int out_stride_qs,
    int out_stride_ks,
    int HK,
    int k_pe_stride_s) {
  int COFF = HR / 2;
  at::parallel_for(0, seq_len, 1024 / rotary_dim, [&](int begin, int end) {
    for (int s = begin; s < end; ++s) {
      for (int n = 0; n < num_head; n++) {
        int in_offset_q = s * in_stride_s + n * rotary_dim;
        int out_offset_q = s * out_stride_qs + n * rotary_dim;
        int out_offset_k = s * out_stride_ks;
        long p = 0;
        scalar_t* sin_start = nullptr;
        scalar_t* cos_start = nullptr;
        // step 0) get the rotary position embedding for the current position
        p = t_pos[s];
        sin_start = t_emb_pos + p * HR + COFF;
        cos_start = t_emb_pos + p * HR;
        // step 1) apply_rotary_pos_emb for the rotary_dim elements in every
        // head of query/key
        for (int h = 0; h < rotary_dim; h += 2) {
          scalar_t cos = cos_start[h >> 1];
          scalar_t sin = sin_start[h >> 1];
          scalar_t in1 = q_pe[in_offset_q + h];
          scalar_t in2 = q_pe[in_offset_q + h + 1];
          scalar_t out1 = in1 * cos - in2 * sin;
          scalar_t out2 = in2 * cos + in1 * sin;
          q_pe_out[out_offset_q + h] = out1;
          q_pe_out[out_offset_q + h + 1] = out2;
        }
        for (int h = 0; h < HK; h += 2) {
          scalar_t cos = cos_start[h >> 1];
          scalar_t sin = sin_start[h >> 1];
          int k_pe_offset = s * k_pe_stride_s;
          scalar_t in1_k = k_pe[k_pe_offset + h];
          scalar_t in2_k = k_pe[k_pe_offset + h + 1];
          scalar_t out1_k = in1_k * cos - in2_k * sin;
          scalar_t out2_k = in2_k * cos + in1_k * sin;
          k_pe_out[out_offset_k + h] = out1_k;
          k_pe_out[out_offset_k + h + 1] = out2_k;
        }
      }
    }
  });
}
}  // namespace

std::tuple<at::Tensor, at::Tensor>
rotary_position_embedding_cpu(at::Tensor& t_pos, at::Tensor& q_pe, at::Tensor& k_pe, at::Tensor& t_emb_pos) {
  RECORD_FUNCTION(
      "sgl-kernel::rotary_position_embedding_cpu", std::vector<c10::IValue>({t_pos, q_pe, k_pe, t_emb_pos}));
  t_pos = t_pos.contiguous();
  q_pe = q_pe.contiguous();
  k_pe = k_pe.contiguous();
  t_emb_pos = t_emb_pos.contiguous();
  CHECK_INPUT(t_pos);
  CHECK_INPUT(q_pe);
  CHECK_INPUT(k_pe);
  CHECK_INPUT(t_emb_pos);
  at::Tensor q_pe_out = at::empty_like(q_pe);
  at::Tensor k_pe_out = at::empty_like(k_pe);

  auto in_sizes = q_pe.sizes();   // in[S][N][rotary_dim]
  auto S = in_sizes[0];           // seq len
  auto N = in_sizes[1];           // number of head
  auto rotary_dim = in_sizes[2];  // qk_rope_head_dim
  auto MP = t_emb_pos.size(0);    // Max Pos
  auto HR = t_emb_pos.size(1);    // rotary_dim
  auto in_stride_s = q_pe.stride(0);
  auto NK = 1;
  auto HK = k_pe.size(-1);
  CHECK_EQ(S, t_pos.size(0));
  CHECK_EQ(HR, rotary_dim);
  CHECK_EQ(rotary_dim, in_sizes[2]);
  CHECK_EQ(HK, rotary_dim);
  auto k_pe_stride_s = k_pe.stride(0);
  auto out_stride_qs = q_pe_out.stride(0);
  auto out_stride_ks = k_pe_out.stride(0);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(q_pe.scalar_type(), "rotary_position_embedding_cpu", [&] {
    rope_kernel_impl<scalar_t>(
        q_pe_out.data_ptr<scalar_t>(),
        k_pe_out.data_ptr<scalar_t>(),
        t_pos.data_ptr<int64_t>(),
        q_pe.data_ptr<scalar_t>(),
        k_pe.data_ptr<scalar_t>(),
        t_emb_pos.data_ptr<scalar_t>(),
        S,
        N,
        rotary_dim,
        HR,
        in_stride_s,
        out_stride_qs,
        out_stride_ks,
        HK,
        k_pe_stride_s);
  });
  return std::make_tuple(q_pe_out, k_pe_out);
}

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
  CHECK_INPUT (t_pos);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT (q_pe);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT (k_pe);
  CHECK_INPUT (t_emb_pos);
  CHECK_DIM(1, t_pos);
  CHECK_DIM(3, q_pe);
  CHECK_DIM(3, k_pe);
  CHECK_DIM(2, t_emb_pos);

  int seq_len = q_pe.size(0);
  int num_head = q_pe.size(1);
  int rotary_dim = q_pe.size(2);
  int HK = k_pe.size(2);
  int HR = t_emb_pos.size(1);
  CHECK_EQ(HR, rotary_dim);
  CHECK_EQ(k_pe.size(0), seq_len);
  CHECK_EQ(k_pe.size(1), 1);
  CHECK_EQ(t_pos.size(0), seq_len);
  CHECK_EQ(HK, rotary_dim);

  at::Tensor q_pe_out = at::empty_like(q_pe);
  at::Tensor k_pe_out = at::empty_like(k_pe);
  int q_pe_stride_s = q_pe.stride(0);
  int k_pe_stride_s = k_pe.stride(0);
  int out_stride_qs = q_pe_out.stride(0);
  int out_stride_ks = k_pe_out.stride(0);

  const auto input_dtype = q_pe.scalar_type();
  TORCH_CHECK(t_pos.scalar_type() == at::kLong, "expect positions to be int64, got ", t_pos.scalar_type());
  TORCH_CHECK(input_dtype == k_pe.scalar_type(), "q_pe and k_pe must have the same data type");
  TORCH_CHECK(input_dtype == t_emb_pos.scalar_type(), "q_pe and t_emb_pos must have the same data type");

  AT_DISPATCH_REDUCED_FLOATING_TYPES(input_dtype, "rotary_position_embedding_cpu", [&] {
    rope_kernel_impl<scalar_t>(
        q_pe_out.data_ptr<scalar_t>(),
        k_pe_out.data_ptr<scalar_t>(),
        t_pos.data_ptr<int64_t>(),
        q_pe.data_ptr<scalar_t>(),
        k_pe.data_ptr<scalar_t>(),
        t_emb_pos.data_ptr<scalar_t>(),
        seq_len,
        num_head,
        rotary_dim,
        HR,
        q_pe_stride_s,
        out_stride_qs,
        out_stride_ks,
        HK,
        k_pe_stride_s);
  });
  return std::make_tuple(q_pe_out, k_pe_out);
}

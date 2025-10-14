#include "common.h"
#include "vec.h"
#include <ATen/native/CPUBlas.h>
#include <ATen/native/cpu/utils.h>

namespace {

inline float softplus(float x) {
    if (x > 20.0f) return x;
    else if (x < -20.0f) return std::exp(x);
    else return std::log1p(std::exp(x));
}

inline at::vec::Vectorized<float> softplus(const at::vec::Vectorized<float>& x) {
    at::vec::Vectorized<float> mask_hi = x > at::vec::Vectorized<float>(20.0f);
    at::vec::Vectorized<float> mask_lo = x < at::vec::Vectorized<float>(-20.0f);

    at::vec::Vectorized<float> expx = x.exp();
    at::vec::Vectorized<float> log1pex = (expx + at::vec::Vectorized<float>(1.0f)).log();

    return at::vec::Vectorized<float>::blendv(at::vec::Vectorized<float>::blendv(log1pex, expx, mask_lo), x, mask_hi);
}

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ src, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  constexpr int kVecSize = bVec::size();
  int64_t d = 0;
#pragma GCC unroll 4
  for (; d <= size - kVecSize; d += kVecSize) {
    bVec out_bvec = bVec::loadu(src + d);
    out_bvec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = src[d];
  }
}


template <typename scalar_t>
void fused_gdn_gating_kernel_impl(float* __restrict__ A_log, 
                                 const scalar_t* __restrict__ a,
                                 const scalar_t* __restrict__ dt_bias,
                                 float* __restrict__ out,
                                 int64_t batch,
                                 int64_t num_heads) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int vec_size = bVec::size();
  constexpr int fvec_size = fVec::size();
  fVec neg_one(-1.0f);
  at::parallel_for(0, batch, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
        int64_t j = 0;
        for(; j < num_heads - (num_heads % vec_size); j += vec_size) {
            fVec A_log_vec0 = fVec::loadu(A_log + j);
            fVec A_log_vec1 = fVec::loadu(A_log + j + fvec_size);
            bVec dt_bias_vec = bVec::loadu(dt_bias + j);
            bVec a_bvec = bVec::loadu(a + i * num_heads + j);
            fVec a0, a1, dt_bias_vec0, dt_bias_vec1;
            std::tie(a0, a1) = at::vec::convert_to_float(a_bvec);
            std::tie(dt_bias_vec0, dt_bias_vec1) = at::vec::convert_to_float(dt_bias_vec);

            fVec g0 = neg_one * A_log_vec0.exp() * softplus(a0 + dt_bias_vec0);
            fVec g1 = neg_one * A_log_vec1.exp() * softplus(a1 + dt_bias_vec1);

            g0.store(out + i * num_heads + j);
            g1.store(out + i * num_heads + j + fvec_size);
        }
        for(; j< num_heads; ++j) {
            out[i * num_heads + j] = -std::exp(A_log[j]) * softplus(float(a[i * num_heads + j]) + float(dt_bias[j]));
        }
    }
  });
}
template <typename T>
at::Tensor causal_conv1d_update_kernel_inner(
    const at::Tensor& hidden_states,
    at::Tensor& conv_states,
    const at::Tensor& cache_indices,
    const at::Tensor& conv_weights,
    const c10::optional<at::Tensor>& conv_bias,
    bool silu_activation,
    const c10::optional<at::Tensor>& cache_seqlens) {
  auto bs = cache_indices.size(0);
  auto channels = conv_states.size(1);
  auto kernel_size = conv_weights.size(1);
  auto state_len = conv_states.size(2);
  auto seqlen = hidden_states.dim() == 3 ? hidden_states.size(2) : 1;
  auto has_bias = conv_bias.has_value();
  auto bias_ptr = has_bias ? conv_bias.value().data_ptr<T>() : nullptr;
  auto conv_states_ptr = conv_states.data_ptr<T>();
  auto conv_weights_ptr = conv_weights.data_ptr<T>();
  auto hidden_states_ptr = hidden_states.data_ptr<T>();
  auto cache_indices_ptr = cache_indices.data_ptr<int>();
  auto hidden_states_strideB = hidden_states.stride(0);
  auto hidden_states_strideC = hidden_states.stride(1);
  auto hidden_states_strideS =
      hidden_states.dim() == 3 ? hidden_states.stride(2) : 0;
  auto res = at::empty_like(hidden_states);
  auto res_ptr = res.data_ptr<T>();
  auto res_strideB = res.stride(0);
  auto res_strideC = res.stride(1);
  auto res_strideS = res.dim() == 3 ? res.stride(2) : 0;
  auto conv_states_strideB = conv_states.stride(0);
  auto conv_states_strideC = conv_states.stride(1);
  auto conv_states_strideK = conv_states.stride(2);
  auto conv_weights_strideC = conv_weights.stride(0);
  bool has_cache_seqlens = cache_seqlens.has_value();
  auto cache_seqlens_ptr =
      has_cache_seqlens ? cache_seqlens.value().data_ptr<int>() : nullptr;
  if (has_cache_seqlens) {
    auto x_new = at::empty(
        {bs, channels, kernel_size - 1 + seqlen}, hidden_states.options());
    auto x_new_ptr = x_new.data_ptr<T>();
    auto x_new_strideB = x_new.stride(0);
    auto x_new_strideC = x_new.stride(1);
#pragma omp parallel for collapse(2)
    for (auto bi = 0; bi < bs; bi++) {
      for (auto ci = 0; ci < channels; ci++) {
        auto cache_seqlen = cache_seqlens_ptr[bi];
        auto copy_idx = cache_seqlen % state_len;
        auto conv_states_start =
            cache_indices_ptr[bi] * conv_states_strideB + ci * conv_states_strideC;
        auto conv_weights_start = ci * conv_weights_strideC;
        auto hidden_states_start =
            bi * hidden_states_strideB + ci * hidden_states_strideC;
        for (auto k = 0; k < kernel_size - 1; k++) {
          auto width_idx =
              (k - (kernel_size - 1) + cache_seqlen + state_len) % state_len;
          x_new_ptr[bi * x_new_strideB + ci * x_new_strideC + k] =
              conv_states_ptr
                  [conv_states_start + width_idx * conv_states_strideK];
        }
        for (auto k = 0; k < seqlen; k++) {
          x_new_ptr
              [bi * x_new_strideB + ci * x_new_strideC + k + kernel_size - 1] =
                  hidden_states_ptr
                      [hidden_states_start + k * hidden_states_strideS];
        }
        float outs[seqlen] = {0.0f};
        for (auto k = 0; k < kernel_size + seqlen; k++) {
          for (auto si = 0; si < seqlen; si++) {
            if (k - si >= 0 && k - si < kernel_size) {
              outs[si] += conv_weights_ptr[conv_weights_start + k - si] *
                  x_new_ptr[bi * x_new_strideB + ci * x_new_strideC + k];
            }
          }
        }
        for (auto si = 0; si < seqlen; si++) {
          if (has_bias) {
            outs[si] += bias_ptr[ci];
          }
          if (silu_activation) {
            outs[si] = outs[si] / (1 + expf(-outs[si]));
          }
          res_ptr[bi * res_strideB + ci * res_strideC + si * res_strideS] =
              outs[si];
        }
        for (auto si = 0; si < state_len; si++) {
          if ((si >= copy_idx && si < copy_idx + seqlen) ||
              (copy_idx + seqlen > state_len &&
               si < (copy_idx + seqlen) % state_len)) {
            conv_states_ptr[conv_states_start + si * conv_states_strideK] =
                hidden_states_ptr
                    [hidden_states_start +
                     ((si + state_len - copy_idx) % state_len) *
                         hidden_states_strideS];
          }
        }
      }
    }

  } else {
#pragma omp parallel for collapse(2)
    for (auto bi = 0; bi < bs; bi++) {
      for (auto ci = 0; ci < channels; ci++) {
        auto conv_states_start =
            cache_indices_ptr[bi] * conv_states_strideB + ci * conv_states_strideC;
        auto conv_weights_start = ci * conv_weights_strideC;
        auto hidden_states_start =
            bi * hidden_states_strideB + ci * hidden_states_strideC;
        float outs[seqlen] = {0.0f};
        for (auto k = state_len - kernel_size; k < state_len + seqlen; k++) {
          for (auto si = 0; si < seqlen; si++) {
            auto k_end = state_len + si + 1;
            auto k_start = k_end - kernel_size;
            if (k >= k_start && k < k_end) {
              if (k < state_len) {
                outs[si] += conv_weights_ptr[conv_weights_start + k - k_start] *
                    conv_states_ptr
                        [conv_states_start + k * conv_states_strideK];
              } else {
                outs[si] += conv_weights_ptr[conv_weights_start + k - k_start] *
                    hidden_states_ptr
                        [hidden_states_start +
                         (k - state_len) * hidden_states_strideS];
              }
            }
          }
        }
        for (auto si = 0; si < seqlen; si++) {
          if (has_bias) {
            outs[si] += bias_ptr[ci];
          }
          if (silu_activation) {
            outs[si] = outs[si] / (1 + expf(-outs[si]));
          }
          res_ptr[bi * res_strideB + ci * res_strideC + si * res_strideS] =
              outs[si];
        }
        for (auto si = 0; si < state_len; si++) {
          if (si + seqlen < state_len) {
            conv_states_ptr[conv_states_start + si * conv_states_strideK] =
                conv_states_ptr
                    [conv_states_start + (si + seqlen) * conv_states_strideK];
          } else {
            conv_states_ptr[conv_states_start + si * conv_states_strideK] =
                hidden_states_ptr
                    [hidden_states_start +
                     (si - state_len + seqlen) * hidden_states_strideS];
          }
        }
      }
    }
  }
  return std::move(res);
}
template <typename T>
std::tuple<at::Tensor, at::Tensor> causal_conv1d_fn_kernel_inner(
    const at::Tensor& x,
    const at::Tensor& conv_weights,
    const c10::optional<at::Tensor>& conv_bias,
    const c10::optional<at::Tensor>& initial_states,
    const at::Tensor& final_states_out,
    bool silu_activation) {
  auto batch = x.size(0);
  auto seqlen = x.size(-1);
  auto dim = conv_weights.size(0);
  auto width = conv_weights.size(1);
  auto has_bias = conv_bias.has_value();
  auto bias_ptr = has_bias ? conv_bias.value().data_ptr<T>() : nullptr;
  auto conv_weights_ptr = conv_weights.data_ptr<T>();
  auto x_ptr = x.data_ptr<T>();
  auto x_strideB = x.stride(0);
  auto x_strideD = x.stride(1);
  auto x_strideS = x.stride(2);
  auto has_initial_states = initial_states.has_value();
  auto initial_states_ptr =
      has_initial_states ? initial_states.value().data_ptr<T>() : nullptr;
  auto initial_len = has_initial_states ? width - 1 : 0;
  auto final_states_out_ptr = final_states_out.data_ptr<T>();
  auto out = at::empty_like(x);
  auto out_ptr = out.data_ptr<T>();
  auto out_strideB = out.stride(0);
  auto out_strideD = out.stride(1);
  auto out_strideS = out.stride(2);
#pragma omp parallel for collapse(2)
  for (auto bi = 0; bi < batch; bi++) {
    for (auto di = 0; di < dim; di++) {
      for (auto li = 0; li < seqlen; li++) {
        auto out_idx = bi * out_strideB + di * out_strideD + li * out_strideS;
        out_ptr[out_idx] = 0;
        for (auto wi = 0; wi < width; wi++) {
          if (li - wi >= 0) {
            out_ptr[out_idx] += conv_weights_ptr[di * width + width - 1 - wi] *
                x_ptr[bi * x_strideB + di * x_strideD + (li - wi) * x_strideS];
          } else if (has_initial_states) {
            out_ptr[out_idx] += conv_weights_ptr[di * width + width - 1 - wi] *
                initial_states_ptr[bi * dim * (width - 1) + di * (width - 1) +
                                   width - 1 + (li - wi)];
          }
        }
        if (has_bias) {
          out_ptr[out_idx] += bias_ptr[di];
        }
        if (silu_activation) {
          out_ptr[out_idx] = out_ptr[out_idx] / (1 + expf(-out_ptr[out_idx]));
        }
      }
      for (auto li = 0; li < width - 1; li++) {
        auto final_states_out_idx =
            bi * dim * (width - 1) + di * (width - 1) + li;
        if (li < width - 1 - seqlen - initial_len) {
          final_states_out_ptr[final_states_out_idx] = 0;
        } else if (li < width - 1 - seqlen) {
          final_states_out_ptr[final_states_out_idx] = initial_states_ptr
              [bi * dim * (width - 1) + di * (width - 1) + li + seqlen];
        } else {
          final_states_out_ptr[final_states_out_idx] = x_ptr
              [bi * out_strideB + di * out_strideD +
               (li - width + 1 + seqlen) * out_strideS];
        }
      }
    }
  }
  return std::make_tuple(std::move(out), std::move(final_states_out));
}

template <typename scalar_t>
void fused_recurrent_gated_delta_rule_kernel_impl(
    const scalar_t* __restrict__ q_ptr,
    const scalar_t* __restrict__ k_ptr,
    const scalar_t* __restrict__ v_ptr,
    const float* __restrict__ g_ptr,
    const scalar_t* __restrict__ beta_ptr,
    const int32_t* __restrict__ indices_ptr,
    float* __restrict__ state_ptr,
    scalar_t* __restrict__ o_ptr,
    float* __restrict__ kv_mem_ptr,
    int64_t seq_len,
    int64_t batch_size,
    int64_t num_heads,
    int64_t head_dim,
    int64_t v_num_heads,
    int64_t v_head_dim,
    int64_t q_strideB,
    int64_t q_strideS,
    int64_t q_strideH,
    int64_t k_strideB,
    int64_t k_strideS,
    int64_t k_strideH,
    int64_t v_strideB,
    int64_t v_strideS,
    int64_t v_strideH) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  constexpr int64_t VecSize = bVec::size();
  constexpr int64_t fVecSize = fVec::size();
  int64_t group_size = v_num_heads / num_heads;
  double scale = 1 / std::sqrt(head_dim);
  fVec scale_vec = fVec(scale);
  at::parallel_for(0, batch_size * seq_len * v_num_heads, 0, [&](int64_t begin, int64_t end) {
    int64_t bi{0}, si{0}, ni{0};
    data_index_init(begin, bi, batch_size, si, seq_len, ni, v_num_heads);
    for (int64_t i = begin; i < end; ++i) {
        int64_t cache_index = indices_ptr[bi];
        int64_t state_offset = (cache_index * v_num_heads + ni) * head_dim * v_head_dim;
        float g_val = g_ptr[ni];
        float g_val_exp = std::exp(g_val);
        fVec g_val_exp_vec = fVec(g_val_exp);
        int64_t q_offset = si * q_strideS + bi * q_strideB + (ni / group_size) * q_strideH;
        int64_t k_offset = si * k_strideS + bi * k_strideB + (ni / group_size) * k_strideH;
        int64_t v_offset = si * v_strideS + bi * v_strideB + ni * v_strideH;
        int64_t o_offset = ((bi * seq_len + si) * v_num_heads + ni) * v_head_dim;
        int64_t dt_kv_mem_offset = ((bi * seq_len + si) * v_num_heads + ni) * v_head_dim;
        float beta_val = beta_ptr[ni];
        fVec beta_vec = fVec(beta_val);
        int64_t dvi = 0;
        for (; dvi <= v_head_dim - fVecSize; dvi += fVecSize) {
          for (int di = 0; di < head_dim; ++di) {
            fVec k_val_vec = fVec(k_ptr[k_offset + di]);
            fVec state_vec = fVec::loadu(state_ptr + state_offset + di * v_head_dim + dvi);
            fVec kv_mem_vec = fVec::loadu(kv_mem_ptr + dt_kv_mem_offset + dvi);
            state_vec = state_vec * g_val_exp_vec;
            kv_mem_vec = kv_mem_vec + state_vec * k_val_vec;
            state_vec.store(state_ptr + state_offset + di * v_head_dim + dvi);
            kv_mem_vec.store(kv_mem_ptr + dt_kv_mem_offset + dvi);
          }
        }
        for(; dvi < v_head_dim; ++dvi) {
          for (int di = 0; di < head_dim; ++di) {
            float k_val = k_ptr[k_offset + di];
            state_ptr[state_offset + di * v_head_dim + dvi] *= g_val_exp;
            kv_mem_ptr[dt_kv_mem_offset + dvi] += state_ptr[state_offset + di * v_head_dim + dvi] * k_val;
          }
        }
        for (dvi = 0; dvi <= v_head_dim - VecSize; dvi += VecSize) {
          bVec v_bvec = bVec::loadu(v_ptr + v_offset + dvi);
          fVec v_vec0, v_vec1;
          std::tie(v_vec0, v_vec1) = at::vec::convert_to_float(v_bvec);
          fVec kv_mem_vec0 = fVec::loadu(kv_mem_ptr + dt_kv_mem_offset + dvi);
          fVec kv_mem_vec1 = fVec::loadu(kv_mem_ptr + dt_kv_mem_offset + dvi + fVecSize);
          fVec dt_vec0 = (v_vec0 - kv_mem_vec0) * beta_vec;
          fVec dt_vec1 = (v_vec1 - kv_mem_vec1) * beta_vec;
          bVec o_vec = bVec::loadu(o_ptr + o_offset + dvi);
          fVec o_vec0, o_vec1;
          std::tie(o_vec0, o_vec1) = at::vec::convert_to_float(o_vec);
          for (int di = 0; di < head_dim; ++di) {
            fVec q_vec = fVec(q_ptr[q_offset + di]);
            fVec k_vec = fVec(k_ptr[k_offset + di]);
            fVec state_vec0 = fVec::loadu(state_ptr + state_offset + di * v_head_dim + dvi);
            fVec state_vec1 = fVec::loadu(state_ptr + state_offset + di * v_head_dim + dvi + fVecSize);
            state_vec0 = state_vec0 + k_vec * dt_vec0;
            state_vec1 = state_vec1 + k_vec * dt_vec1;
            o_vec0 = o_vec0 + state_vec0 * q_vec * scale_vec;
            o_vec1 = o_vec1 + state_vec1 * q_vec * scale_vec;
            state_vec0.store(state_ptr + state_offset + di * v_head_dim + dvi);
            state_vec1.store(state_ptr + state_offset + di * v_head_dim + dvi + fVecSize);
          }
          o_vec = at::vec::convert_from_float<scalar_t>(o_vec0, o_vec1);
          o_vec.store(o_ptr + o_offset + dvi);
        }
        for (; dvi < v_head_dim; ++dvi) {
          float v_val = v_ptr[v_offset + dvi];
          float dt_val = (v_val - kv_mem_ptr[dt_kv_mem_offset + dvi]) * beta_val;
          float o_val = o_ptr[o_offset + dvi];
          for (int di = 0; di < head_dim; ++di) {
            float q_val = q_ptr[q_offset + di];
            float k_val = k_ptr[k_offset + di];
            state_ptr[state_offset + di * v_head_dim + dvi] += k_val * dt_val;
            o_val += state_ptr[state_offset + di * v_head_dim + dvi] * q_val * scale;
          }
          o_ptr[o_offset + dvi] = o_val;
        }
      data_index_step(bi, batch_size, si, seq_len, ni, v_num_heads);
    }
  });
}


template <typename scalar_t>
void fused_sigmoid_gating_delta_rule_update_kernel_impl(
    const scalar_t* __restrict__ q_ptr,
    const scalar_t* __restrict__ k_ptr,
    const scalar_t* __restrict__ v_ptr,
    const float* __restrict__ g_ptr,
    const scalar_t* __restrict__ b_ptr,
    const int32_t* __restrict__ indices_ptr,
    float* __restrict__ state_ptr,
    scalar_t* __restrict__ o_ptr,
    float* __restrict__ kv_mem_ptr,
    int64_t seq_len,
    int64_t batch_size,
    int64_t num_heads,
    int64_t head_dim,
    int64_t v_num_heads,
    int64_t v_head_dim,
    int64_t q_strideB,
    int64_t q_strideS,
    int64_t q_strideH,
    int64_t k_strideB,
    int64_t k_strideS,
    int64_t k_strideH,
    int64_t v_strideB,
    int64_t v_strideS,
    int64_t v_strideH) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  constexpr int64_t VecSize = bVec::size();
  constexpr int64_t fVecSize = fVec::size();
  int64_t group_size = v_num_heads / num_heads;
  double scale = 1 / std::sqrt(head_dim);
  fVec scale_vec = fVec(scale);
  at::parallel_for(0, batch_size * seq_len * v_num_heads, 0, [&](int64_t begin, int64_t end) {
    int64_t bi{0}, si{0}, ni{0};
    data_index_init(begin, bi, batch_size, si, seq_len, ni, v_num_heads);
    for (int64_t i = begin; i < end; ++i) {
        int64_t cache_index = indices_ptr[bi];
        int64_t state_offset = (cache_index * v_num_heads + ni) * head_dim * v_head_dim;
        float g_val = g_ptr[ni];
        float g_val_exp = std::exp(g_val);
        fVec g_val_exp_vec = fVec(g_val_exp);
        int64_t q_offset = si * q_strideS + bi * q_strideB + (ni / group_size) * q_strideH;
        int64_t k_offset = si * k_strideS + bi * k_strideB + (ni / group_size) * k_strideH;
        int64_t v_offset = si * v_strideS + bi * v_strideB + ni * v_strideH;
        int64_t o_offset = ((bi * seq_len + si) * v_num_heads + ni) * v_head_dim;
        int64_t dt_kv_mem_offset = ((bi * seq_len + si) * v_num_heads + ni) * v_head_dim;
        float beta_val = 1 / (1 + std::exp(-b_ptr[ni]));
        fVec beta_vec = fVec(beta_val);
        int64_t dvi = 0;
        for (; dvi <= v_head_dim - fVecSize; dvi += fVecSize) {
          for (int di = 0; di < head_dim; ++di) {
            fVec k_val_vec = fVec(k_ptr[k_offset + di]);
            fVec state_vec = fVec::loadu(state_ptr + state_offset + di * v_head_dim + dvi);
            fVec kv_mem_vec = fVec::loadu(kv_mem_ptr + dt_kv_mem_offset + dvi);
            state_vec = state_vec * g_val_exp_vec;
            kv_mem_vec = kv_mem_vec + state_vec * k_val_vec;
            state_vec.store(state_ptr + state_offset + di * v_head_dim + dvi);
            kv_mem_vec.store(kv_mem_ptr + dt_kv_mem_offset + dvi);
          }
        }
        for(; dvi < v_head_dim; ++dvi) {
          for (int di = 0; di < head_dim; ++di) {
            float k_val = k_ptr[k_offset + di];
            state_ptr[state_offset + di * v_head_dim + dvi] *= g_val_exp;
            kv_mem_ptr[dt_kv_mem_offset + dvi] += state_ptr[state_offset + di * v_head_dim + dvi] * k_val;
          }
        }
        for (dvi = 0; dvi <= v_head_dim - VecSize; dvi += VecSize) {
          bVec v_bvec = bVec::loadu(v_ptr + v_offset + dvi);
          fVec v_vec0, v_vec1;
          std::tie(v_vec0, v_vec1) = at::vec::convert_to_float(v_bvec);
          fVec kv_mem_vec0 = fVec::loadu(kv_mem_ptr + dt_kv_mem_offset + dvi);
          fVec kv_mem_vec1 = fVec::loadu(kv_mem_ptr + dt_kv_mem_offset + dvi + fVecSize);
          fVec dt_vec0 = (v_vec0 - kv_mem_vec0) * beta_vec;
          fVec dt_vec1 = (v_vec1 - kv_mem_vec1) * beta_vec;
          bVec o_vec = bVec::loadu(o_ptr + o_offset + dvi);
          fVec o_vec0, o_vec1;
          std::tie(o_vec0, o_vec1) = at::vec::convert_to_float(o_vec);
          for (int di = 0; di < head_dim; ++di) {
            fVec q_vec = fVec(q_ptr[q_offset + di]);
            fVec k_vec = fVec(k_ptr[k_offset + di]);
            fVec state_vec0 = fVec::loadu(state_ptr + state_offset + di * v_head_dim + dvi);
            fVec state_vec1 = fVec::loadu(state_ptr + state_offset + di * v_head_dim + dvi + fVecSize);
            state_vec0 = state_vec0 + k_vec * dt_vec0;
            state_vec1 = state_vec1 + k_vec * dt_vec1;
            o_vec0 = o_vec0 + state_vec0 * q_vec * scale_vec;
            o_vec1 = o_vec1 + state_vec1 * q_vec * scale_vec;
            state_vec0.store(state_ptr + state_offset + di * v_head_dim + dvi);
            state_vec1.store(state_ptr + state_offset + di * v_head_dim + dvi + fVecSize);
          }
          o_vec = at::vec::convert_from_float<scalar_t>(o_vec0, o_vec1);
          o_vec.store(o_ptr + o_offset + dvi);
        }
        for (; dvi < v_head_dim; ++dvi) {
          float v_val = v_ptr[v_offset + dvi];
          float dt_val = (v_val - kv_mem_ptr[dt_kv_mem_offset + dvi]) * beta_val;
          float o_val = o_ptr[o_offset + dvi];
          for (int di = 0; di < head_dim; ++di) {
            float q_val = q_ptr[q_offset + di];
            float k_val = k_ptr[k_offset + di];
            state_ptr[state_offset + di * v_head_dim + dvi] += k_val * dt_val;
            o_val += state_ptr[state_offset + di * v_head_dim + dvi] * q_val * scale;
          }
          o_ptr[o_offset + dvi] = o_val;
        }
      data_index_step(bi, batch_size, si, seq_len, ni, v_num_heads);
    }
  });
}


template <typename scalar_t>
void fused_qkvzba_split_reshape_cat_impl(
  const scalar_t* __restrict__ mixed_qkvz,
  const scalar_t* __restrict__ mixed_ba,
  scalar_t* __restrict__ mixed_qkv,
  scalar_t* __restrict__ z,
  scalar_t* __restrict__ b,
  scalar_t* __restrict__ a,
  int64_t batch,
  int64_t num_heads_qk,
  int64_t num_heads_v,
  int64_t head_qk,
  int64_t group,
  int64_t head_v,
  int64_t qkv_strideB,
  int64_t qkvz_strideB,
  int64_t ba_strideB
) {
  int64_t qkvz_stride_per_head = head_qk * 2 + head_v * 2 * group;
  at::parallel_for(0, batch * num_heads_qk, 0, [&](int64_t begin, int64_t end) {
    int64_t bi{0}, hi{0};
    data_index_init(begin, bi, batch, hi, num_heads_qk);
    for (int64_t i = begin; i < end; ++i) {
      scalar_t* __restrict__ q_out_ptr = mixed_qkv + bi * qkv_strideB + hi * head_qk;
      const scalar_t* __restrict__ q_in_ptr = mixed_qkvz + bi * qkvz_strideB + hi * qkvz_stride_per_head;
      scalar_t* __restrict__ k_out_ptr = q_out_ptr + num_heads_qk * head_qk;
      const scalar_t* __restrict__ k_in_ptr = q_in_ptr + head_qk;
      scalar_t* __restrict__ v_out_ptr = k_out_ptr + num_heads_qk * head_qk + hi * head_qk * (group - 1);
      const scalar_t* __restrict__ v_in_ptr = k_in_ptr + head_qk;
      scalar_t* __restrict__ z_out_ptr = z + bi * num_heads_v * head_v + hi * group * head_v;
      const scalar_t* __restrict__ z_in_ptr = v_in_ptr + head_qk * group;
      copy_stub(q_out_ptr, q_in_ptr, head_qk);
      copy_stub(k_out_ptr, k_in_ptr, head_qk);
      copy_stub(v_out_ptr, v_in_ptr, head_qk * group);
      copy_stub(z_out_ptr, z_in_ptr, head_qk * group);
      scalar_t* __restrict__ b_out_ptr = b + bi * num_heads_v + hi * group;
      const scalar_t* __restrict__ b_in_ptr = mixed_ba + bi * ba_strideB + hi * group * 2;
      scalar_t* __restrict__ a_out_ptr = a + bi * num_heads_v + hi * group;
      const scalar_t* __restrict__ a_in_ptr = b_in_ptr + group;
      copy_stub(b_out_ptr, b_in_ptr, group);
      copy_stub(a_out_ptr, a_in_ptr, group);
      data_index_step(bi, batch, hi, num_heads_qk);
  }
  });
}


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
}
}  // anonymous namespace

extern at::Tensor qwen3_next_l2norm_cpu(at::Tensor& input, double eps);


// A_log: [num_v_heads]
// a: [batch, num_v_heads]
// dt_bias: [num_v_heads]
// -A_log.float().exp() * F.softplus(a.float() + dt_bias)
at::Tensor fused_gdn_gating_cpu(const at::Tensor& A_log, const at::Tensor& a, const at::Tensor& dt_bias) {
  RECORD_FUNCTION("sgl-kernel::fused_gdn_gating_cpu", std::vector<c10::IValue>({A_log, a, dt_bias}));
  CHECK_DIM(1, A_log);
  CHECK_DIM(2, a);
  CHECK_DIM(1, dt_bias);
  CHECK_CONTIGUOUS(a);
  CHECK_EQ(A_log.size(0), a.size(1));
  CHECK_EQ(A_log.size(0), dt_bias.size(0));
  int batch = a.size(0);
  int num_heads = a.size(1);
  at::Tensor out = at::empty_like(a, a.options().dtype(at::kFloat));
  AT_DISPATCH_REDUCED_FLOATING_TYPES(a.scalar_type(), "fused_gdn_gating_kernel", [&] {
    fused_gdn_gating_kernel_impl<scalar_t>(
        A_log.data_ptr<float>(),
        a.data_ptr<scalar_t>(),
        dt_bias.data_ptr<scalar_t>(),
        out.data_ptr<float>(),
        batch,
        num_heads);
  });
  return out;
}


at::Tensor causal_conv1d_update_cpu(
    const at::Tensor& hidden_states,
    at::Tensor& conv_states,
    const at::Tensor& cache_indices,
    const at::Tensor& conv_weights,
    const c10::optional<at::Tensor>& conv_bias,
    bool silu_activation,
    const c10::optional<at::Tensor>& cache_seqlens) {
  RECORD_FUNCTION("sgl-kernel::causal_conv1d_update_cpu", std::vector<c10::IValue>({hidden_states, conv_states, conv_weights}));
  if (hidden_states.scalar_type() == at::ScalarType::Float) {
    return causal_conv1d_update_kernel_inner<float>(
        hidden_states,
        conv_states,
        cache_indices,
        conv_weights,
        conv_bias,
        silu_activation,
        cache_seqlens);
  } else if (hidden_states.scalar_type() == at::ScalarType::BFloat16) {
    return causal_conv1d_update_kernel_inner<at::BFloat16>(
        hidden_states,
        conv_states,
        cache_indices,
        conv_weights,
        conv_bias,
        silu_activation,
        cache_seqlens);
  } else if (hidden_states.scalar_type() == at::ScalarType::Half) {
    return causal_conv1d_update_kernel_inner<at::Half>(
        hidden_states,
        conv_states,
        cache_indices,
        conv_weights,
        conv_bias,
        silu_activation,
        cache_seqlens);
  } else {
    TORCH_CHECK(
        false,
        "Only support bfloat16, float16 and float for causal_conv1d_update");
  }
}

std::tuple<at::Tensor, at::Tensor> causal_conv1d_fn_cpu(
    const at::Tensor& x,
    const at::Tensor& conv_weights,
    const c10::optional<at::Tensor>& conv_bias,
    const c10::optional<at::Tensor>& initial_states,
    const c10::optional<at::Tensor>& final_states_out,
    bool silu_activation) {
  RECORD_FUNCTION("sgl-kernel::causal_conv1d_fn_cpu", std::vector<c10::IValue>({x, conv_weights}));
  auto final_states_out_ = final_states_out.has_value()
      ? final_states_out.value()
      : at::empty(
            {x.size(0), x.size(1), conv_weights.size(-1) - 1}, x.options());
  if (x.scalar_type() == at::ScalarType::Float) {
    return causal_conv1d_fn_kernel_inner<float>(
        x,
        conv_weights,
        conv_bias,
        initial_states,
        final_states_out_,
        silu_activation);
  } else if (x.scalar_type() == at::ScalarType::BFloat16) {
    return causal_conv1d_fn_kernel_inner<at::BFloat16>(
        x,
        conv_weights,
        conv_bias,
        initial_states,
        final_states_out_,
        silu_activation);
  } else if (x.scalar_type() == at::ScalarType::Half) {
    return causal_conv1d_fn_kernel_inner<at::Half>(
        x,
        conv_weights,
        conv_bias,
        initial_states,
        final_states_out_,
        silu_activation);
  } else {
    TORCH_CHECK(
        false, "Only support bfloat16, float16 and float for causal_conv1d_fn");
  }
}

// query: [seq_len, batch_size, num_heads, head_dim]
// key: [seq_len, batch_size, num_heads, head_dim]
// value: [seq_len, batch_size, v_num_heads, v_head_dim]
// g: [batch_size, v_num_heads]
// beta: [batch_size, v_num_heads]
// cache_indices: [batch_size]
// initial_state:[num_tokens, v_num_heads, head_dim, v_head_dim]
at::Tensor fused_recurrent_gated_delta_rule_cpu(
  const at::Tensor& query,
  const at::Tensor& key,
  const at::Tensor& value,
  const at::Tensor& g,
  const at::Tensor& beta,
  const at::Tensor& cache_indices,
  at::Tensor& initial_state,
  bool use_qk_l2norm_in_kernel
) {
  RECORD_FUNCTION("sgl-kernel::fused_recurrent_gated_delta_rule_cpu", std::vector<c10::IValue>({query, key, value, g, beta, initial_state}));
  CHECK_DIM(4, query);
  CHECK_DIM(4, key);
  CHECK_DIM(4, value);
  CHECK_DIM(2, g);
  CHECK_DIM(2, beta);
  CHECK_DIM(4, initial_state);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(query);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(key);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(value);
  CHECK_CONTIGUOUS(g);
  CHECK_CONTIGUOUS(beta);
  CHECK_CONTIGUOUS(initial_state);
  int64_t seq_len = query.size(0);
  int64_t batch_size = query.size(1);
  int64_t num_heads = query.size(2);
  int64_t head_dim = query.size(3);
  int64_t v_num_heads = value.size(2);
  int64_t v_head_dim = value.size(3);
  CHECK_EQ(key.size(0), seq_len);
  CHECK_EQ(key.size(1), batch_size);
  CHECK_EQ(key.size(2), num_heads);
  CHECK_EQ(key.size(3), head_dim);
  CHECK_EQ(value.size(0), seq_len);
  CHECK_EQ(value.size(1), batch_size);
  CHECK_EQ(value.size(2), v_num_heads);
  CHECK_EQ(value.size(3), v_head_dim);
  CHECK_EQ(g.size(0), batch_size);
  CHECK_EQ(g.size(1), v_num_heads);
  CHECK_EQ(beta.size(0), batch_size);
  CHECK_EQ(beta.size(1), v_num_heads);
  CHECK_EQ(cache_indices.size(0), batch_size);
  CHECK(initial_state.size(0) >= batch_size);
  CHECK_EQ(initial_state.size(1), v_num_heads);
  CHECK_EQ(initial_state.size(2), head_dim);
  CHECK_EQ(initial_state.size(3), v_head_dim);
  CHECK_EQ(v_num_heads % num_heads, 0);

  at::Tensor core_attn_out = at::zeros({batch_size, seq_len, v_num_heads, v_head_dim}, at::kBFloat16);
  at::Tensor kv_mem = at::zeros({batch_size, seq_len, v_num_heads, v_head_dim}, at::kFloat);
  at::Tensor query_ = query;
  at::Tensor key_ = key;
  if (use_qk_l2norm_in_kernel) {
    query_ = qwen3_next_l2norm_cpu(query_, 1e-6);
    key_ = qwen3_next_l2norm_cpu(key_, 1e-6);
  }
  int64_t q_strideS = query_.stride(0);
  int64_t q_strideB = query_.stride(1);
  int64_t q_strideH = query_.stride(2);
  int64_t k_strideS = key_.stride(0);
  int64_t k_strideB = key_.stride(1);
  int64_t k_strideH = key_.stride(2);
  int64_t v_strideS = value.stride(0);
  int64_t v_strideB = value.stride(1);
  int64_t v_strideH = value.stride(2);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(query.scalar_type(), "fused_recurrent_gated_delta_rule_kernel_impl", [&] {
    fused_recurrent_gated_delta_rule_kernel_impl<scalar_t>(
        query_.data_ptr<scalar_t>(),
        key_.data_ptr<scalar_t>(),
        value.data_ptr<scalar_t>(),
        g.data_ptr<float>(),
        beta.data_ptr<scalar_t>(),
        cache_indices.data_ptr<int32_t>(),
        initial_state.data_ptr<float>(),
        core_attn_out.data_ptr<scalar_t>(),
        kv_mem.data_ptr<float>(),
        seq_len,
        batch_size,
        num_heads,
        head_dim,
        v_num_heads,
        v_head_dim,
        q_strideB,
        q_strideS,
        q_strideH,
        k_strideB,
        k_strideS,
        k_strideH,
        v_strideB,
        v_strideS,
        v_strideH);
  });
  return core_attn_out;
}



// query: [seq_len, batch_size, num_heads, head_dim]
// key: [seq_len, batch_size, num_heads, head_dim]
// value: [seq_len, batch_size, v_num_heads, v_head_dim]
// A_log: [v_num_heads]
// a: [batch_size, v_num_heads]
// dt_bias: [v_num_heads]
// b: [batch_size, v_num_heads]
// cache_indices: [batch_size]
// initial_state:[num_tokens, v_num_heads, head_dim, v_head_dim]
at::Tensor fused_sigmoid_gating_delta_rule_update_cpu(
  const at::Tensor& query,
  const at::Tensor& key,
  const at::Tensor& value,
  const at::Tensor& A_log,
  const at::Tensor& a,
  const at::Tensor& dt_bias,
  const at::Tensor& b,
  const at::Tensor& cache_indices,
  at::Tensor& initial_state,
  bool use_qk_l2norm_in_kernel
) {
  RECORD_FUNCTION("sgl-kernel::fused_sigmoid_gating_delta_rule_update_cpu", std::vector<c10::IValue>({query, key, value, A_log, a, dt_bias, b, initial_state}));
  CHECK_DIM(4, query);
  CHECK_DIM(4, key);
  CHECK_DIM(4, value);
  CHECK_DIM(1, A_log);
  CHECK_DIM(2, a);
  CHECK_DIM(1, dt_bias);
  CHECK_DIM(2, b);
  CHECK_DIM(4, initial_state);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(query);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(key);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(value);
  CHECK_CONTIGUOUS(a);
  CHECK_CONTIGUOUS(b);
  CHECK_CONTIGUOUS(initial_state);
  int64_t seq_len = query.size(0);
  int64_t batch_size = query.size(1);
  int64_t num_heads = query.size(2);
  int64_t head_dim = query.size(3);
  int64_t v_num_heads = value.size(2);
  int64_t v_head_dim = value.size(3);
  CHECK_EQ(key.size(0), seq_len);
  CHECK_EQ(key.size(1), batch_size);
  CHECK_EQ(key.size(2), num_heads);
  CHECK_EQ(key.size(3), head_dim);
  CHECK_EQ(value.size(0), seq_len);
  CHECK_EQ(value.size(1), batch_size);
  CHECK_EQ(value.size(2), v_num_heads);
  CHECK_EQ(value.size(3), v_head_dim);
  CHECK_EQ(a.size(0), batch_size);
  CHECK_EQ(a.size(1), v_num_heads);
  CHECK_EQ(dt_bias.size(0), v_num_heads);
  CHECK_EQ(b.size(0), batch_size);
  CHECK_EQ(b.size(1), v_num_heads);
  CHECK_EQ(A_log.size(0), v_num_heads);
  CHECK_EQ(cache_indices.size(0), batch_size);
  CHECK(initial_state.size(0) >= batch_size);
  CHECK_EQ(initial_state.size(1), v_num_heads);
  CHECK_EQ(initial_state.size(2), head_dim);
  CHECK_EQ(initial_state.size(3), v_head_dim);
  CHECK_EQ(v_num_heads % num_heads, 0);

  at::Tensor core_attn_out = at::zeros({batch_size, seq_len, v_num_heads, v_head_dim}, at::kBFloat16);
  at::Tensor kv_mem = at::zeros({batch_size, seq_len, v_num_heads, v_head_dim}, at::kFloat);
  at::Tensor query_ = query;
  at::Tensor key_ = key;
  if (use_qk_l2norm_in_kernel) {
    query_ = qwen3_next_l2norm_cpu(query_, 1e-6);
    key_ = qwen3_next_l2norm_cpu(key_, 1e-6);
  }
  at::Tensor g = fused_gdn_gating_cpu(A_log, a, dt_bias);
  int64_t q_strideB = query_.stride(1);
  int64_t q_strideS = query_.stride(0);
  int64_t q_strideH = query_.stride(2);
  int64_t k_strideB = key_.stride(1);
  int64_t k_strideS = key_.stride(0);
  int64_t k_strideH = key_.stride(2);
  int64_t v_strideB = value.stride(1);
  int64_t v_strideS = value.stride(0);
  int64_t v_strideH = value.stride(2);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(query.scalar_type(), "fused_sigmoid_gating_delta_rule_update_kernel_impl", [&] {
    fused_sigmoid_gating_delta_rule_update_kernel_impl<scalar_t>(
        query_.data_ptr<scalar_t>(),
        key_.data_ptr<scalar_t>(),
        value.data_ptr<scalar_t>(),
        g.data_ptr<float>(),
        b.data_ptr<scalar_t>(),
        cache_indices.data_ptr<int32_t>(),
        initial_state.data_ptr<float>(),
        core_attn_out.data_ptr<scalar_t>(),
        kv_mem.data_ptr<float>(),
        seq_len,
        batch_size,
        num_heads,
        head_dim,
        v_num_heads,
        v_head_dim,
        q_strideB,
        q_strideS,
        q_strideH,
        k_strideB,
        k_strideS,
        k_strideH,
        v_strideB,
        v_strideS,
        v_strideH);
  });
  return core_attn_out;
}


// mixed_qkvz: [batch, num_heads_qk * head_qk * 2 + num_heads_v * head_v * 2]
// mixed_ba: [batch, num_heads_v * 2]
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fused_qkvzba_split_reshape_cat_cpu(
  const at::Tensor& mixed_qkvz,
  const at::Tensor& mixed_ba,
  int64_t num_heads_qk,
  int64_t num_heads_v,
  int64_t head_qk,
  int64_t head_v
) {
  RECORD_FUNCTION("sgl-kernel::fused_qkvzba_split_reshape_cat_cpu", std::vector<c10::IValue>({mixed_qkvz, mixed_ba}));
  CHECK_DIM(2, mixed_qkvz);
  CHECK_DIM(2, mixed_ba);
  CHECK_INPUT(mixed_qkvz);
  CHECK_INPUT(mixed_ba);
  int64_t batch = mixed_qkvz.size(0);
  int64_t qkv_dim = num_heads_qk * head_qk * 2 + num_heads_v * head_v;
  int64_t ba_dim = num_heads_v * 2;
  int64_t expected_dim = qkv_dim + num_heads_v * head_v;
  CHECK_EQ(mixed_qkvz.size(1), expected_dim);
  CHECK_EQ(mixed_ba.size(0), batch);
  CHECK_EQ(mixed_ba.size(1), ba_dim);
  CHECK_EQ(num_heads_v % num_heads_qk, 0);
  at::Tensor mixed_qkv = at::empty({batch, qkv_dim}, mixed_qkvz.options());
  at::Tensor z = at::empty({batch, num_heads_v, head_v}, mixed_qkvz.options());
  at::Tensor b = at::empty({batch, num_heads_v}, mixed_ba.options());
  at::Tensor a = at::empty({batch, num_heads_v}, mixed_ba.options());
  int64_t group = num_heads_v / num_heads_qk;
  int64_t qkvz_strideB = mixed_qkvz.size(1);
  int64_t qkv_strideB = mixed_qkv.size(1);
  int64_t ba_strideB = mixed_ba.size(1);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(mixed_qkvz.scalar_type(), "fused_qkvzba_split_reshape_cat_impl", [&] {
    fused_qkvzba_split_reshape_cat_impl<scalar_t>(
      mixed_qkvz.data_ptr<scalar_t>(),
      mixed_ba.data_ptr<scalar_t>(),
      mixed_qkv.data_ptr<scalar_t>(),
      z.data_ptr<scalar_t>(),
      b.data_ptr<scalar_t>(),
      a.data_ptr<scalar_t>(),
      batch,
      num_heads_qk,
      num_heads_v,
      head_qk,
      group,
      head_v,
      qkv_strideB,
      qkvz_strideB,
      ba_strideB);
    });
  return std::make_tuple(mixed_qkv, z, b, a);
}


// query: [B, T, HK, EK]
// key: [B, T, HK, EK]
// value: [B, T, HV, EV]
// g: [B, T, HV] FP32
// beta: [B, T, HV]
// cu_seqlens: [N + 1] INT32
// initial_state: [N, HV, EK, EV]
// use_qk_l2norm_in_kernel: bool
std::tuple<at::Tensor, at::Tensor> chunk_gated_delta_rule_cpu(
        at::Tensor& query,
        at::Tensor& key,
        at::Tensor& value,
        at::Tensor& g,
        at::Tensor& beta,
        at::Tensor& cu_seqlens,
        at::Tensor& initial_state,
        bool use_qk_l2norm_in_kernel) {
    RECORD_FUNCTION("sgl-kernel::chunk_gated_delta_rule_cpu", std::vector<c10::IValue>({query, key, value, g, beta, initial_state}));
    CHECK_DIM(4, query);
    CHECK_DIM(4, key);
    CHECK_DIM(4, value);
    CHECK_DIM(3, g);
    CHECK_DIM(3, beta);
    CHECK_DIM(4, initial_state);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(query);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(key);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(value);
    CHECK_CONTIGUOUS(g);
    CHECK_CONTIGUOUS(beta);
    CHECK_CONTIGUOUS(initial_state);
    int64_t B = query.size(0);
    int64_t T = query.size(1);
    int64_t HK = query.size(2);
    int64_t EK = query.size(3);
    int64_t HV = value.size(2);
    int64_t EV = value.size(3);
    CHECK_EQ(B, 1);
    CHECK_EQ(key.size(0), B);
    CHECK_EQ(key.size(1), T);
    CHECK_EQ(key.size(2), HK);
    CHECK_EQ(key.size(3), EK);
    CHECK_EQ(value.size(0), B);
    CHECK_EQ(value.size(1), T);
    CHECK_EQ(g.size(0), B);
    CHECK_EQ(g.size(1), T);
    CHECK_EQ(g.size(2), HV);
    CHECK_EQ(beta.size(0), B);
    CHECK_EQ(beta.size(1), T);
    CHECK_EQ(beta.size(2), HV);
    CHECK_EQ(initial_state.size(1), HV);
    CHECK_EQ(initial_state.size(2), EK);
    CHECK_EQ(initial_state.size(3), EV);
    CHECK_EQ(HV % HK, 0);

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
    return std::make_tuple(std::move(output), std::move(final_state));
}
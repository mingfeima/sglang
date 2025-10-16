#include "common.h"
#include "vec.h"

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
std::tuple<at::Tensor, at::Tensor> causal_conv1d_update_kernel_inner(
    const at::Tensor& hidden_states,
    const at::Tensor& conv_states,
    const at::Tensor& conv_weights,
    const c10::optional<at::Tensor>& conv_bias,
    bool silu_activation,
    const c10::optional<at::Tensor>& cache_seqlens) {
  auto bs = conv_states.size(0);
  auto channels = conv_states.size(1);
  auto kernel_size = conv_weights.size(1);
  auto state_len = conv_states.size(2);
  auto seqlen = hidden_states.dim() == 3 ? hidden_states.size(2) : 1;
  auto has_bias = conv_bias.has_value();
  auto bias_ptr = has_bias ? conv_bias.value().data_ptr<T>() : nullptr;
  auto conv_states_ptr = conv_states.data_ptr<T>();
  auto conv_weights_ptr = conv_weights.data_ptr<T>();
  auto hidden_states_ptr = hidden_states.data_ptr<T>();
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
            bi * conv_states_strideB + ci * conv_states_strideC;
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
            bi * conv_states_strideB + ci * conv_states_strideC;
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
  return std::make_tuple(std::move(res), std::move(conv_states));
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


std::tuple<at::Tensor, at::Tensor> causal_conv1d_update_cpu(
    const at::Tensor& hidden_states,
    const at::Tensor& conv_states,
    const at::Tensor& conv_weights,
    const c10::optional<at::Tensor>& conv_bias,
    bool silu_activation,
    const c10::optional<at::Tensor>& cache_seqlens) {
  RECORD_FUNCTION("sgl-kernel::causal_conv1d_update_cpu", std::vector<c10::IValue>({hidden_states, conv_states, conv_weights}));
  if (hidden_states.scalar_type() == at::ScalarType::Float) {
    return causal_conv1d_update_kernel_inner<float>(
        hidden_states,
        conv_states,
        conv_weights,
        conv_bias,
        silu_activation,
        cache_seqlens);
  } else if (hidden_states.scalar_type() == at::ScalarType::BFloat16) {
    return causal_conv1d_update_kernel_inner<at::BFloat16>(
        hidden_states,
        conv_states,
        conv_weights,
        conv_bias,
        silu_activation,
        cache_seqlens);
  } else if (hidden_states.scalar_type() == at::ScalarType::Half) {
    return causal_conv1d_update_kernel_inner<at::Half>(
        hidden_states,
        conv_states,
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
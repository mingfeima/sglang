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
}  // anonymous namespace

// A_log: [num_v_heads]
// a: [batch, num_v_heads]
// dt_bias: [num_v_heads]
// -A_log.float().exp() * F.softplus(a.float() + dt_bias)
at::Tensor fused_gdn_gating_cpu(at::Tensor& A_log, at::Tensor& a, at::Tensor& dt_bias) {
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
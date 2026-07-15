#pragma once
#include "vec.h"

template <typename scalar_t>
inline void fill_stub(scalar_t* __restrict__ out, scalar_t val, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  const Vec data_vec(val);
  at::vec::map<scalar_t>([data_vec](Vec out) { return out = data_vec; }, out, out, size);
}

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, int64_t size) {
  using Vec = at::vec::Vectorized<scalar_t>;
  constexpr int kVecSize = Vec::size();
  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    Vec data = Vec::loadu(input + d);
    data.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = input[d];
  }
}

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ input, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    auto [x0, x1] = load_float_vec2(input + d);
    bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d]);
  }
}

template <>
inline void copy_stub<uint8_t>(uint8_t* __restrict__ out, const uint8_t* __restrict__ input, int64_t size) {
  // size might be 64x + 32
  std::memcpy(out, input, size * sizeof(uint8_t));
}

template <typename scalar_t, typename input_t>
inline void copy_mul_stub(scalar_t* __restrict__ out, const input_t* __restrict__ input, float weight, int64_t size) {
  static_assert(
      std::is_same_v<input_t, float> || std::is_same_v<input_t, scalar_t>,
      "copy_mul_stub only supports input_t == float or input_t == scalar_t");
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec weight_vec = fVec(weight);
  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    auto [x0, x1] = load_float_vec2(input + d);
    x0 = x0 * weight_vec;
    x1 = x1 * weight_vec;
    bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] * weight);
  }
}

// acc from [topk, K] to [K]
template <typename scalar_t>
inline void sum_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, int64_t topk, int64_t K) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  if (topk == 1) {
    // do copy for topk = 1
    copy_stub(out, input, K);
  } else {
    // do sum for topk != 1
    int64_t d;
#pragma GCC unroll 4
    for (d = 0; d <= K - kVecSize; d += kVecSize) {
      fVec sum_fvec0 = fVec(0.f);
      fVec sum_fvec1 = fVec(0.f);
      for (int t = 0; t < topk; ++t) {
        bVec x_bvec = bVec::loadu(input + t * K + d);
        fVec x_fvec0, x_fvec1;
        std::tie(x_fvec0, x_fvec1) = at::vec::convert_to_float(x_bvec);

        sum_fvec0 += x_fvec0;
        sum_fvec1 += x_fvec1;
      }
      bVec out_bvec = convert_from_float_ext<scalar_t>(sum_fvec0, sum_fvec1);
      out_bvec.store(out + d);
    }
    for (; d < K; ++d) {
      float sum_val = 0.f;
      for (int t = 0; t < topk; ++t) {
        sum_val += static_cast<float>(input[t * K + d]);
      }
      out[d] = static_cast<scalar_t>(sum_val);
    }
  }
}

// out = input + input2 * scale
template <typename scalar_t, typename input_t>
inline void add_mul_stub(
    scalar_t* __restrict__ out,
    const input_t* __restrict__ input,
    const scalar_t* __restrict__ input2,
    float scale,
    int64_t size) {
  static_assert(
      std::is_same_v<input_t, float> || std::is_same_v<input_t, scalar_t>,
      "add_mul_stub only supports input_t == float or input_t == scalar_t");

  // out = input (without scale factor)
  if (input2 == nullptr) {
    copy_stub(out, input, size);
    return;
  }

  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec s_vec = fVec(scale);
  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    auto [x0, x1] = load_float_vec2(input + d);

    bVec y_bvec = bVec::loadu(input2 + d);
    fVec y0, y1;
    std::tie(y0, y1) = at::vec::convert_to_float(y_bvec);

    x0 = x0 + y0 * s_vec;
    x1 = x1 + y1 * s_vec;
    bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] + float(input2[d]) * scale);
  }
}

template <typename scalar_t>
inline void silu_and_mul_stub(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ input, const scalar_t* __restrict__ input2, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;

  // no remainder
#pragma GCC unroll 4
  for (int64_t d = 0; d < size; d += bVec::size()) {
    bVec x = bVec::loadu(input + d);
    fVec x0, x1;
    std::tie(x0, x1) = at::vec::convert_to_float(x);
    bVec y = bVec::loadu(input2 + d);
    fVec y0, y1;
    std::tie(y0, y1) = at::vec::convert_to_float(y);
    x0 = fast_silu(x0) * y0;
    x1 = fast_silu(x1) * y1;
    bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
    out_vec.store(out + d);
  }
}

template <typename scalar_t>
inline void copy_mul_stub(scalar_t* __restrict__ out, const float* __restrict__ input, float weight, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec weight_vec = fVec(weight);
  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input + d) * weight_vec;
    fVec data1 = fVec::loadu(input + d + fVec::size()) * weight_vec;
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] * weight);
  }
}

// input = input + input2
inline void add_bias_stub(float* __restrict__ input, const float* __restrict__ input2, int64_t size) {
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = fVec::size();
  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec x_fvec = fVec::loadu(input + d);
    fVec y_fvec = fVec::loadu(input2 + d);
    x_fvec = x_fvec + y_fvec;
    x_fvec.store(input + d);
  }
  for (; d < size; ++d) {
    input[d] = input[d] + input2[d];
  }
}

template <typename scalar_t>
inline void copy_mul_stub(scalar_t* __restrict__ out, const scalar_t* __restrict__ input, float weight, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();
  const fVec weight_vec = fVec(weight);
  int64_t d;
#pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    bVec x = bVec::loadu(input + d);
    fVec x0, x1;
    std::tie(x0, x1) = at::vec::convert_to_float(x);
    x0 = x0 * weight_vec;
    x1 = x1 * weight_vec;
    bVec out_vec = convert_from_float_ext<scalar_t>(x0, x1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] * weight);
  }
}

#if defined(CPU_CAPABILITY_AVX512)
template <>
inline void clamp_sigmoid_and_mul_stub<at::BFloat16>(
    at::BFloat16* __restrict__ out,
    const at::BFloat16* __restrict__ input,
    int64_t size,
    const float alpha,
    const float limit) {
  const __m512 vlimit = _mm512_set1_ps(limit);
  const __m512 vnlimit = _mm512_set1_ps(-limit);
  const __m512 valpha = _mm512_set1_ps(alpha);
  const __m512 vone = _mm512_set1_ps(1.f);

  auto process = [&](__m512 v) -> __m256i {
    __m512 glu, lin;
    _mm512_deinterleave_ps(v, glu, lin);
    glu = _mm512_min_ps(glu, vlimit);
    lin = _mm512_min_ps(vlimit, _mm512_max_ps(vnlimit, lin));
    __m512 result = _mm512_mul_ps(_mm512_rcp14_sigmoid_glu_ps(glu, valpha), _mm512_add_ps(lin, vone));
    return _mm512_cvtneps_pbh(result);
  };

#pragma GCC unroll 4
  for (int64_t d = 0; d < 2 * size; d += 32) {
    __m512i bx = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(input + d));
    __m256i o0 = process(CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(bx, 0)));
    __m256i o1 = process(CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32(bx, 1)));
    _mm512_storeu_si512(
        reinterpret_cast<__m512i*>(out + d / 2),
        _mm512_inserti32x8(_mm512_castsi256_si512(o0), o1, 1));
  }
}
#endif

template <typename scalar_t>
inline void clamp_sigmoid_and_mul_stub(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    int64_t size,
    const float alpha,
    const float limit) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  const fVec one = fVec(1.f);
  const fVec limit_v = fVec(limit);
  const fVec nlimit_v = fVec(-limit);
  const fVec alpha_v = fVec(alpha);

#pragma GCC unroll 4
  for (int64_t d = 0; d < 2 * size; d += bVec::size()) {
    bVec x = bVec::loadu(input + d);
    fVec x0_, y0_;
    std::tie(x0_, y0_) = at::vec::convert_to_float(x);
    float tmp_buffer[fVec::size() * 2];
    float tmp_glu[fVec::size()];
    float tmp_linear[fVec::size()];
    x0_.store(tmp_buffer);
    y0_.store(tmp_buffer + fVec::size());
#pragma GCC unroll 4
    for (int j = 0; j < fVec::size(); ++j) {
      tmp_glu[j] = tmp_buffer[j * 2];
      tmp_linear[j] = tmp_buffer[j * 2 + 1];
    }
    fVec x0 = fVec::loadu(tmp_glu);
    fVec y0 = fVec::loadu(tmp_linear);

    x0 = at::vec::minimum(x0, limit_v);
    y0 = at::vec::minimum(limit_v, at::vec::maximum(nlimit_v, y0));
    x0 = fast_sigmoid_glu(x0, alpha_v) * (y0 + one);
    convert_from_float_and_store<scalar_t>(out + d / 2, x0);
  }
}

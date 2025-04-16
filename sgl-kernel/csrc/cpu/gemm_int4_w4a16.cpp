#include "common.h"
#include "vec.h"
#include "gemm.h"

namespace {

template <typename scalar_t, bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn {
  static inline void apply(
    const scalar_t* __restrict__ A, const uint8_t* __restrict__ B, scalar_t* __restrict__ C,
    const uint8_t* __restrict__ Bz, const scalar_t* __restrict__ Bs,
    const float* __restrict__ bias, int64_t K, int group_size, int64_t lda, int64_t ldb, int64_t ldc,
    int64_t strideBz, int64_t strideBs) {
  TORCH_CHECK(false, "tinygemm_kernel_nn: scalar path not implemented!");
  }
};

// convert packed 8-bit integers to packed 32-bit integers
inline __m512 CVT_INT8_TO_FP32(__m128i x) {
  return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(x));
}

#if defined(CPU_CAPABILITY_AVX512)

// 0-15 in bf16 bits
const uint16_t BF16_LUT[] = {0x0000, 0x3F80, 0x4000, 0x4040,
                             0x4080, 0x40A0, 0x40C0, 0x40E0,
                             0x4100, 0x4110, 0x4120, 0x4130,
                             0x4140, 0x4150, 0x4160, 0x4170};

inline uint32_t f32_as_u32(float x) {
  uint32_t tmp;
  std::memcpy(&tmp, &x, sizeof(tmp));
  return tmp;
}
inline float u32_as_f32(uint32_t x) {
  float tmp;
  std::memcpy(&tmp, &x, sizeof(tmp));
  return tmp;
}

template <bool has_bias, int BLOCK_M, int BLOCK_N>
struct tinygemm_kernel_nn<at::BFloat16, has_bias, BLOCK_M, BLOCK_N> {
  static inline void apply(
    const at::BFloat16* __restrict__ A, const uint8_t* __restrict__ B, at::BFloat16* __restrict__ C,
    const uint8_t* __restrict__ Bz, const at::BFloat16* __restrict__ Bs,
    const float* __restrict__ bias, int64_t K, int group_size, int64_t lda, int64_t ldb, int64_t ldc,
    int64_t strideBz, int64_t strideBs) {

    static_assert(BLOCK_N % 32 == 0);
    constexpr int ROWS = BLOCK_M;
    constexpr int COLS = BLOCK_N / 16;

    // prefetch distance
    constexpr int PREFETCH_SIZE_K = 0;

    __m512bh va;
    __m512bh vb[COLS];
    __m512 vc[ROWS * COLS];
    __m512 vc_master[ROWS * COLS];

    __m256i mask = _mm256_set1_epi8(0xF);  // lower 4 bit
    __m512i bf16_lut = _mm512_castsi256_si512(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(BF16_LUT)));
    __m512 va_sum;
    __m512 scale[COLS];
    __m512 zero_scale[COLS];
    float A_sum[ROWS] = {};

    const int64_t K2 = K >> 1;
    const int64_t lda2 = lda >> 1;
    const int64_t ldb2 = ldb; // ldb * 2 >> 1;
    const int64_t gs2 = group_size >> 1;
    const float* a_ptr = reinterpret_cast<const float*>(A);
    const uint8_t* b_ptr = B;

    auto loadc = [&](auto i) {
      constexpr int col = i % COLS;
      if constexpr (has_bias) {
        vc_master[i] = _mm512_loadu_ps(bias + col * 16);
      } else {
        vc_master[i] = _mm512_set1_ps(0.f);
      }
    };
    Unroll<ROWS * COLS>{}(loadc);

    auto compute = [&](auto i, int64_t k) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      if constexpr (col == 0) {
        va = (__m512bh)(_mm512_set1_ps(a_ptr[row * lda2 + k]));
      }
      if constexpr (row == 0 && col % 2 == 0) {
        __m256i vb_u4 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(b_ptr + k * ldb + col * 16));

        // deinterleave and lookup to BF16
        __m256i vb_u8_lo = vb_u4 & mask;
        __m256i vb_u8_hi = _mm256_srli_epi16(vb_u4, 4) & mask;
        vb[col]   = (__m512bh)_mm512_permutexvar_epi16(_mm512_cvtepi8_epi16(vb_u8_lo), bf16_lut);
        vb[col+1] = (__m512bh)_mm512_permutexvar_epi16(_mm512_cvtepi8_epi16(vb_u8_hi), bf16_lut);

        if constexpr (PREFETCH_SIZE_K > 0) {
          _mm_prefetch(b_ptr + (k + PREFETCH_SIZE_K) * ldb2 + col * 16, _MM_HINT_T0);
        }
      }
      vc[i] = _mm512_dpbf16_ps(vc[i], va, vb[col]);
    };
    auto accumulate_group = [&](auto i, int64_t kgs) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;

      // (x * (w - zero) * scale).sum(-1)
      // (x * (w - zero)).sum(-1) * scale
      // (x * w).sum(-1) * scale - x.sum(-1) * zero * scale

      if constexpr (row == 0 && col % 2 == 0) {
        // Bs layout: [K/gs, BLOCK_N] : [strideBs, 1], dtype=bf16
        __m512i tmp1 = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(Bs + kgs * strideBs + col * 16));
        scale[col]   = _mm512_cvtpbh_ps((__m256bh)_mm512_extracti32x8_epi32(tmp1, 0));
        scale[col+1] = _mm512_cvtpbh_ps((__m256bh)_mm512_extracti32x8_epi32(tmp1, 1));

        // Bz layout: [K/gs, BLOCK_N] : [strideBs, 1], dtype=uint8
        __m256i tmp2 = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(Bz + kgs * strideBz + col * 16));
        zero_scale[col]   = CVT_INT8_TO_FP32(_mm256_extracti32x4_epi32(tmp2, 0)) * scale[col];
        zero_scale[col+1] = CVT_INT8_TO_FP32(_mm256_extracti32x4_epi32(tmp2, 1)) * scale[col+1];
      }
      if constexpr (col == 0) {
        va_sum = _mm512_set1_ps(A_sum[row]);
      }

      vc_master[i] = _mm512_fmadd_ps(vc[i], scale[col], vc_master[i]);
      vc_master[i] = _mm512_fnmadd_ps(va_sum, zero_scale[col], vc_master[i]);
    };
    for (int64_t k = 0; k < K2; k += gs2) {
      Unroll<ROWS * COLS>{}([&](auto i) { vc[i] = _mm512_set1_ps(0.f); });
      Unroll<ROWS>{}([&](auto i) { A_sum[i] = 0.0f; });

      for (int64_t k_offset = 0; k_offset < gs2; ++k_offset) {
        Unroll<ROWS * COLS>{}(compute, k + k_offset);
      }

      // sum of A within this group
      // NOTE: if we do zero point subtraction when loading B, we don't need to
      // compute A_sum
      Unroll<ROWS>{}([&](auto row) {
        __m512 acc = _mm512_cvtpbh_ps((__m256bh)_mm256_loadu_ps(a_ptr + row * lda2 + k));
        for (int j = 1; j < group_size / 16; ++j) {
          acc += _mm512_cvtpbh_ps((__m256bh)_mm256_loadu_ps(a_ptr + row * lda2 + k + j * 8));
        }
        __m256 tmp256 = _mm512_extractf32x8_ps(acc, 0) + _mm512_extractf32x8_ps(acc, 1);
        __m128 tmp128 = _mm256_extractf32x4_ps(tmp256, 0) + _mm256_extractf32x4_ps(tmp256, 1);
        A_sum[row] = (tmp128[0] + tmp128[1]) + (tmp128[2] + tmp128[3]);
      });
      Unroll<ROWS * COLS>{}(accumulate_group, k / gs2);
    }

    auto storec = [&](auto i) {
      constexpr int row = i / COLS;
      constexpr int col = i % COLS;
      if constexpr (col % 2 == 0) {
        _mm512_storeu_si512(
            reinterpret_cast<__m512i*>(C + row * ldc + col * 16),
            (__m512i)(_mm512_cvtne2ps_pbh(vc_master[i + 1], vc_master[i])));
      }
    };
    Unroll<ROWS * COLS>{}(storec);
  }
};
#endif

#define LAUNCH_TINYGEMM_KERNEL_NN(MB_SIZE, NB_SIZE)                          \
    tinygemm_kernel_nn<scalar_t, has_bias, MB_SIZE, NB_SIZE>::apply(         \
        A + mb_start * lda, B + nb_start, C + mb_start * ldc + nb_start, \
        Bz + nb_start, Bs + nb_start, has_bias ? bias + nb_start : nullptr,  \
        K, group_size, lda, ldb, ldc, strideBz, strideBs);

template <typename scalar_t, bool has_bias>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const uint8_t* __restrict__ Bz,
    const scalar_t* __restrict__ Bs,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int group_size,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    int64_t strideBz,
    int64_t strideBs) {

  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = 64;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);
  for (int mb = 0; mb < MB; ++mb) {
    int64_t mb_start = mb * BLOCK_M;
    int64_t mb_size = std::min(BLOCK_M, M - mb_start);
    for (int64_t nb = 0; nb < NB; ++nb) {
      int64_t nb_start = nb * BLOCK_N;
      int64_t nb_size = std::min(BLOCK_N, N - nb_start);

      switch(mb_size << 4 | nb_size >> 4) {
        // mb_size = 1
        case 0x12: LAUNCH_TINYGEMM_KERNEL_NN(1, 32); break;
        case 0x14: LAUNCH_TINYGEMM_KERNEL_NN(1, 64); break;
        // mb_size = 2
        case 0x22: LAUNCH_TINYGEMM_KERNEL_NN(2, 32); break;
        case 0x24: LAUNCH_TINYGEMM_KERNEL_NN(2, 64); break;
        // mb_size = 3
        case 0x32: LAUNCH_TINYGEMM_KERNEL_NN(3, 32); break;
        case 0x34: LAUNCH_TINYGEMM_KERNEL_NN(3, 64); break;
        // mb_size = 4
        case 0x42: LAUNCH_TINYGEMM_KERNEL_NN(4, 32); break;
        case 0x44: LAUNCH_TINYGEMM_KERNEL_NN(4, 64); break;
        default: TORCH_CHECK(false, "Unexpected block size, ", mb_size, "x", "nb_size");
      }
    }
  }
}

template <typename scalar_t>
void int4_w4a16_linear_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ w_zeros,
    const scalar_t* __restrict__ w_scales,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int group_size,
    int64_t mat1_strideM,
    int64_t out_strideM) {

  constexpr int64_t BLOCK_M = block_size_m();
  // constexpr int64_t BLOCK_N = block_size_n();
  constexpr int64_t BLOCK_N = 64;
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  // parallel on [MB, NB]
  AT_DISPATCH_BOOL(bias != nullptr, has_bias, [&] {
    at::parallel_for(0, MB * NB, 0, [&](int64_t begin, int64_t end) {
      int64_t mb{0}, nb{0};
      data_index_init(begin, mb, MB, nb, NB);

      for (int64_t i = begin; i < end; ++i) {
        UNUSED(i);
        int64_t mb_start = mb * BLOCK_M;
        int64_t mb_size = std::min(M - mb_start, BLOCK_M);
        int64_t nb_start = nb * BLOCK_N;
        int64_t nb_size = std::min(N - nb_start, BLOCK_N);

        tinygemm_kernel<scalar_t, has_bias>(
            /*   A */ x + mb_start * mat1_strideM,
            /*   B */ w + nb_start * K / 2,  // divide by 2 since w is u4 packed in u8
            /*   C */ out + mb_start * out_strideM + nb_start,
            /*  Bz */ w_zeros + nb_start,
            /*  Bs */ w_scales + nb_start,
            /* bias*/ bias + nb_start,
            /*   M */ mb_size,
            /*   N */ nb_size,
            /*   K */ K,
            /*  gs */ group_size,
            /* lda */ mat1_strideM,
            /* ldb */ nb_size,
            /* ldc */ out_strideM,
            /* sBz */ N,
            /* sBs */ N);

        // move to the next index
        data_index_step(mb, MB, nb, NB);
      }

    });
  });
}

} // anonymous namespace

// mat1     : [M, K]
// mat2     : [N, K] (appear as [N, K/2] in u8)
// w_zeros  : [K/gs, N]
// w_scales : [K/gs, N]
// bias     : [N]
// out      : [M, N]
//
at::Tensor int4_w4a16_linear(
    at::Tensor& x,
    at::Tensor& w,
    at::Tensor& w_zeros,
    at::Tensor& w_scales,
    std::optional<at::Tensor>& bias) {
  RECORD_FUNCTION(
    "sgl-kernel::int4_w4a16_linear", std::vector<c10::IValue>({x, w, w_zeros, w_scales, bias}));

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(x);
  CHECK_INPUT(w);
  CHECK_INPUT(w_zeros);
  CHECK_INPUT(w_scales);

  int64_t M = x.size(0);
  int64_t N = w.size(0);
  int64_t K = x.size(1);
  int group_size = K / w_zeros.size(0);
  CHECK_EQ(w.size(1), K / 2);  // u4 packed as u8
  CHECK_DIM(2, x);
  CHECK_DIM(2, w);

  auto out = at::empty({M, N}, x.options());

  // strides
  int64_t x_strideM = x.stride(0);
  int64_t out_strideM = out.stride(0);

  const bool has_bias = bias.has_value();
  const float* bias_data = nullptr;
  if (has_bias) {
    CHECK_EQ(bias.value().size(0), N);
    bias_data = bias.value().data_ptr<float>();
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(x.scalar_type(), "int4pack_linear_kernel_impl", [&] {
    int4_w4a16_linear_kernel_impl<scalar_t>(
        out.data_ptr<scalar_t>(),
        x.data_ptr<scalar_t>(),
        w.data_ptr<uint8_t>(),
        w_zeros.data_ptr<uint8_t>(),
        w_scales.data_ptr<scalar_t>(),
        bias_data,
        M,
        N,
        K,
        group_size,
        x_strideM,
        out_strideM);
  });

  return out;
}

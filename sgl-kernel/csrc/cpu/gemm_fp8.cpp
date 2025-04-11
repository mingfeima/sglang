#include "common.h"
#include "vec.h"
#include "gemm.h"

namespace {

template <typename scalar_t>
inline void copy_stub(scalar_t* __restrict__ out, const float* __restrict__ input, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();

  int64_t d;
  #pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input + d);
    fVec data1 = fVec::loadu(input + d + fVec::size());
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d]);
  }
}

template <typename scalar_t>
inline void copy_add_stub(scalar_t* __restrict__ out, const float* __restrict__ input, const float* __restrict__ bias, int64_t size) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int kVecSize = bVec::size();

  int64_t d;
  #pragma GCC unroll 4
  for (d = 0; d <= size - kVecSize; d += kVecSize) {
    fVec data0 = fVec::loadu(input + d) + fVec::loadu(bias + d);
    fVec data1 = fVec::loadu(input + d + fVec::size()) + fVec::loadu(bias + d + fVec::size());
    bVec out_vec = convert_from_float_ext<scalar_t>(data0, data1);
    out_vec.store(out + d);
  }
  for (; d < size; ++d) {
    out[d] = static_cast<scalar_t>(input[d] + bias[d]);
  }
}

inline void unpack_B(
    at::BFloat16* __restrict__ Btmp,
    const at::Float8_e4m3fn* __restrict__ packed_B,
    int N,
    int K,
    int ldb,
    int ldb_tmp,
    float scale) {
  // [K/2, N, 2]
  const int K2 = K >> 1;
  const int ldb2 = ldb; // ldb * 2 >> 1;
  const uint16_t* b_ptr = reinterpret_cast<const uint16_t*>(packed_B);
  const __m512 vd = _mm512_set1_ps(scale);

  constexpr int BLOCK_N = block_size_n();
  static_assert(BLOCK_N == 32);

  for (int k = 0; k < K2; ++k) {
    for (int n = 0; n < N; n += 64) { // BLOCK_N = 32
        __m512i b8 = _mm512_loadu_si512(b_ptr + k * ldb2 + n);

        __m256i b8_0 = _mm512_extracti32x8_epi32(b8, 0);
        __m256i b8_1 = _mm512_extracti32x8_epi32(b8, 1);

        __m512bh bf16_0 = CVT_FP8_TO_BF16(b8_0);
        __m512bh bf16_1 = CVT_FP8_TO_BF16(b8_1);

        // Apply scale
        __m512 f0_lo = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)bf16_0, 0));
        __m512 f0_hi = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)bf16_0, 1));
        __m512 f1_lo = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)bf16_1, 0));
        __m512 f1_hi = CVT_BF16_TO_FP32(_mm512_extracti32x8_epi32((__m512i)bf16_1, 1));

        f0_lo = _mm512_mul_ps(f0_lo, vd);
        f0_hi = _mm512_mul_ps(f0_hi, vd);
        f1_lo = _mm512_mul_ps(f1_lo, vd);
        f1_hi = _mm512_mul_ps(f1_hi, vd);

        bf16_0 = _mm512_cvtne2ps_pbh(f0_hi, f0_lo);
        bf16_1 = _mm512_cvtne2ps_pbh(f1_hi, f1_lo);

        _mm512_storeu_si512(Btmp + k * ldb_tmp * 2 + n * 2 + 0, (__m512i)bf16_0);
        _mm512_storeu_si512(Btmp + k * ldb_tmp * 2 + n * 2 + 32, (__m512i)bf16_1);
    }
  }
}

template <typename scalar_t, typename packed_t, bool has_bias>
struct brgemm {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const packed_t* __restrict__ B,
      scalar_t* __restrict__ C,
      scalar_t* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      const float* __restrict__ bias,
      const float* __restrict__ scale,
      int M,
      int N,
      int K,
      int lda,
      int ldb,
      int ldc) {
    TORCH_CHECK(false, "struct brgemm: primary template not implemented!");
  }
};

template <typename scalar_t, bool has_bias>
struct brgemm<scalar_t, scalar_t, has_bias> {
  static inline void apply(
      const scalar_t* __restrict__ A,
      const scalar_t* __restrict__ B,
      scalar_t* __restrict__ C,
      scalar_t* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      const float* __restrict__ bias,
      const float* __restrict__ scale,
      int M,
      int N,
      int K,
      int lda,
      int ldb,
      int ldc) {
    UNUSED(scale);

    constexpr int BLOCK_N = block_size_n();
    at::native::cpublas::brgemm(
        M, N, K, lda, ldb, BLOCK_N, /* add_C */ false, A, B, Ctmp);

    // copy from Ctmp to C
    for (int m = 0; m < M; ++m) {
      if constexpr (has_bias) {
        copy_add_stub(C + m * ldc, Ctmp + m * BLOCK_N, bias, N);
      } else {
        copy_stub(C + m * ldc, Ctmp + m * BLOCK_N, N);
      }
    }
  }
};

template <bool has_bias>
struct brgemm<at::BFloat16, at::Float8_e4m3fn, has_bias> {
  static inline void apply(
      const at::BFloat16* __restrict__ A,
      const at::Float8_e4m3fn* __restrict__ B,
      at::BFloat16* __restrict__ C,
      at::BFloat16* __restrict__ Btmp,
      float* __restrict__ Ctmp,
      const float* __restrict__ bias,
      const float* __restrict__ scale,
      int M,
      int N,
      int K,
      int lda,
      int ldb,
      int ldc) {
    constexpr int BLOCK_N = block_size_n();

    // [BLOCK_K, BLOCK_N] -> [BLOCK_K / 2, BLOCK_N * 2]
    const int ldb_tmp = block_size_n();

    static_assert(BLOCK_K == 128);

    // accumulate across K per BLOCK_K
    for (int k = 0; k < K; k += BLOCK_K) {
      int kb_size = std::min(BLOCK_K, K - k);

      int idx = k >> 7; // k / BLOCK_K where BLOCK_K = 128
      unpack_B(Btmp, B + k * ldb, N, kb_size, ldb, ldb_tmp, scale[idx]);

      const bool add_C = (k != 0);
      at::native::cpublas::brgemm(
          M, N, kb_size, lda, ldb_tmp, BLOCK_N, add_C, A + k, Btmp, Ctmp);
    }

    // copy from Ctmp to C
    for (int m = 0; m < M; ++m) {
      if constexpr (has_bias) {
        copy_add_stub(C + m * ldc, Ctmp + m * BLOCK_N, bias, N);
      } else {
        copy_stub(C + m * ldc, Ctmp + m * BLOCK_N, N);
      }
    }
  }
};

template <typename scalar_t, bool has_bias>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const at::Float8_e4m3fn* __restrict__ B,
    scalar_t* __restrict__ C,
    scalar_t* __restrict__ Btmp,
    float* __restrict__ Ctmp,
    const float* __restrict__ scale,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg,
    int64_t block_size_K) {

  if (brg) {
    brgemm<scalar_t, at::Float8_e4m3fn, has_bias>::apply(
        A, B, C, Btmp, Ctmp, bias, scale, M, N, K, lda, ldb, ldc);
    return;
  }

  // TODO: add the support for use_brgemm = false;
  TORCH_CHECK(false, "use_brgemm = false is not supported yet");
}

template <typename scalar_t>
void fp8_scaled_mm_kernel_impl(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mat1,
    const at::Float8_e4m3fn* __restrict__ mat2,
    const float* __restrict__ scales2,
    const float* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t mat1_strideM,
    int64_t out_strideM,
    int64_t block_size_N,
    int64_t block_size_K) {

  constexpr int64_t BLOCK_M = block_size_m();
  constexpr int64_t BLOCK_N = block_size_n();
  const int64_t MB = div_up(M, BLOCK_M);
  const int64_t NB = div_up(N, BLOCK_N);

  const int64_t scale_size_N = div_up(N, block_size_N);
  const int64_t scale_size_K = div_up(K, block_size_K);

  const int64_t blocks_n_per_group = block_size_N / BLOCK_N;

  // TODO: add the support for use_brgemm = false;
  // use avx512-bf16 when a) M is small; b) dtype is bfloat16, otherwise use amx
  const bool use_brgemm = can_use_brgemm<at::Float8_e4m3fn>(M);

  // parallel on [MB, NB]
  AT_DISPATCH_BOOL(bias != nullptr, has_bias, [&] {
    at::parallel_for(0, MB * NB, 0, [&](int64_t begin, int64_t end) {
      int64_t mb{0}, nb{0};
      data_index_init(begin, mb, MB, nb, NB);

      // for brgemm, use float32 for accumulate
      alignas(64) float Ctmp[BLOCK_M * BLOCK_N];
      // for brgemm when mat2 is float8_e4m3
      alignas(64) scalar_t Btmp[BLOCK_N * BLOCK_K];

      for (int64_t i = begin; i < end; ++i) {
        UNUSED(i);
        const float* scale_ptr = scales2 + (nb / blocks_n_per_group) * scale_size_K;

        int64_t mb_start = mb * BLOCK_M;
        int64_t mb_size = std::min(M - mb_start, BLOCK_M);
        int64_t nb_start = nb * BLOCK_N;
        int64_t nb_size = std::min(N - nb_start, BLOCK_N);

        tinygemm_kernel<scalar_t, has_bias>(
            /*   A            */ mat1 + mb_start * mat1_strideM,
            /*   B            */ mat2 + nb_start * K, // nb * BLOCK_N * K
            /*   C            */ out + mb_start * out_strideM + nb_start,
            /*   Btmp         */ Btmp,
            /*   Ctmp         */ Ctmp,
            /*   scale        */ scale_ptr,
            /*   bias         */ bias + nb_start,
            /*   M            */ mb_size,
            /*   N            */ nb_size,
            /*   K            */ K,
            /*   lda          */ mat1_strideM,
            /*   ldb          */ nb_size,
            /*   ldc          */ out_strideM,
            /*   brg          */ use_brgemm,
            /*   block_size_K */ block_size_K);

        // move to the next index
        data_index_step(mb, MB, nb, NB);
      }

      if (use_brgemm) {
        at::native::cpublas::brgemm_release();
      }
    });
  });

}

} // anonymous namespace

at::Tensor fp8_scaled_mm_cpu(at::Tensor& mat1, at::Tensor& mat2, at::Tensor& scales2,
    std::vector<int64_t> block_size, std::optional<at::Tensor>& bias,
    at::ScalarType out_dtype, bool is_vnni) {
  RECORD_FUNCTION("sgl-kernel::fp8_scaled_mm_cpu", std::vector<c10::IValue>({mat1, mat2, scales2, block_size, bias}));

  auto packed_w = is_vnni ? mat2 : convert_weight_packed(mat2);

  CHECK_LAST_DIM_CONTIGUOUS_INPUT(mat1);
  CHECK_INPUT(mat2);
  CHECK_INPUT(scales2);
  TORCH_CHECK(scales2.scalar_type() == at::kFloat,
      "fp8_scaled_mm_cpu: expect scales2 to be float32.");

  int64_t M = mat1.size(0);
  int64_t N = mat2.size(0);
  int64_t K = mat2.size(1);

  CHECK_EQ(mat1.size(1), K);
  CHECK_DIM(2, mat1);
  CHECK_DIM(2, mat2);

  TORCH_CHECK(block_size.size() == 2,
      "fp8_scaled_mm_cpu: expect block_size.size() to be 2.");

  int64_t block_size_N = block_size[0];
  int64_t block_size_K = block_size[1];

  constexpr int64_t BLOCK_N = block_size_n();
  TORCH_CHECK(block_size_N % BLOCK_N == 0, "fp8_scaled_mm_cpu: expect block_size_N to be multiples of BLOCK_N");
  TORCH_CHECK(block_size_K == BLOCK_K, "fp8_scaled_mm_cpu: expect block_size_K equals to BLOCK_K");
  CHECK_EQ(scales2.size(0), div_up(N, block_size_N));
  CHECK_EQ(scales2.size(1), div_up(K, block_size_K));

  const auto st = mat1.scalar_type();
  TORCH_CHECK(st == at::kBFloat16 || st == at::kHalf,
      "fp8_scaled_mm_cpu: expect A to be bfloat16 or half.");
  TORCH_CHECK(st == out_dtype,
      "fp8_scaled_mm_cpu: expect A has same dtype with out_dtype.");
  TORCH_CHECK(mat2.scalar_type() == at::kFloat8_e4m3fn,
      "fp8_scaled_mm_cpu: expect mat2 to be fp8_e4m3.");
  TORCH_CHECK(scales2.scalar_type() == at::kFloat,
      "fp8_scaled_mm_cpu: expect scales to be float32.");
  auto out = at::empty({M, N}, mat1.options().dtype(out_dtype));

  // strides
  int64_t mat1_strideM = mat1.stride(0);
  int64_t out_strideM = out.stride(0);

  const bool has_bias = bias.has_value();
  const float* bias_data = nullptr;
  if (has_bias) {
    CHECK_EQ(bias.value().size(0), N);
    bias_data = bias.value().data_ptr<float>();
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(out_dtype, "fp8_scaled_mm_kernel_impl", [&] {
    fp8_scaled_mm_kernel_impl<scalar_t>(
        out.data_ptr<scalar_t>(),
        mat1.data_ptr<scalar_t>(),
        packed_w.data_ptr<at::Float8_e4m3fn>(),
        scales2.data_ptr<float>(),
        bias_data,
        M,
        N,
        K,
        mat1_strideM,
        out_strideM,
        block_size_N,
        block_size_K);
  });

  return out;
}

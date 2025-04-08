#include "common.h"
#include "vec.h"

namespace {
void cvt_e4m3_bf16_intrinsic(const at::Float8_e4m3fn* __restrict__ in, at::BFloat16* out, int64_t len, bool fzt) {
  int64_t i = 0;
  for (; i < len - 31; i += 32) {
    __m256i x0 = _mm256_loadu_si256((__m256i*)&in[i]);
    __m512i bh;
    if (fzt) {
      bh = cvt_e4m3_bf16_intrinsic_without_denorm(x0);
    } else {
      bh = cvt_e4m3_bf16_intrinsic_with_denorm(x0);
    }
    _mm512_storeu_si512((__m512i*)&out[i], bh);
  }
  for (; i < len; i++) {
    out[i] = static_cast<at::BFloat16>(in[i]);
  }
}
}  // namespace

at::Tensor cvt_e4m3_to_bf16(at::Tensor& src, at::Tensor& dst, int64_t len) {
  RECORD_FUNCTION("sgl-kernel::convert_e4m3_to_bf16", c10::ArrayRef<c10::IValue>({src, dst}));

  auto src_ptr = src.data_ptr<at::Float8_e4m3fn>();
  auto dst_ptr = dst.data_ptr<at::BFloat16>();
#ifdef SGLANG_CPU_FP8_CVT_FTZ
  cvt_e4m3_bf16_intrinsic(src_ptr, dst_ptr, len, true);
#else
  cvt_e4m3_bf16_intrinsic(src_ptr, dst_ptr, len, false);
#endif
  return dst;
}

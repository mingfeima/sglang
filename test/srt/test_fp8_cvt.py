import unittest
import expecttest

import torch
import sgl_kernel.cpu

class TestFP8Convert(expecttest.TestCase):
    def test_convert_e4m3_to_bf16(self):
        # Test without denorm
        weight = (torch.rand(4096, 14336, dtype=torch.bfloat16) * 2 - 1) * 448.0
        weight = weight.to(torch.float8_e4m3fn)
        weight_bf16 = torch.empty(4096, 14336, dtype=torch.bfloat16)
        sgl_kernel.cpu.cvt_e4m3_to_bf16(
            weight, weight_bf16, 4096 * 14336
        )
        self.assertTrue(
            torch.allclose(weight.to(torch.bfloat16), weight_bf16),
            f"cvt_e4m3_to_bf16 failed: expected {weight}, got {weight_bf16}",
        )

        # Test with denorm using intrinsic (No Nan support)
        # weight = torch.arange(256, dtype=torch.uint8)
        # weight[127] = 126
        # weight[255] = 254
        # weight = weight.view(torch.float8_e4m3fn)
        # weight_bf16 = torch.empty(256, dtype=torch.bfloat16)
        # sgl_kernel.cpu.cvt_e4m3_to_bf16(weight, weight_bf16, 256)
        # self.assertTrue(
        #     torch.allclose(weight.to(torch.bfloat16), weight_bf16, equal_nan=True),
        #     f"cvt_e4m3_to_bf16 failed: expected {weight}, got {weight_bf16}",
        # )
        # weight2 = torch.randn(4096, 14336, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        # weight2_bf16 = torch.empty(4096, 14336, dtype=torch.bfloat16)
        # sgl_kernel.cpu.cvt_e4m3_to_bf16(
        #     weight2, weight2_bf16, 4096 * 14336
        # )
        # self.assertTrue(
        #     torch.allclose(weight2.to(torch.bfloat16), weight2_bf16),
        #     f"convert_e4m3_to_bf16 failed: expected {weight2}, got {weight2_bf16}",
        # )


if __name__ == "__main__":
    unittest.main()

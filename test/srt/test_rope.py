import unittest
import expecttest

import torch
import sgl_kernel.cpu

class TestFusedMOE(expecttest.TestCase):
    def test_deepseek_v2_rope(self):
        def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            x = torch.stack((-x2, x1), dim=-1)
            return x.flatten(-2)

        def rotary_emb_ref(sin, cos, query, key):
            query_rot = query[..., : qk_rope_head_dim]
            key_rot = key[..., : qk_rope_head_dim]

            rotate_fn = _rotate_gptj
            query_rot = query_rot * cos + rotate_fn(query_rot) * sin
            key_rot = key_rot * cos + rotate_fn(key_rot) * sin
            return query_rot, key_rot

        num_head = 16
        seq_len = 1024
        q_head_dim = 192
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64

        # Create sin, cos
        freqs = torch.rand(seq_len, qk_rope_head_dim // 2)
        positions = torch.arange(seq_len)
        cos = freqs.cos() * 0.7
        sin = freqs.sin() * 0.7
        cache = torch.cat((cos, sin), dim=-1)
        cos_sin = cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
        sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        for dtype in [torch.float, torch.bfloat16]:
            prec = 1e-5 if dtype == torch.float32 else 5e-3
            enable_autocast = dtype == torch.bfloat16

            with torch.no_grad(), torch.cpu.amp.autocast(enabled=enable_autocast):
                q = torch.randn(seq_len, num_head, q_head_dim, dtype=dtype)
                k_pe = torch.randn(seq_len, 1, qk_rope_head_dim, dtype=dtype)
                q_clone = q.clone()
                k_pe_clone = k_pe.clone()

                # ref kernel
                _, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
                q_pe, k_pe = rotary_emb_ref(sin, cos, q_pe, k_pe)
                q[..., qk_nope_head_dim :] = q_pe

                # fused rope kernel
                q_clone, k_pe_clone = sgl_kernel.cpu.rotary_position_embedding(
                    q_clone, k_pe_clone, sin, cos, positions, num_head, q_head_dim,
                    qk_nope_head_dim, qk_rope_head_dim
                )

                assert torch.allclose(q, q_clone, rtol=prec, atol=prec)
                assert torch.allclose(k_pe, k_pe_clone.to(torch.float), rtol=prec, atol=prec)


if __name__ == "__main__":
    unittest.main()
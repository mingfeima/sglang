
import unittest

import sgl_kernel
import torch
from torch.nn.functional import softplus
import torch.nn.functional as F

from sglang.test.test_utils import CustomTestCase
from utils import precision

torch.manual_seed(1234)

def torch_gdn_gating(A_log, a, dt_bias):
    return -A_log.float().exp() * softplus(a.float() + dt_bias)

def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm

def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def sigmoid_gating_delta_rule_update(query, key, value, A_log, a, dt_bias, b, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    beta = b.sigmoid()
    g = -A_log.float().exp() * softplus(a.float() + dt_bias)
    return torch_recurrent_gated_delta_rule(
        query, key, value, g.unsqueeze(0), beta.unsqueeze(0), initial_state, output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel
    )

def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - num_heads % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    tot_heads = num_heads + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, sequence_length, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, tot_heads // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :num_heads]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state

def chunk_gated_delta_rule_update(
    query, # [B, T, HK, K]
    key, # [B, T, HK, K]
    value, # [B, T, HV, V]
    g, # [B, T, HV]
    beta, # [B, T, HV]
    cu_seqlens, # [N+1]
    initial_state, # [N, HV, K, V]
    use_qk_l2norm_in_kernel, # True
):
    num_heads = query.shape[2]
    num_value_heads = value.shape[2]
    batch_size = initial_state.shape[0]
    if num_value_heads // num_heads > 1:
        query = query.repeat_interleave(num_value_heads // num_heads, dim=2)
        key = key.repeat_interleave(num_value_heads // num_heads, dim=2)
    output = torch.empty_like(value)
    final_state = torch.empty_like(initial_state)
    start_q = 0
    for i in range(batch_size):
        end_q = cu_seqlens[i + 1]
        core_attn_outi, last_recurrent_state = torch_chunk_gated_delta_rule(
            query=query[:, start_q:end_q, :, :],
            key=key[:, start_q:end_q, :, :],
            value=value[:, start_q:end_q, :, :],
            g=g[:, start_q:end_q, :],
            beta=beta[:, start_q:end_q, :],
            initial_state=initial_state[i],
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        output[:, start_q:end_q, :, :] = core_attn_outi
        final_state[i] = last_recurrent_state
        start_q = end_q
    return output, final_state

def fix_query_key_value_ordering_reshape_cat(
    mixed_qkvz,
    mixed_ba,
    num_k_heads,
    num_v_heads,
    attn_tp_size,
    head_k_dim,
    head_v_dim):
    new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
        num_k_heads // attn_tp_size,
        (
            head_k_dim
            + head_k_dim
            + (head_v_dim + head_v_dim)
            * num_v_heads
            // num_k_heads
        ),
    )
    new_tensor_shape_ba = mixed_ba.size()[:-1] + (
        num_k_heads // attn_tp_size,
        2 * num_v_heads // num_k_heads,
    )

    mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
    mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

    split_arg_list_qkvz = [
        head_k_dim,
        head_k_dim,
        (num_v_heads // num_k_heads * head_v_dim),
        (num_v_heads // num_k_heads * head_v_dim),
    ]
    split_arg_list_ba = [
        num_v_heads // num_k_heads,
        num_v_heads // num_k_heads,
    ]
    # [b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]
    # --> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]
    (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=2)
    (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=2)

    # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
    value = value.reshape(value.size(0), -1, head_v_dim)
    z = z.reshape(z.size(0), -1, head_v_dim)
    b = b.reshape(b.size(0), num_v_heads // attn_tp_size)
    a = a.reshape(a.size(0), num_v_heads // attn_tp_size)
    query, key, value = map(
        lambda x: x.reshape(x.shape[0], -1), (query, key, value)
    )
    mixed_qkv = torch.cat((query, key, value), dim=-1)

    return mixed_qkv, z, b, a

class TestMambaAttention(CustomTestCase):
    def test_fused_gdn_gating(self):
        dims = [6, 32]
        for dim in dims:
            A_log = torch.rand(dim)
            a = torch.rand(1024, dim, dtype=torch.bfloat16)
            dt_bias = torch.rand(dim, dtype=torch.bfloat16)

            g = torch_gdn_gating(A_log, a, dt_bias)
            g_sgl = torch.ops.sgl_kernel.fused_gdn_gating_cpu(A_log, a, dt_bias)
            atol = rtol = precision[g.dtype]
            self.assertTrue(torch.allclose(g, g_sgl, atol=atol, rtol=rtol))

    def test_recurrent_gated_delta_rule(self):
        batch_size = 1
        num_value_heads = 32
        head_k_dim = 128
        head_v_dim = 128
        num_heads = 16
        seq_len = 1
        query = torch.rand(seq_len, batch_size, num_heads, head_k_dim, dtype=torch.bfloat16)
        key = torch.rand(seq_len, batch_size, num_heads, head_k_dim, dtype=torch.bfloat16)
        value = torch.rand(seq_len, batch_size, num_value_heads, head_v_dim, dtype=torch.bfloat16)
        A_log = torch.rand(num_value_heads, dtype=torch.float32)
        a = torch.rand(batch_size, num_value_heads, dtype=torch.float32)
        dt_bias = torch.rand(num_value_heads, dtype=torch.float32)
        g = -A_log.exp() * softplus(a + dt_bias)
        beta = torch.rand(batch_size, num_value_heads, dtype=torch.bfloat16).sigmoid()
        ssm_states = torch.rand(513, num_value_heads, head_k_dim, head_v_dim, dtype=torch.float32)
        cache_indices = torch.randint(0, 513, (batch_size,), dtype=torch.int32)
        use_qk_l2norm_in_kernel = True
        query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
        query_ref = query.clone()
        key_ref = key.clone()
        if num_value_heads // num_heads > 1:
            query_ref = query_ref.repeat_interleave(num_value_heads // num_heads, dim=2)
            key_ref = key_ref.repeat_interleave(num_value_heads // num_heads, dim=2)
        core_attn_out_ref, last_recurrent_state_ref = torch_recurrent_gated_delta_rule(
            query_ref.transpose(0, 1),
            key_ref.transpose(0, 1),
            value.transpose(0, 1),
            g=g.unsqueeze(0),
            beta=beta.unsqueeze(0),
            initial_state=ssm_states[cache_indices],
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel
        )
        core_attn_out = torch.ops.sgl_kernel.fused_recurrent_gated_delta_rule_cpu(
            query,
            key,
            value,
            g,
            beta,
            cache_indices,
            ssm_states,
            use_qk_l2norm_in_kernel
        )
        last_recurrent_state = ssm_states[cache_indices]
        atol = rtol = precision[core_attn_out.dtype]
        self.assertTrue(torch.allclose(core_attn_out, core_attn_out_ref, atol=atol, rtol=rtol))
        self.assertTrue(torch.allclose(last_recurrent_state, last_recurrent_state_ref, atol=atol, rtol=rtol))

    def test_fused_sigmoid_gating_delta_rule_update(self):
        batch_size = 1
        num_value_heads = 32
        head_k_dim = 128
        head_v_dim = 128
        num_heads = 16
        seq_len = 1
        query = torch.rand(seq_len, batch_size, num_heads, head_k_dim, dtype=torch.bfloat16)
        key = torch.rand(seq_len, batch_size, num_heads, head_k_dim, dtype=torch.bfloat16)
        value = torch.rand(seq_len, batch_size, num_value_heads, head_v_dim, dtype=torch.bfloat16)
        A_log = torch.rand(num_value_heads, dtype=torch.float32)
        a = torch.rand(batch_size, num_value_heads, dtype=torch.bfloat16)
        b = torch.rand(batch_size, num_value_heads, dtype=torch.bfloat16)
        dt_bias = torch.rand(num_value_heads, dtype=torch.bfloat16)
        ssm_states = torch.rand(513, num_value_heads, head_k_dim, head_v_dim, dtype=torch.float32)
        cache_indices = torch.randint(0, 513, (batch_size,), dtype=torch.int32)
        use_qk_l2norm_in_kernel = True
        query_ref = query.clone()
        key_ref = key.clone()
        if num_value_heads // num_heads > 1:
            query_ref = query_ref.repeat_interleave(num_value_heads // num_heads, dim=2)
            key_ref = key_ref.repeat_interleave(num_value_heads // num_heads, dim=2)
        core_attn_out_ref, last_recurrent_state_ref = sigmoid_gating_delta_rule_update(
            query_ref.transpose(0, 1),
            key_ref.transpose(0, 1),
            value.transpose(0, 1),
            A_log,
            a,
            dt_bias,
            b,
            initial_state=ssm_states[cache_indices],
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel
        )
        core_attn_out = torch.ops.sgl_kernel.fused_sigmoid_gating_delta_rule_update_cpu(
            query,
            key,
            value,
            A_log,
            a,
            dt_bias,
            b,
            cache_indices,
            ssm_states,
            use_qk_l2norm_in_kernel
        )
        last_recurrent_state = ssm_states[cache_indices]
        atol = rtol = precision[core_attn_out.dtype]
        self.assertTrue(torch.allclose(core_attn_out, core_attn_out_ref, atol=atol, rtol=rtol))
        self.assertTrue(torch.allclose(last_recurrent_state, last_recurrent_state_ref, atol=atol, rtol=rtol))

    def test_fused_qkvzba_split_reshape_cat(self):
        mixed_qkvz = torch.rand(1024, 12288, dtype=torch.bfloat16)
        mixed_ba = torch.rand(1024, 64, dtype=torch.bfloat16)
        head_k_dim = 128
        head_v_dim = 128
        num_v_heads = 32
        num_k_heads = 16
        attn_tp_size = 1
        mixed_qkv_ref, z_ref, b_ref, a_ref = fix_query_key_value_ordering_reshape_cat(
            mixed_qkvz,
            mixed_ba,
            num_k_heads,
            num_v_heads,
            attn_tp_size,
            head_k_dim,
            head_v_dim
        )
        num_heads_qk = num_k_heads // attn_tp_size
        num_heads_v = num_v_heads // attn_tp_size
        mixed_qkv, z, b, a = torch.ops.sgl_kernel.fused_qkvzba_split_reshape_cat_cpu(
            mixed_qkvz,
            mixed_ba,
            num_heads_qk,
            num_heads_v,
            head_k_dim,
            head_v_dim
        )
        atol = rtol = precision[mixed_qkv.dtype]
        self.assertTrue(torch.allclose(mixed_qkv, mixed_qkv_ref, atol=atol, rtol=rtol))
        self.assertTrue(torch.allclose(z, z_ref, atol=atol, rtol=rtol))
        self.assertTrue(torch.allclose(b, b_ref, atol=atol, rtol=rtol))
        self.assertTrue(torch.allclose(a, a_ref, atol=atol, rtol=rtol))

    def test_chunk_gated_delta_rule(self):
        B, T, HK, HV, EK, EV, N = 1, 512, 16, 32, 128, 128, 4
        query_ = torch.rand((B, T, HK, EK), dtype=torch.bfloat16) * 0.1
        key_ = torch.rand((B, T, HK, EK), dtype=torch.bfloat16) * 0.1
        value_ = torch.rand((B, T, HV, EV), dtype=torch.bfloat16) * 0.1
        g_ = torch.rand((B, T, HV), dtype=torch.float32) * 0.1
        beta_ = torch.rand((B, T, HV), dtype=torch.bfloat16) * 0.1
        cu_seqlens_ = torch.tensor([0, 128, 256, 384, T], dtype=torch.int32)
        initial_state_ = torch.rand((N, HV, EK, EV), dtype=torch.float32) * 0.1

        core_attn_out_ref, last_recurrent_state_ref = chunk_gated_delta_rule_update(
            query=query_,
            key=key_,
            value=value_,
            g=g_,
            beta=beta_,
            cu_seqlens=cu_seqlens_,
            initial_state=initial_state_,
            use_qk_l2norm_in_kernel=True,
        )

        query = query_.clone()
        key = key_.clone()
        value = value_.clone()
        g = g_.clone()
        beta = beta_.clone()
        cu_seqlens = cu_seqlens_.clone()
        initial_state = initial_state_.clone()

        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
        core_attn_out, last_recurrent_state = torch.ops.sgl_kernel.chunk_gated_delta_rule_cpu(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            use_qk_l2norm_in_kernel=True,
        )
        atol = rtol = precision[core_attn_out.dtype]
        self.assertTrue(torch.allclose(core_attn_out, core_attn_out_ref, rtol=rtol, atol=atol))
        self.assertTrue(torch.allclose(last_recurrent_state, last_recurrent_state_ref, rtol=rtol, atol=atol))

if __name__ == "__main__":
    unittest.main()

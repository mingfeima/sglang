from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.configs.load_config import LoadConfig

from sglang.srt.layers.vocab_parallel_embedding import pad_vocab_size

DEFAULT_MOE_PADDING_SIZE = 32

try:
    from sgl_kernel.cpu import convert_weight_packed

    is_intel_amx_backend_available = True
except:
    is_intel_amx_backend_available = False


def get_moe_padding_size(model_config, load_config):
    from sglang.srt.model_loader.loader import _get_quantization_config

    quant_config = _get_quantization_config(model_config, load_config)

    if quant_config is not None and hasattr(quant_config, "weight_block_size"):
        # See NOTE(HandH1998): To ensure proper alignment of the block-wise quantization scales, the output_size of the weights for both the gate and up layers must be divisible by block_n.
        weight_block_size = getattr(quant_config, "weight_block_size")

        assert (
            len(weight_block_size) == 2
        ), "Only len(weight_block_size) == 2 is supported"
        assert (
            weight_block_size[0] == weight_block_size[1]
        ), "Only weight_block_size[0] == weight_block_size[1] is supported"

        return weight_block_size[0]

    return DEFAULT_MOE_PADDING_SIZE


def update_intermediate_size(model_config, attr_name, intermediate_padding_size):
    if hasattr(model_config.hf_config, attr_name):
        attr_value = getattr(model_config.hf_config, attr_name)
        if attr_value % intermediate_padding_size != 0:
            attr_value = pad_vocab_size(attr_value, intermediate_padding_size)
            setattr(model_config.hf_config, attr_name, attr_value)
            setattr(model_config.hf_text_config, attr_name, attr_value)
    return model_config


def update_config(
    model_config: ModelConfig, load_config: LoadConfig, tp_size: int
) -> ModelConfig:
    # Support the case where the num_attention_heads is not divisible by the TP size.
    if model_config.num_attention_heads % tp_size != 0:
        query_heads_per_kv = (
            model_config.num_attention_heads // model_config.get_total_num_kv_heads()
        )
        total_kv_heads = model_config.get_total_num_kv_heads()
        num_key_value_heads = pad_vocab_size(total_kv_heads, tp_size)
        model_config.num_key_value_heads = num_key_value_heads
        model_config.hf_config.num_key_value_heads = num_key_value_heads
        model_config.hf_text_config.num_key_value_heads = num_key_value_heads

        num_attention_heads = num_key_value_heads * query_heads_per_kv
        model_config.num_attention_heads = num_attention_heads
        model_config.hf_config.num_attention_heads = num_attention_heads
        model_config.hf_text_config.num_attention_heads = num_attention_heads

    intermediate_padding_size = tp_size * get_moe_padding_size(
        model_config, load_config
    )
    model_config = update_intermediate_size(
        model_config, "moe_intermediate_size", intermediate_padding_size
    )
    model_config = update_intermediate_size(
        model_config, "intermediate_size", intermediate_padding_size
    )

    return model_config


def get_actual_shard_size(shard_size, weight_start, weight_end):
    return min(shard_size, weight_end - weight_start)


def reset_param_data_if_needed(param_data, dim, start, length):
    if length == 0:
        return

    assert length > 0, f"Length should be positive, but got {length}"

    param_data.narrow(dim, start, length).zero_()
    return


def cpu_has_amx_support():
    return torch._C._cpu._is_amx_tile_supported() and is_intel_amx_backend_available


def prepack_weight_if_needed(weight):
    if weight.device != torch.device("cpu"):
        return weight
    if not cpu_has_amx_support():
        return weight

    return convert_weight_packed(weight)


def _process_weight_after_loading(module, weight_names, transpose_dims=None) -> None:
    # Pack weight for get better performance on CPU
    devices = {getattr(module, weight_name).device for weight_name in weight_names}
    assert len(devices) == 1, f"Expects all weights to be on the same device"
    device = devices.pop()

    if transpose_dims:
        assert len(weight_names) == len(
            transpose_dims
        ), "len(weight_names) should be equal to len(transpose_dims)"

    for i, weight_name in enumerate(weight_names):
        weight_tensor = getattr(module, weight_name)
        if transpose_dims and transpose_dims[i]:
            weight_tensor = weight_tensor.transpose(*transpose_dims[i])

        setattr(
            module,
            weight_name,
            torch.nn.Parameter(
                prepack_weight_if_needed(weight_tensor),
                requires_grad=False,
            ),
        )

    module.use_intel_amx_backend = (
        device == torch.device("cpu") and cpu_has_amx_support()
    )

    if module.use_intel_amx_backend and getattr(module, "bias", None) is not None:
        module.bias = torch.nn.Parameter(module.bias.data.float(), requires_grad=False)


class PackWeightMethod:
    def __init__(self, weight_names, transpose_dims=None):
        self.weight_names = weight_names
        self.transpose_dims = transpose_dims

    def process_weights_after_loading(self, module) -> None:
        _process_weight_after_loading(module, self.weight_names, self.transpose_dims)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = x.to(torch.float32)
    d = x.shape[-1] // 2
    out = torch.nn.functional.silu(x[..., :d]) * x[..., d:]
    return out.to(dtype)

def per_token_quant_int8_cpu(x):
    x = x.float()
    absmax = x.abs().max(dim=-1).values
    absmax = absmax.clamp_min(1e-7).unsqueeze(-1)
    scale_x = absmax / 127
    x_q = x.mul(127 / absmax)
    x_q = torch.round(x_q).to(torch.int8)

    return x_q, scale_x

def native_w8a8_per_token_matmul(A, B, As, Bs, bias=None, output_dtype=torch.bfloat16):
    """Matrix multiplication function that supports per-token input quantization and per-column weight quantization"""
    A = A.to(torch.float32)
    B = B.to(torch.float32)

    assert A.shape[-1] == B.shape[-1], "Dimension mismatch"
    assert B.ndim == 2 and B.is_contiguous(), "B must be a 2D contiguous tensor"

    # Reshape input
    M = A.numel() // A.shape[-1]
    B = B.t()  # Transpose weight matrix
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (K,)
    A = A.reshape(M, N)

    # As is per-token [M, 1], Bs is per-column [1, K]
    C = torch.matmul(A, B)  # [M, K]
    C = As * C * Bs.view(1, -1)  # Broadcast per-column scale

    if bias is not None:
        C.add_(bias.view(1, -1))

    return C.reshape(origin_C_shape).to(output_dtype)

def torch_w8a8_per_column_moe(a, w1, w2, w1_s, w2_s, topk_weight, topk_ids, topk):
    """This function performs fused moe with per-column int8 quantization using native torch."""

    B, D = a.shape
    # Perform per-token quantization
    a_q, a_s = per_token_quant_int8_cpu(a)
    # Repeat tokens to match topk
    a_q = a_q.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    # Also repeat the scale
    a_s = a_s.view(B, -1, 1).repeat(1, topk, 1).reshape(-1, 1)  # [B*topk, 1]

    out = torch.zeros(B * topk, w2.shape[1], dtype=torch.float32, device=a.device)

    # Calculate routing
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    # Process each expert
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            # First MLP layer: note that a_s is now per-token
            inter_out = native_w8a8_per_token_matmul(
                a_q[mask], w1[i], a_s[mask], w1_s[i], output_dtype=torch.float32
            )
            # Activation function
            act_out = silu_and_mul(inter_out)
            # Quantize activation output with per-token
            act_out_q, act_out_s = per_token_quant_int8_cpu(act_out)
            # Second MLP layer
            out[mask] = native_w8a8_per_token_matmul(
                act_out_q, w2[i], act_out_s, w2_s[i], output_dtype=torch.float32
            )
    # Apply routing weights and sum
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1).to(a.dtype)

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


def may_get_weight_block_size(model_config, load_config):
    from sglang.srt.model_loader.loader import _get_quantization_config

    quant_config = _get_quantization_config(model_config, load_config)

    if quant_config is not None and hasattr(quant_config, "weight_block_size"):
        return getattr(quant_config, "weight_block_size")

    return None


def get_num_heads_padding_size(tp_size, weight_block_size):
    pad_size = (
        tp_size * 2 if tp_size % 2 == 1 and weight_block_size is not None else tp_size
    )
    return pad_size


def get_moe_padding_size(weight_block_size):
    if weight_block_size is not None:
        # See NOTE(HandH1998): To ensure proper alignment of the block-wise quantization scales, the output_size of the weights for both the gate and up layers must be divisible by block_n.
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
    weight_block_size = may_get_weight_block_size(model_config, load_config)

    if model_config.num_attention_heads % tp_size != 0:
        query_heads_per_kv = (
            model_config.num_attention_heads // model_config.get_total_num_kv_heads()
        )
        total_kv_heads = model_config.get_total_num_kv_heads()

        pad_size = get_num_heads_padding_size(tp_size, weight_block_size)
        num_key_value_heads = pad_vocab_size(total_kv_heads, pad_size)

        model_config.num_key_value_heads = num_key_value_heads
        model_config.hf_config.num_key_value_heads = num_key_value_heads
        model_config.hf_text_config.num_key_value_heads = num_key_value_heads

        num_attention_heads = num_key_value_heads * query_heads_per_kv
        model_config.num_attention_heads = num_attention_heads
        model_config.hf_config.num_attention_heads = num_attention_heads
        model_config.hf_text_config.num_attention_heads = num_attention_heads

    intermediate_padding_size = tp_size * get_moe_padding_size(weight_block_size)
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

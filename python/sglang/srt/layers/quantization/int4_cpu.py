import logging
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import Tensor, nn

from sglang.srt.cpu_utils import cpu_has_amx_support
from sglang.srt.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.srt.layers.moe.fused_moe_triton import (
    FusedMoE,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from sglang.srt.layers.moe.topk import select_experts
from sglang.srt.layers.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.utils import set_weight_attrs

try:
    import sgl_kernel.cpu

    _has_amx = cpu_has_amx_support()

except ImportError:
    _has_amx = False

logger = logging.getLogger(__name__)


# NOTE: once we have a CPU INT4 kernel in SGLang, especially with support for
# integer zero-point domain, we can replace torch._int4pack_mm
# TODO: support compressed-tensors format
class Int4CPUConfig(QuantizationConfig):
    _supported_group_sizes = [32, 64, 128, 256]

    """Config class for Int4 weight-only Quantization, backed by PyTorch int4pack_mm.
    """

    def __init__(
        self,
        group_size: int,
        lm_head_quantized: bool,
        modules_to_not_convert: Optional[List[str]],
    ) -> None:
        assert _has_amx
        super().__init__()
        self.group_size = group_size
        self.pack_factor = 8
        self.lm_head_quantized = lm_head_quantized
        self.modules_to_not_convert = modules_to_not_convert or []

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        # technically FP32 and FP16 are also supported, but they have bad perf
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_name(self) -> str:
        return "int4_cpu"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Int4CPUConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        zero_point = cls.get_from_keys(config, ["zero_point"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )

        assert (
            weight_bits == 4
            and group_size in cls._supported_group_sizes
            and zero_point == True
        )
        return cls(group_size, lm_head_quantized, modules_to_not_convert)

    def get_quant_method(
        self,
        layer: nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        not_convert = any(name in prefix for name in self.modules_to_not_convert)

        if isinstance(layer, LinearBase):
            return (
                Int4CPULinearMethod(self)
                if not not_convert
                else UnquantizedLinearMethod()
            )
        elif isinstance(layer, ParallelLMHead):
            return (
                Int4CPULinearMethod(self)
                if self.lm_head_quantized and not not_convert
                else UnquantizedLinearMethod()
            )
        elif isinstance(layer, FusedMoE):
            return Int4CPUMoEMethod(self)

        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        if not _has_amx:
            return None

        quant_name = cls.get_name()
        can_convert = cls.is_compatible(hf_quant_cfg)
        is_valid_user_quant = user_quant is None or user_quant == quant_name

        if can_convert and is_valid_user_quant:
            msg = (
                f"The model is convertible to {quant_name} during runtime."
                f" Using {quant_name} kernel."
            )
            logger.info(msg)
            return cls.get_name()

        if can_convert and user_quant == "awq":
            logger.info(
                f"Detected that the model can run with {quant_name}"
                ", however you specified quantization=awq explicitly,"
                " so forcing awq."
            )

        return None

    @classmethod
    def is_compatible(cls, quant_config: Dict[str, Any]):
        from vllm.platforms import current_platform

        if not current_platform.is_cpu():
            return False

        # Extract data from quant config.
        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("bits")
        group_size = quant_config.get("group_size")
        zero_point = quant_config.get("zero_point")

        if quant_method != "awq":
            return False

        # If we cannot find the info needed in the config, cannot convert.
        if num_bits is None or group_size is None or zero_point is None:
            return False

        return (
            num_bits == 4
            and group_size in cls._supported_group_sizes
            and zero_point == True
        )


class Int4CPULinearMethod(LinearMethodBase):

    def __init__(self, quant_config: Int4CPUConfig):
        self.quant_config = quant_config

    # vllm.model_executor.layers.quantization.awq.AWQLinearMethod
    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        group_size = self.quant_config.group_size
        pack_factor = self.quant_config.pack_factor
        num_groups = input_size_per_partition // group_size

        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                output_size_per_partition // pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=pack_factor,
            weight_loader=weight_loader,
        )

        qzeros = PackedvLLMParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition // pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=pack_factor,
            weight_loader=weight_loader,
        )

        scales = GroupQuantScaleParameter(
            data=torch.empty(
                num_groups,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.num_groups = num_groups

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        qweight, qzeros, scales = _autoawq_to_int4pack(
            layer.qweight.data, layer.qzeros.data, layer.scales.data
        )
        layer.qweight = nn.Parameter(qweight, requires_grad=False)
        layer.qzeros = nn.Parameter(qzeros, requires_grad=False)
        layer.scales = nn.Parameter(scales, requires_grad=False)
        if getattr(layer, "bias", None) is not None:
            layer.bias = nn.Parameter(layer.bias.data.float(), requires_grad=False)

        layer.use_intel_amx_backend = False

    def apply(
        self,
        layer: nn.Module,
        x: Tensor,
        bias: Optional[Tensor] = None,
    ):
        return sgl_kernel.cpu.int4_w4a16_linear(
            x, layer.qweight, layer.qzeros, layer.scales, bias
        )


class Int4CPUMoEMethod(FusedMoEMethodBase):

    def __init__(self, quant_config: Int4CPUConfig):
        self.quant_config = quant_config

    # vllm.model_executor.layers.quantization.awq_marlin.AWQMoEMethod
    def create_weights(
        self,
        layer: nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        extra_weight_attrs.update(
            {
                "is_transposed": True,
                "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
            }
        )

        w13_qweight = nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                2 * intermediate_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        w2_qweight = nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        num_groups_w13 = hidden_size // self.quant_config.group_size
        num_groups_w2 = intermediate_size // self.quant_config.group_size

        # WEIGHT_SCALES
        # Allocate 2 scales for w1 and w3 respectively.
        w13_scales = nn.Parameter(
            torch.empty(
                num_experts, num_groups_w13, intermediate_size * 2, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = nn.Parameter(
            torch.empty(num_experts, num_groups_w2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        # WEIGHT_ZERO_POINT
        # Allocate 2 zero points for w1 and w3 respectively.
        w13_qzeros = nn.Parameter(
            torch.empty(
                num_experts,
                num_groups_w13,
                2 * intermediate_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)

        w2_qzeros = nn.Parameter(
            torch.empty(
                num_experts,
                num_groups_w2,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        w13_qweight, w13_qzeros, w13_scales = _autoawq_to_int4pack(
            layer.w13_qweight.data, layer.w13_qzeros.data, layer.w13_scales.data
        )
        layer.w13_qweight = nn.Parameter(w13_qweight, requires_grad=False)
        layer.w13_qzeros = nn.Parameter(w13_qzeros, requires_grad=False)
        layer.w13_scales = nn.Parameter(w13_scales, requires_grad=False)

        w2_qweight, w2_qzeros, w2_scales = _autoawq_to_int4pack(
            layer.w2_qweight.data, layer.w2_qzeros.data, layer.w2_scales.data
        )
        layer.w2_qweight = nn.Parameter(w2_qweight, requires_grad=False)
        layer.w2_qzeros = nn.Parameter(w2_qzeros, requires_grad=False)
        layer.w2_scales = nn.Parameter(w2_scales, requires_grad=False)

        layer.use_intel_amx_backend = False

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        inplace: bool = True,
        no_combine: bool = False,
    ):
        assert activation == "silu"
        # Expert selection
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
        )
        return sgl_kernel.cpu.fused_experts(
            x,
            layer.w13_qweight,
            layer.w2_qweight,
            topk_weights,
            topk_ids,
            inplace=False,
            use_int4_w4a16=True,
            w1_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
            w1_zero=layer.w13_qzeros,
            w2_zero=layer.w2_qzeros,
        )


def _autoawq_to_int4pack(qweight: Tensor, qzeros: Tensor, scales: Tensor):
    """Convert AutoAWQ weight format to sgl-kernel's CPU int4

    Args:
        qweight: (*, K, N / 8), int32
        qzeros: (*, K / group_size, N / 8), int32
        scales: (*, K / group_size, N), bfloat16
    """
    # unpack from AutoAWQ format
    # https://github.com/casper-hansen/AutoAWQ/blob/23d584c2/awq/modules/triton/gemm.py#L73-L86
    bitshifts = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.int32) * 4
    qweight = (qweight.unsqueeze(-1) >> bitshifts) & 0xF
    qweight = qweight.flatten(-2).transpose(-1, -2).to(torch.uint8)

    # convert to VNNI format: (*, N/BLOCK_N, K/2, BLOCK_N, 2)
    BLOCK_N = 32  # must match what's used in the kernel
    *dims, N, K = qweight.shape
    qweight = qweight.reshape(*dims, N // BLOCK_N, BLOCK_N, K // 2, 2)
    qweight = qweight.transpose(-3, -2)

    # bit packing
    COUNT = 32
    qweight = qweight.reshape(-1, COUNT * 2)
    qweight = (qweight[:, COUNT:] << 4) | qweight[:, :COUNT]
    qweight = qweight.reshape(*dims, N, K // 2)

    qzeros = (qzeros.unsqueeze(-1) >> bitshifts) & 0xF
    qzeros = qzeros.flatten(-2).to(torch.uint8)
    return qweight, qzeros, scales

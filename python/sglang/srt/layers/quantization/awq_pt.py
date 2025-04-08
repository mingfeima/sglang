import logging
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn

from sglang.srt.layers.linear import LinearBase, LinearMethodBase
from sglang.srt.layers.moe.fused_moe_triton import (
    FusedMoE,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from sglang.srt.layers.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.utils import set_weight_attrs

logger = logging.getLogger(__name__)


# NOTE: once we have a CPU INT4 kernel in SGLang, especially with support for
# integer zero-point domain, we can replace torch._int4pack_mm
class AWQPTConfig(QuantizationConfig):
    _supported_group_sizes = [32, 64, 128, 256]

    """Config class for Int4 weight-only Quantization, backed by PyTorch int4pack_mm.
    """

    def __init__(
        self,
        group_size: int,
        lm_head_quantized: bool,
        modules_to_not_convert: Optional[List[str]],
    ) -> None:
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
        return "awq_pt"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AWQPTConfig":
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
        if any(name in prefix for name in self.modules_to_not_convert):
            return None

        if isinstance(layer, LinearBase):
            return AWQPTLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return AWQPTMoEMethod(self)
        elif isinstance(layer, VocabParallelEmbedding):
            if not self.lm_head_quantized:
                return None
            return AWQPTLinearMethod(self)

        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
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


class AWQPTLinearMethod(LinearMethodBase):

    def __init__(self, quant_config: AWQPTConfig):
        self.quant_config = quant_config

    # create weight in AutoAWQ layout
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
        # convert AutoAWQ weight format to PyTorch's int4mm
        qweight = layer.qweight.data  # (K, N / 8)
        qzeros = layer.qzeros.data  # (K / group_size, N / 8)
        scales = layer.scales.data  # (K / group_size, N)
        layer.group_size = qweight.shape[0] // qzeros.shape[0]

        # https://github.com/casper-hansen/AutoAWQ/blob/23d584c2/awq/modules/triton/gemm.py#L73-L86
        bitshifts = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.int32) * 4
        qweight_unpacked = (qweight.unsqueeze(-1) >> bitshifts) & 0xF
        qweight_unpacked = qweight_unpacked.flatten(1).T.contiguous()

        # PT int4pack_mm only supports zero_point in float domain, but AWQ uses integer domain.
        # Hence, we need to convert zero_point to float domain.
        # integer domain: w = (qweight - qzeros) * scales
        #   float domain: w = qweight * scales + zeros - scales * 8
        qzeros_unpacked = (qzeros.unsqueeze(-1) >> bitshifts) & 0xF
        qzeros_unpacked = qzeros_unpacked.flatten(1)
        zeros = ((8 - qzeros_unpacked.float()) * scales.float()).to(scales.dtype)

        if qweight.device.type == "cuda":
            convert_weight = torch._convert_weight_to_int4pack
            self._mm = torch._weight_int4pack_mm
            innerKTiles = 8
        elif qweight.device.type == "cpu":
            convert_weight = torch._convert_weight_to_int4pack_for_cpu
            self._mm = torch._weight_int4pack_mm_for_cpu
            innerKTiles = 1

        qweight = convert_weight(qweight_unpacked, innerKTiles)
        scales_zeros = torch.stack([scales, zeros], dim=-1)  # (K / group_size, N, 2)

        layer.qweight = nn.Parameter(qweight, requires_grad=False)
        del layer.qzeros
        del layer.scales
        layer.scales_zeros = nn.Parameter(scales_zeros, requires_grad=False)

    def apply(
        self,
        layer: nn.Module,
        x: Tensor,
        bias: Optional[Tensor] = None,
    ):
        out = self._mm(
            x, layer.qweight, self.quant_config.group_size, layer.scales_zeros
        )
        if bias is not None:
            out = out + bias
        return out


class AWQPTMoEMethod(FusedMoEMethodBase):

    def __init__(self, quant_config: AWQPTConfig):
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
        pass

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from:
# https://github.com/vllm-project/vllm/blob/fb6af8bc086328ca6659e72d11ffd4309ce4de22/vllm/model_executor/models/deepseek_v2.py
"""Inference-only DeepseekV2 model."""

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.cpu_utils import (
    PackWeightMethod,
    cpu_has_amx_support,
    prepack_weight_if_needed,
)
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.triton_ops.rocm_mla_decode_rope import (
    decode_attention_fwd_grouped_rope,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import EPMoE
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_to_tensor_quant,
    input_to_float8,
    normalize_e4m3fn_to_e4m3fnuz,
)
from sglang.srt.layers.quantization.int8_utils import (
    block_dequant as int8_block_dequant,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope, get_rope_wrapper
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, is_cuda_available, is_hip

if cpu_has_amx_support():
    import sgl_kernel.cpu

    _has_amx = True
else:
    _has_amx = False


_is_hip = is_hip()

if is_cuda_available():
    from sgl_kernel import bmm_fp8


class DeepseekV2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MoEGate(nn.Module):
    def __init__(
        self,
        config,
        prefix: str = "",
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((config.n_routed_experts, config.hidden_size))
        )
        if config.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((config.n_routed_experts))
            )
        else:
            self.e_score_correction_bias = None
        self.quant_method = PackWeightMethod(weight_names=["weight"])
        self.sgl_kernel_cpu_weight_packed_linear = (
            sgl_kernel.cpu.weight_packed_linear if _has_amx else None
        )

    def forward(self, hidden_states):
        if self.use_intel_amx_backend:
            return self.sgl_kernel_cpu_weight_packed_linear(
                hidden_states, self.weight, None
            )

        logits = F.linear(hidden_states, self.weight, None)
        return logits


class DeepseekV2MoE(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.gate = MoEGate(config=config, prefix=add_prefix("gate", prefix))

        MoEImpl = EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE
        self.experts = MoEImpl(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            prefix=add_prefix("experts", prefix),
        )

        # Cache shared_experts and related attributes
        self.shared_experts = None
        self.shared_experts_gate_up_proj = None
        self.shared_experts_down_proj = None
        self.shared_experts_is_int8 = None
        self.shared_experts_is_fp8 = None
        self.shared_experts_weight_block_size = None
        self.experts_impl = None
        self.gate_impl = None
        self.sgl_kernel_cpu_shared_expert = (
            sgl_kernel.cpu.shared_expert if _has_amx else None
        )

        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
            )
            # Cache gate_up_proj and down_proj
            self.shared_experts_gate_up_proj = self.shared_experts.gate_up_proj
            self.shared_experts_down_proj = self.shared_experts.down_proj

            # Cache whether shared_experts is int8
            # quantized models might not have .weight
            if (
                hasattr(self.shared_experts_gate_up_proj, "weight")
                and self.shared_experts_gate_up_proj.weight.dtype == torch.int8
            ):
                assert self.shared_experts_down_proj.weight.dtype == torch.int8
                self.shared_experts_is_int8 = True

            if self.shared_experts_gate_up_proj.weight.dtype == torch.float8_e4m3fn:
                assert self.shared_experts_down_proj.weight.dtype == torch.float8_e4m3fn

                assert (
                    self.shared_experts_gate_up_proj.quant_method.quant_config.weight_block_size
                    == self.shared_experts_down_proj.quant_method.quant_config.weight_block_size
                )
                self.shared_experts_weight_block_size = (
                    self.shared_experts_gate_up_proj.quant_method.quant_config.weight_block_size
                )

                self.shared_experts_is_fp8 = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.gate_impl is None:
            self.gate_impl = self.gate.forward
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate_impl(hidden_states)
        if self.experts_impl is None:
            self.experts_impl = self.experts.forward
        fused_experts_out = self.experts_impl(
            hidden_states=hidden_states, router_logits=router_logits
        )

        # Use cached attributes instead of dynamic access
        gate_up_proj = self.shared_experts_gate_up_proj
        down_proj = self.shared_experts_down_proj
        use_intel_amx_backend = gate_up_proj.use_intel_amx_backend
        shared_experts_is_int8 = self.shared_experts_is_int8
        shared_experts_is_fp8 = self.shared_experts_is_fp8
        shared_experts_weight_block_size = self.shared_experts_weight_block_size
        has_shared_experts = self.n_shared_experts is not None

        assert use_intel_amx_backend == down_proj.use_intel_amx_backend
        if has_shared_experts and use_intel_amx_backend:
            # [Note] inplace should be False in fused_experts.
            # If inplace is True in fused_experts (self.experts), hidden_states will be changed after fused_experts
            # While hidden_states is still needed in shared_expert.
            final_hidden_states = self.sgl_kernel_cpu_shared_expert(
                hidden_states,
                gate_up_proj.weight,
                down_proj.weight,
                fused_experts_out,
                self.routed_scaling_factor,
                inplace=True,
                use_int8_w8a8=shared_experts_is_int8,
                use_fp8_w8a16=shared_experts_is_fp8,
                w1_scale=(
                    gate_up_proj.weight_scale
                    if shared_experts_is_int8
                    else (
                        gate_up_proj.weight_scale_inv if shared_experts_is_fp8 else None
                    )
                ),
                w2_scale=(
                    down_proj.weight_scale
                    if shared_experts_is_int8
                    else down_proj.weight_scale_inv if shared_experts_is_fp8 else None
                ),
                block_size=(
                    shared_experts_weight_block_size if shared_experts_is_fp8 else None
                ),
            )
        else:
            shared_output = None
            if has_shared_experts:
                shared_output = self.shared_experts(hidden_states)
            final_hidden_states = fused_experts_out * self.routed_scaling_factor
            if shared_output is not None:
                final_hidden_states = final_hidden_states + shared_output
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV2Attention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        layer_id=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_a_proj", prefix),
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_b_proj", prefix),
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_proj", prefix),
            )

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_a_proj_with_mqa", prefix),
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_b_proj", prefix),
        )
        # O projection.
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        rope_scaling["rope_type"] = "deepseek_yarn"
        self.rotary_emb = get_rope_wrapper(
            qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=False,
            device=global_server_args_dict["device"],
        )

        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        # TODO, support head_size 192
        self.attn = RadixAttention(
            self.num_local_heads,
            256,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a.contiguous())
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = latent_cache[:, :, self.kv_lora_rank :]
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q[..., self.qk_nope_head_dim :] = q_pe
        k = torch.empty_like(q)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe
        q = torch.nn.functional.pad(q, [0, 256 - self.qk_head_dim], value=0).view(
            -1, self.num_local_heads * 256
        )
        k = torch.nn.functional.pad(k, [0, 256 - self.qk_head_dim], value=0).view(
            -1, self.num_local_heads * 256
        )
        v = torch.nn.functional.pad(v, [0, 256 - self.v_head_dim], value=0).view(
            -1, self.num_local_heads * 256
        )
        attn_output = self.attn(q, k, v, forward_batch)
        attn_output = attn_output.view(-1, self.num_local_heads, 256)[
            ..., : self.v_head_dim
        ].reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output


class MParams:
    def __init__(self):
        self.q_lora_rank: Optional[int] = None
        self.q_a_proj: Optional[nn.Module] = None
        self.q_b_proj: Optional[nn.Module] = None
        self.kv_a_proj_with_mqa: Optional[nn.Module] = None
        self.q_a_layernorm: Optional[nn.Module] = None
        self.kv_a_layernorm: Optional[nn.Module] = None
        self.rotary_emb: Optional[nn.Module] = None
        self.qkv_proj_with_rope_is_int8: Optional[bool] = None
        self.qkv_proj_with_rope_is_fp8: Optional[bool] = None
        self.num_local_heads: Optional[int] = None
        self.kv_lora_rank: Optional[int] = None
        self.weight_block_size: Optional[List[int]] = None


class DeepseekV2AttentionMLA(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        layer_id=None,
        use_dp=False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads if use_dp else num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.o_proj_is_int8 = None
        self.o_proj_is_fp8 = None
        self.o_proj_weight_block_size = None

        if use_dp:
            # For data parallel attention
            if self.q_lora_rank is not None:
                self.q_a_proj = ReplicatedLinear(
                    self.hidden_size,
                    self.q_lora_rank,
                    bias=False,
                    quant_config=quant_config,
                    prefix=add_prefix("q_a_proj", prefix),
                )
                self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
                self.q_b_proj = ReplicatedLinear(
                    q_lora_rank,
                    self.num_heads * self.qk_head_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=add_prefix("q_b_proj", prefix),
                )
            else:
                self.q_proj = ReplicatedLinear(
                    self.hidden_size,
                    self.num_heads * self.qk_head_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=add_prefix("q_proj", prefix),
                )
            self.kv_b_proj = ReplicatedLinear(
                self.kv_lora_rank,
                self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("kv_b_proj", prefix),
            )
            # O projection.
            self.o_proj = ReplicatedLinear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("o_proj", prefix),
            )
        else:
            # For tensor parallel attention
            if self.q_lora_rank is not None:
                self.q_a_proj = ReplicatedLinear(
                    self.hidden_size,
                    self.q_lora_rank,
                    bias=False,
                    quant_config=quant_config,
                    prefix=add_prefix("q_a_proj", prefix),
                )
                self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
                self.q_b_proj = ColumnParallelLinear(
                    q_lora_rank,
                    self.num_heads * self.qk_head_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=add_prefix("q_b_proj", prefix),
                )
            else:
                self.q_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.num_heads * self.qk_head_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=add_prefix("q_proj", prefix),
                )
            self.kv_b_proj = ColumnParallelLinear(
                self.kv_lora_rank,
                self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("kv_b_proj", prefix),
            )
            # O projection.
            self.o_proj = RowParallelLinear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("o_proj", prefix),
            )
            assert self.o_proj.input_is_parallel
            assert self.o_proj.reduce_results
            if self.o_proj.weight.dtype == torch.int8:
                self.o_proj_is_int8 = True
            if self.o_proj.weight.dtype == torch.float8_e4m3fn:
                self.o_proj_is_fp8 = True
                self.o_proj_weight_block_size = (
                    self.o_proj.quant_method.quant_config.weight_block_size
                )

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_a_proj_with_mqa", prefix),
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)

        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        self.rotary_emb = get_rope_wrapper(
            qk_rope_head_dim,
            rotary_dim=qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=False,
            device=global_server_args_dict["device"],
        )

        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale
        else:
            self.rotary_emb.forward = self.rotary_emb.forward_native

        self.attn_mqa = RadixAttention(
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.kv_lora_rank,
            prefix=add_prefix("attn_mqa", prefix),
        )

        self.attn_mha = RadixAttention(
            self.num_local_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
            prefix=add_prefix("attn_mha", prefix),
        )

        self.w_kc = None
        self.w_vc = None
        self.w_scale = None

        if cpu_has_amx_support():
            self.quant_method = PackWeightMethod(
                weight_names=["w_kc", "w_vc"], transpose_dims=[[1, 2], [1, 2]]
            )
        else:
            self.quant_method = None
            self.use_intel_amx_backend = False

        self.qkv_proj_with_rope_is_int8 = None
        self.qkv_proj_with_rope_is_fp8 = None
        self.weight_block_size = None
        if self.q_lora_rank is not None:
            # quantized models might not have .weight
            if (
                hasattr(self.q_a_proj, "weight")
                and self.q_a_proj.weight.dtype == torch.int8
            ):
                assert (
                    self.q_a_proj.weight.dtype
                    == self.q_b_proj.weight.dtype
                    == self.kv_a_proj_with_mqa.weight.dtype
                )
                self.qkv_proj_with_rope_is_int8 = True
            if self.q_a_proj.weight.dtype == torch.float8_e4m3fn:
                assert (
                    self.q_a_proj.weight.dtype
                    == self.q_b_proj.weight.dtype
                    == self.kv_a_proj_with_mqa.weight.dtype
                )
                assert (
                    self.q_a_proj.quant_method.quant_config.weight_block_size
                    == self.q_b_proj.quant_method.quant_config.weight_block_size
                    == self.kv_a_proj_with_mqa.quant_method.quant_config.weight_block_size
                )
                self.weight_block_size = (
                    self.q_a_proj.quant_method.quant_config.weight_block_size
                )
                self.qkv_proj_with_rope_is_fp8 = True

        params = MParams()
        params.q_lora_rank = self.q_lora_rank
        if params.q_lora_rank is not None:
            # TODO: only cache weight instead of modules. We need to cache weight after weight packing
            params.q_a_proj = self.q_a_proj
            params.q_b_proj = self.q_b_proj
            params.q_a_layernorm = self.q_a_layernorm
            params.q_a_proj_weight_scale = (
                params.q_a_proj.weight_scale
                if self.qkv_proj_with_rope_is_int8
                else None
            )
            params.q_b_proj_weight_scale = (
                params.q_b_proj.weight_scale
                if self.qkv_proj_with_rope_is_int8
                else None
            )

        params.kv_a_proj_with_mqa = self.kv_a_proj_with_mqa
        params.kv_a_proj_with_mqa_weight_scale = (
            params.kv_a_proj_with_mqa.weight_scale
            if self.qkv_proj_with_rope_is_int8
            else None
        )
        params.kv_a_layernorm = self.kv_a_layernorm
        params.rotary_emb = self.rotary_emb
        params.qkv_proj_with_rope_is_int8 = self.qkv_proj_with_rope_is_int8
        params.qkv_proj_with_rope_is_fp8 = self.qkv_proj_with_rope_is_fp8
        params.num_local_heads = self.num_local_heads
        params.kv_lora_rank = self.kv_lora_rank
        params.weight_block_size = self.weight_block_size

        params.attn_mqa_layer_id = self.attn_mqa.layer_id
        params.attn_mqa_scaling = self.attn_mqa.scaling
        params.attn_mqa_logit_cap = self.attn_mqa.logit_cap
        params.attn_mqa_tp_k_head_num = self.attn_mqa.tp_k_head_num
        params.attn_mqa_qk_head_dim = self.attn_mqa.qk_head_dim
        params.attn_mqa_tp_v_head_num = self.attn_mqa.tp_v_head_num
        params.attn_mqa_v_head_dim = self.attn_mqa.v_head_dim
        params.attn_mqa_tp_q_head_num = self.attn_mqa.tp_q_head_num

        params.o_proj_tp_size = self.o_proj.tp_size
        params.o_proj_tp_rank = self.o_proj.tp_rank
        params.o_proj_is_int8 = self.o_proj_is_int8
        params.o_proj_is_fp8 = self.o_proj_is_fp8
        params.o_proj_weight_block_size = self.o_proj_weight_block_size
        params.o_proj_scale = (
            self.o_proj.weight_scale
            if params.o_proj_is_int8
            else self.o_proj.weight_scale_inv if params.o_proj_is_fp8 else None
        )
        params.device_group = (
            get_tp_group().device_group if params.o_proj_tp_size > 1 else None
        )
        params.reduce_op = (
            torch.distributed.ReduceOp.SUM if params.o_proj_tp_size > 1 else None
        )
        self.params = params

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        def no_absorb() -> bool:
            if global_server_args_dict["enable_flashinfer_mla"]:
                # Flashinfer MLA: Do not absorb when enabling ragged prefill
                return (
                    not global_server_args_dict["flashinfer_mla_disable_ragged"]
                    and forward_batch.forward_mode.is_extend()
                    and not forward_batch.forward_mode.is_target_verify()
                    and not forward_batch.forward_mode.is_draft_extend()
                    and forward_batch.extend_prefix_lens.sum() == 0
                )
            else:
                # Triton: Use normal computation for prefill and use weight absorption for extend/decode
                return (
                    forward_batch.forward_mode.is_extend()
                    and not forward_batch.forward_mode.is_target_verify()
                    and not forward_batch.forward_mode.is_draft_extend()
                    and forward_batch.extend_prefix_lens.sum() == 0
                )

        if no_absorb():
            return self.forward_normal(positions, hidden_states, forward_batch)
        else:
            if _is_hip:
                if (
                    os.getenv("SGLANG_ROCM_FUSED_DECODE_MLA") == "1"
                    and forward_batch.forward_mode.is_decode()
                ):
                    return self.forward_absorb_fused_mla_rope(
                        positions, hidden_states, forward_batch
                    )
                else:
                    return self.forward_absorb(positions, hidden_states, forward_batch)
            else:
                if self.q_lora_rank is not None and self.use_intel_amx_backend:
                    if forward_batch.forward_mode.is_decode():
                        return self.forward_absorb_decode_fused_cpu(
                            positions, hidden_states, forward_batch
                        )
                    else:
                        return self.forward_absorb_fused_mla_rope_cpu(
                            positions, hidden_states, forward_batch
                        )
                else:
                    return self.forward_absorb(positions, hidden_states, forward_batch)

    def forward_normal(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a.contiguous())
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]
        k_pe = latent_cache[:, :, self.kv_lora_rank :]
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q[..., self.qk_nope_head_dim :] = q_pe
        k = torch.empty_like(q)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe

        latent_cache[:, :, : self.kv_lora_rank] = kv_a.unsqueeze(1)
        latent_cache[:, :, self.kv_lora_rank :] = k_pe

        # Save latent cache
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn_mha, forward_batch.out_cache_loc, latent_cache, None
        )
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output

    def forward_absorb(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        q_len = hidden_states.shape[0]
        q_input = hidden_states.new_empty(
            q_len, self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim
        )
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        if self.w_kc.dtype == torch.float8_e4m3fnuz:
            # TODO(kernel): add bmm_fp8 for torch.float8_e4m3fnuz
            q_nope_out = torch.bmm(
                q_nope.to(torch.bfloat16).transpose(0, 1),
                self.w_kc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_kc.dtype == torch.float8_e4m3fn:
            q_nope_val, q_nope_scale = input_to_float8(
                q_nope.transpose(0, 1), torch.float8_e4m3fn
            )
            q_nope_out = bmm_fp8(
                q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
            )
        else:
            if self.use_intel_amx_backend:
                # [Note] Align shapes of bmm inputs.
                # Shapes of inputs:
                #   q_nope: [M, B, K]
                #   original self.w_kc: [B, K, N]
                #   current self.w_kc (which has been converted in PackWeightMethod): [B, N, K]

                # Shapes of inputs to sgl_kernel.cpu.bmm:
                #   out: [B, M, N]
                #   mat1: [B, M, K]
                #   mat2: [B, N, K]
                B = self.w_kc.size(0)
                N = self.w_kc.size(1)
                M = q_nope.size(0)

                q_nope_out = q_input[..., : self.kv_lora_rank].transpose_(0, 1)
                sgl_kernel.cpu.bmm(q_nope_out, q_nope.transpose(0, 1), self.w_kc)
            else:
                q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
                q_input[..., : self.kv_lora_rank] = q_nope_out.transpose(0, 1)

        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        v_input = latent_cache[..., : self.kv_lora_rank]
        v_input = self.kv_a_layernorm(v_input.contiguous()).unsqueeze(1)
        k_input = latent_cache.unsqueeze(1)
        k_input[..., : self.kv_lora_rank] = v_input
        k_pe = k_input[..., self.kv_lora_rank :]

        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q_input[..., self.kv_lora_rank :] = q_pe
        k_input[..., self.kv_lora_rank :] = k_pe

        attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        if self.w_vc.dtype == torch.float8_e4m3fnuz:
            # TODO(kernel): add bmm_fp8 for torch.float8_e4m3fnuz
            attn_bmm_output = torch.bmm(
                attn_output.to(torch.bfloat16).transpose(0, 1),
                self.w_vc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = input_to_float8(
                attn_output.transpose(0, 1), torch.float8_e4m3fn
            )
            attn_bmm_output = bmm_fp8(
                attn_output_val,
                self.w_vc,
                attn_output_scale,
                self.w_scale,
                torch.bfloat16,
            )
        else:
            if self.use_intel_amx_backend:
                # See [Note] Align shapes of bmm inputs.
                B = self.w_vc.size(0)
                N = self.w_vc.size(1)
                M = attn_output.size(0)
                output = torch.empty([M, int(B * N)], dtype=attn_output.dtype)
                attn_bmm_output = output.view([M, B, N]).transpose_(0, 1)
                sgl_kernel.cpu.bmm(
                    attn_bmm_output, attn_output.transpose(0, 1), self.w_vc
                )
                attn_output = output
            else:
                attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)
                attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        output, _ = self.o_proj(attn_output)

        return output

    def forward_absorb_fused_mla_rope(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        enable_rope_fusion = (
            os.getenv("SGLANG_FUSED_MLA_ENABLE_ROPE_FUSION", "1") == "1"
        )
        q_len = hidden_states.shape[0]
        q_input = hidden_states.new_empty(
            q_len, self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim
        )
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        if self.w_kc.dtype == torch.float8_e4m3fnuz:
            # TODO(kernel): add bmm_fp8 for torch.float8_e4m3fnuz
            q_nope_out = torch.bmm(
                q_nope.to(torch.bfloat16).transpose(0, 1),
                self.w_kc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_kc.dtype == torch.float8_e4m3fn:
            q_nope_val, q_nope_scale = input_to_float8(
                q_nope.transpose(0, 1), torch.float8_e4m3fn
            )
            q_nope_out = bmm_fp8(
                q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
            )
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
        q_input[..., : self.kv_lora_rank] = q_nope_out.transpose(0, 1)

        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        v_input = latent_cache[..., : self.kv_lora_rank]
        v_input = self.kv_a_layernorm(v_input.contiguous()).unsqueeze(1)
        k_input = latent_cache.unsqueeze(1)
        k_input[..., : self.kv_lora_rank] = v_input

        if not enable_rope_fusion:
            k_pe = k_input[..., self.kv_lora_rank :]
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
            q_input[..., self.kv_lora_rank :] = q_pe
            k_input[..., self.kv_lora_rank :] = k_pe
            k_pe_output = None
        else:
            k_pe_output = torch.empty_like(k_input[..., self.kv_lora_rank :])

        q_input[..., self.kv_lora_rank :] = q_pe

        # attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
        # Use Fused ROPE with use_rope=OFF.
        attn_output = torch.empty(
            (q_len, self.num_local_heads, self.kv_lora_rank),
            dtype=q.dtype,
            device=q.device,
        )
        attn_logits, _, kv_indptr, kv_indices, _, _, _ = (
            forward_batch.attn_backend.forward_metadata
        )
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        num_kv_split = forward_batch.attn_backend.num_kv_splits
        sm_scale = self.attn_mqa.scaling
        if attn_logits is None:
            attn_logits = torch.empty(
                (
                    forward_batch.batch_size,
                    self.num_local_heads,
                    num_kv_split,
                    self.kv_lora_rank + 1,
                ),
                dtype=torch.float32,
                device=q.device,
            )

        # save current latent cache.
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn_mqa, forward_batch.out_cache_loc, k_input, None
        )
        key_cache_buf = forward_batch.token_to_kv_pool.get_key_buffer(
            self.attn_mqa.layer_id
        )
        val_cache_buf = key_cache_buf[..., : self.kv_lora_rank]

        decode_attention_fwd_grouped_rope(
            q_input,
            key_cache_buf,
            val_cache_buf,
            attn_output,
            kv_indptr,
            kv_indices,
            k_pe_output,
            self.kv_lora_rank,
            self.rotary_emb.rotary_dim,
            cos_sin_cache,
            positions,
            attn_logits,
            num_kv_split,
            sm_scale,
            logit_cap=self.attn_mqa.logit_cap,
            use_rope=enable_rope_fusion,
            is_neox_style=self.rotary_emb.is_neox_style,
        )

        if enable_rope_fusion:
            k_input[..., self.kv_lora_rank :] = k_pe_output
            forward_batch.token_to_kv_pool.set_kv_buffer(
                self.attn_mqa, forward_batch.out_cache_loc, k_input, None
            )

        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        if self.w_vc.dtype == torch.float8_e4m3fnuz:
            # TODO(kernel): add bmm_fp8 for torch.float8_e4m3fnuz
            attn_bmm_output = torch.bmm(
                attn_output.to(torch.bfloat16).transpose(0, 1),
                self.w_vc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = input_to_float8(
                attn_output.transpose(0, 1), torch.float8_e4m3fn
            )
            attn_bmm_output = bmm_fp8(
                attn_output_val,
                self.w_vc,
                attn_output_scale,
                self.w_scale,
                torch.bfloat16,
            )
        else:
            attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)
        attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        output, _ = self.o_proj(attn_output)

        return output

    def forward_absorb_fused_mla_rope_cpu(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        params = self.params
        assert (
            params.q_lora_rank is not None and self.use_intel_amx_backend
        ), "forward_absorb_fused_mla_rope_cpu requires q_lora_rank is not None and use_intel_amx_backend"
        q_input, k_input, v_input = sgl_kernel.cpu.qkv_proj_with_rope(
            hidden_states,
            params.q_a_proj.weight,
            params.q_b_proj.weight,
            params.kv_a_proj_with_mqa.weight,
            self.w_kc,
            params.q_a_layernorm.weight,
            params.kv_a_layernorm.weight,
            positions,
            params.rotary_emb.cos_sin_cache,
            params.kv_a_layernorm.variance_epsilon,
            use_int8_w8a8=params.qkv_proj_with_rope_is_int8,
            use_fp8_w8a16=params.qkv_proj_with_rope_is_fp8,
            q_a_proj_scale=(
                params.q_a_proj.weight_scale
                if params.qkv_proj_with_rope_is_int8
                else (
                    params.q_a_proj.weight_scale_inv
                    if params.qkv_proj_with_rope_is_fp8
                    else None
                )
            ),
            q_b_proj_scale=(
                params.q_b_proj.weight_scale
                if params.qkv_proj_with_rope_is_int8
                else (
                    params.q_b_proj.weight_scale_inv
                    if params.qkv_proj_with_rope_is_fp8
                    else None
                )
            ),
            kv_a_proj_scale=(
                params.kv_a_proj_with_mqa.weight_scale
                if params.qkv_proj_with_rope_is_int8
                else (
                    params.kv_a_proj_with_mqa.weight_scale_inv
                    if params.qkv_proj_with_rope_is_fp8
                    else None
                )
            ),
            weight_block_size=(
                params.weight_block_size if params.qkv_proj_with_rope_is_fp8 else None
            ),
        )
        attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
        attn_output = attn_output.view(-1, params.num_local_heads, params.kv_lora_rank)

        w_vc = self.w_vc
        w_scale = self.w_scale
        if w_vc.dtype == torch.float8_e4m3fnuz:
            # TODO(kernel): add bmm_fp8 for torch.float8_e4m3fnuz
            attn_bmm_output = torch.bmm(
                attn_output.to(torch.bfloat16).transpose(0, 1),
                w_vc.to(torch.bfloat16) * w_scale,
            )
        elif w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = input_to_float8(
                attn_output.transpose(0, 1), torch.float8_e4m3fn
            )
            attn_bmm_output = bmm_fp8(
                attn_output_val,
                w_vc,
                attn_output_scale,
                w_scale,
                torch.bfloat16,
            )
        else:
            # See [Note] Align shapes of bmm inputs.
            B = w_vc.size(0)
            N = w_vc.size(1)
            M = attn_output.size(0)
            output = torch.empty([M, int(B * N)], dtype=attn_output.dtype)
            attn_bmm_output = output.view([M, B, N]).transpose_(0, 1)
            sgl_kernel.cpu.bmm(attn_bmm_output, attn_output.transpose(0, 1), w_vc)
            attn_output = output
        output, _ = self.o_proj(attn_output)

        return output

    def forward_absorb_decode_fused_cpu(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        params = self.params
        assert (
            params.q_lora_rank is not None and self.use_intel_amx_backend
        ), "forward_absorb_decode_fused_cpu requires q_lora_rank is not None and use_intel_amx_backend"

        # TODO: add FP8 support
        assert self.w_vc.dtype not in [torch.float8_e4m3fnuz, torch.float8_e4m3fn]
        output = sgl_kernel.cpu.forward_absorb_decode_fused(
            hidden_states,
            params.q_a_proj.weight,
            params.q_b_proj.weight,
            params.kv_a_proj_with_mqa.weight,
            self.w_kc,
            params.q_a_layernorm.weight,
            params.kv_a_layernorm.weight,
            positions,
            params.rotary_emb.cos_sin_cache,
            forward_batch.token_to_kv_pool.get_key_buffer(params.attn_mqa_layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(params.attn_mqa_layer_id),
            forward_batch.out_cache_loc,
            forward_batch.attn_backend.forward_metadata[0],  # attn_logits
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            self.w_vc,
            self.o_proj.weight,
            self.o_proj.bias,
            params.kv_a_layernorm.variance_epsilon,
            params.qkv_proj_with_rope_is_int8,
            params.attn_mqa_scaling,
            params.attn_mqa_logit_cap,
            params.attn_mqa_tp_k_head_num,
            params.attn_mqa_qk_head_dim,
            params.attn_mqa_tp_v_head_num,
            params.attn_mqa_v_head_dim,
            params.attn_mqa_tp_q_head_num,
            params.num_local_heads,
            params.kv_lora_rank,
            params.o_proj_tp_size,
            params.o_proj_tp_rank,
            params.o_proj_is_int8,
            params.o_proj_is_fp8,
            hidden_states.dtype,
            params.q_a_proj_weight_scale,
            params.q_b_proj_weight_scale,
            params.kv_a_proj_with_mqa_weight_scale,
            None,  # bmm_scale
            params.device_group,
            params.reduce_op,
            params.o_proj_scale,
            params.o_proj_weight_block_size,
        )

        return output


def all_gather(
    input_tensor: torch.Tensor, forward_batch: ForwardBatch, rank, world_size, group
):
    all_lens = forward_batch.global_num_tokens_cpu
    max_len = max(forward_batch.global_num_tokens_cpu)

    if world_size == 1:
        return input_tensor, 0, all_lens[0]

    padded_tensor = torch.nn.functional.pad(
        input_tensor, (0, 0, 0, max_len - input_tensor.shape[0])
    )

    group.all_gather_into_tensor(forward_batch.gathered_buffer, padded_tensor)

    gathered_tensors = torch.concat(
        [
            forward_batch.gathered_buffer[i * max_len : i * max_len + all_lens[i]]
            for i in range(world_size)
        ]
    )

    start_index = 0 if rank == 0 else sum(all_lens[:rank])
    end_index = start_index + all_lens[rank]

    return gathered_tensors, start_index, end_index


class DeepseekV2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.enable_dp_attention = (
            not global_server_args_dict["disable_mla"]
            and global_server_args_dict["enable_dp_attention"]
        )
        if self.enable_dp_attention:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.tp_group = get_tp_group()
        if not global_server_args_dict["disable_mla"]:
            self.self_attn = DeepseekV2AttentionMLA(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=(
                    config.q_lora_rank if hasattr(config, "q_lora_rank") else None
                ),
                kv_lora_rank=config.kv_lora_rank,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                layer_id=layer_id,
                use_dp=self.enable_dp_attention,
                prefix=add_prefix("self_attn", prefix),
            )
        else:
            self.self_attn = DeepseekV2Attention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=(
                    config.q_lora_rank if hasattr(config, "q_lora_rank") else None
                ),
                kv_lora_rank=config.kv_lora_rank,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                layer_id=layer_id,
                prefix=add_prefix("self_attn", prefix),
            )
        if is_nextn or (
            config.n_routed_experts is not None
            and layer_id >= config.first_k_dense_replace
            and layer_id % config.moe_layer_freq == 0
        ):
            self.mlp = DeepseekV2MoE(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention
        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )

        # Fully Connected
        if self.enable_dp_attention:
            hidden_states, start_idx, end_idx = all_gather(
                hidden_states, forward_batch, self.tp_rank, self.tp_size, self.tp_group
            )
            hidden_states = self.mlp(hidden_states)
            hidden_states = hidden_states[start_idx:end_idx]
        else:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class DeepseekV2Model(nn.Module):

    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not global_server_args_dict["enable_dp_attention"],
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(
                    config,
                    layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        layers = self.layers
        for i in range(len(layers)):
            layer = layers[i]
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )
        if not forward_batch.forward_mode.is_idle():
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


def _awq_dequantize(qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor):
    if qweight.is_cuda:
        from vllm import _custom_ops as ops

        # on CPU, this is not available
        return ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)

    # qweight: (K, N / 8), int32
    # qzeros: (K / group_size, N / 8), int32
    # scales: (K / group_size, N), bfloat16

    # https://github.com/casper-hansen/AutoAWQ/blob/23d584c2/awq/modules/triton/gemm.py#L73-L86
    bitshifts = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.int32) * 4
    qweight_unpacked = (qweight.unsqueeze(-1) >> bitshifts) & 0xF
    qweight_unpacked = qweight_unpacked.flatten(-2)  # (K, N)

    qzeros_unpacked = (qzeros.unsqueeze(-1) >> bitshifts) & 0xF
    qzeros_unpacked = qzeros_unpacked.flatten(-2)  # (K / group_size, N)

    num_groups = qzeros.shape[0]
    qweight_unpacked = qweight_unpacked.unflatten(0, (num_groups, -1))
    qweight = qweight_unpacked - qzeros_unpacked.unsqueeze(1)
    weight = qweight.float() * scales.unsqueeze(1).float()
    weight = weight.flatten(0, 1).to(scales.dtype)
    return weight


class DeepseekV2ForCausalLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = DeepseekV2Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        if global_server_args_dict["enable_dp_attention"]:
            # TODO: use weight_packed_linear for enable_dp_attention
            self.lm_head = ReplicatedLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                prefix=add_prefix("lm_head", prefix),
            )
            self.logits_processor = LogitsProcessor(config, skip_all_gather=True)
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
            self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        MoEImpl = EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE
        expert_params_mapping = MoEImpl.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            # TODO(HandH1998): Modify it when nextn is supported.
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                if num_nextn_layers > 0 and name.startswith("model.layers"):
                    name_list = name.split(".")
                    if (
                        len(name_list) >= 3
                        and int(name_list[2]) >= self.config.num_hidden_layers
                    ):
                        continue
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

        if not global_server_args_dict["disable_mla"]:
            for layer_id in range(self.config.num_hidden_layers):
                self_attn = self.model.layers[layer_id].self_attn
                if hasattr(self_attn.kv_b_proj, "qweight"):
                    # AWQ compatible
                    w = _awq_dequantize(
                        self_attn.kv_b_proj.qweight,
                        self_attn.kv_b_proj.scales,
                        self_attn.kv_b_proj.qzeros,
                    ).T
                else:
                    w = self_attn.kv_b_proj.weight
                # NOTE(HandH1998): Since `bmm_fp8` only supports per-tensor scale, we have to requantize `self_attn.kv_b_proj`.
                # This may affect the accuracy of fp8 model.
                if hasattr(self.quant_config, "weight_block_size") and w.dtype in (
                    torch.float8_e4m3fn,
                    torch.float8_e4m3fnuz,
                ):
                    weight_block_size = self.quant_config.weight_block_size
                    if weight_block_size is not None:
                        assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                        if _is_hip:
                            weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                                weight=w,
                                weight_scale=self_attn.kv_b_proj.weight_scale_inv,
                                input_scale=None,
                            )
                        else:
                            weight = w
                            weight_scale = self_attn.kv_b_proj.weight_scale_inv

                        w, scale = block_quant_to_tensor_quant(
                            weight, weight_scale, weight_block_size
                        )
                        self_attn.w_scale = scale
                if w.dtype == torch.int8:
                    if hasattr(self.quant_config, "weight_block_size"):
                        # block-wise int8 need it
                        weight_block_size = self.quant_config.weight_block_size
                        if weight_block_size is not None:
                            assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                            weight = w
                            weight_scale = self_attn.kv_b_proj.weight_scale_inv
                            w = int8_block_dequant(
                                weight, weight_scale, weight_block_size
                            ).to(torch.bfloat16)
                    else:
                        # channel-wise int8 need it
                        w = w.to(torch.bfloat16) * self_attn.kv_b_proj.weight_scale.to(
                            torch.bfloat16
                        )
                w_kc, w_vc = w.unflatten(
                    0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
                ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
                self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
                self_attn.w_vc = w_vc.contiguous().transpose(1, 2)
                if (
                    hasattr(self_attn.kv_b_proj, "weight_scale")
                    and self_attn.w_scale is None
                ):
                    self_attn.w_scale = self_attn.kv_b_proj.weight_scale
                    if _is_hip:
                        self_attn.w_scale *= 2.0
                if (
                    w_kc.device == torch.device("cpu")
                    and cpu_has_amx_support()
                    and w.dtype == torch.float8_e4m3fn
                ):
                    self_attn.w_kc = (
                        self_attn.w_kc.to(torch.bfloat16) * self_attn.w_scale
                    )
                    self_attn.w_vc = (
                        self_attn.w_vc.to(torch.bfloat16) * self_attn.w_scale
                    )

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass


EntryClass = [DeepseekV2ForCausalLM, DeepseekV3ForCausalLM]

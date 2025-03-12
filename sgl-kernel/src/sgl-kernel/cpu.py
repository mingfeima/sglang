import sgl_kernel
import torch


def fused_experts(
    x,
    w13_weight,
    w2_weight,
    topk_weights,
    topk_ids,
    inplace,
    is_vnni,
):
    return sgl_kernel.ops._kernels.fused_experts_cpu(
        x,
        w13_weight,
        w2_weight,
        topk_weights,
        topk_ids,
        inplace,
        is_vnni,
    )


def convert_weight_packed(weight):
    return sgl_kernel.ops._kernels.convert_weight_packed(weight)

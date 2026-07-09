from .paged_mqa_logits import (
    aiter_paged_mqa_logits,
    cutedsl_paged_mqa_logits,
    deepgemm_paged_mqa_logits_native,
    deepgemm_paged_mqa_logits_split,
)

_LAZY_CUTEDSL_EXPORTS = {
    "CuteDSLPagedMQALogitsRunner",
    "pick_dsl_expand",
}


def __getattr__(name: str):
    if name in _LAZY_CUTEDSL_EXPORTS:
        from . import cutedsl_paged_mqa_logits as _cutedsl

        return getattr(_cutedsl, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CuteDSLPagedMQALogitsRunner",
    "pick_dsl_expand",
    "aiter_paged_mqa_logits",
    "cutedsl_paged_mqa_logits",
    "deepgemm_paged_mqa_logits_native",
    "deepgemm_paged_mqa_logits_split",
]

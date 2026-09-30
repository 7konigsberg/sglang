"""Config-time override declarations for ernie45_vl."""

from typing import Any

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    resolving_view,
)


@_register_for("Ernie4_5_VLMoeForConditionalGeneration")
def _ernie45_vl_overrides(server_args: Any, hf_config: Any) -> dict:
    if resolving_view(server_args).enable_dp_attention:
        raise ValueError(
            "ERNIE 4.5 VL MoE does not support DP attention: its attention "
            "heads are split over the full TP group."
        )
    return {}

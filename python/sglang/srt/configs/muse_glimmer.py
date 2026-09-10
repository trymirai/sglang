# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Text configuration for Meta's native Muse Glimmer HF checkpoints."""

import math

from transformers import PretrainedConfig


class MuseGlimmerAssistantConfig(PretrainedConfig):
    model_type = "muse_glimmer_assistant"
    vocab_size = 202048
    is_causal = False


class MuseGlimmerConfig(PretrainedConfig):
    model_type = "muse_glimmer"

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        if text_config is not None:
            text = dict(text_config)
            text.pop("model_type", None)
            kwargs.update(text)
            kwargs["hidden_act"] = kwargs.pop("hidden_activation")
            kwargs["no_rope_layers"] = [
                bool(theta) for theta in kwargs.pop("layer_rope_theta")
            ]
            kwargs["rope_theta"] = kwargs["rope_parameters"]["rope_theta"]
            kwargs["output_soft_cap_temp"] = kwargs.pop("final_logit_softcapping")
            # Runtime uses scale / head_dim; HF uses scale / sqrt(head_dim).
            kwargs["qk_scale_factor"] *= math.sqrt(kwargs["head_dim"])
        kwargs.setdefault("use_qk_norm", True)
        kwargs.setdefault("use_attn_output_gate", True)
        kwargs.setdefault("normalize_tok_embeddings", True)
        kwargs.setdefault("rope_is_neox_style", True)
        super().__init__(**kwargs)

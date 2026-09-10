# Copyright 2023-2026 SGLang Team
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

import logging
import re
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
)
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix, is_cuda

_is_cuda = is_cuda()

if _is_cuda:
    from sglang.srt.layers.elementwise import fused_sigmoid_mul

logger = logging.getLogger(__name__)

_VISION_NAME_FRAGMENTS = (
    "vision_encoder",
    "vision_tower",
    "vision_adapter",
    "vision_projection",
    "perception_emb_norm",
)

# Vendor tensor names -> this port's; applied simultaneously.
_VENDOR_RENAMES = {
    "post_attention_layernorm": "post_attn_norm",
    "pre_feedforward_layernorm": "post_attention_layernorm",
    "post_feedforward_layernorm": "post_ffn_norm",
    "self_attn.gate_proj": "self_attn.output_gate_proj",
}

_VENDOR_RENAME_RE = re.compile("|".join(re.escape(key) for key in _VENDOR_RENAMES))


def _vendor_weight_name(name: str) -> str:
    name = name.replace("model.language_model.", "model.", 1)
    return _VENDOR_RENAME_RE.sub(lambda m: _VENDOR_RENAMES[m.group(0)], name)


def get_attention_sliding_window_size(config) -> int:
    return config.sliding_window - 1


class MuseGlimmerMLP(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        if config.hidden_act != "silu":
            raise ValueError(
                f"Muse Glimmer expects hidden_act=silu, got {config.hidden_act}"
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MuseGlimmerAttention(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        tp_size = get_parallel().tp_size
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.scaling = config.qk_scale_factor / self.head_dim

        self.use_rope = config.no_rope_layers[layer_id] == 1
        self.is_sliding = config.layer_types[layer_id] == "sliding_attention"
        self.use_qk_norm = config.use_qk_norm
        self.use_output_gate = config.use_attn_output_gate

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        if self.use_output_gate:
            self.output_gate_proj = ColumnParallelLinear(
                config.hidden_size,
                self.total_num_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("output_gate_proj", prefix),
            )

        self.qk_norm = (
            RMSNorm(
                hidden_size=self.head_dim,
                eps=config.rms_norm_eps,
                has_weight=False,
            )
            if self.use_qk_norm
            else None
        )

        # HF checkpoints use NeoX RoPE; the config controls the convention.
        self.rotary_emb = (
            get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_theta,
                is_neox_style=config.rope_is_neox_style,
            )
            if self.use_rope
            else None
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=(
                get_attention_sliding_window_size(config) if self.is_sliding else -1
            ),
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.qk_norm is not None:
            q, k = apply_qk_norm(
                q,
                k,
                q_norm=self.qk_norm,
                k_norm=self.qk_norm,
                head_dim=self.head_dim,
            )

        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)

        attn_out = self.attn(q, k, v, forward_batch)

        if self.use_output_gate:
            gate, _ = self.output_gate_proj(hidden_states)
            if _is_cuda:
                attn_out = fused_sigmoid_mul(attn_out, gate, inplace=True)
            else:
                attn_out = torch.sigmoid(gate) * attn_out

        out, _ = self.o_proj(attn_out)
        return out


class MuseGlimmerDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MuseGlimmerAttention(
            config,
            layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.post_attn_norm = GemmaRMSNorm(config.hidden_size, eps=config.post_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = MuseGlimmerMLP(
            config, quant_config=quant_config, prefix=add_prefix("mlp", prefix)
        )
        self.post_ffn_norm = GemmaRMSNorm(config.hidden_size, eps=config.post_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states = residual + self.post_attn_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.post_ffn_norm(hidden_states)
        return hidden_states


class MuseGlimmerModel(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.embed_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps, has_weight=False)
            if config.normalize_tok_embeddings
            else None
        )
        self.layers = nn.ModuleList(
            [
                MuseGlimmerDecoderLayer(
                    config,
                    i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        # Reference MuseGlimmerFinalRMSNorm: weight is the scale, not an offset.
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layers_to_capture: List[int] = []

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        if self.embed_norm is not None:
            hidden_states = self.embed_norm(hidden_states)

        aux_hidden_states = []
        for i, layer in enumerate(self.layers):
            hidden_states = layer(positions, hidden_states, forward_batch)
            if i in self.layers_to_capture:
                aux_hidden_states.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class MuseGlimmerForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    checkpoint_uses_vendor_names = False

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        if quant_config is not None:
            raise ValueError(
                "Muse Glimmer currently supports unquantized HF weights only."
            )
        self.config = config
        self.quant_config = quant_config

        if config.output_soft_cap_temp is not None:
            config.final_logit_softcapping = config.output_soft_cap_temp

        self.model = MuseGlimmerModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            padding_size=128,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(
            config, logit_scale=config.output_multiplier
        )
        self.capture_aux_hidden_states = False

    def get_attention_sliding_window_size(self):
        return get_attention_sliding_window_size(self.config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def set_dflash_layers_to_capture(self, layer_ids: List[int]):
        if layer_ids is None:
            raise ValueError(
                "DFLASH requires explicit layer_ids for aux hidden capture."
            )
        num_layers = len(self.model.layers)
        bad = [i for i in layer_ids if not 0 <= i < num_layers]
        if bad:
            raise ValueError(
                f"DFLASH target layer ids {bad} are out of range for a "
                f"{num_layers}-layer Muse Glimmer target."
            )
        self.capture_aux_hidden_states = True
        self.model.layers_to_capture = list(layer_ids)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded = set()
        expected = {
            (name, shard)
            for name in params_dict
            for shard in (
                ("q", "k", "v")
                if ".qkv_proj." in name
                else (0, 1)
                if ".gate_up_proj." in name
                else (None,)
            )
        }

        for name, loaded_weight in weights:
            # SGLang derives RoPE itself; the checkpoint ships a cached freqs buffer.
            if "rotary_emb.freqs" in name or "rotary_emb.inv_freq" in name:
                continue
            if any(fragment in name for fragment in _VISION_NAME_FRAGMENTS):
                continue

            if self.checkpoint_uses_vendor_names:
                name = _vendor_weight_name(name)

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "output_gate_proj" in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded.add((mapped, shard_id))
                break
            else:
                if name not in params_dict:
                    logger.warning(
                        "Muse Glimmer: unexpected checkpoint weight %s", name
                    )
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add((name, None))

        missing = expected - loaded
        if missing:
            raise ValueError(
                f"Muse Glimmer checkpoint is missing weights: {sorted(missing)}"
            )
        logger.info("Muse Glimmer: loaded %d weight tensors", len(loaded))


class MuseGlimmerForConditionalGeneration(MuseGlimmerForCausalLM):
    """Text-only execution of Meta's native multimodal HF checkpoint."""

    checkpoint_uses_vendor_names = True


EntryClass = [MuseGlimmerForCausalLM, MuseGlimmerForConditionalGeneration]

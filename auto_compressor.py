import logging
from typing import Optional, Union, List, Tuple, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import OPTForCausalLM
from modeling_flash_llama import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)

PastKVType = Optional[Tuple[Tuple[torch.FloatTensor]]]


@dataclass
class SummaryConfig:
    """Keep track of token constitution of current input sequence"""

    softprompt_length: int = 0
    past_key_values_softprompt_length: int = 0
    control_length: int = 0

    def reset(self):
        self.softprompt_length = 0
        self.past_key_values_softprompt_length = 0
        self.control_length = 0


@dataclass
class CausalACOutputWithPast(CausalLMOutputWithPast):
    softprompt: Optional[torch.FloatTensor] = None


class AutoCompressorMixin:
    """Mixin class to turn a AutoModelForCausalLM into an AutoCompressor."""

    def setup_autocompressor(self, config):
        # Defaults for variable-length compression
        if not hasattr(config, "compression_max_len"):
            config.compression_max_len = 0
        if not hasattr(config, "compression_lambda"):
            config.compression_lambda = 0.0
        if not hasattr(config, "recompress_memory"):
            config.recompress_memory = True
        if not hasattr(config, "truncate_bptt_segments"):
            config.truncate_bptt_segments = 1
        if not hasattr(config, "sum_token_id"):
            config.sum_token_id = None
        if not hasattr(config, "eoc_token_id"):
            config.eoc_token_id = None

        if config.sum_token_id is None or config.eoc_token_id is None:
            logger.warning(
                "AutoCompressor config is missing sum/eoc token ids; ensure tokenizer is updated before use."
            )

        self.summary_config = SummaryConfig()

    # ------------------------------------------------
    # Helpers
    # ------------------------------------------------
    def get_sum_embedding(self, device, dtype):
        if self.config.sum_token_id is None:
            raise ValueError("sum_token_id must be set on config")
        token = torch.tensor([self.config.sum_token_id], device=device)
        return self.get_input_embeddings()(token).to(dtype)

    def get_past_key_values_len(self, past_key_values):
        if past_key_values is None:
            return 0
        try:
            v = past_key_values[0][1]
            if v is None:
                pkv = past_key_values[0][0]
                return int(pkv.size(2)) if torch.is_tensor(pkv) else 0
            return int(v)
        except Exception:
            return 0

    def _decoder(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.LongTensor,
        past_key_values: PastKVType,
        output_hidden_states: bool,
        use_cache: bool,
        output_attentions: bool,
        softprompt_length: int,
        past_key_values_softprompt_length: int,
        control_length: int = 0,
        segment_gradient_checkpointing: bool = False,
    ):
        def _run(segment_embeds, segment_attention_mask, segment_past_key_values, soft_len, past_soft_len, ctrl_len):
            self.summary_config.softprompt_length = soft_len
            self.summary_config.past_key_values_softprompt_length = past_soft_len
            self.summary_config.control_length = ctrl_len
            return self.model(
                inputs_embeds=segment_embeds,
                attention_mask=segment_attention_mask,
                past_key_values=segment_past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        if segment_gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                _run,
                inputs_embeds,
                attention_mask,
                past_key_values,
                softprompt_length,
                past_key_values_softprompt_length,
                control_length,
                use_reentrant=False,
            )
        return _run(
            inputs_embeds,
            attention_mask,
            past_key_values,
            softprompt_length,
            past_key_values_softprompt_length,
            control_length,
        )

    def _compress_context(
        self,
        prefix_outputs,
        attention_mask: torch.LongTensor,
        past_key_values_softprompt_length: int,
        original_length: int,
        training: bool = True,
    ):
        """Run variable-length compression loop starting after <sum>.

        Returns
        -------
        new_softprompt: torch.FloatTensor
            Memory vectors to prepend to the next segment.
        stats: Dict
            Contains probabilities and expected length (training only).
        """

        if self.config.compression_max_len <= 0:
            return prefix_outputs.last_hidden_state[:, :0, :], {}
        if self.config.eoc_token_id is None:
            raise ValueError("eoc_token_id must be set on config")

        past_key_values = prefix_outputs.past_key_values
        z_prev = prefix_outputs.last_hidden_state[:, -1:, :]
        attn_mask = torch.cat(
            [attention_mask, torch.ones(attention_mask.size(0), 1, device=attention_mask.device, dtype=attention_mask.dtype)],
            dim=1,
        )

        prob_list = []
        survival_list = []
        memory_vectors = []
        expected_length = None

        survival = torch.ones(z_prev.size(0), 1, 1, device=z_prev.device, dtype=z_prev.dtype)

        for step in range(self.config.compression_max_len):
            self.summary_config.softprompt_length = 0
            self.summary_config.past_key_values_softprompt_length = past_key_values_softprompt_length
            self.summary_config.control_length = 0

            step_outputs = self.model(
                inputs_embeds=z_prev,
                attention_mask=attn_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

            z_t = step_outputs.last_hidden_state[:, -1:, :]
            logits = self.lm_head(z_t)
            p_t = logits.softmax(dim=-1)[..., self.config.eoc_token_id]
            prob_list.append(p_t)
            if training:
                survival_list.append(survival)
                memory_vectors.append(survival * z_t)
                survival = survival * (1 - p_t.unsqueeze(-1))
            else:
                token_id = logits.argmax(dim=-1)
                stop_mask = token_id.eq(self.config.eoc_token_id)
                if not torch.any(stop_mask):
                    memory_vectors.append(z_t)
                else:
                    keep_mask = ~stop_mask
                    if torch.any(keep_mask):
                        memory_vectors.append(z_t * keep_mask.unsqueeze(-1).type_as(z_t))
                    break

            z_prev = z_t
            past_key_values = step_outputs.past_key_values
            attn_mask = torch.cat(
                [attn_mask, torch.ones(attn_mask.size(0), 1, device=attn_mask.device, dtype=attn_mask.dtype)], dim=1
            )

        if training:
            if len(memory_vectors) > 0:
                new_softprompt = torch.cat(memory_vectors, dim=1)
            else:
                new_softprompt = z_prev[:, :0, :]
            if len(survival_list) > 0:
                survival_tensor = torch.cat(survival_list, dim=1)
                expected_length = survival_tensor.sum(dim=1)  # [B,1]
            else:
                survival_tensor = None
                expected_length = None
        else:
            new_softprompt = torch.cat(memory_vectors, dim=1) if len(memory_vectors) > 0 else z_prev[:, :0, :]
            survival_tensor = None

        stats = {
            "p_eoc": torch.stack(prob_list, dim=1) if len(prob_list) > 0 else None,
            "survival": survival_tensor,
            "expected_length": expected_length,
            "original_length": original_length,
        }

        return new_softprompt, stats

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Union[PastKVType, Dict] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        segment_lengths: Optional[Union[List[int], int]] = None,
        softprompt: Optional[torch.FloatTensor] = None,
        output_softprompt: Optional[bool] = None,
        past_key_values_softprompt_length: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if past_key_values is not None and isinstance(past_key_values, dict):
            past_key_values, softprompt = past_key_values["past_key_values"], past_key_values.get("softprompt", softprompt)
            if softprompt is not None:
                past_key_values_softprompt_length = max(past_key_values_softprompt_length, softprompt.size(1))

        past_key_values_length = self.get_past_key_values_len(past_key_values) - past_key_values_softprompt_length

        if head_mask is not None:
            raise ValueError("Compressor does not support head_mask")
        if inputs_embeds is not None and input_ids is not None:
            raise ValueError("Compressor does not support both input_ids and input_embeds")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if past_key_values is None:
            segment_lengths = segment_lengths if segment_lengths is not None else input_ids.size(1)
            if attention_mask is None:
                attention_mask = torch.ones(
                    inputs_embeds.size(0), inputs_embeds.size(1), dtype=torch.long, device=inputs_embeds.device
                )

            inputs_embeds_list = torch.split(inputs_embeds, segment_lengths, dim=1)
            attention_mask_list = torch.split(attention_mask, segment_lengths, dim=1)
        else:
            if attention_mask is None:
                attention_mask = torch.ones(
                    inputs_embeds.size(0), inputs_embeds.size(1) + past_key_values_length, dtype=torch.long, device=inputs_embeds.device
                )
            inputs_embeds_list = (inputs_embeds,)
            attention_mask_list = (attention_mask,)

        text_hidden_state_list = []
        output_attentions_list = []
        output_hidden_states_list = []
        compression_penalties = []

        if softprompt is None:
            softprompt = inputs_embeds[:, :0, :]

        segments_since_detach = 0

        for step, segment_embeds in enumerate(inputs_embeds_list):
            is_last_step = step == len(inputs_embeds_list) - 1
            compress_this_segment = self.config.compression_max_len > 0 and (not is_last_step or output_softprompt)

            segment_gradient_checkpointing = (
                getattr(self.config, "segment_gradient_checkpointing", False)
                and self.training
                and compress_this_segment
            )

            sum_embed = self.get_sum_embedding(segment_embeds.device, segment_embeds.dtype)
            sum_embed = sum_embed.unsqueeze(0).expand(segment_embeds.size(0), -1, -1)

            if past_key_values_softprompt_length > 0:
                softprompt_length = 0
                combined_embeds = torch.cat([segment_embeds, sum_embed], dim=1)
                attn_mask = torch.cat(
                    [
                        torch.ones(
                            segment_embeds.size(0), past_key_values_softprompt_length, device=segment_embeds.device, dtype=attention_mask_list[step].dtype
                        ),
                        attention_mask_list[step],
                        torch.ones(segment_embeds.size(0), 1, device=segment_embeds.device, dtype=attention_mask_list[step].dtype),
                    ],
                    dim=1,
                )
            else:
                softprompt_length = softprompt.size(1)
                combined_embeds = torch.cat([softprompt, segment_embeds, sum_embed], dim=1)
                attn_mask = torch.cat(
                    [
                        torch.ones(segment_embeds.size(0), softprompt_length, device=segment_embeds.device, dtype=attention_mask_list[step].dtype),
                        attention_mask_list[step],
                        torch.ones(segment_embeds.size(0), 1, device=segment_embeds.device, dtype=attention_mask_list[step].dtype),
                    ],
                    dim=1,
                )

            decoder_outputs = self._decoder(
                combined_embeds,
                attn_mask,
                past_key_values,
                output_hidden_states,
                use_cache if use_cache is not None else False,
                output_attentions,
                softprompt_length,
                past_key_values_softprompt_length,
                control_length=0,
                segment_gradient_checkpointing=segment_gradient_checkpointing,
            )

            total_length = decoder_outputs.last_hidden_state.size(1)
            segment_length = segment_embeds.size(1)
            segment_hiddens = decoder_outputs.last_hidden_state[:, softprompt_length : softprompt_length + segment_length]
            text_hidden_state_list.append(segment_hiddens)

            output_attentions_list.append(decoder_outputs.attentions)
            output_hidden_states_list.append(decoder_outputs.hidden_states)

            if compress_this_segment:
                original_length = softprompt_length + past_key_values_softprompt_length + segment_length
                new_softprompt, stats = self._compress_context(
                    decoder_outputs,
                    attn_mask,
                    past_key_values_softprompt_length + softprompt_length,
                    original_length=original_length,
                    training=self.training,
                )

                if self.training and stats.get("expected_length") is not None and self.config.compression_lambda > 0:
                    expected_len = stats["expected_length"].squeeze(-1)
                    length_penalty = self.config.compression_lambda * (expected_len / max(float(original_length), 1.0))
                    compression_penalties.append(length_penalty.mean())

                if self.config.recompress_memory:
                    softprompt = new_softprompt
                else:
                    softprompt = torch.cat([softprompt, new_softprompt], dim=1)

                segments_since_detach += 1
                if segments_since_detach >= max(1, int(getattr(self.config, "truncate_bptt_segments", 1))):
                    softprompt = softprompt.detach()
                    segments_since_detach = 0
            else:
                softprompt = softprompt

            past_key_values = None
            past_key_values_softprompt_length = 0

        self.summary_config.reset()

        text_hiddens = torch.cat(text_hidden_state_list, dim=1)
        logits = self.lm_head(text_hiddens).contiguous()

        loss = None
        if labels is not None:
            labels = labels[..., : logits.size(1)]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            if len(compression_penalties) > 0:
                loss = loss + sum(compression_penalties)
        elif len(compression_penalties) > 0:
            loss = sum(compression_penalties)

        output = CausalACOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=output_hidden_states_list if output_hidden_states_list[0] is not None else None,
            attentions=output_attentions_list if output_attentions_list[0] is not None else None,
            softprompt=softprompt if output_softprompt else None,
        )

        if return_dict:
            return output
        else:
            return tuple(output.values())

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values, attention_mask, inputs_embeds, **kwargs
        )
        model_inputs["softprompt"] = kwargs.get("softprompt", None)
        model_inputs["segment_lengths"] = kwargs.get("segment_lengths", None)
        model_inputs["past_key_values_softprompt_length"] = kwargs.get("past_key_values_softprompt_length", 0)
        return model_inputs

    def _update_model_kwargs_for_generation(
        self, outputs, model_kwargs: Dict[str, torch.Tensor], is_encoder_decoder: bool = False
    ) -> Dict[str, torch.Tensor]:
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder)

        softprompt = getattr(outputs, "softprompt", None)
        if softprompt is not None:
            model_kwargs["softprompt"] = softprompt
            model_kwargs["past_key_values_softprompt_length"] = softprompt.size(1)
        else:
            model_kwargs.pop("softprompt", None)
            model_kwargs.pop("past_key_values_softprompt_length", None)

        return model_kwargs


class OPTLearnedPositionalEmbeddingWithPadding(nn.Embedding):
    """Overwrite the default OPTLearnedPositionalEmbedding to disable position on memory tokens"""

    def __init__(self, num_embeddings: int, embedding_dim: int, summary_config: Optional[SummaryConfig] = None):
        super().__init__(num_embeddings + 2, embedding_dim, padding_idx=1)
        self.summary_config = summary_config if summary_config is not None else SummaryConfig()

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        attention_mask = attention_mask.long()
        bsz = attention_mask.size(0)

        total_softprompt_length = self.summary_config.softprompt_length + self.summary_config.past_key_values_softprompt_length
        left_placeholder = torch.ones(
            bsz, self.summary_config.softprompt_length, dtype=torch.long, device=attention_mask.device
        )
        effective_mask = attention_mask[:, total_softprompt_length :]
        positions = effective_mask.cumsum(dim=1) * effective_mask + 1

        start = max(past_key_values_length - self.summary_config.past_key_values_softprompt_length, 0)
        positions = positions[:, start:]
        positions = torch.cat([left_placeholder, positions], dim=1)

        return super().forward(positions)


class OPTAutoCompressorModel(AutoCompressorMixin, OPTForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.setup_autocompressor(config)

        self.model.decoder.embed_positions = OPTLearnedPositionalEmbeddingWithPadding(
            config.max_position_embeddings, config.hidden_size, summary_config=self.summary_config
        )

        self.post_init()


# For backwards compatibility
AutoCompressorModel = OPTAutoCompressorModel


class LlamaAutoCompressorModel(AutoCompressorMixin, LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.setup_autocompressor(config)

        self.post_init()

    def get_past_key_values_len(self, past_key_values):
        if past_key_values is None:
            return 0
        try:
            v = past_key_values[0][1]
            if v is None:
                pkv = past_key_values[0][0]
                return int(pkv.size(1)) if torch.is_tensor(pkv) else 0
            return int(v)
        except Exception:
            return 0

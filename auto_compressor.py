import logging
import os
from typing import Optional, Union, List, Tuple, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import OPTForCausalLM
from modeling_flash_llama import LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

import os

logger = logging.getLogger(__name__)

PastKVType = Optional[Tuple[Tuple[torch.FloatTensor]]]

@dataclass
class SummaryConfig:
    """Keep track of token constitution of current input sequence"""
    softprompt_length: int = 0
    past_key_values_softprompt_length: int = 0
    summary_length: int = 0

    def reset(self):
        self.softprompt_length = 0
        self.past_key_values_softprompt_length = 0
        self.summary_length = 0


@dataclass
class CausalACOutputWithPast(CausalLMOutputWithPast):
    softprompt: Optional[torch.FloatTensor]= None


class AutoCompressorMixin:
    """Mixin class to turn a AutoModelForCausalLM into an AutoCompressor."""

    def setup_autocompressor(self, config):
        """Initialize the autocompressor. Override for custom behaviour"""

        # Ensure new configuration attributes exist with sensible defaults
        defaults = {
            "compression_max_len": 0,
            "compression_lambda": 0.0,
            "recompress_memory": True,
            "gate_memory_mode": "embed",
            "truncate_bptt_segments": 1,
        }
        for key, value in defaults.items():
            if not hasattr(config, key):
                setattr(config, key, value)

        if not hasattr(config, "sum_token_id"):
            setattr(config, "sum_token_id", None)
        if not hasattr(config, "eoc_token_id"):
            setattr(config, "eoc_token_id", None)

        self.summary_config = SummaryConfig()

    def forward_segment(
        self,
        softprompt: torch.FloatTensor,
        segment_embeds: torch.FloatTensor,
        summary_token_embeds: torch.FloatTensor,
        segment_attention_mask: torch.LongTensor,
        past_key_values: PastKVType,
        output_hidden_states: bool,
        use_cache: bool,
        output_attentions: bool,
        segment_gradient_checkpointing: bool,
        past_key_values_softprompt_length: int
    ):

        bsz = segment_embeds.size(0)
        summary_length = summary_token_embeds.size(1)
        if past_key_values_softprompt_length > 0: # Softprompt should already be in past_key_values
            softprompt_length = 0
            segment_embeds = torch.cat([segment_embeds, summary_token_embeds], dim=1)

            device, attn_dtype = segment_embeds.device, segment_attention_mask.dtype
            segment_attention_mask = torch.cat([
                torch.ones(bsz, past_key_values_softprompt_length, device=device, dtype=attn_dtype),
                segment_attention_mask,
                torch.ones(bsz, summary_length, device=device, dtype=attn_dtype)
            ], dim=1)
        else:
            softprompt_length = softprompt.size(1)
            segment_embeds = torch.cat([softprompt, segment_embeds, summary_token_embeds], dim=1)

            device, attn_dtype = segment_embeds.device, segment_attention_mask.dtype
            segment_attention_mask = torch.cat([
                torch.ones(bsz, softprompt_length, device=device, dtype=attn_dtype),
                segment_attention_mask,
                torch.ones(bsz, summary_length, device=device, dtype=attn_dtype)
            ], dim=1)
        
        def decoder(segment_embeds,
                    segment_attention_mask,
                    segment_past_key_values,
                    softprompt_length,
                    past_key_values_softprompt_length,
                    summary_length):
            self.summary_config.softprompt_length = softprompt_length
            self.summary_config.past_key_values_softprompt_length = past_key_values_softprompt_length
            self.summary_config.summary_length = summary_length

            return self.model(
                inputs_embeds=segment_embeds,
                attention_mask=segment_attention_mask,
                past_key_values=segment_past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,)

        if segment_gradient_checkpointing:
            outputs = torch.utils.checkpoint.checkpoint(
                decoder, segment_embeds, segment_attention_mask, past_key_values,
                softprompt_length, past_key_values_softprompt_length, summary_length,
                use_reentrant=False)
        else:
            outputs = decoder(
                segment_embeds, segment_attention_mask, past_key_values,
                softprompt_length, past_key_values_softprompt_length, summary_length)

        total_length = outputs.last_hidden_state.size(1)
        segment_last_hiddens = (
            outputs.last_hidden_state[:, softprompt_length:total_length - summary_length]
        )
        new_softprompt = outputs.last_hidden_state[:, total_length - summary_length:]

        return outputs, segment_last_hiddens, new_softprompt

    def get_past_key_values_len(self, past_key_values):
        if past_key_values is None:
            return 0
        try:
            v = past_key_values[0][1]
            if v is None:
                # 텐서 기반 복원 시도
                pkv = past_key_values[0][0]
                return int(pkv.size(2)) if torch.is_tensor(pkv) else 0
            return int(v)
        except Exception:
            return 0

    def _call_decoder_with_config(
        self,
        inputs_embeds,
        attention_mask,
        past_key_values,
        softprompt_length,
        past_key_values_softprompt_length,
        summary_length,
        use_cache,
        output_attentions,
        output_hidden_states,
    ):
        """Wrapper to ensure SummaryConfig is set before calling the underlying model."""
        self.summary_config.softprompt_length = softprompt_length
        self.summary_config.past_key_values_softprompt_length = past_key_values_softprompt_length
        self.summary_config.summary_length = summary_length

        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

    def compress_context(
        self,
        softprompt: torch.FloatTensor,
        segment_embeds: torch.FloatTensor,
        segment_attention_mask: torch.LongTensor,
        past_key_values: PastKVType,
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
        past_key_values_softprompt_length: int,
        training: bool = True,
    ):
        """Run the variable-length compression loop.

        Returns a tuple of (new_softprompt, stats_dict).
        """

        device, dtype = segment_embeds.device, segment_embeds.dtype
        bsz = segment_embeds.size(0)
        softprompt_length = softprompt.size(1)

        sum_token_id = getattr(self.config, "sum_token_id", None)
        eoc_token_id = getattr(self.config, "eoc_token_id", None)
        if sum_token_id is None or eoc_token_id is None:
            raise ValueError("sum_token_id and eoc_token_id must be set on the config for compression mode.")

        sum_embed = self.get_input_embeddings()(torch.full((bsz, 1), sum_token_id, device=device))

        # Build prefix input with <sum> trigger
        prefix_embeds = torch.cat([softprompt, segment_embeds, sum_embed.to(dtype)], dim=1)
        attn_dtype = segment_attention_mask.dtype
        prefix_attention_mask = torch.cat(
            [
                torch.ones(bsz, softprompt_length, device=device, dtype=attn_dtype),
                segment_attention_mask,
                torch.ones(bsz, 1, device=device, dtype=attn_dtype),
            ],
            dim=1,
        )

        # Process prefix to seed compression
        prefix_outputs = self._call_decoder_with_config(
            prefix_embeds,
            prefix_attention_mask,
            past_key_values,
            softprompt_length=softprompt_length,
            past_key_values_softprompt_length=past_key_values_softprompt_length,
            summary_length=0,
            use_cache=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        z_prev = prefix_outputs.last_hidden_state[:, -1:, :]
        past_kv = prefix_outputs.past_key_values
        compression_attention_mask = prefix_attention_mask

        z_list = []
        survival = torch.ones(bsz, 1, device=device, dtype=dtype)
        survival_list = []
        p_eoc_list = []

        max_len = getattr(self.config, "compression_max_len", 0)
        if max_len <= 0:
            return softprompt, {
                "expected_length": torch.zeros([], device=device, dtype=dtype),
                "length_penalty": torch.zeros([], device=device, dtype=dtype),
                "p_eoc_list": [],
            }

        for _ in range(max_len):
            compression_attention_mask = torch.cat(
                [compression_attention_mask, torch.ones(bsz, 1, device=device, dtype=attn_dtype)], dim=1
            )

            outputs = self._call_decoder_with_config(
                z_prev.to(dtype),
                compression_attention_mask,
                past_kv,
                softprompt_length=0,
                past_key_values_softprompt_length=softprompt_length,
                summary_length=0,
                use_cache=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            z_t = outputs.last_hidden_state[:, -1:, :]
            logits = self.lm_head(z_t)
            probs = torch.softmax(logits, dim=-1)
            p_eoc = probs[..., eoc_token_id:eoc_token_id + 1]
            p_eoc_list.append(p_eoc)
            survival_list.append(survival)

            if training:
                z_list.append(survival * z_t)
                survival = survival * (1.0 - p_eoc)
            else:
                # Greedy termination for inference
                next_token = torch.argmax(logits, dim=-1)
                cont_mask = (next_token != eoc_token_id).float().unsqueeze(-1)
                if cont_mask.sum() == 0:
                    break
                z_list.append(z_t * cont_mask)
                survival = cont_mask

            z_prev = z_t
            past_kv = outputs.past_key_values

            if not training and survival.sum() == 0:
                break

        if not z_list:
            new_softprompt = softprompt[:, :0, :]
            expected_length = torch.zeros([], device=device, dtype=dtype)
        else:
            new_softprompt = torch.cat(z_list, dim=1)
            if training:
                survival_stack = torch.stack(survival_list, dim=1)
                expected_length = survival_stack.sum(dim=1)
            else:
                expected_length = torch.tensor(new_softprompt.size(1), device=device, dtype=dtype)

        original_length = softprompt_length + segment_attention_mask.sum(dim=1, keepdim=True)
        length_penalty = 0.0
        if training and getattr(self.config, "compression_lambda", 0.0) > 0:
            length_penalty = (
                getattr(self.config, "compression_lambda", 0.0)
                * (expected_length / original_length.clamp(min=1)).mean()
            )

        return new_softprompt, {
            "expected_length": expected_length,
            "length_penalty": length_penalty if torch.is_tensor(length_penalty) else torch.tensor(length_penalty, device=device, dtype=dtype),
            "p_eoc_list": p_eoc_list,
            "prefix_outputs": prefix_outputs,
        }

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
        # Support legacy dict-shaped past_key_values while preferring tuples for newer transformers versions
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

        summary_token_embeds = inputs_embeds[:, :0, :]

        if softprompt is None:
            softprompt = inputs_embeds[:, :0, :]

        # If no past_key_values are given, we will process the sequence in multiple segments
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

        last_hidden_state_list = []
        output_attentions_list = []
        output_hidden_states_list = []
        compression_penalty = 0.0
        segments_since_detach = 0

        for step, segment_embeds in enumerate(inputs_embeds_list):
            segment_attention_mask = attention_mask_list[step]
            is_last_step = step == len(inputs_embeds_list) - 1
            segment_gradient_checkpointing = (
                getattr(self.config, "segment_gradient_checkpointing", False) and
                self.training and not is_last_step
            )

            outputs, segment_hidden_states, _ = self.forward_segment(
                softprompt.to(inputs_embeds.dtype), segment_embeds, summary_token_embeds, segment_attention_mask,
                past_key_values, output_hidden_states, use_cache, output_attentions,
                segment_gradient_checkpointing, past_key_values_softprompt_length)

            last_hidden_state_list.append(segment_hidden_states)
            output_attentions_list.append(outputs.attentions)
            output_hidden_states_list.append(outputs.hidden_states)

            should_compress = past_key_values is None and (not is_last_step or output_softprompt)
            if should_compress:
                new_softprompt, stats = self.compress_context(
                    softprompt.to(inputs_embeds.dtype), segment_embeds, segment_attention_mask,
                    outputs.past_key_values, use_cache, output_attentions, output_hidden_states,
                    past_key_values_softprompt_length, training=self.training,
                )

                if getattr(self.config, "recompress_memory", True):
                    softprompt = new_softprompt
                else:
                    softprompt = torch.cat([softprompt, new_softprompt], dim=1)

                if self.training:
                    compression_penalty = compression_penalty + stats.get("length_penalty", 0.0)

                segments_since_detach += 1
                if self.config.truncate_bptt_segments and segments_since_detach >= self.config.truncate_bptt_segments:
                    softprompt = softprompt.detach()
                    segments_since_detach = 0

            past_key_values = None
            past_key_values_softprompt_length = 0

        past_key_values = outputs.past_key_values

        self.summary_config.reset()

        last_hiddens = torch.cat(last_hidden_state_list, dim=1)
        logits = self.lm_head(last_hiddens).contiguous()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            if self.training:
                loss = loss + compression_penalty

        output = CausalACOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=output_hidden_states_list if output_hidden_states_list and output_hidden_states_list[0] is not None else None,
            attentions=output_attentions_list if output_attentions_list and output_attentions_list[0] is not None else None,
            softprompt=softprompt if output_softprompt else softprompt[:,:0,:],
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
    """Overwrite the default OPTLearnedPositionalEmbedding to disable position on summary tokens"""

    def __init__(self, num_embeddings: int, embedding_dim: int, summary_config: Optional[SummaryConfig] = None):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        super().__init__(num_embeddings + 2, embedding_dim, padding_idx=1)

        self.summary_config = summary_config if summary_config is not None else SummaryConfig()

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        attention_mask = attention_mask.long()
        bsz = attention_mask.size(0)

        left_placeholder = torch.ones(bsz, self.summary_config.softprompt_length, dtype=torch.long, device=attention_mask.device) # <pad> -> zero vector
        right_placeholder = torch.ones(bsz, self.summary_config.summary_length, dtype=torch.long, device=attention_mask.device) # <pad> -> zero vector

        total_softprompt_length = self.summary_config.softprompt_length + self.summary_config.past_key_values_softprompt_length
        attention_mask = attention_mask[:, total_softprompt_length : attention_mask.size(1)-self.summary_config.summary_length]

        positions = attention_mask.cumsum(dim=1) * attention_mask + 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length - self.summary_config.past_key_values_softprompt_length :]
        positions = torch.cat([left_placeholder, positions, right_placeholder], dim=1)

        return super().forward(positions)


class OPTAutoCompressorModel(AutoCompressorMixin, OPTForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.setup_autocompressor(config)

        # Custom positional embeddings
        self.model.decoder.embed_positions = OPTLearnedPositionalEmbeddingWithPadding(
            config.max_position_embeddings, config.hidden_size, summary_config=self.summary_config
        )

        # Initialize weights and apply final processing
        self.post_init()


# For backwards compatibility
AutoCompressorModel = OPTAutoCompressorModel


class LlamaAutoCompressorModel(AutoCompressorMixin, LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.setup_autocompressor(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_past_key_values_len(self, past_key_values):
        # modeling_flash_llama has slightly different layout of past key vlaues
        if past_key_values is None:
            return 0
        try:
            v = past_key_values[0][1]
            if v is None:
                # 텐서 기반 복원 시도
                pkv = past_key_values[0][0]
                return int(pkv.size(1)) if torch.is_tensor(pkv) else 0
            return int(v)
        except Exception:
            return 0

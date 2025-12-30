import logging
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
    compression_stats: Optional[Dict[str, torch.FloatTensor]] = None


class AutoCompressorMixin:
    """Mixin class to turn a AutoModelForCausalLM into an AutoCompressor."""

    def setup_autocompressor(self, config):
        """Call this function in the subclass __init__ to initialize the autocompressor. Override for custom behaviour"""
        # Backwards compatibility with fixed-length AutoCompressors
        default_config = {
            "compression_max_len": getattr(config, "compression_max_len", getattr(config, "summary_length", 0)),
            "compression_lambda": getattr(config, "compression_lambda", 0.0),
            "recompress_memory": getattr(config, "recompress_memory", True),
            "gate_memory_mode": getattr(config, "gate_memory_mode", "embed"),
            "truncate_bptt_segments": getattr(config, "truncate_bptt_segments", 1),
        }
        for k, v in default_config.items():
            setattr(config, k, v)

        # Explicitly disable old summary-token path
        config.summary_length = 0

        if not hasattr(config, "sum_token_id"):
            config.sum_token_id = None
        if not hasattr(config, "eoc_token_id"):
            config.eoc_token_id = None

        self.summary_config = SummaryConfig()

    def _apply_position_placeholders(self, softprompt_length: int, summary_length: int = 0, past_softprompt_length: int = 0):
        self.summary_config.softprompt_length = softprompt_length
        self.summary_config.past_key_values_softprompt_length = past_softprompt_length
        self.summary_config.summary_length = summary_length

    def _concat_attention(self, base_mask: torch.Tensor, prefix: int, suffix: int):
        bsz = base_mask.size(0)
        device, dtype = base_mask.device, base_mask.dtype
        return torch.cat([
            torch.ones(bsz, prefix, device=device, dtype=dtype),
            base_mask,
            torch.ones(bsz, suffix, device=device, dtype=dtype)
        ], dim=1)

    def compress_context(
        self,
        softprompt: torch.FloatTensor,
        segment_embeds: torch.FloatTensor,
        sum_token_embed: torch.FloatTensor,
        segment_attention_mask: torch.LongTensor,
        output_attentions: bool,
        output_hidden_states: bool,
        segment_gradient_checkpointing: bool,
        training: bool,
    ):
        """Compress the concatenated memory + segment after the <sum> trigger.

        Returns (lm_outputs, segment_hidden_states, new_memory, stats)
        """
        if self.config.eoc_token_id is None:
            raise ValueError("compression requires a configured eoc_token_id")
        bsz = segment_embeds.size(0)
        mem_len = softprompt.size(1)

        base_embeds = torch.cat([softprompt, segment_embeds, sum_token_embed], dim=1)
        attn_mask = self._concat_attention(segment_attention_mask, mem_len, 1)

        def decoder(inputs_embeds, attention_mask, past_key_values, softprompt_length, past_softprompt_length, summary_length=0):
            self._apply_position_placeholders(softprompt_length, summary_length, past_softprompt_length)
            return self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

        if segment_gradient_checkpointing:
            outputs = torch.utils.checkpoint.checkpoint(
                decoder,
                base_embeds,
                attn_mask,
                None,
                mem_len,
                0,
                0,
                use_reentrant=False,
            )
        else:
            outputs = decoder(base_embeds, attn_mask, None, mem_len, 0, 0)

        hidden_states = outputs.last_hidden_state
        segment_hidden_states = hidden_states[:, mem_len:, :]
        sum_hidden = hidden_states[:, -1:, :]

        total_length = attn_mask.size(1)
        running_attention_mask = attn_mask
        past_key_values = outputs.past_key_values

        # Manual compression loop
        z_list = []
        p_list = []
        s_list = []

        survival = torch.ones(bsz, 1, device=segment_embeds.device, dtype=hidden_states.dtype)
        z_prev = sum_hidden
        for step in range(self.config.compression_max_len):
            running_attention_mask = torch.cat([
                running_attention_mask,
                torch.ones(bsz, 1, device=running_attention_mask.device, dtype=running_attention_mask.dtype)
            ], dim=1)

            self._apply_position_placeholders(0, 0, mem_len)
            step_out = self.model(
                inputs_embeds=z_prev,
                attention_mask=running_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

            z_t = step_out.last_hidden_state[:, -1:, :]
            logits_t = self.lm_head(z_t)
            past_key_values = step_out.past_key_values

            p_t = torch.softmax(logits_t, dim=-1)[..., self.config.eoc_token_id]
            p_list.append(p_t)
            s_list.append(survival)

            z_list.append(z_t)
            z_prev = z_t

            if not training:
                next_token = torch.argmax(logits_t, dim=-1)
                if (next_token == self.config.eoc_token_id).all():
                    # Do not include the terminating state in the memory
                    z_list = z_list[:-1]
                    s_list = s_list[:-1]
                    p_list = p_list[:-1]
                    break

            survival = survival * (1 - p_t)

        if len(z_list) == 0:
            new_memory = softprompt[:, :0, :]
        else:
            z_cat = torch.cat(z_list, dim=1)
            s_cat = torch.cat(s_list, dim=1)
            gated_memory = z_cat * s_cat if training else z_cat
            new_memory = gated_memory

        stats = {}
        if training and len(p_list) > 0:
            p_cat = torch.cat(p_list, dim=1)
            s_cat = torch.cat(s_list, dim=1)
            expected_length = torch.sum(s_cat, dim=1, keepdim=True)
            stats = {
                "p_eoc": p_cat,
                "survival": s_cat,
                "expected_length": expected_length,
            }

        return outputs, segment_hidden_states, new_memory, stats

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
        if head_mask is not None:
            raise ValueError("Compressor does not support head_mask")
        if inputs_embeds is not None and input_ids is not None:
            raise ValueError("Compressor does not support both input_ids and input_embeds")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if softprompt is None:
            softprompt = inputs_embeds[:, :0, :]

        if past_key_values is not None:
            raise ValueError("Variable-length compression currently does not support passing past_key_values directly.")

        if segment_lengths is None:
            segment_lengths = [input_ids.size(1)]
        elif isinstance(segment_lengths, int):
            segment_lengths = [segment_lengths]

        # Build segment tuples (segment_tokens, sum_token)
        ptr = 0
        segments = []
        for seg_len in segment_lengths:
            seg_tokens = input_ids[:, ptr:ptr + seg_len]
            seg_mask = attention_mask[:, ptr:ptr + seg_len]
            ptr += seg_len
            sum_tok = None
            sum_mask = None
            if ptr < input_ids.size(1):
                sum_tok = input_ids[:, ptr:ptr + 1]
                sum_mask = attention_mask[:, ptr:ptr + 1]
                ptr += 1
            segments.append((seg_tokens, seg_mask, sum_tok, sum_mask))

        if ptr != input_ids.size(1):
            raise ValueError("segment_lengths and <sum> placements do not cover the full input")

        logits_list = []
        attentions_list = []
        hidden_states_list = []
        compression_stats = {}
        final_had_sum_token = False

        loss = None
        total_loss = 0.0
        for seg_idx, (seg_tokens, seg_mask, sum_tok, sum_mask) in enumerate(segments):
            segment_embeds = self.get_input_embeddings()(seg_tokens)
            sum_embeds = self.get_input_embeddings()(sum_tok) if sum_tok is not None else None

            if sum_tok is not None and self.config.sum_token_id is not None:
                if not torch.all(sum_tok == self.config.sum_token_id):
                    raise ValueError("Expected <sum> token at segment boundary")

            segment_gradient_checkpointing = (
                getattr(self.config, "segment_gradient_checkpointing", False) and
                self.training
            )

            outputs, segment_hidden_states, new_memory, stats = self.compress_context(
                softprompt.to(segment_embeds.dtype),
                segment_embeds,
                sum_embeds if sum_embeds is not None else segment_embeds[:, :0, :],
                seg_mask,
                output_attentions,
                output_hidden_states,
                segment_gradient_checkpointing,
                self.training,
            ) if sum_tok is not None else (self.model(
                inputs_embeds=torch.cat([softprompt, segment_embeds], dim=1),
                attention_mask=self._concat_attention(seg_mask, softprompt.size(1), 0),
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            ), None, softprompt, {})

            segment_logits = self.lm_head(outputs.last_hidden_state[:, softprompt.size(1):, :])
            logits_list.append(segment_logits)
            attentions_list.append(outputs.attentions)
            hidden_states_list.append(outputs.hidden_states)

            if stats:
                compression_stats[f"segment_{seg_idx}"] = stats
                if self.training:
                    original_length = float(softprompt.size(1) + seg_tokens.size(1) + 1)
                    expected_length = stats["expected_length"]
                    total_loss = total_loss + self.config.compression_lambda * (expected_length.mean() / original_length)

            if sum_tok is not None:
                final_had_sum_token = True
            softprompt = new_memory if sum_tok is not None and self.config.recompress_memory else torch.cat([softprompt, new_memory], dim=1)
            if self.config.truncate_bptt_segments <= 1:
                softprompt = softprompt.detach()

        logits = torch.cat(logits_list, dim=1)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            total_loss = total_loss + ce_loss
            loss = total_loss

        output = CausalACOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=hidden_states_list if hidden_states_list and hidden_states_list[0] is not None else None,
            attentions=attentions_list if attentions_list and attentions_list[0] is not None else None,
            softprompt=softprompt if output_softprompt or final_had_sum_token else softprompt[:, :0, :],
            compression_stats=compression_stats if compression_stats else None,
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

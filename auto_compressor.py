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
    """Keep track of token constitution of current input sequence."""

    # Number of softprompt (memory) vectors prepended directly to the input
    softprompt_length: int = 0
    # Number of softprompt (memory) vectors already stored inside past_key_values
    past_key_values_softprompt_length: int = 0
    # Number of compression-step vectors that should receive zero positional embeddings
    compression_length: int = 0

    def reset(self):
        self.softprompt_length = 0
        self.past_key_values_softprompt_length = 0
        self.compression_length = 0


@dataclass
class CausalACOutputWithPast(CausalLMOutputWithPast):
    softprompt: Optional[torch.FloatTensor] = None


class AutoCompressorMixin:
    """Mixin class to turn a AutoModelForCausalLM into an AutoCompressor."""

    def setup_autocompressor(self, config):
        """Call this function in the subclass __init__ to initialize the autocompressor."""
        defaults = dict(
            compression_max_len=8,
            compression_lambda=0.0,
            compression_alpha=1.0,
            sum_token_id=None,
            eoc_token_id=None,
            recompress_memory=True,
            truncate_bptt_segments=1,
            compress_stop_threshold=None,
        )

        for k, v in defaults.items():
            if not hasattr(self.config, k):
                setattr(self.config, k, v)

        self.summary_config = SummaryConfig()

    def _mask_control_tokens_in_labels(self, input_ids, labels):
        if labels is None:
            return None
        masked_labels = labels.clone()
        if self.config.sum_token_id is not None:
            masked_labels = masked_labels.masked_fill(input_ids == self.config.sum_token_id, -100)
        if self.config.eoc_token_id is not None:
            masked_labels = masked_labels.masked_fill(input_ids == self.config.eoc_token_id, -100)
        return masked_labels

    def _mask_logits_for_control_tokens(self, logits):
        if logits is None:
            return logits
        for tok_id in (self.config.sum_token_id, self.config.eoc_token_id):
            if tok_id is not None:
                logits[..., tok_id] = -float("inf")
        return logits

    def compress_context(
        self,
        prefix_hidden: torch.FloatTensor,
        past_key_values: PastKVType,
        memory_length: int,
        segment_text_length: torch.LongTensor,
        training: bool,
    ):
        """Run the variable-length compression loop."""

        device = prefix_hidden.device
        hidden_size = prefix_hidden.size(-1)
        batch_size = prefix_hidden.size(0)
        z_prev = prefix_hidden
        past = past_key_values
        logits_hist = []
        p_hist = []
        survival_hist = []
        gated_memory = []

        survival = torch.ones(batch_size, device=device, dtype=prefix_hidden.dtype)

        def step_attention_mask(past_len: int):
            return torch.ones(batch_size, past_len + 1, device=device, dtype=torch.long)

        past_total_len = self.get_past_key_values_len(past)

        for _ in range(self.config.compression_max_len):
            attn_mask = step_attention_mask(past_total_len)
            self.summary_config.softprompt_length = 0
            self.summary_config.past_key_values_softprompt_length = memory_length
            self.summary_config.compression_length = 1

            outputs = self.model(
                inputs_embeds=z_prev.unsqueeze(1),
                attention_mask=attn_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            z_t = outputs.last_hidden_state[:, -1, :]
            logits = self.lm_head(z_t)
            probs = logits.softmax(dim=-1)
            p_t = probs[:, self.config.eoc_token_id]

            logits_hist.append(logits)
            p_hist.append(p_t)
            survival_hist.append(survival)

            if training:
                gated_memory.append((survival.unsqueeze(-1) * z_t).unsqueeze(1))
                survival = survival * (1 - p_t)
            else:
                stop_decision = (
                    (self.config.compress_stop_threshold is not None and p_t > self.config.compress_stop_threshold)
                    or (self.config.compress_stop_threshold is None and torch.bernoulli(p_t).bool())
                )
                keep_mask = (~stop_decision).unsqueeze(-1)
                if keep_mask.any():
                    gated_memory.append((z_t * keep_mask).unsqueeze(1))
                if torch.all(stop_decision):
                    break

            past = outputs.past_key_values
            past_total_len = self.get_past_key_values_len(past)
            z_prev = z_t

        if training:
            new_memory = torch.cat(gated_memory, dim=1) if gated_memory else torch.zeros(
                batch_size, 0, hidden_size, device=device, dtype=prefix_hidden.dtype
            )
            expected_length = torch.stack(survival_hist, dim=1).sum(dim=1) if survival_hist else torch.zeros(
                batch_size, device=device, dtype=prefix_hidden.dtype
            )
            original_length = segment_text_length + self.config.compression_alpha * memory_length
            length_penalty = self.config.compression_lambda * (expected_length / original_length)
        else:
            new_memory = torch.cat(gated_memory, dim=1) if gated_memory else prefix_hidden[:, :0]
            expected_length = None
            length_penalty = None

        return new_memory, {
            "logits": logits_hist,
            "p": p_hist,
            "survival": survival_hist,
            "expected_length": expected_length,
            "length_penalty": length_penalty,
        }, past

    def get_past_key_values_len(self, past_key_values):
        return 0 if past_key_values is None else past_key_values[0][0].size(2)

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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if past_key_values is not None and isinstance(past_key_values, dict):
            past_key_values, softprompt = past_key_values["past_key_values"], past_key_values["softprompt"]
        past_key_values_softprompt_length = softprompt.size(1) if softprompt is not None else 0

        if head_mask is not None:
            raise ValueError("Compressor does not support head_mask")
        if inputs_embeds is not None and input_ids is not None:
            raise ValueError("Compressor does not support both input_ids and input_embeds")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.size(0), inputs_embeds.size(1), dtype=torch.long, device=inputs_embeds.device
            )

        if past_key_values is not None:
            # Generation path without recompression
            self.summary_config.softprompt_length = past_key_values_softprompt_length
            self.summary_config.past_key_values_softprompt_length = 0
            self.summary_config.compression_length = 0
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            logits = self.lm_head(outputs.last_hidden_state)
            if not self.training:
                logits = self._mask_logits_for_control_tokens(logits)

            loss = None
            if labels is not None:
                masked_labels = self._mask_control_tokens_in_labels(input_ids, labels)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = masked_labels[..., 1:].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

            output = CausalACOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values={"past_key_values": outputs.past_key_values, "softprompt": softprompt},
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                softprompt=softprompt,
            )

            return output if return_dict else tuple(output.values())

        # Training/eval path with segmentation
        segment_lengths = segment_lengths if segment_lengths is not None else input_ids.size(1)

        inputs_embeds_list = torch.split(inputs_embeds, segment_lengths, dim=1)
        attention_mask_list = torch.split(attention_mask, segment_lengths, dim=1)
        input_ids_list = torch.split(input_ids, segment_lengths, dim=1)
        labels_list = torch.split(
            self._mask_control_tokens_in_labels(input_ids, labels) if labels is not None else None,
            segment_lengths,
            dim=1,
        ) if labels is not None else [None] * len(inputs_embeds_list)

        if softprompt is None:
            softprompt = inputs_embeds[:, :0, :]
        softprompt_detach_flag = False

        last_hidden_state_list = []
        output_attentions_list = []
        output_hidden_states_list = []
        logits_list = []
        length_penalties = []

        for step, segment_embeds in enumerate(inputs_embeds_list):
            segment_input_ids = input_ids_list[step]
            segment_attention_mask = attention_mask_list[step]
            segment_labels = labels_list[step]

            if self.config.truncate_bptt_segments == 1 and softprompt_detach_flag and softprompt is not None:
                softprompt = softprompt.detach()
                softprompt_detach_flag = False

            bsz = segment_embeds.size(0)
            softprompt_length = softprompt.size(1)

            segment_gradient_checkpointing = (
                getattr(self.config, "segment_gradient_checkpointing", False)
                and self.training
                and step < len(inputs_embeds_list) - 1
            )

            def decoder(segment_embeds, segment_attention_mask, segment_past_key_values, softprompt_length):
                self.summary_config.softprompt_length = softprompt_length
                self.summary_config.past_key_values_softprompt_length = 0
                self.summary_config.compression_length = 0
                return self.model(
                    inputs_embeds=segment_embeds,
                    attention_mask=segment_attention_mask,
                    past_key_values=segment_past_key_values,
                    use_cache=True,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=True,
                )

            segment_input = torch.cat([softprompt, segment_embeds], dim=1)
            device, attn_dtype = segment_embeds.device, segment_attention_mask.dtype
            full_attention_mask = torch.cat(
                [torch.ones(bsz, softprompt_length, device=device, dtype=attn_dtype), segment_attention_mask], dim=1
            )

            if segment_gradient_checkpointing:
                outputs = torch.utils.checkpoint.checkpoint(
                    decoder,
                    segment_input,
                    full_attention_mask,
                    None,
                    softprompt_length,
                    use_reentrant=False,
                )
            else:
                outputs = decoder(segment_input, full_attention_mask, None, softprompt_length)

            segment_hidden_states = outputs.last_hidden_state[:, softprompt_length:, :]
            segment_logits = self.lm_head(segment_hidden_states)
            if not self.training:
                segment_logits = self._mask_logits_for_control_tokens(segment_logits)

            logits_list.append(segment_logits)
            last_hidden_state_list.append(segment_hidden_states)
            output_attentions_list.append(outputs.attentions)
            output_hidden_states_list.append(outputs.hidden_states)

            # Compression is triggered by <sum> at the end of the segment
            sum_mask = segment_input_ids == self.config.sum_token_id
            if sum_mask.any():
                position_ids = torch.arange(segment_input_ids.size(1), device=device).unsqueeze(0)
                masked_positions = sum_mask * position_ids
                sum_index = masked_positions.max(dim=1).values
                sum_hidden = segment_hidden_states[torch.arange(bsz, device=device), sum_index, :]

                text_mask = segment_attention_mask * (segment_input_ids != self.config.sum_token_id) * (
                    segment_input_ids != self.config.eoc_token_id
                )
                segment_text_length = text_mask.sum(dim=1).to(segment_hidden_states.dtype)

                softprompt, stats, _ = self.compress_context(
                    sum_hidden,
                    outputs.past_key_values,
                    softprompt_length,
                    segment_text_length,
                    training=self.training,
                )

                if self.training and stats["length_penalty"] is not None:
                    length_penalties.append(stats["length_penalty"])
            elif self.config.recompress_memory:
                # No trigger token; keep memory as-is
                softprompt = softprompt

            softprompt_detach_flag = True

        self.summary_config.reset()

        logits = torch.cat(logits_list, dim=1)
        loss = None
        if labels is not None:
            all_labels = torch.cat(labels_list, dim=1)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = all_labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            if length_penalties:
                loss = loss + torch.stack(length_penalties, dim=0).mean()
        elif length_penalties:
            loss = torch.stack(length_penalties, dim=0).mean()

        output = CausalACOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values={"past_key_values": None, "softprompt": softprompt},
            hidden_states=output_hidden_states_list if output_hidden_states_list[0] is not None else None,
            attentions=output_attentions_list if output_attentions_list[0] is not None else None,
            softprompt=softprompt if output_softprompt or output_softprompt is None else softprompt[:, :0],
        )

        return output if return_dict else tuple(output.values())

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        model_inputs = super().prepare_inputs_for_generation(input_ids, past_key_values, attention_mask, inputs_embeds, **kwargs)
        model_inputs["softprompt"] = kwargs.get("softprompt", None)
        model_inputs["segment_lengths"] = kwargs.get("segment_lengths", None)
        return model_inputs


class OPTLearnedPositionalEmbeddingWithPadding(nn.Embedding):
    """Overwrite the default OPTLearnedPositionalEmbedding to disable position on softprompt/compression tokens."""

    def __init__(self, num_embeddings: int, embedding_dim: int, summary_config: Optional[SummaryConfig] = None):
        super().__init__(num_embeddings + 2, embedding_dim, padding_idx=1)
        self.summary_config = summary_config if summary_config is not None else SummaryConfig()

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        attention_mask = attention_mask.long()
        bsz = attention_mask.size(0)

        left_placeholder = torch.ones(
            bsz, self.summary_config.softprompt_length, dtype=torch.long, device=attention_mask.device
        )  # <pad> -> zero vector
        right_placeholder = torch.ones(
            bsz, self.summary_config.compression_length, dtype=torch.long, device=attention_mask.device
        )  # <pad> -> zero vector

        total_softprompt_length = self.summary_config.softprompt_length + self.summary_config.past_key_values_softprompt_length
        attention_mask = attention_mask[:, total_softprompt_length : attention_mask.size(1) - self.summary_config.compression_length]

        positions = attention_mask.cumsum(dim=1) * attention_mask + 1
        positions = positions[:, past_key_values_length - self.summary_config.past_key_values_softprompt_length :]
        positions = torch.cat([left_placeholder, positions, right_placeholder], dim=1)

        return super().forward(positions)


class OPTAutoCompressorModel(AutoCompressorMixin, OPTForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.setup_autocompressor(config)

        self.model.decoder.embed_positions = OPTLearnedPositionalEmbeddingWithPadding(
            config.max_position_embeddings, config.hidden_size, summary_config=self.summary_config
        )

        self.post_init()


AutoCompressorModel = OPTAutoCompressorModel


class LlamaAutoCompressorModel(AutoCompressorMixin, LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.setup_autocompressor(config)
        self.post_init()

    def get_past_key_values_len(self, past_key_values):
        return 0 if past_key_values is None else past_key_values[0][1]

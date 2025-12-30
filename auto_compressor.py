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
    summary_length: int = 0

    def reset(self):
        self.softprompt_length = 0
        self.past_key_values_softprompt_length = 0
        self.summary_length = 0


@dataclass
class CausalACOutputWithPast(CausalLMOutputWithPast):
    softprompt: Optional[torch.FloatTensor] = None
    compression_stats: Optional[Dict] = None


class AutoCompressorMixin:
    """Mixin class to turn a AutoModelForCausalLM into an AutoCompressor."""

    def setup_autocompressor(self, config):
        """Call this function in the subclass __init__ to initialize the autocompressor. Override for custom behaviour"""

        # Defaults for new variable-length compression behaviour
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

        self.summary_config = SummaryConfig()

    def get_past_key_values_len(self, past_key_values):
        return 0 if past_key_values is None else past_key_values[0][0].size(2)

    def _split_segments(self, input_ids, attention_mask, segment_lengths: Optional[Union[List[int], int]] = None):
        """Split the input sequence into text segments followed by a <sum> trigger.

        If ``segment_lengths`` is provided, we assume a <sum> token follows each segment. Otherwise, we search for
        sum_token_id positions in the first batch element and split accordingly.
        """

        _, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Determine valid length using the first example
        valid_length = attention_mask[0].sum().item()

        if segment_lengths is not None:
            if isinstance(segment_lengths, int):
                # Treat int as a uniform length; expect <sum> tokens in between
                num_segments = int(valid_length // (segment_lengths + 1))
                segment_lengths = [segment_lengths] * num_segments

            pointer = 0
            segments = []
            for seg_len in segment_lengths:
                start, end = pointer, pointer + seg_len
                if end >= seq_len:
                    raise ValueError("segment_lengths exceed input sequence length")
                segments.append(
                    (
                        input_ids[:, start : end + 1],
                        attention_mask[:, start : end + 1],
                    )
                )
                pointer = end + 1
            if pointer < valid_length:
                # Process trailing tokens without a compression request
                segments.append((input_ids[:, pointer:valid_length], attention_mask[:, pointer:valid_length]))
            return segments

        # Fallback: use <sum> token occurrences
        if self.config.sum_token_id is None:
            raise ValueError("sum_token_id must be set to split segments without explicit lengths")

        sum_positions = (input_ids[0, :valid_length] == self.config.sum_token_id).nonzero(as_tuple=True)[0].tolist()
        pointer = 0
        segments: List[Tuple[torch.LongTensor, torch.LongTensor]] = []
        for pos in sum_positions:
            if pos < pointer:
                continue
            segments.append((input_ids[:, pointer : pos + 1], attention_mask[:, pointer : pos + 1]))
            pointer = pos + 1
        if pointer < valid_length:
            segments.append((input_ids[:, pointer:valid_length], attention_mask[:, pointer:valid_length]))
        return segments

    def _compute_lm_loss(self, logits, labels):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        return loss

    def _run_decoder(
        self,
        inputs_embeds,
        attention_mask,
        past_key_values,
        softprompt_length,
        past_key_values_softprompt_length,
        use_cache,
        output_attentions,
        output_hidden_states,
    ):
        """Shared decoder wrapper that configures placeholder lengths for position embeddings."""

        self.summary_config.softprompt_length = softprompt_length
        self.summary_config.past_key_values_softprompt_length = past_key_values_softprompt_length
        self.summary_config.summary_length = 0

        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

    def _compress_context(
        self,
        initial_hidden: torch.FloatTensor,
        past_key_values,
        base_attention_mask: torch.LongTensor,
        softprompt_length: int,
        training: bool,
    ):
        """
        Run the manual compression loop to produce a variable-length soft prompt.

        Args:
            initial_hidden: last hidden state from the <sum> token (B, D)
            past_key_values: cache containing prefix context
            base_attention_mask: attention mask covering memory + segment + <sum>
            softprompt_length: length of the current memory already in past_key_values
            training: toggles soft termination vs. hard termination
        Returns:
            new_softprompt, stats
        """

        compression_max_len = self.config.compression_max_len
        if compression_max_len <= 0:
            return initial_hidden[:, None, :].detach() * 0, {}

        past_key_values_softprompt_length = softprompt_length
        current_past = past_key_values
        current_attention_mask = base_attention_mask

        z_prev = initial_hidden.unsqueeze(1)  # (B, 1, D)
        new_memory = []
        p_list = []
        survival_list = []

        bsz = initial_hidden.size(0)
        survival = torch.ones(bsz, 1, device=initial_hidden.device, dtype=initial_hidden.dtype)

        for _ in range(compression_max_len):
            attn_mask_step = torch.ones(bsz, 1, device=current_attention_mask.device, dtype=current_attention_mask.dtype)
            attn_mask_full = torch.cat([current_attention_mask, attn_mask_step], dim=1)

            outputs = self._run_decoder(
                z_prev,
                attn_mask_full,
                current_past,
                softprompt_length=0,
                past_key_values_softprompt_length=past_key_values_softprompt_length,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=True,
            )

            hidden = outputs.last_hidden_state[:, -1:, :]
            logits = self.lm_head(hidden).contiguous()
            if self.config.eoc_token_id is None:
                raise ValueError("eoc_token_id must be set before compression")
            p_t = torch.softmax(logits, dim=-1)[..., self.config.eoc_token_id]

            if training:
                survival_list.append(survival)
                p_list.append(p_t)

                gated_hidden = survival.unsqueeze(-1) * hidden
                new_memory.append(gated_hidden)
                survival = survival * (1 - p_t)
                z_prev = hidden
            else:
                next_token = torch.argmax(logits, dim=-1)
                stop_mask = next_token.squeeze(-1) == self.config.eoc_token_id
                continue_mask = (~stop_mask).to(hidden.dtype).view(bsz, 1, 1)

                if continue_mask.sum() > 0:
                    new_memory.append(hidden * continue_mask)
                    z_prev = hidden
                if stop_mask.all():
                    break

            current_past = outputs.past_key_values
            current_attention_mask = attn_mask_full

        if not new_memory:
            # No compression steps taken
            return initial_hidden[:, None, :].detach() * 0, {}

        softprompt = torch.cat(new_memory, dim=1)

        stats = {}
        if training and p_list:
            survival_stack = torch.cat(survival_list, dim=1)  # (B, L, 1)
            p_stack = torch.stack(p_list, dim=1)  # (B, L, 1)
            expected_length = survival_stack.sum(dim=1)  # (B, 1)
            stats = {
                "survival": survival_stack,
                "p_stop": p_stack,
                "expected_length": expected_length.squeeze(-1),
            }

        return softprompt, stats

    def forward_segment(
        self,
        softprompt: torch.FloatTensor,
        segment_embeds: torch.FloatTensor,
        segment_attention_mask: torch.LongTensor,
        past_key_values: PastKVType,
        output_hidden_states: bool,
        use_cache: bool,
        output_attentions: bool,
        segment_gradient_checkpointing: bool,
        past_key_values_softprompt_length: int,
    ):

        bsz = segment_embeds.size(0)
        if past_key_values_softprompt_length > 0:  # Softprompt should already be in past_key_values
            softprompt_length = 0
            inputs_embeds = segment_embeds

            device, attn_dtype = segment_embeds.device, segment_attention_mask.dtype
            attention_mask = torch.cat(
                [
                    torch.ones(bsz, past_key_values_softprompt_length, device=device, dtype=attn_dtype),
                    segment_attention_mask,
                ],
                dim=1,
            )
        else:
            softprompt_length = softprompt.size(1)
            inputs_embeds = torch.cat([softprompt, segment_embeds], dim=1)

            device, attn_dtype = segment_embeds.device, segment_attention_mask.dtype
            attention_mask = torch.cat(
                [
                    torch.ones(bsz, softprompt_length, device=device, dtype=attn_dtype),
                    segment_attention_mask,
                ],
                dim=1,
            )

        def decoder(inputs_embeds, attention_mask, segment_past_key_values, softprompt_length, past_key_values_softprompt_length):
            return self._run_decoder(
                inputs_embeds,
                attention_mask,
                segment_past_key_values,
                softprompt_length=softprompt_length,
                past_key_values_softprompt_length=past_key_values_softprompt_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        if segment_gradient_checkpointing:
            outputs = torch.utils.checkpoint.checkpoint(
                decoder,
                inputs_embeds,
                attention_mask,
                past_key_values,
                softprompt_length,
                past_key_values_softprompt_length,
                use_reentrant=False,
            )
        else:
            outputs = decoder(
                inputs_embeds,
                attention_mask,
                past_key_values,
                softprompt_length,
                past_key_values_softprompt_length,
            )

        total_length = outputs.last_hidden_state.size(1)
        segment_last_hiddens = outputs.last_hidden_state[:, softprompt_length:total_length]

        return outputs, segment_last_hiddens, attention_mask

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
        # We formulate the past_key_values as a tuple where the second entry is the softprompt already in the past key values
        if past_key_values is not None and isinstance(past_key_values, dict):
            # Replace softprompt in direct argument with the softprompt in past_key_values
            past_key_values, softprompt = past_key_values["past_key_values"], past_key_values["softprompt"]
            past_key_values_softprompt_length = softprompt.size(1)
        else:
            past_key_values_softprompt_length = 0

        past_key_values_length = self.get_past_key_values_len(past_key_values) - past_key_values_softprompt_length

        if head_mask is not None:
            raise ValueError("Compressor does not support head_mask")
        if inputs_embeds is not None and input_ids is not None:
            raise ValueError("Compressor does not support both input_ids and input_embeds")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if attention_mask is None and input_ids is not None:
            attention_mask = torch.ones_like(input_ids)

        # If no past_key_values are given, we will process the sequence in multiple segments
        if past_key_values is None:
            segments = self._split_segments(input_ids, attention_mask, segment_lengths) if input_ids is not None else []

            last_hidden_state_list = []
            output_attentions_list = []
            output_hidden_states_list = []
            compression_stats_list = []
            outputs = None

            if softprompt is None:
                softprompt = inputs_embeds[:, :0, :]

            segments_since_detach = 0

            pointer = 0
            for step, (segment_ids, segment_mask) in enumerate(segments):
                is_last_step = step == len(segments) - 1
                segment_gradient_checkpointing = (
                    getattr(self.config, "segment_gradient_checkpointing", False)
                    and self.training
                    and not is_last_step
                )

                seg_len = segment_ids.size(1)
                segment_embeds = inputs_embeds[:, pointer : pointer + seg_len, :]
                pointer += seg_len

                outputs, segment_hidden_states, combined_attention_mask = self.forward_segment(
                    softprompt.to(inputs_embeds.dtype),
                    segment_embeds,
                    segment_mask,
                    past_key_values,
                    output_hidden_states,
                    use_cache,
                    output_attentions,
                    segment_gradient_checkpointing,
                    past_key_values_softprompt_length,
                )

                last_hidden_state_list.append(segment_hidden_states)

                output_attentions_list.append(outputs.attentions)
                output_hidden_states_list.append(outputs.hidden_states)

                # Trigger compression if a <sum> token ends the segment
                has_sum = (
                    input_ids is not None
                    and segment_ids.size(1) > 0
                    and segment_ids[0, -1].item() == self.config.sum_token_id
                )
                if has_sum and self.config.compression_max_len > 0:
                    sum_hidden = segment_hidden_states[:, -1, :]
                    new_softprompt, comp_stats = self._compress_context(
                        sum_hidden,
                        outputs.past_key_values,
                        combined_attention_mask,
                        softprompt.size(1),
                        training=self.training,
                    )
                    compression_stats_list.append(comp_stats)

                    softprompt = new_softprompt
                    segments_since_detach += 1
                    if self.config.truncate_bptt_segments and segments_since_detach >= self.config.truncate_bptt_segments:
                        softprompt = softprompt.detach()
                        segments_since_detach = 0
                elif has_sum:
                    # Even without compression steps, drop the <sum> control token from memory pipeline
                    softprompt = softprompt.detach()

                if self.config.recompress_memory and has_sum:
                    past_key_values = None
                    past_key_values_softprompt_length = 0

            # Reset placeholder positions
            self.summary_config.reset()

            last_hiddens = (
                torch.cat(last_hidden_state_list, dim=1) if last_hidden_state_list else inputs_embeds[:, :0, :]
            )
            logits = self.lm_head(last_hiddens).contiguous()

            loss = None
            length_penalty = None
            if labels is not None:
                loss = self._compute_lm_loss(logits, labels)

            if compression_stats_list and self.training and self.config.compression_lambda > 0:
                expected_lengths = [stats.get("expected_length") for stats in compression_stats_list if stats]
                if expected_lengths:
                    expected_stack = torch.stack(expected_lengths, dim=0).mean(dim=0)
                    original_lengths = []
                    for seg_ids, seg_mask in segments:
                        if seg_ids.size(1) == 0 or seg_ids[0, -1].item() != self.config.sum_token_id:
                            continue
                        content_length = seg_mask[:, :-1].sum(dim=1)
                        original_lengths.append(content_length)
                    if original_lengths:
                        original_length = torch.stack(original_lengths, dim=0).float().mean(dim=0)
                        length_penalty = self.config.compression_lambda * (
                            expected_stack / torch.clamp(original_length, min=1.0)
                        )
                        loss = loss + length_penalty.mean() if loss is not None else length_penalty.mean()

            past_to_return = outputs.past_key_values if outputs is not None else None
            output = CausalACOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values={"past_key_values": past_to_return, "softprompt": softprompt},
                hidden_states=output_hidden_states_list
                if output_hidden_states_list and output_hidden_states_list[0] is not None
                else None,
                attentions=output_attentions_list if output_attentions_list and output_attentions_list[0] is not None else None,
                softprompt=softprompt,
                compression_stats={"length_penalty": length_penalty} if length_penalty is not None else None,
            )

            if return_dict:
                return output
            else:
                return tuple(output.values())

        # With past_key_values we will process the input in a single pass (for generation), except when generating memory
        else:
            if attention_mask is None:
                attention_mask = torch.ones(
                    inputs_embeds.size(0), inputs_embeds.size(1) + past_key_values_length, dtype=torch.long, device=inputs_embeds.device
                )

            outputs = self._run_decoder(
                inputs_embeds,
                attention_mask,
                past_key_values,
                softprompt_length=0,
                past_key_values_softprompt_length=past_key_values_softprompt_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            logits = self.lm_head(outputs.last_hidden_state).contiguous()

            loss = None
            if labels is not None:
                loss = self._compute_lm_loss(logits, labels)

            output = CausalACOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values={"past_key_values": outputs.past_key_values, "softprompt": softprompt},
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                softprompt=softprompt,
            )

            if return_dict:
                return output
            else:
                return tuple(output.values())

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        model_inputs = super().prepare_inputs_for_generation(input_ids, past_key_values, attention_mask, inputs_embeds, **kwargs)
        model_inputs["softprompt"] = kwargs.get("softprompt", None)
        model_inputs["segment_lengths"] = kwargs.get("segment_lengths", None)
        return model_inputs


class OPTLearnedPositionalEmbeddingWithPadding(nn.Embedding):
    """Overwrite the default OPTLearnedPositionalEmbedding to disable position on softprompt tokens"""

    def __init__(self, num_embeddings: int, embedding_dim: int, summary_config: Optional[SummaryConfig] = None):
        super().__init__(num_embeddings + 2, embedding_dim, padding_idx=1)

        self.summary_config = summary_config if summary_config is not None else SummaryConfig()

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        attention_mask = attention_mask.long()
        bsz = attention_mask.size(0)

        left_placeholder = torch.ones(
            bsz,
            self.summary_config.softprompt_length + self.summary_config.past_key_values_softprompt_length,
            dtype=torch.long,
            device=attention_mask.device,
        )  # <pad> -> zero vector
        right_placeholder = torch.ones(
            bsz, self.summary_config.summary_length, dtype=torch.long, device=attention_mask.device
        )  # <pad> -> zero vector

        total_softprompt_length = self.summary_config.softprompt_length + self.summary_config.past_key_values_softprompt_length
        attention_mask = attention_mask[
            :, total_softprompt_length : attention_mask.size(1) - self.summary_config.summary_length
        ]

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
        # modeling_flash_llama has slightly different layout of past key values
        return 0 if past_key_values is None else past_key_values[0][1]

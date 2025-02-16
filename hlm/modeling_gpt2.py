# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
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
"""PyTorch OpenAI GPT-2 model."""

import logging
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP, GPT2Attention, GPT2FlashAttention2, GPT2LMHeadModel, GPT2Model, GPT2SdpaAttention
from vector_quantize_pytorch import ResidualVQ, SimVQ, VectorQuantize, LFQ

from models import get_model
from utils import load_checkpoint


logger = logging.getLogger(__name__)


class HLMGPT2Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None, chunk_size=None):
        super().__init__(config, is_cross_attention, layer_idx)
        self.chunk_size = chunk_size
        max_positions = config.max_position_embeddings
        self.max_positions = max_positions

        mask = torch.tril(torch.ones(max_positions, max_positions))

        attention_mask = mask.masked_fill(mask == 0, float('-inf')) -1
        self.base_attn_mask = attention_mask

        mask = torch.tril(torch.ones(int(max_positions // self.chunk_size), int(max_positions // self.chunk_size)))

        attention_mask = mask.masked_fill(mask == 0, float('-inf')) -1
        self.ctx_attn_mask = attention_mask
        self.ctx_pred_attn_mask = self.ctx_token_attention_mask(int(max_positions // self.chunk_size) + max_positions, self.chunk_size+1, self.chunk_size+1)

    def ctx_token_attention_mask(self, seq_len, window_size, window_position):
        
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = torch.zeros(seq_len, seq_len)
        # 处理每个 window 的额外规则
        for start in range(0, seq_len, window_size):
            end = min(start + window_size, seq_len)
            
            # 当前 window 内的 tokens 可以看到自己窗口内的所有 tokens
            mask[start:end, start:end] = 1
            
            # 如果不是第一个 window，那么每个 window 的第一个 token 可以看到前一个 window 的第一个 token
            for prev_start in range(0, start, window_size):
                mask[start:end, prev_start] = 1
        mask *= causal_mask
        # import pdb; pdb.set_trace()
        mask = mask.masked_fill(mask == 0, float('-inf')) -1
        # mask = torch.zeros_like(causal_mask)
        return mask

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        
        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)


        if attention_mask is not None:
            # Apply the attention mask
            # import pdb; pdb.set_trace()
            if attn_weights.shape[-1] == self.max_positions:
                attn_weights = attn_weights + self.base_attn_mask.to(attn_weights.device)
                
            elif attn_weights.shape[-1] == int(self.max_positions // self.chunk_size):
                attn_weights = attn_weights + self.ctx_attn_mask.to(attn_weights.device)
                
            elif attn_weights.shape[-1] == self.max_positions + int(self.max_positions // self.chunk_size):
                attn_weights = attn_weights + self.ctx_pred_attn_mask.to(attn_weights.device)
            

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with torch.amp.autocast(query.device.type, enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            
            if attn_weights.shape[-1] == self.max_positions:
                attn_weights = attn_weights + self.base_attn_mask.to(attn_weights.device)
                
            elif attn_weights.shape[-1] == int(self.max_positions // self.chunk_size):
                attn_weights = attn_weights + self.ctx_attn_mask.to(attn_weights.device)
                
            elif attn_weights.shape[-1] == self.max_positions + int(self.max_positions // self.chunk_size):
                attn_weights = attn_weights + self.ctx_pred_attn_mask.to(attn_weights.device)
                

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights


GPT2_ATTENTION_CLASSES = {"eager": HLMGPT2Attention, "flash_attention_2": GPT2FlashAttention2, "sdpa": GPT2SdpaAttention}


class HLMGPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None, chunk_size=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        attention_class = GPT2_ATTENTION_CLASSES['eager']

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = attention_class(config=config, layer_idx=layer_idx, chunk_size=chunk_size)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = attention_class(config=config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        # import pdb; pdb.set_trace()
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual


        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class HLMGPT2Model(GPT2Model):
    _supports_param_buffer_assignment = False

    def __init__(self, config, input_layers=None, vae_model_config=None, ctx_layers=None, chunk_size=None):
        super().__init__(config)
        self.input_layers = input_layers
        self.embed_dim = config.hidden_size
        self.ctx_layers = ctx_layers

        self.vqvae = None
        if vae_model_config is not None:
            self.vqvae = get_model(vae_model_config['vae_config_path'])
            if vae_model_config['vae_pretrained_model_path'] is not None:
                load_checkpoint(self.vqvae, None, vae_model_config['vae_pretrained_model_path'])
                logger.info(f"vae model loaded with config {vae_model_config}")
        # self.vae_model = vae_model if vae_model != None else None
        self.chunk_size = chunk_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([HLMGPT2Block(config, layer_idx=i, chunk_size=self.chunk_size) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        self._attn_implementation = 'eager'

        # Initialize weights and apply final processing
        self.post_init()

        self.ctx_attn_mask = torch.tensor(1)
        
        self.ctx_lm_head = nn.Linear(self.embed_dim, self.vqvae.codebook_size * self.vqvae.num_quantizers)
        self.loss_fct = CrossEntropyLoss()

    def ctx_token_attention_mask(self, seq_len, window_size, window_position):
        
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = torch.zeros(seq_len, seq_len)
        # 处理每个 window 的额外规则
        for start in range(0, seq_len, window_size):
            end = min(start + window_size, seq_len)
            
            # 当前 window 内的 tokens 可以看到自己窗口内的所有 tokens
            mask[start:end, start:end] = 1
            
            # 如果不是第一个 window，那么每个 window 的第一个 token 可以看到前一个 window 的第一个 token
            for prev_start in range(0, start, window_size):
                mask[start:end, prev_start] = 1
        mask *= causal_mask
        # import pdb; pdb.set_trace()
        return mask


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        qualitized_loss = None

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Attention mask.
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        if self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif _use_sdpa:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask=attention_mask,
                input_shape=(batch_size, input_shape[-1]),
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_length,
            )
        else:
            if attention_mask is not None:
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif not self._attn_implementation == "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        
        
        if self.input_layers is None:
            
            for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
                # Model parallel
                
                if self.model_parallel:
                    torch.cuda.set_device(hidden_states.device)
                    # Ensure layer_past is on same device as hidden_states (might not be correct)
                    if layer_past is not None:
                        layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                    # Ensure that attention_mask is always on the same device as hidden_states
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(hidden_states.device)
                    if isinstance(head_mask, torch.Tensor):
                        head_mask = head_mask.to(hidden_states.device)
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    outputs = self._gradient_checkpointing_func(
                        block.__call__,
                        hidden_states,
                        None,
                        attention_mask,
                        head_mask[i],
                        encoder_hidden_states,
                        encoder_attention_mask,
                        use_cache,
                        output_attentions,
                    )
                else:
                    outputs = block(
                        hidden_states,
                        layer_past=layer_past,
                        attention_mask=attention_mask,
                        head_mask=head_mask[i],
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                    )

                hidden_states = outputs[0]
                if use_cache is True:
                    presents = presents + (outputs[1],)

                if output_attentions:
                    all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                    if self.config.add_cross_attention:
                        all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

                # Model Parallel: If it's the last layer for that device, put things on the next device
                if self.model_parallel:
                    for k, v in self.device_map.items():
                        if i == v[-1] and "cuda:" + str(k) != self.last_device:
                            hidden_states = hidden_states.to("cuda:" + str(k + 1))
        else:
            for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
                # Model parallel
                
                if self.model_parallel:
                    torch.cuda.set_device(hidden_states.device)
                    # Ensure layer_past is on same device as hidden_states (might not be correct)
                    if layer_past is not None:
                        layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                    # Ensure that attention_mask is always on the same device as hidden_states
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(hidden_states.device)
                    if isinstance(head_mask, torch.Tensor):
                        head_mask = head_mask.to(hidden_states.device)
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # import pdb; pdb.set_trace()
                if i == self.input_layers:
                    ctx_features = hidden_states
                    
                    batch_size, seq_len, hidden_size = hidden_states.size()
                    num_ctx_tokens = seq_len // self.chunk_size  # ctx tokens 的数量

                    hidden_states_reshaped = hidden_states.view(batch_size, num_ctx_tokens, self.chunk_size, hidden_size)
                    ctx_hidden_states = hidden_states_reshaped.mean(dim=2)

                    if isinstance(self.vqvae, VectorQuantize) or isinstance(self.vqvae, SimVQ) or isinstance(self.vqvae, ResidualVQ):
                        ctx_tokens, ctx_token_ids, cmt_loss = self.vqvae(ctx_hidden_states)
                        qualitized_loss = (ctx_tokens - ctx_hidden_states).abs().mean()
                        qualitized_loss += cmt_loss.mean()

                    if isinstance(self.vqvae, LFQ): 
                        ctx_tokens, ctx_token_ids, entropy_aux_loss = self.vqvae(ctx_hidden_states)
                        qualitized_loss = F.l1_loss(ctx_tokens, ctx_hidden_states)
                        qualitized_loss += entropy_aux_loss
                        
                    
                else: ctx_tokens=None
       
                
                if self.gradient_checkpointing and self.training:
                    outputs = self._gradient_checkpointing_func(
                        block.__call__,
                        hidden_states,
                        None,
                        attention_mask,
                        head_mask[i],
                        encoder_hidden_states,
                        encoder_attention_mask,
                        use_cache,
                        output_attentions,
                    )
                else:
                    if i == self.input_layers:
                        residual_hidden_states = hidden_states
                    # import pdb; pdb.set_trace()
                    if i < self.input_layers:
                        outputs = block(
                            hidden_states,
                            layer_past=layer_past,
                            attention_mask=attention_mask,
                            head_mask=head_mask[i],
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            use_cache=use_cache,
                            output_attentions=output_attentions,
                        )
                    elif self.input_layers <= i and i < self.ctx_layers:
                        if ctx_tokens is not None:    
                            # import pdb; pdb.set_trace()                      
                            outputs = block(
                                ctx_tokens,
                                layer_past=layer_past,
                                attention_mask=attention_mask,
                                head_mask=head_mask[i],
                                encoder_hidden_states=encoder_hidden_states,
                                encoder_attention_mask=encoder_attention_mask,
                                use_cache=use_cache,
                                output_attentions=output_attentions,
                            )
                        else:
                            outputs = block(
                                hidden_states,
                                layer_past=layer_past,
                                attention_mask=attention_mask,
                                head_mask=head_mask[i],
                                encoder_hidden_states=encoder_hidden_states,
                                encoder_attention_mask=encoder_attention_mask,
                                use_cache=use_cache,
                                output_attentions=output_attentions,
                            )
                        if i == self.ctx_layers - 1:
                            # import pdb; pdb.set_trace()
                            ctx_pred = self.ctx_lm_head(outputs[0]).view(ctx_token_ids.shape[0], ctx_token_ids.shape[1], ctx_token_ids.shape[2], self.vqvae.codebook_size)
                            ctx_loss = self.loss_fct(ctx_pred[..., :-1, :].contiguous().view(-1, ctx_pred.size(-1)), ctx_token_ids[..., 1:].contiguous().view(-1).detach())

                    else:
                        if i == self.ctx_layers:
                            max_indices = torch.argmax(ctx_pred, dim=-1)
                            # max_indices = ctx_token_ids
                            
                            if isinstance(self.vqvae, VectorQuantize) or isinstance(self.vqvae, ResidualVQ):
                                qualitized_states = self.vqvae.get_output_from_indices(max_indices)
                            
                            elif isinstance(self.vqvae, SimVQ) or isinstance(self.vqvae, LFQ):
                                qualitized_states = self.vqvae.indices_to_codes(max_indices)


                            qualitized_states = qualitized_states[:, :-1, :]
                            first_states = hidden_states[:, 0:1, :]
                            qualitized_states = torch.cat((first_states, qualitized_states), dim=1)


                            hidden_states_res = []
                            for i_token in range(qualitized_states.shape[1]):  # x2.shape[1] = 1024 / n
                                hidden_states_slice = residual_hidden_states[:, i_token * self.chunk_size: (i_token + 1) * self.chunk_size]  # Slice of n elements from x1    
                                slided_ctx_tokens = qualitized_states[:, i_token:i_token + 1]  # One token from x2                                
                                hidden_states_res.append(torch.cat([slided_ctx_tokens, hidden_states_slice], dim=1))  # Concatenate along the second dimension
                            # Concatenate the result list into a single tensor
                            hidden_states = torch.cat(hidden_states_res, dim=1)
                        outputs = block(
                            hidden_states,
                            layer_past=layer_past,
                            attention_mask=self.ctx_attn_mask.to(hidden_states.device),
                            # attention_mask=attention_mask,
                            head_mask=head_mask[i],
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            use_cache=use_cache,
                            output_attentions=output_attentions,
                        )

                hidden_states = outputs[0]
                if use_cache is True:
                    presents = presents + (outputs[1],)

                if output_attentions:
                    all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                    if self.config.add_cross_attention:
                        all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

                # Model Parallel: If it's the last layer for that device, put things on the next device
                if self.model_parallel:
                    for k, v in self.device_map.items():
                        if i == v[-1] and "cuda:" + str(k) != self.last_device:
                            hidden_states = hidden_states.to("cuda:" + str(k + 1))
        
        if hidden_states.shape[1] != 1024:
            hidden_states = hidden_states[:, torch.arange(hidden_states.size(1)) % (self.chunk_size+1) != 0, :]
            hidden_states = self.ln_f(hidden_states)
        else:
            hidden_states = self.ln_f(hidden_states)
        
        
        try:
            hidden_states = hidden_states.view(output_shape)
        except:
            print(hidden_states.shape)
            print(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        ), qualitized_loss, ctx_loss


class HLMGPT2LMHeadModel(GPT2LMHeadModel, PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, input_layers=None, ctx_layers=None, vae_model_config=None, training_type=None):

        super().__init__(config)
        
        self.input_layers = input_layers
        self.ctx_layers = ctx_layers
        self.training_type = training_type
        self.chunk_size = vae_model_config['chunk_size']

        self.transformer = HLMGPT2Model(config, self.input_layers, vae_model_config, self.ctx_layers, self.chunk_size)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_layers: Optional[int] = None,
        ctx_layers: Optional[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs, qualitized_loss, ctx_loss = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)
        loss = None
        if labels is None:
            labels = input_ids
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()           

            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            gpt_loss = loss
            
            if self.training_type == 'full' or self.training_type == 'ours':
                loss += qualitized_loss
                loss += ctx_loss

            elif self.training_type == 'after_input_layer_include_cb' or self.training_type == 'after_input_layer_exclude_cb':
                loss += ctx_loss

            elif self.training_type == 'except_codebook':
                loss += ctx_loss

            elif self.training_type == 'codebook':
                loss = qualitized_loss
            
            elif self.training_type == 'only_ctx_layer_exclude_cb' or 'after_input_layer_include_cb':
                loss = ctx_loss

            elif self.training_type == 'only_output_layer':
                loss = loss

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        ), qualitized_loss, ctx_loss, gpt_loss

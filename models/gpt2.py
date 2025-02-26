from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel, GPT2Model
from transformers import GenerationMixin
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from typing import Callable, Optional, Tuple, Union


class HLMGPT2(GPT2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, gpt2_config, model_name_or_path, vq_config):
        super().__init__(gpt2_config)
        self.transformer = GPT2Model(gpt2_config).from_pretrained(model_name_or_path)
        self.codebook_dim = vq_config['codebook_dim']
        self.embedding_dim = vq_config['embedding_dim']
        self.codebook_size = vq_config['codebook_size']
        self.num_quantizer = vq_config['num_quantizers']
        self.wte = torch.nn.Sequential(*[torch.nn.Embedding(self.codebook_size, self.embedding_dim) for _ in range(self.num_quantizer)])
        # model.wte_proj = torch.nn.Linear(self.codebook_dim, self.embedding_dim)
        self.vqhead = torch.nn.Sequential(*[torch.nn.Linear(self.embedding_dim, self.codebook_size) for _ in range(self.num_quantizer)])

        self.transformer.wte = nn.Identity()
        self.lm_head = nn.Identity()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        # self.post_init()

    # def to(self, device):
    #     self.transformer.to(device)
    #     self.wte = [w.to(device) for w in self.wte]
    #     self.vqhead = [v.to(device) for v in self.vqhead]
    #     self.model_parallel = device.type == "cuda"
    #     self.device_map = get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
    #     self.is_parallelizable = True
    #     self.transformer.model_parallel = self.model_parallel
    #     # self.transformer.half() if device.type == "cuda" else self.transformer.float()

    # def parameters(self, recurse = True):
    #     return super().parameters(recurse)

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        # self.wte = [w.to(self.transformer.first_device) for w in self.wte]
        # self.vqhead = [h.to(self.transformer.last_device) for h in self.vqhead]
        self.wte.to(self.transformer.first_device)
        self.vqhead.to(self.transformer.last_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        # self.wte = [w.to("cpu") for w in self.wte]
        # self.vqhead = [h.to('cpu') for h in self.vqhead]
        self.wte.to("cpu")
        self.vqhead.to('cpu')
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

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
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        assert input_ids is not None
        inputs_embeds = None
        for i in range(self.num_quantizer):
            if inputs_embeds is None:
                inputs_embeds = self.wte[i](input_ids[:, :, i])
            else:
                inputs_embeds += self.wte[i](input_ids[:, :, i])
        assert inputs_embeds is not None

        transformer_outputs = self.transformer(
            input_ids=None,
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
            hidden_states = hidden_states.to(self.vqhead[0].weight.device)

        lm_logits = []
        loss = None
        labels = labels.to(self.vqhead[0].weight.device)
        for i in range(self.num_quantizer):
            lm_logits.append(self.vqhead[i](hidden_states)) # Bx1024x8192
            if loss is None:
                loss = F.cross_entropy(lm_logits[-1].view(-1, lm_logits[-1].shape[-1]), labels[:,:,i].view(-1).long(), ignore_index=-100)
            else:
                loss += F.cross_entropy(lm_logits[-1].view(-1, lm_logits[-1].shape[-1]), labels[:,:,i].view(-1).long(), ignore_index=-100)
        loss = loss / self.num_quantizer
        lm_logits = lm_logits[-1] # avoid too much memory usage
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
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
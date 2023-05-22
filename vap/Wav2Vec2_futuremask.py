import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    Wav2Vec2BaseModelOutput,
    XVectorOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers import Wav2Vec2Model, Wav2Vec2Config
#from transformers.modeling_wav2vec2 import Wav2Vec2FeatureEncoder,Wav2Vec2FeatureProjection
import pdb

class Wav2Vec2ModelLeft2Right(Wav2Vec2Model):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        # self.config = config
        # self.feature_extractor = Wav2Vec2FeatureEncoder(config)
        # self.feature_projection = Wav2Vec2FeatureProjection(config)

        # # model only needs masking vector if mask prob is > 0.0
        # if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
        #     self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        # if config.do_stable_layer_norm:
        #     self.encoder = Wav2Vec2EncoderStableLayerNorm(config)
        # else:
        #     self.encoder = Wav2Vec2Encoder(config)

        # self.adapter = Wav2Vec2Adapter(config) if config.add_adapter else None

        # # Initialize weights and apply final processing
        # self.post_init()

    # def __init__(self):
    #     super().__init__()
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        #pdb.set_trace()
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        #if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            #attention_mask = self._get_feature_vector_attention_mask(
            #    extract_features.shape[1], attention_mask, add_adapter=False
            #)

         ## here mask out the future
        batch_size, seq_length, hidden_size = extract_features.shape
        # extract_features = extract_features.unsqueeze(1).repeat(1, seq_length, 1, 1)
        # future_mask = torch.triu(torch.ones(seq_length, seq_length)).bool()
        # future_mask = future_mask.unsqueeze(0).repeat(batch_size, 1,1)
        # extract_features = extract_features.masked_fill(future_mask.unsqueeze(-1), -float("inf"))
        hidden_states, extract_features = self.feature_projection(extract_features)
        # hidden_states = hidden_states.reshape(-1, hidden_states.shape[-2], hidden_states.shape[-1])


        future_mask = torch.triu(torch.ones(seq_length, seq_length)).bool()

        # hidden_states = self._mask_hidden_states(
        #     hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        # )
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=future_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        print('done encoding')
        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

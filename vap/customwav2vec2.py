from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from torch import Tensor
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import pdb
from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    Wav2Vec2BaseModelOutput,
    XVectorOutput,
    
)
from transformers import Wav2Vec2Processor,Wav2Vec2ForCTC,AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC

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
class W2V2Transformers(torch.nn.Module):
    def __init__(self, name="TencentGameMate/chinese-wav2vec2-base"):
        super().__init__()
        model = Wav2Vec2Model.from_pretrained(name)
        #model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

        self.feature_extractor = model.feature_extractor
        #self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        
        #processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

        self.feature_projection = model.feature_projection

        # Transformer
        self.pos_conv_embed = model.encoder.pos_conv_embed
        self.layer_norm = model.encoder.layer_norm
        self.dropout = model.encoder.dropout
        self.layers = model.encoder.layers
        self.decrease_dimension = nn.Linear(768, 256)




    def create_causal_mask(self, x: Tensor) -> Tensor:
        assert (
            x.ndim == 3
        ), f"Expected x to b of [B, N_FRAMES, D] dimensions, got {x.shape}"
        b, n, _ = x.size()
        mask = torch.tril(torch.ones((n, n), device=x.device, dtype=torch.bool)).view(
            n, n
        )
        mask = mask.repeat(b, 1, 1).unsqueeze(1)
        mask.requires_grad_(False)
        return mask

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = x.squeeze(1)
        x = self.feature_extractor(x)
       # pdb.set_trace()
        x = x.transpose(1, 2)
        return self.feature_projection(x)
    def encoder(self, hidden_states, causal: bool = True):
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        attention_mask = None
        if causal:
            attention_mask = self.create_causal_mask(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        
        # hid_copy = torch.clone(hidden_states)

        # for layer in self.layers:
        #     layer_outputs = layer(
        #         hid_copy,
        #         output_attentions=False,
        #     )
        #     hid_copy = layer_outputs[0]

        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]
        # print(hid_copy == hidden_states)
        # pdb.set_trace()
        #lhs_lowdim = self.decrease_dimension(hidden_states)
        #lhs_lowdim = torch.relu(lhs_lowdim)
        return hidden_states
    
    def forward(self, x, causal: bool = True):
        hidden_states, extract_features = self.encode(x)
        hidden_states = self.encoder(hidden_states, causal)
        return hidden_states
    
    # def forward(self, x, causal: bool = True):
    #     hidden_states, extract_features = self.encode(x)

    #     # Transformer, aka Wav2Vec2Encoder
    #     position_embeddings = self.pos_conv_embed(hidden_states)
    #     hidden_states = hidden_states + position_embeddings
    #     hidden_states = self.dropout(hidden_states)

    #     attention_mask = None
    #     if causal:
    #         attention_mask = self.create_causal_mask(hidden_states)
    #     hidden_states = self.layer_norm(hidden_states)
    #     for layer in self.layers:
    #         layer_outputs = layer(
    #             hidden_states,
    #             attention_mask=attention_mask,
    #             output_attentions=False,
    #         )
    #         hidden_states = layer_outputs[0]

    #     lhs_lowdim = self.decrease_dimension(hidden_states)
    #     lhs_lowdim = torch.relu(lhs_lowdim)

    #     return lhs_lowdim
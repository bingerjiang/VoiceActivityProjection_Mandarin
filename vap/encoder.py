import torch
import torch.nn as nn
import einops
from vap.encoder_components import load_CPC, get_cnn_layer
from torch import Tensor
## import wav2vec stuff
import torch.nn.functional as F
import soundfile as sf
from fairseq import checkpoint_utils

from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
   # Wav2Vec2Model,
    HubertModel,
)
from vap.Wav2Vec2_futuremask import Wav2Vec2ModelLeft2Right
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
from transformers import AutoProcessor, AutoModelForPreTraining

import pdb

class EncoderCPC(nn.Module):
    """
    Encoder: waveform -> h
    pretrained: default='cpc'

    A simpler version of the Encoder
    check paper (branch) version to see other encoders...
    """

    def __init__(self, load_pretrained=True, freeze=True, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.encoder = load_CPC(load_pretrained)
        self.output_dim = self.encoder.gEncoder.conv4.out_channels
        self.dim = self.output_dim

        self.downsample_ratio = 160
        self.downsample = get_cnn_layer(
            dim=self.output_dim,
            kernel=[5],
            stride=[2],
            dilation=[1],
            activation="GELU",
        )
        self.downsample_ratio = 320

        if freeze:
            self.freeze()

    def get_default_conf(self):
        return {""}

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        print(f"Froze {self.__class__.__name__}!")

    def unfreeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(True)
        print(f"Trainable {self.__class__.__name__}!")

    def forward(self, waveform):
        if waveform.ndim < 3:
            waveform = waveform.unsqueeze(1)  # channel dim

        # Backwards using only the encoder encounters:
        # ---------------------------------------------------
        # RuntimeError: one of the variables needed for gradient computation
        # has been modified by an inplace operation:
        # [torch.FloatTensor [4, 256, 1000]], which is output 0 of ReluBackward0, is at version 1;
        # expected version 0 instead. Hint: enable anomaly detection to find
        # the operation that failed to compute its gradient, with
        # torch.autograd.set_detect_anomaly(True).
        # HOWEVER, if we feed through encoder.gAR we do not encounter that problem...
        #pdb.set_trace()
        z = self.encoder.gEncoder(waveform)
        z = einops.rearrange(z, "b c n -> b n c")
        z = self.encoder.gAR(z)
        z = self.downsample(z)
        #pdb.set_trace()
        return z

class EncoderWav2vec(nn.Module):
    """
    Encoder: waveform -> h
    pretrained: default='cpc'

    A simpler version of the Encoder
    check paper (branch) version to see other encoders...
    """

    def __init__(self, load_pretrained=True, freeze=True, sample_rate=16000):
        super().__init__()
        model = Wav2Vec2ModelLeft2Right.from_pretrained("TencentGameMate/chinese-wav2vec2-base")
        self.sample_rate = sample_rate
        #self.encoder = Wav2Vec2ModelLeft2Right.from_pretrained("TencentGameMate/chinese-wav2vec2-base",return_dict=True)
        #self.encoder = HubertModel.from_pretrained("TencentGameMate/chinese-hubert-base")

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("TencentGameMate/chinese-wav2vec2-base")
        self.decrease_dimension = nn.Linear(768, 256)
        
        #self.output_dim = self.encoder.gEncoder.conv4.out_channels
        #self.dim = self.output_dim

        #self.downsample_ratio = 160
        # self.downsample = get_cnn_layer(
        #     dim=self.output_dim,
        #     kernel=[5],
        #     stride=[2],
        #     dilation=[1],
        #     activation="GELU",
        # )
        #self.downsample_ratio = 320
        self.feature_projection = model.feature_projection

        # Transformer
        self.pos_conv_embed = model.encoder.pos_conv_embed
        self.layer_norm = model.encoder.layer_norm
        self.dropout = model.encoder.dropout
        self.layers = model.encoder.layers
        if freeze:
            self.freeze()

    def get_default_conf(self):
        return {""}

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        print(f"Froze {self.__class__.__name__}!")

    def unfreeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(True)
        print(f"Trainable {self.__class__.__name__}!")
    
   
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
        x = self.feature_extractor(x,return_tensors="pt",
                                            sampling_rate=16000,
                                            padding=True).input_values
        pdb.set_trace()
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        return self.feature_projection(x)
    
    def forward(self, waveform, causal: bool = True):
        #pdb.set_trace()
        if waveform.ndim < 3:
            waveform = waveform.unsqueeze(1)  # channel dim
        if waveform.ndim >3:
            waveform = wavform.squeeze(0)
            pdb.set_trace()
        waveform = waveform.squeeze(1)
        input_values = self.feature_extractor(waveform,return_tensors="pt",
                                            sampling_rate=16000,
                                            padding=True).input_values
        pdb.set_trace()
        extract_features = self.feature_extractor(input_values,sampling_rate=16000).input_values
        pdb.set_trace()
        extract_features = extract_features[0].transpose(1, 2)
        hidden_states, extract_features = self.feature_projection(extract_features)
        # if input_values.ndim>3:
 
        #     input_values = input_values.squeeze(0).squeeze(1)


        #hidden_states, extract_features = self.encode(waveform)

        # Transformer, aka Wav2Vec2Encoder
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        attention_mask = None
        if causal:
            attention_mask = self.create_causal_mask(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=False,
            )
            last_hidden_states = layer_outputs[0]        
        

        lhs_lowdim = self.decrease_dimension(last_hidden_state)
        lhs_lowdim = torch.relu(lhs_lowdim)

        
        ## 
        # batch_size, seq_length, hidden_size = lhs_lowdim.shape
        # future_mask = torch.triu(torch.ones(seq_length, seq_length)).bool()
        # masked_last_hidden_layer = lhs_lowdim.masked_fill(future_mask, 0)
        ###
        

        return lhs_lowdim


if __name__ == '__main__':

    

    model_path="TencentGameMate/chinese-wav2vec2-base"
    wav_path="/home/binger/repo/vap/VoiceActivityProjection/example/example.wav"
    mask_prob=0.0
    mask_length=10
    #model = AutoModelForPreTraining.from_pretrained("TencentGameMate/chinese-wav2vec2-base")

    model = EncoderWav2vec()

    pdb.set_trace()

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("TencentGameMate/chinese-wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained(model_path)

    # for pretrain: Wav2Vec2ForPreTraining
    # model = Wav2Vec2ForPreTraining.from_pretrained(model_path)
    device = "cpu"
    if torch.cuda.is_available():
        model = model.to("cuda")
        device = "cuda"
    
    model = model.to(device)
    model = model.half()
    model.eval()

    
    
    wav, sr = sf.read(wav_path)
    input_values = feature_extractor(wav, return_tensors="pt").input_values
    input_values = input_values.half()
    input_values = input_values.to(device)
    pdb.set_trace()
    # for Wav2Vec2ForPreTraining
    # batch_size, raw_sequence_length = input_values.shape
    # sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)
    # mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.0, mask_length=2)
    # mask_time_indices = torch.tensor(mask_time_indices, device=input_values.device, dtype=torch.long)

    with torch.no_grad():
        outputs = model(input_values[:,:,0])
        pdb.set_trace()

        last_hidden_state = outputs.last_hidden_state
        
        # for Wav2Vec2ForPreTraining
        # outputs = model(input_values, mask_time_indices=mask_time_indices, output_hidden_states=True)
        # last_hidden_state = outputs.hidden_states[-1]
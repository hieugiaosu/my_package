import torch
import torch.nn as nn
from ...modules.whyv2 import CausalIntraAndInterBandModule
from ...modules.whyv import AllHeadPReLULayerNormalization4DC, LayerNormalization, WHYVFilterGate
from typing import Optional, Tuple, List
from fla.layers import GatedLinearAttention, DeltaNet
from fla.models.utils import Cache
from einops import rearrange
from ...modules.whyv import DimensionEmbedding, WHYVDeconv
from ...modules.input import RMSNormalizeInput, STFTInput
from ...modules.output import RMSDenormalizeOutput, WaveGeneratorByISTFT

class WHYV2BlockWithGLA(nn.Module):
    def __init__(
        self,
        emb_dim = 48,
        kernel_size:int = 4,
        emb_hop_size:int = 1,
        n_freqs = 65,
        hidden_channels:int = 192,
        n_head=4,
        activation="PReLU",
        eps = 1e-5,
        layer_idx = 0
    ):
        super().__init__()
        assert emb_dim % n_head == 0, "emb_dim must be divisible by n_head"
        E = emb_dim // n_head
        self.intra_and_inter_band_module = CausalIntraAndInterBandModule(
            emb_dim=emb_dim,
            kernel_size=kernel_size,
            emb_hop_size=emb_hop_size,
            hidden_channels=hidden_channels,
            eps=eps
        )

        self.conv = nn.Conv2d(emb_dim,emb_dim,1)
        self.norm = AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps)
        self.down_sample = nn.Linear(E*n_freqs, 256)
        self.attention = GatedLinearAttention(
            # hidden_size=E*n_freqs,
            hidden_size=256,
            expand_k=0.5,
            expand_v = 1.0,
            num_heads=1,
            layer_idx = layer_idx
        )
        # self.attention = DeltaNet(
        #     hidden_size=256,
        #     expand_k=1,
        #     expand_v = 1.0,
        #     num_heads=1,
        #     layer_idx = layer_idx,
        #     # use_fast_conv1d = False
        # )
        self.up_sample = nn.Linear(256, E*n_freqs)
        self.concat_proj = nn.Sequential(
            nn.Conv2d(emb_dim,emb_dim,1),
            getattr(nn,activation)(),
            LayerNormalization(emb_dim, dim=-3, total_dim=4, eps=eps),
        )

        self.filter_gate = WHYVFilterGate(emb_dim,n_freqs)
    
    def forward(
            self,
            x,gtf,gtb,
            gla_past_key_values: Optional[Cache] = None,
            lstm_hidden: Optional[Tuple[torch.Tensor,torch.Tensor]] = None,
            return_hidden: bool = False
            ):
        y, h = self.intra_and_inter_band_module(x, lstm_hidden, return_hidden=True)
        batch, _,  freq, frame = y.shape
        y = rearrange(y,"B C Q T -> B C T Q")
        Q = self.norm(self.conv(y)) # [B, n_head, C, T, Q]
        n_head = Q.shape[1]
        Q = rearrange(Q, "B H C T Q -> (B H) T (C Q)")
        Q = self.down_sample(Q)
        att, _ , gla_hidden = self.attention(Q, past_key_values=gla_past_key_values, use_cache=return_hidden)
        att = self.up_sample(att)
        att = rearrange(att, "(B H) T (C Q) -> B (H C) T Q", Q=freq, H = n_head, B = batch, T=frame)
        att = self.concat_proj(att)
        o = att + y
        o = rearrange(o, "B C T Q -> B C Q T")
        o = self.filter_gate(o,gtf,gtb)
        if return_hidden:
            return o, gla_hidden, h
        else:
            return o

class WHYV2WithGLA(nn.Module):
    def __init__(
            self,
            n_fft=128,
            hop_length=64,
            window="hann",
            n_audio_channel=1,
            n_layers=5,
            input_kernel_size_T = 3,
            input_kernel_size_F = 3,
            output_kernel_size_T = 3,
            output_kernel_size_F = 3,
            lstm_hidden_units=192,
            attn_n_head=4,
            emb_dim=48,
            emb_ks=4,
            emb_hs=1,
            activation="PReLU",
            eps=1.0e-5,
            conditional_dim = 256,
            ):
        super().__init__()
        n_freqs = n_fft//2 + 1
        self.input_normalize = RMSNormalizeInput((1,2),keepdim=True)
        self.stft = STFTInput(
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            window=window,
        )

        self.istft = WaveGeneratorByISTFT(
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            window=window
        )

        self.output_denormalize = RMSDenormalizeOutput()

        self.dimension_embedding = DimensionEmbedding(
            audio_channel=n_audio_channel,
            emb_dim=emb_dim,
            kernel_size=(input_kernel_size_F,input_kernel_size_T),
            eps=eps
        )

        self.whyv2_with_gla_block = nn.ModuleList(
            [
                WHYV2BlockWithGLA(
                    emb_dim=emb_dim,
                    kernel_size=emb_ks,
                    emb_hop_size=emb_hs,
                    n_freqs=n_freqs,
                    hidden_channels=lstm_hidden_units,
                    n_head=attn_n_head,
                    activation=activation,
                    eps=eps,
                    layer_idx = i
                ) for i in range(n_layers)
            ]
        )

        self.filter_gen = nn.Linear(conditional_dim,emb_dim*n_freqs)
        self.bias_gen = nn.Linear(conditional_dim,emb_dim*n_freqs)

        self.deconv = WHYVDeconv(
            emb_dim=emb_dim,
            n_srcs=1,
            kernel_size_T=output_kernel_size_T,
            kernel_size_F=output_kernel_size_F,
            padding_F=output_kernel_size_F//2,
            padding_T=output_kernel_size_T//2
            )
        
        self.n_layers = n_layers

    def forward(self, input, speaker_embedding, lstm_hidden: List[Tuple[torch.Tensor,torch.Tensor]] = [], gla_hidden:Optional[Cache] = None, return_hidden=False):
        if gla_hidden is None and return_hidden:
            gla_hidden = Cache()
        audio_length = input.shape[-1]
        x = input

        if x.dim() == 2:
            x = x.unsqueeze(1)

        x, std = self.input_normalize(x)

        x = self.stft(x)

        x = self.dimension_embedding(x)

        n_freqs = x.shape[-2]
        gtf = self.filter_gen(speaker_embedding)
        gtb = self.bias_gen(speaker_embedding)
        gtf = rearrange(gtf,"b (d q) -> b d q 1", q = n_freqs)
        gtb = rearrange(gtb,"b (d q) -> b d q 1", q = n_freqs)
        new_hidden = []
        if return_hidden:
            for i in range(self.n_layers):
                layer_lstm_hidden = lstm_hidden[i] if i < len(lstm_hidden) else None
                x,gla_hidden, new_layer_lstm_hidden = self.whyv2_with_gla_block[i](x,gtf,gtb,gla_past_key_values=gla_hidden,lstm_hidden=layer_lstm_hidden,return_hidden=return_hidden)
                new_hidden.append(new_layer_lstm_hidden)
        else:
            for i in range(self.n_layers):
                layer_lstm_hidden = lstm_hidden[i] if i < len(lstm_hidden) else None
                x = self.whyv2_with_gla_block[i](x,gtf,gtb,gla_past_key_values=gla_hidden,lstm_hidden=layer_lstm_hidden,return_hidden=return_hidden)
        
        x = self.deconv(x)

        x = rearrange(x,"B C N F T -> B N C F T") #becasue in istft, the 1 dim is for real and im part

        x = self.istft(x,audio_length)

        x = self.output_denormalize(x,std)

        if return_hidden:
            return x[:,0], gla_hidden, new_hidden
        else:
            return x[:,0]
        

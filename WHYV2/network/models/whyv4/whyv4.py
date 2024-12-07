import torch
import torch.nn as nn
from einops import rearrange

from ...modules.input import RMSNormalizeInput, SplitBandEmbedding, STFTInput
from ...modules.output import (RMSDenormalizeOutput, SplitBandDeconv,
                               WaveGeneratorByISTFT)
from ...modules.whyv import *
from ...modules.whyv4 import WHYV4Block


class WHYV4(nn.Module):
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
            qk_output_channel=4,
            emb_dim=48,
            emb_ks=4,
            emb_hs=1,
            activation="PReLU",
            eps=1.0e-5,
            conditional_dim = 256
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
        # self.dimension_embedding = SplitBandEmbedding(
        #     emb_dim=emb_dim,
        #     input_n_freqs=n_freqs,
        #     kernel_size=(input_kernel_size_F,input_kernel_size_T),
        #     padding="same",
        #     audio_channel=n_audio_channel,
        #     eps=eps
        # )

        # n_freqs = self.dimension_embedding.output_n_freqs
        self.whyv_block = nn.ModuleList(
            [
                WHYV4Block(
                    emb_dim=emb_dim,
                    kernel_size=emb_ks,
                    emb_hop_size=emb_hs,
                    hidden_channels=lstm_hidden_units,
                    n_head=attn_n_head,
                    qk_output_channel=qk_output_channel,
                    activation=activation,
                    eps=eps,
                    n_freqs=n_freqs
                ) for _ in range(n_layers)
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
        
        # self.deconv = SplitBandDeconv(
        #     emb_dim=emb_dim,
        #     input_n_freqs=n_freqs,
        #     kernel_size=(output_kernel_size_F,output_kernel_size_T),
        #     padding=(output_kernel_size_F//2,output_kernel_size_T//2),
        #     eps=eps
        # )

        self.n_layers = n_layers
    def forward(self,input, clue):
        audio_length = input.shape[-1]

        x = input

        if x.dim() == 2:
            x = x.unsqueeze(1)

        x, std = self.input_normalize(x)

        x = self.stft(x)

        x = self.dimension_embedding(x)

        n_freqs = x.shape[-2]
        f = self.filter_gen(clue)
        b = self.bias_gen(clue)
        f = rearrange(f,"b (d q) -> b d q 1", q = n_freqs)
        b = rearrange(b,"b (d q) -> b d q 1", q = n_freqs)

        for i in range(self.n_layers):
            x = self.whyv_block[i](x,f,b)

        x = self.deconv(x)

        # print(x.shape)
        x = rearrange(x,"B C N F T -> B N C F T") #becasue in istft, the 1 dim is for real and im part

        x = self.istft(x,audio_length)

        x = self.output_denormalize(x,std)
        return x[:,0]
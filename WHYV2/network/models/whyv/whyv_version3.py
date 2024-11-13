import torch.nn as nn
import torch
from einops import rearrange
from ...modules.input import RMSNormalizeInput, STFTInput
from ...modules.output import RMSDenormalizeOutput, WaveGeneratorByISTFT
from ...modules.whyv import *
from typing import Iterable
import torch.nn.functional as F

class WHYVv3(nn.Module):
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
            conditional_dim = 256,
            pretrain_encoder = None
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
        if pretrain_encoder is not None:
            self.dimension_embedding.load_state_dict(
                torch.load(pretrain_encoder, map_location='cpu')
            )
            for param in self.dimension_embedding.parameters():
                param.requires_grad = False

        self.whyv_block = nn.ModuleList(
            [
                WHYVBlock(
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
        
        self.n_layers = n_layers
    def forward(
            self,
            input, 
            clue, 
            inference = False, 
            hidden_present:Iterable[torch.Tensor] = []):
        audio_length = input.shape[-1]

        x = input

        if x.dim() == 2:
            x = x.unsqueeze(1)
        if not inference:
            for idx in range(len(hidden_present)):
                hidden_present[idx] = hidden_present[idx].unsqueeze(1) if hidden_present[idx].dim() == 2 else hidden_present[idx]
                hidden_present[idx], _ = self.input_normalize(hidden_present[idx])
                hidden_present[idx] = self.stft(hidden_present[idx])
                hidden_present[idx] = self.dimension_embedding(hidden_present[idx]) 
            
        x, std = self.input_normalize(x)

        x = self.stft(x)

        x = self.dimension_embedding(x)

        n_freqs = x.shape[-2]
        f = self.filter_gen(clue)
        b = self.bias_gen(clue)
        f = rearrange(f,"b (d q) -> b d q 1", q = n_freqs)
        b = rearrange(b,"b (d q) -> b d q 1", q = n_freqs)
        loss = 0
        if not inference:
            for i in range(self.n_layers):
                x = self.whyv_block[i](x,f,b)
                loss += F.mse_loss(x, hidden_present[i])
        else:
            for i in range(self.n_layers):
                x = self.whyv_block[i](x,f,b)

        x = self.deconv(x)

        x = rearrange(x,"B C N F T -> B N C F T") #becasue in istft, the 1 dim is for real and im part

        x = self.istft(x,audio_length)

        x = self.output_denormalize(x,std)
        if not inference:
            return x[:,0], loss
        else:
            return x[:,0]
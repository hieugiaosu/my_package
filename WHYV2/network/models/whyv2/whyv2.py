import torch.nn as nn
from einops import rearrange
from ...modules.input import RMSNormalizeInput, STFTInput
from ...modules.output import RMSDenormalizeOutput, WaveGeneratorByISTFT
from ...modules.whyv2 import *
from ...modules.whyv import DimensionEmbedding, WHYVDeconv
from .schema import Whyv2Reference

class WHYV2(nn.Module):
    def __init__(
            self,
            n_fft=128,
            hop_length=64,
            window="hann",
            n_audio_channel=1,
            n_layers=4,
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
            n_selection_frame = 50
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

        self.whyv2_block = nn.ModuleList(
            [
                WHYV2block(
                    emb_dim=emb_dim,
                    kernel_size=emb_ks,
                    emb_hop_size=emb_hs,
                    n_selection_frame=n_selection_frame,
                    n_freqs=n_freqs,
                    hidden_channels=lstm_hidden_units,
                    n_head=attn_n_head,
                    activation=activation,
                    eps=eps
                )
                for _ in range(n_layers)
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
    def regigster_reference(self, reference_waveform, speaker_embedding):
        """
        Args: 
            reference_waveform (torch.Tensor): shape (batch, frame)
            speaker_embedding (torch.Tensor): shape (batch, conditional_dim)
        """
        x = reference_waveform

        if x.dim() == 2:
            x = x.unsqueeze(1)

        x, _ = self.input_normalize(x)

        x = self.stft(x)

        x = self.dimension_embedding(x)

        n_freqs = x.shape[-2]
        gtf = self.filter_gen(speaker_embedding)
        gtb = self.bias_gen(speaker_embedding)
        gtf = rearrange(gtf,"b (d q) -> b d q 1", q = n_freqs)
        gtb = rearrange(gtb,"b (d q) -> b d q 1", q = n_freqs)

        ref_out_put = Whyv2Reference(
            gtf=gtf,
            gtb=gtb,
            whyv2_block_ref=[]
        )
        i = x
        for block in self.whyv2_block:
            o = block.register_reference(i,gtf,gtb)
            ref_out_put.whyv2_block_ref.append(o)
            i = o.output
        
        return ref_out_put
    
    def forward(self, input, whyv2_reference: Whyv2Reference):
        audio_length = input.shape[-1]
        x = input
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x, std = self.input_normalize(x)
        x = self.stft(x)

        x = self.dimension_embedding(x)

        for idx, block in enumerate(self.whyv2_block):
            x = block(
                Whyv2BlockForwardInput(
                    x=x,
                    gtf=whyv2_reference.gtf,
                    gtb=whyv2_reference.gtb,
                    lstm_hidden=whyv2_reference.whyv2_block_ref[idx].lstm_hidden,
                    selection_frame=whyv2_reference.whyv2_block_ref[idx].selection_frame
                )
            )
        
        x = self.deconv(x)

        x = rearrange(x,"B C N F T -> B N C F T") #becasue in istft, the 1 dim is for real and im part

        x = self.istft(x,audio_length)

        x = self.output_denormalize(x,std)

        return x[:,0]


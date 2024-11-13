import torch
import torch.nn as nn
from ...modules.input import RMSNormalizeInput, STFTInput
from ...modules.output import RMSDenormalizeOutput, WaveGeneratorByISTFT
from ...modules.whyv import DimensionEmbedding, WHYVDeconv, TFGridnetBlock, \
WHYVFilterGate, IntraAndInterBandModule, CrossFrameSelfAttention
from einops import rearrange
import torch.nn.functional as F

class TFEncoder(nn.Module):
    def __init__(
            self,
            n_fft=128,
            hop_length=64,
            window="hann",
            n_audio_channel=1,
            emb_dim=48,
            kernel_size_T = 3,
            kernel_size_F = 3,
            eps=1.0e-5
            ):
        super().__init__()
        self.input_normalize = RMSNormalizeInput((1,2),keepdim=True)
        self.stft = STFTInput(
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            window=window,
        )
        self.dimension_embedding = DimensionEmbedding(
            audio_channel=n_audio_channel,
            emb_dim=emb_dim,
            kernel_size=(kernel_size_F,kernel_size_T),
            eps=eps
        )
    def forward(self,input):
        x = input
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x,std = self.input_normalize(x)

        x = self.stft(x)

        x = self.dimension_embedding(x)

        return x,std

class TFDecoder(nn.Module):
    def __init__(
            self,
            n_fft=128,
            hop_length=64,
            window="hann",
            emb_dim=48,
            kernel_size_T = 3,
            kernel_size_F = 3
            ):
        super().__init__()
        self.deconv = WHYVDeconv(
            emb_dim=emb_dim,
            n_srcs=1,
            kernel_size_T=kernel_size_T,
            kernel_size_F=kernel_size_F,
            padding_F=kernel_size_F//2,
            padding_T=kernel_size_T//2
            )
        self.istft = WaveGeneratorByISTFT(
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            window=window
        )
        self.output_denormalize = RMSDenormalizeOutput()
    def forward(self,x,std,audio_length):
        x = self.deconv(x)
        x = rearrange(x,"B C N F T -> B N C F T") #becasue in istft, the 1 dim is for real and im part
        x = self.istft(x,audio_length)
        x = self.output_denormalize(x,std)
        return x

class WHYV3(nn.Module):
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
            conditional_dim = 256
            ):
        super().__init__()
        n_freqs = n_fft//2 + 1
        self.input_encoder = TFEncoder(
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            n_audio_channel=n_audio_channel,
            emb_dim=emb_dim,
            kernel_size_T=input_kernel_size_T,
            kernel_size_F=input_kernel_size_F,
            eps=eps
        )

        self.filter_gen = nn.Linear(conditional_dim,emb_dim*n_freqs)
        self.bias_gen = nn.Linear(conditional_dim,emb_dim*n_freqs)

        self.ref_extractor = nn.ModuleList([
            TFGridnetBlock(
                emb_dim=emb_dim,
                kernel_size=emb_ks,
                emb_hop_size=emb_hs,
                hidden_channels=lstm_hidden_units,
                n_head=attn_n_head,
                qk_output_channel=emb_dim,
                activation=activation,
                eps=eps
            ) for _ in range(n_layers)
        ])

        self.mix_extractor = nn.ModuleList([
            WHYV3Block(
                emb_dim=emb_dim,
                kernel_size=emb_ks,
                emb_hop_size=emb_hs,
                hidden_channels=lstm_hidden_units,
                n_head=attn_n_head,
                qk_output_channel=emb_dim,
                activation=activation,
                eps=eps
            ) for _ in range(n_layers)
        ])

        self.output_decoder = TFDecoder(
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            emb_dim=emb_dim,
            kernel_size_T=output_kernel_size_T,
            kernel_size_F=output_kernel_size_F
        )

        self.n_layers = n_layers
    def forward(self,input,ref,emb):
        input_length = input.shape[-1]
        x, std = self.input_encoder(input)
        r, _ = self.input_encoder(ref)
        gtf = self.filter_gen(emb)
        gtb = self.bias_gen(emb)
        n_freqs = x.shape[-2]
        gtf = rearrange(gtf,"b (d q) -> b d q 1", q = n_freqs)
        gtb = rearrange(gtb,"b (d q) -> b d q 1", q = n_freqs)
        for i in range(self.n_layers):
            r = self.ref_extractor[i](r)
            x = self.mix_extractor[i](x,r,gtf,gtb)
        
        y = self.output_decoder(x,std,input_length)
        return y[:,0]


class WHYV3Block(nn.Module):
    def __init__(
            self,
            emb_dim=48,
            kernel_size=4,
            emb_hop_size=1,
            hidden_channels=192,
            n_head=4,
            qk_output_channel=4,
            activation="PReLU",
            eps=1.0e-5,
            n_freqs=65
    ):
        super().__init__()
        self.intra_and_inter_band_module = IntraAndInterBandModule(
                emb_dim=emb_dim,
                kernel_size=kernel_size,
                emb_hop_size=emb_hop_size,
                hidden_channels=hidden_channels,
                eps=eps
            )
        
        self.attention = WHYV3CrossAttention(
                emb_dim=emb_dim,
                n_freqs=n_freqs,
                n_head=n_head,
                qk_output_channel=qk_output_channel,
                activation=activation,
                eps=eps
            )
        
        self.gate = WHYVFilterGate(emb_dim,n_freqs)
    
    def forward(self,x,kv,gtf,gtb):
        y = self.intra_and_inter_band_module(x)
        y = self.attention(y,kv)
        y = self.gate(y,gtf,gtb)
        return y

class WHYV3CrossAttention(CrossFrameSelfAttention):
    def __init__(self, emb_dim=48, n_freqs=65, n_head=4, qk_output_channel=4, activation="PReLU", eps=0.00001):
        super().__init__(emb_dim, n_freqs, n_head, qk_output_channel, activation, eps)

    def forward(self,q,kv):
        input_q = rearrange(q,"B C Q T -> B C T Q")
        input_kv = rearrange(kv,"B C Q T -> B C T Q")
        Q = self.norm_Q(self.conv_Q(input_q)) # [B, n_head, C, T, Q]
        K = self.norm_K(self.conv_K(input_kv))
        V = self.norm_V(self.conv_V(input_kv))

        Q = rearrange(Q, "B H C T Q -> (B H) T (C Q)")
        K = rearrange(K, "B H C T Q -> (B H) (C Q) T").contiguous()
        batch, n_head, channel, _, freq = V.shape
        frame = Q.shape[1]
        V = rearrange(V, "B H C T Q -> (B H) T (C Q)")
        emb_dim = K.shape[-2]
        qkT = torch.matmul(Q, K) / (emb_dim**0.5)
        qkT = F.softmax(qkT,dim=2)
        att = torch.matmul(qkT,V)
        att = rearrange(att, "(B H) T (C Q) -> B (H C) T Q", C=channel, Q=freq, H = n_head, B = batch, T=frame)
        att = self.concat_proj(att)
        out = att + input_q
        out = rearrange(out, "B C T Q -> B C Q T")
        return out
        
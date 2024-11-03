import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from ..whyv import AllHeadPReLULayerNormalization4DC, LayerNormalization, WHYVFilterGate
from .whyv2_block import CausalIntraAndInterBandModule
from ...layers import CausalLinearAttention
from typing import Optional, Tuple
from .schema import CausalWHYV2BlockHidden

class CausalFrameLinearAttention(nn.Module):
    def __init__(
            self,
            emb_dim = 48,
            n_head=4,
            activation="PReLU",
            eps = 1e-5,
            n_freqs = 65
    ):
        super().__init__()
        assert emb_dim % n_head == 0
        E = emb_dim // n_head
        self.conv_Q = nn.Conv2d(emb_dim,emb_dim,1)
        self.norm_Q = AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps)

        self.conv_K = nn.Conv2d(emb_dim,emb_dim,1)
        self.norm_K = AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps)

        self.conv_V = nn.Conv2d(emb_dim, emb_dim, 1)
        self.norm_V = AllHeadPReLULayerNormalization4DC((n_head, emb_dim // n_head), eps=eps)

        self.attention = CausalLinearAttention(E*n_freqs, eps=eps)

        self.concat_proj = nn.Sequential(
            nn.Conv2d(emb_dim,emb_dim,1),
            getattr(nn,activation)(),
            LayerNormalization(emb_dim, dim=-3, total_dim=4, eps=eps),
        )
        self.emb_dim = emb_dim  
        self.n_head = n_head
    
    def forward(self, x, h:Optional[Tuple[torch.tensor,torch.tensor]]=None, return_hidden = False):
        input = rearrange(x,"B C Q T -> B C T Q")
        Q = self.norm_Q(self.conv_Q(input)) # [B, n_head, C, T, Q]
        K = self.norm_K(self.conv_K(input))
        V = self.norm_V(self.conv_V(input))
        Q = rearrange(Q, "B H C T Q -> (B H) T (C Q)")
        K = rearrange(K, "B H C T Q -> (B H) T (C Q)").contiguous()
        batch, n_head, channel, frame, freq = V.shape
        V = rearrange(V, "B H C T Q -> (B H) T (C Q)")

        att, new_h = self.attention(Q, K, V, h, n_chunks=16)
        att = rearrange(att, "(B H) T (C Q) -> B (H C) T Q", C=channel, Q=freq, H = n_head, B = batch, T=frame)
        att = self.concat_proj(att)
        out = att + input
        out = rearrange(out, "B C T Q -> B C Q T")
        if return_hidden:
            return out, new_h
        return out

class WHYV2version2Block(nn.Module):
    def __init__(
            self,
            emb_dim = 48,
            kernel_size:int = 4,
            emb_hop_size:int = 1,
            n_freqs = 65,
            hidden_channels:int = 192,
            n_head=4,
            activation="PReLU",
            eps = 1e-5
    ):
        super().__init__()

        self.intra_and_inter_band_module = CausalIntraAndInterBandModule(
            emb_dim=emb_dim,
            kernel_size=kernel_size,
            emb_hop_size=emb_hop_size,
            hidden_channels=hidden_channels,
            eps=eps
        )

        self.attention = CausalFrameLinearAttention(
            emb_dim=emb_dim,
            n_head=n_head,
            activation=activation,
            eps=eps,
            n_freqs=n_freqs
        )

        self.filter_gate = WHYVFilterGate(emb_dim,n_freqs)
    
    def forward(
            self, 
            x,
            gtf,
            gtb, 
            h:CausalWHYV2BlockHidden = {"lstm_h":None,"attention_h":None}, 
            return_hidden = False
            ):
        lstm_o = self.intra_and_inter_band_module(x, h["lstm_h"], return_hidden=return_hidden)
        if return_hidden:
            y_lstm, new_lstm_h = lstm_o
        else:
            y_lstm = lstm_o
            new_lstm_h = None
        
        att_o = self.attention(y_lstm, h["attention_h"], return_hidden=return_hidden)
        if return_hidden:
            y_att, new_att_h = att_o
        else:
            y_att = att_o
            new_att_h = None
        y = self.filter_gate(y_att, gtf, gtb)
        if return_hidden:
            return y, {"lstm_h":new_lstm_h, "attention_h":new_att_h}
        return y
    
__all__ = ["WHYV2version2Block","CausalFrameLinearAttention"]
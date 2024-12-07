import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..whyv import CrossFrameSelfAttention, WHYVFilterGate


class SelfLinearAttention(nn.Module):
    def __init__(
            self, in_features = 192,
            hidden_size = 192, num_head = 4,
            eps = 1e-6, feature_mapping = "1+elu",
            activation = "PReLU"
            ):
        super().__init__()
        assert hidden_size % num_head == 0, "hidden_size must be divided by num_head"
        E = hidden_size // num_head
        self.q_linear = nn.Conv2d(E, E, kernel_size=1)
        self.k_linear = nn.Conv2d(E, E, kernel_size=1)
        self.v_linear = nn.Conv2d(E, E, kernel_size=1)
        # self.q_linear = nn.Linear(in_features, hidden_size)
        # self.k_linear = nn.Linear(in_features, hidden_size)
        # self.v_linear = nn.Linear(in_features, hidden_size)
        # self.q_norm = nn.LayerNorm(hidden_size, eps=eps)
        # self.k_norm = nn.LayerNorm(hidden_size, eps=eps)
        # self.v_norm = nn.LayerNorm(hidden_size, eps=eps)
        
        if feature_mapping == "1+elu":
            self.feature_mapping = lambda x: 1 + F.elu(x)
        else:
            self.feature_mapping = getattr(F, feature_mapping)
        
        self.concat_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            getattr(nn,activation)(),
            nn.LayerNorm(hidden_size, eps=eps)
        )

        self.num_head = num_head
    def forward(self,input):
        x = rearrange(input, "b l (h d) -> b d h l", h = self.num_head)
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        q = rearrange(q, "b d h l -> (b h) l d")
        k = rearrange(k, "b d h l -> (b h) l d")
        v = rearrange(v, "b d h l -> (b h) l d")
        # q = rearrange(q, "b l (h d) -> (b h) l d", h = self.num_head)
        # k = rearrange(k, "b l (h d) -> (b h) l d", h = self.num_head)
        # v = rearrange(v, "b l (h d) -> (b h) l d", h = self.num_head)

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        v = F.normalize(v, p=2, dim=-1)

        q = self.feature_mapping(q)
        k = self.feature_mapping(k)
        kv = torch.bmm(k.transpose(1,2),v)
        # print(kv.shape)
        # kv = torch.einsum("bld,blv->bdv",k,v)
        # qk = torch.einsum("bld,bld->bl",q,k)
        attn = torch.einsum("bld,bdv->blv",q, kv) #/ (qk.unsqueeze(-1) + 1e-6)
        attn = rearrange(attn, "(b h) l d -> b l (h d)", h = self.num_head)
        attn = self.concat_projection(attn)
        return input + attn

class WHYV4Block(nn.Module):
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
        in_channels = emb_dim * kernel_size
        self.kernel_size = kernel_size
        self.emb_hs = emb_hop_size
        self.intra_norm = nn.LayerNorm(emb_dim,eps=eps)
        self.intra_attention = SelfLinearAttention(
            in_features=in_channels, 
            hidden_size=hidden_channels, 
            num_head=n_head, 
            eps=eps, 
            activation=activation
            )
        if kernel_size == emb_hop_size:
            self.intra_linear = nn.Linear(hidden_channels, in_channels)
        else:
            self.intra_linear = nn.ConvTranspose1d(hidden_channels, emb_dim,kernel_size ,emb_hop_size)
        
        self.inter_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.inter_attention = SelfLinearAttention(
            in_features=in_channels, 
            hidden_size=hidden_channels, 
            num_head=n_head, 
            eps=eps, 
            activation=activation
            )
        if kernel_size == emb_hop_size:
            self.inter_linear = nn.Linear(hidden_channels, in_channels)
        else:
            self.inter_linear = nn.ConvTranspose1d(hidden_channels, emb_dim,kernel_size ,emb_hop_size)
        
        self.cross_attn = CrossFrameSelfAttention(
                emb_dim=emb_dim,
                n_freqs=n_freqs,
                n_head=n_head,
                qk_output_channel=emb_dim//n_head,
                activation=activation,
                eps=eps
            )
        self.gate = WHYVFilterGate(emb_dim,n_freqs)

    def forward(self,x,gtf,gtb):
        B, C, old_Q, old_T = x.shape

        padding = self.kernel_size - self.emb_hs

        T = (
            math.ceil((old_T + 2 * padding - self.kernel_size) / self.emb_hs) * self.emb_hs
            + self.kernel_size
        )
        Q = (
            math.ceil((old_Q + 2 * padding - self.kernel_size) / self.emb_hs) * self.emb_hs
            + self.kernel_size
        )

        input = rearrange(x, "B C Q T -> B T Q C")
        input = F.pad(input, (0, 0, padding, Q - old_Q - padding, padding, T - old_T - padding))
        intra_attn = self.intra_norm(input)
        if self.kernel_size == self.emb_hs:
            intra_attn = intra_attn.view([B * T, -1, self.kernel_size * C])
            intra_attn = self.intra_attention(intra_attn)
            intra_attn = self.intra_linear(intra_attn)
            intra_attn = intra_attn.view([B, T, Q, C])
        else:
            intra_attn = rearrange(intra_attn,"B T Q C -> (B T) C Q")
            intra_attn = F.unfold(
                intra_attn[...,None],(self.kernel_size,1),stride=(self.emb_hs,1)
            )
            intra_attn = intra_attn.transpose(1, 2)  # [BT, -1, C*I]
            intra_attn = self.intra_attention(intra_attn)
            intra_attn = intra_attn.transpose(1, 2)  # [BT, H, -1]
            intra_attn = self.intra_linear(intra_attn)  # [BT, C, Q]
            intra_attn = intra_attn.view([B, T, C, Q])
            intra_attn = intra_attn.transpose(-2, -1)  # [B, T, Q, C]
        intra_attn = intra_attn + input
        inter_input = rearrange(intra_attn, "B T Q C -> B Q T C")
        inter_attn = self.inter_norm(inter_input)
        if self.kernel_size == self.emb_hs:
            inter_attn = inter_attn.view([B * Q, -1, self.kernel_size * C])
            inter_attn = self.inter_attention(inter_attn)
            inter_attn = self.inter_linear(inter_attn)
            inter_attn = inter_attn.view([B, Q, T, C])
        else:
            inter_attn = rearrange(inter_attn,"B Q T C -> (B Q) C T")
            inter_attn = F.unfold(
                inter_attn[...,None],(self.kernel_size,1),stride=(self.emb_hs,1)
            )
            inter_attn = inter_attn.transpose(1, 2)  # [BQ, -1, C*I]
            inter_attn = self.inter_attention(inter_attn)
            inter_attn = inter_attn.transpose(1, 2)  # [BQ, H, -1]
            inter_attn = self.inter_linear(inter_attn)  # [BQ, C, T]
            inter_attn = inter_attn.view([B, Q, C, T])
            inter_attn = inter_attn.transpose(-2, -1)  # [B, Q, T, C]
        inter_attn = inter_attn + inter_input

        inter_attn = rearrange(inter_attn,"B Q T C -> B C Q T")
        inter_attn = inter_attn[..., padding : padding + old_Q, padding : padding + old_T]

        o = self.cross_attn(inter_attn)
        o = self.gate(o,gtf,gtb)
        return o
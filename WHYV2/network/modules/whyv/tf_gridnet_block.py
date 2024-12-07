import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)

class IntraAndInterBandModule(nn.Module):
    def __init__(
            self, emb_dim:int = 48,
            kernel_size:int = 4,
            emb_hop_size:int = 1,
            hidden_channels:int = 192,
            eps = 1e-5
            ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.emb_hs = emb_hop_size
        self.kernel_size = kernel_size
        in_channels = emb_dim * kernel_size

        self.intra_norm = nn.LayerNorm(emb_dim,eps=eps)

        self.intra_lstm = nn.LSTM(
            in_channels,hidden_channels,1,batch_first=True,bidirectional=True
        )

        if kernel_size == emb_hop_size:
            self.intra_linear = nn.Linear(hidden_channels*2, in_channels)
        else:
            self.intra_linear = nn.ConvTranspose1d(hidden_channels*2, emb_dim,kernel_size ,emb_hop_size)

        self.inter_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.inter_lstm = nn.LSTM(
            in_channels,hidden_channels,1,batch_first=True,bidirectional=True
        )

        if kernel_size == emb_hop_size:
            self.inter_linear = nn.Linear(hidden_channels*2, in_channels)
        else:
            self.inter_linear = nn.ConvTranspose1d(hidden_channels*2, emb_dim,kernel_size ,emb_hop_size)
    def forward(self,x):
        """
        Args:
            input (torch.Tensor): [B C Q T]
        output:
            ouput (torch.Tensor): [B C Q T]
        """
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
        intra_rnn = self.intra_norm(input)
        if self.kernel_size == self.emb_hs:
            intra_rnn = intra_rnn.view([B * T, -1, self.kernel_size * C])
            intra_rnn, _ = self.intra_lstm(intra_rnn)
            intra_rnn = self.intra_linear(intra_rnn)
            intra_rnn = intra_rnn.view([B, T, Q, C])
        else:
            intra_rnn = rearrange(intra_rnn,"B T Q C -> (B T) C Q")
            intra_rnn = F.unfold(
                intra_rnn[...,None],(self.kernel_size,1),stride=(self.emb_hs,1)
            )
            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*I]
            intra_rnn, _ = self.intra_lstm(intra_rnn)
            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
            intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
            intra_rnn = intra_rnn.view([B, T, C, Q])
            intra_rnn = intra_rnn.transpose(-2, -1)  # [B, T, Q, C]
        intra_rnn = intra_rnn + input
        inter_input = rearrange(intra_rnn, "B T Q C -> B Q T C")
        inter_rnn = self.inter_norm(inter_input)
        if self.kernel_size == self.emb_hs:
            inter_rnn = inter_rnn.view([B * Q, -1, self.kernel_size * C])
            inter_rnn, _ = self.inter_lstm(inter_rnn)
            inter_rnn = self.inter_linear(inter_rnn)
            inter_rnn = inter_rnn.view([B, Q, T, C])
        else:
            inter_rnn = rearrange(inter_rnn,"B Q T C -> (B Q) C T")
            inter_rnn = F.unfold(
                inter_rnn[...,None],(self.kernel_size,1),stride=(self.emb_hs,1)
            )
            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, -1, C*I]
            inter_rnn,_ = self.inter_lstm(inter_rnn)
            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, H, -1]
            inter_rnn = self.inter_linear(inter_rnn)  # [BQ, C, T]
            inter_rnn = inter_rnn.view([B, Q, C, T])
            inter_rnn = inter_rnn.transpose(-2, -1)  # [B, Q, T, C]
        inter_rnn = inter_rnn + inter_input

        inter_rnn = rearrange(inter_rnn,"B Q T C -> B C Q T")
        inter_rnn = inter_rnn[..., padding : padding + old_Q, padding : padding + old_T]

        return inter_rnn

class LayerNormalization(nn.Module):
    def __init__(self, input_dim, dim=1, total_dim=4, eps=1e-5):
        super().__init__()
        self.dim = dim if dim >= 0 else total_dim + dim
        param_size = [1 if ii != self.dim else input_dim for ii in range(total_dim)]
        self.gamma = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        self.eps = eps

    @torch.amp.autocast(enabled=False, device_type="cuda")
    def forward(self, x):
        if x.ndim - 1 < self.dim:
            raise ValueError(
                f"Expect x to have {self.dim + 1} dimensions, but got {x.ndim}"
            )
        if x.dtype in HALF_PRECISION_DTYPES:
            dtype = x.dtype
            x = x.float()
        else:
            dtype = None
        mu_ = x.mean(dim=self.dim, keepdim=True)
        std_ = torch.sqrt(x.var(dim=self.dim, unbiased=False, keepdim=True) + self.eps)
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat.to(dtype=dtype) if dtype else x_hat

class AllHeadPReLULayerNormalization4DC(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2, input_dimension
        H, E = input_dimension
        param_size = [1, H, E, 1, 1]
        self.gamma = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        self.act = nn.PReLU(num_parameters=H, init=0.25)
        self.eps = eps
        self.H = H
        self.E = E

    def forward(self, x):
        assert x.ndim == 4
        B, _, T, F = x.shape
        x = x.view([B, self.H, self.E, T, F])
        x = self.act(x)  # [B,H,E,T,F]
        stat_dim = (2,)
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,H,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,H,1,T,1]
        x = ((x - mu_) / std_) * self.gamma + self.beta  # [B,H,E,T,F]
        return x

class CrossFrameSelfAttention(nn.Module):
    def __init__(
            self,
            emb_dim = 48,
            n_freqs = 65,
            n_head=4,
            qk_output_channel=4,
            activation="PReLU",
            eps = 1e-5

    ):
        super().__init__()
        assert emb_dim % n_head == 0
        E = qk_output_channel
        self.conv_Q = nn.Conv2d(emb_dim,n_head*E,1)
        self.norm_Q = AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps)

        self.conv_K = nn.Conv2d(emb_dim,n_head*E,1)
        self.norm_K = AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps)

        self.conv_V = nn.Conv2d(emb_dim, emb_dim, 1)
        self.norm_V = AllHeadPReLULayerNormalization4DC((n_head, emb_dim // n_head), eps=eps)

        self.concat_proj = nn.Sequential(
            nn.Conv2d(emb_dim,emb_dim,1),
            getattr(nn,activation)(),
            LayerNormalization(emb_dim, dim=-3, total_dim=4, eps=eps),
        )
        self.emb_dim = emb_dim  
        self.n_head = n_head
    def forward(self,x):
        """
        arg:
            x: (torch.Tensor) [B C Q T]
        output:
            output: (torch.Tensor) [B C Q T]
        """

        input = rearrange(x,"B C Q T -> B C T Q")
        Q = self.norm_Q(self.conv_Q(input)) # [B, n_head, C, T, Q]
        K = self.norm_K(self.conv_K(input))
        V = self.norm_V(self.conv_V(input))
        
        Q = rearrange(Q, "B H C T Q -> (B H) T (C Q)")
        K = rearrange(K, "B H C T Q -> (B H) (C Q) T").contiguous()
        batch, n_head, channel, frame, freq = V.shape
        V = rearrange(V, "B H C T Q -> (B H) T (C Q)")
        emb_dim = Q.shape[-1]
        qkT = torch.matmul(Q, K) / (emb_dim**0.5)
        qkT = F.softmax(qkT,dim=2)
        att = torch.matmul(qkT,V)
        att = rearrange(att, "(B H) T (C Q) -> B (H C) T Q", C=channel, Q=freq, H = n_head, B = batch, T=frame)
        att = self.concat_proj(att)
        out = att + input
        out = rearrange(out, "B C T Q -> B C Q T")
        return out

class TFGridnetBlock(nn.Module):
    def __init__(
            self,
            emb_dim = 48,
            kernel_size:int = 4,
            emb_hop_size:int = 1,
            n_freqs = 65,
            hidden_channels:int = 192,
            n_head=4,
            qk_output_channel=4,
            activation="PReLU",
            eps = 1e-5
    ):
        super().__init__()
        self.tf_grid_block = nn.Sequential(
            IntraAndInterBandModule(
                emb_dim=emb_dim,
                kernel_size=kernel_size,
                emb_hop_size=emb_hop_size,
                hidden_channels=hidden_channels,
                eps=eps
            ),
            CrossFrameSelfAttention(
                emb_dim=emb_dim,
                n_freqs=n_freqs,
                n_head=n_head,
                qk_output_channel=qk_output_channel,
                activation=activation,
                eps=eps
            )
        )
    def forward(self,input):
        return self.tf_grid_block(input)

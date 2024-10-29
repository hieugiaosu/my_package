import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from ..whyv import AllHeadPReLULayerNormalization4DC, LayerNormalization, WHYVFilterGate

class CausalIntraAndInterBandModule(nn.Module):
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
        self.hidden_channels = hidden_channels

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
            in_channels,hidden_channels,1,batch_first=True,bidirectional=False
        )

        if kernel_size == emb_hop_size:
            self.inter_linear = nn.Linear(hidden_channels, in_channels)
        else:
            self.inter_linear = nn.ConvTranspose1d(hidden_channels, emb_dim,kernel_size ,emb_hop_size)
    def forward(self,x, prev=None, return_hidden = False):
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
            intra_rnn, intra_hidden = self.intra_lstm(intra_rnn)
            intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
            intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
            intra_rnn = intra_rnn.view([B, T, C, Q])
            intra_rnn = intra_rnn.transpose(-2, -1)  # [B, T, Q, C]
        intra_rnn = intra_rnn + input
        inter_input = rearrange(intra_rnn, "B T Q C -> B Q T C")
        inter_rnn = self.inter_norm(inter_input)
        if self.kernel_size == self.emb_hs:
            inter_rnn = inter_rnn.view([B * Q, -1, self.kernel_size * C])
            inter_rnn, inter_hidden = self.inter_lstm(inter_rnn, prev)
            inter_rnn = self.inter_linear(intra_rnn)
            inter_rnn = inter_rnn.view([B, Q, T, C])
        else:
            inter_rnn = rearrange(inter_rnn,"B Q T C -> (B Q) C T")
            inter_rnn = F.unfold(
                inter_rnn[...,None],(self.kernel_size,1),stride=(self.emb_hs,1)
            )
            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, -1, C*I]
            inter_rnn, inter_hidden = self.inter_lstm(inter_rnn, prev)
            inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, H, -1]
            inter_rnn = self.inter_linear(inter_rnn)  # [BQ, C, T]
            inter_rnn = inter_rnn.view([B, Q, C, T])
            inter_rnn = inter_rnn.transpose(-2, -1)  # [B, Q, T, C]
        inter_rnn = inter_rnn + inter_input

        inter_rnn = rearrange(inter_rnn,"B Q T C -> B C Q T")
        inter_rnn = inter_rnn[..., padding : padding + old_Q, padding : padding + old_T]
        return (inter_rnn, inter_hidden) if return_hidden else inter_rnn
   
class SelectionFrameAttention(nn.Module):
    def __init__(
            self,
            emb_dim = 48,
            n_head=4,
            activation="PReLU",
            eps = 1e-5,
            n_selection_frame = 50

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


        self.concat_proj = nn.Sequential(
            nn.Conv2d(emb_dim,emb_dim,1),
            getattr(nn,activation)(),
            LayerNormalization(emb_dim, dim=-3, total_dim=4, eps=eps),
        )
        self.emb_dim = emb_dim  
        self.n_head = n_head
        self.n_selection_frame = n_selection_frame
    
    def register_reference(self, reference_frame):
        """
        Args:
            reference_frame (torch.Tensor): [B C Q T]
        """
        input = rearrange(reference_frame,"B C Q T -> B C T Q")
        Q = self.norm_Q(self.conv_Q(input)) # [B, n_head, C, T, Q]
        K = self.norm_K(self.conv_K(input))
        V = self.norm_V(self.conv_V(input))

        Q = rearrange(Q, "B H C T Q -> (B H) T (C Q)")
        K = rearrange(K, "B H C T Q -> (B H) T (C Q)").contiguous()
        batch, n_head, channel, frame, freq = V.shape
        V = rearrange(V, "B H C T Q -> (B H) T (C Q)")

        emb_dim = Q.shape[-1]
        qkT = torch.matmul(Q, K.transpose(1,2)) / (emb_dim**0.5)
        qkT = F.softmax(qkT,dim=2)
        att = torch.matmul(qkT,V)


        #############
        att = rearrange(att, "(B H) T (C Q) -> B (H C) T Q", C=channel, Q=freq, H = n_head, B = batch, T=frame)
        att = self.concat_proj(att)

        ##############
        out = att + input
        out = rearrange(out, "B C T Q -> B C Q T")
        return {"output":out, "selection_frame":out[..., -self.n_selection_frame:]}

    
    def forward(self,x,selection_frame):
        """
        arg:
            x: (torch.Tensor) [B C Q T_input]
            selection_frame: (torch.Tensor) [B C Q T_selection]
        output:
            output: (torch.Tensor) [B C Q T_input]
        """
        frame = x.shape[-1]  
        input = rearrange(x,"B C Q T -> B C T Q")
        kv = rearrange(selection_frame,"B C Q T -> B C T Q")
        Q = self.norm_Q(self.conv_Q(input)) # [B, n_head, C, T, Q]
        K = self.norm_K(self.conv_K(torch.cat([input,kv],dim=-2)))
        V = self.norm_V(self.conv_V(torch.cat([input,kv],dim=-2)))

        Q = rearrange(Q, "B H C T Q -> (B H) T (C Q)")
        K = rearrange(K, "B H C T Q -> (B H) T (C Q)").contiguous()
        batch, n_head, channel, _, freq = V.shape
        V = rearrange(V, "B H C T Q -> (B H) T (C Q)")

        emb_dim = Q.shape[-1]
        qkT = torch.matmul(Q, K.transpose(1,2)) / (emb_dim**0.5)
        qkT = F.softmax(qkT,dim=2)
        att = torch.matmul(qkT,V)


        #############
        att = rearrange(att, "(B H) T (C Q) -> B (H C) T Q", C=channel, Q=freq, H = n_head, B = batch, T=frame)
        att = self.concat_proj(att)

        ##############
        out = att + input
        out = rearrange(out, "B C T Q -> B C Q T")
        return out
        
class WHYV2block(nn.Module):
    def __init__(
            self,
            emb_dim = 48,
            kernel_size:int = 4,
            emb_hop_size:int = 1,
            n_selection_frame = 50,
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

        self.selection_frame_attention = SelectionFrameAttention(
            emb_dim=emb_dim,
            n_head=n_head,
            activation=activation,
            eps=eps,
            n_selection_frame=n_selection_frame
        )

        self.filter_gate = WHYVFilterGate(emb_dim,n_freqs)

    
    def register_reference(self, reference_frame, gtf, gtb):
        y, h = self.intra_and_inter_band_module(reference_frame, return_hidden=True)
        att_ref_output = self.selection_frame_attention.register_reference(y)
        output = self.filter_gate(att_ref_output['output'], gtf, gtb)
        return {
            "output":output,
            "selection_frame":att_ref_output["selection_frame"],
            "lstm_hidden":h
        }
        

    def forward(self,input):
        y = self.intra_and_inter_band_module(input['x'], input['lstm_hidden'], return_hidden=False)
        y = self.selection_frame_attention(y, input['selection_frame'])
        y = self.filter_gate(y, input['gtf'], input['gtb'])
        return y

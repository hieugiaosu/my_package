import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange
from ..functions.linear_attention import causal_linear_attention
import time
class CausalLinearAttention(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim

        self.phi = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ELU()
        )

        self.eps = eps
    
    def forward(
            self,
            q:torch.Tensor,
            k:torch.Tensor,
            v:torch.Tensor,
            h:Optional[Tuple[torch.Tensor,torch.Tensor]]=None
            ):
        """
        Args:
            q (torch.Tensor): query tensor [Batch, T, Dim]
            k (torch.Tensor): key tensor [Batch, T, Dim]
            v (torch.Tensor): value tensor [Batch, T, Dim_value]
            h (Tuple[torch.Tensor,torch.Tensor]): tuple of history z (with shape [B,1,Dim]) and s (with shape [B,1,D,Dim_value])
        """
        batch, frame, dim = q.shape
        _,_,dim_value = v.shape

        if h is not None:
            z_init, s_init = h
        else:
            z_init = torch.zeros(batch,1,dim,dtype=q.dtype,device=q.device)
            s_init = torch.zeros(batch,1,dim,dim_value,dtype=q.dtype, device=q.device)
        query = self.phi(q) + 1
        key = self.phi(k) + 1
        return causal_linear_attention(query,key,v,s_init,z_init,self.eps)
        # z = torch.cumsum(key, dim=1) + z_init
        # s = torch.cumsum(torch.einsum('btfi,btgi->btfg',key.unsqueeze(-1),v.unsqueeze(-1)), dim=1) + s_init

        # return torch.einsum('nli,nliv->nlv',query,s)/(torch.einsum('nli,nli->nl',query,z).unsqueeze(-1)+self.eps), tuple((
        #     z[:,-1:,...],s[:,-1:,...]
        # ))
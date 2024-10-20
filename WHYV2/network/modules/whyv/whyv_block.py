import torch.nn as nn
import torch
import torch.nn.functional as F
from .tf_gridnet_block import TFGridnetBlock

class WHYVFilterGate(nn.Module):
    def __init__(self,emb_dim=48, n_freqs = 65):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1,emb_dim,n_freqs,1).to(torch.float32))
        self.beta = nn.Parameter(torch.empty(1,emb_dim,n_freqs,1).to(torch.float32))
        nn.init.xavier_normal_(self.alpha)
        nn.init.xavier_normal_(self.beta)
    def forward(self,input,filters,bias):
        f = F.sigmoid(self.alpha*filters)
        b = F.tanh(self.beta*bias)
        return f*input + b
    
class WHYVBlock(nn.Module):
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
        self.tf_gridnet_block = TFGridnetBlock(
                    emb_dim=emb_dim,
                    kernel_size=kernel_size,
                    emb_hop_size=emb_hop_size,
                    hidden_channels=hidden_channels,
                    n_head=n_head,
                    qk_output_channel=qk_output_channel,
                    activation=activation,
                    eps=eps
                )
        self.gate = WHYVFilterGate(emb_dim,n_freqs)
    
    def forward(self,x,filter,bias):
        y = self.tf_gridnet_block(x)
        y = self.gate(y,filter,bias)
        return y
import torch
import torch.nn as nn
from typing import Iterable

class RMSNormalizeInput(nn.Module):
    def __init__(self, dim: Iterable[int], keepdim:bool = True):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
    def forward(self,input):
        std = torch.std(input,dim=self.dim,keepdim=self.keepdim)
        output = input/std
        return output, std
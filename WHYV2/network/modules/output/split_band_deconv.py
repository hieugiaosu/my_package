import torch.nn as nn
import torch
from typing import Dict, Tuple
from einops import rearrange

class SplitBandDeconv(nn.Module):
    def __init__(
            self,
            emb_dim:int = 48,
            input_n_freqs:int = 65,
            kernel_size: Tuple[int,int] = (3,3),
            padding = (1,1),
            split_points: Dict[int,int] = {
                40:1,
                60:2,
                64:5,
                65:29
            },
            eps=1.0e-5
            ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.input_n_freqs = input_n_freqs
        assert input_n_freqs == max(split_points.keys()), "input_n_freqs should be equal to the maximum key of split_points"
        self.split_points = split_points
        split_points_list = sorted(list(split_points.keys()))
        self.band_deconv = nn.ModuleDict({})
        for idx, split_point in enumerate(split_points_list):
            if idx == 0:
                self.band_deconv[str((0,split_point))] = nn.ConvTranspose2d(emb_dim,2*split_points[split_point],kernel_size=kernel_size,padding=padding)
            else:
                self.band_deconv[str((split_points_list[idx-1],split_point))] = nn.ConvTranspose2d(emb_dim,2*split_points[split_point],kernel_size=kernel_size,padding=padding)
    def forward(self,input):
        output_list = []
        for k, model in self.band_deconv.items():
            start, end = eval(k)
            x = input[:,:,start:end,:]
            y = model(x)
            y = rearrange(y,"B (C N) F T -> B C 1 (N F) T", C=2)
            output_list.append(y)
        return torch.cat(output_list,dim=-2)

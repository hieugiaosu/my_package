import torch.nn as nn
import torch
from typing import Dict, Tuple
from einops import rearrange

class SplitBandEmbedding(nn.Module):
    def __init__(
            self,
            emb_dim:int = 48,
            input_n_freqs:int = 129,
            kernel_size: Tuple[int,int] = (3,3),
            padding = "same",
            audio_channel:int = 1,
            split_points: Dict[int,int] = {
                40:1,
                80:2,
                100:5,
                129:29
            },
            eps=1.0e-5
            ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.input_n_freqs = input_n_freqs
        assert input_n_freqs == max(split_points.keys()), "input_n_freqs should be equal to the maximum key of split_points"
        self.split_points = split_points
        split_points_list = sorted(list(split_points.keys()))
        self.output_n_freqs = 0
        self.band_embedding = nn.ModuleDict({})
        for idx, split_point in enumerate(split_points_list):
            if idx == 0:
                assert (split_point - 0) % split_points[split_point] == 0, f"split_points should be divisible by the difference between the current split_point and the previous split_point get {split_point} - 0 % {split_points[split_point]}"
                self.output_n_freqs += split_point // split_points[split_point]
                self.band_embedding[str((0,split_point))] = nn.Sequential(
                    nn.Conv2d(2*audio_channel*split_points[split_point],emb_dim,kernel_size=kernel_size,padding=padding),
                    nn.GroupNorm(1,emb_dim,eps=eps)
                )
            else:
                assert (split_point - split_points_list[idx-1]) % split_points[split_point] == 0, f"split_points should be divisible by the difference between the current split_point and the previous split_point get {split_point} - {split_points_list[idx-1]} % {split_points[split_point]}"
                self.output_n_freqs += (split_points_list[idx] - split_points_list[idx-1]) // split_points[split_point]
                self.band_embedding[str((split_points_list[idx-1],split_point))] = nn.Sequential(
                    nn.Conv2d(2*audio_channel*split_points[split_point],emb_dim,kernel_size=kernel_size,padding=padding),
                    nn.GroupNorm(1,emb_dim,eps=eps)
                )
    def forward(self,input):
        output_list = []
        for k, model in self.band_embedding.items():
            start, end = eval(k)
            band_wide = self.split_points[end]
            x = input[:,:,start:end,:]
            x = rearrange(x,'B D (N F) T -> B (D N) F T',N=band_wide)
            y = model(x)
            output_list.append(y)
        return torch.cat(output_list,dim=-2)

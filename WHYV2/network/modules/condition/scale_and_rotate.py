import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TFPositionalEncoding(nn.Module):
    def __init__(self, max_freq_dim):
        super().__init__()
        # Initialize max frequency dimension if required
        self.max_freq_dim = max_freq_dim

    def forward(self, x):
        """
        Add sinusoidal positional encoding to the frequency dimension of the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, feature_dim, freq_dim, time_dim)
        
        Returns:
            torch.Tensor: Tensor with positional encodings added to the frequency dimension.
        """
        device = x.device
        batch_size, feature_dim, freq_dim, time_dim = x.shape
        # Generate sinusoidal positional encodings
        position = torch.arange(freq_dim, dtype=torch.float32, device=device).unsqueeze(1)  # Shape: (freq_dim, 1)
        div_term = torch.exp(
            torch.arange(0, feature_dim, 2, dtype=torch.float32, device=device) * 
            (-math.log(10000.0) / feature_dim)
        )  # Shape: (feature_dim // 2,)
        
        # Compute sin and cos encodings
        pos_enc = torch.zeros((freq_dim, feature_dim), device=device)  # Shape: (freq_dim, feature_dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pos_enc[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
        
        # Reshape for broadcasting and add to input tensor
        pos_enc = pos_enc.unsqueeze(0).unsqueeze(3)  # Shape: (1, feature_dim, freq_dim, 1)
        x = x + pos_enc.transpose(1,2)  # Broadcasting adds pos_enc to freq_dim across batch and time
        
        return x


class ScaleAndRotateCondition(nn.Module):
    def __init__(
            self,
            input_dim: int,
            condition_dim: int,
            kernel_size: int = (3,3),
            n_freqs: int = 65,
            ):
        super().__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim

        self.create_projection = nn.Linear(self.condition_dim, self.input_dim * 2)
        self.create_sin_rotation = nn.Sequential(
            nn.Conv2d(self.input_dim+self.condition_dim, 1, kernel_size=kernel_size, padding="same"),
            nn.Tanh()
        )

        self.create_scale = nn.Sequential(
            nn.Conv2d(self.input_dim+self.condition_dim, 1, kernel_size=kernel_size, padding="same"),
            nn.ReLU()
        )

        self.pos_encoding = TFPositionalEncoding(max_freq_dim=n_freqs)
        

    def forward(self, x, e):
        down_sampling_matrix = self.create_projection(e)
        down_sampling_matrix = rearrange(down_sampling_matrix, 'b (n d) -> b n d', n=2)
        down_sampling_matrix = F.normalize(down_sampling_matrix, p=2, dim=1)
        down_presentation = torch.einsum("bnd,bdft->bnft", down_sampling_matrix, x)


        condition = torch.cat(
            [x, e.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])],
            dim=1
        )

        ## apply positional encoding to condition
        condition = self.pos_encoding(condition)

        sin_term = self.create_sin_rotation(condition)
        scale = self.create_scale(condition)
        cos_term = torch.sqrt(1 - sin_term ** 2)
        rotation_term = torch.cat([sin_term, cos_term, cos_term, -sin_term], dim=1)
        rotation_term = rearrange(rotation_term, 'b (n c) f t -> b n c f t', n=2)


        down_presentation = torch.einsum("bndft,bdft->bnft", rotation_term, down_presentation)
        y = torch.einsum("bnd,bnft->bdft", down_sampling_matrix, down_presentation)
        y = y * scale
        return y
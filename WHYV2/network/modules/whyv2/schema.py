from dataclasses import dataclass
from typing import Tuple, Optional
import torch

@dataclass
class SelectionAttentionRefOutput:
    output: torch.Tensor
    selection_frame: torch.Tensor

@dataclass
class Whyv2BlockRefOutput:
    output: torch.Tensor
    selection_frame: torch.Tensor
    lstm_hidden: Tuple[torch.Tensor, torch.Tensor]

@dataclass
class Whyv2BlockForwardInput:
    x: torch.Tensor
    selection_frame: torch.Tensor
    lstm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]
    gtb: torch.Tensor
    gtf: torch.Tensor


__all__ = [
    "SelectionAttentionRefOutput",
    "Whyv2BlockRefOutput",
    "Whyv2BlockForwardInput"
]
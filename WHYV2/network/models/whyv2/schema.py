from ...modules.whyv2 import Whyv2BlockRefOutput
from dataclasses import dataclass
import torch
from typing import List

@dataclass
class Whyv2Reference:
    gtf: torch.Tensor
    gtb: torch.Tensor
    whyv2_block_ref: List[Whyv2BlockRefOutput]
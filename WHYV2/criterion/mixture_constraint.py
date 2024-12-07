import torch.nn as nn
import torch

class Mixture_constraint_loss(nn.Module):
    def __init__(self,reduction="mean"):
        super().__init__()
        if reduction == "mean":
            self.loss_reduction = lambda loss: loss.mean(0)
        elif reduction == "sum":
            self.loss_reduction = lambda loss: loss.sum(0)
        else:
            self.loss_reduction = lambda loss: loss
    #@torch.amp.custom_fwd(cast_inputs=torch.float32,device_type='cuda')
    def forward(self,input,label):
        input = input.float()
        label = label.float()
        N = input.shape[0]
        alpha = torch.bmm(input.unsqueeze(1),label.unsqueeze(-1))/torch.clamp(torch.bmm(input.unsqueeze(1),input.unsqueeze(-1)),min=torch.tensor(1e-7,dtype=input.dtype,device=input.device))
        alpha = alpha.view(N,1)
        return self.loss_reduction(torch.norm(alpha*input-label,1,dim=-1))
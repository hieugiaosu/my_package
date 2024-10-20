from torch.nn import Module

class RMSDenormalizeOutput(Module):
    def __init__(self):
        super().__init__()
    def forward(self,input,std):
        return input*std
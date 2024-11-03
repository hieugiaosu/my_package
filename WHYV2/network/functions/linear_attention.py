import torch

class CausalLinearAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(query,key,value,s_init,z_init,eps):
        z = torch.cumsum(key, dim=1) + z_init
        s = torch.cumsum(torch.einsum('btf,btg->btfg',key,value), dim=1) + s_init

        return torch.einsum('nli,nliv->nlv',query,s)/(torch.einsum('nli,nli->nl',query,z).unsqueeze(-1)+eps), tuple((
            z[:,-1:,...],s[:,-1:,...]
        ))

    @staticmethod
    def setup_context(ctx, inputs, output):
        query,key,value,s_init,z_init,eps = inputs
        ctx.save_for_backward(query,key,value,s_init)
    
    @staticmethod
    def backward(ctx, grad_output, *args):
        """
        ctx: (query,key,value,s_init)
            -query, key, value: (B, T, Dim)
            -s_init: (B, 1, Dim, Dim_value)
        grad_output: (B, T, Dim_value)
        """
        query,key,value,s_init = ctx.saved_tensors
        s = torch.cumsum(torch.einsum('btf,btg->btfg',key,value), dim=1) + s_init
        dq = torch.einsum('btv,btdv->btd',grad_output,s)
        s_kv = torch.cumsum(torch.einsum('btd,btv->btdv',query,grad_output), dim=1)
        dv = torch.einsum('btdv,btd->btv',s_kv,key)
        dk = torch.einsum('btdv,btv->btv',s_kv,value)
        return dq, dk, dv, None, None, None

def causal_linear_attention(query,key,value,s_init,z_init,eps):
    return CausalLinearAttentionFunction.apply(query,key,value,s_init,z_init,eps)

__all__ = [
    "causal_linear_attention"
]
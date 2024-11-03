import torch
import math

class CausalLinearAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(query,key,value,s_init,z_init,eps, n_chunks=1):
        num_frames = query.shape[1]
        chunk_size = num_frames // n_chunks
        n_chunks = math.ceil(num_frames / chunk_size)
        z = torch.cumsum(key, dim=1) + z_init
        prev_s = s_init
        v_list = []
        for chunk in range(n_chunks):
            from_idx = chunk*chunk_size
            to_idx = min((chunk+1)*chunk_size, num_frames)
            s = torch.cumsum(
                torch.einsum(
                    'btf,btg->btfg',
                    key[:,from_idx:to_idx,:],value[:,from_idx:to_idx,:]
                    ), 
                dim=1
                ) + prev_s
            prev_s = s[:,-1:,...]
            v = torch.einsum(
                'nli,nliv->nlv',query[:,from_idx:to_idx,:],s
                )/(torch.einsum(
                    'nli,nli->nl',query[:,from_idx:to_idx,:],z[:,from_idx:to_idx,:]
                    ).unsqueeze(-1)+eps)
            v_list.append(v)
        return torch.cat(v_list, dim=1), tuple((z[:,-1:,...],prev_s))
        # z = torch.cumsum(key, dim=1) + z_init
        # s = torch.cumsum(torch.einsum('btf,btg->btfg',key,value), dim=1) + s_init

        # return torch.einsum('nli,nliv->nlv',query,s)/(torch.einsum('nli,nli->nl',query,z).unsqueeze(-1)+eps), tuple((
        #     z[:,-1:,...],s[:,-1:,...]
        # ))

    @staticmethod
    def setup_context(ctx, inputs, output):
        query,key,value,s_init,z_init,eps, n_chunks = inputs
        chunk_size = query.shape[1] // n_chunks
        ctx.save_for_backward(query,key,value,s_init, torch.tensor(chunk_size,device='cpu'))
    
    @staticmethod
    def backward(ctx, grad_output, *args):
        """
        ctx: (query,key,value,s_init)
            -query, key, value: (B, T, Dim)
            -s_init: (B, 1, Dim, Dim_value)
        grad_output: (B, T, Dim_value)
        """
        query,key,value,prev_s, chunk_size = ctx.saved_tensors
        n_chunks = math.ceil(query.shape[1] / chunk_size.item())
        dq_list = list([])
        dk_list = list([])
        dv_list = list([])
        for chunk in range(n_chunks):
            from_idx = chunk*chunk_size
            to_idx = (chunk+1)*chunk_size
            s = torch.cumsum(
                torch.einsum(
                    'btf,btg->btfg',
                    key[:,from_idx:to_idx,:],value[:,from_idx:to_idx,:]
                    ), 
                dim=1
                ) + prev_s
            prev_s = s[:,-1:,...]
            dq = torch.einsum(
                'btv,btdv->btd',grad_output[:,from_idx:to_idx,:],s
                )
            dq_list.append(dq)
        dq = torch.cat(dq_list, dim=1)

        prev_s_kv = 0
        for chunk in range(n_chunks):
            from_idx = chunk*chunk_size
            to_idx = (chunk+1)*chunk_size
            s_kv = torch.cumsum(
                torch.einsum(
                    'btd,btv->btdv',query[:,from_idx:to_idx,:],grad_output[:,from_idx:to_idx,:]
                    ), 
                dim=1
                ) + prev_s_kv
            prev_s_kv = s_kv[:,-1:,...]
            dv = torch.einsum(
                'btdv,btd->btv',s_kv,key[:,from_idx:to_idx,:]
                )
            dk = torch.einsum(
                'btdv,btv->btv',s_kv,value[:,from_idx:to_idx,:]
                )
            dv_list.append(dv)
            dk_list.append(dk)
        dv = torch.cat(dv_list, dim=1)
        dk = torch.cat(dk_list, dim=1)
        # s = torch.cumsum(torch.einsum('btf,btg->btfg',key,value), dim=1) + prev_s
        # dq = torch.einsum('btv,btdv->btd',grad_output,s)
        # s_kv = torch.cumsum(torch.einsum('btd,btv->btdv',query,grad_output), dim=1)
        # dv = torch.einsum('btdv,btd->btv',s_kv,key)
        # dk = torch.einsum('btdv,btv->btv',s_kv,value)
        return dq, dk, dv, None, None, None, None

def causal_linear_attention(query,key,value,s_init,z_init,eps,n_chunks=1):
    return CausalLinearAttentionFunction.apply(query,key,value,s_init,z_init,eps,n_chunks)

__all__ = [
    "causal_linear_attention"
]
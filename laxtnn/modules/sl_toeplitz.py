# Differentiable Causal Sparse+Low-Rank Toeplitz Function
# The Sparse component will be handled by a 1D convolution
# The Low-rank component will be handled by fast causal masking
# We peform the fw pass via the S+L representation
#  (already faster than FFT at seq len 512)
#  while performing the bw pass via the FFT
#  (faster than S+L at seq len 512 but maybe slower at 10k)

import torch
import torch.nn.functional as F

from fast_transformers.causal_product import causal_dot_product

@torch.no_grad()
def apply_causal_sl(U, V, s, x):
    # x (b, n, hd)
    # U (..., n, r)
    # V (..., n, r)
    #out (b, n, hd)

    b = x.shape[0]
    hd, nk = s.shape
    nk2 = nk - 1
    xt = x.transpose(1,2)
    U, V, = [
        mat_[None].expand(b, -1, -1, -1)  # 'hd n 2(r+1) -> b hd n 2(r+1)'
        for mat_ in [U, V]
    ]
    out = (
        causal_dot_product(U, V, xt[..., None].contiguous())[..., 0] + 
        F.conv1d(xt, s[:, None], padding=nk2, groups=hd)[..., :-nk2]
    )
    return out.transpose(1,2)

class CausalSLToeplitz(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, V, c, s, x):
        """
        U and V should have `requires_grad` == False
        s is the sparse convolution kernel
        L = U V^T is the low-rank component (used for fw pass)
        c is the first column of L (redundant, but used in bw pass during training)
        U,V (hd, n, r)
        c (hd, n)
        s (hd, n_k)
        x (b, n, hd): the vectors to which the Toeplitz matrices are being applied
        """
        hd, nk = s.shape
        nk2 = nk - 1
        out = apply_causal_sl(U, V, s, x)
        ctx.save_for_backward(U, V, s, x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        U, V, s, x = ctx.saved_tensors
        n = x.shape[-2]
        n2 = 2 * n
        hd, nk = s.shape
        
        dy_reversed = grad_output.flip(-2)
        # we can also do this using fft/ifft at 4n*log(2n)
        grad_x_rev = apply_causal_sl(U, V, s, dy_reversed)
        grad_x = grad_x_rev.flip(-2)
        
        x_pad = torch.cat([x, x[..., :1,:]], dim=-2)
        x_f = torch.fft.rfft(x_pad, n2, dim=-2)  # since we didn't do this during fw pass
        out_f = torch.fft.rfft(dy_reversed, n2, dim=-2)
        grad_c_rev = torch.fft.irfft(out_f * x_f, n2, dim=-2)[..., :n, :]
        # reduce over batch dim for grad_c and grad_s
        grad_c_rev = grad_c_rev.sum(dim=0)
        grad_c = grad_c_rev.flip(-2).transpose(-2, -1)
        grad_s = grad_c[..., :nk]
        return None, None, grad_c, grad_s, grad_x

causal_sl_toeplitz = CausalSLToeplitz.apply
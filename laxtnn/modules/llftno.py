# Learned Laplace Features Toeplitz Neural Operator
# The Sparse component will be handled by a 1D convolution in parent class Sltno
# The Low-rank component will be handled by Learned exponential weighted sinusoids
import math

import torch
import torch.nn as nn
from fast_transformers.causal_product import causal_dot_product

# from custom fairseq
from fairseq.modules.helpers import get_activation_fn, print_params

from .sltno import Sltno
from ..utils.causal_product import causal_product_trio

PI = torch.Tensor([math.pi])
PI2 = 2 * PI

relu = nn.ReLU()

def gen_block_ixes(r):
    """
    indices for corresponding entries of diagonal 2x2 blocks in last 2 dims.
    
    Example:
    in: r = 3
        B = torch.zeros((1,2*r,2*r))
        ixs = gen_block_ixes(r)
        B[ixs[0]] = torch.Tensor([1,5,9])
        B[ixs[1]] = torch.Tensor([2,6,10])
        B[ixs[2]] = torch.Tensor([3,7,11])
        B[ixs[3]] = torch.Tensor([4,8,12])
        print(B)
    out:tensor([[[ 1.,  3.,  0.,  0.,  0.,  0.],
                 [ 2.,  4.,  0.,  0.,  0.,  0.],
                 [ 0.,  0.,  5.,  7.,  0.,  0.],
                 [ 0.,  0.,  6.,  8.,  0.,  0.],
                 [ 0.,  0.,  0.,  0.,  9., 11.],
                 [ 0.,  0.,  0.,  0., 10., 12.]]])
        
    """
    out0 = torch.arange(0, 2*r, 2, dtype=torch.long)
    out1 = out0+1
    out = ((...,out0,out0), (...,out1,out0), (...,out0,out1), (...,out1,out1))
    return out

class Llftno(Sltno):
    """
    Learned Laplace features for low-rank component
    """
    def __init__(self, h, dim, r, nk, **kwargs):
        super().__init__(h, dim, r, nk, **kwargs)
        hd = h * dim
        # alpha includes extra param for const offset:
        self.alphas = nn.Parameter(torch.randn((hd, r+1)))  # [-inf, inf]
        self.phis = nn.Parameter(torch.rand((hd, r)) * PI)  # [0, pi]
        # fixed grid, not params:
        self.register_buffer('omegas', torch.linspace(1, r, r) * PI)
        # falloff
        if self.causal:
            self.falloff_lambda = nn.Parameter(torch.rand(1))  # [0, inf)
        else:
            self.falloff_omega = nn.Parameter(PI)  # [0, pi)

    def apply_low_rank(self, x, **unused_kwargs):
        """
        non-causal

        x `torch.Tensor` (b, n, hd): sequences of tokens
        have access to self.r for the rank
        """
        _, n, _ = x.shape
        t = self.gen_pos(n, absolute=True, device=x.device)
        U, S = self.gen_low_rank_US(t)  # (hd, n or r, r)

        # falloff
        if not self.causal:
            omega_ft = torch.clamp(self.falloff_omega, 0, PI.item()) * t[..., 0]  # (n, )
            Fc = torch.cos(omega_ft)
            Fs = torch.sin(omega_ft)  # (n, )

        x = x.transpose(1, 2)  # (b, hd, n)

        if self.causal:
            out = causal_product_trio(U, S, U, x)  # (b hd n)
        else:
            VT = U.transpose(-1, -2)  #  hd, 2r+1, n
            # out = (
            #     (U @ (S @ (VT @ x)))
            #     + Fc * (U @ (S @ (VT @ (Fc * x))))
            #     + Fs * (U @ (S @ (VT @ (Fs * x))))
            # ) / 2
            out = (U[None] @ (S[None] @ (VT[None] @ x[..., None]))).squeeze(-1)  # b, hd, n

        out = out.transpose(1, 2)  # (b n hd)
        return out

    def gen_low_rank_US(self, t):
        n = t.shape[0]
        hd = self.h * self.dim
        r = self.r

        # padded with 0's to represent constant offset (cheaper than cat and deterministic unlike F.pad)
        omega_t = torch.zeros((hd, n, r+1), device=t.device)
        omega_t[..., :r] = self.omegas * t  # (hd, n, r+1)
        # sin(x) = cos(x+3pi/2)
        omega_t = torch.Tensor([0, 1.5*PI.item()]).to(omega_t) + omega_t[..., None]   # (hd, n, r+1, 2)
        # interleave cos/sin, drop the 0's vector:
        U = torch.cos(omega_t).reshape((hd, n, 2*(r+1)))[..., :-1]  # (hd, n, 2r+1)
        S = torch.zeros((hd, 2*r+1, 2*r+1), device=self.phis.device)
        ixs = gen_block_ixes(r)
        block03 = self.alphas[..., :-1] * torch.cos(self.phis)
        block12 = self.alphas[..., :-1] * torch.sin(self.phis)
        S[ixs[0]] = block03
        S[ixs[1]] =  -block12
        S[ixs[2]] =  block12
        S[ixs[3]] =  block03
        S[..., -1, -1] = self.alphas[..., -1]
        return U, S
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

PI = torch.Tensor([math.pi])
PI2 = 2 * PI

relu = nn.ReLU()

class Llftno(Sltno):
    """
    Learned Laplace features for low-rank component
    """
    def __init__(self, h, dim, r, nk, **kwargs):
        super().__init__(h, dim, r, nk, **kwargs)
        hd = h * dim
        # alpha includes extra param for const offset:
        self.alphas = nn.Parameter(torch.randn((hd, 1, r+1, 1)))  # [-inf, inf]
        self.omegas = nn.Parameter(torch.rand((hd, 1, r)) * 8 * PI2)  # [0, inf)
        self.phis = nn.Parameter(torch.rand((hd, 1, r, 1)) * PI)  # [0, pi]
        self.laplace = kwargs.get('laplace', False)
        if self.laplace:
            self.lambdas = nn.Parameter(torch.rand((hd, 1, r)))  # [0, inf)
        # falloff
        if self.causal:
            self.falloff_lambda = nn.Parameter(torch.rand(1))  # [0, inf)
        else:
            self.falloff_omega = nn.Parameter(PI2)  # [0, 2pi)

    def apply_low_rank(self, x, t, causal=False, **unused_kwargs):
        """
        x `torch.Tensor` (b, n, hd): sequences of tokens
        t `torch.Tensor` (n, 1): relative position indices
        have access to self.r for the rank
        """
        b, n, hd = x.shape
        r = self.r
        U, V = self.gen_low_rank_UV(t)
        # falloff
        if not self.causal:
            omega_ft = (PI2.item() + relu(self.falloff_omega)) * t  # (1, 1, n, 1)
            Fc = torch.cos(omega_ft)
            Fs = torch.sin(omega_ft)  # (1, 1, n, 1)

        x = x.transpose(1, 2)  # (b, hd, n)
        x = x[..., None]  # 'b hd n 1'
        if self.causal:
            U, V, = [
                mat_[None].expand(b, -1, -1, -1)  # 'hd n 2(r+1) -> b hd n 2(r+1)'
                for mat_ in [U, V]
            ]
            x = x.contiguous()
            out = causal_dot_product(U, V, x)  # (b hd n 1)
        else:
            VT = V.transpose(1, 2)  #  hd, 2(r+1), n
            out = Fc * (U @ (VT @ (Fc * x))) + Fs * (U @ (VT @ (Fs * x)))

        out = out[..., 0].transpose(1, 2)  # (b n hd)
        return out

    def gen_causal_low_rank_c(self, t):
        cos = self.alphas[..., :-1, :] * torch.cos(self.omegas[..., None] * t[..., None] + 2 * self.phis)
        if self.laplace:
            cos *= torch.exp(-t[..., None] * relu(self.lambdas[..., None]))
        c = torch.exp(-t * relu(self.falloff_lambda)) * (
            self.alphas[..., -1, :] + torch.sum(cos, dim=-2)
        )  # (hd, n, 1)
        return c[..., 0]

    def gen_low_rank_UV(self, t):
        n = t.shape[0]
        hd = self.h * self.dim
        r = self.r
        # padded with 0's to represent constant offset (cheaper than cat and deterministic unlike F.pad)
        omega_t = torch.zeros((hd, n, r+1), device=self.omegas.device)
        omega_t[..., :r] = self.omegas * t  # (hd, n, r+1)
        # sin(x) = cos(x+3pi/2)
        omega_t = torch.Tensor([0, 1.5*PI.item()]).to(omega_t) + omega_t[..., None]   # (hd, n, r+1, 2)
        # padded with 0's to represent constant offset
        phi_pad = torch.zeros((hd, 1, r+1, 1), device=self.phis.device)
        phi_pad[..., :r, :] = self.phis  # (hd, 1, r+1, 1)
        otp = omega_t + phi_pad
        otm = omega_t - phi_pad  # (hd, n, r+1, 2)
        U = torch.cos(otp)
        # preapply alphas
        V = torch.cos(otm) * self.alphas  # (hd, n, r+1, 2)
        
        if self.laplace:
            lambdas_clamp = torch.zeros((hd, 1, r+1), device=self.lambdas.device)  # hd, 1, r+1
            lambdas_clamp[..., :r] = relu(self.lambdas)  # hd, 1, r+1
            exponential = torch.exp(-lambdas_clamp * t)  # hd, n, r+1
            exponential = exponential[..., None]  # hd, n, r+1, 1
            U *= exponential
            exponential = torch.exp(lambdas_clamp * t)
            V *= exponential
        U = U.reshape([hd, n, -1])
        V = V.reshape([hd, n, -1])  # (hd, n, 2(r+1))
        if self.causal:
            lambda_clamp = relu(self.falloff_lambda)  # (1, )
            exponentialn = torch.exp(-lambda_clamp * t)  # n, 1
            U *= exponentialn
            exponentialp = torch.exp(lambda_clamp * t)  # n, 1
            V *= exponentialp
        return U, V
    
    @torch.no_grad()
    def gen_causal_low_rank_UV(self, t):
        U, V = self.gen_low_rank_UV(t)
        U, V = U.detach(), V.detach()
        return U, V

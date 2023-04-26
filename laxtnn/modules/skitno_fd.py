import math 

import torch
from torch import nn

# from custom fairseq
from fairseq.modules.helpers import get_activation_fn

from .sltno import Sltno
from ..utils.toep_mat import ToepMat

PI = torch.Tensor([math.pi])
PI2 = 2 * PI

class SKITnoFD(Sltno):
    def __init__(self, h, dim, r, nk, **kwargs):
        """
        SKI using LLF instead of RPE
        r: num inducing locations
        """
        super().__init__(h, dim, r, nk, **kwargs)
        hd = h*dim
        # real part of fft
        self.alphas = nn.Parameter(torch.randn((hd, r)))  # [-inf, inf]
        if not self.causal:
            # imaginary part of fft
            self.betas = nn.Parameter(torch.randn((hd, r)))  # [-inf, inf]
        self.register_buffer('inducing', torch.linspace(0, 1, r))
    
    def rpe(self, n, fd=True):
        """
        get interpolated coeffs for RPE in either frequency or time domain
        """
        # fake batch dim
        reals = torch.interpolate(self.alphas[None], (n,), mode='linear', align_corners=True)[0]
        if self.causal:
            irfft_real_part = torch.fft.irfft(reals, dim=1)
            irfft_real_part[:, n:] = 0  # make it causal
            irfft_real_part[:, 0] *= 0.5
            if fd:
                out = 2 * torch.fft.rfft(irfft_real_part, dim=1)
            else:
                out = 2 * irfft_real_part
        else:
            imags = torch.interpolate(self.betas[None], (n,), mode='linear', align_corners=True)[0]
            imags[:, (0, -1)] = 0  # force DC and Nyquist to be real
            out = torch.complex(reals, imags)
            if not fd:
                out = torch.rfft.irfft(out)
        return out

    def gen_low_rank_U(self, t):
        n, _ = t.shape
        #original locations
        falloff = self.gamma ** (t * n)  # (n, 1)
        original_loc = t[..., 0]  # (n, )

        # get largest point in the inducing < the original location
        # TODO: closed form since it's all uniform? (not super important)
        indices = torch.searchsorted(self.inducing*(n-1), original_loc)
        # should be unnecessary with side='left' above as default, but *shrug*
        indices.clamp_(min=0, max=self.inducing.shape[0]-2)

        #lerp weights: n
        weights_l = (original_loc - self.inducing[indices])*(self.r-1)/(n-1)
        weights_u = 1 - weights_l

        # TODO: init as sparse (not super important, small matrix)
        W = torch.zeros((n, self.r)).to(t)  # (n, r)
        W[torch.arange(W.size(0)), indices] = weights_l
        W[torch.arange(W.size(0)), indices+1] = weights_u
        W = falloff * W
        return W

    def apply_low_rank(self, x):
        """
        non-causal
        """
        b, n, hd = x.shape  # hd = num_heads*num_dims
        t = self.get_pos(n, absolute=True, device=x.device)

        W = self.gen_low_rank_U(t)

        # build inducing kernel matrix
        positions = self.inducing[1:][..., None]  # (r-1, 1) 
        zero = self.rpe(self.inducing[:1][..., None])  # (1, hd)
        pos = self.rpe(positions)  # (r-1, hd)
        # following needs flip to match circulant structure!
        neg = self.rpe(-positions.flip(0))  # (r-1, hd)
        a = torch.cat([zero, pos, zero, neg], dim=0)  # (2r, hd)
        a = self.act_fun(a).transpose(0,1)  # (hd, 2r)
        S = ToepMat(a, self.r) 

        # this one is still slow
        x = x.transpose(0, 1).reshape((n, b*hd))  # (n, b*hd)
        Wt = W.T
        W = W.to_sparse()
        Wt = Wt.to_sparse()  # (r, n)
        # these `view` and `permute` required to use ToepMat aren't too expensive
        output = torch.sparse.mm(Wt, x).view((self.r, b, hd)).permute((1,2,0))[..., None]  # (b, hd, r, 1)
        output = (S @ output)[..., 0].permute((2,0,1)).view((self.r, b*hd))  # (r, b*hd)
        output = torch.sparse.mm(W, output)  # (n, b*hd)
        output = output.view((n, b, hd)).transpose(0, 1)  # (b, n, hd)
        return output

    def apply_toeplitz(self, x):
        b, n, hd = x.shape  # hd = num_heads*num_dims
        # build inducing kernel matrix
        positions = self.inducing[..., None]*(n-1)  # (r, 1) 
        pos = self.rpe(positions)  # (r, hd)
        # apply interpolation (need fake batch dim)
        pos = pos[None].transpose(-1,-2) # (1, hd, r)
        pos = torch.nn.functional.interpolate(pos, (n,), mode='linear', align_corners=True)[0]  # (hd, n)

        a = self.act_fun(pos)  # (hd, n)
        a = a * (self.gamma ** torch.arange(n, device=x.device))
        a[:, :self.nk2+1] += self.conv_kernel
        S = ToepMat(a, n)
        x = x.transpose(-1, -2)[..., None]  # (b, hd, n, 1)
        out = (S @ x)[..., 0]  # (b, hd, n)
        return out.transpose(-1, -2)
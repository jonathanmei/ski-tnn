import torch
from torch import nn
from torch.nn import functional as F

from .sltno import Sltno
from ..utils.toep_mat import ToepMat

def sign0(x):
    """
    sign function that treats sign(x)==1 instead of sign(x)==0
    """
    return torch.sign(x) + (x==0).int()

class SKITnoInvTime(Sltno):
    def __init__(self, h, dim, r, nk, **kwargs):
        """
        r: num inducing locations
        """
        super().__init__(h, dim, r, nk, **kwargs)
        hd = h*dim
        # explicit interpolation grid to represent the interval [-1, 1]
        self.alphas = nn.Parameter(torch.randn((1, hd, 1, r)))  # [-inf, inf]
        # negative
        if not self.causal:
            self.betas = nn.Parameter(torch.randn((1, hd, 1, r)))  # [-inf, inf]

    def rpe(self, t):
        """
        t: (r, 1) values in in interval [1, infty)
        """
        t = t.transpose(-2, -1)[None]  # (1, 1, r)  fake batch and height dims
        # doesn't do much to cache this computation, it's not the bottleneck:
        t = sign0(t) * (self.gamma ** torch.abs(t))
        if self.causal:
            t = 2*t - 1 # shift [0,1] to normalized [-1,1]
            vals = F.pad(self.alphas, (1, 0), value=0)
        else:
            shape = self.betas.shape[:-1] + (2*self.r + 1, )
            vals = torch.zeros(shape, device=self.betas.device)
            vals[..., :self.r] = self.betas
            vals[..., -self.r:] = self.alphas
        t = F.pad(t[..., None], (0, 1), value=0)  # (1, 1, r, 2) == (n, h, w, 2)
        # bottleneck
        interpd = F.grid_sample(vals, t, mode='bilinear', align_corners=True)  # (1, hd, 1, r)
        return interpd[0, :, 0].transpose(-2,-1)  # (r, hd)

    def gen_low_rank_U(self, t):
        """
        t: vector [0, ..., n-1]
        """
        n, _ = t.shape
        #original locations
        falloff = self.gamma ** (t * n)  # (n, 1)
        original_loc = t[..., 0]  # (n, )

        inducing = torch.arange(self.r, device=t.device) * (n-1) / (self.r-1)
        # get largest point in the inducing < the original location
        # TODO: closed form since it's all uniform? (not super important)
        indices = torch.searchsorted(inducing, original_loc)
        # should be unnecessary with side='left' above as default, but *shrug*
        indices.clamp_(min=0, max=inducing.shape[0]-2)

        #lerp weights: n
        weights_l = (original_loc - inducing[indices])*(self.r-1)/(n-1)
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
        positions = torch.arange(self.r, device=x.device) * (n-1) / (self.r-1)
        positions = positions[:, None]  # (r, 1)
        nonneg = self.rpe(positions)  # (r, hd)
        zero = nonneg[:1]  # (1, hd)
        # following needs flip to match circulant structure!
        neg = self.rpe(-positions[1:].flip(0))  # (r-1, hd)
        a = torch.cat([nonneg, zero, neg], dim=0)  # (2r, hd)
        a = a.transpose(0,1)  # (hd, 2r)
        S = ToepMat(a, self.r) 

        output = W @ (S @ ( W.T @ x.transpose(-1, -2)[..., None]))[..., 0].transpose(-1, -2)
        return output

    def apply_toeplitz(self, x):
        b, n, hd = x.shape  # hd = num_heads*num_dims
        # two interpolations
        # build inducing kernel matrix
        positions = torch.arange(self.r, device=x.device) * (n-1) / (self.r-1)
        # nonneg only (causal)
        nonneg = self.rpe(positions[:, None])  # (r, hd)
        # apply interpolation (need fake batch dim)
        nonneg = nonneg[None].transpose(-1,-2) # (1, hd, r)
        a = F.interpolate(nonneg, (n,), mode='linear', align_corners=True)[0]  # (hd, n)

        # positions = torch.arange(n, device=x.device)
        # pos = self.rpe(positions[:, None])  # (n, hd)
        # a = pos.transpose(-1, -2)  # (hd, n)

        a[:, :self.nk2+1] += self.conv_kernel
        S = ToepMat(a, n)
        x = x.transpose(-1, -2)[..., None]  # (b, hd, n, 1)
        out = (S @ x)[..., 0]  # (b, hd, n)
        return out.transpose(-1, -2)
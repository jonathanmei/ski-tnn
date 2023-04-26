import torch
from torch import nn

# from custom fairseq

from .rpe import Rpe
from .sltno import Sltno
from ..utils.toep_mat import ToepMat

class SKITnoInvTime(Sltno):
    def __init__(self, h, dim, r, nk, **kwargs):
        """
        r: num inducing locations
        """
        super().__init__(h, dim, r, nk, **kwargs)
        hd = h*dim
        # explicit interpolation grid to represent the interval [-1, 1]
        self.zeros = nn.Parameter(torch.randn((1, hd)))  # [-inf, inf]
        if not self.causal:
            if r % 2 != 0:  # make r odd so we can force the center to be 0
                r += 1
                self.r = r
        self.alphas = nn.Parameter(torch.randn((1, hd, 1, r)))  # [-inf, inf]

    def rpe(self, t):
        """
        t: (r, 1) values in in interval [1, infty)
        """
        grid = self.alphas
        t = t.transpose(-2, -1)[None]  # (1, 1, r)  fake batch and height dims
        t = torch.sign(t) * (self.gamma ** torch.abs(t))
        if self.causal:
            t = 2*t - 1 # shift [0,1] to normalized [-1,1]
            grid[..., 0] = 0
        else:
            # grid = self.alphas  # (1, hd, 1, r) == (n, c, h, w)
            grid[..., self.r // 2] = 0
        t = torch.stack([t, torch.zeros_like(t)], dim=-1)  # (1, 1, r, 2) == (n, h, w, 2)
        return nn.functional.grid_sample(t, grid, mode='linear', align_corners=True)[0, :, 0].transpose(-2,-1)  # (r, hd)

    def gen_low_rank_U(self, t):
        """
        t: vector [0, ..., n-1]
        """
        n, _ = t.shape
        #original locations
        falloff = self.gamma ** (t * n)  # (n, 1)
        original_loc = t[..., 0]  # (n, )

        inducing = torch.arange(self.r) * (n-1) / (self.r-1)
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
        positions = torch.arange(1, self.r) * (n-1) / (self.r-1)
        positions = positions[:, None]  # (r-1, 1)
        zero = self.zeros  # (1, hd)
        pos = self.rpe(positions)  # (r-1, hd)
        # following needs flip to match circulant structure!
        neg = self.rpe(-positions.flip(0))  # (r-1, hd)
        a = torch.cat([zero, pos, zero, neg], dim=0)  # (2r, hd)
        a = self.a.transpose(0,1)  # (hd, 2r)
        S = ToepMat(a, self.r) 

        # this one is still slow
        # x = x.transpose(0, 1).reshape((n, b*hd))  # (n, b*hd)
        # Wt = W.T
        # W = W.to_sparse()
        # Wt = Wt.to_sparse()  # (r, n)
        # # these `view` and `permute` required to use ToepMat aren't too expensive
        # output = torch.sparse.mm(Wt, x).view((self.r, b, hd)).permute((1,2,0))[..., None]  # (b, hd, r, 1)
        # output = (S @ output)[..., 0].permute((2,0,1)).view((self.r, b*hd))  # (r, b*hd)
        # output = torch.sparse.mm(W, output)  # (n, b*hd)
        # output = output.view((n, b, hd)).transpose(0, 1)  # (b, n, hd)
        output = W @ (S @ ( W.T @ x.transpose(-1, -2)[..., None]))
        return output

    def apply_toeplitz(self, x):
        b, n, hd = x.shape  # hd = num_heads*num_dims
        ### two interpolations... why?
        # # build inducing kernel matrix
        # positions = torch.arange(1, self.r) * (n-1) / (self.r-1)
        # # positive only (causal)
        # pos = self.rpe(positions[:, None])  # (r-1, hd)
        # pos_r = torch.cat([self.zeros, pos], dim=0)  # (r, hd)
        # # apply interpolation (need fake batch dim)
        # pos_r = pos_r[None].transpose(-1,-2) # (1, hd, r)
        # a = torch.nn.functional.interpolate(pos_r, (n,), mode='linear', align_corners=True)[0]  # (hd, n)

        # directly get all positions from RPE
        positions = torch.arange(1, n)
        pos = self.rpe(positions[:, None])  # (r-1, hd)
        a = torch.cat([self.zeros, pos], dim=0)  # (r, hd)

        a[:, :self.nk2+1] += self.conv_kernel
        S = ToepMat(a, n)
        x = x.transpose(-1, -2)[..., None]  # (b, hd, n, 1)
        out = (S @ x)[..., 0]  # (b, hd, n)
        return out.transpose(-1, -2)
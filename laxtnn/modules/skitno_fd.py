import torch
from torch import nn

# from custom fairseq
from fairseq.modules.helpers import get_activation_fn

from .sltno import Sltno
from ..utils.causal_product import causal_product_trio_toep
from ..utils.toep_mat import ToepMat

class SKITno(Sltno):
    def __init__(self, h, dim, r, nk, **kwargs):
        """
        r: num inducing locations
        """
        super().__init__(h, dim, r, nk, **kwargs)
        gamma = torch.Tensor([kwargs.get('gamma', 0.99)])
        self.causal = kwargs.get('causal', False)

        self.pos = nn.Parameter(torch.randn((h*dim, r)))
        if not self.causal:
            self.neg = nn.Parameter(torch.randn((h*dim, r-1)))
        self.register_buffer('inducing', torch.linspace(0, 1-1/r, r))
        # falloff
        self.register_buffer('gamma', gamma)

    def gen_low_rank_U(self, t):
        n, _ = t.shape
        #original locations
        falloff = self.gamma ** (t * n)  # (n, 1)
        original_loc = t[..., 0]  # (n, )

        # get largest point in the inducing < the original location
        # TODO: closed form since it's all uniform (not super important)
        indices = torch.searchsorted(self.inducing, original_loc)
        # should be unnecessary with side='left' above as default, but *shrug*
        indices.clamp_(min=0, max=self.inducing.shape[0]-2)

        #lerp weights: n
        weights_l = (original_loc - self.inducing[indices]) * self.r  # spacing of 1/r
        weights_u = 1 - weights_l

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
        t = self.gen_pos(n, absolute=True, device=x.device)

        W = self.gen_low_rank_U(t)

        # build inducing kernel matrix
        neg = self.neg if not self.causal else torch.zeros_like(self.pos)  # (hd, r)
        a = torch.cat([self.pos, neg], dim=0)  # (hd, 2r)

        # this one is still slow
        x = x.transpose(0, 1).reshape((n, b*hd))  # (n, b*hd)
        Wt = W.T
        W = W.to_sparse()
        Wt = Wt.to_sparse()  # (r, n)
        S = ToepMat(a, self.r) 
        # these `view` and `permute` required to use ToepMat aren't too expensive
        output = torch.sparse.mm(Wt, x).view((self.r, b, hd)).permute((1,2,0))[..., None]  # (b, hd, r, 1)
        output = (S @ output)[..., 0].permute((2,0,1)).view((self.r, b*hd))  # (r, b*hd)
        output = torch.sparse.mm(W, output)  # (n, b*hd)
        output = output.view((n, b, hd)).transpose(0, 1)  # (b, n, hd)
        return output
    
    def apply_toeplitz(self, x):

        return 
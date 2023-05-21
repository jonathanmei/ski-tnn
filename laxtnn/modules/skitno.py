import torch

from .rpe import Rpe
from .sltno import Sltno
from ..utils.misc import get_activation_fn
from ..utils.toep_mat import ToepMat


class SKITno(Sltno):
    def __init__(self, h, dim, r, nk, **kwargs):
        """
        r: num inducing locations
        """
        super().__init__(h, dim, r, nk, **kwargs)
        rpe_dim = kwargs.get('rpe_dim', 4)
        residual = kwargs.get('residual', False)
        act = kwargs.get('act', 'relu')
        bias = kwargs.get('bias', True)
        layers = kwargs.get('layers', 3)
        norm_type = kwargs.get('norm_type', 'simplermsnorm')
        act_type = kwargs.get('act_type', 'none')
        self.rpe = Rpe(
            dim=rpe_dim, 
            outdim=h * dim, 
            residual=residual,
            act=act,
            bias=bias, 
            layers=layers,
            norm_type=norm_type,
        )
        self.act_fun = get_activation_fn(act_type)
        self.register_buffer('inducing', torch.linspace(0, 1, r))

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
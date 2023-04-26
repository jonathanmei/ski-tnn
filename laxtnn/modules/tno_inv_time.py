import torch
import torch.nn as nn
from einops import rearrange

# from custom fairseq
from fairseq.modules.helpers import get_activation_fn, print_params
from .rpe import Rpe

from ..utils.profiling import pytorch_profile

class TnoInvTime(nn.Module):
    def __init__(
        self, 
        h, 
        dim, 
        rpe_dim, 
        causal=False, 
        use_decay=False, 
        use_multi_decay=False, 
        residual=False, 
        act="relu", 
        par_type=1, 
        gamma=0.999,
        bias=True,
        act_type="none",
        layers=3,
        norm_type="simplermsnorm",
    ):
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        self.h = h
        self.dim = dim
        self.causal = causal
        self.par_type = par_type
        self.zero_value = 0
        self.use_decay = use_decay
        if self.use_decay:
            self.gamma = nn.Parameter(torch.ones(h, 1, dim) * gamma, requires_grad=False)
        self.use_multi_decay = use_multi_decay
        if self.use_multi_decay:
            self.lambda_ = gamma
            self.gamma = nn.Parameter(torch.randn(h, 1, dim))

        self.rpe = Rpe(
            dim=rpe_dim, 
            outdim=h * dim, 
            residual=residual,
            act=act,
            bias=bias, 
            layers=layers,
            norm_type=norm_type,
        )
        
        if self.causal:
            self.forward = self.forward_causal
        else:
            self.forward = self.forward_non_causal
            
        self.act_fun = get_activation_fn(act_type)

    def get_pos(self, n):
        index = 1 / torch.arange(2, 1+n).reshape(n-1, -1)
        return index
        
    def get_zero(self):
        index = torch.ones(1).reshape(1, -1)
        return index

    def get_neg(self, n):
        if self.causal:
            index = torch.zeros(self.h * n * self.dim).reshape(self.h, n, self.dim)
        else:
            index = -1 / torch.arange(n, 1, -1).reshape(n-1, -1)
        return index
    
    def rpe_transform(self, x):
        # n, 1 -> n, (d * h)
        res = self.rpe(x)
        # n, (d * h) -> h, n, d
        res = rearrange(res, 'n (h d) -> h n d', h=self.h)

        return res
    
    def forward_causal(self, x, dim=-2, normalize=False):
        # x: b, h, n, d
        n = x.shape[dim]
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        ##### coef
        # 1, d, 1 -> h, 1, d
        zero = self.rpe_transform(self.get_zero().to(x))
        pos = self.rpe_transform(self.get_pos(n).to(x))

        if self.use_decay or self.use_multi_decay:
            coef = torch.arange(1, n).reshape(1, -1, 1).to(x)
            if self.use_decay:
                gamma = self.gamma
            else:
                gamma = torch.sigmoid(self.gamma)
                gamma = self.lambda_ + (1 - self.lambda_) * gamma
            gamma = gamma ** coef
            pos = gamma * pos
        a = torch.cat([zero, pos, zero], dim=1)
        a = self.act_fun(a)

        # x: b, h, n, d
        # a: h, l, d
        output = self.compute(x, a, dim, n)

        if normalize:
            size = list(x.shape[:-1]) + [1]
            ones = torch.ones(size).to(x)
            denorm = self.compute(ones, a, dim, n)
            output = output / denorm
        return output

    def forward_non_causal(self, x, dim=-2, normalize=False):
        # x: b, h, n, d
        n = x.shape[dim]
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        ##### coef
        # 1, d, 1 -> h, 1, d
        zero = self.rpe_transform(self.get_zero().to(x))
        pos = self.rpe_transform(self.get_pos(n).to(x))
        neg_index = self.get_neg(n).to(x)
        if self.causal:
            neg = neg_index
        else:
            neg = self.rpe_transform(neg_index)

        if self.use_decay or self.use_multi_decay:
            coef = torch.arange(1, n).reshape(1, -1, 1).to(x)
            if self.use_decay:
                gamma = self.gamma
            else:
                gamma = torch.sigmoid(self.gamma)
                gamma = self.lambda_ + (1 - self.lambda_) * gamma
            gamma = gamma ** coef
            pos = gamma * pos
            neg = torch.flip(gamma, dims=[1]) * neg
        a = torch.cat([zero, pos, zero, neg], dim=1)
        a = self.act_fun(a)
        # x: b, h, n, d
        # a: h, l, d
        output = self.compute(x, a, dim, n)

        if normalize:
            size = list(x.shape[:-1]) + [1]
            ones = torch.ones(size).to(x)
            denorm = self.compute(ones, a, dim, n)
            output = output / denorm

        return output
    
    def compute(self, x, a, dim, n):
        # x: b, h, n, d
        # a: h, n, d
        y = torch.fft.rfft(x, 2 * n, dim=dim)
        v = torch.fft.rfft(a, 2 * n, dim=dim).unsqueeze(0)
        u = v * y
        output = torch.fft.irfft(u, 2 * n, dim=dim)[:, :, :n, :]

        return output


# Sparse+Low-Rank Toeplitz Neural Operator that all Lax operators should build on
# The Sparse component will be handled by a 1D convolution
# The Low-rank component will be handled by various Linear approximations

import torch
import torch.nn as nn
import torch.nn.functional as F

# from custom fairseq
from fairseq.modules.helpers import get_activation_fn, print_params

from .sl_toeplitz import causal_sl_toeplitz
from ..utils.profiling import pytorch_profile
DEFAULT_GAMMA = 0.999

class Sltno(nn.Module):
    def __init__(
        self, 
        h, 
        dim, 
        r,
        nk,  # convolution kernel size (for causal model, half will be unused)
        causal=False, 
        fft_bw=True,  # use the custom function for bw pass?
        use_decay=False, 
        use_multi_decay=False, 
        residual=False, 
        act="relu", 
        par_type=1, 
        gamma=DEFAULT_GAMMA,
        bias=True,
        act_type="none",
        layers=3,
        norm_type="simplermsnorm",
        laplace=False,
        falloff=True,
        **unused_kwargs,
    ):
        """
        Attributes:

            h `int`: number of heads
            r `int`: rank of low-rank component
            nk `int`: size of neighborhood for convolution kernel. should be odd. if not, increased to the nearest odd int
        """
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)
        
        self.h = h
        self.dim = dim
        self.r = r
        odd = nk % 2
        self.nk = nk + 1 - odd
        self.causal = causal
        self.fft_bw = fft_bw
        self.par_type = par_type
        self.nk2 = self.nk // 2
        self.conv_kernel = nn.Parameter(torch.randn(h * dim, self.nk2+1 if self.causal else self.nk))

    def get_pos(self, n, device='cuda'):
        """
        get n uniformly spaced position indices in [0, 1)
        """
        index = torch.linspace(0, 1-1/n, n, device=device).reshape(n, -1)
        return index

    def compute(self, x):
        """
        x (b, n, hd)
        """
        n = x.shape[-2]
        output = torch.zeros_like(x)
        if self.nk > 0:
            output += self.apply_sparse(x)
        if self.r > 0:
            output += self.apply_low_rank(x, self.get_pos(n, device=x.device))
        return output

    def forward(self, x, dim=-2, normalize=False):
        # x: b, n, hd
        ## should take care of the decay in apply_low_rank
        ## we have the ability to pass in arbitrary position indices as the 2nd argument. at the moment we generate it uniformly
        output = self.compute(x)
        if normalize:
            size = list(x.shape[:-1]) + [1]
            ones = torch.ones(size, device=x.device)
            denorm = self.compute(ones)
            output = output / (1e-9 + denorm)
        return output

    def apply_sparse(self, x):
        # x: b, n, hd
        # kernel: hd, n_k
        hd = x.shape[-1]
        # Letting pytorch optimize how to apply the small 1D convolution:
        output = F.conv1d(
            x.transpose(1, 2), # b n hd -> b hd n
            self.conv_kernel[:, None],  # hd n_k -> hd 1 n_k
            #padding='same',  # newer versions
            padding=self.nk2,  # pytorch==1.8.1
            groups=hd
        )  # b, hd, n+nk2
        if self.causal:
            output = output[..., :-self.nk2]
        return output.transpose(1, 2)

    def apply_low_rank(self, x, t):
        """
        Abstract
        Params:
            x `torch.Tensor` (b, n, hd): sequences of tokens
            t `torch.Tensor` (n, 1): relative position indices
        """
        raise NotImplementedError

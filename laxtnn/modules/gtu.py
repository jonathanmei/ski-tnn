from dataclasses import dataclass
from einops import rearrange

import torch
import torch.nn as nn

from .tno import Tno  # Vanilla Toeplitz Neural Operator (TNO)
from .tno_fd import TnoFD
from .skitno import SKITno  # Structured Kernel Interpolation (SKI) TNO
from .skitno_inv_time import SKITnoInvTime
from ..utils.misc import get_activation_fn, get_norm_fn

@dataclass
class TnoConfig:
    h: int
    dim: int
    rpe_dim: str
    causal: bool
    use_decay: bool
    use_multi_decay: bool
    residual: bool
    act: str
    par_type: str
    gamma: float
    bias: bool
    act_type: str
    layers: int
    norm_type: str


class Gtu(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        act_fun="silu",
        causal=False,
        expand_ratio=3,
        resi_param=False,
        use_norm=False,
        norm_type="simplermsnorm",
        use_decay=False,
        use_multi_decay=False,
        rpe_layers=3,
        rpe_embedding=512,
        rpe_act="relu",
        normalize=False,
        par_type=1,
        residual=False,
        gamma=0.99,
        act_type="none",
        # lax
        args=None,
        tno_fd=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.expand_ratio = expand_ratio
        self.resi_param = resi_param
        self.num_heads = num_heads
        self.normalize = normalize

        if self.resi_param:
            self.d = nn.Parameter(torch.randn(embed_dim))

        d1 = int(self.expand_ratio * embed_dim)
        d1 = (d1 // self.num_heads) * self.num_heads
        self.head_dim = d1 // num_heads
        # linear projection
        self.v_proj = nn.Linear(embed_dim, d1, bias=bias)
        self.u_proj = nn.Linear(embed_dim, d1, bias=bias)
        self.o = nn.Linear(d1, embed_dim, bias=bias)
        self.act = get_activation_fn(act_fun)

        TnoModule = TnoFD if tno_fd else Tno
        kwargs = {}
        self.tno_type = getattr(args, 'tno_type', 'tno')
        config = TnoConfig(
            h=num_heads, 
            dim=self.head_dim,
            rpe_dim=rpe_embedding,
            causal=causal,
            use_decay=use_decay,
            use_multi_decay=use_multi_decay,
            residual=residual,
            act=rpe_act,
            par_type=par_type,
            gamma=gamma,
            bias=bias,
            act_type=act_type,
            layers=rpe_layers,
            norm_type=norm_type,
        ).__dict__
        if self.tno_type == 'tno':  # Vanilla Toeplitz Neural Operator (TNO)
            self.toep = TnoModule(**config)
        elif self.tno_type == 'skitno':  # Structured Kernel Interpolation (SKI) TNO
            self.rank = args.rank
            self.nk = args.nk
            self.toep = SKITno(r=self.rank, nk=self.nk,  # conv kernel width
                **config
            )
        else:  # self.tno_type == 'skitno_inv_time':
            self.rank = args.rank
            self.nk = args.nk
            self.toep = SKITnoInvTime(r=self.rank, nk=self.nk,  # conv kernel width
                **config
            )

        # norm
        self.norm_type = norm_type
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = get_norm_fn(self.norm_type)(d1)

    def forward(self, x):
        # x: b, h, w, d
        num_heads = self.num_heads
        u = self.act(self.u_proj(x))  # gating
        v = self.act(self.v_proj(x))  # (b, n, hd)  input to TNO
        # maybe reshape
        if self.tno_type == 'tno' or self.tno_type == 'tno_inv_time':  # Vanilla Toeplitz Neural Operator (TNO) or TNO_FD
            v = rearrange(v, 'b n (h d) -> b h n d', h=num_heads)
            output = self.toep(v, dim=-2, normalize=self.normalize)
            output = rearrange(output, 'b h n d -> b n (h d)')
        else:  # self.tno_type == 'skitno':  # Structured Kernel Interpolation (SKI) TNO
            output = self.toep(v, normalize=self.normalize)
        output = u * output
        if self.use_norm:
            output = self.norm(output)

        output = self.o(output)

        return output

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

# from custom fairseq
from fairseq.modules.helpers import get_activation_fn, print_params

from .rpe import Rpe


class TnoFD(nn.Module):
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
            self.gamma = nn.Parameter(
                torch.ones(h, 1, dim) * gamma, requires_grad=False
            )
        self.use_multi_decay = use_multi_decay
        if self.use_multi_decay:
            self.lambda_ = gamma
            self.gamma = nn.Parameter(torch.randn(h, 1, dim))

        # import ipdb; ipdb.set_trace()
        self.rpe = Rpe(
            dim=rpe_dim,
            outdim=h * dim * (1 if causal else 2),
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

    def get_w(self, n):
        # if self.par_type == 1:
        w_axis = torch.linspace(0.0, np.pi, int(n / 2 + 1)).reshape(int(n / 2 + 1), -1)
        # elif self.par_type == 2:
        #     index = torch.arange(1, 1 + n).reshape(n, -1) * 1.0 / n
        # elif self.par_type == 3:
        #     index = torch.exp(torch.arange(1, 1 + n).reshape(n, -1) * 1.0 / n)

        return w_axis

    def get_pos(self, n):
        if self.par_type == 1:
            index = torch.arange(1, 1 + n).reshape(n, -1) * 1.0
        elif self.par_type == 2:
            index = torch.arange(1, 1 + n).reshape(n, -1) * 1.0 / n
        elif self.par_type == 3:
            index = torch.exp(torch.arange(1, 1 + n).reshape(n, -1) * 1.0 / n)

        return index

    def get_zero(self):
        index = torch.zeros(1).reshape(1, -1) * 1.0
        if self.par_type == 3:
            index = torch.exp(index)

        return index

    def get_neg(self, n):
        if self.causal:
            index = (
                torch.ones(self.h * n * self.dim).reshape(self.h, n, self.dim)
                * self.zero_value
            )
        else:
            if self.par_type == 1:
                index = -torch.arange(1, 1 + n).flip(0).reshape(n, -1) * 1.0
            elif self.par_type == 2:
                index = -torch.arange(1, 1 + n).flip(0).reshape(n, -1) * 1.0 / n

        return index

    def rpe_transform(self, x):
        # n, 1 -> n, (d * h)
        res = self.rpe(x)
        # n, (d * h) -> h, n, d
        res = rearrange(res, "n (h d) -> h n d", h=self.h)

        if self.causal:
            # generate imag part from real part to enforce spectrum of causal kernel
            f_dim = 1
            num_freqs = res.shape[f_dim]
            irfft_rp = torch.fft.irfft(
                res, dim=f_dim
            )  # ifft of real part of rpe kernel
            irfft_rp[:, num_freqs::, :] = 0.0
            irfft_rp[:, 0, :] *= 0.5
            res = 2.0 * torch.fft.rfft(irfft_rp, dim=f_dim)

        else:
            D = res.shape[2] // 2
            real, imag = res[:, :, 0:D], res[:, :, D::]
            imag[:, (0, -1), :] = 0.0  # force DC and Nyquist to be real only
            res = torch.complex(real, imag)

        return res

    def forward_causal(self, x, dim=-2, normalize=False):
        # import ipdb; ipdb.set_trace()
        # x: b, h, n, d
        n = x.shape[dim]
        # a0, a1, ... , a(n-1), a0, a(-(n-1)), ... , a(-1)
        ##### coef
        # 1, d, 1 -> h, 1, d
        # import ipdb; ipdb.set_trace()
        # zero = self.rpe_transform(self.get_zero().to(x))
        # pos = self.rpe_transform(self.get_pos(n - 1).to(x))
        fft_size = 2 * n
        pos_w = self.rpe_transform(self.get_w(fft_size).to(x))

        # if self.use_decay or self.use_multi_decay:
        #     coef = torch.arange(0, n).reshape(1, -1, 1).to(x)
        #     if self.use_decay:
        #         gamma = self.gamma
        #     else:
        #         gamma = torch.sigmoid(self.gamma)
        #         gamma = self.lambda_ + (1 - self.lambda_) * gamma
        #     gamma = gamma**coef
        # pos = gamma * pos
        # a = torch.cat([zero, pos, zero], dim=1)

        a = self.act_fun(pos_w)

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
        # zero = self.rpe_transform(self.get_zero().to(x))
        # pos = self.rpe_transform(self.get_pos(n - 1).to(x))

        # neg_index = self.get_neg(n - 1).to(x)

        fft_size = 2 * n
        pos_w = self.rpe_transform(self.get_w(fft_size).to(x))

        # if self.causal:
        #     neg = neg_index
        # else:
        #     neg = self.rpe_transform(neg_index)

        # if self.use_decay or self.use_multi_decay:
        #     coef = torch.arange(1, n).reshape(1, -1, 1).to(x)
        #     if self.use_decay:
        #         gamma = self.gamma
        #     else:
        #         gamma = torch.sigmoid(self.gamma)
        #         gamma = self.lambda_ + (1 - self.lambda_) * gamma
        #     gamma = gamma**coef
        #     pos = gamma * pos
        #     neg = torch.flip(gamma, dims=[1]) * neg
        # a = torch.cat([zero, pos, zero, neg], dim=1)
        a = self.act_fun(pos_w)
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
        # v = torch.fft.rfft(a, 2 * n, dim=dim).unsqueeze(0)
        u = a * y
        output = torch.fft.irfft(u, 2 * n, dim=dim)[:, :, :n, :]

        return output

    def toeplizt_matrix(self, x, dim):
        assert dim == -2
        # shape of x: b, h, n, d
        n = x.shape[dim]
        # c: first col, r: first row
        # 1, d, 1 -> h, 1, d
        zero = self.rpe_transform(self.get_zero().to(x))
        pos = self.rpe_transform(self.get_pos(n - 1).to(x))
        neg_index = self.get_neg(n - 1).to(x)
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
            gamma = gamma**coef
            pos = gamma * pos
            neg = torch.flip(gamma, dims=[1]) * neg
        zero = self.act_fun(zero)
        pos = self.act_fun(pos)
        if not self.causal:
            neg = self.act_fun(neg)
        c = torch.cat([zero, pos], dim=-2)
        r = torch.cat([zero, neg.flip(1)], dim=-2)
        vals = torch.cat([r, c[:, 1:].flip(1)], dim=-2)
        n = c.shape[-2]
        shape = self.h, n, n
        i, j = torch.ones(n, n).nonzero().T
        T = vals[:, j - i].reshape(self.h, n, n, -1)

        res = torch.einsum("h n m d, b h m d -> b h n d", T, x)
        return res, T

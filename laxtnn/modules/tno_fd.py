import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from ..utils.misc import print_params
from .rpe import Rpe


class TnoFD(nn.Module):
    """Implements the toeplitz neural operator for FD TNN.

    NOTE: This is the TNO used for FD TNN in "SKI to go Faster: Accelerating
    Toeplitz Neural Networks via Asymmetric Kernels"
    """

    def __init__(
        self,
        h: int,
        dim: int,
        rpe_dim: int,
        causal: bool = False,
        residual: bool = False,
        act: str = "relu",
        bias: bool = True,
        layers: int = 3,
        norm_type: str = "simplermsnorm",
        **kwargs,
    ) -> nn.Module:
        """Instantiate the TNO for FD TNN.

        Args:
            h (int): number of heads
            dim (int): token embedding length
            rpe_dim (int): width of the RPE hidden layers
            causal (bool, optional): Set for causal sequence modeling.
                Defaults to False.
            residual (bool, optional): Set to enable RPE residual
                connections. Defaults to False.
            act (str, optional): Activation function used by the RPE.
                Defaults to "relu".
            bias (bool, optional): Set to false to remove bias
                parameters from the RPE. Defaults to True.
            layers (int, optional): The number of RPE hidden layers.
                Defaults to 3.
            norm_type (str, optional): Normalization to use in RPE MLP.
                Defaults to "simplermsnorm".

        Returns:
            nn.Module: a TnoFD module
        """
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)

        self.h = h
        self.dim = dim
        self.causal = causal
        self.layers = layers

        # rpe output has 2x output width in non-causal case
        self.rpe = Rpe(
            dim=rpe_dim,
            outdim=h * dim * (1 if causal else 2),
            residual=residual,
            act=act,
            bias=bias,
            layers=layers,
            norm_type=norm_type,
        )

    @staticmethod
    def get_w(fft_size: int) -> torch.Tensor:
        """Generates the vector of frequencies used by RPE.

        Args:
            fft_size (int) : fft size (assumed to be even length)

        Returns:
            torch.Tensor: the frequency axis, a fft_size/2 + 1 length vector of
                ordered frequencies.
        """
        w_axis = torch.linspace(0.0, np.pi, int(fft_size / 2 + 1)).reshape(
            int(fft_size / 2 + 1), -1
        )

        return w_axis

    def rpe_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Generate the kernels' complex frequency responses via the RPE.

        Args:
            x (torch.Tensor): the frequency axis

        Returns:
            torch.Tensor: complex frequency responses of all kernels
        """
        # f, 1 -> f, (d * h)
        res = self.rpe(x)
        # f, (d * h) -> h, f, d
        res = rearrange(res, "f (h d) -> h f d", h=self.h)
        
        if self.causal:
            # generate imag part from real part to enforce spectrum of causal
            # kernel
            res = causal_spectrum(res)

        else:
            # use half of the RPE outputs for real part of kernel frequency
            # responses and the other half for the imaginary part
            D = res.shape[2] // 2
            real, imag = res[:, :, 0:D], res[:, :, D::]
            imag[:, (0, -1), :] = 0.0  # force DC and Nyquist to be real only
            res = torch.complex(real, imag)

        return res

    def forward(
        self, x: torch.Tensor, dim: int = -2, **kwargs
    ) -> torch.Tensor:
        """Compute the forward pass for the TNO FD.

        Args:
            x (torch.Tensor): token sequences
            dim (int, optional): sequence dimension. Defaults to -2.

        Returns:
            torch.Tensor: output token sequences convolved with tno kernels
        """
        # x: b, h, n, d
        n = x.shape[dim]  # sequence length

        # set fft_size to nearest power of 2 greater than 2x sequence length
        fft_size = int(2 ** np.ceil(np.log2(2 * n)))

        w = self.get_w(fft_size).to(x)  # frequency axis, input to rpe

        # a: h, fft_size/2+1, d
        a = self.rpe_transform(w)  # kernel frequency responses

        output = fft_convolve(x, a, dim, n, fft_size)

        return output


def fft_convolve(
    x: torch.Tensor, a: torch.Tensor, dim: int, n: int, fft_size: int
) -> torch.Tensor:
    """Convolve inputs and kernels via frequency domain multiplication.

    Args:
        x (torch.Tensor): input token sequences
        a (torch.Tensor): kernels' complex frequency responses
        dim (int): sequence dimension, where fft is applied
        n (int): input sequence length (time domain)
        fft_size (int): the fft size

    Returns:
        torch.Tensor: output token sequences
    """
    # x: b, h, n, d
    # a: h, fft_size/2+1, d
    y = torch.fft.rfft(x, fft_size, dim=dim)

    u = a * y

    # output sequences are truncated to input sequence length
    output = torch.fft.irfft(u, fft_size, dim=dim)[:, :, :n, :]

    return output


def causal_spectrum(x: torch.Tensor, f_dim: int = 1) -> torch.Tensor:
    """Computes the frequency response of causal sequence given real part.

    Args:
        x (torch.Tensor) : real part of the desired causal frequency response
        f_dim (int, optional) : frequency dimension of x. Defaults to 1.

    Returns:
        torch.Tensor: complex frequency response of causal
            time sequence having the given real part frequency response
    """
    num_freqs = x.shape[f_dim]
    irfft_rp = torch.fft.irfft(x, dim=f_dim)  # ifft of real part of rpe kernel
    irfft_rp[:, num_freqs::, :] = 0.0
    irfft_rp[:, 0, :] *= 0.5
    return 2.0 * torch.fft.rfft(irfft_rp, dim=f_dim)

import numpy as np
import torch
import torch.nn as nn

from ..utils.misc import print_params
from .tno_fd import causal_spectrum, fft_convolve


class TnoFDNoRPE(nn.Module):
    """Implements a toeplitz neural operator variant of fd tnn w/o an RPE MLP.

    This version directly parameterizes the kernel frequency response at a
    a discrete number of uniformly spaced frequencies, using linear
    interpolation to upsample the frequency response when extrapolating
    to longer sequence lengths. For causal sequence modeling, only the real
    values are parameterized and the hilbert transform is used to recover the
    imaginary part.

    NOTE: This is NOT the version used for FD TNN in our paper "SKI to go
    Faster", but one thing we tried that worked ok, improving speed (not 
    as well as FD TNN/TnoFD), & suffering accuracy loss when interpolating 
    more than 4x (for longer sequences). Smoother interp may improve this 
    issue.
    """

    def __init__(
        self,
        h: int,
        dim: int,
        causal: bool = False,
        num_freqs: int = 128,
        **kwargs,
    ) -> nn.Module:
        """Instantiate the FD TNO with an RPE MLP.

        Args:
            h (int): number of heads
            dim (int): token embedding length
            causal (bool, optional): Set for causal sequence modeling.
                Defaults to False.
            num_freqs (int, optional): Number of frequencies to parameterize.
                Defaults to 128.

        Returns:
            nn.Module: a TnoFDNoRPE module
        """
        super().__init__()
        # get local varables
        params = locals()
        # print params
        print_params(**params)

        self.h = h
        self.dim = dim
        self.causal = causal
        self.num_freqs = num_freqs
        self.time_factor = 2 * num_freqs

        if self.causal:
            self.coeffs = nn.Parameter(
                torch.randn(dim, num_freqs), requires_grad=True
            )
        else:
            self.coeffs_re = nn.Parameter(
                torch.randn(dim, num_freqs), requires_grad=True
            )
            self.coeffs_im = nn.Parameter(
                torch.randn(dim, num_freqs), requires_grad=True
            )

        self.dc = nn.Parameter(torch.ones(dim, 1), requires_grad=True)

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
        if self.causal:
            a = torch.cat(
                        [
                            self.dc,
                            nn.functional.interpolate(
                                self.coeffs.unsqueeze(0),
                                scale_factor=fft_size / self.time_factor,
                                mode="linear",
                            ).squeeze(0),
                        ],
                        dim=1,
                    )
            a = causal_spectrum(a.transpose(1, 0).unsqueeze(0))
        else:
            sf = fft_size / self.time_factor

            pos_r = nn.functional.interpolate(
                self.coeffs_re.unsqueeze(0),
                scale_factor=sf,
                mode="linear",
            ).squeeze(0)

            pos_i = nn.functional.interpolate(
                self.coeffs_im.unsqueeze(0),
                scale_factor=sf,
                mode="linear",
            ).squeeze(0)

            pos_i[:, -1] = 0.0  # Nyquist bin should be real valued

            a = torch.cat(
                [
                    self.dc,
                    torch.complex(pos_r, pos_i),
                ],
                dim=1,
            )
            a = a.transpose(1, 0).unsqueeze(0)

        output = fft_convolve(x, a, dim, n, fft_size)

        return output

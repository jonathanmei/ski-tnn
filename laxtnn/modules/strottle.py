import copy
import math
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.functional as F
import torch.nn as nn
from torch import Tensor


class MultiChannelLinear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        num_chs: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_chs = num_chs
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((num_chs, out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty((num_chs, out_features), **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for n in range(self.num_chs):
            nn.init.kaiming_uniform_(self.weight[n, :, :], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[n, :, :])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias[n, :], -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.einsum("Bcx,cyx->Bcy", input, self.weight) + self.bias

    def extra_repr(self) -> str:
        return "num_chs={}, in_features={}, out_features={}, bias={}".format(
            self.num_chs, self.in_features, self.out_features, self.bias is not None
        )


class DepthWiseStridedConvEncoder(nn.Module):
    def __init__(
        self,
        emb_len: int,
        seq_len: int,
        kernel_sizes: List[int] = None,
        strides: List[int] = None,
        act_fn: Callable[[Tensor], Tensor] = nn.ReLU(),
        out_fn: Optional[Callable[[Tensor], Tensor]] = None,
        target_latent_len: int = None,
    ):
        super(DepthWiseStridedConvEncoder, self).__init__()
        self.emb_len = emb_len
        self.seq_len = seq_len
        self.act_fn = act_fn
        self.out_fn = out_fn

        # arg checking
        assert (target_latent_len is None) != (
            kernel_sizes is None
        ), "Cannot specific kernel_sizes when setting target_latent_len. It's either or."

        if target_latent_len is None:
            strides = strides or kernel_sizes
            strides = strides * len(kernel_sizes) if len(strides) == 1 else strides
            assert len(strides) == len(
                kernel_sizes
            ), "strides and kernel_sizes lengths are incompatible."
            num_layers = len(kernel_sizes)
        else:
            num_layers = round(np.log2(seq_len) - np.log2(target_latent_len))
            kernel_sizes = [2] * num_layers
            strides = [2] * num_layers

        self.kernel_sizes = kernel_sizes
        self.strides = strides

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    emb_len,
                    emb_len,
                    kernel_size=k,
                    stride=s,
                    groups=emb_len,
                )
                for k, s in zip(kernel_sizes, strides)
            ]
        )

        out_len = copy.deepcopy(seq_len)
        for k, s in zip(kernel_sizes, strides):
            out_len = int(np.floor((out_len - (k - 1) - 1) / s + 1))

        assert (
            out_len > 1
        ), f"kernel_sizes and strides result in latent dimension {out_len}"

        self.latent_len = out_len

    def forward(self, x, return_activations=False):
        activations = []
        num_layers = len(self.conv_layers)
        for l, layer in enumerate(self.conv_layers):
            x = layer(x)
            if l < num_layers - 1:
                x = self.act_fn(x)
                if return_activations:
                    activations.append(x)

        x = x if self.out_fn is None else self.out_fn(x)

        return (x, activations) if return_activations else x


class DepthWiseStridedConvDecoder(nn.Module):
    def __init__(
        self,
        emb_len: int,
        latent_len: int,
        kernel_sizes: List[int] = None,
        strides: List[int] = None,
        act_fn: Callable[[Tensor], Tensor] = nn.ReLU(),
        out_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        super(DepthWiseStridedConvDecoder, self).__init__()

        self.emb_len = emb_len
        self.latent_len = latent_len
        self.act_fn = act_fn
        self.out_fn = out_fn

        # arg checking
        strides = strides or kernel_sizes
        strides = strides * len(kernel_sizes) if len(strides) == 1 else strides
        assert len(strides) == len(
            kernel_sizes
        ), "strides and kernel_sizes lengths are incompatible."
        num_layers = len(kernel_sizes)

        self.kernel_sizes = kernel_sizes
        self.strides = strides

        self.deconv_layers = nn.ModuleList(
            [
                nn.ConvTranspose1d(
                    emb_len,
                    emb_len,
                    kernel_size=k,
                    stride=s,
                    groups=emb_len,
                )
                for k, s in zip(kernel_sizes, strides)
            ]
        )

    def forward(self, x, encoder_activations=None):
        num_layers = len(self.deconv_layers)
        for l, layer in enumerate(self.deconv_layers):
            x = layer(x)
            if l < num_layers - 1:
                x = self.act_fn(x)
                x = x + (
                    0.0
                    if encoder_activations is None
                    else encoder_activations[num_layers - l - 2]
                )

        x = x if self.out_fn is None else self.out_fn(x)

        return x


class Strottleneck(nn.Module):
    def __init__(
        self,
        encoder: DepthWiseStridedConvEncoder,
        decoder: DepthWiseStridedConvDecoder,
        latent_net: Optional[nn.Module] = None,
        unet_connections: bool = False,
    ):
        super(Strottleneck, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_net = latent_net
        self.unet_connections = unet_connections

    def forward(self, x):
        encodings = self.encoder(x, return_activations=self.unet_connections)

        if self.unet_connections:
            encodings, activations = encodings

        encodings = encodings if self.latent_net is None else self.latent_net(encodings)

        decodings = self.decoder(
            encodings,
            encoder_activations=activations if self.unet_connections else None,
        )
        return decodings


def param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

from typing import List, Optional, Any
from collections import OrderedDict
from functools import partial
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


class ImplicitDecoder(nn.Module):

    def __init__(self,
                 dims: List[int] = [512, 512, 512, 512],
                 injection_dims: Optional[List[int]] = None,
                 latent_size: int = 256,
                 groups: int = 1,
                 final_out_dim: int = 1,
                 dropout_layers: List[int] = [0, 1, 2, 3],
                 dropout_prob: float = 0.0,
                 norm_layers: List[int] = [0, 1, 2, 3],
                 latent_in: List[int] = [2],
                 normalization: Optional[str] = 'weight',
                 activation: Optional[str] = None,
                 latent_dropout: bool = False,
                 query_dim: int = 0,
                 residual: bool = False,
                 fft: bool = False,
                 fft_dim: int = 10,
                 fft_progressive_factor: float = 0,
                 radius_init: float = 1,
                 init_std: float = 0.000001,
                 beta: int = 100,
                 **kwargs):
        super(ImplicitDecoder, self).__init__()

        if fft:
            input_dim = latent_size + 3*(2*fft_dim+1)
        else:
            input_dim = latent_size + query_dim
        dims = [input_dim] + dims + [final_out_dim]

        if fft:
            self.gauss_fft = Embedding(
                in_channels=query_dim,
                N_freqs=fft_dim,
                progressive_factor=fft_progressive_factor)

        self.num_layers = len(dims)
        self.dims = dims
        self.query_dim = query_dim
        self.latent_size = latent_size
        self.groups = groups

        if latent_dropout:
            self.latent_dropout = nn.Dropout(p=0.2)

        activation = "igr"
        nl, weight_init, first_layer_init, last_layer_init = (nn.Softplus(beta=beta),
                    partial(init_weights_geometric, p=1.0), None,
                    partial(init_weights_geometric_last_layer,
                            p=1.0, radius_init=radius_init,
                            init_std=init_std))
        final_nl = None

        layers_dict = OrderedDict()
        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
            in_dim = dims[layer]
            if injection_dims is not None and layer != 0:
                in_dim = in_dim + injection_dims[layer - 1]

            first_layer = layer == 0
            last_layer = layer == self.num_layers - 2

            tmp_weight_init = weight_init
            if first_layer and first_layer_init is not None:
                tmp_weight_init = first_layer_init
            if last_layer and last_layer_init is not None:
                tmp_weight_init = last_layer_init

            tmp_nl = nl if not last_layer else final_nl

            layers_dict['layer{}'.format(layer)] = AssembledLayer(
                in_features=in_dim,
                out_features=out_dim,
                groups=groups,
                bias=True,
                beta=beta,
                normalization=normalization if layer in norm_layers else None,
                dropout_prob=dropout_prob if layer in dropout_layers else None,
                activation=tmp_nl,
                weight_init=tmp_weight_init,
                first_layer=first_layer,
                last_layer=last_layer,
                residual=residual,
                geometric_init=activation == 'igr',
                latent_in=layer in latent_in)
        self.model = nn.Sequential(layers_dict)

    def forward(self, input, progress=1.0):
        if self.latent_size == 0:
            latent, xyz = input.new_empty([input.shape[0], 0]), input
        elif self.query_dim == 0:
            latent, xyz = input, input.new_empty([input.shape[0], 0])
        else:
            latent, xyz = torch.split(input, [self.latent_size, self.query_dim],
                                      dim=-1)

        if hasattr(self, 'latent_dropout'):
            latent = self.dropout(latent)
        if hasattr(self, 'gauss_fft'):
            xyz_feat = self.gauss_fft(xyz, progress=progress)
            x = torch.cat([latent, xyz_feat], dim=-1)
            skip = x.clone()
        else:
            x = torch.cat([latent, xyz], dim=-1)
            skip = input

        data = {'x': x, 'latent': latent, 'xyz': xyz, 'skip': skip}
        data = self.model(data)

        return data['x']


class ImplicitLayer(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 normalization: Optional[str] = 'weight',
                 dropout_prob: Optional[float] = None,
                 activation: Optional[Any] = None,
                 weight_init: Optional[Any] = None,
                 last_layer: bool = False,
                 first_layer: bool = False,
                 residual: bool = False):
        super(ImplicitLayer, self).__init__()

        self.last_layer = last_layer
        self.first_layer = first_layer
        self.residual = residual

        self.lin = nn.Linear(in_features=in_features,
                             out_features=out_features,
                             bias=bias)
        if weight_init is not None:
            self.lin.apply(weight_init)

        if normalization is not None:
            if normalization == 'weight':
                self.lin = nn.utils.weight_norm(self.lin)
            elif normalization == 'layer' and not last_layer:
                self.norm = nn.LayerNorm(out_features)

        if activation is not None:
            self.activation = activation

        if dropout_prob is not None and not last_layer:
            self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, input):
        x = self.lin(input)
        if hasattr(self, 'norm'):
            x = self.norm(x)
        if hasattr(self, 'activation'):
            if self.last_layer:
                x = self.activation(x) + 1.0 * x
            else:
                if self.residual and not self.first_layer:
                    x = x + input
                x = self.activation(x)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        return x


class AssembledLayer(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 groups: int = 1,
                 bias: bool = True,
                 beta: Optional[int] = 100,
                 normalization: Optional[str] = 'weight',
                 dropout_prob: Optional[float] = None,
                 activation: Optional[Any] = None,
                 weight_init: Optional[Any] = None,
                 last_layer: bool = False,
                 first_layer: bool = False,
                 residual: bool = False,
                 geometric_init: bool = True,
                 latent_in: bool = False,
                 query_in: bool = False):
        super(AssembledLayer, self).__init__()

        self.last_layer = last_layer
        self.geometric_init = geometric_init
        self.latent_in = latent_in
        self.query_in = query_in and not first_layer

        self.implicit_layer = ImplicitLayer(in_features=in_features,
                                            out_features=out_features,
                                            bias=bias,
                                            normalization=normalization,
                                            dropout_prob=dropout_prob,
                                            activation=activation,
                                            weight_init=weight_init,
                                            last_layer=last_layer,
                                            first_layer=first_layer,
                                            residual=residual)

    def forward(self, data):
        x = data['x']
        if self.latent_in:
            x = torch.cat([x, data['skip']], dim=-1)
            if self.geometric_init:
                x /= np.sqrt(2)
        elif self.query_in:
            x = torch.cat([x, data['xyz']], dim=-1)
        x = self.implicit_layer(x)
        data['x'] = x
        return data


def init_weights_geometric(m, p=1.0):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 0.0,
                            np.sqrt(2) / np.sqrt(p * m.weight.shape[0]))
        if hasattr(m, 'bias'):
            torch.nn.init.constant_(m.bias, 0.0)


def init_weights_geometric_last_layer(m, p=1.0, init_std=1e-6, radius_init=1.0):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight,
                            mean=np.sqrt(np.pi) /
                            np.sqrt(p * m.weight.shape[1]),
                            std=init_std)
        if hasattr(m, 'bias'):
            torch.nn.init.constant_(m.bias, -radius_init)


class Embedding(nn.Module):

    def __init__(self,
                 in_channels,
                 N_freqs,
                 logscale: bool = True,
                 progressive_factor: float = 0):
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.out_channels = in_channels * (2 * N_freqs + 1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs - 1), N_freqs)

        self.progressive_factor = progressive_factor

    def forward(self, x, progress=1.0):
        out = [x]
        funcs = [torch.sin, torch.cos]
        if self.progressive_factor > 0:
            prog_w = self.progressive_weight(np.arange(len(self.freq_bands)),
                                             progress)
        for i_freq, freq in enumerate(self.freq_bands):
            for func in funcs:
                if self.progressive_factor > 0:
                    out += [prog_w[i_freq] * func(freq * x)]
                else:
                    out += [func(freq * x)]

        return torch.cat(out, -1)

    def progressive_weight(self, freq_band, progress):
        alpha = progress * self.N_freqs * self.progressive_factor
        return (1 - np.cos(np.clip(alpha - freq_band, 0, 1))) / 2

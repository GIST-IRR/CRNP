import torch
import torch.nn as nn
import torch.nn.functional as F

from parser.modules.norm import MeanOnlyLayerNorm


class Sine(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(input)


class XSinusoid(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.sin(input)


class PXSinusoid(nn.Module):
    def __init__(self, a=2.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.sin(input) / self.a


class Relu_Sinusoid(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu(input) + torch.sin(input)


class ResidualConfig:
    def __init__(
        self,
        version=1,
        activation="relu",
        norm="layer",
        elementwise_affine=True,
        add_first=False,
        norm_first=False,
        dropout=0.0,
    ):
        self.version = version
        self.activation = activation
        self.norm = norm
        self.elementwise_affine = elementwise_affine
        self.add_first = add_first
        self.norm_first = norm_first
        self.dropout = dropout


default_residual_config = {
    "version": 1,
    "activation": "relu",
    "norm": None,
    "elementwise_affine": True,
    "add_first": False,
    "norm_first": False,
    "dropout": 0.0,
}


class ResLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        activation="relu",
        norm=None,
        dropout=0.0,
        elementwise_affine=True,
        add_first=False,
        norm_first=False,
        version=1,
    ):
        super(ResLayer, self).__init__()
        self.activation = activation
        self.add_first = add_first
        self.norm_first = norm_first
        self.version = version
        self.dropout = dropout

        if activation == "relu":
            activation = nn.ReLU
        elif activation == "leaky_relu":
            activation = nn.LeakyReLU
        elif activation == "gelu":
            activation = nn.GELU
        elif activation == "tanh":
            activation = nn.Tanh
        elif activation == "sine":
            activation = Sine
        elif activation == "px-sine":
            activation = PXSinusoid
        elif activation == "relu+sine":
            activation = Relu_Sinusoid

        if norm == "layer":
            norm = nn.LayerNorm
        elif norm == "mo-layer":
            norm = MeanOnlyLayerNorm
        elif norm == "batch":
            norm = nn.BatchNorm1d

        if version == 1:
            if norm is None:
                self.linear = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    activation(),
                    nn.Linear(out_dim, out_dim),
                )
                self.register_module("norm", None)
            else:
                self.linear = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    norm(out_dim, elementwise_affine=elementwise_affine),
                    activation(),
                    nn.Linear(out_dim, out_dim),
                    # norm(out_dim, elementwise_affine=elementwise_affine),
                )
                self.norm = norm(
                    out_dim, elementwise_affine=elementwise_affine
                )
            self.last_activation = activation()
        elif version == 2:
            self.linear = nn.Sequential(
                norm(in_dim, elementwise_affine=elementwise_affine),
                activation(),
                # nn.Dropout(dropout),
                nn.Linear(in_dim, out_dim),
                norm(out_dim, elementwise_affine=elementwise_affine),
                activation(),
                # nn.Dropout(dropout),
                nn.Linear(out_dim, out_dim),
            )
            # self.register_module("last_activation", None)

    def forward(self, x):
        if self.version == 1:
            if self.add_first:
                if self.norm:
                    if self.norm_first:
                        x = x + self.norm(self.linear(x))
                    else:
                        x = self.norm(self.linear(x) + x)
                else:
                    x = x + self.linear(x)
                return self.last_activation(x)
            else:
                return x + self.last_activation(self.linear(x))
        elif self.version == 2:
            return x + self.linear(x)


class Bilinear_ResLayer(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        norm=None,
        elementwise_affine=True,
        *args,
        **kwargs
    ):
        super(Bilinear_ResLayer, self).__init__()

        if norm == "layer":
            norm = nn.LayerNorm
        elif norm == "batch":
            norm = nn.BatchNorm1d

        self.relu_layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            norm(out_dim, elementwise_affine=elementwise_affine),
            nn.ReLU(),
        )
        self.sine_layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            norm(out_dim, elementwise_affine=elementwise_affine),
            Sine(),
        )
        self.sine_amplitude = nn.Parameter(torch.ones(out_dim))

    def forward(self, x):
        r_x = self.relu_layer(x)
        s_x = self.sine_amplitude * self.sine_layer(x)
        return x + r_x + s_x


if __name__ == "__main__":
    import torch

    x = torch.randn(2, 4)

    net = ResLayer(4, 4, 3, activation="tanh")
    print(net(x))

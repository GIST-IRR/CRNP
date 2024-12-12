import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ResLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        activation="relu",
        norm=None,
        elementwise_affine=True,
    ):
        super(ResLayer, self).__init__()
        if activation == "relu":
            activation = nn.ReLU
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
        elif norm == "batch":
            norm = nn.BatchNorm1d

        if norm is None:
            self.linear = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                activation(),
                nn.Linear(out_dim, out_dim),
                activation(),
            )
        else:
            self.linear = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                norm(out_dim, elementwise_affine=elementwise_affine),
                activation(),
                nn.Linear(out_dim, out_dim),
                norm(out_dim, elementwise_affine=elementwise_affine),
                activation(),
            )

    def forward(self, x):
        return self.linear(x) + x


class ResLayerNorm(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResLayerNorm, self).__init__()
        self.linear = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_dim, out_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(out_dim, out_dim)),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.linear(x) + x


if __name__ == "__main__":
    import torch

    x = torch.randn(2, 4)

    net = ResLayer(4, 4, 3, activation="tanh")
    print(net(x))

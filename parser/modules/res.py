import torch.nn as nn


class ResLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation="relu", norm=None):
        super(ResLayer, self).__init__()
        if activation == "relu":
            activation = nn.ReLU
        elif activation == "tanh":
            activation = nn.Tanh

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
                norm(out_dim),
                activation(),
                nn.Linear(out_dim, out_dim),
                norm(out_dim),
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

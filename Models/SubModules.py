import math

import torch
from torch import nn


class DepthWiseConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size, kernels_per_layer, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * kernels_per_layer,
                                   kernel_size=kernel_size, groups=in_channels, bias=bias, padding='same')

    def forward(self, x):
        return self.depthwise(x)


class PointWiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernels_per_layer=1, bias=False):
        super().__init__()
        self.pointwise = nn.Conv2d(in_channels=in_channels * kernels_per_layer, out_channels=out_channels,
                                   kernel_size=(1, 1), bias=bias, padding="valid")

    def forward(self, x):
        return self.pointwise(x)


class MaxNormLayer(nn.Linear):
    def __init__(self, in_features, out_features, max_norm=1.0):
        super(MaxNormLayer, self).__init__(in_features=in_features, out_features=out_features)
        self.max_norm = max_norm

    def forward(self, x):
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.data = torch.renorm(
                    self.weight.data, p=2, dim=0, maxnorm=self.max_norm
                )
        return super(MaxNormLayer, self).forward(x)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, kernels_per_layer=1, bias=False):
        super().__init__()
        self.depthwise = DepthWiseConv2d(in_channels=in_channels, kernels_per_layer=kernels_per_layer,
                                         kernel_size=kernel_size, bias=bias)
        self.pointwise = PointWiseConv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernels_per_layer=kernels_per_layer, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ViewConv(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view((x.shape[0], x.shape[1], 1, x.shape[2]))


class ToTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view((x.shape[1], x.shape[0], x.shape[2]))


class Unsqueeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view((x.shape[1], x.shape[0], 1))

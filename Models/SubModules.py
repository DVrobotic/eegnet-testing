import math

import torch
from torch import nn
import torch.nn.functional as F


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
        return x.view((x.shape[0], x.shape[1], 1))


class EmbeddingLayer(nn.Module):
    def __init__(
            self,
            n_times,
            patches_num,
            embedding_dim,
            dropout_rate,
            depth=1
    ):
        super().__init__()
        self.patches_num = patches_num
        self.embedding = nn.Linear(in_features=math.ceil(n_times * depth / self.patches_num),
                                   out_features=embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.positional_encode_table = nn.Embedding(num_embeddings=self.patches_num, embedding_dim=embedding_dim)

    # output shape -> [Batches, embeding dim, pathces,]
    def forward(self, x):
        extract_image_patches()
        patches = list(x.chunk(self.patches_num, dim=-1))
        patches[-1] = F.pad(patches[-1], (0, patches[0].shape[-1] - patches[-1].shape[-1]))
        patches = torch.stack(patches, dim=-2)

        # creating new dimesion for embedding (its not -1 to not bug the matrix mul)
        patches = patches.unsqueeze(-2)
        embedding_result = self.embedding(patches)

        # creating removing the unnecessary dimension to add the positional encode
        embedding_result = embedding_result.squeeze(-2)
        positional_result = embedding_result + self.positional_encode_table(
            (torch.arange(self.patches_num)))
        out = self.dropout(positional_result)

        return out.view(out.shape[0], out.shape[2], out.shape[1])
        # return out.view(out.shape[0], out.shape[1], out.shape[3], out.shape[2])


class VisionTransformer(nn.Module):
    def __init__(
            self,
            patches,
            transformer_ffd,
            nhead=8,
            num_layers=2
    ):
        super().__init__()
        self.transformer = nn.Sequential(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=patches, dim_feedforward=transformer_ffd, nhead=nhead,
                                           batch_first=True),
                num_layers=num_layers,
            ),
        )

    def forward(self, x):
        return self.transformer(x)


class ConvolutionSimplifier(nn.Module):
    def __init__(self, embedding_dim, dropout_rate):
        super().__init__()
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.ELU(),
        )

    def forward(self, x):
        return self.conv(x)


class FeatureExtraction(nn.Module):
    def __init__(
            self,
            n_channels,
            kernel_length,
            F1,
            D,
            F2,
            pool1_stride,
            pool2_stride,
    ):
        super().__init__()
        self.net = nn.Sequential(
            ViewConv(),
            nn.Conv2d(in_channels=n_channels, out_channels=F1, kernel_size=(1, kernel_length), bias=False,
                      padding='same'),
            nn.BatchNorm2d(num_features=F1, momentum=0.01, eps=0.001, track_running_stats=False),
            DepthWiseConv2d(in_channels=F1, kernel_size=(n_channels, 1), kernels_per_layer=D, bias=False),
            nn.BatchNorm2d(num_features=F1 * D, momentum=0.01, eps=0.001, track_running_stats=False),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool1_stride), stride=pool1_stride),
            SeparableConv2d(in_channels=F1 * D, kernel_size=(1, 16), out_channels=F2, bias=False),
            nn.BatchNorm2d(num_features=F2, momentum=0.01, eps=0.001, track_running_stats=False),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool2_stride), stride=pool2_stride),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)

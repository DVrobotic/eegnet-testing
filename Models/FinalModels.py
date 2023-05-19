import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from x_transformers import Encoder

from utils.utils import initialize_weight
from base.layers import Conv2dWithConstraint


def exists(val):
    return val is not None


class MyViTransformerWrapper(nn.Module):
    def __init__(
            self,
            *,
            image_size,
            patch_size,
            attn_layers,
            channels,
            num_classes=None,
            dropout=0.,
            post_emb_norm=False,
            emb_dropout=0.
    ):
        super().__init__()
        assert isinstance(attn_layers, Encoder), 'attention layers must be an Encoder'
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size)
        patch_dim = channels * patch_size
        dim = attn_layers.dim
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(num_patches, dim))

        self.patch_to_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.post_emb_norm = nn.LayerNorm(dim) if post_emb_norm else nn.Identity()
        self.dropout = nn.Dropout(emb_dropout)

        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Linear(dim, num_classes) if exists(num_classes) else nn.Identity()

    def forward(
            self,
            img,
            return_embeddings=False
    ):
        p = self.patch_size
        img = img.unsqueeze(-2)
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=p)
        x = self.patch_to_embedding(x)
        n = x.shape[1]
        x = x + self.pos_embedding

        x = self.post_emb_norm(x)
        x = self.dropout(x)

        x = self.attn_layers(x)
        x = self.norm(x)

        if not exists(self.mlp_head) or return_embeddings:
            return x

        x = x.mean(dim=-2)
        return self.mlp_head(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *config, max_norm=1, **kwconfig):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*config, **kwconfig)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(LinearWithConstraint, self).forward(x)


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
            dropout_rate,
            weight_init_method=None
    ):
        super().__init__()

        # Spectral
        self.rearrange = Rearrange('b c w -> b 1 c w')
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), bias=False, padding='same')
        self.batch1 = nn.BatchNorm2d(F1)

        # Spatial
        self.conv2 = Conv2dWithConstraint(F1, F1 * D, (n_channels, 1), bias=False, groups=F1)
        self.batch2 = nn.BatchNorm2d(F1 * D)
        self.activation1 = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, pool1_stride))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Temporal
        self.conv3 = nn.Conv2d(F1 * D, F2, (1, n_channels), bias=False, padding='same', groups=F2)
        self.conv4 = nn.Conv2d(F2, F2, 1, bias=False)
        self.batch3 = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.avgpool2 = nn.AvgPool2d((1, pool2_stride))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Classifier
        self.flatten = nn.Flatten()

        initialize_weight(self, weight_init_method)

    def forward(self, x):
        out = self.rearrange(x)
        out = self.conv1(out)
        out = self.batch1(out)

        # Spatial
        out = self.conv2(out)
        out = self.batch2(out)
        out = self.activation1(out)
        out = self.avgpool1(out)
        out = self.dropout1(out)

        # Temporal
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.batch3(out)
        out = self.activation2(out)
        out = self.avgpool2(out)
        out = self.dropout2(out)
        out = self.flatten(out)

        return out


class EegnetTemporal(nn.Module):
    def __init__(
            self,
            n_times,
            n_classes,
            n_channels,
            kernel_length,
            patches_size=5,
            transformer_dim=4,
            transformer_layers=1,
            transformer_heads=1,
            dropout_rate=0.5,
            max_norm=0.25,
            F1=8,
            F2=16,
            D=2,
            pool1_stride=16,
            pool2_stride=8,
            n_hidden=4,
    ):
        super().__init__()
        self.feature_extraction_output = F2 * (
                    (((n_times - pool1_stride) // pool1_stride + 1) - pool2_stride) // pool2_stride + 1)

        self.feature_extraction = nn.Sequential(
            FeatureExtraction(n_channels=n_channels, kernel_length=kernel_length, F1=F1, D=D, F2=F2,
                              pool1_stride=pool1_stride, pool2_stride=pool2_stride, dropout_rate=dropout_rate),
            nn.Linear(in_features=self.feature_extraction_output, out_features=n_hidden),
            nn.ELU(),
        )

        self.transformer = nn.Sequential(
            MyViTransformerWrapper(
                image_size=n_times,
                patch_size=patches_size,
                channels=n_channels,
                dropout=dropout_rate,
                attn_layers=Encoder(
                    dim=transformer_dim,
                    depth=transformer_layers,
                    heads=transformer_heads,
                    macaron=True,
                    rel_pos_bias=True,
                    # attn_dim_head = 4,
                    # value_dim_head=4,
                    # dim_head=4,
                    # ff_mult=1,
                ),
            ),
            nn.Linear(in_features=transformer_dim, out_features=n_hidden),
            nn.ELU(),
        )

        self.head = nn.Sequential(
            LinearWithConstraint(in_features=2*n_hidden, out_features=n_classes, max_norm=max_norm),
        )

    def forward(self, x, targets):
        out_values = {}
        feature_extraction_result = self.feature_extraction(x)
        transformer_result = self.transformer(x)
        logits = self.head(torch.cat([
            feature_extraction_result,
            transformer_result,
        ], dim=-1))

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets)

        return logits, loss, out_values


class EegnetTemporalSpatial(nn.Module):
    def __init__(
            self,
            n_times,
            n_classes,
            n_channels,
            kernel_length,
            patches_size=5,
            transformer_dim=4,
            transformer_layers=1,
            transformer_heads=1,
            dropout_rate=0.5,
            max_norm=0.25,
            F1=8,
            F2=16,
            D=2,
            pool1_stride=16,
            pool2_stride=8,
            n_hidden=4,
    ):
        super().__init__()
        self.feature_extraction_output = F2 * (
                (((n_times - pool1_stride) // pool1_stride + 1) - pool2_stride) // pool2_stride + 1)

        self.feature_extraction = nn.Sequential(
            FeatureExtraction(n_channels=n_channels, kernel_length=kernel_length, F1=F1, D=D, F2=F2,
                              pool1_stride=pool1_stride, pool2_stride=pool2_stride, dropout_rate=dropout_rate),
            nn.Linear(in_features=self.feature_extraction_output, out_features=n_hidden),
            nn.ELU(),
        )

        self.transformer = nn.Sequential(
            MyViTransformerWrapper(
                image_size=n_times,
                patch_size=patches_size,
                channels=n_channels,
                dropout=dropout_rate,
                attn_layers=Encoder(
                    dim=transformer_dim,
                    depth=transformer_layers,
                    heads=transformer_heads,
                    macaron=True,
                    rel_pos_bias=True,
                    # attn_dim_head = 4,
                    # value_dim_head=4,
                    # dim_head=4,
                    # ff_mult=1,
                ),
            ),
            nn.Linear(in_features=transformer_dim, out_features=n_hidden),
            nn.ELU(),
        )

        self.spatial_transformer = nn.Sequential(
            nn.AdaptiveAvgPool1d(32),
            Rearrange("b c t -> b t c"),
            MyViTransformerWrapper(
                image_size=n_channels,
                patch_size=n_channels//2,
                channels=32,
                dropout=dropout_rate,
                attn_layers=Encoder(
                    dim=transformer_dim,
                    depth=transformer_layers,
                    heads=transformer_heads,
                    macaron=True,
                    rel_pos_bias=True,
                    attn_dim_head = 32,
                ),
            ),
            nn.Linear(in_features=transformer_dim, out_features=n_hidden),
            nn.ELU(),
        )

        self.head = nn.Sequential(
            LinearWithConstraint(in_features=3*n_hidden, out_features=n_classes, max_norm=max_norm),
        )

    def forward(self, x, targets):
        out_values = {}
        feature_extraction_result = self.feature_extraction(x)
        transformer_result = self.transformer(x)
        spatial_transformer_result = self.spatial_transformer(x)
        logits = self.head(torch.cat([
            feature_extraction_result,
            transformer_result,
            spatial_transformer_result
        ], dim=-1))

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets)

        return logits, loss, out_values


class EegnetSpatial(nn.Module):
    def __init__(
            self,
            n_times,
            n_classes,
            n_channels,
            kernel_length,
            patches_size=5,
            transformer_dim=4,
            transformer_layers=1,
            transformer_heads=1,
            dropout_rate=0.5,
            max_norm=0.25,
            F1=8,
            F2=16,
            D=2,
            pool1_stride=16,
            pool2_stride=8,
            n_hidden=4,
    ):
        super().__init__()
        self.feature_extraction_output = F2 * (
                (((n_times - pool1_stride) // pool1_stride + 1) - pool2_stride) // pool2_stride + 1)

        self.feature_extraction = nn.Sequential(
            FeatureExtraction(n_channels=n_channels, kernel_length=kernel_length, F1=F1, D=D, F2=F2,
                              pool1_stride=pool1_stride, pool2_stride=pool2_stride, dropout_rate=dropout_rate),
            nn.Linear(in_features=self.feature_extraction_output, out_features=n_hidden),
            nn.ELU(),
        )

        self.spatial_transformer = nn.Sequential(
            nn.AdaptiveAvgPool1d(32),
            Rearrange("b c t -> b t c"),
            MyViTransformerWrapper(
                image_size=n_channels,
                patch_size=n_channels//2,
                channels=32,
                dropout=dropout_rate,
                attn_layers=Encoder(
                    dim=transformer_dim,
                    depth=transformer_layers,
                    heads=transformer_heads,
                    macaron=True,
                    rel_pos_bias=True,
                    attn_dim_head = 32,
                ),
            ),
            nn.Linear(in_features=transformer_dim, out_features=n_hidden),
            nn.ELU(),
        )

        self.head = nn.Sequential(
            LinearWithConstraint(in_features=2*n_hidden, out_features=n_classes, max_norm=max_norm),
        )

    def forward(self, x, targets):
        out_values = {}
        feature_extraction_result = self.feature_extraction(x)
        spatial_transformer_result = self.spatial_transformer(x)
        logits = self.head(torch.cat([
            feature_extraction_result,
            spatial_transformer_result
        ], dim=-1))

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets)

        return logits, loss, out_values


class Eegnet(nn.Module):
    def __init__(
            self,
            n_times,
            n_classes,
            n_channels,
            kernel_length,
            patches_size=5,
            transformer_dim=4,
            transformer_layers=1,
            transformer_heads=1,
            dropout_rate=0.5,
            max_norm=0.25,
            F1=8,
            F2=16,
            D=2,
            pool1_stride=16,
            pool2_stride=8,
            n_hidden=4,
    ):
        super().__init__()
        self.feature_extraction_output = F2 * (
                (((n_times - pool1_stride) // pool1_stride + 1) - pool2_stride) // pool2_stride + 1)

        self.feature_extraction = nn.Sequential(
            FeatureExtraction(n_channels=n_channels, kernel_length=kernel_length, F1=F1, D=D, F2=F2,
                              pool1_stride=pool1_stride, pool2_stride=pool2_stride, dropout_rate=dropout_rate),
        )

        self.head = nn.Sequential(
            LinearWithConstraint(in_features=self.feature_extraction_output, out_features=n_classes, max_norm=max_norm),
        )

    def forward(self, x, targets):
        out_values = {}
        feature_extraction_result = self.feature_extraction(x)
        logits = self.head(feature_extraction_result)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets)

        return logits, loss, out_values
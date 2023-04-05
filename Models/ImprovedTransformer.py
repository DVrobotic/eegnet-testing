from torch import nn
from Models.SubModules import ViewConv, DepthWiseConv2d, SeparableConv2d, Unsqueeze, PositionalEncoding, ToTransformer, \
    MaxNormLayer
import torch.nn.functional as F


class EEGNET(nn.Module):
    def __init__(
            self,
            n_channels,
            n_times,
            n_classes,
            kernel_length=64,
            F1=8,
            D=2,
            F2=16,
            signal_size=32,
            pool1_stride=4,
            pool2_stride=8,
            dropout_rate=0.5,
            norm_rate=0.25,
            transformer_ffd=516,
    ):
        super().__init__()
        print('model instantianting...')
        self.net = nn.Sequential(
            ViewConv(),
            nn.Conv2d(in_channels=n_channels, out_channels=F1, kernel_size=(1, kernel_length), bias=False, padding='same'),
            nn.BatchNorm2d(num_features=F1, momentum=0.01, eps=0.001, track_running_stats=False),
            DepthWiseConv2d(in_channels=F1, kernel_size=(n_channels, 1), kernels_per_layer=D, bias=False),
            nn.BatchNorm2d(num_features=F1*D, momentum=0.01, eps=0.001, track_running_stats=False),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool1_stride), stride=pool1_stride),
            nn.Dropout(dropout_rate),
            SeparableConv2d(in_channels=F1*D, kernel_size=(1, 16), out_channels=F2, bias=False),
            nn.BatchNorm2d(num_features=F2, momentum=0.01, eps=0.001, track_running_stats=False),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool2_stride), stride=pool2_stride),
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(in_features=F2 * ((((n_times - pool1_stride) // pool1_stride + 1) - pool2_stride) // pool2_stride + 1), out_features=signal_size, bias=False),
            nn.ELU(),
            Unsqueeze(),
            PositionalEncoding(d_model=signal_size, dropout=0.1),
            ToTransformer(),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=signal_size, dim_feedforward=transformer_ffd, nhead=4, batch_first=True),
                num_layers=2,
            ),
            # ViewConv(),
            # nn.Conv2d(in_channels=signal_size, out_channels=1, kernel_size=(1, signal_size), bias=False, padding='same'),
            # nn.Flatten(),

            nn.AvgPool1d(kernel_size=4, stride=signal_size),
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.BatchNorm1d(num_features=signal_size, momentum=0.01, eps=0.001, track_running_stats=False),
            nn.ELU(),
            MaxNormLayer(in_features=signal_size, out_features=n_classes, max_norm=norm_rate),
            nn.Softmax(dim=1),
        )

    def forward(self, x, targets):
        out_values = {}
        out = x

        for layer in self.net.children():
            out = layer(out)
            out_values[layer.__class__.__name__] = out.clone()

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(out, targets)

        return out, loss, out_values
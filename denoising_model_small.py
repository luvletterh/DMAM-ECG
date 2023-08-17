import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log as ln


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        nn.init.zeros_(self.bias)


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        noise_level = noise_level.view(-1)
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-ln(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)

        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super().__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1)
        return x


class Bridge(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(512, num_heads=1)
        self.input_conv = Conv1d(input_size, input_size, 3, padding=1, padding_mode='reflect')
        self.encoding = FeatureWiseAffine(input_size, hidden_size, use_affine_level=1)
        self.output_conv = Conv1d(input_size, hidden_size, 3, padding=1, padding_mode='reflect')

    def forward(self, x, noise_embed):
        x = self.input_conv(x)
        x = self.encoding(x, noise_embed)
        x, _ = self.attention(x, x, x)

        return self.output_conv(x)


class CIL_FilBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dilation):
        super().__init__()
        self.cv1 = Conv(input_size, hidden_size, 3, dilation=dilation, n=1)  # 1
        self.cv3 = Conv(hidden_size, hidden_size, 5, dilation=dilation, n=2)  # 2
        self.filters = nn.ModuleList([
            Conv1d(input_size, hidden_size // 4, 3, dilation=dilation, padding=1 * dilation, padding_mode='reflect'),
            Conv1d(hidden_size, hidden_size // 4, 5, dilation=dilation, padding=2 * dilation, padding_mode='reflect'),
            Conv1d(hidden_size, hidden_size // 4, 9, dilation=dilation, padding=4 * dilation, padding_mode='reflect'),
            Conv1d(hidden_size, hidden_size // 4, 15, dilation=dilation, padding=7 * dilation, padding_mode='reflect'),
        ])
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool1d(kernel_size=7, stride=1, padding=3)
        self.m = nn.ModuleList([nn.MaxPool1d(kernel_size=x, stride=1, padding=x // 2) for x in [5, 9, 13]])  # 池化三个
        self.conv_1 = Conv(hidden_size, hidden_size, 9, 1, 4)
        self.conv_2 = Conv1d(hidden_size, hidden_size, 9, padding=4, padding_mode='reflect')

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv3(x)
        filts = []
        for layer in self.filters:
            filts.append(layer(x1))
        filts = torch.cat(filts, dim=1)
        filts1, filts2 = self.conv_1(filts).chunk(2, dim=1)
        filts1 = self.pool(filts1)
        filts2 = self.pool2(filts2)
        filts = F.leaky_relu(torch.cat([filts1, filts2], dim=1), 0.2)
        x2 = F.leaky_relu(x2, 0.2)
        return filts + x2 + x


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def autopad(kernel_size, stride):
    # Calculate total padding needed to preserve spatial dimensions.
    total_padding = (stride - 1) * kernel_size - stride + 2

    # Calculate padding on each side of the input tensor.
    padding = total_padding // 2

    return padding


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, k, dilation, n, s=1, p=1, g=1,
                 act=SiLU()):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, k, dilation=dilation, padding=n * dilation,
                              padding_mode='reflect')
        self.bn = nn.InstanceNorm1d(out_channels, eps=0.001, momentum=0.03)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class ConditionalModel(nn.Module):
    def __init__(self, feats=64):
        super(ConditionalModel, self).__init__()
        self.attention = nn.MultiheadAttention(512, num_heads=8)
        self.stream_x = nn.ModuleList([
            nn.Sequential(Conv1d(1, feats, 9, padding=4, padding_mode='reflect'),
                          nn.LeakyReLU(0.2)),
            CIL_FilBlock(feats, feats, 1),
            CIL_FilBlock(feats, feats, 2),
            CIL_FilBlock(feats, feats, 4),
            CIL_FilBlock(feats, feats, 2),
            CIL_FilBlock(feats, feats, 1),
            CIL_FilBlock(feats, feats, 1),
        ])

        self.stream_cond = nn.ModuleList([
            nn.Sequential(Conv1d(1, feats, 9, padding=4, padding_mode='reflect'),
                          nn.LeakyReLU(0.2)),
            CIL_FilBlock(feats, feats, 1),
            CIL_FilBlock(feats, feats, 2),
            CIL_FilBlock(feats, feats, 4),
            CIL_FilBlock(feats, feats, 2),
            CIL_FilBlock(feats, feats, 1),
            CIL_FilBlock(feats, feats, 1),
        ])

        self.embed = PositionalEncoding(feats)

        self.bridge = nn.ModuleList([
            Bridge(feats, feats),
            Bridge(feats, feats),
            Bridge(feats, feats),
            Bridge(feats, feats),
            Bridge(feats, feats),
            Bridge(feats, feats),
        ])

        self.conv_out = Conv1d(feats, 1, 9, padding=4, padding_mode='reflect')

    def forward(self, x, cond, noise_scale):
        noise_embed = self.embed(noise_scale)

        xs = []
        for layer, br in zip(self.stream_x, self.bridge):
            x = layer(x)
            xs.append(br(x, noise_embed))

        for x, layer in zip(xs, self.stream_cond):
            cond = layer(cond) + x

        return self.conv_out(cond)

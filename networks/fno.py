import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from networks.unet_parts import *


class Unet(nn.Module):
    def __init__(self, in_channels):
        super(Unet, self).__init__()

        self.n_channels = in_channels
        self.n_classes = in_channels * 2
        self.bilinear = False

        self.inc = DoubleConv(self.n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        factor = 2 if self.bilinear else 1
        self.up2 = Up(128, 64 // factor, self.bilinear)
        self.up3 = Up(64, 32 // factor, self.bilinear)
        self.outc = OutConv(32, self.n_classes)
        self.conv1 = nn.Conv2d(self.n_classes, self.n_classes, 3)
        self.conv2 = nn.Conv2d(self.n_classes, self.n_classes // 2, 3)
        self.prelu1 = nn.PReLU(self.n_classes)
        self.prelu2 = nn.PReLU(self.n_classes // 2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up2(x3, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        gap = self.conv1(logits.detach())
        gap = self.prelu1(gap)
        gap = self.conv2(gap)
        gap = self.prelu2(gap)
        gap = F.adaptive_avg_pool2d(gap, (1, 1))
        return logits, gap


class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.unet = Unet(out_channels)

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.randn(1, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(1, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        factor = x_ft.abs()
        factor = torch.cat((factor[:, :, :self.modes1, :self.modes2], factor[:, :, -self.modes1:, :self.modes2]), dim=2)
        factor, gap = self.unet(factor)
        factor = factor.reshape((2, 1, x_ft.shape[1], self.modes1 * 2, self.modes2))

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1.mul(torch.view_as_complex(torch.stack((factor[0][:, :, :self.modes1, :self.modes2], factor[1][:, :, :self.modes1, :self.modes2]), dim=-1))))
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2.mul(torch.view_as_complex(torch.stack((factor[0][:, :, -self.modes1:, :self.modes2], factor[1][:, :, -self.modes1:, :self.modes2]), dim=-1))))

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x, gap


class FNO2d(nn.Module):
    def __init__(self, modes, width, in_channel, out_channel):
        super(FNO2d, self).__init__()

        self.scales_per_block = [1, 1, 1, 2, 2, 2]
        self.share_block = [False, False, True, False, True, True]
        self.num_per_block = [2, 2, 2, 2, 2, 2]

        self.modes = modes
        self.width = width
        self.conv_begin_0 = nn.Conv2d(in_channel + 2, self.width, 1)
        self.conv_begin_1 = nn.Conv2d(self.width, self.width, 1)
        self.prelu_begin = nn.PReLU(self.width)

        self.SConv2d_list = []
        self.w_list = []
        self.prelu_list = []
        self.ssc_list = []
        self.conv_list = []

        current_width = self.width
        total_width = 0
        for i in range(len(self.scales_per_block)):
            if self.share_block[i]:
                self.conv_list.append(nn.Conv2d(current_width, current_width, 1))
                self.SConv2d_list.append(SpectralConv2d_fast(current_width, current_width, self.modes // self.scales_per_block[i], self.modes // self.scales_per_block[i]))
                self.w_list.append(nn.Conv2d(current_width, current_width, 1))
                self.prelu_list.append(nn.PReLU(current_width))
                total_width += current_width * self.num_per_block[i]
            else:
                for _ in range(self.num_per_block[i]):
                    self.conv_list.append(nn.Conv2d(current_width, current_width, 1))
                    self.SConv2d_list.append(SpectralConv2d_fast(current_width, current_width, self.modes // self.scales_per_block[i], self.modes // self.scales_per_block[i]))
                    self.w_list.append(nn.Conv2d(current_width, current_width, 1))
                    self.prelu_list.append(nn.PReLU(current_width))
                    total_width += current_width
            self.ssc_list.append(nn.Conv2d(current_width, current_width, 1))
            current_width += current_width

        self.conv_list = nn.ModuleList(self.conv_list)
        self.SConv2d_list = nn.ModuleList(self.SConv2d_list)
        self.w_list = nn.ModuleList(self.w_list)
        self.prelu_list = nn.ModuleList(self.prelu_list)
        self.ssc_list = nn.ModuleList(self.ssc_list)

        self.conv_end1 = nn.Conv2d(current_width, current_width, 1)
        self.conv_end2 = nn.Conv2d(current_width, out_channel, 1)
        self.prelu_end = nn.PReLU(current_width)

        self.mlps = nn.Sequential(*[nn.Linear(total_width, total_width // 8), nn.ReLU(), nn.Linear(total_width // 8, in_channel)])

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
        x = self.conv_begin_0(x)
        x = self.prelu_begin(x)
        x = self.conv_begin_1(x)

        features = [x]
        gap_list = []

        pointer = 0
        for i in range(len(self.scales_per_block)):
            x = torch.cat(features, 1)
            x_s = x
            if self.share_block[i]:
                for _ in range(self.num_per_block[i]):
                    x_t = x
                    x = self.conv_list[pointer](x)
                    result, gap = self.SConv2d_list[pointer](x)
                    gap_list.append(gap)
                    x = result + self.w_list[pointer](x)
                    x = self.prelu_list[pointer](x)
                    x = x + x_t
                pointer += 1
            else:
                for _ in range(self.num_per_block[i]):
                    x_t = x
                    x = self.conv_list[pointer](x)
                    result, gap = self.SConv2d_list[pointer](x)
                    gap_list.append(gap)
                    x = result + self.w_list[pointer](x)
                    x = self.prelu_list[pointer](x)
                    x = x + x_t
                    pointer += 1
            x = self.ssc_list[i](x)
            x = x + x_s
            features.append(x)

        x = torch.cat(features, 1)

        x = self.conv_end1(x)
        x = self.prelu_end(x)
        x = self.conv_end2(x)

        gaps = torch.cat(gap_list, dim=1)[:, :, 0, 0]
        pred_z = self.mlps(gaps)

        return x, pred_z

    def get_grid(self, shape, device):
        n, c, h, w = shape
        gridx = torch.tensor(np.linspace(0, 1, h), dtype=torch.float)
        gridx = gridx.reshape(1, 1, h, 1).repeat([n, 1, 1, w])
        gridy = torch.tensor(np.linspace(0, 1, w), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, w).repeat([n, 1, h, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)

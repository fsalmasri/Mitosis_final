import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import padding, upasample, norm


def activ():
    return nn.PReLU()

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            padding(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0),
        )

        self.double_conv = nn.Sequential(
            padding(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            norm(out_channels),
            activ(),
            padding(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            # norm(out_channels),
            # activ()
        )

    def forward(self, x):
        d = self.maxpool_conv(x)
        return self.double_conv(d)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                padding(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0),
            )
            # self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # self.conv = DoubleConv(in_channels, out_channels)

        self.conv = nn.Sequential(
            norm(in_channels),
            activ(),
            padding(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0),
            norm(out_channels),
            activ(),
            padding(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)
        )

    def forward(self, x1, x2):
        x = torch.cat([x2, self.up(x1)], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Unet2(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, bilinear=True):
        super(Unet2, self).__init__()
        self.inc = nn.Sequential(
            padding(1),
            nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0),
            activ(),
            padding(1),
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=0),
            # activ(),

        )
        self.down1 = Down(ngf, ngf * 2)
        self.down2 = Down(ngf * 2, ngf * 4)
        self.down3 = Down(ngf * 4, ngf * 8)
        # self.down4 = Down(ngf * 8, ngf * 16)
        # self.up1 = Up(ngf * 16, ngf * 8, bilinear)
        self.up2 = Up(ngf * 8, ngf * 4, bilinear)
        self.up3 = Up(ngf * 4, ngf * 2, bilinear)
        self.up4 = Up(ngf * 2, ngf, bilinear)
        self.outc = OutConv(ngf, output_nc)

        self.act = nn.Tanh()
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        #
        # x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        out = self.outc(x)
        return nn.Sigmoid()(out)

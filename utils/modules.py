import torch
import torch.nn as nn
import functools

def activ():
    return nn.PReLU()

def norm(out_channels):
    return nn.InstanceNorm2d(out_channels, affine=False)


def padding(n):
    return nn.ReflectionPad2d(n) #nn.ZeroPad2d(n) # nn.ConstantPad2d(n, -1) #

def upasample(inner_nc, outer_nc, mode='transposed', use_bias=False):
    if mode == 'transposed':
        return nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)

    elif mode == 'bilinear':
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            padding(1),
            nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=0, bias=False, groups=1)
        )
    elif mode == 'resized':
        return nn.Sequential(

            nn.Upsample(scale_factor=2, mode='nearest'), #nearest , align_corners=False
            padding(1),
            nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=0, bias=False, groups=1),
        )


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
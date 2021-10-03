import torch
import torch.nn as nn
from utils.modules import activ, norm

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            activ(),
            nn.Conv2d(in_channels=16, out_channels=32, stride=2, kernel_size=3, padding=1),
            norm(16),
            activ(),
        )


        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            norm(32),
            activ(),
            nn.Conv2d(in_channels=32, out_channels=32, stride=2, kernel_size=3, padding=1),
            norm(32),
            activ(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            norm(32),
            activ(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            norm(32),
            activ(),
        )

        self.conv4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, padding=0),
            activ(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, padding=0)
        )


    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x) 
        x = self.conv3(x)
        x = self.conv4(x)

        return nn.Sigmoid()(x.flatten(1))
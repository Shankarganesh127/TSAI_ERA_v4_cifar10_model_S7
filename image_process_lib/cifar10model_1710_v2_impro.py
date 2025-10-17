import torch
import torch.nn as nn
import torch.nn.functional as F
from image_process import image_processing

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution = Depthwise + Pointwise"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()

        # Depthwise: each input channel convolved separately
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                 stride=stride, padding=padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)

        # Pointwise: 1x1 conv to mix channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x

class Net(nn.Module):
    """CIFAR-10 CNN with C1C2C3C4 architecture, Depthwise Sep Conv, Dilated Conv, and GAP.
    - Parameters:
        - input_channels: Number of input channels (3 for RGB, 4 for RGB + processed).
    """
    def __init__(self, input_channels=4):
        super(Net, self).__init__()
        self.img_pro = image_processing()

        # C1: Initial feature extraction (32x32 -> 32x32)
        self.c1 = nn.Sequential(
            nn.Conv2d(input_channels, 8, 3, padding=1, bias=False),  # Dynamic input channels
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05),
        )

        # C2: Feature extraction with Dilated Convolutions (32x32 -> 32x32)
        self.c2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(32, 32, 3, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.05),
        )

        # C3: Pattern recognition with Depthwise Separable Conv (32x32 -> 16x16)
        self.c3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            DepthwiseSeparableConv(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.05),
        )

        # C4: Final convolution with stride=1 (16x16 -> 8x8), then GAP + FC
        self.c4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.img_pro.extract_image_features(x, include_processed_channel=(self.c1[0].in_channels == 4))
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        return F.log_softmax(x, dim=1)
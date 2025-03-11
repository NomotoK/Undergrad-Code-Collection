import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sam import SAM


# 定义Wide ResNet的基本残差块
class WideBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(WideBasicBlock, self).__init__()
        width = 4  # Wide ResNet的宽度因子
        self.conv1 = nn.Conv2d(in_channels, out_channels * width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels * width)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels * width, out_channels * width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * width)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out
    



    # 定义Wide ResNet模型
class WideResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(WideResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion * 4, num_classes)  # 乘以4是因为Wide ResNet的宽度因子为4

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion * 4  # 乘以4是因为Wide ResNet的宽度因子为4
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
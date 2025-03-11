import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dropout_rate):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # Shortcut connection when input and output dimensions are different
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = self.dropout(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes, dropout_rate=0.3):
        super(WideResNet, self).__init__()

        assert (depth - 4) % 6 == 0, 'Depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor

        n_stages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(1, n_stages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._wide_layer(BasicBlock, n_stages[0], n_stages[1], n, 1, dropout_rate)
        self.layer2 = self._wide_layer(BasicBlock, n_stages[1], n_stages[2], n, 2, dropout_rate)
        self.layer3 = self._wide_layer(BasicBlock, n_stages[2], n_stages[3], n, 2, dropout_rate)
        self.bn = nn.BatchNorm2d(n_stages[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_stages[3], num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _wide_layer(self, block, in_channels, out_channels, num_blocks, stride, dropout_rate):
        layers = []
        layers.append(block(in_channels, out_channels, stride, dropout_rate))
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, 1, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.bn(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Example usage
# depth = 28  # Depth of the Wide ResNet
# widen_factor = 10  # Widening factor
# num_classes = 10  # Number of classes in your classification problem

# model = WideResNet(depth, widen_factor, num_classes)
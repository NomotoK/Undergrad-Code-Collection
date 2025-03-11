import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from tqdm import tqdm

# 定义ShuffleNet模型
class ShuffleNet(nn.Module):
    def __init__(self, groups=3, in_channels=1, num_classes=10):
        super(ShuffleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ShuffleUnit(24, 24, groups=groups),
            ShuffleUnit(24, 24, groups=groups),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ShuffleUnit(24, 48, groups=groups),
            ShuffleUnit(48, 48, groups=groups),
            ShuffleUnit(48, 48, groups=groups),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ShuffleUnit(48, 96, groups=groups),
            ShuffleUnit(96, 96, groups=groups),
            ShuffleUnit(96, 96, groups=groups),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 定义ShuffleUnit
class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super(ShuffleUnit, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=groups, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 通道混洗
        channels = x.size(1)
        x1 = x[:, :channels // 2, :, :]
        x2 = x[:, channels // 2:, :, :]
        x1 = self.branch1(x1)
        out = torch.cat([x1, x2], 1)
        return out

# 准备数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = FashionMNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ShuffleNet(groups=3, in_channels=1, num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

# 测试模型
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")

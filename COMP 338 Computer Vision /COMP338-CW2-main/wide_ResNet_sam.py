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

# 实例化Wide ResNet模型
def WideResNet18():
    return WideResNet(WideBasicBlock, [2, 2, 2, 2])


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    def closure():
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        return loss
    

    # 训练模型
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step(closure)

        total_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f'Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    end_time = time.time()
    epoch_time = end_time - start_time
    print(f'Training Time for Epoch: {epoch_time:.2f} seconds')
    print(f'Average Loss for Epoch: {total_loss / len(train_loader)}')


def test(model, test_loader, device):
# 在测试集上评估模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy * 100:.2f}%')


# 在代码中添加验证损失的记录
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    return val_loss / len(val_loader)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device')

    # 准备Fashion-MNIST数据集，并进行数据增强
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomCrop(28, padding=4),  # 随机裁剪到28x28，并在边缘填充4个像素
        # transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

    # 将模型和数据移动到GPU上
    model = WideResNet18().to(device)
    

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, rho=0.05,adaptive= True)  # 使用 SAM 作为优化器
    #如果连续8个epoch的loss都没有下降，就将学习率降低为原来的0.2
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=8, verbose=True)

       # 训练模型
    num_epochs = 100

    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        train_loss = train(model, train_loader, criterion, optimizer, device)

        # 验证集上的损失
        val_loss = validate(model, test_loader, criterion, device)

        scheduler.step(val_loss)

        train(model, train_loader, criterion, optimizer, device)

    # 测试模型
    test(model, test_loader, device)
    




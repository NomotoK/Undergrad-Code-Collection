import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from _model2 import WideResNet  # 请确保模型的实现在一个名为model.py的文件中

def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


    average_loss = total_loss / len(train_loader)
    print(f"Training Loss: {average_loss:.4f}")

def test_model(model, test_loader, device):
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
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")

def main():
    # 超参数
    depth = 28
    width_factor = 10
    dropout = 0.3
    in_channels = 1  # 单通道灰度图像
    num_labels = 10
    batch_size = 64
    learning_rate = 0.001
    epochs = 10

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = WideResNet(depth=depth, width_factor=width_factor, dropout=dropout, in_channels=in_channels, labels=num_labels)
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 加载数据
    train_loader, test_loader = load_data(batch_size=batch_size)

    # 训练模型
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_model(model, train_loader, criterion, optimizer, device)

    # 评估模型
    test_model(model, test_loader, device)

if __name__ == "__main__":
    main()

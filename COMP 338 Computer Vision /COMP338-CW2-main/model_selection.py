import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# 定义数据转换
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 下载训练集和测试集
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# 加载数据
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# 定义模型
mobilenet = models.mobilenet_v2(pretrained=False)
# 修改输出层以适应Fashion-MNIST的类别数（10个类别）
mobilenet.classifier[1] = torch.nn.Linear(1280, 10)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

cnn = SimpleCNN()

resnet = models.resnet18(pretrained=False)
# 修改输出层以适应Fashion-MNIST的类别数（10个类别）
resnet.fc = nn.Linear(512, 10)

shufflenet = models.shufflenet_v2_x1_0(pretrained=False)
# 修改输出层以适应Fashion-MNIST的类别数（10个类别）
shufflenet.fc = nn.Linear(1024, 10)

import torchvision.models as models

widresnet = models.wide_resnet50_2(pretrained=False)
# 修改输出层以适应Fashion-MNIST的类别数（10个类别）
widresnet.fc = nn.Linear(2048, 10)




import torch.optim as optim

criterion = nn.CrossEntropyLoss()

# 选择适当的优化器和学习率
optimizer_mobilenet = optim.Adam(mobilenet.parameters(), lr=0.001)
optimizer_cnn = optim.Adam(cnn.parameters(), lr=0.001)
optimizer_resnet = optim.Adam(resnet.parameters(), lr=0.001)
optimizer_shufflenet = optim.Adam(shufflenet.parameters(), lr=0.001)
optimizer_widresnet = optim.Adam(widresnet.parameters(), lr=0.001)


def train_model(model, optimizer, criterion, trainloader, epochs=5):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:  # 每100个小批量打印一次损失
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0


# train_model(mobilenet, optimizer_mobilenet, criterion, trainloader)
train_model(cnn, optimizer_cnn, criterion, trainloader)
train_model(resnet, optimizer_resnet, criterion, trainloader)
train_model(shufflenet, optimizer_shufflenet, criterion, trainloader)
train_model(widresnet, optimizer_widresnet, criterion, trainloader)


def evaluate_model(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print('Accuracy on test set: %d %%' % (100 * accuracy))

# 评估每个模型
# evaluate_model(mobilenet, testloader)
evaluate_model(cnn, testloader)
evaluate_model(resnet, testloader)
evaluate_model(shufflenet, testloader)
evaluate_model(widresnet, testloader)




def plot_accuracy_curve(train_accuracy, test_accuracy, epochs):
    epochs_list = np.arange(1, epochs + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, train_accuracy, label='Training Accuracy', marker='o')
    plt.plot(epochs_list, test_accuracy, label='Test Accuracy', marker='o')
    
    plt.title('Training and Test Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
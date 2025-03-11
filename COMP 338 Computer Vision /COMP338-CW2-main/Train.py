import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sam import SAM
from tqdm import tqdm
from cutout import Cutout
from Model import WideResNet18


def load_and_preprocess_data(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomCrop(28, padding=4),  # 随机裁剪到28x28，并在边缘填充4个像素
        transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),# 转换为张量
        Cutout(size=8, p=0.5),  # 随机擦除
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader





def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
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
        optimizer.first_step(zero_grad=True)

        criterion(model(images), labels).backward()
        optimizer.second_step(zero_grad=True)

        total_loss += loss.item()

        loop.set_description(f"Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        loop.update(1)
        # if (i + 1) % 100 == 0:
        #     print(f'Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    print('Finished Training')
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
        return accuracy




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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device')

    # Hyperparameters
    batch_size = 128
    num_epochs = 100


    train_loader, test_loader = load_and_preprocess_data(batch_size)
    # 将模型和数据移动到GPU上
    model = WideResNet18().to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=0.01, rho=0.05,adaptive= True)  # 使用 SAM 作为优化器

    #如果连续8个epoch的loss都没有下降，就将学习率降低为原来的0.2,最多下降到原来的0.1倍
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=8, verbose=True, min_lr=0.0005)
    

    # 训练模型
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]\n ----------------------------------------------------------------------')

        train(model, train_loader, criterion, optimizer, device)
        
        val_loss = validate(model, test_loader, criterion, device) 
        acc = test(model, test_loader, device)

        # 更新学习率，传入训练集上的loss
        scheduler.step(val_loss)
        #将每个epoch的loss和acc存储在一个txt文件中
        with open('loss_acc.txt', 'a') as f:
            f.write(f'Epoch: {epoch+1}, acc: {acc:.4f}, Val Loss: {val_loss:.4f}\n')
            f.close()

        # test(model, test_loader, device)
        # 当学习率过小时， 停止训练
        # if optimizer.param_groups[0]['lr'] < 1e-5:
        #     break
        # #保存模型
        # torch.save(model.state_dict(), './wide_resnet_sam.pth')


    # 测试模型
    test(model, test_loader, device)

if __name__ == '__main__':

    main()
    #100epochs: 0,9462
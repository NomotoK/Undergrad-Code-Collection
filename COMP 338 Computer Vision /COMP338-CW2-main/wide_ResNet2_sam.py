import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sam import SAM
from tqdm import tqdm
from torch.nn import functional as F
from WideResNet_Model import WideResNet


mean, std = 0, 0
num_epochs = 100
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 导入数据
def get_train_data(mean, std):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)
        ])
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    return train_loader



def get_mean_std():
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST('./data', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                            ])),
        batch_size=64, shuffle=True)
    mean = 0.
    std = 0.
    nb_samples = len(train_loader.dataset)
    for data, _ in tqdm(train_loader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
    mean /= nb_samples
    std /= nb_samples
    return mean, std



def get_test_data(mean, std):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)
        ])
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    return test_loader



def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    start_time = time.time()
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for batch_idx, (inputs, targets) in loop:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        criterion(model(inputs), targets).backward()
        optimizer.second_step(zero_grad=True)
        total_loss += loss.item()
        loop.set_description(f"Train Loss: {loss.item():.4f}")
        loop.update(1)
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f'Training Time for Epoch: {epoch_time:.2f} seconds')
    print(f'Average Loss for Epoch: {total_loss / len(train_loader)}')


def main():
    mean, std = get_mean_std()
    print(mean, std)


if __name__ == '__main__':
    main()
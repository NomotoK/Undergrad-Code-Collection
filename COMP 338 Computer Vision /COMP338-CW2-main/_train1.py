import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from _model1 import WideResNet
from tqdm import tqdm
import time

def load_and_preprocess_data(batch_size):
    transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),  # 随机裁剪到28x28，并在边缘填充4个像素
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

def train_model(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    epoch = 0
    loop = tqdm(enumerate(trainloader), total=len(trainloader), leave=True)


        
    for i, (inputs, labels) in enumerate(trainloader, 0):

        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        loop.set_description(f"Step [{i+1}/{len(trainloader)}], Loss: {loss.item():.4f}")
        loop.set_postfix(loss=loss.item())
        loop.update(1)


    print('Finished Training')
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f'Training Time for Epoch: {epoch_time:.2f} seconds')
    print(f'Average Loss for Epoch: {running_loss / len(trainloader)}')



def test_model(model, testloader, device):

# 在测试集上评估模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Test Accuracy: {accuracy * 100:.2f}%')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device')
    # Hyperparameters
    batch_size = 64
    depth = 28
    widen_factor = 2
    num_classes = 10
    num_epochs = 10 
    # Load and preprocess data
    trainloader, testloader = load_and_preprocess_data(batch_size)

    # Initialize the Wide ResNet model
    model = WideResNet(depth, widen_factor, num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}\n--------------------------------------------------------------')
        train_model(model, trainloader, criterion, optimizer, device)
    # Train the model

    # Test the model
    test_model(model, testloader, device)



if __name__ == "__main__":
    main()
# without transform: 92.60%
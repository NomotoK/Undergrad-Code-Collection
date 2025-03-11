import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

# Define the Wide ResNet model
class WideBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(WideBasicBlock, self).__init__()
        width = 4
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
        self.fc = nn.Linear(256 * block.expansion * 4, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion * 4
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Instantiate the model
model = WideResNet(WideBasicBlock, [2, 2, 2])

# Load pre-trained weights
pretrained_dict = torch.load('path/to/your/pretrained_model.pth')

# If the model's state_dict has a 'module.' prefix, remove it
if 'module.' in list(pretrained_dict.keys())[0]:
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}

# Load the weights into the model
model.load_state_dict(pretrained_dict)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Assuming img_path is the path to your test image
img = Image.open(img_path).convert('L')  # Convert to grayscale
img = transform(img)
img = img.unsqueeze(0)  # Add batch dimension
# Set the model to evaluation mode
model.eval()

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load Fashion-MNIST test dataset
test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Inference
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print('Accuracy on Fashion-MNIST: {:.2%}'.format(accuracy))

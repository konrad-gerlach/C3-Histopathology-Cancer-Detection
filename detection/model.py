import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision


class Model1(nn.Module):
    def __init__(self, num_classes=1):
        super(Model1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7056, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

    def print_layers(self):
        X = torch.rand(size=(1, 3, 96, 96), dtype=torch.float32)
        print("layer1:")
        for layer in self.layer1:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape: \t', X.shape)
        print("layer2:")
        for layer in self.layer2:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape: \t', X.shape)


if __name__ == "__main__":
    model1 = Model1()
    model1.print_layers()

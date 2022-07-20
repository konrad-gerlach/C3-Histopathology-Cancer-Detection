import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision


class Small_LeNet(nn.Module):
    def __init__(self, num_classes=1):
        super(Small_LeNet, self).__init__()
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

    def get_layers(self):
        output = "Model summary:\n"
        X = torch.rand(size=(1, 3, 96, 96), dtype=torch.float32)
        
        output += "layer1:\n"        
        for layer in self.layer1:
            X = layer(X)
            output += (layer.__class__.__name__ + 'output shape: \t' + str(X.shape) +'\n')

        output += "layer2:\n"
        for layer in self.layer2:
            X = layer(X)
            output += (layer.__class__.__name__ + 'output shape: \t' + str(X.shape) + '\n')

        return output
    
    def print_layers(self):
        print(model1.get_layers())

class BigLeNet(nn.Module):
    def __init__(self, num_classes=1):
        super(BigLeNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 18, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(18),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(18, 48, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(21168, 6350)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(6350, 4445)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4445, num_classes)

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

    def get_layers(self):
        output = "Model summary:\n"
        X = torch.rand(size=(1, 3, 96, 96), dtype=torch.float32)
        
        output += "layer1:\n"        
        for layer in self.layer1:
            X = layer(X)
            output += (layer.__class__.__name__ + 'output shape: \t' + str(X.shape) +'\n')

        output += "layer2:\n"
        for layer in self.layer2:
            X = layer(X)
            output += (layer.__class__.__name__ + 'output shape: \t' + str(X.shape) + '\n')

        return output
    
    def print_layers(self):
        print(model1.get_layers())


if __name__ == "__main__":
    model1 = Model1()
    model1.print_layers()

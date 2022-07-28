import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
from abc import ABC, abstractmethod

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = self.get_layers()

    @abstractmethod
    def get_layers(self):
        pass

    def forward(self, x):
        logits = self.layers(x)
        return logits 


class Small_LeNet(Model):
    
    def get_layers(self):
        return nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(7056, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1)
        )

class BigLeNet(Model):
    
    def get_layers(self):
        return nn.Sequential(
            nn.Conv2d(3, 18, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(18),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(18, 48, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(21168, 6350),
            nn.ReLU(),
            nn.Linear(6350, 4445),
            nn.ReLU(),
            nn.Linear(4445, 1)
        )    

class Tillus(Model):
    def __init__(self,s , c, f):
        super(Model, self).__init__()
        self.layers = self.get_layers(s=s, c=c, f=f)

    def get_layers(self, s, c, f):
        return nn.Sequential(
            nn.Dropout2d(p=c),
            nn.Conv2d(3, 32, kernel_size=11, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout2d(p=c),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout2d(p=c),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Dropout(p=f),
            nn.Linear(8192, s, bias=False),
            nn.BatchNorm1d(s),
            nn.ReLU(),
            nn.Dropout(p=f),
            nn.Linear(s, s, bias=False),
            nn.BatchNorm1d(s),
            nn.ReLU(),
            nn.Dropout(p=f),
            nn.Linear(s, 1)
        ) 

# inspired by: https://blog.ineuron.ai/AlexNet-CNN-architecture-With-Implementation-in-Keras-Q4strWr4iZ
class Alex_Net(Model):

    def get_layers(self):
        return nn.Sequential(
            nn.Conv2d(3,96,kernel_size=7,stride=4,padding=0, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            nn.Conv2d(96,256,kernel_size=5,padding=2, stride=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),

            nn.Conv2d(256,384,kernel_size=3,padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(384,384,kernel_size=3,padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(384,256,kernel_size=3,padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=0),

            nn.Flatten(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

class Big_Konrad(Model):
    def __init__(self,s , c, f):
        super(Model, self).__init__()
        print("->", self.__class__.__name__)
        self.layers = self.get_layers(s=s, c=c, f=f)

    def get_layers(self, s, c, f):
        return nn.Sequential(
            nn.Dropout2d(p=c),
            nn.Conv2d(3,128,kernel_size=7,padding='same', bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),

            nn.Dropout2d(p=c),
            nn.Conv2d(128,256,kernel_size=5,padding='same', bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),

            nn.Dropout2d(p=c),
            nn.Conv2d(256,512,kernel_size=3,padding='same', bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),

            nn.Dropout2d(p=c),
            nn.Conv2d(512,64,kernel_size=1,padding='same', bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            
            nn.Flatten(),
            nn.Dropout(p=f),
            nn.Linear(int(64*96/16*96/16), s),
            nn.BatchNorm1d(s),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=f),
            nn.Linear(s, 2*s),
            nn.BatchNorm1d(2*s),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=f),
            nn.Linear(2*s, 1)
        )

class Very_Big_Konrad(Model):

    def get_layers(self):
        return nn.Sequential(
            nn.Conv2d(3,128,kernel_size=7,padding='same'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),

            nn.Conv2d(128,256,kernel_size=5,padding='same'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),

            nn.Conv2d(256,512,kernel_size=3,padding='same'),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512,512,kernel_size=3,padding='same'),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512,512,kernel_size=3,padding='same'),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),

            nn.Conv2d(512,64,kernel_size=1,padding='same'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(int(64*96/16*96/16), 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1000, 400),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(400, 1)
        )

class VGG_16(Model):

    def get_layers(self):
        return nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding="same"),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,padding="same"),
            nn.ReLU(),            
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),

            nn.Conv2d(64,128,kernel_size=3,padding="same"),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,padding="same"),
            nn.ReLU(),         
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),

            nn.Conv2d(128,256,kernel_size=3,padding="same"),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,padding="same"),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,padding="same"),
            nn.ReLU(),         
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),

            nn.Conv2d(256,512,kernel_size=3,padding="same"),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,padding="same"),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,padding="same"),
            nn.ReLU(),         
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),

            nn.Conv2d(512,512,kernel_size=3,padding="same"),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,padding="same"),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,padding="same"),
            nn.ReLU(),         
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0),

           
            nn.Flatten(),
            nn.Linear(4608, 4608),
            nn.ReLU(),
            nn.Linear(4608, 1)
        )

class No_Conv(Model):
    
    def get_layers(self):
        return nn.Sequential(           
            nn.Flatten(),
            nn.Linear(27648, 9216),
            nn.ReLU(),
            nn.Linear(9216, 1)
        )   

if __name__ == "__main__":
    model = Small_LeNet()
    print(list(model.modules()))
    
    

    

from __future__ import print_function, division
from data import get_dl, get_ds
import zipfile
import torchvision
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
from kaggle.api.kaggle_api_extended import KaggleApi
from skimage import io, transform
import torch
import os
import pandas as pd
from torch import nn
import model
import data

#https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

def get_model(img_shape,normalize):
    return model.Classifier(img_shape,normalize)

def train(model,dataloader,batch_size=64,learning_rate=1e-3,epochs=5):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    train_loop(model,dataloader,loss_fn,optimizer)

def train_loop(model,dataloader,loss_fn,optimizer):

    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        y = y.view(-1,1).to(torch.float)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    

def run_classifier():
    batch_size = 64
    num_workers = 4
    dataloader = data.get_dl(batch_size=batch_size,num_workers=num_workers)
    model = get_model([3,96,96],True)
    train (model,dataloader,batch_size=batch_size,learning_rate=1e-3,epochs=5)


if __name__ == "__main__":
    run_classifier()
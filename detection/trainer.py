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
import wandb

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
            wandb.log({"loss":loss})
    

def run_classifier(trainer_config,model_config):
    wandb.init(project=trainer_config["project"])
    wandb.config = model_config
    dataloader,img_shape = data.get_dl(batch_size=model_config["batch_size"],num_workers=model_config["num_workers"])
    model = get_model(img_shape,True)
    wandb.watch(model)
    train (model,dataloader,batch_size=model_config["batch_size"],learning_rate=model_config["learning_rate"],epochs=model_config["max_epochs"])

# decreases logging for better performance! mostly relevant for small dsets
PERFORMANCE_MODE = False
PRJ = "histo_cancer"

MODEL_CONFIG = dict(
    batch_size = 64,
    num_workers = 4,
    learning_rate = 1e-3,
    max_epochs=100,
)

GPUS = 1
TRAINER_CONFIG = dict(
    project = PRJ,
    gpus=GPUS,
    strategy="ddp" if GPUS > 1 else None,
    fast_dev_run=False,
    log_every_n_steps=250 if PERFORMANCE_MODE else 100,
    accumulate_grad_batches=1
)

if __name__ == "__main__":
    run_classifier(TRAINER_CONFIG,MODEL_CONFIG)
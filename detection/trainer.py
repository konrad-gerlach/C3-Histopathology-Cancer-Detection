from __future__ import print_function, division
from data import get_dl, get_ds
from test import test_loop
from helper import predicted_lables
import zipfile
import torchvision
import os
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


# https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

def get_model(img_shape, normalize):
    return model.Model1()


def train(model, train_dataloader, test_dataloader, device, batch_size=64, learning_rate=1e-3, epochs=5):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    train_loop(model, train_dataloader, test_dataloader, loss_fn, optimizer, device, epochs)


def train_loop(model, train_dataloader, test_dataloader, loss_fn, optimizer, device, epochs):
    model = model.to(device)
    size = len(train_dataloader.dataset)

    #to be set
    batch_limit=20
    batch_limit -= 1

    for epoch in range(epochs):
        acc_accum = 0
        train_iter = enumerate(train_dataloader)
        train_epoch_loss = 0        

        for batch, (X, y) in train_iter:
            # Compute prediction and loss
            X = X.to(device)
            y = y.to(device)
            y = y.view(-1, 1).to(torch.float)

            pred = model(X)
            loss = loss_fn(pred, y) #does this work without sigmoid on pred?
            train_epoch_loss += loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = predicted_lables(pred) 
            acc_accum += (pred == y).sum()       

            print(str(batch), end='\r')


            # loss and accuracy for some batches
            if batch % 5 == 0:
                loss  = loss.item()
                current = batch * len(X)
                train_acc = acc_accum / (current + len(X))
                print(f"train loss: {loss:>7f} train accuracy: {train_acc:>7f} [{current:>5d}/{size:>5d}]")
                wandb.log({"loss": loss})
                wandb.log({"train accuracy per batch": train_acc})

            if batch == batch_limit:
                break
        
        # loss while training
        train_epoch_loss /= batch_limit
        wandb.log({"train loss per epoch": train_epoch_loss})
        print('epoch {}, train loss {}'.format(epoch+1,  train_epoch_loss))

        test_loop(model, test_dataloader, loss_fn, device, epoch)


def run_classifier(trainer_config, model_config):
    wandb.init(project=trainer_config["project"], entity="histo-cancer-detection")
    wandb.config = model_config
    train_dataloader, test_dataloader, img_shape = data.get_dl(batch_size=model_config["batch_size"], num_workers=model_config["num_workers"])
    model = get_model(img_shape, True)
    wandb.watch(model)
    print(trainer_config["device"])
    train(model, train_dataloader, test_dataloader, trainer_config["device"], batch_size=model_config["batch_size"],
          learning_rate=model_config["learning_rate"], epochs=model_config["max_epochs"])


# decreases logging for better performance! mostly relevant for small dsets
PERFORMANCE_MODE = False
PRJ = "histo_cancer"

MODEL_CONFIG = dict(
    batch_size=64,
    num_workers=4,
    learning_rate=0.01,
    max_epochs=10
)

GPUS = 1
TRAINER_CONFIG = dict(
    project=PRJ,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
)

if __name__ == "__main__":
    run_classifier(TRAINER_CONFIG, MODEL_CONFIG)

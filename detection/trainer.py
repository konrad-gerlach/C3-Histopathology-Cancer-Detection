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
import config


# https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

def get_model(img_shape, normalize):
    return model.Model1()

def log_model(conv_model):
    columns = ["Text"]
    data = []

    lines = conv_model.get_layers().split("\n")
    for line in lines:
        data.append([line])

    table =  wandb.Table(data=data, columns=columns)
    wandb.log({"The model we use": table}) 



def train(model, train_dataloader, test_dataloader, optimizer_config, device, learning_rate=1e-3, epochs=5):
    loss_fn = nn.BCEWithLogitsLoss()
    if optimizer_config["use_optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=optimizer_config["betas"], eps=optimizer_config["eps"], weight_decay=optimizer_config["weight_decay"], amsgrad=optimizer_config["amsgrad"])
    elif optimizer_config["use_optimizer"] == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, rho=optimizer_config["rho"], eps=optimizer_config["eps"], weight_decay=optimizer_config["weight_decay"])
    elif optimizer_config["use_optimizer"] == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, lr_decay=optimizer_config["lr_decay"], weight_decay=optimizer_config["weight_decay"])
    elif optimizer_config["use_optimizer"] == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=optimizer_config["alpha"], eps=optimizer_config["eps"], weight_decay=optimizer_config["weight_decay"], momentum=optimizer_config["momentum"])
    elif optimizer_config["use_optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=optimizer_config["momentum"], weight_decay=optimizer_config["weight_decay"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    print("You are currently using the optimizer: {}".format(optimizer))
    train_loop(model, train_dataloader, test_dataloader, loss_fn, optimizer, device, epochs)


def train_loop(model, train_dataloader, test_dataloader, loss_fn, optimizer, device, epochs):
    model = model.to(device)
    size = len(train_dataloader.dataset)

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
            loss = loss_fn(pred, y)
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
        
        # loss while training
        train_epoch_loss /= batch + 1
        wandb.log({"train loss per epoch": train_epoch_loss})
        print('epoch {}, train loss {}'.format(epoch+1,  train_epoch_loss))

        test_loop(model, test_dataloader, loss_fn, device, epoch)


def run_classifier(trainer_config, model_config, optimizer_config):
    wandb.init(project=trainer_config["project"], entity="histo-cancer-detection")
    wandb.config = model_config
    train_dataloader, test_dataloader, img_shape = data.get_dl(batch_size=model_config["batch_size"], num_workers=model_config["num_workers"])
    model = get_model(img_shape, True)
    log_model(model)
    wandb.watch(model)
    print(trainer_config["device"])
    train(model, train_dataloader, test_dataloader, optimizer_config, trainer_config["device"],
          learning_rate=model_config["learning_rate"], epochs=model_config["max_epochs"])


# decreases logging for better performance! mostly relevant for small dsets
PERFORMANCE_MODE = False

GPUS = 1


if __name__ == "__main__":
    run_classifier(config.TRAINER_CONFIG, config.MODEL_CONFIG, config.OPTIMIZER_CONFIG)

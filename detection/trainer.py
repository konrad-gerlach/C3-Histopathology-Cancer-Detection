from __future__ import print_function, division
from calendar import c
import logging
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
import test


# https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
def get_model(img_shape, normalize, fc_layer_size, conv_dropout, fully_dropout):
    return model.Big_Konrad(fc_layer_size, conv_dropout, fully_dropout)
    

def log_metadata(model, model_config, optimizer):
    lines = str(optimizer).split("\n")
    logging_config = dict(
        batch_size= model_config["batch_size"],
        learning_rate= model_config["learning_rate"],
        max_epochs= model_config["max_epochs"],
        train_portion=config.DATA_CONFIG["train_portion"],
        test_portion=config.DATA_CONFIG["test_portion"],
        optimizer= lines[0].split(" ")[0],
        optimizer_parameters= lines[1:-1]
    )
    return logging_config


def choose_optimizer(optimizer_config, parameters, gradient_accumulation, learning_rate=1e-3, ):
    use_optimizer= optimizer_config["use_optimizer"].lower()
    learning_rate = learning_rate / gradient_accumulation
    if use_optimizer == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, betas=optimizer_config["betas"], eps=optimizer_config["eps"], weight_decay=optimizer_config["weight_decay"], amsgrad=optimizer_config["amsgrad"])
    elif use_optimizer == "adadelta":
        return torch.optim.Adadelta(parameters, lr=learning_rate, rho=optimizer_config["rho"], eps=optimizer_config["eps"], weight_decay=optimizer_config["weight_decay"])
    elif use_optimizer == "adagrad":
        return torch.optim.Adagrad(parameters, lr=learning_rate, lr_decay=optimizer_config["lr_decay"], weight_decay=optimizer_config["weight_decay"])
    elif use_optimizer == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=learning_rate, alpha=optimizer_config["alpha"], eps=optimizer_config["eps"], weight_decay=optimizer_config["weight_decay"], momentum=optimizer_config["momentum"])
    elif use_optimizer == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=optimizer_config["momentum"], weight_decay=optimizer_config["weight_decay"])
       
    else:
        return torch.optim.SGD(parameters, lr=learning_rate)

def train(model, train_dataloader, test_dataloader, optimizer, device, gradient_accumulation,epochs=5):
    loss_fn = nn.BCEWithLogitsLoss()
    train_loop(model, train_dataloader, test_dataloader, loss_fn, optimizer, device, epochs,gradient_accumulation)


def train_loop(model, train_dataloader, test_dataloader, loss_fn, optimizer, device, epochs,gradient_accumulation=1):
    model = model.to(device)
    size = len(train_dataloader.dataset)

    for epoch in range(epochs):
        acc_accum = 0
        train_iter = enumerate(train_dataloader)
        train_epoch_loss = 0        
        model.train()
        wandb.log({"epoch": epoch})
        for batch, (X, y) in train_iter:
            # Compute prediction and loss
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y = y.view(-1, 1).to(torch.float)

            pred = model(X)
            loss = loss_fn(pred, y)
            # fix oom https://pytorch.org/docs/stable/notes/faq.html
            train_epoch_loss += float(loss)

            # Backpropagation with gradient accumulation
            loss.backward()
            if batch % gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
            

            pred = predicted_lables(pred)
            batch_acc_accum = float((pred == y).sum())
            acc_accum += float(batch_acc_accum)
            

            print(str(batch), end='\r')


            # loss and accuracy for some batches
            if batch % 100 == 0:
                loss  = loss.item()
                current = batch * len(X)
                batch_acc = batch_acc_accum / len(X)
                train_acc = acc_accum / (current + len(X))
                print(f"train loss: {loss:>7f} train accuracy: {train_acc:>7f} [{current:>5d}/{size:>5d}]")
                wandb.log({"loss": loss})
                wandb.log({"train accuracy per batch":batch_acc})
                wandb.log({"train accuracy rolling avg per epoch": train_acc})
        
        # loss while training
        train_epoch_loss /= batch + 1
        wandb.log({"train loss per epoch": train_epoch_loss})
        print('epoch {}, train loss {}'.format(epoch+1,  train_epoch_loss))

        test.test_loop(model, test_dataloader, loss_fn, device, epoch)

def introduce_sweep():
    sweep_config = {'method': 'random'}

    metric = {
    'name': 'test accuracy per epoch',
    'goal': 'maximize'   
    }
    sweep_config['metric'] = metric

    parameters_dict = {
    'optimizer': {
        'values': ['adam', 'sgd', 'rmspropp']
        },
    'fc_layer_size': {
        'values': [128, 256, 512]
        },
    'fully_dropout': {
          'values': [0.4, 0.5, 0.6]
        },
    'conv_dropout': {
        'values': [0.1, 0.2, 0.3]
    },
    'batch_size': {
          'values': [4, 16, 64]
        },
    'learning_rate': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
      },
    'weight_decay': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
      },
    }
    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project="histo_cancer")

    wandb.agent(sweep_id, run_classifier, count=1000)

def run_classifier(sweep_config=None):

    trainer_config = config.TRAINER_CONFIG
    
    with wandb.init(project=trainer_config["project"], entity="histo-cancer-detection", config=sweep_config):
        sweep_config=wandb.config        
        model_config = config.MODEL_CONFIG
        model_config["lr"]=sweep_config.learning_rate
        model_config["batch_size"]=sweep_config.batch_size
        optimizer_config = config.OPTIMIZER_CONFIG
        optimizer_config["use_optimizer"]=sweep_config.optimizer
        optimizer_config["weight_decay"]=sweep_config.weight_decay
        train_dataloader, test_dataloader, img_shape = data.get_dl(batch_size=model_config["batch_size"], num_workers=model_config["num_workers"])
        model = get_model(img_shape, True, fc_layer_size=sweep_config.fc_layer_size, conv_dropout=sweep_config.conv_dropout, fully_dropout=sweep_config.fully_dropout)
        optimizer=choose_optimizer(optimizer_config, model.parameters(), model_config["gradient_accumulation"], learning_rate=model_config["lr"])
        #logging_config = log_metadata(model, model_config, optimizer)

        #wandb.config = model_config
        
        
        #wandb.watch(model, criterion=None, log="gradients", log_freq=1000, idx=None, log_graph=(True))
    


        print("You are currently using the optimizer: {}".format(optimizer))
        print(trainer_config["device"])

        train(model, train_dataloader, test_dataloader, optimizer, trainer_config["device"], model_config["gradient_accumulation"], epochs=model_config["max_epochs"])
        #wandb.finish()


# decreases logging for better performance! mostly relevant for small dsets
PERFORMANCE_MODE = False

GPUS = 1


if __name__ == "__main__":
    introduce_sweep()
    run_classifier(config.TRAINER_CONFIG, config.MODEL_CONFIG, config.OPTIMIZER_CONFIG)

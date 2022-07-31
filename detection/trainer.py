from __future__ import print_function, division
import argparse
from calendar import c
from cmath import log
from distutils.file_util import copy_file
import logging
from data import get_dl, get_ds
from test import test_loop
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
import helper


# https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
def get_model(img_shape, normalize, fc_layer_size=config.SP_MODEL_CONFIG["fc_layer_size"], conv_dropout=config.SP_MODEL_CONFIG["conv_dropout"], fully_dropout=config.SP_MODEL_CONFIG["fully_dropout"]):
    #first parameters?
    #insert values from SP_MODEL_CONFIG here if necessary
    return model.Big_Konrad(fc_layer_size,conv_dropout,fully_dropout)

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
            

            pred = helper.predicted_lables(pred)
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

def classifier():
    trainer_config = config.TRAINER_CONFIG
    wandb.init(project=trainer_config["project"], entity=trainer_config["entity"])
    run_classifier

def run_classifier():
    trainer_config = config.TRAINER_CONFIG
    model_config = config.MODEL_CONFIG
    optimizer_config = config.OPTIMIZER_CONFIG

    train_dataloader, test_dataloader, img_shape = data.get_dl(batch_size=model_config["batch_size"], num_workers=model_config["num_workers"])
    model = get_model(img_shape, True)
    optimizer = helper.choose_optimizer(optimizer_config, model.parameters(), model_config["gradient_accumulation"], learning_rate=model_config["lr"])
    logging_config = helper.log_metadata(model_config, optimizer)
 
    #wandb.init(project=trainer_config["project"], entity="histo-cancer-detection", config=logging_config)
    wandb.config = logging_config
   
    wandb.watch(model, criterion=None, log="gradients", log_freq=1000, idx=None, log_graph=(True))


    print("You are currently using the optimizer: {}".format(optimizer))
    print(trainer_config["device"])

    train(model, train_dataloader, test_dataloader, optimizer, trainer_config["device"], model_config["gradient_accumulation"], epochs=model_config["max_epochs"])
    wandb.finish()

    """
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
        optimizer=helper.choose_optimizer(optimizer_config, model.parameters(), model_config["gradient_accumulation"], learning_rate=model_config["lr"])
        #logging_config = log_metadata(model, model_config, optimizer)

        #wandb.config = model_config
        
        
        #wandb.watch(model, criterion=None, log="gradients", log_freq=1000, idx=None, log_graph=(True))
    


        print("You are currently using the optimizer: {}".format(optimizer))
        print(trainer_config["device"])

        train(model, train_dataloader, test_dataloader, optimizer, trainer_config["device"], model_config["gradient_accumulation"], epochs=model_config["max_epochs"])
        #wandb.finish()
        """


# decreases logging for better performance! mostly relevant for small dsets
PERFORMANCE_MODE = False

GPUS = 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='configure project')
    parser.add_argument('--ds_path', default=config.DATA_CONFIG["ds_path"],
                        help='the location where the dataset is or should be located')

    args = parser.parse_args()
    config.DATA_CONFIG["ds_path"] = args.ds_path
    print(config.DATA_CONFIG["ds_path"])
    classifier()

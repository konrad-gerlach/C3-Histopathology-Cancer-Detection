from __future__ import print_function, division
import argparse
from calendar import c
from cmath import log
from curses.ascii import SP
import logging
from data import get_dl, get_ds
import zipfile
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import torch
import os
import pandas as pd
from torch import nn
import feature_visualization
import model
import data
import wandb
import config
import test
import helper
import generic_train_loop

# https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
def get_model():
    #first parameters?
    #insert values from SP_MODEL_CONFIG here if necessary
    return config.MODEL_CONFIG["model_class"](**config.SP_MODEL_CONFIG)

def train(model, train_dataloader, test_dataloader, optimizer, device, gradient_accumulation,epochs=5):
    loss_fn = training_loss_function
    train_loop(model, train_dataloader, test_dataloader, loss_fn, optimizer, device, epochs, gradient_accumulation)

def training_loss_function(outputs,y):
    return nn.BCEWithLogitsLoss()(outputs[-1],y)

def training_logger(outputs,loss,batch,X,y,inputs):
    inputs["train_epoch_loss"] += float(loss)
    
    pred = outputs[-1]
    pred = helper.predicted_lables(pred)
    
    batch_acc_accum = float((pred == y).sum())
    inputs["acc_accum"] += float(batch_acc_accum)

    print(str(batch), end='\r')
    size = inputs["size"]

    # loss and accuracy for some batches
    if batch % 100 == 0:
        loss  = loss.item()
        current = batch * len(X)
        batch_acc = batch_acc_accum / len(X)
        train_acc = inputs["acc_accum"] / (current + len(X))
        print(f"train loss: {loss:>7f} train accuracy: {train_acc:>7f} [{current:>5d}/{size:>5d}]")
        wandb.log({"loss": loss})
        wandb.log({"train accuracy per batch":batch_acc})
        wandb.log({"train accuracy rolling avg per epoch": train_acc})
    
    return inputs


def train_loop(model, train_dataloader, test_dataloader, loss_fn, optimizer, device, epochs,gradient_accumulation=1):
    model = model.to(device)
    for epoch in range(epochs):
        train_iter = enumerate(train_dataloader)
        inputs = dict(
            acc_accum = 0,
            train_epoch_loss = 0,
            size = len(train_dataloader.dataset)
        )
       
        model.train()
        wandb.log({"epoch": epoch})
        
        for batch, (X, y) in train_iter:
            inputs = generic_train_loop.train_loop(batch,X,y,device,model,loss_fn,gradient_accumulation,optimizer,training_logger,inputs)
        
        # loss while training
        inputs["train_epoch_loss"] /= len(train_iter)
        wandb.log({"train loss per epoch": inputs["train_epoch_loss"]})
        print('epoch {}, train loss {}'.format(epoch+1,  inputs["train_epoch_loss"]))

        test_loss_epoch, epoch_acc = test.test_loop(model, test_dataloader, loss_fn, device, epoch)
        if epoch_acc > config.TRAINER_CONFIG["accuracy_goal"]:
            return

def classifier():
    trainer_config = config.TRAINER_CONFIG
    continue_training = config.TRAINER_CONFIG["continue_training"]
    if continue_training:
        job_type = "resume_training_classifier"
    else:
        job_type = "train_classifier"
    wandb.config = {}
    run = wandb.init(project=trainer_config["project"], entity=trainer_config["entity"], job_type=job_type)
    run_classifier(run,continue_training)


def run_classifier(run,continue_training):
    trainer_config = config.TRAINER_CONFIG
    model_config = config.MODEL_CONFIG
    optimizer_config = config.OPTIMIZER_CONFIG

    train_dataloader, test_dataloader, img_shape = data.get_dl(batch_size=model_config["batch_size"], num_workers=model_config["num_workers"])
    if continue_training:
        model = helper.load_model(run)
    else:
        model = get_model()

    optimizer = helper.choose_optimizer(optimizer_config, model.parameters(), model_config["gradient_accumulation"], learning_rate=model_config["lr"])
    logging_config = helper.log_metadata(model_config, optimizer)
 
    #wandb.init(project=trainer_config["project"], entity="histo-cancer-detection", config=logging_config)
    wandb.config.update(logging_config)
   
    wandb.watch(model, criterion=None, log="gradients", log_freq=1000, idx=None, log_graph=(True))


    print("You are currently using the optimizer: {}".format(optimizer))
    print(trainer_config["device"])

    train(model, train_dataloader, test_dataloader, optimizer, trainer_config["device"], model_config["gradient_accumulation"], epochs=model_config["max_epochs"])
    helper.log_model(run,model,optimizer)
    
    wandb.finish()


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
    feature_visualization.visualizer()

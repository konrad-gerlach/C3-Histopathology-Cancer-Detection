from __future__ import print_function, division
import torch
from torch import nn
import data
import wandb
import config
import test
import helper
import generic_train_loop

# https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
def get_model():
    # first parameters?
    # insert values from SP_MODEL_CONFIG here if necessary
    return config.MODEL_CONFIG["model_class"](**config.SP_MODEL_CONFIG)

def run_trainer(run=None):
    model_config = config.MODEL_CONFIG
    optimizer_config = config.OPTIMIZER_CONFIG
    trainer_config = config.TRAINER_CONFIG
    wandb_config = config.WANDB_CONFIG
    continue_training = trainer_config["continue_training"]
    gradient_accumulation=trainer_config["gradient_accumulation"]
    loss_fn = training_loss_function

    helper.define_dataset_location()
    if run is None:
        run = setup_wandb(wandb_config)

    train_dataloader, test_dataloader, model, optimizer = setup_training(optimizer_config, model_config, trainer_config,
                                                                        run, continue_training, gradient_accumulation)

    log_with_wandb(trainer_config, optimizer_config, model, optimizer)

    train_loop(model, train_dataloader, test_dataloader, loss_fn, optimizer, trainer_config["device"],
                trainer_config["max_epochs"], gradient_accumulation)

    finish__training(run, model, optimizer)


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


def train_loop(model, train_dataloader, test_dataloader, loss_fn, optimizer, device, epochs, gradient_accumulation=1):
    model = model.to(device)
    for epoch in range(epochs):
        inputs = dict(
            acc_accum = 0,
            train_epoch_loss = 0,
            size = len(train_dataloader.dataset)
        )
       
        model.train()
        wandb.log({"epoch": epoch})
        
        for batch, (X, y) in enumerate(train_dataloader):
            inputs = generic_train_loop.train_loop(batch,X,y,device,model,loss_fn,gradient_accumulation,optimizer,training_logger,inputs)
        
        # loss while training
        inputs["train_epoch_loss"] /= len(train_dataloader.dataset)
        wandb.log({"train loss per epoch": inputs["train_epoch_loss"]})
        print('epoch {}, train loss {}'.format(epoch+1,  inputs["train_epoch_loss"]))

        epoch_acc = test.test_loop(model, test_dataloader, loss_fn, device, epoch)
        if epoch_acc > config.TRAINER_CONFIG["accuracy_goal"]:
            return


def setup_wandb(wandb_config):
    wandb.config = wandb_config
    return wandb.init(project=wandb_config["project"], entity=wandb_config["entity"])


def setup_training(optimizer_config, model_config, trainer_config, run, continue_training, gradient_accumulation):
    model = get_model()
    if continue_training:
        model.load_state_dict(torch.load(trainer_config["model_path"]))
    model = model.to(trainer_config["device"])
    optimizer = optimizer_config["optimizer_class"](model.parameters(), **optimizer_config["optimizer_config"])
    train_dataloader = data.get_dataloader(data.TRAIN_DATASET, trainer_config["batch_size"], trainer_config["device"])
    test_dataloader = data.get_dataloader(data.TEST_DATASET, trainer_config["batch_size"], trainer_config["device"])
    return train_dataloader, test_dataloader, model, optimizer


def log_with_wandb(trainer_config, optimizer_config, model, optimizer):
    logging_config = helper.log_metadata(trainer_config, optimizer_config, optimizer)

    wandb.config.update(logging_config)

    wandb.watch(model, criterion=None, log="gradients", log_freq=1000, idx=None, log_graph=(True))

    print("You are currently using the optimizer: {}".format(optimizer))
    print(trainer_config["device"])


def finish__training(run, model, optimizer):
    helper.log_model(run, model)
    wandb.finish()


if __name__ == "__main__":
    helper.define_dataset_location()
    run_trainer()

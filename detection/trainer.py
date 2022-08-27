from __future__ import print_function, division
import torch
from torch import nn
import data
import wandb
import config
import test
import helper
import generic_train_loop

def get_model():
    return config.MODEL_CONFIG["model_class"](**config.SP_MODEL_CONFIG)

def run_trainer(run=None, continue_training=False):
    optimizer_config = config.OPTIMIZER_CONFIG
    trainer_config = config.TRAINER_CONFIG
    wandb_config = config.WANDB_CONFIG
    gradient_accumulation=trainer_config["gradient_accumulation"]

    helper.define_dataset_location()
    if run is None:
        run = setup_wandb(wandb_config)
        continue_training = trainer_config["continue_training"]

    train_dataloader, test_dataloader, model, optimizer = setup_training(optimizer_config, run, continue_training, gradient_accumulation)

    log_with_wandb(trainer_config, optimizer_config, model, optimizer)

    train_loop(model, train_dataloader, test_dataloader, optimizer, trainer_config["device"], trainer_config["max_epochs"], gradient_accumulation)

    finish__training(run, model)


def setup_wandb(wandb_config):
    wandb.config = {}
    return wandb.init(project=wandb_config["project"], entity=wandb_config["entity"], job_type=helper.job_type_of_training())


def setup_training(optimizer_config, run, continue_training, gradient_accumulation):
    train_dataloader, test_dataloader, __ = data.get_dl(batch_size=optimizer_config["batch_size"])
    if continue_training:
        model = helper.load_model(run)
    else:
        model = config.MODEL_CONFIG["model_class"](**config.SP_MODEL_CONFIG)

    optimizer = helper.choose_optimizer(optimizer_config, model.parameters(), learning_rate=optimizer_config["lr"] / gradient_accumulation)

    return train_dataloader, test_dataloader, model, optimizer


def log_with_wandb(trainer_config, optimizer_config, model, optimizer):
    logging_config = helper.log_metadata(trainer_config, optimizer_config, optimizer)

    wandb.config.update(logging_config)

    wandb.watch(model, criterion=None, log="gradients", log_freq=1000, idx=None, log_graph=(True))

    print("You are currently using the optimizer: {}".format(optimizer))
    print(trainer_config["device"])


def training_logger(outputs, loss, batch, X, y, metrics):
    metrics["train_epoch_loss"] += float(loss)
    
    pred = outputs[-1]
    pred = helper.predicted_lables(pred)
    
    batch_acc_accum = float((pred == y).sum())
    metrics["acc_accum"] += float(batch_acc_accum)

    print(str(batch), end='\r')
    size = metrics["size"]

    # loss and accuracy for some batches
    if batch % 100 == 0:
        loss  = loss.item()
        current = batch * len(X)
        batch_acc = batch_acc_accum / len(X)
        train_acc = metrics["acc_accum"] / (current + len(X))
        print(f"train loss: {loss:>7f} train accuracy: {train_acc:>7f} [{current:>5d}/{size:>5d}]")
        wandb.log({"loss": loss})
        wandb.log({"train accuracy per batch":batch_acc})
        wandb.log({"train accuracy rolling avg per epoch": train_acc})
    
    return metrics


def train_loop(model, train_dataloader, test_dataloader, optimizer, device, epochs, gradient_accumulation=1):
    model = model.to(device)
    for epoch in range(epochs):
        metrics = dict(
            acc_accum = 0,
            train_epoch_loss = 0,
            size = len(train_dataloader.dataset)
        )
       
        model.train()
        wandb.log({"epoch": epoch})
        
        for batch, (X, y) in enumerate(train_dataloader):
            metrics = generic_train_loop.train_loop(X, y, device, model, training_logger, metrics, gradient_accumulation, optimizer, batch)
        
        # loss while training
        metrics["train_epoch_loss"] /= len(train_dataloader.dataset)
        wandb.log({"train loss per epoch": metrics["train_epoch_loss"]})
        print('epoch {}, train loss {}'.format(epoch+1,  metrics["train_epoch_loss"]))

        epoch_acc = test.test_loop(model, test_dataloader, device, epoch)
        if epoch_acc > config.TRAINER_CONFIG["accuracy_goal"]:
            return


def finish__training(run, model):
    helper.log_model(run, model)
    wandb.finish()


if __name__ == "__main__":
    helper.define_dataset_location()
    run_trainer()

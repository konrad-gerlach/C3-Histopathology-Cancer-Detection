from __future__ import print_function, division
import torch
from torch import nn
import data
import wandb
import config
import test
import helper


# https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

def train(run=None):
    model_config = config.MODEL_CONFIG
    optimizer_config = config.OPTIMIZER_CONFIG
    trainer_config = config.TRAINER_CONFIG
    wandb_config = config.WANDB_CONFIG
    continue_training = trainer_config["continue_training"]
    gradient_accumulation=trainer_config["gradient_accumulation"]
    loss_fn = nn.BCEWithLogitsLoss()

    helper.define_dataset_location()
    if run is None:
        run = setup_wandb(wandb_config)

    train_dataloader, test_dataloader, model, optimizer = setup_training(optimizer_config, model_config, trainer_config,
                                                                        run, continue_training, gradient_accumulation)

    log_with_wandb(trainer_config, optimizer_config, model, optimizer)

    train_loop(model, train_dataloader, test_dataloader, loss_fn, optimizer, trainer_config["device"],
                trainer_config["max_epochs"], gradient_accumulation)

    finish__training(run, model, optimizer)


def train_loop(model, train_dataloader, test_dataloader, loss_fn, optimizer, device, epochs, gradient_accumulation=1):
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

            log_batch_progress(loss, batch, pred, acc_accum, size, X, y)

        train_epoch_loss /= batch + 1
        log_epoch_progress(train_epoch_loss, epoch)

        __, epoch_acc = test.test_loop(model, test_dataloader, loss_fn, device, epoch)
        if epoch_acc > config.TRAINER_CONFIG["accuracy_goal"]:
            return


def log_batch_progress(loss, batch, pred, acc_accum, size, X, y):
    print(str(batch), end='\r')

    # loss and accuracy for some batches
    if batch % 100 == 0:
        pred = helper.predicted_lables(pred)
        batch_acc_accum = float((pred == y).sum())
        acc_accum += float(batch_acc_accum)

        loss = loss.item()
        current = batch * len(X)
        batch_acc = batch_acc_accum / len(X)
        train_acc = acc_accum / (current + len(X))
        print(f"train loss: {loss:>7f} train accuracy: {train_acc:>7f} [{current:>5d}/{size:>5d}]")
        wandb.log({"loss": loss})
        wandb.log({"train accuracy per batch": batch_acc})
        wandb.log({"train accuracy rolling avg per epoch": train_acc})


def log_epoch_progress(train_epoch_loss, epoch):
    # loss while training
    wandb.log({"train loss per epoch": train_epoch_loss})
    print('epoch {}, train loss {}'.format(epoch + 1, train_epoch_loss))


def setup_wandb(wandb_config):
    wandb.config = {}
    return wandb.init(project=wandb_config["project"], entity=wandb_config["entity"], job_type=helper.job_type_of_training())


def setup_training(optimizer_config, model_config, trainer_config, run, continue_training, gradient_accumulation):

    train_dataloader, test_dataloader, __ = data.get_dl(batch_size=optimizer_config["batch_size"],
                                                               num_workers=model_config["num_workers"])
    if continue_training:
        model = helper.load_model(run)
    else:
        model = config.MODEL_CONFIG["model_class"](**config.SP_MODEL_CONFIG)

    optimizer = helper.choose_optimizer(optimizer_config, model.parameters(),
                                        learning_rate=optimizer_config["lr"] / gradient_accumulation)

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
    train()

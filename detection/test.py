import torch
import helper
import wandb
import generic_train_loop


def test_loop(model, test_dataloader, device, epoch):
    model.eval()
    metrics = dict(
            test_loss_epoch = 0.0,
            epoch_acc = 0,
            correct_pred = 0,
            n = 0
        )

    with torch.no_grad():
        test_iter = enumerate(test_dataloader)
        for _, (X_test, y_test) in test_iter:
            metrics = generic_train_loop.train_loop(X_test, y_test, device, model, test_logger, metrics)

        metrics["test_loss_epoch"] /= len(test_dataloader)
        metrics["epoch_acc"] = metrics["correct_pred"] / metrics["n"]

    wandb.log({"test loss per epoch": metrics["test_loss_epoch"]})
    wandb.log({"test accuracy per epoch": metrics["epoch_acc"]})
    print('epoch {}, test loss {}, accuracy {}'.format(epoch + 1, metrics["test_loss_epoch"], metrics["epoch_acc"]))
    return metrics["epoch_acc"]

# batch and X are not used in this function but are needed for the generic_train_loop
def test_logger(outputs, loss, batch, X, y_test, metrics):
    pred = outputs[-1]
    metrics["test_loss_epoch"] += float(loss)
    pred_lables_test = helper.predicted_lables(pred)
    metrics["n"] += len(y_test)
    metrics["correct_pred"] += float((pred_lables_test == y_test).sum())

    return metrics

from turtle import pos
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
            num = 0,
            #naming convention following https://en.wikipedia.org/wiki/Precision_and_recall (Aug 2022)
            tp=0,
            fp=0,
            tn=0,
            fn=0,
            p=0,
            n=0,
            pp = 0,
            nn = 0,
        )

    with torch.no_grad():
        test_iter = enumerate(test_dataloader)
        for _, (X_test, y_test) in test_iter:
            metrics = generic_train_loop.train_loop(X_test, y_test, device, model, test_logger, metrics)

        metrics["test_loss_epoch"] /= len(test_dataloader)
        metrics["epoch_acc"] = metrics["correct_pred"] / metrics["num"]

    wandb.log({"test loss per epoch": metrics["test_loss_epoch"]})
    wandb.log({"test accuracy per epoch": metrics["epoch_acc"]})
    wandb.log(metrics)
    wandb.log({"test recall/sensitivity": metrics["tp"]/metrics["p"]})
    wandb.log({"test accuracy validation": (metrics["tp"]+metrics["tn"])/(metrics["p"]+metrics["n"])})
    wandb.log({"test precision": metrics["tp"]/metrics["pp"]})
    wandb.log({"test specificity": metrics["tn"]/metrics["n"]})
    wandb.log({"test F1score": (2*metrics["tp"])/(2*metrics["tp"]+metrics["fp"]+metrics["fn"])})
    print('epoch {}, test loss {}, accuracy {}'.format(epoch + 1, metrics["test_loss_epoch"], metrics["epoch_acc"]))
    return metrics["epoch_acc"]

# batch and X are not used in this function but are needed for the generic_train_loop
def test_logger(outputs, loss, batch, X, y_test, metrics):
    pred = outputs[-1]
    metrics["test_loss_epoch"] += float(loss)
    pred_lables_test = helper.predicted_lables(pred)
    metrics["num"] += len(y_test)
    metrics["correct_pred"] += float((pred_lables_test == y_test).sum())

    pred_lables_test = pred_lables_test.flatten()
    y_test = y_test.flatten()

    positive_indices = (pred_lables_test == 1.0).nonzero().flatten()
    negative_indices = (pred_lables_test == 0.0).nonzero().flatten()


    metrics["tp"] += int((1.0 == y_test.index_select(0,positive_indices)).sum())
    metrics["fp"] += int((0.0 == y_test.index_select(0,positive_indices)).sum())
    metrics["tn"] += int((0.0 == y_test.index_select(0,negative_indices)).sum())
    metrics["fn"] += int((1.0 == y_test.index_select(0,negative_indices)).sum())
    metrics["p"] += int((y_test == 1.0).sum())
    metrics["n"] += int((y_test == 0.0).sum())
    metrics["pp"] += len(positive_indices)
    metrics["nn"] += len(negative_indices)
    return metrics

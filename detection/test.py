import torch
import helper
import wandb

from detection import data, config


def test_loop(model, test_dataloader, loss_fn, device, epoch):
    # test loss and accuracy
    # https://colab.research.google.com/drive/1LHDUxiuG1niSOTiK9vRBlX47403OI15H?authuser=1#scrollTo=yoek_AxoQiNg
    model.eval()
    correct_pred = 0
    n = 0

    # TODO Refactor to use generic training loop instead
    with torch.no_grad():
        test_loss_epoch = 0.0
        test_iter = enumerate(test_dataloader)
        for _, (X_test, y_test) in test_iter:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            y_test = y_test.view(-1, 1).to(torch.float)

            pred_test = model(X_test)

            batch_loss = loss_fn([model(X_test)], y_test)
            test_loss_epoch += float(batch_loss)

            pred_lables_test = helper.predicted_lables(pred_test)
            n += len(y_test)
            correct_pred += float((pred_lables_test == y_test).sum())

        test_loss_epoch /= len(test_dataloader)
        epoch_acc = correct_pred / n

    wandb.log({"test loss per epoch": test_loss_epoch})
    wandb.log({"test accuracy per epoch": epoch_acc})
    print('epoch {}, test loss {}, accuracy {}'.format(epoch + 1, test_loss_epoch, epoch_acc))
    return test_loss_epoch, epoch_acc


if __name__ == "__main__":
    config.LOAD_CONFIG["alias"] = "usable-colored-with-metrics"
    config.DATA_CONFIG["grayscale"] = False
    wandb_config = config.WANDB_CONFIG
    job_type = "precision-recall"
    run = wandb.init(project=wandb_config["project"], entity=wandb_config["entity"], job_type=job_type)
    model = helper.load_model(run)
    device = config.TRAINER_CONFIG["device"]
    model = model.to(device)
    model.eval()

    _, data_loader, _ = data.get_dl(batch_size=32, num_workers=0)
    with torch.no_grad():
        import csv

        f = open('./submission3.csv', 'w')

        writer = csv.writer(f)
        header = ['id', 'label']
        writer.writerow(header)

        for batch, (X, y, names) in enumerate(data_loader):
            output = model(X)
            output = helper.predicted_lables(output)
            for i, x in enumerate(X):
                writer.writerow([names[i], output[i][0].item()])
                #print(i, int(output[i][0]), names[i])
            print(batch)
        f.close()
        wandb.finish()

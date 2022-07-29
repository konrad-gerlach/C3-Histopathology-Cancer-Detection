import torch
import config

def predicted_lables(pred):
    pred = torch.sigmoid(pred)
    pred = torch.round(pred, decimals=0)
    return pred

def choose_optimizer(optimizer_config, parameters, gradient_accumulation, learning_rate=1e-3):
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
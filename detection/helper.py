import os
import torch
import wandb
import config
from config import MODEL_CONFIG
from detection.config import SP_MODEL_CONFIG

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

def log_metadata(model_config, optimizer):
    lines = str(optimizer).split("\n")
    logging_config = dict(
        batch_size= model_config["batch_size"],
        learning_rate= model_config["lr"],
        max_epochs= model_config["max_epochs"],
        train_portion=config.DATA_CONFIG["train_portion"],
        test_portion=config.DATA_CONFIG["test_portion"],
        optimizer= lines[0].split(" ")[0],
        optimizer_parameters= lines[1:-1]
    )
    return logging_config


def log_model(run,model,optimizer):
    log_model_as_artifact(run,model,str(MODEL_CONFIG["model_class"].__name__),"the trained parameters",SP_MODEL_CONFIG)

def log_model_as_artifact(run,model,name,description,config):
    model_artifact = wandb.Artifact(
            name, type="model",
            description=description,
            metadata=dict(config))
    torch.save(model.state_dict(), "trained_model.pth")
    model_artifact.add_file("trained_model.pth")
    wandb.save("trained_model.pth")
    run.log_artifact(model_artifact)


def load_model(run):
    return load_model_from_artifact(run,MODEL_CONFIG["model_class"])

def load_model_from_artifact(run,model_class):
    model_artifact = run.use_artifact(str(model_class.__name__)+":latest")
    model_dir = model_artifact.download()
    model_path = os.path.join(model_dir, "trained_model.pth")
    model_config = model_artifact.metadata
    wandb.config.update(model_config)

    model = model_class(**model_config)
    model.load_state_dict(torch.load(model_path))
    return model
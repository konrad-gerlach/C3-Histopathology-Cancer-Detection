import os
import torch
import wandb
import config
import argparse


def predicted_lables(pred):
    pred = torch.sigmoid(pred)
    pred = torch.round(pred, decimals=0)
    return pred


def choose_optimizer(optimizer_config, parameters, learning_rate):
    use_optimizer = optimizer_config["use_optimizer"].lower()
    weigth_decay = optimizer_config["weight_decay"]

    if use_optimizer == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, betas=optimizer_config["betas"],
                                eps=optimizer_config["eps"], weight_decay=weigth_decay,
                                amsgrad=optimizer_config["amsgrad"])
    elif use_optimizer == "adadelta":
        return torch.optim.Adadelta(parameters, lr=learning_rate, rho=optimizer_config["rho"],
                                    eps=optimizer_config["eps"], weight_decay=weigth_decay)
    elif use_optimizer == "adagrad":
        return torch.optim.Adagrad(parameters, lr=learning_rate, lr_decay=optimizer_config["lr_decay"],
                                   weight_decay=weigth_decay)
    elif use_optimizer == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=learning_rate, alpha=optimizer_config["alpha"],
                                   eps=optimizer_config["eps"], weight_decay=weigth_decay,
                                   momentum=optimizer_config["momentum"])
    elif use_optimizer == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=optimizer_config["momentum"],
                               weight_decay=weigth_decay)

    else:
        return torch.optim.SGD(parameters, lr=learning_rate)


def log_metadata(optimizer):
    lines = str(optimizer).split("\n")
    logging_config = dict(
        batch_size=config.OPTIMIZER_CONFIG["batch_size"],
        learning_rate=config.OPTIMIZER_CONFIG["lr"],
        max_epochs=config.TRAINER_CONFIG["max_epochs"],
        train_portion=config.DATA_CONFIG["train_portion"],
        test_portion=config.DATA_CONFIG["test_portion"],
        optimizer=lines[0].split(" ")[0],
        optimizer_parameters=lines[1:-1],
        model_config = config.MODEL_CONFIG,
        sp_model_config = config.SP_MODEL_CONFIG,
        wandb_config = config.WANDB_CONFIG,
        trainer_config = config.TRAINER_CONFIG,
        optimizer_config = config.OPTIMIZER_CONFIG,
        data_config = config.DATA_CONFIG,
        load_config = config.LOAD_CONFIG,
        sweep_config = config.SWEEP_CONFIG
    )
    return logging_config


def log_model(run,model,optimizer):
    log_model_as_artifact(run,model,config.LOAD_CONFIG["name"],"the trained parameters",config.SP_MODEL_CONFIG)


def log_model_as_artifact(run, model, name, description, config):
    model_artifact = wandb.Artifact(
        name, type="model",
        description=description,
        metadata=dict(config))
    torch.save(model.state_dict(), "trained_model.pth")
    model_artifact.add_file("trained_model.pth")
    wandb.save("trained_model.pth")
    run.log_artifact(model_artifact)


def load_model(run):
    return load_model_from_artifact(run,config.MODEL_CONFIG["model_class"],config.LOAD_CONFIG["name"],config.LOAD_CONFIG["alias"])

def load_model_from_artifact(run,model_class,name, alias):
    model_artifact = run.use_artifact(name+":"+alias)
    model_dir = model_artifact.download()
    model_path = os.path.join(model_dir, "trained_model.pth")
    model_config = model_artifact.metadata
    wandb.config.update(model_config)

    model = model_class(**model_config)
    model.load_state_dict(torch.load(model_path, map_location=config.TRAINER_CONFIG["device"]))
    return model


def define_dataset_location():
    parser = argparse.ArgumentParser(description='configure project')
    parser.add_argument('--ds_path', default=config.DATA_CONFIG["ds_path"],
                        help='the location where the dataset is or should be located')

    args = parser.parse_args()
    config.DATA_CONFIG["ds_path"] = args.ds_path
    print(config.DATA_CONFIG["ds_path"])

def job_type_of_training():
    continue_training = config.TRAINER_CONFIG["continue_training"]
    if continue_training:
        return "resume_training_classifier"
    else:
        return "train_classifier"

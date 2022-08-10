import wandb
import trainer
import config


def run_sweep():
    sweep_config = {'method': 'bayes'}

    metric = {
        'name': 'test accuracy per epoch',
        'goal': 'maximize'
    }
    sweep_config['metric'] = metric

    parameters_dict = {
            'fc_layer_size': {
            'values': [128, 256]
        },
        'conv_dropout': {
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
        },
        'lr': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0.001,
            'max': 0.01
        },
        'weight_decay': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0.001,
            'max': 0.01
        },
    }
    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project=config.WANDB_CONFIG["project"], entity=config.WANDB_CONFIG["entity"])

    # calls run_classifier_with mulitple times with different configs
    wandb.agent(sweep_id, run_classifier_with, count=config.SWEEP_CONFIG["runs"])


def run_classifier_with(sweep_config=None):
    continue_training = config.TRAINER_CONFIG["continue_training"]
    if continue_training:
        job_type = "resume_training_classifier"
    else:
        job_type = "train_classifier"

    with wandb.init(project=config.WANDB_CONFIG["project"], entity=config.WANDB_CONFIG["entity"], config=sweep_config,
                    job_type=job_type) as run:
        sweep_config = wandb.config
        # change the configs globally so they can be unsed elsewhere
        config.TRAINER_CONFIG["max_epochs"] = config.SWEEP_CONFIG["epochs"]
        config.OPTIMIZER_CONFIG["lr"] = sweep_config.lr
        #config.OPTIMIZER_CONFIG["batch_size"] = sweep_config.batch_size

        #config.OPTIMIZER_CONFIG["use_optimizer"] = sweep_config.optimizer
        config.OPTIMIZER_CONFIG["weight_decay"] = sweep_config.weight_decay

        config.DATA_CONFIG["train_portion"] = config.SWEEP_CONFIG["train_portion"]
        config.DATA_CONFIG["test_portion"] = config.SWEEP_CONFIG["test_portion"]

        config.SP_MODEL_CONFIG["fc_layer_size"] = sweep_config.fc_layer_size
        config.SP_MODEL_CONFIG["conv_dropout"] = sweep_config.conv_dropout
        #config.SP_MODEL_CONFIG["fully_dropout"] = sweep_config.fully_dropout

        trainer.run_classifier(run, continue_training)


if __name__ == "__main__":
    run_sweep()

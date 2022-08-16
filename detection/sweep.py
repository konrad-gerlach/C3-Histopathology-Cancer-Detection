import argparse
import math
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
    'fc_layer_size': {
        'values': [128, 256, 512, 1028]
        },
    'fully_dropout': {
          'values': [0.4, 0.5, 0.6, 0.7]
        },
    'conv_dropout': {
        'values': [0.1, 0.2, 0.3]
    },
    'batch_size': {
          'values': [8, 16, 64]
        },
    'lr': {
        # a flat distribution between 0 and 0.1
        'distribution': 'log_uniform',
        'min': math.log(0.0001),
        'max': math.log(0.1)
      },
    'weight_decay': {
        # a flat distribution between 0 and 0.1
        'distribution': 'log_uniform',
        'min': math.log(0.0001),
        'max': math.log(0.1)
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

        trainer.run_classifier(run, False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='configure project')
    parser.add_argument('--ds_path', default=config.DATA_CONFIG["ds_path"],
                        help='the location where the dataset is or should be located')

    args = parser.parse_args()
    config.DATA_CONFIG["ds_path"] = args.ds_path
    run_sweep()
    run_sweep()

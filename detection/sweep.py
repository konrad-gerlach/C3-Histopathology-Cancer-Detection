import wandb

def run_sweep():
    sweep_config = {'method': 'random'}

    metric = {
    'name': 'test accuracy per epoch',
    'goal': 'maximize'   
    }
    sweep_config['metric'] = metric

    parameters_dict = {
    'optimizer': {
        'values': ['adam', 'sgd', 'rmspropp']
        },
    'fc_layer_size': {
        'values': [128, 256, 512]
        },
    'fully_dropout': {
          'values': [0.4, 0.5, 0.6]
        },
    'conv_dropout': {
        'values': [0.1, 0.2, 0.3]
    },
    'batch_size': {
          'values': [4, 16, 64]
        },
    'learning_rate': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
      },
    'weight_decay': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
      },
    }
    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project="histo_cancer")

    wandb.agent(sweep_id, run_classifier, count=1000)
    
    return 0

if __name__ == "__main__":
    run_sweep()
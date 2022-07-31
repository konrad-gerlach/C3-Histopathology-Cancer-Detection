import torch

import model

PRJ = "histo_cancer"
#default values
MODEL_CONFIG = dict(
    batch_size=64,
    gradient_accumulation = 1, #https://stackoverflow.com/questions/63815311/what-is-the-correct-way-to-implement-gradient-accumulation-in-pytorch approach no 1. was chosen
    num_workers=4,
    lr=0.01,
    max_epochs=1,
    model_class = model.Big_Konrad
)

#specific for different models, just put in all the values
#you want to adjust, that are not captured above
#default values
SP_MODEL_CONFIG = dict(
    conv_dropout=0,
    fully_dropout=0.5,
    fc_layer_size=200
)

TRAINER_CONFIG = dict(
    project=PRJ,
    entity="histo-cancer-detection",
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
)

#supports adam, adadelta, rmsprop, adagrad, sgd (with weight decay and momentum)
#if none is selected, sgd is used
#https://pytorch.org/docs/stable/optim.html
#default values
OPTIMIZER_CONFIG = dict(
    use_optimizer = "", 
    alpha = 0.99, #For RmsProp
    betas= (0.9, 0.999), #For Adam
    rho=0.9, #For Adadelta
    eps=1e-08,
    weight_decay=0,
    amsgrad=False,
    momentum=0,
    lr_decay=0.1,
)

#default values
DATA_CONFIG = dict(
    train_portion = 0.01,
    test_portion = 0.01,
    ds_path = 'datasets/cancer',
    use_cache = True
)

#default values
SWEEP_CONFIG = dict(
    train_portion = 0.001,
    test_portion = 0.001,
    epochs = 1
    ,
    runs = 1000
)
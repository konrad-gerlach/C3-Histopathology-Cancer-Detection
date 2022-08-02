import torch
import model

PRJ = "histo_cancer"
#default values
MODEL_CONFIG = dict(
    batch_size=64,
    gradient_accumulation = 1, #https://stackoverflow.com/questions/63815311/what-is-the-correct-way-to-implement-gradient-accumulation-in-pytorch approach no 1. was chosen
    num_workers=4,
    lr=0.020769733168812123,
    max_epochs=100,
    model_class = model.Big_K
)

#constructor arguments for model selected with MODEL_CONFIG["model_class"]
SP_MODEL_CONFIG = dict(
    conv_dropout=0.1,
    fully_dropout=0.6,
    fc_layer_size=512
)

TRAINER_CONFIG = dict(
    project=PRJ,
    entity="histo-cancer-detection",
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    continue_training = False,  #if set to true the latest model for MODEL_CONFIG["model_class"] with alias LOAD_CONFIG["alias"] will be downloaded and used for training
    accuracy_goal = 0.95 #model will be saved once this testing accuracy has been reached
)
LOAD_CONFIG = dict(
    alias="usable"
)

#supports adam, adadelta, rmsprop, adagrad, sgd (with weight decay and momentum)
#if none is selected, sgd is used
#https://pytorch.org/docs/stable/optim.html
#default values
OPTIMIZER_CONFIG = dict(
    use_optimizer = "adam", 
    alpha = 0.99, #For RmsProp
    betas= (0.9, 0.999), #For Adam
    rho=0.9, #For Adadelta
    eps=1e-08,
    weight_decay=0.00550143838583892,
    amsgrad=False,
    momentum=0,
    lr_decay=0.1,
)

#default values
DATA_CONFIG = dict(
    train_portion = 0.66,
    test_portion = 0.33,
    ds_path = 'datasets/cancer',
    use_cache = True
)

#default values
SWEEP_CONFIG = dict(
    train_portion = 0.001,
    test_portion = 0.001,
    epochs = 1,
    runs = 1000
)
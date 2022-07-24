import torch

PRJ = "histo_cancer"

MODEL_CONFIG = dict(
    batch_size=64,
    num_workers=4,
    learning_rate=0.01,
    max_epochs=100
)

TRAINER_CONFIG = dict(
    project=PRJ,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
)

#supports adam, adadelta, rmsprop, adagrad, sgd (with weight decay and momentum)
#if none is selected, sgd is used
#https://pytorch.org/docs/stable/optim.html
OPTIMIZER_CONFIG = dict(
    #use_optimizer = "sgd", 
    alpha = 0.99, #For RmsProp
    betas= (0.9, 0.999), #For Adam
    rho=0.9, #For Adadelta
    eps=1e-08,
    weight_decay=0,
    amsgrad=False,
    momentum=0,
    lr_decay=0.1,
)

DATA_CONFIG = dict(
    train_portion = 0.9,
    test_portion = 0.1
)
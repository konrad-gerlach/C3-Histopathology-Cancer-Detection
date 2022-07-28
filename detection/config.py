import torch

PRJ = "histo_cancer"

MODEL_CONFIG = dict(
    batch_size=16,
    gradient_accumulation = 4, #https://stackoverflow.com/questions/63815311/what-is-the-correct-way-to-implement-gradient-accumulation-in-pytorch approach no 1. was chosen
    num_workers=4,
    learning_rate=0.01,
    max_epochs=10
)

TRAINER_CONFIG = dict(
    project=PRJ,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
)

#supports adam, adadelta, rmsprop, adagrad, sgd (with weight decay and momentum)
#if none is selected, sgd is used
#https://pytorch.org/docs/stable/optim.html
OPTIMIZER_CONFIG = dict(
    use_optimizer = "adam", 
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
    train_portion = 0.1,
    test_portion = 0.1
)
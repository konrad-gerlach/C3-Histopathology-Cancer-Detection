import torch
import model

MODEL_CONFIG = dict(
    model_class=model.Big_Konrad
)

# constructor arguments for model selected with MODEL_CONFIG["model_class"]
SP_MODEL_CONFIG = dict(
    conv_dropout=0.1,
    fully_dropout=0.5,
    fc_layer_size=200
)

WANDB_CONFIG = dict(
    project="histo_cancer",
    entity="histo-cancer-detection"
)

TRAINER_CONFIG = dict(
    max_epochs=25,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    continue_training=True,
    # if set to true the latest model for MODEL_CONFIG["model_class"] with alias LOAD_CONFIG["alias"] will be
    # downloaded and used for training
    accuracy_goal=0.97,  # model will be saved once this testing accuracy has been reached
    gradient_accumulation=1,
    # https://stackoverflow.com/questions/63815311/what-is-the-correct-way-to-implement-gradient-accumulation-in
    # (pytorch approach no 1. was chosen)
    mode = "training" #supports training,sweeps,feature_visualization
)

# supports adam, rmsprop, sgd (with weight decay and momentum)
# default is sgd
# https://pytorch.org/docs/stable/optim.html
OPTIMIZER_CONFIG = dict(
    batch_size=64,
    use_optimizer="adam",
    lr=0.01,
    weight_decay=0,
    alpha=0.99,  # For RmsProp
    betas=(0.9, 0.999),  # For Adam
    eps=1e-08,
    amsgrad=False,
    momentum=0,
    lr_decay=0.1,
)

DATA_CONFIG = dict(
    num_workers=4,
    train_portion=0.67,
    test_portion=0.33,
    ds_path='datasets/cancer',
    use_cache=False,
    grayscale=False
)

#set the alias accordingly for continuos training
if DATA_CONFIG["grayscale"]:
    ALIAS="usable-black-and-white"
else:
    ALIAS="usable-colored"

#available aliasas usable-black-and-white, usable-colored, bad_colored
LOAD_CONFIG = dict(
    alias=ALIAS,
    name="Big_Konrad"
)

SWEEP_CONFIG = dict(
    train_portion=0.1,
    test_portion=0.1,
    epochs=10,
    runs=1000
)

from tkinter.tix import X_REGION
import torch
import matplotlib.pyplot as plt
import torchvision
import wandb
import config
import helper
import generic_train_loop

def sample(img_shape,device):
    return torch.ones(img_shape,device=device,requires_grad=True)


def visualizer():
    trainer_config = config.TRAINER_CONFIG
    job_type = "visualization"
    wandb.config = {}
    run = wandb.init(project=trainer_config["project"], entity=trainer_config["entity"], job_type=job_type)
    run_visualizer(run)

def run_visualizer(run):
    trainer_config = config.TRAINER_CONFIG
    model_config = config.MODEL_CONFIG
    optimizer_config = config.OPTIMIZER_CONFIG
    model = helper.load_model(run)

    input = sample((3,96,96),trainer_config["device"])
    optimizer = helper.choose_optimizer(optimizer_config,[input], model_config["gradient_accumulation"], learning_rate=model_config["lr"])
    logging_config = helper.log_metadata(model_config, optimizer)
 
    wandb.config.update(logging_config)
   
    wandb.watch(model, criterion=None, log="gradients", log_freq=1000, idx=None, log_graph=(True))


    print("You are currently using the optimizer: {}".format(optimizer))
    print(trainer_config["device"])

    visualize(model, optimizer,input, trainer_config["device"], model_config["gradient_accumulation"], epochs=model_config["max_epochs"])
    
    wandb.finish()

def visualizer_loss_fn(outputs,y):
    return (-outputs[-1]).mean()

def visualize(model, optimizer, input, device, gradient_accumulation,epochs=5):
    loss_fn = visualizer_loss_fn
    visualizer_loop(model, loss_fn, input, optimizer, device, epochs, gradient_accumulation)

def logger(outputs,loss,batch,X,y,inputs):
    X = torch.clamp(X,0,1)
    wandb.log({"loss":loss})
    wandb.log({"inputs" : [wandb.Image(x) for x in X]})

def visualizer_loop(model, loss_fn, input, optimizer, device, epochs, gradient_accumulation):
    model = model.to(device)
    input = input.view(1,*input.shape)
    y = torch.zeros(1)
    inputs = dict()
    model.eval()
    for i in range(100000):
        generic_train_loop.train_loop(1,input.clamp(0,1),y,device,model,loss_fn,gradient_accumulation, optimizer, logger, inputs)


def show(images):    
    # Here _ means that we ignore (not use) variables
    _, figs = plt.subplots(1, len(images), figsize=(200, 200))
    for f, img in zip(figs, images):
        f.imshow(torchvision.transforms.ToPILImage()(img))
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)

if __name__ == "__main__":
    show(sample((3,3,32,32),'cpu'))
    plt.show()
from tkinter.tix import X_REGION
import torch
from matplotlib import pyplot as plt
import torchvision
import wandb
import config
import helper
import generic_train_loop
import data

def sample(img_shape,device):
    return torch.rand(img_shape,device=device,requires_grad=True)


def visualizer():
    job_type = "visualization"
    wandb.config = {}
    run = wandb.init(project=config.WANDB_CONFIG["project"], entity=config.WANDB_CONFIG["entity"], job_type=job_type)
    run_visualizer(run)

def run_visualizer(run):
    model = helper.load_model(run)

    input = sample((1,3,96,96),config.TRAINER_CONFIG["device"])
    optimizer = helper.choose_optimizer(config.OPTIMIZER_CONFIG,[input], config.TRAINER_CONFIG["gradient_accumulation"], learning_rate=config.OPTIMIZER_CONFIG["lr"])
    logging_config = helper.log_metadata()
 
    wandb.config.update(logging_config)
   
    wandb.watch(model, criterion=None, log="gradients", log_freq=1000, idx=None, log_graph=(True))


    print("You are currently using the optimizer: {}".format(optimizer))

    visualize(model, optimizer,input, config.TRAINER_CONFIG["device"], config.TRAINER_CONFIG["gradient_accumulation"], epochs=config.TRAINER_CONFIG["max_epochs"])
    
    wandb.finish()

def visualizer_loss_fn(outputs,y):
    layer = outputs[-1]
    loss = torch.zeros(1,device=layer.device)
    for i in range(len(layer)):
        loss += layer[i,i]
    return loss

def data_example_loss_fn(outputs,y):
    layer = outputs[-1]
    loss = torch.zeros(len(layer),device=layer.device)
    for i in range(len(layer)):
        loss[i] += layer[i][0]
    return loss

def visualize(model, optimizer, input, device, gradient_accumulation,epochs=5):
    loss_fn = visualizer_loss_fn
    visualizer_loop(model, loss_fn, input, optimizer, device, epochs, gradient_accumulation)
    loss_fn = data_example_loss_fn
    get_data_examples(model,device,loss_fn)

def logger(outputs,loss,batch,X,y,inputs):
    X = torch.clamp(X,0,1)
    wandb.log({"loss":loss})
    wandb.log({"inputs_transformed" : [wandb.Image(x) for x in X]})

def visualizer_loop(model, loss_fn, input, optimizer, device, epochs, gradient_accumulation):
    model = model.to(device)
    y = torch.zeros(1)
    inputs = dict()
    model.eval()
    for i in range(1500):
        wandb.log({"inputs" : [wandb.Image(x) for x in input]})
        X = random_transform(input.clamp(0,1))
        generic_train_loop.train_loop(1,X,y,device,model,loss_fn,gradient_accumulation, optimizer, logger, inputs)

def random_transform(inputs):
    inputs = torchvision.transforms.GaussianBlur(5)(inputs)
    inputs = torchvision.transforms.RandomAffine(2,translate=(0.1,0.1),scale=(0.8,1.2))(inputs)
    return inputs

def get_data_examples(model,device,loss_fn):
    model = model.to(device)
    model.eval()
    train_dl, test_dl, img_shape = data.get_dl(64,4)
    results = []
    with torch.no_grad():
        for batch, (X,y) in enumerate(test_dl):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y = y.view(-1, 1).to(torch.float)

            outputs = model.forward_per_layer(X)
            loss = loss_fn(outputs, y)
            X = X.to("cpu", non_blocking=True)
            y = y.to("cpu", non_blocking=True)
            loss = loss.to("cpu", non_blocking=True)
            for i in range(len(X)):
                results.append((X[i],loss[i],y[i]))
    results.sort(key= lambda img_and_loss_tuple: img_and_loss_tuple[1])
    wandb.log({"minimzer_images" : [wandb.Image(img_and_loss_tuple[0],caption=("cancer: "+str(img_and_loss_tuple[2].item()))+ " loss: " + str(img_and_loss_tuple[1].item())) for img_and_loss_tuple in results[:10]]})
   


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
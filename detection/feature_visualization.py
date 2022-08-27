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


def generate_initial_sample(img_shape):
    width = img_shape[1]
    height = img_shape[2]
    if True or config.DATA_CONFIG["grayscale"]:
        return sample((1,1,width,height),config.TRAINER_CONFIG["device"])
    else:
        return sample((1,3,width,height),config.TRAINER_CONFIG["device"])


#conversion from single channel grayscale to three channel grayscale
def pad_image_channels(image):
    if True or config.DATA_CONFIG["grayscale"]:
        return image.repeat(1,3,1,1)
    else:
        return image

def random_transform(inputs):
    transformed = torchvision.transforms.GaussianBlur(5,sigma = 0.75)(inputs)
    transformed = torchvision.transforms.RandomAffine(15,translate=(0.1,0.1),scale=(0.8,1.2))(transformed)
    return transformed

#loss function for a batch of multiple images to be optimized to minimize loss functions for multiple neurons
def visualizer_loss_fn(outputs,y):
    layer = outputs[-1]
    loss = torch.zeros(1,device=layer.device)
    for i in range(len(layer)):
        loss += layer[i,i]
    return loss

#loss function for a batch of multiple dataset images minimizing the same loss function
def data_example_loss_fn(outputs,y):
    layer = outputs[-1]
    loss = torch.zeros(len(layer),device=layer.device)
    for i in range(len(layer)):
        loss[i] += layer[i][0]
    return loss

def visualize(model, optimizer, device, gradient_accumulation,sample_input,epochs=5):
    loss_fn = visualizer_loss_fn
    visualizer_loop(model, loss_fn, optimizer, device, epochs, gradient_accumulation,sample_input)
    #loss_fn = data_example_loss_fn
    #get_data_examples(model,device,loss_fn)

def logger(outputs,loss,batch,X,y,inputs):
    X = torch.clamp(X,0,1)
    wandb.log({"loss":loss})
    wandb.log({"inputs_transformed" : [wandb.Image(x) for x in X]})

def visualizer_loop(model, loss_fn, optimizer, device, epochs, gradient_accumulation,sample_input):
    model = model.to()
    y = torch.zeros(1)
    inputs = dict()
    model.eval()
    for i in range(1500):
        wandb.log({"inputs" : [wandb.Image(x) for x in sample_input]})
        X = random_transform(pad_image_channels(sample_input.clamp(0,1)))
        generic_train_loop.train_loop(1,X,y,device,model,loss_fn,gradient_accumulation, optimizer, logger, inputs)

def get_data_examples(model,device,loss_fn):
    model = model.to(device)
    model.eval()
    train_dl, test_dl, img_shape = data.get_dl(config.OPTIMIZER_CONFIG["batch_size"])
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

def setup_wandb():
    job_type = "visualization"
    wandb.config = {}
    return wandb.init(project=config.WANDB_CONFIG["project"], entity=config.WANDB_CONFIG["entity"], job_type=job_type)

def run_visualizer(run):
    model = helper.load_model(run)

    _, _, img_shape = data.get_dl(config.OPTIMIZER_CONFIG["batch_size"])
    sample_input = generate_initial_sample(img_shape)
    optimizer = helper.choose_optimizer(config.OPTIMIZER_CONFIG,[sample_input], config.TRAINER_CONFIG["gradient_accumulation"], learning_rate=config.OPTIMIZER_CONFIG["lr"])
    logging_config = helper.log_metadata(optimizer)

    wandb.config.update(logging_config)
    wandb.watch(model, criterion=None, log="gradients", log_freq=1000, idx=None, log_graph=(True))

    visualize(model, optimizer, config.TRAINER_CONFIG["device"], config.TRAINER_CONFIG["gradient_accumulation"],sample_input, epochs=config.TRAINER_CONFIG["max_epochs"])
    
    wandb.finish()

def visualizer():
    run = setup_wandb()
    run_visualizer(run)


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
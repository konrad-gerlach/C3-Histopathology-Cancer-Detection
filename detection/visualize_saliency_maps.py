import matplotlib.pyplot as plt
import torch
import config
import data
import wandb
import helper


# https://towardsdatascience.com/saliency-map-using-pytorch-68270fe45e80

wandb_config = config.WANDB_CONFIG
job_type = "saliency"
run = wandb.init(project=wandb_config["project"], entity=wandb_config["entity"], job_type=job_type)

model = helper.load_model(run)
device = config.TRAINER_CONFIG["device"]
model = model.to(device)

# Set the model on Eval Mode
model.eval()
wandb.finish()


# Open the image file
data_loader, _, _ = data.get_dl(batch_size=1, num_workers=1)
image=0
for batch, (X, y) in enumerate(data_loader):
    imageBatch = X
    imageBatch = imageBatch.to(device)
    imageBatch.requires_grad_()
    break



# Retrieve output from the image
output = model(imageBatch)

# Catch the output
output_idx = output.argmax()
output_max = output[0, output_idx]

# Do backpropagation to get the derivative of the output based on the image
output_max.backward()

# Retireve the saliency map and also pick the maximum value from channels on each pixel.
# In this case, we look at dim=1. Recall the shape (batch_size, channel, width, height)

def saliency_visualizer():
    pass


if __name__ == "__main__":
    saliency_visualizer()
    saliency, _ = torch.max(imageBatch.grad.data.abs(), dim=1)
    saliency = saliency.reshape(96, 96)
    # Visualize the image and the saliency map
    fig, ax = plt.subplots(1, 2)
    image = next(iter(imageBatch)).reshape(-1, 96, 96)
    ax[0].imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
    ax[0].axis('off')
    ax[1].imshow(saliency.cpu(), cmap='hot')
    ax[1].axis('off')
    plt.tight_layout()
    fig.suptitle('The Image and Its Saliency Map')
    plt.show()


import re
import matplotlib.pyplot as plt
import torch
import config
import data
import wandb
import helper


# Inspired by: https://towardsdatascience.com/saliency-map-using-pytorch-68270fe45e80
def setup():
    
    wandb_config = config.WANDB_CONFIG
    job_type = "saliency"
    run = wandb.init(project=wandb_config["project"], entity=wandb_config["entity"], job_type=job_type)

    model = helper.load_model(run)
    device = config.TRAINER_CONFIG["device"]
    model = model.to(device)

    # Set the model on Eval Mode
    model.eval()
    # Open the image file
    data_loader, _, _ = data.get_dl(batch_size=1, num_workers=1)
    wandb.finish()

    return data_loader, device, model


def show_saliencies(images):
    fig, ax = plt.subplots(5, len(images))
    for i, image in enumerate(images):
        
        #red = larger absolut gradient in one of the 3 channels
        # ... just shows certainty (not if for or against cancer)
        sal_abs, _ = torch.max(image.grad.data.abs(), dim=1)
        sal_abs = sal_abs.reshape(96, 96)



        #red = one of the channels has super high gradient 
        #vs black = one of the channels has super low gradient
        # ... most certain channel wins
        sal_max, _ = torch.max(image.grad.data, dim=1)
        sal_min, _ = torch.min(image.grad.data, dim=1)

        geq = sal_max.abs() >= sal_min.abs()
        geq = geq.type(torch.int)
        sal_max = sal_max * geq
        sal_min = sal_min * geq.neg()
        saliency = sal_max + sal_min
        saliency = saliency.reshape(96, 96)

        # Visualize the image and the saliency map
        img = next(iter(image)).reshape(-1, 96, 96)
        ax[0, i].imshow(img.cpu().detach().numpy().transpose(1, 2, 0))
        ax[0, i].axis('off')

        ax[1, i].imshow(sal_abs.cpu(), cmap='hot')
        ax[1, i].axis('off')
        ax[2, i].imshow(sal_abs.cpu(), cmap='RdGy')
        ax[2, i].axis('off')

        ax[3, i].imshow(saliency.cpu(), cmap='hot')
        ax[3, i].axis('off')
        ax[4, i].imshow(saliency.cpu(), cmap='RdGy')
        ax[4, i].axis('off')

    plt.tight_layout(pad=0.7)
    fig.suptitle('Images of cancer and corresponding saliency maps')
    plt.show()


def saliency_visualizer():
    image_data, device, model = setup()
    images = []
    num_images = 10
    for batch, (X, y) in enumerate(image_data):
        if y:
            X = X.to(device)
            X.requires_grad_()
            images.append(X)
        if len(images) >= num_images:
            break

    for image in images:
        # Retrieve output from the image
        output = model(image)
        # Catch the output
        output_idx = output.argmax()
        output_max = output[0, output_idx]
        # Do backpropagation to get the derivative of the output based on the image
        output_max.backward()

    show_saliencies(images)


if __name__ == "__main__":
    saliency_visualizer()

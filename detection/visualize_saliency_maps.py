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
    wandb.finish()
    # Open the image file
    data_loader, _, _ = data.get_dl(batch_size=1, num_workers=1)

    return data_loader, device, model


def show_saliencies(images):
    fig, ax = plt.subplots(2, len(images))
    for i, image in enumerate(images):
        saliency, _ = torch.max(image.grad.data.abs(), dim=1)
        saliency = saliency.reshape(96, 96)
        # Visualize the image and the saliency map

        img = next(iter(image)).reshape(-1, 96, 96)
        ax[0, i].imshow(img.cpu().detach().numpy().transpose(1, 2, 0))
        ax[0, i].axis('off')
        ax[1, i].imshow(saliency.cpu(), cmap='afmhot')
        ax[1, i].axis('off')

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

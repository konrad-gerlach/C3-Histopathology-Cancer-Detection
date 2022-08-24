import matplotlib.pyplot as plt
import torch
import config
import data
import wandb
import helper


# Inspired by: https://towardsdatascience.com/saliency-map-using-pytorch-68270fe45e80
def setup(grayscale, good_model):
    # order important :/
    if good_model:
        if grayscale:
            config.DATA_CONFIG["grayscale"] = True
        else:
            config.LOAD_CONFIG["alias"] = "usable-colored"
    else:
        if grayscale:
            config.DATA_CONFIG["grayscale"] = True
        else:
            config.LOAD_CONFIG["alias"] = "bad_colored"

    if grayscale:
        config.DATA_CONFIG["grayscale"] = True
    else:
        config.DATA_CONFIG["grayscale"] = False

    wandb_config = config.WANDB_CONFIG
    job_type = "saliency"
    run = wandb.init(project=wandb_config["project"], entity=wandb_config["entity"], job_type=job_type)

    model = helper.load_model(run)
    device = config.TRAINER_CONFIG["device"]
    model = model.to(device)

    # Set the model on Eval Mode
    model.eval()
    data_loader, _, _ = data.get_dl(batch_size=1, num_workers=1)

    return data_loader, device, model


def show_saliencies(images):
    fig, ax = plt.subplots(5, len(images))
    for i, image in enumerate(images):
        image_grads = image.grad.data.abs()
        
        red_grads = image_grads[0:1,0]
        green_grads = image_grads[0:1,1]
        blue_grads = image_grads[0:1,2]
        
        red_grads = red_grads.reshape(96, 96)
        green_grads = green_grads.reshape(96, 96)
        blue_grads = blue_grads.reshape(96, 96)

        # red = larger absolut gradient in one of the 3 channels
        # ... just shows certainty (not if for or against cancer?)
        sal_abs, _ = torch.max(image_grads, dim=1)
       
        sal_abs = sal_abs.reshape(96, 96)

        # Visualize the image and the saliency map
        img = next(iter(image)).reshape(-1, 96, 96)
        ax[0, i].imshow(img.cpu().detach().numpy().transpose(1, 2, 0))
        ax[0, i].axis('off')

        ax[1, i].imshow(red_grads.cpu(), cmap='inferno')
        ax[1, i].axis('off')
        ax[2, i].imshow(green_grads.cpu(), cmap='inferno')
        ax[2, i].axis('off')
        ax[3, i].imshow(blue_grads.cpu(), cmap='inferno')
        ax[3, i].axis('off')
        ax[4, i].imshow(sal_abs.cpu(), cmap='inferno')
        ax[4, i].axis('off')

        inferno = plt.get_cmap('inferno')
        wandb.log({"input" : wandb.Image(img.cpu().detach().numpy().transpose(1, 2, 0))})
        wandb.log({"red" : wandb.Image(inferno(red_grads))})
        wandb.log({"green" : wandb.Image(inferno(green_grads))})
        wandb.log({"blue" : wandb.Image(inferno(blue_grads))})
        wandb.log({"full" : wandb.Image(inferno(sal_abs))})


    wandb.log({"Cancer images with saliency maps": plt})

    plt.tight_layout(pad=0.7)
    fig.suptitle('Images of cancer and corresponding saliency maps')
    plt.show()
    cancer_regions(sal_abs, img)


def cancer_regions(sal_abs, image):
    # setup
    threshold = 0.995
    off = 3

    value_treshold = torch.quantile(sal_abs, q=threshold)
    cancer_areas = torch.zeros(96, 96)

    for i in range(0, 95):
        for k in range(0, 95):
            if sal_abs[i, k] >= value_treshold:
                cancer_areas[i - off:i + off, k - off:k + off] = 1

    regions = torch.mul(image, cancer_areas)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
    ax[0].axis('off')
    ax[1].imshow(sal_abs.cpu(), cmap='hot')
    ax[1].axis('off')
    ax[2].imshow(regions.cpu().detach().numpy().transpose(1, 2, 0))
    ax[2].axis('off')

    plt.tight_layout(pad=0.7)
    fig.suptitle('Focus on regions that made the model predict cancer')
    plt.show()


def collect_images_with_gradient(grayscale, good_model, num_images, images):
    image_data, device, model = setup(grayscale, good_model)
    num_images = len(images) + num_images

    highly_cancerous = 0.999

    highly_non_cancerous = 0.01

    for batch, (X, y) in enumerate(image_data):
        if y==1:
            X = X.to(device)
            X.requires_grad_()
            output = model(X)
            if output > highly_cancerous:
                images.append(X)
        if len(images) >= num_images:
            break

    for image in images:
        # Retrieve output from the image
        output = model(image)
        # Catch the output
        output.backward()

        """
        print(output)
        output_idx = output.argmax()
        output_max = output[0, output_idx]
        print(output_max)
        # Do backpropagation to get the derivative of the output based on the image
        output_max.backward()
        """

    return images


def saliency_visualizer():
    # configure here

    num_images = 8
    images = []
    images = collect_images_with_gradient(False, True, num_images, images)
    #images = collect_images_with_gradient(False, False, num_images, images)
    #images = collect_images_with_gradient(True, True, num_images, images)

    show_saliencies(images)


if __name__ == "__main__":
    saliency_visualizer()

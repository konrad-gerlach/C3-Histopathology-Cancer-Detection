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
    _, data_loader, _ = data.get_dl(batch_size=1, num_workers=1)

    return data_loader, device, model


def show_saliencies(images):
    fig, ax = plt.subplots(2, len(images))
    for i, image_and_gradient in enumerate(images):
        image = image_and_gradient[0]
        gradient = image_and_gradient[1]
        # red = larger absolut gradient in one of the 3 channels
        # ... just shows certainty (not if for or against cancer?)
        sal_abs, _ = torch.max(gradient.abs(), dim=1)
        sal_abs = sal_abs.reshape(96, 96)

        # red = one of the channels has super high gradient
        # vs black = one of the channels has super low gradient
        # ... most certain channel wins
        sal_max, _ = torch.max(gradient, dim=1)
        sal_min, _ = torch.min(gradient, dim=1)

        geq = sal_max.abs() >= sal_min.abs()
        geq = geq.type(torch.int)
        sal_max = sal_max * geq
        sal_min = sal_min * geq.neg()

        sal_max = sal_max.reshape(96, 96)
        sal_min = sal_min.reshape(96, 96)

        saliency = sal_max + sal_min
        saliency = saliency.reshape(96, 96)

        # Visualize the image and the saliency map
        img = next(iter(image)).reshape(-1, 96, 96)
        ax[0, i].imshow(img.cpu().detach().numpy().transpose(1, 2, 0))
        ax[0, i].axis('off')

        ax[1, i].imshow(sal_abs.cpu(), cmap='inferno')
        ax[1, i].axis('off')

        inferno = plt.get_cmap('inferno')
        wandb.log({"input" : wandb.Image(img.cpu().detach().numpy().transpose(1, 2, 0))})
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


def collect_images_with_gradient(grayscale, good_model, num_images, images_with_gradient):
    image_data, device, model = setup(grayscale, good_model)
    num_images = len(images_with_gradient) + num_images
    
    for batch, (image, y) in enumerate(image_data):
        
        if y:
            image = image.to(device)
            print(image.grad == None)
            image.requires_grad_()   

            # Retrieve output from the image
            output = model(image)
            # Do backpropagation to get the derivative of the output based on the image
            output.backward()
            #print(image.grad.data)
            images_with_gradient.append([image, image.grad.data])
            #image.grad = None
           
                
        if len(images_with_gradient) >= num_images:
            break
    

    return images_with_gradient


def saliency_visualizer():
    # configure here

    num_images = 3
    images_with_gradient = []    
    images_with_gradient = collect_images_with_gradient(False, True, num_images, images_with_gradient)
    images_with_gradient = collect_images_with_gradient(False, False, num_images, images_with_gradient)
    images_with_gradient = collect_images_with_gradient(True, True, num_images, images_with_gradient)
    

    show_saliencies(images_with_gradient)


if __name__ == "__main__":
    saliency_visualizer()

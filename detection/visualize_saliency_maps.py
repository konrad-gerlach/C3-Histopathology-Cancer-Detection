from email.mime import image
import matplotlib.pyplot as plt
import torch
import config
import data
import wandb
import helper


# Inspired by: https://towardsdatascience.com/saliency-map-using-pytorch-68270fe45e80
def setup(grayscale, good_model):

    #setup to choose the desired artifacts and get right data
    if grayscale:
        config.LOAD_CONFIG["alias"] = "usable-black-and-white"
        config.DATA_CONFIG["grayscale"] = True
    else:
        config.DATA_CONFIG["grayscale"] = False
        if good_model:
            config.LOAD_CONFIG["alias"] = "usable-colored"
        else:
            config.LOAD_CONFIG["alias"] = "bad_colored"         

    wandb_config = config.WANDB_CONFIG
    job_type = "saliency"
    run = wandb.init(project=wandb_config["project"], entity=wandb_config["entity"], job_type=job_type)
    model = helper.load_model(run)
    device = config.TRAINER_CONFIG["device"]
    model = model.to(device)

    # Set the model on Eval Mode
    model.eval()
    # always use the images in same order
    _, data_loader, _ = data.get_dl(batch_size=1)

    return data_loader, device, model


def show_saliencies(images):
    if not config.TRAINER_CONFIG["plot_figures"]:
        return

    fig, ax = plt.subplots(5, len(images), figsize=(100,100))
    for i, image in enumerate(images):
        image_grads = image.grad.data.abs()
        
        red_grads = image_grads[0:1,0]
        green_grads = image_grads[0:1,1]
        blue_grads = image_grads[0:1,2]
        
        red_grads = red_grads.reshape(96, 96)
        green_grads = green_grads.reshape(96, 96)
        blue_grads = blue_grads.reshape(96, 96)


        #retrieve largest absolute value across all channels 
        sal_abs, _ = torch.max(image_grads.abs(), dim=1)   
        
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

        #also log in wandb
        #inferno = plt.get_cmap('inferno')
        #wandb.log({"Sal_abs": wandb.Image(inferno(sal_abs.cpu()))})



    #wandb.log({"Cancer images with saliency maps": plt})

    plt.tight_layout(pad=0.7)
    fig.suptitle('Images of cancer and corresponding saliency maps')
    plt.show()
    cancer_regions(sal_abs, img)




def cancer_regions(sal_abs, image):
    # configure which quantile of pixels are of interest and size of surrounding

    threshold = 0.995
    off = 3

    value_treshold = torch.quantile(sal_abs, q=threshold)
    cancer_areas = torch.zeros(96, 96)

    for i in range(0, 95):
        for k in range(0, 95):
            if sal_abs[i, k] >= value_treshold:
                cancer_areas[i - off:i + off, k - off:k + off] = 1

    regions = torch.mul(image.cpu(), cancer_areas.cpu())

    cancer_surrounding = torch.ones(3, 96, 96)
    cancer_surr_color = torch.zeros(3, 96, 96)

    for i in range(0, 95):
        for k in range(0, 95):
            if cancer_areas[i,k] == 0:
                if (cancer_areas[i-1,k] == 1 or cancer_areas[i+1,k] == 1 or cancer_areas[i,k-1] == 1 or cancer_areas[i,k+1] == 1):
                    if (cancer_areas[i-1,k] == 0 or cancer_areas[i+1,k] == 0 or cancer_areas[i,k-1] == 0 or cancer_areas[i,k+1] == 0):
                        cancer_surrounding[0,i,k]=0
                        cancer_surrounding[1,i,k]=0
                        cancer_surrounding[2,i,k]=0
                        cancer_surr_color[0,i,k]=1

    
    surrounding = torch.mul(image.cpu(), cancer_surrounding.cpu())
    surrounding = torch.add(surrounding.cpu(), cancer_surr_color.cpu())

    fig, ax = plt.subplots(1, 4, figsize=(100, 100))
    ax[0].imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
    ax[0].axis('off')
    ax[1].imshow(sal_abs.cpu(), cmap='inferno')
    ax[1].axis('off')
    ax[2].imshow(regions.cpu().detach().numpy().transpose(1, 2, 0))
    ax[2].axis('off')
    ax[3].imshow(surrounding.cpu().detach().numpy().transpose(1, 2, 0))
    ax[3].axis('off')

    plt.tight_layout(pad=0.7)
    fig.suptitle('Focus on regions that made the model predict cancer')
    plt.show()


# https://towardsdatascience.com/saliency-map-using-pytorch-68270fe45e80
def collect_images_with_gradient(cancerous , grayscale, good_model, num_images, images):
    image_data, device, model = setup(grayscale, good_model)
    num_images = len(images) + num_images

    #configure
    #no cancer threshold to get same images for (non)-colored
    cancer_threshold = 0
    non_cancer_threshold = 0.01

    for batch, (image, y) in enumerate(image_data):
        if cancerous:
            if y == 1:
                image = image.to(device)
                image.requires_grad_()
                # Retrieve output from the image
                output = model(image)
                
                if torch.sigmoid(output) > cancer_threshold:
                    # Do backpropagation to get the derivative of the output based on the image
                    output.backward()
                    images.append(image)
        else:
            if y == 0:
                image = image.to(device)
                image.requires_grad_()
                # Retrieve output from the image
                output = model(image)
                if torch.sigmoid(output) < non_cancer_threshold:
                    # Do backpropagation to get the derivative of the output based on the image
                    output.backward()
                    images.append(image)            

        if len(images) >= num_images:
            break

    return images


def run_saliency_visualizer():
    images = []

    # configure. minimum num_images is 2
    num_images = 3
    images = collect_images_with_gradient(cancerous=True ,grayscale=True, good_model=True, num_images=num_images, images=images)
    images = collect_images_with_gradient(cancerous=True ,grayscale=False, good_model=True, num_images=num_images, images=images)    
    images = collect_images_with_gradient(cancerous=True ,grayscale=False, good_model=False, num_images=num_images, images=images)
    images = collect_images_with_gradient(cancerous=True ,grayscale=True, good_model=True, num_images=num_images, images=images)

    show_saliencies(images)


if __name__ == "__main__":
    helper.define_dataset_location()
    run_saliency_visualizer()

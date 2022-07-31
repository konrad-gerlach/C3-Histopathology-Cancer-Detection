from __future__ import print_function, division
import re
import zipfile
import torchvision
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import os
import pandas as pd
import config


# https://lindevs.com/download-dataset-from-kaggle-using-api-and-python/
# https://www.kaggle.com/docs/api
# https://github.com/Kaggle/kaggle-api/blob/master/kaggle/api/kaggle_api_extended.py

def unzip_competition_files(competition, path):
    outfile = os.path.join(path, competition + '.zip')
    try:
        with zipfile.ZipFile(outfile) as z:
            z.extractall(path)
    except zipfile.BadZipFile as e:
        raise ValueError(
            'Bad zip file, please report on '
            'www.github.com/kaggle/kaggle-api', e)
    try:
        os.remove(outfile)
    except OSError as e:
        print('Could not delete zip file, got %s' % e)


def load_competition_from_kaggle(competition, path):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(competition, path, quiet=False)
    unzip_competition_files(competition, path)


def load_cancer_ds():
    competition = 'histopathologic-cancer-detection'
    path = config.DATA_CONFIG["ds_path"]

    if not os.path.exists(path):
        load_competition_from_kaggle(competition, path)

#decorates another Dataset and caches its results
class CachingDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = {}
    
    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self,idx):
        if idx not in self.cache:
            self.cache[idx] = self.dataset.__getitem__(idx)
        return self.cache[idx]

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class CancerDataset(Dataset):
    def __init__(self, path, csvfile, transform=None):
        self.path = path
        self.labels_frame = pd.read_csv(csvfile)
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.path,
                                self.labels_frame.iloc[idx, 0] + ".tif")
        image = io.imread(img_name)
        label = self.labels_frame.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        sample = [image, label]

        return sample


def get_ds():
    load_cancer_ds()
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Resize([96, 96])])
    path = config.DATA_CONFIG["ds_path"]
    use_cache = config.DATA_CONFIG["use_cache"]
    full_ds = CancerDataset(os.path.join(path, "train"), os.path.join(path, "train_labels.csv"), transforms)
    if use_cache:
        full_ds = CachingDataset(full_ds)
    train_ds, test_ds = split_ds(full_ds)
    return train_ds, test_ds

def split_ds(full_ds):
    train_size = int(config.DATA_CONFIG["train_portion"] * len(full_ds))
    test_size = int(config.DATA_CONFIG["test_portion"] * len(full_ds))
    remainder = len(full_ds) - train_size
    train_ds, remainder_ds = torch.utils.data.random_split(full_ds, [train_size, remainder])
    test_ds, remainder_ds = torch.utils.data.random_split(remainder_ds, [test_size, remainder-test_size])
    return train_ds, test_ds


def get_dl(batch_size, num_workers, pin_memory=True):
    train_ds, test_ds = get_ds()
    img_shape = train_ds[0][0].shape
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_dl, test_dl, img_shape


def show(images, labels):    
    # Here _ means that we ignore (not use) variables
    _, figs = plt.subplots(1, len(images), figsize=(200, 200))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(torchvision.transforms.ToPILImage()(img))
        if lbl == 0:
            lbl = 'No Cancer'
        else:
            lbl = 'Cancer'
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)

if __name__ == "__main__":
    train_dataloader, test_dataloader, img_shape = get_dl(batch_size=4, num_workers=4)

    for batch, (X, y) in enumerate(train_dataloader):
        show(X,y)
        if batch == 0:
            break
    plt.show()
    


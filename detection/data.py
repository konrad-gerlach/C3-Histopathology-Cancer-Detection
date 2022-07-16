from __future__ import print_function, division
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
from kaggle.api.kaggle_api_extended import KaggleApi
from skimage import io, transform
import torch
import os
import pandas as pd

ds_path = 'datasets/cancer'


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
    path = ds_path

    if not os.path.exists(path):
        load_competition_from_kaggle(competition, path)


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
    ds = CancerDataset(os.path.join(ds_path, "train"), os.path.join(ds_path, "train_labels.csv"), transforms)
    return ds


def get_dl(batch_size, num_workers, pin_memory=True):
    ds = get_ds()
    img_shape = ds[0][0].shape
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    return dl, img_shape


if __name__ == "__main__":
    ds = get_ds()
    sample = ds.__getitem__(0)
    print(sample[1])
    plt.imshow(torchvision.transforms.ToPILImage()(sample[0]))
    plt.show()

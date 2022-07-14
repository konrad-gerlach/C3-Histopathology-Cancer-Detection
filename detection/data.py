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
#https://lindevs.com/download-dataset-from-kaggle-using-api-and-python/
#https://www.kaggle.com/docs/api
#https://github.com/Kaggle/kaggle-api/blob/master/kaggle/api/kaggle_api_extended.py

def unzip_competition_files(competition,path):
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

def load_competition_from_kaggle(competition,path):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(competition, path, quiet=False)
    unzip_competition_files(competition,path)

def load_cancer_ds():
    competition = 'histopathologic-cancer-detection'
    path = ds_path

    if not os.path.exists(path):
        load_competition_from_kaggle(competition,path)

#https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class CancerDataset(Dataset):
    def __init__(self,path,csvfile):
        self.path = path
        self.labels = pd.read_csv(csvfile)
    
    def __len__(self):
        return 0
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_ds():
    load_cancer_ds()
    return

if __name__ == "__main__":
    get_ds()
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
from kaggle.api.kaggle_api_extended import KaggleApi
import os

ds_path = 'datasets/cancer'
#https://lindevs.com/download-dataset-from-kaggle-using-api-and-python/
#https://www.kaggle.com/docs/api
def load_competition_from_kaggle(competition,path):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(competition, path)

def load_cancer_ds():
    competition = 'histopathologic-cancer-detection'
    path = ds_path

    if not os.path.exists(path):
        load_competition_from_kaggle(competition,path)

class CancerDataset(Dataset):
    def __init__(self,path):
        self.path = path
    
    def __len__(self):
        return 0
    
    def __getitem__(self,idx):
        return 0


def get_ds():
    load_cancer_ds()
    return CancerDataset()

if __name__ == "__main__":
    get_ds()
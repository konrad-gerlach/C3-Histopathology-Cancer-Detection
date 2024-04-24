## Updates
A revision to the report has been added on 6th of April 2024 and can be found in the file report_revision_2024_04_06.pdf .

## Security Warning / Disclaimer
Please note, that this repository is not being actively maintained and has received security advisories regarding potential vulnerabilities from Github. Therefore, using or executing the code is not advised without appropriate precautions. The disclaimer as part of the license still applies.
# C3-Histopathology-Cancer-Detection
You can find our Git repository at: https://github.com/konrad-gerlach/C3-Histopathology-Cancer-Detection .
Our data source is: https://www.kaggle.com/competitions/histopathologic-cancer-detection/data .
Our code contains links to articles relevant to understanding certain sections.
List of referenced and used resources:
https://stackoverflow.com/questions/63815311/what-is-the-correct-way-to-implement-gradient-accumulation-in
https://pytorch.org/docs/stable/optim.html
https://lindevs.com/download-dataset-from-kaggle-using-api-and-python/
https://www.kaggle.com/docs/api
https://github.com/Kaggle/kaggle-api/blob/master/kaggle/api/kaggle_api_extended.py
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
https://en.wikipedia.org/wiki/Precision_and_recall
https://towardsdatascience.com/saliency-map-using-pytorch-68270fe45e80
https://towardsdatascience.com/saliency-map-using-pytorch-68270fe45e80

Our models can be downloaded from:
https://www.dropbox.com/sh/898sa4puz1bk2xl/AAAyenjUrHw6T48X-9BkBtZJa?dl=0
There are three models:
We used one model for the coulour visualizations contained in our Report (colour net), another model - with identical architecture and trained with the same hyperparameters - was used to report the metrics (AUC score, precision, recall, specificity, F1 score) (metric net) and a third model - with the same architecture as the previous two - was used for the grayscale visualization (grayscale net). All of them use the Big-K (Big-Konrad) architecture. Apart from the grayscale net having used a learning rate of 0.01 instead of 0.001 all models were trained using the same hyperparameters and architecture. Both colour models achieved an accuracy of 95%.

# Setup
create virtual python environment in a desired directory - execute:
```bash
sudo apt-get install -y python3-venv
python3 -m venv pytorch-venv
source pytorch-venv/bin/activate
```
clone this repo into directory and
execute `pip3 install -r requirements.txt`

# Training on Google Colab
1. Runtime >> Change runtime type >> Select GPU
2. Create a new directory (like detection in our Github)
3. Upload all .py files from the detection folder in our GitHub to the new directory. You can edit the files by double-clicking on them (restart the runtime to incorporate changes)
4. Create a new subdirectory in root called .kaggle and upload your kaggle.json there
```bash
!pip3 install wandb
!python3 detection/data.py
!python3 detection/main.py
```
5. You will be asked to log into wandb, use option 2 and input your API key

The whole program execution is determined by the contents of the config.py file to prevent long chains of arguments in the command line.
Optionally you can specify --ds_path (dataset path) to specify the location where your dataset is or should be located.


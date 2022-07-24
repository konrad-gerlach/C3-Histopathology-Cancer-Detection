# C3-Histopathology-Cancer-Detection

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
1. Runtime >> Change runtime type >> GPU auswählen 
2. Neuen ordner erstellen (wie detection in unserem Github)
3. In den Ordner alle .py Dateien aus detection hochladen
4. Bei dem root ordner einen Unterordner .kaggle erstellen und dort kaggle.json hochladen
```bash
!pip3 install wandb
!python3 detection/data.py
!python3 detection/trainer.py
```
5. Dann wird man gefragt, wie man sich bei wandb einloggen will, mit Option 2 und seinen key einfügen

Mit einem Doppelklick auf die Dateien kann man sie direkt dort bearbeiten.

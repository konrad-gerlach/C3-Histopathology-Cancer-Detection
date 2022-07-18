import torch

def predicted_lables(pred):
    pred = torch.sigmoid(pred)
    pred = torch.round(pred, decimals=0)
    return pred
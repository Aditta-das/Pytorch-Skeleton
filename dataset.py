import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
import joblib

class TriggerDataset(Dataset):
    def __init__(self):
        pass
        

    def __len__(self):
        # len of dataset
        pass
        # return len(something)

    def __getitem__(self, idx):
        '''
        idx: get idx value
        '''
        data = None
        label = None
        return {
            "data": torch.tensor(data, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.float)
        }

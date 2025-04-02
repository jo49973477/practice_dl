import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

class LinearRegressionDataset(Dataset):
    def __init__(self, train: bool):
        self.mode = "train" if train else "test"
        self.x = np.load("data/X_{}.npy".format(self.mode))
        self.y = np.load("data/Y_{}.npy".format(self.mode))
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return {"input": torch.Tensor(self.x[idx]), 
                "label": torch.Tensor(self.y[idx])}
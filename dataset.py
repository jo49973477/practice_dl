import os
from PIL import Image
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

class CIFAR10(Dataset):
    def __init__(self, data_dir: str, train: bool):
        self.train_dirs = [os.path.join(data_dir, "data_batch_{}".format(i))
                           for i in range(1, 6)]
        
        self.test_dirs = [os.path.join(data_dir, "test_batch")]
        # directory of train set and test set
        
        self.train = train # if the model is training, it's true
        
        self.labels = []
        self.datas = []
        self.filenames = [] # the list where filenames, datas, labels are stored
        
        self.extract_dataset()
        
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        return {"data": self.datas[idx], 
                "label": self.labels[idx],
                "filename": self.filenames[idx]}
    
    def getImg(self, idx):
        flattened = np.array(self.datas[idx])
        r = flattened[0:1024].reshape(32,32)
        g = flattened[1024:2048].reshape(32,32)
        b = flattened[2048:].reshape(32,32)
        restored_img = np.stack([r,g,b], axis = -1)
        return restored_img
    
    #---- getting dictionary file from the file ----#
    def unpack(self, dir):
        try:
            with open(dir, 'rb') as fo:
                dict = pickle.load(fo, encoding = "latin1")
            return dict
        except:
            raise ValueError("💀Check the dataset directory again! File doesn't exist.💀")
        
    
    #---- extracting dataset from file ----#
        
    def extract_dataset(self):
        dirs = self.train_dirs if self.train else self.test_dirs
        
        for dir in dirs:
            data_dict = self.unpack(dir)
            self.labels += data_dict["labels"]
            self.datas += data_dict["data"].tolist()
            self.filenames += data_dict["filenames"]
        
        assert len(self.labels) == len(self.datas), "💀Labels and datas must have same number.💀"
        
if __name__ == "__main__":
    dataset = CIFAR10("/home/yeongyoo/cifar-10-batches-py", train=True)
    idx2val = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    print("dataset_len:", len(dataset))
    
    for i in range(10):
        data = dataset[i]
        plt.subplot(5, 2, i+1)
        plt.imshow(dataset.getImg(i))
        plt.title(idx2val[data['label']])
        
    plt.show()
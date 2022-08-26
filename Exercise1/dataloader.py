from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torchvision import datasets, transforms
from PIL import images
import glob

class CustomDataset(Dataset):
    def __init__(self, path, n_classes=10, transform = False):
        
        self.transform = transform
        self.filelist = glob.glob(path) 
        
        self.labels =  #... load the labels (copy from the notebook)
        
    def __len__(self):
       
        return len(self.filelist)

    def __getitem__(self, idx):
        
        img = Image.open(self.filelist[idx])
        
        x = #.... transform to tensor and flatten it to a tensor of 69*69 = 4761
        
        x = # flatten img
        
        y = self.labels[idx]
    
        return x, y
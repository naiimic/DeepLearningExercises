
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
import glob

class CustomDataset(Dataset):
    def __init__(self, path, n_classes = 10, transform = False):
        
        self.transform = transform
        
        self.filelist = glob.glob(path+'/*.png') 

        self.labels =  np.zeros(len(self.filelist))
        
        for class_i in range(n_classes):
            files_that_are_of_this_class = ['class'+str(class_i) in x for x in self.filelist]
            self.labels[ files_that_are_of_this_class ] = class_i
            
        self.labels = torch.LongTensor(self.labels)

    def __len__(self):
       
        return len(self.filelist)

    def __getitem__(self, idx):
        
        img = Image.open(self.filelist[idx])
        
        if self.transform == True:
            img = transforms.RandomRotation(180)( img )
            
        x = transforms.ToTensor()( img )
        # ... repeat x over 3 dimensions...
        
        y = self.labels[idx]
    
        return x, y
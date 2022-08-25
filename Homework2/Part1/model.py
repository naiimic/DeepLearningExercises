
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Conv2d,ReLU,MaxPool2d,Linear

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # take the part of the model you decided 
        self.features = nn.Sequential(
            ...
        )

        # build a simple FC that takes as input the flatten output of the features layers
        # the output should be 10
        self.classifier = nn.Sequential(
            
        )
        
    def forward(self,x):
        
        out = self.features(x)
        out = torch.flatten(out,1)
        out = self.classifier(out)
        
        return out
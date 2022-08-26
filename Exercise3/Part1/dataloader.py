
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import glob
import dgl
import pandas as pd

def collate_graphs(samples):
    
    # The input `samples` is a list, a batch of whatever comes out of your dataset object
    
    graphs = [x[0] for x in samples]
    labels = [x[1] for x in samples]
    
    batched_graph = dgl.batch(graphs)
    targets = torch.tensor(labels).long()
    
    return batched_graph, targets

class PointCloudMNISTdataset(Dataset):
    def __init__(self, path):
         
        self.df = pd.read_hdf(path)
        self.labels = torch.LongTensor(self.df.label)
        self.n_points = self.df.n_points.values
        
    def __len__(self):
       
        return len(self.n_points)

    def __getitem__(self, idx):
        
        node_xy = self.df.iloc[idx]['xy']
        
        g = dgl.graph(([],[]),num_nodes=self.n_points[idx])
        
        g.ndata['xy'] = torch.tensor(node_xy).float()
        
        y = self.labels[idx]
        
        return g, y
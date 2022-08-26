
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import glob
import dgl
import networkx as nx
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_graph(g_input,ax,size=1):
    g = g_input.to(torch.device('cpu'))
    x = g.nodes['points'].data['xy'][:,0].data.numpy()
    y = g.nodes['points'].data['xy'][:,1].data.numpy()
    object_centers = g.nodes['objects'].data['centers'].data.numpy()
    object_width = g.nodes['objects'].data['width'].data.numpy()
    object_height = g.nodes['objects'].data['height'].data.numpy()

    ax.scatter(x,y,c='cornflowerblue',cmap='tab10',s=size)
    
    for i in range(len(object_height)):
    
        bounding_box = patches.Rectangle((object_centers[i][0]-object_width[i]/2, object_centers[i][1]-object_height[i]/2), 
                             object_width[i], object_height[i], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(bounding_box)
        
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)

def collate_graphs(samples):
    
    # The input `samples` is a list, a batch of whatever comes out of your dataset object
    
    batched_graph = dgl.batch(samples)

    return batched_graph

class RandomShapeDataset(Dataset):
    def __init__(self,path):

        self.graphs, _ = dgl.data.utils.load_graphs(path)
        # if torch.cuda.is_available():
        #     self.graphs = [g.to(torch.device('cuda')) for g in self.graphs]
    def __len__(self):
       
        return len(self.graphs)

    def __getitem__(self, idx):

        return self.graphs[idx]
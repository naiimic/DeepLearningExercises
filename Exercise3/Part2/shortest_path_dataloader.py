
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import glob
import dgl
import networkx as nx
from tqdm import tqdm
import json

def collate_graphs(samples):
    
    # The input `samples` is a list, a batch of whatever comes out of your dataset object
    
    batched_graph = dgl.batch(samples,ndata=['node_features','on_path'],edata=['distance','on_path'])

    return batched_graph

class ShortestPathDataset(Dataset):
    def __init__(self, path,limit=-1):
        
        filelist = glob.glob(path+'/*.json')
        
        if limit > 0:
            filelist = filelist[:limit]
        
        self.graphs = []
        for fname in tqdm(filelist):
            with open(fname) as jfile:
                json_data = json.load(jfile)
                json_data['directed'] = 'true'
                graph = nx.node_link_graph(json_data)
                g = dgl.from_networkx(graph, node_attrs=['node_features'],edge_attrs=['distance','on_path'])
                
                edge_distance = torch.cat([g.edata['distance'],g.edata['distance']],dim=0)
            
                edge_target = torch.cat([g.edata['on_path'],g.edata['on_path']],dim=0)
                
                g.add_edges(g.edges()[1],g.edges()[0])
                
                g.edata['distance'] = edge_distance
                g.edata['on_path'] = edge_target.float()
                
                g.update_all(dgl.function.copy_edge('on_path','on_path'),dgl.function.max('on_path','on_path'))
                self.graphs.append(g)
        
        
    def __len__(self):
       
        return len(self.graphs)


    def __getitem__(self, idx):
        
    
        return self.graphs[idx]
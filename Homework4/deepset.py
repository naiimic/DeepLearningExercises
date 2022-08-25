
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import dgl

# You can be smart and use the same as the previous homework...
# Just make sure to change g.ndata with g.nodes[...].data[...] etc. (it's a heterogenous graph)

class Deepset(nn.Module):
    def __init__(self):
        super().__init__()
        
        input_size = 2
        hidden_size = 50
        
        self.node_init = nn.Linear(input_size,hidden_size)
        self.hidden_layers = nn.ModuleList()
        
        for i in range(2):
            self.hidden_layers.append( 
                nn.Sequential(
                     ## hidden_size x 2 -> hidden_size -> hidden_size... (5 times in total)
                    nn.BatchNorm1d(hidden_size)
                )
            )
            
    def forward(self,g):
        
        g.nodes['points'].data['hidden rep'] = self.node_init(g.nodes['points'].data['xy'])
        
        for layer_i, layer in enumerate(self.hidden_layers):
                                
            mean_of_node_rep = dgl.mean_nodes(g,'hidden rep',ntype = 'points')
            broadcasted_mean = dgl.broadcast_nodes(g,mean_of_node_rep,ntype = 'points')
            
            ...

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d,ReLU,MaxPool2d,Linear
import dgl

# This is the edge network..

class EdgeNetwork(nn.Module):
    def __init__(self,inputsize,hidden_layer_size,output_size):
        super().__init__()
    
        self.net = nn.Sequential(
            # ...
        )
        
    def forward(self, x):
        
        # x.dst['node_features'], x.src['node_features'], x.dst['node_hidden_rep'], x.src['node_hidden_rep'], x.data['distance'], x.data['prediction'] ....         
        input_data = torch.cat([ ... ],dim=1)
                                      
        # use a neural network to create an edge hidden representation
        # you return a dictionary with what you want to "send" to the reciving node
                
        #print(input_data.shape)
        output_data = self.net(input_data)

        return {'edge hidden representation': output_data }

# This is the node network..
    
class NodeNetwork(nn.Module):
    def __init__(self,inputsize,hidden_layer_size,output_size):
        super().__init__()

        self.net = nn.Sequential(
            ...
        )
        
    def forward(self, x):
        
        # this time your input x has the information on what has been sent to it by the edges
        # x.mailbox['edge hidden represetation'] -> (Batch size, number of nodes in neighborhood, edge hidden rep size)
        
        message_sum = torch.sum(... ,dim=1)
        
        # you need to torch.cat the:
        #  - message sum
        #  - the current hidden rep of nodes (x.data['node_hidden_rep']
        #  - the node features 

        input_data = torch.cat([...],dim=1)
             
        # and then apply some fully connected neural network    

        out = self.net(input_data)
        
        return {'node_hidden_rep': out }


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        # node_representation_size = ...
        # hidden_size = ..
        # output_size_edges_message = ..
        # output_size_nodes = ...
        
        #output classifier should be 1!
        output_size_edges_classifier = 1 
        output_size_nodes_classifier = 1

        # a network to init the hidden rep of the nodes
        self.node_init = nn.Sequential(
            nn.Linear(2,50),
            nn.ReLU(),
            nn.Linear(50,node_representation_size)
        )
        
        # Creating the message
        self.edge_network = EdgeNetwork(...) # You need to understand which dimension in input to give, which hidden and output...
        # Recieving the message
        self.node_network = NodeNetwork(...) # You need to understand which dimension in input to give, which hidden and output...
        
        # Classification
        self.edge_classifier = EdgeNetwork(...)
        
        self.node_classifier = nn.Sequential(...
        )
                
    def forward(self, g):
        
        g.ndata['node_hidden_rep'] = self.node_init(g.ndata['node_features'])

        g.edata['prediction'] = torch.zeros(g.num_edges(),device=g.device)
        g.ndata['prediction'] = torch.zeros(g.num_nodes(),device=g.device)
       
        gn_blocks_iterations = # number of Graph block iterations, you decide, I did 7... 
        
        for i in range(gn_blocks_iterations):
                        
            g.update_all(self.edge_network,self.node_network) #Build in function that updates the edge and nodes network..

            g.apply_edges(...) #Edge classifier
            
            prediction = g.edata['edge hidden representation'].view(-1)
            g.edata['prediction'] += prediction
            
            g.ndata['prediction'] += self.node_classifier(g.ndata['node_hidden_rep']).view(-1)
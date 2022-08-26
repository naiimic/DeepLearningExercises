import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import dgl

from deepset import Deepset

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

class Set2SetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.crossentropy = nn.CrossEntropyLoss(reduction='none')
        self.regression_loss = nn.MSELoss(reduction='none')

    def compute_pair_loss(self,edges):

        target_properties = torch.cat([edges.dst['height'].unsqueeze(1),
                                           edges.dst['width'].unsqueeze(1),
                                           edges.dst['centers'] ],dim=1)
        properties_loss = torch.sum( self.regression_loss(edges.src['properties'],target_properties) ,dim=1)
      
        loss = properties_loss
        
        return {'loss': loss}

    def forward(self, g):
        
        g.apply_edges(self.compute_pair_loss,etype='pred_to_target')
        
        data = g.edges['pred_to_target'].data['loss'].cpu().data.numpy()+0.00000001
        u = g.all_edges(etype='pred_to_target')[0].cpu().data.numpy().astype(int)
        v = g.all_edges(etype='pred_to_target')[1].cpu().data.numpy().astype(int)
        m = csr_matrix((data,(u,v)))
        
        selected_columns = min_weight_full_bipartite_matching(m)[1]

        n_objects_per_event = [n.item() for n in g.batch_num_nodes('objects')]
        col_offest = np.repeat( np.cumsum([0]+n_objects_per_event[:-1]), n_objects_per_event)
        row_offset = np.concatenate([[0]]+[[n]*n for n in n_objects_per_event])[:-1]
        row_offset = np.cumsum(row_offset)

        edge_indices = selected_columns-col_offest+row_offset

        g.nodes['predicted objects'].data['loss'] = g.edges['pred_to_target'].data['loss' ][edge_indices] 
        
        loss = dgl.sum_nodes(g,'loss',ntype='predicted objects').mean()

        return loss 


        
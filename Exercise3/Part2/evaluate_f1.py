import torch
import numpy as np
from shortest_path_dataloader import ShortestPathDataset, collate_graphs
from MPNN_model import Classifier
import sys
from torch.utils.data import Dataset, DataLoader

def evaluate_on_dataset(path_to_ds=None):

    test_ds = ShortestPathDataset(path_to_ds)
    dataloader = DataLoader(test_ds,batch_size=300,collate_fn=collate_graphs)

    net = Classifier()

    net.load_state_dict(torch.load('trained_model.pt',map_location=torch.device('cpu')))

    net.eval()

    edge_true_pos = 0
    edge_false_pos = 0
    edge_false_neg = 0

    node_true_pos = 0
    node_false_pos = 0
    node_false_neg = 0

    with torch.no_grad():
        for batched_g in dataloader:

            net(batched_g)

            edge_target = batched_g.edata['on_path']
            edge_pred = batched_g.edata['prediction']

            node_target = batched_g.ndata['on_path']
            node_pred = batched_g.ndata['prediction']

            edge_true_pos+=len(torch.where( (edge_pred>0) & (edge_target==1) )[0])
            edge_false_pos+=len(torch.where( (edge_pred>0) & (edge_target==0) )[0])
            edge_false_neg+=len(torch.where( (edge_pred<0) & (edge_target==1) )[0])

            node_true_pos+=len(torch.where( (node_pred>0) & (node_target==1) )[0])
            node_false_pos+=len(torch.where( (node_pred>0) & (node_target==0) )[0])
            node_false_neg+=len(torch.where( (node_pred<0) & (node_target==1) )[0])

    f1_edge = edge_true_pos/(edge_true_pos+0.5*(edge_false_pos+edge_false_neg))
    f1_node = node_true_pos/(node_true_pos+0.5*(node_false_pos+node_false_neg))

    return f1_edge, f1_node

if __name__ == "__main__":

    f1_edge, f1_node = evaluate_on_dataset(sys.argv[1])

    print(f1_edge, f1_node)


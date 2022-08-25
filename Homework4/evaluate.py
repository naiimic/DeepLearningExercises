import torch
import numpy as np
from dataloader import RandomShapeDataset, collate_graphs
from model import Net
import sys
from torch.utils.data import Dataset, DataLoader
from loss import Set2SetLoss

def evaluate_on_dataset(path_to_ds=None):

    loss_func = Set2SetLoss()
    test_ds = RandomShapeDataset(path_to_ds)
    dataloader = DataLoader(test_ds,batch_size=300,collate_fn=collate_graphs)

    net = Net()

    net.load_state_dict(torch.load('trained_model.pt',map_location=torch.device('cpu')))

    net.eval()

    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batched_g in dataloader:
            n_batches+=1

            predicted_g,size_pred = net(batched_g,use_target_size=True)

            loss = loss_func(batched_g) 

            total_loss+=loss.item()


    return total_loss/n_batches

if __name__ == "__main__":

    avg_loss = evaluate_on_dataset(sys.argv[1])

    print(avg_loss)


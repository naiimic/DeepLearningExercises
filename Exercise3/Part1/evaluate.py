import torch
import numpy as np
from dataloader import PointCloudMNISTdataset, collate_graphs
from model import Net
import sys
from torch.utils.data import Dataset, DataLoader

def evaluate_on_dataset(path_to_ds=None):

    test_ds = PointCloudMNISTdataset(path_to_ds)
    dataloader = DataLoader(test_ds,batch_size=300,collate_fn=collate_graphs)

    net = Net()

    net.load_state_dict(torch.load('trained_model.pt',map_location=torch.device('cpu')))

    net.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for x,y in dataloader:

            pred = net(x)
            pred = torch.argmax(pred,dim=1)
            correct+=len(torch.where(pred==y)[0])
            total+=len(y)

    return correct/total

if __name__ == "__main__":

    accuracy = evaluate_on_dataset(sys.argv[1])

    print(accuracy)


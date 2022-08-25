import torch.nn as nn

# Make it simple at the beginning..

class Net(nn.Module):
    
    '''
        input = dimension 69*69
        output = dimension 10
    '''
    
    def __init__(self):
        super(Net,self).__init__()
        
        self.layer1 = #.... nn.Linear(69*69,....)
        # nn.ReLU()
        #....
        # at the end you want a nn.Linear(...,10) (NO RELU AFTER)
    
    def forward(self,x):
        
        # ... you need to pass the input through all the layers

        return out
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class PreProcess(nn.Module):
    def __init__(self):
        super().__init__()

        pass

    def forward(self, state):

        if state is None:
            return torch.zeros(75, 80)

        img = state[35:185] # crop 
        img = img[::2,::2,0] # downsample by factor of 2.
        img[img == 144] = 0 # erase background (background type 1)
        img[img == 109] = 0 # erase background (background type 2)
        img[img != 0] = 1 # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively
        
        return torch.from_numpy(img.astype(np.float32)).float()

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(6000*2, 512), 
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self,state,previous_state):

        combined = torch.cat([previous_state,state],dim=1)
        logits = self.layers(combined)

        return logits

    def sample_action(self,state,previous_state):

        logits = self(state,previous_state)

        # Converts these number to probability
        c = torch.distributions.Categorical(logits=logits)
        # We sample based on the probability
        action = int(c.sample().cpu().numpy()[0])
        action_prob = float(c.probs[0, action].detach().cpu().numpy())

        return action, action_prob

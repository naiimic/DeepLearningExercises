import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from torch.utils.data import Dataset, DataLoader

class GamesMemoryBank(Dataset):

    def __init__(self):

        self.gamma = 0.99
        self.clear_memory()

    def clear_memory(self):

        self.state_history = []
        self.previous_state_history = []
        self.action_history = []
        self.action_prob_history = []
        self.reward_history = []

    def add_event(self, state, previous_state, action, action_prob, reward):

        self.state_history.append(state)
        self.previous_state_history.append(previous_state)
        self.action_history.append(action)
        self.action_prob_history.append(action_prob)
        self.reward_history.append(reward)

    def compute_reward_history(self):

        R = 0
        self.discounted_rewards = []

        for r in self.reward_history[::-1]:
            if r != 0: 
                R = 0 # scored/lost a point in pong, so reset reward sum
            R = r + self.gamma * R
            self.discounted_rewards.insert(0, R)
        
        self.discounted_rewards = torch.tensor(self.discounted_rewards).float()
        self.discounted_rewards = (self.discounted_rewards - self.discounted_rewards.mean()) / self.discounted_rewards.std()

    def __len__(self):
        
        return len(self.state_history)

    def __getitem__(self, idx):

        state = self.state_history[idx]
        previous_state = self.previous_state_history[idx]
        action = torch.tensor( self.action_history[idx] )
        action_prob = torch.tensor( self.action_prob_history[idx] )
        reward = torch.tensor( self.reward_history[idx] )
        discounted_reward = self.discounted_rewards[idx]

        return state, previous_state, action, action_prob, reward, discounted_reward

    def get_sample(self,batch_size):

        idxs = np.random.permutation(range(len(self.state_history)))[:batch_size]

        state = torch.stack( [ self.state_history[idx] for idx in  idxs ],dim=0)
        previous_state = torch.stack( [ self.previous_state_history[idx]  for idx in  idxs ],dim=0 )
        action = torch.stack( [ torch.tensor(self.action_history[idx])  for idx in  idxs ] )
        action_prob = torch.stack( [ torch.tensor(self.action_prob_history[idx])   for idx in  idxs ])
        reward = torch.stack( [ torch.tensor(self.reward_history[idx])   for idx in  idxs ] )

        discounted_reward = torch.stack([ self.discounted_rewards[idx] for idx in  idxs ] )

        return state, previous_state, action, action_prob, reward, discounted_reward

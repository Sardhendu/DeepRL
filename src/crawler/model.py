import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    The actor outputs the best action to take based on the
    """
    def __init__(self, state_size, action_size, seed, fc1_units=512, fc2_units=256):
        self.seed = torch.manual_seed(seed)
        self.bn1 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.out = nn.Linear(fc2_units, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(self.bn1(state)))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return F.tanh(x)


class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=512, fc2_units=256):
        self.seed = torch.manual_seed(seed)
        self.bn1 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.out = nn.Linear(fc2_units, action_size)
    
    def forward(self, state):
        """
        The idea of a critic is t
        :param state:
        :return:
        """
        x = F.relu(self.fc1(self.bn1(state)))
        x = F.relu(self.fc2(x))
        return self.out(x)
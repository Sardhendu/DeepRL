
import torch
import torch.nn as nn
import torch.nn.functional as F



class Actor(nn.Module):
    def __init__(self, state_size, action_size, layer_in_out, seed):
        super(Actor, self).__init__()
        print('[Actor] Initializing the Actor network ..... ')
        self.seed = torch.manual_seed(seed)
        
        self.bn1 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, layer_in_out[0])
        self.fc2 = nn.Linear(layer_in_out[0], layer_in_out[1])
        self.fc3 = nn.Linear(layer_in_out[1], action_size)
        
    def forward(self, states):
        x = F.relu(self.fc1(self.bn1(states)))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x
        
        
class Critic(nn.Module):
    def __init__(self, state_size, action_size, layer_in_out, seed):
        super(Critic, self).__init__()
        print('[Critic] Initializing the Critic Network')
        self.seed = torch.manual_seed(seed)
        self.bn1 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, layer_in_out[0])
        self.fc2 = nn.Linear(layer_in_out[0] + action_size, layer_in_out[1])
        self.fc3 = nn.Linear(layer_in_out[1], 1)
        
    def forward(self, states, actions):
        xs = F.relu(self.fc1(self.bn1(states)))
        # xs = self.drop1(xs)
        x = torch.cat((xs, actions), dim=1)
        x = F.relu(self.fc2(x))
        # x = self.drop2(x)
        return self.fc3(x)

        
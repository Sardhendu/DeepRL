
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
        This module would take input as the state vector and output a action vector
    """
    def __init__(self, state_size, action_size, seed, network_name):
        '''
        :param state_size:
        :param fc_layers:
        :param action_size:
        :param seed:

        Initialize Network Layer Objects
        '''
        
        super(QNetwork, self).__init__()
        
        self.network_name = network_name
        print('[INIT] Initializing Network (%s) .... .... ....'%str(network_name))
        
        if network_name == 'net1':
            self.net1(state_size, action_size, seed)
        elif network_name == 'net2':
            self.net2(state_size, action_size, seed)
        else:
            raise ValueError('Network Name not Understood')
        
    def net1(self, state_size, action_size, seed):
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def net2(self, state_size, action_size, seed):
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)
        
    def forward_net1(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_net2(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

    def forward(self, x):
        '''
        :param x:
        :return:
            Outputs the relu activation for the final layer.
        '''
        if self.network_name == 'net1':
            return self.forward_net1(x)
        elif self.network_name == 'net2':
            return self.forward_net2(x)
        else:
            raise ValueError('Proper net name required')
        



class Optimize:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def adam(self, params):
        return optim.Adam(params, lr=self.learning_rate)


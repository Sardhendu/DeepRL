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
        self.seed = torch.manual_seed(seed)
        self.network_name = network_name
        print('[INIT] Initializing Network (%s) .... .... ....' % str(network_name))
        
        if network_name == 'net1':
            self.net1(state_size, action_size)
            self.forward_func = self.forward_net1
        elif network_name == 'net2':
            self.net2(state_size, action_size)
            self.forward_func = self.forward_net2
        else:
            raise ValueError('Network Name not Understood')
    
    def net1(self, state_size, action_size):
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def net2(self, state_size, action_size):
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, action_size)
    
    def forward_net1(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
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
        return self.forward_func(x)


class VisualQNEtwork(nn.Module):
    def __init__(self, state_size, action_size, seed, network_name):
        super(VisualQNEtwork, self).__init__()
        
        print('[INIT] Initializing Visual QNetwork (%s) .... .... ....' % str(network_name))
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        if network_name == 'net1':
            self.net1(seed)
        else:
            raise ValueError('No other network initialized')
    
    def net1(self, seed):
        """

        :return:
        According to DQN paper:
        1. First hidden layer, kernel = 8x8x32, stride = 4 + RELU
        2. Second hidden layer, kernel = 4x4x64, stride = 2 + RELU
        3. Third hident layer, kernel = 3x3x64, stride = 1 + RELU
        4. Fully Connected layer with 512 uints lead to the the final 1 neuron output that outputs the action
        probability distribution.
        """
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(1, 8, 8), stride=(1, 4, 4))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(1, 4, 4), stride=(1, 2, 2))
        self.conv3 = nn.Conv3d(64, 64, kernel_size=(4, 3, 3), stride=(1, 1, 1))
        
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(64)
    
    def forward(self, x):
        # print('2347023 ', x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        # print(x.shape)
        
        # Flatten
        x = x.view(x.size(0), -1)
        # print()
        x = nn.Linear(len(x[0]), 512)(x)
        x = F.relu(x)
        x = nn.Linear(512, 4)(x)
        # print(x.shape)
        
        return x


# import sys
# sys.path.insert(0,'/Users/sam/All-Program/App/deep-reinforcement-learning/')

class Optimize:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def adam(self, params):
        return optim.Adam(params, lr=self.learning_rate)
    
    def rmsprop(self, params):
        return optim.RMSprop(params, lr=self.learning_rate)  # default smoothing=0.99, eps = 1e-08, momentum=0




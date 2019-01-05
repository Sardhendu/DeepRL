
from torch import nn
import torch.nn.functional as F
import torch.optim as optim



class DDPGActor(nn.Module):
    def __init__(self, state_size, action_size):
        """
        
        :param state_size:
        :param action_size:
        
        """

        super(DDPGActor, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x)) # Since we have to squash actor action output within -1 to 1
        
        return x
    
    
class DDPGCritic:
    def __init__(self):
        pass
    
    def forward(self):
        pass


class Optimize:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def adam(self, params):
        return optim.Adam(params, lr=self.learning_rate)
    
    def rmsprop(self, params):
        return optim.RMSprop(params, lr=self.learning_rate)  # default smoothing=0.99, eps = 1e-08, momentum=0



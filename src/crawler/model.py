import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    The actor outputs the best action to take based on the
    """
    def __init__(self, state_size, action_size, seed, fc1_units=512, fc2_units=256):
        super(Actor, self).__init__()
        
        print('[Actor] Initializing the Actor network ..... ')
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
        """ THe critic plays the role of an examiner that evaluated the actor's output
        
        The actor given a state (st) decides the opportune action (at) and gets the reward(r_t+1).
        The Critic computes the value of taking the same state and action <st, at>.

        :param state_size:
        :param action_size:
        :param seed:
        :param fc1_units:
        :param fc2_units:
        """
        super(Critic, self).__init__()
    
        print('[Actor] Initializing the Critic network ..... ')
        self.seed = torch.manual_seed(seed)
        self.bn1 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.out = nn.Linear(fc2_units, 1)
    
    def forward(self, state, action):
        """
        The idea of a critic is t
        :param state:
        :return:
        """
        xs = F.relu(self.fc1(self.bn1(state)))
        # xs = self.drop1(xs)
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.out(x)
    
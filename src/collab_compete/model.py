import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    def __init__(self, state_size, action_size, layer_in_out, seed):
        super(Actor, self).__init__()
        print('[Actor] Initializing the Actor network ..... ')
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, layer_in_out[0])
        self.bn1 = nn.BatchNorm1d(layer_in_out[0])
        self.fc2 = nn.Linear(layer_in_out[0], layer_in_out[1])
        self.fc3 = nn.Linear(layer_in_out[1], action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        if states.dim() == 1:
            states = torch.unsqueeze(states, 0)
        x = self.bn1(F.relu(self.fc1(states)))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x



class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, layer_in_out, seed):
        """Initialize parameters and build model.
        Params
        ======
            input_size (int): number of dimensions for input layer
            seed (int): random seed
            fc1_units (int): number of nodes in the first hidden layer
            fc2_units (int): number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear((state_size + action_size)*2, layer_in_out[0])
        self.fc2 = nn.Linear(layer_in_out[0], layer_in_out[1])
        self.fc3 = nn.Linear(layer_in_out[1], 1)
        self.bn1 = nn.BatchNorm1d(layer_in_out[0])
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights with near zero values."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        """Build a critic network that maps (states, actions) pairs to Q-values."""
        # #print('Model Critic', states.shape, actions.shape)
        # print(states.shape, actions.shape)
        xs = torch.cat((states, actions), dim=1)
        # print('34343 ', xs.shape)
        # print(self.fc1(xs).shape)
        x = F.relu(self.fc1(xs))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# class Actor_Critic_Models():
#     """
#     Create object containing all models required per DDPG agent:
#     local and target actor and local and target critic
#     """
#
#     def __init__(self, n_agents, state_size=24, action_size=2, seed=0):
#         """
#         Params
#         ======
#             n_agents (int): number of agents
#             state_size (int): number of state dimensions for a single agent
#             action_size (int): number of action dimensions for a single agent
#             seed (int): random seed
#         """
#         self.actor_local = Actor(state_size, action_size, seed).to(device)
#         self.actor_target = Actor(state_size, action_size, seed).to(device)
#         critic_input_size = (state_size + action_size) * n_agents
#         self.critic_local = Critic(critic_input_size, seed).to(device)
#         self.critic_target = Critic(critic_input_size, seed).to(device)


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# def hidden_init(layer):
#     fan_in = layer.weight.data.size()[0]
#     lim = 1. / np.sqrt(fan_in)
#     return (-lim, lim)
#
#
# class Actor(nn.Module):
#     """Actor (Policy) Model."""
#
#     def __init__(self, input_size, output_size, seed, fc1_units=256, fc2_units=256):
#         """Initialize parameters and build actor model.
#         Params
#         ======
#             input_size (int):  number of dimensions for input layer
#             output_size (int): number of dimensions for output layer
#             seed (int): random seed
#             fc1_units (int): number of nodes in first hidden layer
#             fc2_units (int): number of nodes in second hidden layer
#         """
#         super(Actor, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(input_size, fc1_units)
#         self.fc2 = nn.Linear(fc1_units, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, output_size)
#         self.bn = nn.BatchNorm1d(fc1_units)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         """Initialize weights with near zero values."""
#         self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
#         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
#         self.fc3.weight.data.uniform_(-3e-3, 3e-3)
#
#     def forward(self, state):
#         """Build an actor network that maps states to actions."""
#         # #print('Model Actor: ', state.shape)
#         if state.dim() == 1:
#             state = torch.unsqueeze(state,0)
#         x = F.relu(self.fc1(state))
#         x = self.bn(x)
#         x = F.relu(self.fc2(x))
#         x = F.tanh(self.fc3(x))
#         return x
#
#
# class Critic(nn.Module):
#     """Critic (Value) Model."""
#
#     def __init__(self, input_size, seed, fc1_units=256, fc2_units=256):
#         """Initialize parameters and build model.
#         Params
#         ======
#             input_size (int): number of dimensions for input layer
#             seed (int): random seed
#             fc1_units (int): number of nodes in the first hidden layer
#             fc2_units (int): number of nodes in the second hidden layer
#         """
#         super(Critic, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(input_size, fc1_units)
#         self.fc2 = nn.Linear(fc1_units, fc2_units)
#         self.fc3 = nn.Linear(fc2_units, 1)
#         self.bn1 = nn.BatchNorm1d(fc1_units)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         """Initialize weights with near zero values."""
#         self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
#         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
#         self.fc3.weight.data.uniform_(-3e-3, 3e-3)
#
#     def forward(self, states, actions):
#         """Build a critic network that maps (states, actions) pairs to Q-values."""
#         # #print('Model Critic', states.shape, actions.shape)
#         # print(states.shape)
#         xs = torch.cat((states, actions), dim=1)
#         x = F.relu(self.fc1(xs))
#         x = self.bn1(x)
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
# class Actor_Critic_Models():
#     """
#     Create object containing all models required per DDPG agent:
#     local and target actor and local and target critic
#     """
#
#     def __init__(self, n_agents, state_size=24, action_size=2, seed=0):
#         """
#         Params
#         ======
#             n_agents (int): number of agents
#             state_size (int): number of state dimensions for a single agent
#             action_size (int): number of action dimensions for a single agent
#             seed (int): random seed
#         """
#         self.actor_local = Actor(state_size, action_size, seed).to(device)
#         self.actor_target = Actor(state_size, action_size, seed).to(device)
#         critic_input_size = (state_size + action_size) * n_agents
#         self.critic_local = Critic(critic_input_size, seed).to(device)
#         self.critic_target = Critic(critic_input_size, seed).to(device)







# class CentralizedCritic(nn.Module):
#     def __init__(self, state_size, action_size, layer_in_out, seed):
#         super(CentralizedCritic, self).__init__()
#         print('[Critic] Initializing the Critic Network')
#         self.seed = torch.manual_seed(seed)
#         self.bn1 = nn.BatchNorm1d(state_size)
#         self.fc1 = nn.Linear(state_size, layer_in_out[0])
#         # self.fc2 = nn.Linear(layer_in_out[0] + action_size, layer_in_out[1])
#         self.fc2 = nn.Linear(layer_in_out[0] + action_size, layer_in_out[1])
#         self.fc3 = nn.Linear(layer_in_out[1], 1)
#         self.reset_parameters()
#
#
#     def reset_parameters(self):
#         self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
#         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
#         self.fc3.weight.data.uniform_(-1e-3, 1e-3)
#
#
#     def forward(self, states, actions):
#         xs = F.relu(self.fc1(self.bn1(states)))
#         # xs = self.drop1(xs)
#         x = torch.cat((xs, actions), dim=1)
#         x = F.relu(self.fc2(x))
#         # x = self.drop2(x)
#         return self.fc3(x)


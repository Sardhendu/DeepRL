#
# import numpy as np
# import torch
# from torch import nn
# import torch.nn.functional as F
import torch.optim as optim
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# #
# #
# #
# # def hidden_init(layer):
# #     fan_in = layer.weight.data.size()[0]
# #     lim = 1. / np.sqrt(fan_in)
# #     return (-lim, lim)
# #
# #
# # class DDPGActor(nn.Module):
# #     def __init__(self, state_size, action_size, seed):
# #         """
# #         :param state_size:      Dimension of state
# #         :param action_size:     Number of action
# #         """
# #         super(DDPGActor, self).__init__()
# #         self.seed = torch.manual_seed(seed)
# #         self.fc1 = nn.Linear(state_size, 400)
# #         self.bn1 = nn.BatchNorm1d(128)
# #         self.fc2 = nn.Linear(128, 128)
# #         self.bn2 = nn.BatchNorm1d(128)
# #         self.fc3 = nn.Linear(128, action_size)
# #         self.reset_parameters()
# #
# #     def reset_parameters(self):
# #         self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
# #         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
# #         self.fc3.weight.data.uniform_(-3e-3, 3e-3)
# #
# #     def forward(self, x):
# #         """Forward propagation
# #
# #         :param x:
# #         :return:
# #
# #         Since we are dealing with continuous space we have to use a tanh nonlinearity. The tanh nonlinearity squashes the output between -1 and 1. We would also go away with a
# #         sigmoid depending on our action value limits.
# #
# #         If the limit says -n to n, then we just multiply n to the tanh output.
# #         """
# #         x = F.relu(self.bn1(self.fc1(x)))
# #         x = F.relu(self.bn2(self.fc2(x)))
# #         return F.tanh(self.fc3(x))
# #
# #
# # class DDPGCritic(nn.Module):
# #     def __init__(self, state_size, action_size, seed):
# #         """THe DDPG Critic would output a single value for the input <state, action>
# #
# #         :param state_size:
# #         :param action_size:
# #         :param seed:
# #         """
# #         super(DDPGCritic, self).__init__()
# #         self.seed = torch.manual_seed(seed)
# #         self.fc1 = nn.Linear(state_size, 128)
# #         self.bn1 = nn.BatchNorm1d(128)
# #         self.fc2 = nn.Linear(128 + action_size, 128)
# #         self.bn2 = nn.BatchNorm1d(128)
# #         self.fc3 = nn.Linear(128, 1)
# #         self.reset_parameters()
# #
# #     def reset_parameters(self):
# #         self.fc1.weight.data.uniform_(*hidden_init(self.fcs1))
# #         self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
# #         self.fc3.weight.data.uniform_(-3e-3, 3e-3)
# #
# #     def forward(self, state, action_dist):
# #         x = F.relu(self.bn1(self.fc1(state)))
# #         x = torch.cat((x, action_dist), dim=1)
# #         x = F.relu(self.bn2(self.fc2(x)))
# #         return self.fc3(x)
# #
# #
#
# #
# # def debug():
# #     state_torch = torch.from_numpy(np.random.random((1,33))).float().to(device)
# #
# #     actor_local_network = DDPGActor(state_size=33, action_size=4)
# #     critic_local_network = DDPGCritic(state_size=33, action_size=4)
# #     a = actor_local_network.forward(state_torch)
# #     b = critic_local_network.forward(state_torch, a)
# #
# #
# #     print(a.shape, a)
# #     print(b)
#
#
#
#


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=512, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        # self.drop1 = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        # self.drop2 = nn.Dropout(p=0.5)
        # self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.bn1(self.fc1(state)))
        # x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        # x = self.drop2(x)
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=512, fc2_units=512):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        # self.drop1 = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fc1(state))
        # xs = self.drop1(xs)
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        return self.fc3(x)


#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
#
#
# def hidden_init(layer):
#     fan_in = layer.weight.data.size()[0]
#     lim = 1. / np.sqrt(fan_in)
#     return (-lim, lim)
#
#
# class DDPGActor(nn.Module):
#     """Actor (Policy) Model."""
#
#     def __init__(self, state_size, action_size, hidden_units=(512, 256), seed=2, gate=F.relu, final_gate=F.tanh):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             hidden_units (array): Number of nodes for layers
#             seed (int): Random seed
#             gate (function): activation function
#             final_gate (function): final activation function
#         """
#         super(DDPGActor, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.gate = gate
#         self.final_gate = final_gate
#         self.normalizer = nn.BatchNorm1d(state_size)
#         dims = (state_size,) + hidden_units
#         self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
#         self.output = nn.Linear(dims[-1], action_size)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         for layer in self.layers:
#             layer.weight.data.uniform_(*hidden_init(layer))
#         self.output.weight.data.uniform_(-3e-3, 3e-3)
#
#     def forward(self, states):
#         """Build an actor (policy) network that maps states -> actions."""
#         x = self.normalizer(states)
#         for layer in self.layers:
#             x = self.gate(layer(x))
#         return self.final_gate(self.output(x))
#
#
# class DDPGCritic(nn.Module):
#     """Critic (Value) Model."""
#
#     def __init__(self, state_size, action_size, hidden_units=(512, 256), seed=2, gate=F.relu, dropout=0.2):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             hidden_units (array): Number of nodes for layers
#             seed (int): Random seed
#             gate (function): activation function
#         """
#         super(DDPGCritic, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.gate = gate
#         self.dropout = nn.Dropout(p=dropout)
#         self.normalizer = nn.BatchNorm1d(state_size)
#         dims = (state_size,) + hidden_units
#         self.layers = nn.ModuleList()
#         count = 0
#         for dim_in, dim_out in zip(dims[:-1], dims[1:]):
#             if count == 1:
#                 self.layers.append(nn.Linear(dim_in + action_size, dim_out))
#             else:
#                 self.layers.append(nn.Linear(dim_in, dim_out))
#             count += 1
#         self.output = nn.Linear(dims[-1], 1)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         for layer in self.layers:
#             layer.weight.data.uniform_(*hidden_init(layer))
#         self.output.weight.data.uniform_(-3e-3, 3e-3)
#
#     def forward(self, states, actions):
#         """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
#         xs = self.normalizer(states)
#         xs = self.gate(self.layers[0](xs))
#         x = torch.cat((xs, actions), dim=1)
#         for i in range(1, len(self.layers)):
#             x = self.gate(self.layers[i](x))
#         x = self.dropout(x)
#         return self.output(x)
#
#



class Optimize:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def adam(self, params):
        return optim.Adam(params, lr=self.learning_rate)

    def rmsprop(self, params):
        return optim.RMSprop(params, lr=self.learning_rate)  # default smoothing=0.99, eps = 1e-08, momentum=0








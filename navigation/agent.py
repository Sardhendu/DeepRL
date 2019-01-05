import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from navigation import model

import commons as cmn
from buffer import MemoryER, MemoryPER



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# class
class RLAgent:
    def __init__(self, args, env_type, seed):
        print('[INIT] Initializing RLAgent .... .... ....')
        self.args = args
        self.env_type = env_type
        self.MODEL_NAME = self.args.MODEL_NAME
        
        self.STATS_JSON_PATH = self.args.STATS_JSON_PATH
        self.ACTION_SIZE = self.args.ACTION_SIZE
        self.Q_LEARNING_TYPE = self.args.Q_LEARNING_TYPE
        self.SOFT_UPDATE = self.args.SOFT_UPDATE
        self.DECAY_TAU = self.args.DECAY_TAU
        self.TAU = self.args.TAU
        self.TAU_DECAY = self.args.TAU_DECAY
        self.TAU_MIN = self.args.TAU_MIN
        self.UPDATE_AFTER_STEP = self.args.UPDATE_AFTER_STEP
        self.BATCH_SIZE = self.args.BATCH_SIZE
        self.GAMMA = self.args.GAMMA
        self.HARD_UPDATE_FREQUENCY = self.args.HARD_UPDATE_FREQUENCY
        self.HARD_UPDATE = self.args.HARD_UPDATE
        
        self.seed = random.seed(seed)
        
        # 2. Initialize two networks, Local Network and the Target Network.
        if env_type == 'vector':
            self.local_network = model.QNetwork(args.STATE_SIZE, args.ACTION_SIZE, seed,
                                                network_name=args.NET_NAME).to(device)
            self.target_network = model.QNetwork(args.STATE_SIZE, args.ACTION_SIZE, seed,
                                                 network_name=args.NET_NAME).to(device)
        elif env_type == 'visual':
            self.local_network = model.VisualQNEtwork(args.STATE_SIZE, args.ACTION_SIZE, seed,
                                                      network_name=args.NET_NAME).to(device)
            self.target_network = model.VisualQNEtwork(args.STATE_SIZE, args.ACTION_SIZE, seed,
                                                       network_name=args.NET_NAME).to(device)
        else:
            raise ValueError('Env type not understood')
        
        # Create an optimizer for the local Network, since this is to be trained
        self.optimizer = model.Optimize(args.LEARNING_RATE).adam(params=self.local_network.parameters())
        
        # Count number of steps
        self.t_step = 0
        
        # Create a Statistics dictionary
        self.get_stats(reset=True)
        
        self.dump_stats_at_eposide = 100
    
    def dump_stats(self):
        if not os.path.exists(os.path.dirname(self.STATS_JSON_PATH)):
            os.makedirs(os.path.dirname(self.STATS_JSON_PATH))
        
        cmn.dump_json(self.STATS_JSON_PATH, self.stats)
    
    def get_stats(self, reset):
        if reset:
            self.stats = defaultdict(list)
        elif os.path.exists(self.STATS_JSON_PATH):
            self.stats = cmn.read_json(self.STATS_JSON_PATH)
        else:
            self.stats = defaultdict(list)
        
        return self.stats
    
    def action(self, state, eps=0.):
        """
        Select Random actions based on a policy (epsilon greedy policy)
        :param state:   array [37,]  -> for Banana environment
        :param eps:     float
        :return:
        """
        # print('[Action] Agent taking an action .......... ')
        # Convert into torch variable
        if self.env_type == "visual":
            state = torch.from_numpy(state).float().to(device)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)  # similar to expand_dims(axis=0) in numpy
        # print(state.shape)
        
        # Set the network to evaluation state
        state.requires_grad = False
        self.local_network.eval()
        with torch.no_grad():
            action_values = self.local_network(state)
        
        # Reset the network to training state
        self.local_network.train()
        
        # Find the best action based on epsilon-greedy policy
        if np.random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice(np.arange(self.ACTION_SIZE))
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter

        Idea
        ======
            Instead of performing a hard update on Fixed-Q-targets after say 10000 timesteps, perform soft-update
            (move slightly) every time-step
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def hard_update(self, local_model, target_model):
        """Hard update model parameters.
        θ_target = θ_local

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to

        Idea
        ======
            After t time-step copy the weights of local network to target network
        """
        # print('[Hard Update] Performuing hard update .... ')
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
    
    def learn(self, experiences, gamma, episode_num, **kwargs):
        """
        :param experiences:     <state, action, reward, state_next, done>
        :param gamma:
        :param episode_num:
        :return:

        Q-learning with Q-table:
            Step 1) expected_value = Q_table[state][action]
            Step 2) value = max(Q_table[state_next]
            Step 3) target_value = reward + gamma * value
            Step 4) new_value = current_value + alpha*(target_value - expected_value)

        Q-learning with Neural Net function approximator:
            Step 1) expected_value = forward_pass(state)[action]  (Using Local Network)
            Step 2) value = max(forward_pass(state_next))         (Using Target Network)
            Step 3) target_value = reward + gamma * value
            Step 4) optimize(loss(target_value, expected_value))
                      w = w - alpha(w
            Step 5) Update parameters for target model:
                      θ_target = τ * θ_local + (1 - τ) * θ_target
        """
        pass
    
    def step(self, state, action, reward, next_state, done, episode_num, running_time_step):
        """

        :param state:               time-step t (Current State)
        :param action:              time-step t (Current Action)
        :param reward:              time-step t+1 (reqard obtained)
        :param next_state:          time-step t+1 (next state obtained)
        :param done:                {0, 1} experiences that lead to episode termination
        :param episode_num:         (int) Current episode number
        :param running_time_step:   (int) Current timestep
        :return:

        The idea of experience replay was to avoid the learning (changing the network parameters) while playing.
        Because if we learn while we play, then the user action may sway to a particular direction
        and it would become very difficult to come back from that position. Therefore, it is good to
        explore the space (select random actions and store SARS) while playing. Then after few
        steps learn from the previous experiences and update the network parameter
        """
        # print ('[Step] Agent taking a step .......... ')
        # Exploration: Take multiple steps and fill the buffer with random actions based no epsilon greedy policy. (
        # obtained from the network). THis is like filling up the q-table.
        # If the buffer is filled then we learn using Neural Network.
        # print ('[Step] Current Step is: ', self.tstep)
        # Insert the tuple into the memory buffer


class DDQNAgent(RLAgent):
    def __init__(self, args, env_type='vector', seed=0):
        # 1. Create the Replay-Memory-Buffer
        self.memory = MemoryER(args.BUFFER_SIZE, args.BATCH_SIZE, seed)
        
        super().__init__(args, env_type, seed)
    
    def step(self, state, action, reward, next_state, done, episode_num=0, running_time_step=0):
        # print('Taking a step: ', state.shape, action, reward, next_state.shape, done, episode_num, running_time_step)
        # Insert the tuple into the memory buffer
        self.memory.add(state, action, reward, next_state, done)
        
        # print ('self.memory: ', self.memory)
        
        # Update the weights of local network and soft-update the weighs of the target_network
        self.t_step = (self.t_step + 1) % self.UPDATE_AFTER_STEP  # Run from {1->UPDATE_AFTER_STEP}
        # print('[Step] Current Step is: ', self.tstep)
        if self.t_step == 0:
            # print(len(self.memory), self.args.BUFFER_SIZE)
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                # print('asdafadfadfsdfs ', experiences)
                self.learn(experiences, self.GAMMA, episode_num)
        
        if self.HARD_UPDATE:
            if ((running_time_step + 1) % self.HARD_UPDATE_FREQUENCY) == 0:
                self.hard_update(self.local_network, self.target_network)
    
    def act(self, state, eps=0.):
        """
        Select Random actions based on a policy (epsilon greedy policy)
        :param state:   array [37,]  -> for Banana environment
        :param eps:     float
        :return:
        """
        # print('[Action] Agent taking an action state.shape ', state.shape)
        return self.action(state, eps)

    def learn(self, experiences, gamma, episode_num, **kwargs):
    
        # print('[Learn] Agent learning .......... ')
        states, actions, rewards, states_next, dones = experiences
        # print(states.shape, actions.shape, rewards.shape, states_next.shape, dones.shape)
    
        # Action and value taken based on the network parameters for input states
        expected_value = self.local_network.forward(states).gather(1, actions)  # [batch_size, n_actions] then Gather
        # values for the corresponding action
    
        if self.Q_LEARNING_TYPE == 'dqn':
            # Action and value taken based on the target network parameters for next_states
            t_values = self.target_network.forward(states_next).detach().max(dim=1)[0].unsqueeze(1)  # [batch_size, 1]
        elif self.Q_LEARNING_TYPE == 'ddqn':
            # Fetch Actions for state_next using Local Network
            l_action = self.local_network.forward(states_next).detach().argmax(dim=1).unsqueeze(1)  # [batch_size, 1]
            # Fetch Value for Actions selected using Target network
            t_values = self.target_network.forward(states_next).gather(1, l_action)  # [batch_size, 1]
        else:
            raise ValueError('Only dqn and ddqn handled')
    
        # target_value =
        target_value = rewards + (gamma * t_values * (1 - dones))
    
        # Get the td-Error and update the buffer with new priorities
        avg_td_error = np.average(
                np.abs(target_value.detach().cpu().numpy() - expected_value.detach().cpu().numpy()).reshape(-1))
        self.stats['td_error'].append(float(avg_td_error))
    
        # Update local_network parameters
        loss = F.mse_loss(expected_value, target_value)
        self.stats['loss'].append(float(loss))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        # Soft-update the target network.
        if self.SOFT_UPDATE:
            if self.DECAY_TAU:
                tau = cmn.exp_decay(self.TAU, self.TAU_DECAY, episode_num, self.TAU_MIN)
            else:
                tau = self.TAU
        
            self.soft_update(self.local_network, self.target_network, tau)
    
        # DUMP STATS:
        if (episode_num + 1) % self.dump_stats_at_eposide == 0:
            self.dump_stats()


class DDQNAgentPER(RLAgent):
    def __init__(self, args, env_type='vector', seed=0):
        self.memory = MemoryPER(args.BUFFER_SIZE, args.BATCH_SIZE, seed)
        # self.memory = MemoryER(args.BUFFER_SIZE, args.BATCH_SIZE, seed)
        
        super().__init__(args, env_type, seed)
    
    def step(self, state, action, reward, next_state, done, episode_num, running_time_step):
        # print ('Taking a step: ', state.shape, action, reward, next_state.shape, done, episode_num, running_time_step)
        self.memory.add(state, action, reward, next_state, done)
        
        # Update the weights of local network and soft-update the weighs of the target_network
        self.t_step = (self.t_step + 1) % self.UPDATE_AFTER_STEP  # Run from {1->UPDATE_AFTER_STEP}
        # print('[Step] Current Step is: ', self.t_step, running_time_step)
        if self.t_step == 0:
            # print(len(self.memory), self.args.BUFFER_SIZE)
            if len(self.memory) > self.BATCH_SIZE:
                # print('Going to learn: ...............', self.t_step)
                # b_idx, experiences, weights = self.memory.sample()
                b_idx, experiences, weights = self.memory.sample()
                # print(weights)
                self.learn(experiences, self.GAMMA, episode_num, tree_idx=b_idx, weights=weights)
        
        if self.HARD_UPDATE:
            if ((running_time_step + 1) % self.HARD_UPDATE_FREQUENCY) == 0:
                # print('Going to Hard Update: ...............', self.t_step, running_time_step)
                self.hard_update(self.local_network, self.target_network)
    
    def act(self, state, eps=0.):
        """
        Select Random actions based on a policy (epsilon greedy policy)
        :param state:   array [37,]  -> for Banana environment
        :param eps:     float
        :return:
        """
        # print('[Action] Agent taking an action state.shape ', state.shape)
        # Convert into torch variable
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)  # similar to expand_dims(axis=0) in numpy
        # print(state.shape)
        
        # Set the network to evaluation state
        self.local_network.eval()
        with torch.no_grad():
            action_values = self.local_network(state)
        
        # Reset the network to training state
        self.local_network.train()
        
        # Find the best action based on epsilon-greedy policy
        if np.random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice(np.arange(self.ACTION_SIZE))
    
    # Here we learn using the network and the Temporal difference Q-learning (SARSA)
    def learn(self, experiences, gamma, episode_num, **kwargs):
        
        tree_idx, weights = kwargs['tree_idx'], kwargs['weights']
        # print('experiencesexperiences: ', experiences)
        # print('[Learn] Agent learning .......... ')
        states, actions, rewards, states_next, dones = experiences
        # print(state.shape, action.shape, reward.shape, state_next.shape, done.shape)
        
        # Action and value taken based on the network parameters for input states
        # print('actionsactionsactions ', actions)
        expected_value = self.local_network.forward(states).gather(1, actions)  # [batch_size, n_actions] then Gather
        # values for the corresponding action
        
        if self.Q_LEARNING_TYPE == 'dqn':
            # Action and value taken based on the target network parameters for next_states
            t_values = self.target_network.forward(states_next).detach().max(dim=1)[0].unsqueeze(1)  # [batch_size, 1]
        elif self.Q_LEARNING_TYPE == 'ddqn':
            # Fetch Actions for state_next using Local Network
            l_action = self.local_network.forward(states_next).detach().argmax(dim=1).unsqueeze(1)  # [batch_size, 1]
            # Fetch Value for Actions selected using Target network
            t_values = self.target_network.forward(states_next).gather(1, l_action)  # [batch_size, 1]
        else:
            raise ValueError('Only dqn and ddqn handled')
        
        # target_value =
        target_value = rewards + (gamma * t_values * (1 - dones))  # reward + discount return following
        
        # Get the td-Error and update the buffer with new priorities
        td_error = np.abs(target_value.detach().numpy() - expected_value.detach().numpy()).reshape(-1)
        avg_td_error = np.average(td_error)
        self.memory.update(tree_idx, td_error)
        self.stats['td_error'].append(float(avg_td_error))
        
        # Update local_network parameters
        # print(weights)
        # TODO: If weights are 0, then it means that the PER_weights returns zero values. This happens when we
        # TODO: sample experiences without filling the buffer completely.
        loss = torch.sum((expected_value - target_value) ** 2)
        # loss = torch.sum(weights * (expected_value - target_value) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft-update the target network.
        if self.SOFT_UPDATE:
            # print('Going to Soft Update: ...............', self.t_step)
            if self.DECAY_TAU:
                tau = cmn.exp_decay(self.TAU, self.TAU_DECAY, episode_num, self.TAU_MIN)
            else:
                tau = self.TAU
            
            self.soft_update(self.local_network, self.target_network, tau)
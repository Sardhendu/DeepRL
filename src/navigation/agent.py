import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from src.buffer import MemoryER, MemoryPER
from src.navigation import model
from src import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# class
class RLAgent:
    def __init__(self, args, env_type, mode, agent_id=0):
        print('[INIT] Initializing RLAgent .... .... ....')
        self.env_type = env_type
        self.mode = mode
        self.agent_id = agent_id
        
        self.args = args
        if mode == 'train':
            # Environment parameters
            self.ACTION_SIZE = args.ACTION_SIZE
            self.STATE_SIZE = args.STATE_SIZE
            
            # Learning Hyperparametes
            self.BATCH_SIZE = args.BATCH_SIZE
            self.BUFFER_SIZE = args.BUFFER_SIZE
            self.DATA_TO_BUFFER_BEFORE_LEARNING = args.DATA_TO_BUFFER_BEFORE_LEARNING
            self.LEARNING_FREQUENCY = args.LEARNING_FREQUENCY
            self.Q_LEARNING_TYPE = args.Q_LEARNING_TYPE
            
            self.IS_SOFT_UPDATE = args.IS_SOFT_UPDATE
            self.SOFT_UPDATE_FREQUENCY = args.SOFT_UPDATE_FREQUENCY
            self.IS_HARD_UPDATE = args.IS_HARD_UPDATE
            self.HARD_UPDATE_FREQUENCY = args.HARD_UPDATE_FREQUENCY
            
            self.GAMMA = args.GAMMA
            self.TAU = args.TAU
            
            # Exploration Parameter
            self.EPSILON_GREEDY = args.EPSILON_GREEDY()
        
            # Network Initializers
            self.local_network = args.LOCAL_NETWORK()
            self.target_network = args.TARGET_NETWORK()
            self.optimizer = args.OPTIMIZER_FN(self.local_network.parameters())
            
            # Checkpoint path
            self.CHECKPOINT_DIR = args.CHECKPOINT_DIR
            self.SUMMARY_LOGGER = args.SUMMARY_LOGGER
            
        else:
            print('[Agent] Loading Agent weights')
            self.local_network = args.LOCAL_NETWORK()
            self.CHECKPOINT_DIR = args.CHECKPOINT_DIR
            self.CHECKPOINT_NUMBER = args.CHECKPOINT_NUMBER
            
    def load_weights(self):
        """
        :return:
        """
        if self.mode == 'train':
            checkpoint_path = os.path.join(
                    self.CHECKPOINT_DIR,
                    'local_%s_%s.pth' % (str(self.agent_id), str(self.CHECKPOINT_NUMBER))
            )
            print('Loading weights for agent_%s' % (str(self.agent_id)))
    
            self.local_network.load_state_dict(torch.load(checkpoint_path))
            
        elif self.mode == 'test':
            checkpoint_path = os.path.join(
                    self.CHECKPOINT_DIR,
                    'local_%s_%s.pth' % (str(self.agent_id), str(self.CHECKPOINT_NUMBER))
            )
            print(
                'Loading weights for actor_local for agent_%s from \n %s' % (str(self.agent_id), str(checkpoint_path)))
            self.local_network.load_state_dict(torch.load(checkpoint_path))

        else:
            raise ValueError('mode =  train or test permitted ....')

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
        pass
    
    def act(self, state, running_timestep):
        pass
    
    def log(self, tag, value_dict, step):
        self.SUMMARY_LOGGER.add_scalars(tag, value_dict, step)
        
    def checkpoint(self, episode_num):
    
        e_num = episode_num
        torch.save(
                self.local_network.state_dict(), os.path.join(
                        self.CHECKPOINT_DIR, 'local_%s_%s.pth' % (str(self.agent_id), str(e_num)))
        )
        torch.save(
                self.target_network.state_dict(), os.path.join(
                        self.CHECKPOINT_DIR, 'target_%s_%s.pth' % (str(self.agent_id), str(e_num)))
        )
       
class DDQNAgent(RLAgent):
    def __init__(self, args, env_type, mode):
        """
        :param args:            Agent parameters
        :param env_type:        str ("Visual", "Vector")
        :param mode:            str ("train", "test")
        """
        # 1. Create the Replay-Memory-Buffer
        self.memory = MemoryER(
                buffer_size=args.BUFFER_SIZE, batch_size=args.BATCH_SIZE, seed=args.SEED, action_dtype='long'
        )
        super().__init__(args, env_type, mode)

    def act(self, state, running_timestep):
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
    
        # Set the network to evaluation state
        state.requires_grad = False
        self.local_network.eval()
        with torch.no_grad():
            action_values = self.local_network(state)
    
        # Reset the network to training state
        self.local_network.train()
    
        # Find the best action based on epsilon-greedy policy
        epsilon = self.EPSILON_GREEDY.sample()
        if np.random.random() > epsilon:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = np.random.choice(np.arange(self.ACTION_SIZE))
        
        if self.mode == 'train':
            tag = 'agent%i/epsilon' % self.agent_id
            value_dict = {
                'epsilon': float(epsilon)
            }
            self.log(tag, value_dict, running_timestep)
            
        return action

    def step(self, states, actions, rewards, next_states, dones, running_timestep):
        # print('Taking a step: ', state.shape, action, reward, next_state.shape, done, episode_num, running_time_step)
        # Insert the tuple into the memory buffer
        self.memory.add(states, actions, rewards, next_states, dones)

        if (running_timestep % self.LEARNING_FREQUENCY) == 0:
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, running_timestep)

        if running_timestep > self.DATA_TO_BUFFER_BEFORE_LEARNING:
            if self.IS_HARD_UPDATE:
                if (running_timestep % self.HARD_UPDATE_FREQUENCY) == 0:
                    utils.hard_update(self.local_network, self.target_network)
    
            elif self.IS_SOFT_UPDATE:
                if (running_timestep % self.SOFT_UPDATE_FREQUENCY) == 0:
                    utils.soft_update(self.local_network, self.target_network, self.TAU)
            else:
                raise ValueError('Only One of HARD_UPDATE and SOFT_UPDATE is to be activated')

    def learn(self, experiences, running_timestep):
    
        # print('[Learn] Agent learning .......... ')
        states, actions, rewards, states_next, dones = experiences
        # print(states.shape, actions.shape, rewards.shape, states_next.shape, dones.shape)
    
        # Action and value taken based on the network parameters for input states
        expected_value = self.local_network.forward(states).gather(1, actions)  # [batch_size, n_actions] then Gather
        # values for the corresponding action
    
        if self.Q_LEARNING_TYPE == 'dqn':
            # Action and value taken based on the target network parameters for next_states
            t_values = self.target_network.forward(states_next).detach().max(dim=1)[0].unsqueeze(1)  # [batch_size, 1]
        elif self.Q_LEARNING_TYPE == 'dbl_dqn':
            # Fetch Actions for state_next using Local Network
            l_action = self.local_network.forward(states_next).detach().argmax(dim=1).unsqueeze(1)  # [batch_size, 1]
            # Fetch Value for Actions selected using Target network
            t_values = self.target_network.forward(states_next).gather(1, l_action)  # [batch_size, 1]
        else:
            raise ValueError('Only dqn and dbl_dqn handled')
    
        target_value = rewards + (self.GAMMA * t_values * (1 - dones))
    
        # Update local_network parameters
        loss = F.mse_loss(expected_value, target_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        # ------------------------- LOGGING --------------------------#
        if self.log:
            tag = 'agent%i/loss' % self.agent_id
            value_dict = {
                'agent_loss': abs(float(loss))
            }
            self.log(tag, value_dict, running_timestep)



# class DDQNAgentPER(RLAgent):
#     def __init__(self, args, env_type='vector', seed=0):
#         self.memory = MemoryPER(args.BUFFER_SIZE, args.BATCH_SIZE, seed)
#         # self.memory = MemoryER(args.BUFFER_SIZE, args.BATCH_SIZE, seed)
#
#         super().__init__(args, env_type, seed)
#
#     def step(self, state, action, reward, next_state, done, episode_num, running_time_step):
#         # print ('Taking a step: ', state.shape, action, reward, next_state.shape, done, episode_num, running_time_step)
#         self.memory.add(state, action, reward, next_state, done)
#
#         # Update the weights of local network and soft-update the weighs of the target_network
#         self.t_step = (self.t_step + 1) % self.UPDATE_AFTER_STEP  # Run from {1->UPDATE_AFTER_STEP}
#         # print('[Step] Current Step is: ', self.t_step, running_time_step)
#         if self.t_step == 0:
#             # print(len(self.memory), self.args.BUFFER_SIZE)
#             if len(self.memory) > self.BATCH_SIZE:
#                 # print('Going to learn: ...............', self.t_step)
#                 # b_idx, experiences, weights = self.memory.sample()
#                 b_idx, experiences, weights = self.memory.sample()
#                 # print(weights)
#                 self.learn(experiences, self.GAMMA, episode_num, tree_idx=b_idx, weights=weights)
#
#         if self.HARD_UPDATE:
#             if ((running_time_step + 1) % self.HARD_UPDATE_FREQUENCY) == 0:
#                 # print('Going to Hard Update: ...............', self.t_step, running_time_step)
#                 self.hard_update(self.local_network, self.target_network)
#
#     def act(self, state, eps=0.):
#         """
#         Select Random actions based on a policy (epsilon greedy policy)
#         :param state:   array [37,]  -> for Banana environment
#         :param eps:     float
#         :return:
#         """
#         # print('[Action] Agent taking an action state.shape ', state.shape)
#         # Convert into torch variable
#         state = torch.from_numpy(state).float().unsqueeze(0).to(device)  # similar to expand_dims(axis=0) in numpy
#         # print(state.shape)
#
#         # Set the network to evaluation state
#         self.local_network.eval()
#         with torch.no_grad():
#             action_values = self.local_network(state)
#
#         # Reset the network to training state
#         self.local_network.train()
#
#         # Find the best action based on epsilon-greedy policy
#         if np.random.random() > eps:
#             return np.argmax(action_values.cpu().data.numpy())
#         else:
#             return np.random.choice(np.arange(self.ACTION_SIZE))
#
#     # Here we learn using the network and the Temporal difference Q-learning (SARSA)
#     def learn(self, experiences, gamma, episode_num, **kwargs):
#
#         tree_idx, weights = kwargs['tree_idx'], kwargs['weights']
#         # print('[Learn] Agent learning .......... ')
#         states, actions, rewards, states_next, dones = experiences
#         # print(state.shape, action.shape, reward.shape, state_next.shape, done.shape)
#
#         # Action and value taken based on the network parameters for input states
#         # print('actionsactionsactions ', actions)
#         expected_value = self.local_network.forward(states).gather(1, actions)  # [batch_size, n_actions] then Gather
#         # values for the corresponding action
#
#         if self.Q_LEARNING_TYPE == 'dqn':
#             # Action and value taken based on the target network parameters for next_states
#             t_values = self.target_network.forward(states_next).detach().max(dim=1)[0].unsqueeze(1)  # [batch_size, 1]
#         elif self.Q_LEARNING_TYPE == 'ddqn':
#             # Fetch Actions for state_next using Local Network
#             l_action = self.local_network.forward(states_next).detach().argmax(dim=1).unsqueeze(1)  # [batch_size, 1]
#             # Fetch Value for Actions selected using Target network
#             t_values = self.target_network.forward(states_next).gather(1, l_action)  # [batch_size, 1]
#         else:
#             raise ValueError('Only dqn and ddqn handled')
#
#         # target_value =
#         target_value = rewards + (gamma * t_values * (1 - dones))  # reward + discount return following
#
#         # Get the td-Error and update the buffer with new priorities
#         td_error = np.abs(target_value.detach().numpy() - expected_value.detach().numpy()).reshape(-1)
#         avg_td_error = np.average(td_error)
#         self.memory.update(tree_idx, td_error)
#         self.stats['td_error'].append(float(avg_td_error))
#
#         # Update local_network parameters
#         # print(weights)
#         # TODO: If weights are 0, then it means that the PER_weights returns zero values. This happens when we
#         # TODO: sample experiences without filling the buffer completely.
#         loss = torch.sum((expected_value - target_value) ** 2)
#         # loss = torch.sum(weights * (expected_value - target_value) ** 2)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#
#         # Soft-update the target network.
#         if self.SOFT_UPDATE:
#             # print('Going to Soft Update: ...............', self.t_step)
#             tau = self.TAU
#             utils.soft_update(self.local_network, self.target_network, tau)
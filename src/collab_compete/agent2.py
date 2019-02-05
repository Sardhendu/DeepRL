from src.buffer import MemoryER
from collections import defaultdict
import os
import numpy as np
import torch
import torch.nn.functional as F

from src import utils
from src import common as cmn
from src import buffer
from src.collab_compete.model import Actor_Critic_Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MDDPG():
    def __init__(self, args, mode):
        self.MODE = mode
        self.STATE_SIZE = args.STATE_SIZE
        self.ACTION_SIZE = args.ACTION_SIZE
        self.GAMMA = args.GAMMA
        self.NUM_AGENTS = args.NUM_AGENTS
        
        self.SOFT_UPDATE = args.SOFT_UPDATE
        self.HARD_UPDATE = args.HARD_UPDATE
        self.HARD_UPDATE_FREQUENCY = args.HARD_UPDATE_FREQUENCY
        self.SOFT_UPDATE_FREQUENCY = args.SOFT_UPDATE_FREQUENCY
        
        self.DECAY_TAU = args.DECAY_TAU
        self.TAU = args.TAU
        self.TAU_MIN = args.TAU_MIN
        self.TAU_DECAY_RATE = args.TAU_DECAY_RATE
        self.DATA_TO_BUFFER_BEFORE_LEARNING = args.DATA_TO_BUFFER_BEFORE_LEARNING
        
        if mode == 'train':
            self.LEARNING_FREQUENCY = args.LEARNING_FREQUENCY
            self.BATCH_SIZE = args.BATCH_SIZE
            # self.memory = args.MEMORY_FN()  # [args.MEMORY_FN() for _ in range(0, self.NUM_AGENTS)]
            
            self.hard_update = utils.hard_update
            
            self.CHECKPOINT_DIR = args.CHECKPOINT_DIR
        
            # create two agents, each with their own actor and critic
            
            # self.AGENTS = [DDPG(args, agent_id, mode='train', log=True) for agent_id in range(self.NUM_AGENTS)]
            models = [Actor_Critic_Models(n_agents=self.NUM_AGENTS) for _ in range(self.NUM_AGENTS)]
            # self.AGENTS = [DDPG(args, models[agent_id], agent_id, mode='train', log=True) for agent_id in range(
            #         self.NUM_AGENTS)]
    
            # create shared replay buffer
            self.memory = buffer.MemoryER(buffer_size=10000, batch_size=256, seed=0, action_dtype='float')
            # self.MEMORY = args.MEMORY_FN()
        
        else:
            pass

    def act(self, state, action_value_range, running_time_step):
        all_actions = []
        for agent, state in zip(self.AGENTS, state):
            action = agent.act(state, action_value_range, running_time_step)
            all_actions.append(action)
        return np.concatenate(all_actions)

    def step(self, states, actions, rewards, next_states, dones, episode_num, running_time_step, log=True):

        learning_flag = 0
        
        
        # print(rewards)
        # print(dones)

        states = np.expand_dims(np.stack(states, axis=0), axis=0)
        actions = np.expand_dims(np.stack(actions, axis=0), axis=0)
        rewards = np.expand_dims(np.stack(np.array(rewards, dtype=np.float32).reshape(-1,1), axis=0), axis=0)
        next_states = np.expand_dims(np.stack(next_states, axis=0), axis=0)
        dones = np.expand_dims(np.stack(np.array(dones, dtype=np.bool).reshape(-1,1), axis=0), axis=0)

        # print(states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)
        # print('rewards: ', rewards)
        self.memory.add(states, actions, rewards, next_states, dones)

        if (running_time_step % self.LEARNING_FREQUENCY) == 0:
            if len(self.memory) > self.DATA_TO_BUFFER_BEFORE_LEARNING:
                # print('Agent %s : Memory Size: ', len(self.memory[num]))
                experiences = self.memory.sample()
                
                all_states, all_actions, all_rewards, all_next_states, all_dones = experiences
                # print(all_states.shape, all_actions.shape, all_rewards.shape, all_next_states.shape, all_dones.shape)
                
                all_states = all_states.view(all_states.shape[0], -1)
                all_actions = all_actions.view(all_actions.shape[0], -1)
                all_rewards = all_rewards.view(all_rewards.shape[0], -1)
                all_next_states = all_next_states.view(all_next_states.shape[0], -1)
                all_dones = all_dones.view(all_dones.shape[0], -1)

                # print(all_states.shape, all_actions.shape, all_rewards.shape, all_next_states.shape, all_dones.shape)
                #
                # all_next_states = next_state.view(next_state.shape[0], -1)
                #
                # state = state.permute(1, 0, 2)  # convert to [num_agents, batch_size, num_states]
                # action = action.permute(1, 0, 2)  # convert to [num_agents, batch_size, num_states]
                # all_rewards = reward.permute(1, 0)  # convert to [num_agents, batch_size]
                # next_states = next_state.permute(1, 0, 2)  # convert to [num_agents, batch_size, num_states]
                # all_dones = done.permute(1, 0)  # convert to [num_agents,batch_size]
                self.learn(all_states, all_actions, all_rewards, all_next_states, all_dones, running_time_step)
                learning_flag = 1

        # if learning_flag == 1:
        #     if self.HARD_UPDATE:
        #         if (running_time_step % self.HARD_UPDATE_FREQUENCY) == 0:
        #             for num in range(0, self.NUM_AGENTS):
        #                 self.hard_update(self.actor_local[num], self.actor_target[num])
        #                 self.hard_update(self.critic_local[num], self.critic_target[num])
        #
        #     elif self.SOFT_UPDATE:
        #         if (running_time_step % self.SOFT_UPDATE_FREQUENCY) == 0:
        #             # print('Performing soft-update at: ', running_time_step)
        #
        #             for num in range(0, self.NUM_AGENTS):
        #                 self.soft_update(self.critic_local[num], self.critic_target[num], tau)
        #                 self.soft_update(self.actor_local[num], self.actor_target[num], tau)
        #     else:
        #         raise ValueError('Only One of HARD_UPDATE and SOFT_UPDATE is to be activated')
                
                
    def learn(self, all_states, all_actions, all_rewards, all_next_states, all_dones, running_time_step):
        all_next_actions = []
        all_action_pred = []

        state = all_states.reshape(-1, self.NUM_AGENTS, self.STATE_SIZE)
        next_state = all_next_states.reshape(-1, self.NUM_AGENTS, self.STATE_SIZE)
        for ag_num, agent in enumerate(self.AGENTS):
            next_actions = agent.actor_target.forward(state[:, ag_num, :])
            action_pred = agent.actor_local.forward(next_state[:, ag_num, :])
            # You might be wondering why do we get actions for the state, if we already have action for those
            # state using the replay buffer. Well the network weights might have changed after we stashed
            # the experience into the replay buffer. So here we get the fresh actions for the same states.
            all_next_actions.append(next_actions)
            all_action_pred.append(action_pred)
        
        all_next_actions = torch.cat((all_next_actions[0], all_next_actions[1]), 1)
        
        for ag_num, agent in enumerate(self.AGENTS):
            agent.learn(all_states, all_actions, all_next_states, all_next_actions, all_action_pred,
            all_rewards, all_dones, self.GAMMA, ag_num, running_time_step)
            
            
            

import torch.optim as optim


import random

class DDPG:
    def __init__(self, args, model, agent_id, mode, log):
        random.seed(0)
        self.MODE = mode
        self.AGENT_ID = agent_id
        self.LOG = log
        self.TAU = args.TAU
        
        self.soft_update = utils.soft_update
        self.noise = args.NOISE_FN()
        self.noise_amplitude_decay = args.NOISE_AMPLITUDE_FN()

        self.actor_local = model.actor_local
        self.actor_target = model.actor_target
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=1e-4)

        # Critic Network
        self.critic_local = model.critic_local
        self.critic_target = model.critic_target
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=1e-3, weight_decay=0.0)

        self.summary_logger = args.SUMMARY_LOGGER

        self.hard_copy_weights(self.actor_target, self.actor_local)
        self.hard_copy_weights(self.critic_target, self.critic_local)

    def hard_copy_weights(self, target, source):
        """ copy weights from source to target network (part of initialization)"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
        
    def act(self, state, action_value_range, running_time_step):
        # print(state)
        self.noise_amplitude = self.noise_amplitude_decay.sample()
        state = torch.from_numpy(np.expand_dims(state, axis=0)).float().to(device)
    
        state.require_grad = False
        self.actor_local.eval()
        with torch.no_grad():
            # action = self.actor_local[num].forward(state)
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        # print(action)
        if self.MODE == 'train':
            self.noise_val = self.noise.sample() * self.noise_amplitude
            action += self.noise_val
    
        # Clip the actions to the the min and max limit of action probs
        action = np.clip(action, action_value_range[0], action_value_range[1])

        if self.LOG:
            tag = 'agent%i/noise_amplitude' % self.AGENT_ID
            value_dict = {
                'noise_amplitude': self.noise_amplitude
            }
            self.log(tag, value_dict, running_time_step)
        return action

    # def get_central_actions(self):


    def learn(
            self, all_states, all_actions, all_next_states, all_next_actions, all_action_pred,
            all_rewards, all_dones, gamma, agent_num, running_time_step, log=True
    ):
        # ------------------------- CRITIC --------------------------- #
        # 1. First we fetch the expented value using critic local and the state-action pair
        expected_returns = self.critic_local.forward(all_states, all_actions)
    
        # 2. Target value is computed using critic/actor target and next_state-next_action pair.
        # TODO: next_states should also be serverd from both the networks
        # with torch.no_grad():
        target_value = self.critic_target.forward(all_next_states, all_next_actions)
        

        target_returns = all_rewards[:, agent_num].reshape(-1,1) + \
                         (gamma * target_value * (1 - all_dones[:, agent_num].reshape(-1, 1)))

        # 3. Now we calculate critic loss which is simple the mse of expected and target
        self.critic_optimizer.zero_grad()
        critic_loss = F.mse_loss(expected_returns, target_returns.detach())
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_local[agent_num].parameters(), 1)
        self.critic_optimizer.step()
    
        # ------------------------- ACTOR --------------------------- #
        # 1. We re-fetch the actions from the actor network. The reason being that the current experience
        # might be old and the actor weights might have been updated many times since the experience was collected
        # TODO: NOTE: all_action_pred ([action_agent_0, action_agent_1]) is a result of running the "actor_local"
        # network
        # TODO: for both the agents, however at this point only one of the agent is active for backpropagation and the
        # TODO: buffer for the other agent no longer exists. Therefore we detach the other agent's contribution from the
        # TODO: current graph and only use its actions as another set of features to guide training.
        actions_pred = [act_p if i == self.AGENT_ID else act_p.detach() for i, act_p in enumerate(all_action_pred)]
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        actor_loss = -self.critic_local.forward(all_states, actions_pred).mean()
    
        # 2. Now we train the actor network, optimize the actor loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_local[agent_num].parameters(), 1)
        self.actor_optimizer.step()

        self.soft_update(self.critic_local, self.critic_target, self.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.TAU)
    
        # ------------------------- LOGGING --------------------------#
        if self.LOG:
            tag = 'agent%i/actor_loss' % self.AGENT_ID
            value_dict = {
                'actor_loss': abs(float(actor_loss))
            }
            self.log(tag, value_dict, running_time_step)
        
            tag = 'agent%i/critic_loss' % self.AGENT_ID
            value_dict = {
                'critic loss': abs(float(critic_loss)),
            }
            self.log(tag, value_dict, running_time_step)

    def reset(self):
        self.noise.reset()
        
    def log(self, tag, value_dict, step):
        self.summary_logger.add_scalars(tag, value_dict, step)
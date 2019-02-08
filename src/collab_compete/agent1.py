from src.buffer import MemoryER
from collections import defaultdict
import os
import numpy as np
import torch
import torch.nn.functional as F

from src import utils
from src import common as cmn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AgentParams():
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
            self.memory = args.MEMORY_FN()#[args.MEMORY_FN() for _ in range(0, self.NUM_AGENTS)]
            
            self.actor_local = [args.ACTOR_NETWORK_FN() for _ in range(0, self.NUM_AGENTS)]
            self.actor_target = [args.ACTOR_NETWORK_FN() for _ in range(0, self.NUM_AGENTS)]
            
            self.critic_local = [args.CRITIC_NETWORK_FN() for _ in range(0, self.NUM_AGENTS)]
            self.critic_target = [args.CRITIC_NETWORK_FN() for _ in range(0, self.NUM_AGENTS)]
            
            self.actor_local_optimizer = [
                args.ACTOR_OPTIMIZER_FN(self.actor_local[num].parameters()) for num in range(0, self.NUM_AGENTS)
            ]
            self.critic_local_optimizer = [
                args.CRITIC_OPTIMIZER_FN(self.critic_local[num].parameters()) for num in range(0, self.NUM_AGENTS)
            ]

            self.soft_update = utils.soft_update
            self.hard_update = utils.hard_update

            self.noise_amplitude_decay = args.NOISE_AMPLITUDE_FN()
            self.noise = [args.NOISE_FN() for _ in range(0, self.NUM_AGENTS)]
            
            self.SUMMARY_LOGGER = args.SUMMARY_LOGGER
            self.CHECKPOINT_DIR = args.CHECKPOINT_DIR

            # Set weights for local and target actor, respectively, critic the same
            for ag_num in range(self.NUM_AGENTS):
                self.hard_update(self.actor_target[ag_num], self.actor_local[ag_num])
                self.hard_update(self.critic_target[ag_num], self.critic_local[ag_num])
        else:
            pass


class CentralizedAgent(AgentParams):
    def __init__(self, args, mode='train'):
        super().__init__(args, mode)
    
    def act(self, states, action_value_range):
        actions = []
        self.noise_amplitude = self.noise_amplitude_decay.sample()
        for num, state in enumerate(states):
            # print(state)
            state = torch.from_numpy(np.expand_dims(state, axis=0)).float().to(device)
            
            state.require_grad = False
            self.actor_local[num].eval()
            with torch.no_grad():
                # action = self.actor_local[num].forward(state)
                action = self.actor_local[num](state).cpu().data.numpy()
            self.actor_local[num].train()
            # print(action)
            if self.MODE == 'train':
                action += self.noise[num].sample() * self.noise_amplitude
            
            # Clip the actions to the the min and max limit of action probs
            action = np.clip(action, action_value_range[0], action_value_range[1])
            # print('Actions Value: ', actions)
            
            actions.append(action)
        
        return np.concatenate(actions)
    
    # def get_central_actions(self):
    
    
    def learn(
            self, all_states, all_actions, all_next_states, all_next_actions, all_action_pred,
            reward, done, gamma, agent_num, running_time_step, log=True
    ):
        # ------------------------- CRITIC --------------------------- #
        # 1. First we fetch the expented value using critic local and the state-action pair
        expected_returns = self.critic_local[agent_num].forward(all_states, all_actions)
        
        # 2. Target value is computed using critic/actor target and next_state-next_action pair.
        # TODO: next_states should also be serverd from both the networks
        # with torch.no_grad():
        target_value = self.critic_target[agent_num].forward(all_next_states, all_next_actions)
        target_returns = reward + gamma * target_value * (1 - done)
        self.critic_local_optimizer[agent_num].zero_grad()
        # 3. Now we calculate critic loss which is simple the mse of expected and target
        critic_loss = F.mse_loss(expected_returns, target_returns.detach())
        
        # 4. Now we train the critic network, optimize the critic loss
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_local[agent_num].parameters(), 1)
        self.critic_local_optimizer[agent_num].step()
        
        # ------------------------- ACTOR --------------------------- #
        # 1. We re-fetch the actions from the actor network. The reason being that the current experience
        # might be old and the actor weights might have been updated many times since the experience was collected
        # TODO: NOTE: all_action_pred ([action_agent_0, action_agent_1]) is a result of running the "actor_local" network
        # TODO: for both the agents, however at this point only one of the agent is active for backpropagation and the
        # TODO: buffer for the other agent no longer exists. Therefore we detach the other agent's contributionfrom the
        # TODO: current graph and only use its actions as another set of features to guide training.
        actions_pred = [act_p if i == agent_num else act_p.detach() for i, act_p in enumerate(all_action_pred)]
        actions_pred = torch.cat(actions_pred, dim=1)
        actor_loss = -self.critic_local[agent_num].forward(all_states, actions_pred).mean()
        
        # 2. Now we train the actor network, optimize the actor loss
        self.actor_local_optimizer[agent_num].zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_local[agent_num].parameters(), 1)
        self.actor_local_optimizer[agent_num].step()

        # ------------------------- LOGGING --------------------------#
        if log:
            tag = 'agent%i/actor_loss' % agent_num
            value_dict = {
                'actor_loss': abs(float(actor_loss))
            }
            self.log(tag, value_dict, running_time_step)
            
            tag = 'agent%i/critic_loss' % agent_num
            value_dict = {
                'critic loss': abs(float(critic_loss)),
            }
            self.log(tag, value_dict, running_time_step)
            
            tag = 'common/noise_amplitude'
            value_dict = {
                'noise_amplitude': self.noise_amplitude
            }
            self.log(tag, value_dict, running_time_step)
    
    def step(self, states, actions, rewards, next_states, dones, episode_num, running_time_step, log=True):
        
        learning_flag = 0
        
        states = np.expand_dims(np.stack(states, axis=0), axis=0)
        actions = np.expand_dims(np.stack(actions, axis=0), axis=0)
        rewards = np.expand_dims(np.stack(rewards, axis=0), axis=0)
        next_states = np.expand_dims(np.stack(next_states, axis=0), axis=0)
        dones = np.expand_dims(np.stack(dones, axis=0), axis=0)
        
        self.memory.add(states, actions, rewards, next_states, dones)
        
        # print('adding: ', state.shape, action.shape, reward, next_state.shape, done)
        # self.memory.add(state, action, reward, next_state, done)
        # self.memory[num].add(states[num], actions[num], rewards[num], next_states[num], dones[num])
        
        if (running_time_step % self.LEARNING_FREQUENCY) == 0:
            if len(self.memory) > self.DATA_TO_BUFFER_BEFORE_LEARNING:
                # print('Agent %s : Memory Size: ', len(self.memory[num]))
                
                experiences = self.memory.sample()
                state, action, reward, next_state, done = experiences
                all_states = state.view(state.shape[0], -1)
                all_next_states = next_state.view(next_state.shape[0], -1)
                
                state = state.permute(1, 0, 2)  # convert to [num_agents, batch_size, num_states]
                action = action.permute(1, 0, 2)  # convert to [num_agents, batch_size, num_states]
                reward = reward.permute(1, 0)  # convert to [num_agents, batch_size]
                next_state = next_state.permute(1, 0, 2)  # convert to [num_agents, batch_size, num_states]
                done = done.permute(1, 0)  # convert to [num_agents,batch_size]

                
                
                all_next_actions = []
                all_action_pred = []
                for num in range(0, self.NUM_AGENTS):
                    next_actions = self.actor_target[num].forward(next_state[num])
                    action_pred = self.actor_local[num].forward(state[num])
                    # You might be wondering why do we get actions for the state, if we already have action for those
                    # state using the replay buffer. Well the network weights might have changed after we stashed
                    # the experience into the replay buffer. So here we get the fresh actions for the same states.
                    all_next_actions.append(next_actions)
                    all_action_pred.append(action_pred)
                
                all_actions = torch.cat((action[0], action[1]), 1)
                all_next_actions = torch.cat((all_next_actions[0], all_next_actions[1]), 1)
                
                # print(state.shape)
                for agent_num in range(0, self.NUM_AGENTS):
        
                    self.learn(
                        all_states, all_actions, all_next_states, all_next_actions, all_action_pred,
                        reward[agent_num].unsqueeze(1), done[agent_num].unsqueeze(1), self.GAMMA, agent_num, running_time_step
                    )
                    
                    learning_flag = 1
        
        if learning_flag == 1:
            if self.HARD_UPDATE:
                if (running_time_step % self.HARD_UPDATE_FREQUENCY) == 0:
                    for num in range(0, self.NUM_AGENTS):
                        self.hard_update(self.actor_local[num], self.actor_target[num])
                        self.hard_update(self.critic_local[num], self.critic_target[num])
            
            elif self.SOFT_UPDATE:
                if (running_time_step % self.SOFT_UPDATE_FREQUENCY) == 0:
                    # print('Performing soft-update at: ', running_time_step)
                    
                    if self.DECAY_TAU:
                        tau = utils.exp_decay(self.TAU, self.TAU_DECAY_RATE, episode_num, self.TAU_MIN)
                    
                    else:
                        tau = self.TAU
                    
                    for num in range(0, self.NUM_AGENTS):
                        self.soft_update(self.critic_local[num], self.critic_target[num], tau)
                        self.soft_update(self.actor_local[num], self.actor_target[num], tau)
            else:
                raise ValueError('Only One of HARD_UPDATE and SOFT_UPDATE is to be activated')
    
    def reset(self):
        _ = [self.exploration_policy[nag].reset() for nag in range(0, self.NUM_AGENTS)]
    
    def log(self, tag, value_dict, step):
        self.SUMMARY_LOGGER.add_scalars(tag, value_dict, step)
    
    def checkpoint(self, episode_num):
        e_num = episode_num
        for num_ in range(self.NUM_AGENTS):
            torch.save(
                    self.actor_local[num_].state_dict(), os.path.join(
                            self.CHECKPOINT_DIR, 'actor_local_%s_%s.pth' % (str(num_), str(e_num)))
            )
            torch.save(
                    self.actor_target[num_].state_dict(),
                    os.path.join(self.CHECKPOINT_DIR, 'actor_target_%s_%s.pth' % (str(num_), str(e_num)))
            )
            torch.save(
                    self.critic_local[num_].state_dict(),
                    os.path.join(self.CHECKPOINT_DIR, 'critic_local_%s_%s.pth' % (str(num_), str(e_num)))
            )
            torch.save(
                    self.critic_target[num_].state_dict(),
                    os.path.join(self.CHECKPOINT_DIR, 'critic_target_%s_%s.pth' % (str(num_), str(e_num)))
            )







class Agent(AgentParams):
    def __init__(self, args, mode='train'):
        super().__init__(args, mode)
    
    def act(self, states, action_value_range):
        actions = []
        
        self.noise_amplitude = self.noise_amplitude_decay()
        for num, state in enumerate(states):
            state = torch.from_numpy(np.expand_dims(state, axis=0)).float().to(device)
            state.require_grad = False
            self.actor_local[num].eval()
            with torch.no_grad():
                action = self.actor_local[num].forward(state)
            self.actor_local[num].train()
            
            if self.MODE == 'train':
                action += self.exploration_policy[num].sample() * self.noise_amplitude
            
            # Clip the actions to the the min and max limit of action probs
            action = np.clip(action, action_value_range[0], action_value_range[1])
            # print('Actions Value: ', actions)
            
            actions.append(action)
        
        return np.concatenate(actions)
    
    def learn(self, experiences, gamma, agent_num, running_time_step, log=True):
        state, action, reward, next_state, done = experiences

        # ------------------------- CRITIC ---------------------------#
        # 1. First we fetch the expented value using critic local and the state-action pair
        expected_returns = self.critic_local[agent_num].forward(state, action)
        
        # 2. Target value is computed using critic/actor target and next_state-next_action pair
        next_actions = self.actor_target[agent_num].forward(next_state)
        target_value = self.critic_target[agent_num].forward(next_state, next_actions)
        target_returns = reward + gamma * target_value * (1 - done)
        
        # 3. Now we calculate critic loss which is simple the mse of expected and target
        critic_loss = F.mse_loss(expected_returns, target_returns.detach())
        
        # 4. Now we train the critic network, optimize the critic loss
        self.critic_local_optimizer[agent_num].zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local[agent_num].parameters(), 1)
        self.critic_local_optimizer[agent_num].step()
        
        # ------------------------- ACTOR --------------------------- #
        # 1. We re-get the actions from the actor network. The reason being that the current experience
        # might be old and the actor weights might have been updated many times since the experience was collected
        action_pred = self.actor_local[agent_num].forward(state)
        actor_loss = -self.critic_local[agent_num].forward(state, action_pred).mean()
        
        # 2. Now we train the actor network, optimize the actor loss
        self.actor_local_optimizer[agent_num].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local[agent_num].parameters(), 1)
        self.actor_local_optimizer[agent_num].step()

        # ------------------------- LOGGING -------------------------- #
        if log:
            tag = 'agent%i/actor_loss' % agent_num
            value_dict = {
                'actor_loss': abs(float(actor_loss))
            }
            self.log(tag, value_dict, running_time_step)
            
            tag = 'agent%i/critic_loss' % agent_num
            value_dict = {
                'critic loss': abs(float(critic_loss)),
            }
            self.log(tag, value_dict, running_time_step)
            
            tag = 'common/noise_amplitude'
            value_dict = {
                'noise_amplitude': self.noise_amplitude
            }
            self.log(tag, value_dict, running_time_step)
    
    def step(self, states, actions, rewards, next_states, dones, episode_num, running_time_step, log=True):
        
        learning_flag = 0
        
        states = np.expand_dims(np.stack(states, axis=0), axis=0)
        actions = np.expand_dims(np.stack(actions, axis=0), axis=0)
        rewards = np.expand_dims(np.stack(rewards, axis=0), axis=0)
        next_states = np.expand_dims(np.stack(next_states, axis=0), axis=0)
        dones = np.expand_dims(np.stack(dones, axis=0), axis=0)
        
        self.memory.add(states, actions, rewards, next_states, dones)
        
        # print('adding: ', state.shape, action.shape, reward, next_state.shape, done)
        # self.memory.add(state, action, reward, next_state, done)
        # self.memory[num].add(states[num], actions[num], rewards[num], next_states[num], dones[num])
        
        if (running_time_step % self.LEARNING_FREQUENCY) == 0:
            if len(self.memory) > self.DATA_TO_BUFFER_BEFORE_LEARNING:
                # print('Agent %s : Memory Size: ', len(self.memory[num]))
                
                experiences = self.memory.sample()
                state, action, reward, next_state, done = experiences
                state = state.permute(1, 0, 2)  # convert to [batch_size, num_agents, num_states]
                action = action.permute(1, 0, 2)  # convert to [batch_size, num_agents, num_states]
                reward = reward.permute(1, 0)  # convert to [batch_size, num_agents]
                next_state = next_state.permute(1, 0, 2)  # convert to [batch_size, num_agents, num_states]
                done = done.permute(1, 0)  # convert to [batch_size, num_agents]
                
                for num in range(0, self.NUM_AGENTS):
                    experiences = [state[num], action[num], reward[num], next_state[num], done[num]]
                    self.learn(experiences, self.GAMMA, num, running_time_step)
                    
                    learning_flag = 1
        
        if learning_flag == 1:
            if self.HARD_UPDATE:
                if (running_time_step % self.HARD_UPDATE_FREQUENCY) == 0:
                    for num in range(0, self.NUM_AGENTS):
                        self.hard_update(self.actor_local[num], self.actor_target[num])
                        self.hard_update(self.critic_local[num], self.critic_target[num])
            
            elif self.SOFT_UPDATE:
                if (running_time_step % self.SOFT_UPDATE_FREQUENCY) == 0:
                    # print('Performing soft-update at: ', running_time_step)
                    
                    if self.DECAY_TAU:
                        tau = utils.exp_decay(self.TAU, self.TAU_DECAY_RATE, episode_num, self.TAU_MIN)
                    
                    else:
                        tau = self.TAU
                    
                    for num in range(0, self.NUM_AGENTS):
                        self.soft_update(self.critic_local[num], self.critic_target[num], tau)
                        self.soft_update(self.actor_local[num], self.actor_target[num], tau)
            else:
                raise ValueError('Only One of HARD_UPDATE and SOFT_UPDATE is to be activated')
    
    def reset(self):
        _ = [self.exploration_policy[nag].reset() for nag in range(0, self.NUM_AGENTS)]
    
    def log(self, tag, value_dict, step):
        self.SUMMARY_LOGGER.add_scalars(tag, value_dict, step)
    
    def checkpoint(self, episode_num):
        e_num = episode_num
        for num_ in range(self.NUM_AGENTS):
            torch.save(
                    self.actor_local[num_].state_dict(), os.path.join(
                            self.CHECKPOINT_DIR, 'actor_local_%s_%s.pth' % (str(num_), str(e_num)))
            )
            torch.save(
                    self.actor_target[num_].state_dict(),
                    os.path.join(self.CHECKPOINT_DIR, 'actor_target_%s_%s.pth' % (str(num_), str(e_num)))
            )
            torch.save(
                    self.critic_local[num_].state_dict(),
                    os.path.join(self.CHECKPOINT_DIR, 'critic_local_%s_%s.pth' % (str(num_), str(e_num)))
            )
            torch.save(
                    self.critic_target[num_].state_dict(),
                    os.path.join(self.CHECKPOINT_DIR, 'critic_target_%s_%s.pth' % (str(num_), str(e_num)))
            )



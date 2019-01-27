from src.buffer import MemoryER
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from src import utils
from src import commons as cmn

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
            
            self.actor_local = [ag() for ag in args.ACTOR_NETWORK_FN]
            self.actor_target = [ag() for ag in args.ACTOR_NETWORK_FN]
            
            self.critic_local = [ag() for ag in args.CRITIC_NETWORK_FN]
            self.critic_target = [ag() for ag in args.CRITIC_NETWORK_FN]
            
            self.actor_local_optimizer = [args.ACTOR_OPTIMIZER_FN(ag.parameters()) for ag in self.actor_local]
            self.critic_local_optimizer = [args.CRITIC_OPTIMIZER_FN(ag.parameters()) for ag in self.critic_local]

            self.soft_update = utils.soft_update
            self.hard_update = utils.hard_update

            self.exploration_policy = [args.EXPLORATION_POLICY_FN() for _ in range(0, self.NUM_AGENTS)]
            
            self.SUMMARY_LOGGER = args.SUMMARY_LOGGER
        else:
            pass


class Agent(AgentParams):
    def __init__(self, args, mode='train'):
        super().__init__(args, mode)
        
        
    def act(self, states, action_value_range=(-1,1)):
        actions = []
        for num, state in enumerate(states):
            state = torch.from_numpy(np.expand_dims(state, axis=0)).float().to(device)
            state.require_grad = False
            self.actor_local[num].eval()
            with torch.no_grad():
                action = self.actor_local[num].forward(state)
            self.actor_local[num].train()
            
            if self.MODE == 'train':
                action += self.exploration_policy[num].sample()
            # Clip the actions to the the min and max limit of action probs
            action = np.clip(action, action_value_range[0], action_value_range[1])
            # print('Actions Value: ', actions)

            actions.append(action)
        return np.concatenate(actions)


    def learn(self, experiences, gamma, agent_num, running_time_step, log=True):
        state, action, reward, next_state, done = experiences

        #------------------------- CRITIC ---------------------------#
        # 1. First we fetch the expented value using critic local and the state-action pair
        expected_returns = self.critic_local[agent_num].forward(state, action)
        
        # 2. Target value is computed using critic/actor target and next_state-next_action pair
        next_actions = self.actor_target[agent_num].forward(next_state)
        target_value = self.critic_target[agent_num].forward(next_state, next_actions)
        target_returns = reward + gamma*target_value * (1-done)
        
        # 3. Now we calculate critic loss which is simple the mse of expected and target
        critic_loss = F.mse_loss(expected_returns, target_returns)
        
        # 4. Now we train the critic network, optimize the critic loss
        self.critic_local_optimizer[agent_num].zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local[agent_num].parameters(), 1)
        self.critic_local_optimizer[agent_num].step()

        # ------------------------- ACTOR ---------------------------#
        # 1. We re-get the actions from the actor network. The reason being that the current experience
        # might be old and the actor weights might have been updated many times since the experience was collected
        action_pred = self.actor_local[agent_num].forward(state)
        actor_loss = -self.critic_local[agent_num].forward(state, action_pred).mean()

        # 2. Now we train the actor network, optimize the actor loss
        self.actor_local_optimizer[agent_num].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local[agent_num].parameters(), 1)
        self.actor_local_optimizer[agent_num].step()

        #------------------------- LOGGING --------------------------#
        if log:
            tag = 'agent%i/losses' % agent_num
            value_dict = {
                'critic loss': critic_loss,
                'actor_loss': actor_loss
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
                state = state.permute(1, 0, 2)              # convert to [batch_size, num_agents, num_states]
                action = action.permute(1, 0, 2)            # convert to [batch_size, num_agents, num_states]
                reward = reward.permute(1, 0)               # convert to [batch_size, num_agents]
                next_state = next_state.permute(1, 0, 2)    # convert to [batch_size, num_agents, num_states]
                done = done.permute(1, 0)                   # convert to [batch_size, num_agents]
                
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
                    print('Performing soft-update at: ', running_time_step)

                    if self.DECAY_TAU:
                        tau = utils.exp_decay(self.TAU, self.TAU_DECAY_RATE, episode_num, self.TAU_MIN)

                    else:
                        tau = self.TAU

                    for num in range(0, self.NUM_AGENTS):
                        self.soft_update(self.critic_local[num], self.critic_target[num], tau)
                        self.soft_update(self.actor_local[num], self.actor_target[num], tau)
            else:
                raise ValueError('Only One of HARD_UPDATE and SOFT_UPDATE is to be activated')
    
    
    def log(self, tag, value_dict, step):
        self.SUMMARY_LOGGER.add_scalars(tag, value_dict, step)
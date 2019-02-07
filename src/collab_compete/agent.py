
import os
import numpy as np
import torch
import torch.nn.functional as F

from src import utils
from src import buffer
from src.collab_compete.model import Actor, Critic
import torch.optim as optim
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MDDPG():
    def __init__(self, args, mode):
        self.MODE = mode
        self.STATE_SIZE = args.STATE_SIZE
        self.ACTION_SIZE = args.ACTION_SIZE
        self.GAMMA = args.GAMMA
        self.NUM_AGENTS = args.NUM_AGENTS
        self.DATA_TO_BUFFER_BEFORE_LEARNING = args.DATA_TO_BUFFER_BEFORE_LEARNING

        self.summary_logger = args.SUMMARY_LOGGER
        
        if mode == 'train':
            self.LEARNING_FREQUENCY = args.LEARNING_FREQUENCY
            self.BATCH_SIZE = args.BATCH_SIZE
            self.hard_update = utils.hard_update
    
            self.AGENTS = [DDPG(args, agent_id, mode=self.MODE, log=True) for agent_id in range(self.NUM_AGENTS)]
            self.MEMORY = buffer.MemoryER(buffer_size=10000, batch_size=256, seed=0, action_dtype='float')
            
        else:
            pass
    
    def act(self, state, action_value_range, running_time_step):
        all_actions = []
        for agent, state in zip(self.AGENTS, state):
            action = agent.act(state, action_value_range, running_time_step, self.summary_logger)
            all_actions.append(action)
        return np.concatenate(all_actions)
    
    def step(self, states, actions, rewards, next_states, dones, running_time_step):
        
        

        states = np.expand_dims(np.stack(states, axis=0), axis=0)
        actions = np.expand_dims(np.stack(actions, axis=0), axis=0)
        rewards = np.expand_dims(np.stack(np.array(rewards, dtype=np.float32).reshape(-1, 1), axis=0), axis=0)
        next_states = np.expand_dims(np.stack(next_states, axis=0), axis=0)
        dones = np.expand_dims(np.stack(np.array(dones, dtype=np.bool).reshape(-1, 1), axis=0), axis=0)
        
        # print(states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)
        # print('rewards: ', rewards)
        self.MEMORY.add(states, actions, rewards, next_states, dones)
        
        if (running_time_step % self.LEARNING_FREQUENCY) == 0:
            if len(self.MEMORY) > self.DATA_TO_BUFFER_BEFORE_LEARNING:
                # print('Agent %s : Memory Size: ', len(self.MEMORY[num]))
                experiences = self.MEMORY.sample()
                
                all_states, all_actions, all_rewards, all_next_states, all_dones = experiences
                # print(all_states.shape, all_actions.shape, all_rewards.shape, all_next_states.shape, all_dones.shape)
                
                all_states = all_states.view(all_states.shape[0], -1)
                all_actions = all_actions.view(all_actions.shape[0], -1)
                all_rewards = all_rewards.view(all_rewards.shape[0], -1)
                all_next_states = all_next_states.view(all_next_states.shape[0], -1)
                all_dones = all_dones.view(all_dones.shape[0], -1)

                self.learn(all_states, all_actions, all_rewards, all_next_states, all_dones, running_time_step)
        
        if running_time_step  > self.DATA_TO_BUFFER_BEFORE_LEARNING:
            for ag_num, agent in enumerate(self.AGENTS):
                agent.update(running_time_step)
                
        
    
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
                        all_rewards, all_dones, self.GAMMA, ag_num, running_time_step, self.summary_logger)
            
    def log(self, tag, value_dict, step):
        self.summary_logger.add_scalars(tag, value_dict, step)
            
    def checkpoint(self, episode_num):
        for ag_num, agent in enumerate(self.AGENTS):
            agent.checkpoint(ag_num, episode_num)


class DDPG:
    def __init__(self, args, agent_id, mode, log):
        random.seed(0)
        self.MODE = mode
        self.AGENT_ID = agent_id
        self.LOG = log
        self.TAU = args.TAU
        self.NUM_AGENTS = args.NUM_AGENTS
        self.CHECKPOINT_DIR = args.CHECKPOINT_DIR

        self.SOFT_UPDATE = args.SOFT_UPDATE
        self.HARD_UPDATE = args.HARD_UPDATE
        self.HARD_UPDATE_FREQUENCY = args.HARD_UPDATE_FREQUENCY
        self.SOFT_UPDATE_FREQUENCY = args.SOFT_UPDATE_FREQUENCY
        
        self.noise = args.NOISE_FN()
        self.noise_amplitude_decay = args.NOISE_AMPLITUDE_FN()
        
        self.actor_local = Actor(
                state_size=args.STATE_SIZE, action_size=args.ACTION_SIZE, layer_in_out=[256, 256, 2], seed=0
        )
        self.actor_target = Actor(
                state_size=args.STATE_SIZE, action_size=args.ACTION_SIZE, layer_in_out=[256, 256, 2], seed=0
        )
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=1e-4)
        
        # Critic Network
        self.critic_local = Critic(
                state_size=args.STATE_SIZE, action_size=args.ACTION_SIZE, layer_in_out=[256, 256, 1], seed=0
        )
        self.critic_target = Critic(
                state_size=args.STATE_SIZE, action_size=args.ACTION_SIZE, layer_in_out=[256, 256, 1], seed=0
        )
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=1e-3, weight_decay=0.0)

        self.soft_update = utils.soft_update
        utils.hard_update(local_model=self.actor_local, target_model=self.actor_target)
        utils.hard_update(local_model=self.critic_local, target_model=self.critic_target)
    
    def act(self, state, action_value_range, running_time_step, summary_logger):
        # print(state)
        self.noise_amplitude = self.noise_amplitude_decay.sample()
        state = torch.from_numpy(np.expand_dims(state, axis=0)).float().to(device)
        
        state.require_grad = False
        self.actor_local.eval()
        with torch.no_grad():
            # action = self.actor_local[num].forward(state)
            action = self.actor_local.forward(state).cpu().data.numpy()
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
            self.log(tag, value_dict, running_time_step, summary_logger)
        return action
    
    def update(self, running_time_step):
        
        if self.HARD_UPDATE:
            if (running_time_step % self.HARD_UPDATE_FREQUENCY) == 0:
                for num in range(0, self.NUM_AGENTS):
                    self.hard_copy_weights(self.actor_local, self.actor_target)
                    self.hard_copy_weights(self.critic_local, self.critic_target)

        elif self.SOFT_UPDATE:
            if (running_time_step % self.SOFT_UPDATE_FREQUENCY) == 0:
                # print('Performing soft-update at: ', running_time_step)
                self.soft_update(self.critic_local, self.critic_target, self.TAU)
                self.soft_update(self.actor_local, self.actor_target, self.TAU)
        else:
            raise ValueError('Only One of HARD_UPDATE and SOFT_UPDATE is to be activated')

    
    def learn(
            self, all_states, all_actions, all_next_states, all_next_actions, all_action_pred,
            all_rewards, all_dones, gamma, agent_num, running_time_step, summary_logger
    ):
        # ------------------------- CRITIC --------------------------- #
        # 1. First we fetch the expented value using critic local and the state-action pair
        expected_returns = self.critic_local.forward(all_states, all_actions)
        
        # 2. Target value is computed using critic/actor target and next_state-next_action pair.
        # TODO: next_states should also be serverd from both the networks
        # with torch.no_grad():
        target_value = self.critic_target.forward(all_next_states, all_next_actions)
        
        target_returns = all_rewards[:, agent_num].reshape(-1, 1) + \
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
        # TODO: buffer for the other agent no longer exists. Therefore we detach the other agent's contribution from the]=[[[[[[[[[[[[[[k
        # TODO: current graph and only use its actions as another set of features to guide training.
        actions_pred = [act_p if i == self.AGENT_ID else act_p.detach() for i, act_p in enumerate(all_action_pred)]
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        actor_loss = -self.critic_local.forward(all_states, actions_pred).mean()
        
        # 2. Now we train the actor network, optimize the actor loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_local[agent_num].parameters(), 1)
        self.actor_optimizer.step()
        
        # self.soft_update(self.critic_local, self.critic_target, self.TAU)
        # self.soft_update(self.actor_local, self.actor_target, self.TAU)
        
        # ------------------------- LOGGING --------------------------#
        if self.LOG:
            tag = 'agent%i/actor_loss' % self.AGENT_ID
            value_dict = {
                'actor_loss': abs(float(actor_loss))
            }
            self.log(tag, value_dict, running_time_step, summary_logger)
            
            tag = 'agent%i/critic_loss' % self.AGENT_ID
            value_dict = {
                'critic loss': abs(float(critic_loss)),
            }
            self.log(tag, value_dict, running_time_step, summary_logger)
    
    def reset(self):
        self.noise.reset()
    
    def log(self, tag, value_dict, step, summary_logger):
        summary_logger.add_scalars(tag, value_dict, step)
        
    def checkpoint(self, ag_num, episode_num):
        torch.save(
                self.actor_local.state_dict(), os.path.join(
                        self.CHECKPOINT_DIR, 'actor_local_%s_%s.pth' % (str(ag_num), str(episode_num)))
        )
        torch.save(
                self.actor_target.state_dict(),
                os.path.join(self.CHECKPOINT_DIR, 'actor_target_%s_%s.pth' % (str(ag_num), str(episode_num)))
        )
        torch.save(
                self.critic_local.state_dict(),
                os.path.join(self.CHECKPOINT_DIR, 'critic_local_%s_%s.pth' % (str(ag_num), str(episode_num)))
        )
        torch.save(
                self.critic_target.state_dict(),
                os.path.join(self.CHECKPOINT_DIR, 'critic_target_%s_%s.pth' % (str(ag_num), str(episode_num)))
        )


############### This was the other person implementation.




import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque

from src import utils
from src.collab_compete import model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPG():
    """Meta agent that contains the two DDPG agents and shared replay buffer."""
    
    def __init__(self, args, mode):
        """

        :param args:    Configuration arguments
        :param mode:    Training or Testing mode
        """
        self.SEED = args.SEED
        self.ACTION_SIZE = args.ACTION_SIZE
        self.BUFFER_SIZE = args.BUFFER_SIZE
        self.BATCH_SIZE = args.BATCH_SIZE
        self.LEARNING_FREQUENCY = args.LEARNING_FREQUENCY
        self.GAMMA = args.GAMMA
        self.NUM_AGENT = args.NUM_AGENTS
        self.DATA_TO_BUFFER_BEFORE_LEARNING = args.DATA_TO_BUFFER_BEFORE_LEARNING
        
        self.SUMMARY_LOGGER = args.SUMMARY_LOGGER
        self.AGENTS = [DDPG(args, i, mode) for i in range(self.NUM_AGENT)]
        
        # create shared replay buffer
        self.MEMORY = ReplayBuffer(self.ACTION_SIZE, self.BUFFER_SIZE, self.BATCH_SIZE, self.SEED)
    
    def step(self, all_states, all_actions, all_rewards, all_next_states, all_dones, running_time_step):
        all_states = all_states.reshape(1, -1)  # reshape 2x24 into 1x48 dim vector
        all_next_states = all_next_states.reshape(1, -1)  # reshape 2x24 into 1x48 dim vector
        self.MEMORY.add(all_states, all_actions, all_rewards, all_next_states, all_dones)
        
        # Learn every update_every time steps.
        if running_time_step % self.LEARNING_FREQUENCY == 0:
            if len(self.MEMORY) > self.BATCH_SIZE:
                experiences = [self.MEMORY.sample() for _ in range(self.NUM_AGENT)]
                self.learn(experiences, self.GAMMA, running_time_step)
        
        if running_time_step > self.DATA_TO_BUFFER_BEFORE_LEARNING:
            for ag_num, agent in enumerate(self.AGENTS):
                agent.update(running_time_step)
    
    def act(self, all_states, action_value_range, running_time_step):
        # pass each agent's state from the environment and calculate its action
        all_actions = []
        for agent, state in zip(self.AGENTS, all_states):
            action = agent.act(state)
            all_actions.append(action)
        return np.array(all_actions).reshape(1, -1)  # reshape 2x2 into 1x4 dim vector
    
    def learn(self, experiences, gamma, running_time_step):
        # each agent uses its own actor to calculate next_actions
        all_next_actions = []
        all_action_pred = []
        for i, agent in enumerate(self.AGENTS):
            states, _, _, next_states, _ = experiences[i]
            agent_id = torch.tensor([i]).to(device)
            state = states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            next_state = next_states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            
            action_pred = agent.actor_local(state)
            next_actions = agent.actor_target(next_state)
            
            # You might be wondering why do we get actions for the state, if we already have action for those
            # state using the replay buffer. Well the network weights might have changed after we stashed
            # the experience into the replay buffer. So here we get the fresh actions for the same states.
            all_action_pred.append(action_pred)
            all_next_actions.append(next_actions)
        
        # each agent learns from its experience sample
        for i, agent in enumerate(self.AGENTS):
            agent.learn(i, experiences[i], gamma, all_next_actions, all_action_pred, running_time_step)
    
    def log(self, tag, value_dict, step):
        self.SUMMARY_LOGGER.add_scalars(tag, value_dict, step)
    
    def checkpoint(self, episode_num):
        for ag_num, agent in enumerate(self.AGENTS):
            agent.checkpoint(ag_num, episode_num)


class DDPG():
    """DDPG agent with own actor and critic."""
    
    def __init__(self, args, agent_id, mode):
        """
        :param args:            Config parameters
        :param agent_id:        The agent id to run
        :param mode:            train or test
        """
        self.agent_id = agent_id
        self.mode = mode
        self.SEED = args.SEED
        random.seed(self.SEED)
        
        self.NOISE = args.NOISE_FN()
        self.STATE_SIZE = args.STATE_SIZE
        self.ACTION_SIZE = args.ACTION_SIZE
        self.TAU = args.TAU
        self.ACTOR_LEARNING_RATE = args.ACTOR_LEARNING_RATE
        self.CRITIC_LEARNING_RATE = args.CRITIC_LEARNING_RATE
        self.WEIGHT_DECAY = args.WEIGHT_DECAY
        
        self.IS_HARD_UPDATE = args.IS_HARD_UPDATE
        self.IS_SOFT_UPDATE = args.IS_SOFT_UPDATE
        self.SOFT_UPDATE_FREQUENCY = args.SOFT_UPDATE_FREQUENCY
        self.HARD_UPDATE_FREQUENCY = args.HARD_UPDATE_FREQUENCY
        self.NOISE_AMPLITUDE_DEACAY = args.NOISE_AMPLITUDE_DECAY_FN()
        
        self.CHECKPOINT_DIR = args.CHECKPOINT_DIR
        self.SUMMARY_LOGGER = args.SUMMARY_LOGGER
        
        # Actor Network
        self.actor_local = model.Actor(self.STATE_SIZE, self.ACTION_SIZE, [256, 256, 2], self.SEED).to(device)
        self.actor_target = model.Actor(self.STATE_SIZE, self.ACTION_SIZE, [256, 256, 2], self.SEED).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.ACTOR_LEARNING_RATE)
        
        # Critic Network
        self.critic_local = model.Critic(self.STATE_SIZE, self.ACTION_SIZE, [256, 256, 1], self.SEED).to(device)
        self.critic_target = model.Critic(self.STATE_SIZE, self.ACTION_SIZE, [256, 256, 1], self.SEED).to(device)
        self.critic_optimizer = optim.Adam(
                self.critic_local.parameters(), lr=self.CRITIC_LEARNING_RATE, weight_decay=self.WEIGHT_DECAY
        )
        
        # Set weights for local and target actor, respectively, critic the same
        utils.hard_update(self.actor_local, self.actor_target)
        utils.hard_update(self.critic_local, self.critic_target)
        
        # Noise process
        self.noise = args.NOISE_FN()
    
    def act(self, state):
        """Returns actions for given state as per current policy."""
        self.noise_amplitude = self.NOISE_AMPLITUDE_DEACAY.sample()
        state = torch.from_numpy(state).float().to(device)
        
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if self.mode == 'train':
            self.noise_val = self.noise.sample() * self.noise_amplitude
            # print(self.noise_val)
            action += self.noise_val
            action += self.noise_val
        
        return np.clip(action, -1, 1)
    
    def reset(self):
        self.noise.reset()
    
    def learn(self, agent_id, experiences, gamma, all_next_actions, all_actions_pred, running_time_step):
        """
        :param agent_id:            The agent id to run
        :param experiences:         [all_states, all_actions, all_rewards, all_next_states, all_dones]
        :param gamma:               Discount
        :param all_next_actions:    next_actions computed by actor_target for both the agents using next_states
        :param all_actions_pred:    action_pred computed by actor_local for both the agents using states
        :param running_time_step:   The running time step passed form the environment code
        :return:
        """
        
        all_states, all_actions, all_rewards, all_next_states, all_dones = experiences
        
        # ---------------------------- update critic ---------------------------- #
        self.critic_optimizer.zero_grad()
        agent_id = torch.tensor([agent_id]).to(device)
        all_next_actions = torch.cat(all_next_actions, dim=1).to(device)
        with torch.no_grad():
            q_targets_next = self.critic_target(all_next_states, all_next_actions)
        
        q_expected = self.critic_local(all_states, all_actions)
        q_targets = all_rewards.index_select(1, agent_id) + (gamma * q_targets_next * (1 - all_dones.index_select(1,
                                                                                                                  agent_id)))
        critic_loss = F.mse_loss(q_expected, q_targets.detach())
        # minimize loss
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #
        self.actor_optimizer.zero_grad()
        actions_pred = [actions if i == self.agent_id else actions.detach()
                        for i, actions in enumerate(all_actions_pred)]
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        actor_loss = -self.critic_local(all_states, actions_pred).mean()
        # minimize loss
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ------------------------- LOGGING --------------------------#
        if self.log:
            tag = 'agent%i/actor_loss' % self.agent_id
            value_dict = {
                'actor_loss': abs(float(actor_loss))
            }
            self.log(tag, value_dict, running_time_step)
            
            tag = 'agent%i/critic_loss' % self.agent_id
            value_dict = {
                'critic loss': abs(float(critic_loss)),
            }
            self.log(tag, value_dict, running_time_step)
    
    def update(self, running_time_step):
        
        if self.IS_HARD_UPDATE:
            if (running_time_step % self.HARD_UPDATE_FREQUENCY) == 0:
                utils.hard_update(self.actor_local, self.actor_target)
        
        elif self.IS_SOFT_UPDATE:
            if (running_time_step % self.SOFT_UPDATE_FREQUENCY) == 0:
                utils.soft_update(self.critic_local, self.critic_target, self.TAU)
                utils.soft_update(self.actor_local, self.actor_target, self.TAU)
        else:
            raise ValueError('Only One of HARD_UPDATE and SOFT_UPDATE is to be activated')
    
    def log(self, tag, value_dict, step):
        self.SUMMARY_LOGGER.add_scalars(tag, value_dict, step)
    
    def checkpoint(self, ag_num, episode_num):
        torch.save(
                self.actor_local.state_dict(), os.path.join(
                        self.CHECKPOINT_DIR, 'actor_local_%s_%s.pth' % (str(ag_num), str(episode_num)))
        )
        torch.save(
                self.critic_local.state_dict(),
                os.path.join(self.CHECKPOINT_DIR, 'critic_local_%s_%s.pth' % (str(ag_num), str(episode_num)))
        )


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): Random seed
        """
        random.seed(seed)
        self.action_size = action_size
        self.MEMORY = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.MEMORY.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.MEMORY, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
                device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
                device)
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.MEMORY)

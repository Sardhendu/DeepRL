import copy
import numpy as np
import random


class Config:
    def __init__(self):
        self.device = 'cpu'
        self.seed = 0
        self.network_fn = None
        self.optimizer_fn = None
        self.memory_fn = None
        self.noise_fn = None
        self.hidden_units = None
        self.num_agents = 1
        
        self.actor_hidden_units = (64, 64)
        self.actor_network_fn = None
        self.actor_optimizer_fn = None
        self.actor_learning_rate = 0.004
        
        self.critic_hidden_units = (64, 64)
        self.critic_network_fn = None
        self.critic_optimizer_fn = None
        self.critic_learning_rate = 0.003
        
        self.tau = 1e-3
        self.weight_decay = 0
        self.states = None
        self.state_size = None
        self.action_size = None
        self.learning_rate = 0.001
        self.gate = None
        self.batch_size = 256
        self.buffer_size = int(1e5)
        self.discount = 0.999
        self.update_every = 16
        self.gradient_clip = None
        self.entropy_weight = 0.01
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.995
        
        
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
    
    
    
import torch
import numpy as np
from collections import deque
from unityagents import UnityEnvironment
from src.continuous_control2.model import Actor, Critic
from src.continuous_control2.memory import ReplayBuffer
from src.continuous_control2.agent import DDPGAgent

import matplotlib.pyplot as plt

env = UnityEnvironment(file_name='Reacher.app')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]


config = Config()

config.seed = 2
config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config.action_size = brain.vector_action_space_size
config.states = env_info.vector_observations
config.state_size = config.states.shape[1]
config.num_agents = len(env_info.agents)

config.actor_hidden_units = (512, 256)
config.actor_learning_rate = 1e-4
config.actor_network_fn = lambda: Actor(config.action_size, config.state_size, config.actor_hidden_units, config.seed).to(config.device)
config.actor_optimizer_fn = lambda params: torch.optim.Adam(params, lr=config.actor_learning_rate)

config.critic_hidden_units = (512, 256)
config.critic_learning_rate = 3e-4
config.weight_decay = 0
config.critic_network_fn = lambda: Critic(config.action_size, config.state_size, config.critic_hidden_units, config.seed).to(config.device)
config.critic_optimizer_fn = lambda params: torch.optim.Adam(params, lr=config.critic_learning_rate)

config.batch_size = 512
config.buffer_size = int(1e6)
config.discount = 0.99
config.update_every = 4
config.memory_fn = lambda: ReplayBuffer(config.action_size, config.buffer_size, config.batch_size, config.seed, config.device)

config.noise_fn = lambda: OUNoise(config.action_size, config.seed)


agent = DDPGAgent(config)


def ddpg(n_episodes=5000, max_t=2000):
    all_scores = []
    scores_window = deque(maxlen=100)
    
    for i_episode in range(1, n_episodes + 1):
        
        agent.reset()
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(config.num_agents)
        
        for pp in range(max_t):
            print (pp)
            actions = agent.act(states)
            # print(actions)
            env_info = env.step(actions)[brain_name]
            rewards = env_info.rewards
            next_states = env_info.vector_observations
            dones = env_info.local_done
            
            agent.step(states, actions, rewards, next_states, dones)
            
            scores += rewards
            states = next_states
        
        avg_score = np.mean(scores)
        scores_window.append(avg_score)
        all_scores.append(avg_score)
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
    
    return all_scores


ddpg()
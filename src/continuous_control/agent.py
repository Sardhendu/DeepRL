import copy
import random

import numpy as np
import torch
from src.buffer import MemoryER

from src.continuous_control.model import DDPGActor, DDPGCritic, Optimize

#TODO: remove epsilon and add noise as per the DDPG Paper


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


class RLAgent:
    def __init__(self, args, env_type):
        self.args = args
        self.env_type = env_type

        self.ACTION_SIZE = self.args.ACTION_SIZE
        self.BATCH_SIZE = self.args.BATCH_SIZE
        self.UPDATE_AFTER_STEP = self.args.UPDATE_AFTER_STEP
        self.GAMMA = self.args.GAMMA
        self.HARD_UPDATE = self.args.HARD_UPDATE
        self.HARD_UPDATE_FREQUENCY = self.args.HARD_UPDATE_FREQUENCY
        self.BUFFER_SIZE = self.args.BUFFER_SIZE
        self.BATCH_SIZE = self.args.BATCH_SIZE
        
        
        # Create the Local Network and Target Network for the Actor
        self.local_actor = DDPGActor(self.args.STATE_SIZE, self.args.ACTION_SIZE, seed=34)
        self.target_actor = DDPGActor(self.args.STATE_SIZE, self.args.ACTION_SIZE, seed=98)
        self.local_actor_optimizer = Optimize(learning_rate=self.args.LEARNING_RATE).adam(self.local_actor.parameters())

        # Create the Local Network and Target Network for the Critic
        self.local_critic = DDPGCritic(self.args.STATE_SIZE, self.args.ACTION_SIZE, seed=294)
        self.target_critic = DDPGCritic(self.args.STATE_SIZE, self.args.ACTION_SIZE, seed=551)
        self.local_critic_optimizer = Optimize(learning_rate=self.args.LEARNING_RATE).adam(
                self.local_critic.parameters())

        self.t_step = 0

    def action(self, state):
        """
        
        :param state:
        :return:        A numpy array with action value distribution
        
        ### Actor Action: Actons are chosen by actor
        Select action "at" according to the current policy and the exploration noise
        at = μ(st | θμ) + Nt
        
        Here
            --> θμ : Weights of Actor Network
            --> Nt : Are the random noise that jitters the action distribution introducing randomness to for exploration
            
        """
        
        # Set the state to evaluation get actions and reset network to training phase
        # print('4223423 ', state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        state.requires_grad = False
        self.local_actor.eval()
        with torch.no_grad():
            actions = self.local_actor.forward(state).cpu().data.numpy()
        self.local_actor.train()
        return actions

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

    
    
class DDPGAgent(RLAgent):
    def __init__(self, args, env_type, seed):
        super().__init__(args, env_type)

        # Get the Noise
        self.noise = OUNoise(size=self.ACTION_SIZE, seed=seed)

        # Create The memory Buffer
        self.memory = MemoryER(self.BUFFER_SIZE, self.BATCH_SIZE, seed, action_dtype='float')
        
    def act(self, state, add_noise=True, action_value_range=(-1, 1)):
        """
        :param state:               Input n dimension state feature space
        :param add_noise:           bool True or False
        :param action_prob_range:   The limit range of action values
        :return:                    The action value distribution with added noise (if required)
        
        at = μ(st | θμ) + Nt
        """
        # Add Random noise to the action space distribution to foster exploration
        if add_noise:
            actions = self.action(state) + self.noise.sample()
        else:
            actions = self.action(state)
        
        # Clip the actions to the the min and max limit of action probs
        actions = np.clip(actions, action_value_range[0], action_value_range[1])
        return actions
        
    def step(self, state, action, reward, next_state, done, episode_num, running_time_step):
        
        # Store experience to the replay buffer
        print('actionactionaction ', action)
        self.memory.add(state, action, reward, next_state, done)
        
        # When the memory is atleast full as the batch size and if the step num is a factor of UPDATE_AFTER_STEP
        # then we learn the parameters of the network
        # Update the weights of local network and soft-update the weighs of the target_network
        self.t_step = (self.t_step + 1) % self.UPDATE_AFTER_STEP  # Run from {1->UPDATE_AFTER_STEP}
        # print('[Step] Current Step is: ', self.tstep)
        if self.t_step == 0:
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.GAMMA, episode_num, running_time_step)
        
        if self.HARD_UPDATE:
            if ((running_time_step + 1) % self.HARD_UPDATE_FREQUENCY) == 0:
                self.hard_update(self.local_actor, self.target_actor)
                self.hard_update(self.local_critic, self.target_critic)
                
    def learn(self, experiences, gamma, episode_num, running_time_step):
        print('Learning .... ', episode_num, running_time_step)
        states, actions, rewards, states_next, dones = experiences
        print(states.shape)
        print('')
        print(actions)
        print('')
        print(rewards)
        print(states_next)
        print('')
        print(dones)
        pass
    
    # def learn(self):
    
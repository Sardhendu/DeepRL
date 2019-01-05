import copy
import random
import numpy as np


from buffer import MemoryER
from continuous_control.model import DDPGActor, DDPGCritic, Optimize

#TODO: remove epsilon and add noise as per the DDPG Paper



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
    def __init__(self, args, env_type, seed):
        self.args = args
        self.env_type = env_type
        self.seed = seed
        
        # Create the Local Network and Target Network for the Actor
        self.local_actor = DDPGActor(self.args.STATE_SIZE, self.args.ACTION_SIZE)
        self.target_actor = DDPGActor(self.args.STATE_SIZE, self.args.ACTION_SIZE)
        self.local_actor_optimizer = Optimize(learning_rate=self.args.LEARNING_RATE).adam(self.local_actor.parameters())

        # Create the Local Network and Target Network for the Critic
        self.local_critic = DDPGCritic(self.args.STATE_SIZE, self.args.ACTION_SIZE)
        self.target_actor = DDPGCritic(self.args.STATE_SIZE, self.args.ACTION_SIZE)
        self.local_critic_optimizer = Optimize(learning_rate=self.args.LEARNING_RATE).adam(
                self.local_critic.parameters())
        
        
        
        # pass
    def action(self, state, eps):
        """
        
        :param state:
        :param eps:
        :return:
        
        ### Actor Action
        Select action "at" according to the current policy and the exploration noise
        at = μ(st | θμ) + Nt
        
        Here
            --> θμ : Weights of Actor Network
            --> Nt : Are the random noise that jitters the action distribution introducing randomness to for exploration
            
        # Critic Action
        """
        
    
    
    
    
    
class DDPGAgent(RLAgent):
    def __init__(self, args, env_type, seed):
        # Get the Noise
        self.noise = OUNoise(size=args.ACTION_SIZE, seed=4)
        
        # Create The memory Buffer
        self.memory = MemoryER(args.BUFFER_SIZE, args.BATCH_SIZE, seed)
        
        super().__init__(args, env_type, seed)
        
    def act(self):
        pass
    
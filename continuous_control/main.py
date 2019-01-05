
from unityagents import UnityEnvironment
import numpy as np
from collections import deque

class ContinuousControl:
    def __init__(self, env_type='single', mode='train'):
        self.env_type = env_type
        if env_type == 'single':
            self.base_env = UnityEnvironment(file_name='Reacher.app')
        elif env_type == 'double':
            pass
        else:
            raise ValueError('Environment type not understood ....')

        self.brain_name = self.base_env.brain_names[0]
        self.brain = self.base_env.brains[self.brain_name]
        
        if mode == 'train':
            self.train = True
        else:
            self.train = False
        self.reset()

        if env_type == 'single':
            self.state_size = len(self.state)
        elif env_type == 'double':
            self.state_size = self.state.shape
        else:
            raise ValueError('Environment type not understood ....')

        print(self.state_size)
        
    def reset(self):
        self.env_info = self.base_env.reset(train_mode=self.train)[self.brain_name]
        self.get_state()
        return self.state

    def get_state(self):
        if self.env_type == 'single':
            self.state = self.env_info.vector_observations[0]
        elif self.env_type == 'double':
            pass
        else:
            raise ValueError('Environment type not understood ....')
        
    def close(self):
        self.base_env.close()
        
        
class PolicyGradients:
    def __init__(self, args, env, env_type='vector'):
        self.env = env
        self.args = args
        self.score_window_size = 100
        
    def train(self):
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=self.score_window_size)  # last score_window_size scores
    
        running_time_step = 0
        for i_episode in range(1, self.args.NUM_EPISODES + 1):
            state = self.env.reset()
    
class Config:
    NUM_EPISODES = 2000

    
env = ContinuousControl(mode='train')
dqn = PolicyGradients(Config, env)
        
        
    
    
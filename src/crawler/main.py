# import matplotlib.pyplot as plt

import numpy as np
from src.crawler import agent
from src.crawler.config import TrainConfig
from unityagents import UnityEnvironment


class ENV:
    def __init__(self, env_type, mode='train'):
        if mode != 'train':
            print('[Mode] Setting to Test Mode')
            self.train = False
        else:
            print('[Mode] Setting to Train Mode')
            self.train = True
        
        if env_type == 'vector':
            self.base_env = UnityEnvironment(file_name='Crawler.app')
        else:
            raise ValueError('Environment type not understood ....')
        
        self.brain_name = self.base_env.brain_names[0]
        self.brain = self.base_env.brains[self.brain_name]
        self.action_size = self.brain.vector_action_space_size
        self.reset()
    
    def reset(self):
        self.env_info = self.base_env.reset(train_mode=self.train)[self.brain_name]
        return self.get_state()
    
    def num_agents(self):
        return len(self.env_info.agents)
    
    def get_state(self):
        return self.env_info.vector_observations
    
    def step(self, action):
        self.env_info = self.base_env.step(action)[self.brain_name]  # send the action to the environment
        next_states = self.get_state()
        rewards = self.env_info.rewards
        dones = self.env_info.local_done
        return next_states, rewards, dones, None
    
    def close(self):
        self.base_env.close()


def run():
    env = ENV(env_type='vector', mode='train')
    env_info = env.reset()  # reset the environment
    num_agents = env.num_agents()
    states = env.get_state()
    scores = np.zeros(num_agents)  # initialize the score (for each agent)
    
    print('Number of agents: ', num_agents)
    print('State shape: ', states.shape)
    print('Action size: ', env.action_size)
    
    agent_obj = agent.Agent(TrainConfig)
    
    running_timestep = 0
    for i in range(0, 1):
        for j in range(0, 1000):
            actions = agent_obj.act(states)
            # print(np.min(actions), np.max(actions))
            next_states, rewards, dones, _ = env.step(actions)
            agent_obj.step(states, actions, rewards, next_states, dones, running_timestep)
            # print(rewards)
            scores += rewards
            states = next_states
            if np.any(dones):
                break
                
            running_timestep += 1
        print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


run()

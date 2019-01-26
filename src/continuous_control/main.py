# import numpy as np
# from collections import deque
# from unityagents import UnityEnvironment
#
#
# from src.continuous_control.agent import DDPGAgent
# from src.exploration import OUNoise
# from src.continuous_control.model import Actor, Critic
# from src.buffer import MemoryER
# import torch

import numpy as np
from collections import deque
from unityagents import UnityEnvironment


class ContinuousControl:
    def __init__(self, env_type, mode='train'):
        if mode != 'train':
            print('[Mode] Setting to Test Mode')
            self.train = False
        else:
            print('[Mode] Setting to Train Mode')
            self.train = True
            
        if env_type == 'single':
            self.base_env = UnityEnvironment(file_name='Reacher_single.app')
        elif env_type == 'multi':
            self.base_env = UnityEnvironment(file_name='Reacher_multi.app')
        else:
            raise ValueError('Environment type not understood ....')
        
        self.brain_name = self.base_env.brain_names[0]
        self.brain = self.base_env.brains[self.brain_name]
        self.action_size = self.brain.vector_action_space_size
    
    def reset(self):
        self.env_info = self.base_env.reset(train_mode=self.train)[self.brain_name]
        return self.get_state()
    
    def get_state(self):
        return self.env_info.vector_observations
    
    def step(self, action):
        # print(self.brain_name)
        # print(action)
        self.env_info = self.base_env.step(action)[self.brain_name]  # send the action to the environment
        next_states = self.get_state()
        rewards = self.env_info.rewards
        dones = self.env_info.local_done
        return next_states, rewards, dones, None
    
    def close(self):
        self.base_env.close()




class DDPG:
    
    @staticmethod
    def train(env, agent, n_episodes=5000, max_t=2000):
        all_scores = []
        scores_window = deque(maxlen=100)
        NUM_AGENTS = 20
        running_time_step = 1
        for i_episode in range(1, n_episodes + 1):
            
            agent.reset()
            states = env.reset()
            scores = np.zeros(NUM_AGENTS)  # NUM_AGENTS
            for pp in range(max_t):
                actions = agent.act(states)
                next_states, rewards, dones, _ = env.step(actions)
                
                agent.step(states, actions, rewards, next_states, dones, i_episode, running_time_step)
                
                scores += rewards
                states = next_states
                running_time_step += 1
            
            avg_score = np.mean(scores)
            scores_window.append(avg_score)
            all_scores.append(avg_score)
            
            agent.stats_dict['score'].append(avg_score)
            agent.save_stats()
            
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                agent.save_checkpoints(i_episode)
            
            if np.mean(scores_window) >= 30.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                        i_episode - 100, np.mean(scores_window))
                )
                agent.save_checkpoints(i_episode)
                break
        
        return all_scores


    @staticmethod
    def test(env, agent, trials=3, steps=200):
        NUM_AGENTS = 20
        for i in range(trials):
            scores = np.zeros(NUM_AGENTS)
            print('Starting Testing ...')
            state = env.reset()
            for j in range(steps):
                action = agent.act(state)
                # env.render()
                state, reward, done, _ = env.step(action)
                print(reward)
                scores += reward
                if reward != 0:
                    print("Current Reward:", reward, "Total Reward:", scores)
                if done:
                    print('Done.')
                    break
        env.close()
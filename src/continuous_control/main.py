

import numpy as np
from collections import deque
from src.continuous_control.config import TrainConfig
from unityagents import UnityEnvironment
from src.continuous_control.agent import DDPGAgent


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
    def __init__(self, args, env_type, mode):
        self.args = args
        self.env_type = env_type
        self.env = ContinuousControl(env_type=env_type, mode=mode)
        
    def train(self):
        if self.env_type == 'single':
            NUM_AGENTS = 1
        elif self.env_type == 'multi':
            NUM_AGENTS = 20
        else:
            raise ValueError('Only single and multi environment accepted')
            
        self.agent = DDPGAgent(self.args, self.env_type, mode='train', agent_id=0)
        
        all_scores = []
        scores_window = deque(maxlen=100)
        running_timestep = 1
        
        for i_episode in range(1, self.args.NUM_EPISODES + 1):
            # agent.reset()
            states = self.env.reset()
            scores = np.zeros(NUM_AGENTS)  # NUM_AGENTS
            for pp in range(self.args.NUM_TIMESTEPS):
                actions = self.agent.act(states, action_value_range=(-1, 1), running_timestep=running_timestep)
                next_states, rewards, dones, _ = self.env.step(actions)
                
                self.agent.step(states, actions, rewards, next_states, dones, running_timestep)
                
                scores += rewards
                states = next_states
                running_timestep += 1
            
            avg_score = np.mean(scores)
            scores_window.append(avg_score)
            all_scores.append(avg_score)
            
            
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                self.agent.save_checkpoints(i_episode)
            
            if np.mean(scores_window) >= 30.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                        i_episode - 100, np.mean(scores_window))
                )
                self.agent.save_checkpoints(i_episode)
                break
        
        return all_scores

    def test(self, agent, trials=3, steps=200):
        NUM_AGENTS = 20
        for i in range(trials):
            scores = np.zeros(NUM_AGENTS)
            print('Starting Testing ...')
            state = self.env.reset()
            for j in range(steps):
                action = agent.act(state)
                # env.render()
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                # print(reward)
                scores += reward
                if reward != 0:
                    print("Current Reward:", reward, "Total Reward:", scores)
                if done:
                    print('Done.')
                    break
        self.env.close()
        
obj_ = DDPG(args=TrainConfig, env_type='multi', mode='train')
obj_.train()

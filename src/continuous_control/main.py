

import numpy as np
from collections import deque
from src.continuous_control.config import TrainConfig, TestConfig
from unityagents import UnityEnvironment
from src.continuous_control.agent import DDPGAgent


class ContinuousControlEnv:
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




class ContinuousControl:
    def __init__(self, args, env_type, mode):
        self.args = args
        self.env_type = env_type
        self.agent_id = 0
        self.env = ContinuousControlEnv(env_type=env_type, mode=mode)
        
        if mode == 'train':
            self.agent = DDPGAgent(self.args, self.env_type, mode='train', agent_id=self.agent_id)
        elif mode == 'test':
            self.agent = DDPGAgent(self.args, self.env_type, mode='test', agent_id=self.agent_id)
        else:
            raise ValueError('Only "train", "test" mode excepted')
        
    def train(self):
        all_scores = []
        scores_window = deque(maxlen=100)
        running_timestep = 1
        
        for i_episode in range(1, self.args.NUM_EPISODES + 1):
            # agent.reset()
            states = self.env.reset()
            scores = np.zeros(self.args.NUM_AGENTS)  # NUM_AGENTS
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

            ########################
            tag = 'common/avg_score'
            value_dict = {'avg_score_100_episode': np.mean(scores_window)}
            step = i_episode

            self.agent.log(tag, value_dict, step)
            
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

    def test(self, trials=5, steps=200):
        self.agent.load_weights()
        for i in range(trials):
            total_reward = np.zeros(self.args.NUM_AGENTS)
            print('Starting Testing ...')
            state = self.env.reset()
            for j in range(steps):
                action = self.agent.act(state, action_value_range=(-1, 1), running_timestep=j)
                next_states, rewards, dones, _ = self.env.step(action)
                total_reward += rewards
        
                if rewards != 0:
                    print("Current Reward:", np.max(rewards), "Total Reward:", np.max(total_reward))
                state = next_states
                if any(dones):
                    print('Done.')
                    break



if __name__ == "__main__":
    mode = 'test'
    if mode == 'train':
        obj_ = ContinuousControl(args=TrainConfig, env_type='multi', mode='train')
        obj_.train()
    else:
        pass
        # obj_ = ContinuousControl(args=TestConfig, env_type='multi', mode='test')
        # obj_.test()

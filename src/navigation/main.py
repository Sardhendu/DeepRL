from collections import deque

import numpy as np
import torch
from unityagents import UnityEnvironment

from src.navigation.agent import DDQNAgent


class CollectBananaENV:
    def __init__(self, env_type='vector', mode='train'):
        """
        This is a wrapper on top of the brain environment that provides useful function to render the environment
        call very similar to like calling the open AI gym environement.

        Wrapper Code referred from : https://github.com/yingweiy/drlnd_project1_navigation

        :param env_type:
        """
        self.env_type = env_type
        if env_type == 'vector':
            self.base_env = UnityEnvironment('Banana.app')
        elif env_type == 'visual':
            self.base_env = UnityEnvironment('VisualBanana.app')
        else:
            raise ValueError('Env Name not understood ....')
        # get the default brain
        self.brain_name = self.base_env.brain_names[0]
        self.brain = self.base_env.brains[self.brain_name]
        self.action_size = self.brain.vector_action_space_size
        
        if mode == 'train':
            self.train = True
        else:
            self.train = False
        
        self.frame1 = None
        self.frame2 = None
        self.frame3 = None
        self.reset()
        
        if env_type == 'vector':
            self.state_size = len(self.state)
        elif env_type == 'visual':
            self.state_size = self.state.shape
        else:
            raise ValueError('Environment type not understood ....')
        
        print(self.state_size)
    
    def get_state(self):
        if self.env_type == 'visual':
            
            # The DQN paper says to stack 4 frames while running the image through the neural network
            # state size is 1,84,84,3
            # Rearrange from NHWC to NCHW (Pytorch uses 3d covolution in NCHW format, cross corellation across channels)
            frame = np.transpose(self.env_info.visual_observations[0], (0, 3, 1, 2))[:, :, :, :]
            frame_size = frame.shape  # 1,3,84,84
            # print(frame_size)
            self.state = np.zeros((1, frame_size[1], 4, frame_size[2], frame_size[3]))
            self.state[0, :, 0, :, :] = frame
            
            if self.frame1 is not None:
                self.state[0, :, 1, :, :] = self.frame1
            if self.frame2 is not None:
                self.state[0, :, 2, :, :] = self.frame2
            if self.frame3 is not None:
                self.state[0, :, 3, :, :] = self.frame3
            
            # Keep the last 3 frames in the memory to be accessed or stacked with the new input frame to supply as
            # input to the convolution network
            self.frame3 = self.frame2
            self.frame2 = self.frame1
            self.frame1 = frame
            
            # self.state = np.squeeze(self.state)  # We squeeze it becasue the code implemented in buffer will
            # unsqueeze the array
        elif self.env_type == 'vector':
            self.state = self.env_info.vector_observations[0]
        
        else:
            raise ValueError('Environment name %s not understood.' % str(self.env_type))
    
    def reset(self):
        self.env_info = self.base_env.reset(train_mode=self.train)[self.brain_name]
        self.get_state()
        return self.state
    
    def step(self, action):
        """
        This function returns the value in the format of Open AI gym
        :param action:
        :return:
        """
        self.env_info = self.base_env.step(action)[self.brain_name]  # send the action to the environment
        self.get_state()
        reward = self.env_info.rewards[0]
        done = self.env_info.local_done[0]
        return self.state, reward, done, None
    
    def close(self):
        self.base_env.close()


class CollectBanana:
    def __init__(self, args, env_type, mode, buffer_type='ER'):
        """
        :param args:            Config class
        :param env:             environment (Unity)
        :param env_type:        (str) Vector or Visual
        :param mode:            (str) train or test
        :param buffer_type:     (str) ER (experience replay buffer), PER (Priority Experience Replay buffer)

        """
        self.agent_id = 0
        if buffer_type == 'ER':
            self.agent = DDQNAgent(args, env_type, mode=mode)
        # elif buffer_type == 'PER':
        #     self.agent = DDQNAgentPER(args, env_type, seed=0)
        
        self.env = CollectBananaENV(env_type=env_type, mode=mode)
        self.args = args
        self.score_window_size = 100
    
    def train(self, target_score=13.0):
        """
        :param target_score:    (float) target score to solve the environment
        :return:
        """
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=self.score_window_size)  # last score_window_size scores
        running_timestep = 1
        for i_episode in range(1, self.args.NUM_EPISODES + 1):
            state = self.env.reset()
            score = 0
            for t in range(self.args.NUM_TIMESTEPS):
                action = self.agent.act(state, running_timestep)
                next_state, reward, done, _ = self.env.step(action)
                # print('[Train]: ', next_state.shape, reward, done)
                self.agent.step(state, action, reward, next_state, done, running_timestep)
                state = next_state
                score += reward
                running_timestep += 1
                
                if done:
                    break
            
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            avg_score = np.mean(scores_window)
            
            ########################
            tag = 'common/avg_score'
            value_dict = {'avg_score_100_episode': avg_score}
            step = i_episode
            
            self.agent.log(tag, value_dict, step)
            
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                self.agent.checkpoint(i_episode)
            
            if np.mean(scores_window) >= target_score:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                        i_episode, np.mean(scores_window))
                )
                self.agent.checkpoint(i_episode)
                break
        
        # torch.save(self.agent.local_network.state_dict(), self.saved_network)
        self.env.close()
        
        return scores
    
    def test(self, trials=3, steps=200):
        self.agent.load_weights()
        for i in range(trials):
            total_rewards = 0
            print('Starting Testing ...')
            state = self.env.reset()
            for j in range(steps):
                action = self.agent.act(state, running_timestep=j)
                # self.env.render()
                next_states, rewards, dones, _ = self.env.step(action)
                total_rewards += rewards
                if rewards != 0:
                    print("Current Reward:", rewards, "Total Reward:", total_rewards)
                state = next_states
                if dones:
                    print('Done.')
                    break
        self.env.close()




if __name__ =="__main__":
    mode = 'test'
    env_type = 'vector'
    if mode == 'train':
        if env_type == 'vector':
            from src.navigation.config import TrainVectorConfig
            args = TrainVectorConfig
            obj_cb = CollectBanana(args, env_type='vector', mode='train', buffer_type='ER')
            obj_cb.train()
    elif mode == 'test':
        if env_type == 'vector':
            from src.navigation.config import TestVectorConfig
            args = TestVectorConfig
            obj_cb = CollectBanana(args, env_type='vector', mode='test', buffer_type='ER')
            obj_cb.test()
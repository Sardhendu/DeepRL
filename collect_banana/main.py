from collections import deque

import numpy as np
import torch
from unityagents import UnityEnvironment

from DeepRL.collect_banana.agent import DDQNAgent


class CollectBanana:
    def __init__(self, env_type='vector'):
        self.env_type = env_type
        if env_type == 'vector':
            self.base_env = UnityEnvironment(
                    "/Users/sam/All-Program/App/deep-reinforcement-learning/DeepRL/collect_banana/Banana.app")
        elif env_type == 'visual':
            self.base_env = UnityEnvironment(
                    "/Users/sam/All-Program/App/deep-reinforcement-learning/DeepRL/collect_banana/VisualBanana.app")
        else:
            raise ValueError('Env Name not understood ....')
        # get the default brain
        self.brain_name = self.base_env.brain_names[0]
        self.brain = self.base_env.brains[self.brain_name]
        self.action_size = self.brain.vector_action_space_size
        self.train_mode = True
        self.last_frame = None
        self.last2_frame = None
        self.last3_frame = None
        self.reset()

        if env_type == 'visual':
            self.state_size = self.state.shape
        else:
            self.state_size = len(self.state)
    
    def get_state(self):
        if self.env_type == 'visual':
            # frame = np.transpose(self.env_info.visual_observations[0], (0, 3, 1, 2))[:, :, :, :]
            # The DQN paper says to stack 4 frames while running the image through the neural network
            # state size is 1,84,84,3
            # Rearrange from NHWC to NCHW (Pytorch uses 3d covolution in NCHW format, cross corellation across channels)
            frame = np.transpose(self.env_info.visual_observations[0], (0, 3, 1, 2))[:, :, :, :]  # cut the image partially
            frame_size = frame.shape  # 1,3,84,84
            # print(frame_size)
            # NCDHW
            nframes = 4
            self.state = np.zeros((1, frame_size[1], nframes, frame_size[2], frame_size[3]))
            self.state[0, :, 0, :, :] = frame
            
            if not (self.last_frame is None):
                self.state[0, :, 1, :, :] = self.last_frame
            if not (self.last2_frame is None):
                self.state[0, :, 2, :, :] = self.last2_frame
            if not (self.last3_frame is None):
                self.state[0, :, 3, :, :] = self.last3_frame
                
            self.last3_frame = self.last2_frame
            self.last2_frame = self.last_frame
            self.last_frame = frame
            
            # self.state = np.squeeze(self.state)  # We squeeze it becasue the code implemented in buffer will
            # unsqueeze the array
        elif self.env_type == 'vector':
            self.state = self.env_info.vector_observations[0]
            
        else:
            raise ValueError('Environment name is not correct.')
        
    def reset(self):
        self.env_info = self.base_env.reset(train_mode=self.train_mode)[self.brain_name]
        self.get_state()
        return self.state
    
    def render(self):
        pass
    
    def step(self, action):
        self.env_info = self.base_env.step(action)[self.brain_name]  # send the action to the environment
        self.get_state()
        reward = self.env_info.rewards[0]  # get the reward
        done = self.env_info.local_done[0]  # see if episode has finished
        return self.state, reward, done, None  # info is none
    
    def close(self):
        self.base_env.close()


class DDQN:
    def __init__(self, args, env, env_type='vector'):
        self.agent = DDQNAgent(args, env_type, seed=0)
        self.env = env
        self.args = args
        self.score_window_size = 100
    
    def train(self, target_score=13.0, verbose=1):
        """Deep Q-Learning.
            Params
            ======
                n_episodes (int): maximum number of training episodes
                max_t (int): maximum number of timesteps per episode
                eps_start (float): starting value of epsilon, for epsilon-greedy action selection
                eps_end (float): minimum value of epsilon
                eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=self.score_window_size)  # last score_window_size scores
        eps = self.args.EPSILON  # initialize epsilon
        
        running_time_step = 0
        for i_episode in range(1, self.args.NUM_EPISODES + 1):
            state = self.env.reset()
            score = 0
            for t in range(self.args.NUM_TIMESTEPS):
                action = self.agent.act(state, eps)
                next_state, reward, done, _ = self.env.step(action)
                # print('[Train]: ', next_state.shape, reward, done)
                self.agent.step(state, action, reward, next_state, done, i_episode, running_time_step)
                state = next_state
                score += reward
                running_time_step += 1
                
                if done:
                    break
            
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            eps = max(self.args.EPSILON_MIN, self.args.EPSILON_DECAY * eps)  # decrease epsilon
            avg_score = np.mean(scores_window)
            
            if avg_score >= target_score and i_episode > 100:
                if verbose:
                    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                                 np.mean(
                                                                                                         scores_window)))
                torch.save(self.agent.local_network.state_dict(), self.args.CHECKPOINT_PATH)
                break
            
            if verbose:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
                if i_episode % 100 == 0:
                    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        
        # torch.save(self.agent.local_network.state_dict(), self.saved_network)
        
        return scores

    def test(self, trials=3, steps=200):
        self.agent.local_network.load_state_dict(torch.load(self.args.CHECKPOINT_PATH))
    
        for i in range(trials):
            total_reward = 0
            print('Starting Testing ...')
            state = self.env.reset()
            for j in range(steps):
                action = self.agent.act(state)
                self.env.render()
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if reward != 0:
                    print("Current Reward:", reward, "Total Reward:", total_reward)
                if done:
                    print('Done.')
                    break
        self.env.close()



# device = 'cpu'
# from DeepRL.src.collect_banana.model import VisualQNEtwork
# env_type = 'visual'
# env = CollectBanana(env_type)
#
# print(env.state.shape)
# states = torch.from_numpy(env.state).float().to(device)
# my_net = VisualQNEtwork(state_size=env.state_size, action_size=4, seed=0, network_name='net1')
# my_net.forward(states)





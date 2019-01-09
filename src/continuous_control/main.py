import numpy as np
from collections import deque
from unityagents import UnityEnvironment
from src.continuous_control.agent import DDPGAgent

"""
Notes:
1. Having both Actor and Critic learning rate same (0.0001) make the leanring very-very slow. FOr 2000 timesteps the action for all the agents were the same. Which is bad. (NEVER HAVE LEARNING RATE SAME FOR ACTOR AND CRITIC)


"""

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ContinuousControl:
    def __init__(self, env_type='single', mode='train'):
        """
        
        :param env_type:    "single" and "multi"
        :param mode:        "train" or "test"
        """
        self.env_type = env_type
        if env_type == 'single':
            self.base_env = UnityEnvironment(file_name='Reacher_single.app')
        elif env_type == 'multi':
            self.base_env = UnityEnvironment(file_name='Reacher_multi.app')
        else:
            raise ValueError('Environment type not understood ....')

        self.brain_name = self.base_env.brain_names[0]
        self.brain = self.base_env.brains[self.brain_name]
        
        print('Brain Name: ', self.brain_name)
        
        if mode == 'train':
            self.train = True
        else:
            self.train = False
            
        self.reset()

        if env_type == 'single':
            self.state_size = len(self.state)
            self.action_size = self.brain.vector_action_space_size
        elif env_type == 'multi':
            self.state_size = self.state.shape[1]
            self.action_size = self.brain.vector_action_space_size
        else:
            raise ValueError('Environment type not understood ....')

        print('Number of dimensions in state space: ', self.state_size)
        print('Number of action space: ', self.action_size)
        print('Number of agents: ', self.num_agents)
        
    def reset(self):
        self.env_info = self.base_env.reset(train_mode=self.train)[self.brain_name]
        self.num_agents = len(self.env_info.agents)
        self.get_state()
        return self.state

    def get_state(self):
        if self.env_type == 'single':
            self.state = self.env_info.vector_observations
        elif self.env_type == 'multi':
            self.state = self.env_info.vector_observations
        else:
            raise ValueError('Environment type not understood ....')
        
    def step(self, action):
        # print(self.brain_name)
        # print(action)
        self.env_info = self.base_env.step(action)[self.brain_name]  # send the action to the environment
        self.get_state()
        reward = self.env_info.rewards
        done = self.env_info.local_done
        return self.state, reward, done, None
        
    def close(self):
        self.base_env.close()
        
        
class DDPG:
    def __init__(self, args, env, env_type='vector'):
        self.env = env
        self.args = args
        self.score_window_size = 100
        
        self.agent = DDPGAgent(args, env_type)
        
    def train(self, target_score=30, verbose=1):
        all_scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=self.score_window_size)  # last score_window_size scores

        # eps = self.args.EPSILON  # initialize epsilon
        running_time_step = 1
        for i_episode in range(1, self.args.NUM_EPISODES + 1):
            states = self.env.reset()
            # print(states.shape)
            self.agent.reset()
            scores = np.zeros(self.env.num_agents)
            for t in range(self.args.NUM_TIMESTEPS):
                # print(t)
                actions = self.agent.act(states)#np.random.randn(self.env.num_agents,
                # print(actions)
                # self.env.action_size)#self.agent.act(
                # state, eps)
                # print('23424234234234234 ', actions.shape)
                next_states, rewards, dones, _ = self.env.step(actions)
                # print(next_states.shape, rewards, dones)
                self.agent.step(states, actions, rewards, next_states, dones, i_episode, running_time_step)
                states = next_states
                scores += rewards
                running_time_step += 1

                # if dones:
                #     break

            avg_score = np.mean(scores)
            scores_window.append(avg_score)
            all_scores.append(avg_score)
            
            self.agent.stats_dict['score'].append(avg_score)
            self.agent.save_stats()

            if (i_episode % 100) == 0:
                self.agent.save_checkpoints(i_episode)
            

            if avg_score >= target_score and i_episode > 600:
                if verbose:
                    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                            i_episode, np.mean(scores_window))
                    )
                self.agent.save_stats()
                break

            if verbose:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
                if i_episode % 100 == 0:
                    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        
        self.env.close()

from src.exploration import OUNoise
from src.continuous_control.model import Actor, Critic
from src.buffer import MemoryER

class Config:
    import os
    # ENVIRONMEMT PARAMETER
    STATE_SIZE = 33
    ACTION_SIZE = 4
    NUM_EPISODES = 2000
    NUM_TIMESTEPS = 1000

    # MODEL PARAMETERS
    SEED = 0
    BUFFER_SIZE = int(1e05)
    BATCH_SIZE = 512
    
    # Exploration parameter
    NOISE = True
    EPSILON_GREEDY = False
    EPSILON = 1
    EPSILON_DECAY = 0.995  # Epsilon decay for epsilon greedy policy
    EPSILON_MIN = 0.01  # Minimum epsilon to reach
    
    if (NOISE and EPSILON_GREEDY) or (not NOISE and not EPSILON_GREEDY):
        raise ValueError('Only one exploration policy either NOISE or EPSILON_GREEDY si to be chosen ..')

    # LEARNING PARAMETERS
    ACTOR_LEARNING_RATE = 0.0001
    CRITIC_LEARNING_RATE = 0.0005
    GAMMA = 0.99           # Discounts
    LEARNING_FREQUENCY = 4

    # WEIGHTS UPDATE PARAMENTER
    SOFT_UPDATE = True
    TAU = 0.001  # Soft update parameter for target_network
    SOFT_UPDATE_FREQUENCY = 4
    DECAY_TAU = False
    TAU_DECAY_RATE = 0.003
    TAU_MIN = 0.05
    
    HARD_UPDATE = False
    HARD_UPDATE_FREQUENCY = 1000
    
    if (SOFT_UPDATE and HARD_UPDATE) or (not SOFT_UPDATE and not HARD_UPDATE):
        raise ValueError('Only one of Hard Update and Soft Update is to be chosen ..')
    
    if SOFT_UPDATE_FREQUENCY < LEARNING_FREQUENCY:
        raise ValueError('Soft update frequency can not be smaller than the learning frequency')

    # USE PATH
    MODEL_NAME = 'model_1'
    model_dir = os.path.join(os.path.abspath(os.path.join(__file__, "../../..")), 'models')
    base_dir = os.path.join(model_dir, 'continuous_control', '%s'%(MODEL_NAME))
    if not os.path.exists(base_dir):
        print('creating .... ', base_dir)
        os.makedirs(base_dir)
    #
    STATS_JSON_PATH = os.path.join(base_dir, 'stats.json')
    CHECKPOINT_DIR = base_dir
    
    
    # Lambda Functions:
    EXPLORATION_POLICY_FN = lambda: OUNoise(size=Config.ACTION_SIZE, seed=2)
    ACTOR_NETWORK_FN = lambda: Actor(Config.STATE_SIZE, Config.ACTION_SIZE, seed=2, fc1_units=512, fc2_units=256).to(
            device)
    ACTOR_OPTIMIZER_FN = lambda params: torch.optim.Adam(params, lr=Config.ACTOR_LEARNING_RATE)
    
    CRITIC_NETWORK_FN = lambda: Critic(Config.STATE_SIZE, Config.ACTION_SIZE, seed=2, fc1_units=512, fc2_units=256).to(
            device)
    CRITIC_OPTIMIZER_FN = lambda params: torch.optim.Adam(params, lr=Config.CRITIC_LEARNING_RATE)
    
    MEMORY_FN = lambda: MemoryER(Config.BUFFER_SIZE, Config.BATCH_SIZE, seed=2, action_dtype='float')
    
# Config()
# from src.continuous_control.model import Actor
# STATE_SIZE = 2
# ACTION_SIZE = 2
# ACTOR_NETWORK = lambda: Actor(STATE_SIZE, ACTION_SIZE, seed=2, fc1_units=512, fc2_units=256).to(device)
env = ContinuousControl(env_type='multi', mode='train')
dqn = DDPG(Config, env).train()
        
        
    


# Brain Name:  ReacherBrain
# Number of dimensions in state space:  33
# Number of action space:  4
# Number of agents:  20
# [INIT] Initializing Replay Buffer .... .... ....
# Episode 100	Average Score: 0.95
# Episode 200	Average Score: 3.78
# Episode 300	Average Score: 5.73
# Episode 400	Average Score: 3.94
# Episode 500	Average Score: 7.56
# Episode 600	Average Score: 11.82
#
# Environment solved in 601 episodes!	Average Score: 11.89
#
# Process finished with exit code 0

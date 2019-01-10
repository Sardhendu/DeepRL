import torch
from src.exploration import OUNoise
from src.continuous_control.model import Actor, Critic
from src.buffer import MemoryER

import numpy as np
from collections import deque
from unityagents import UnityEnvironment

from src.continuous_control.agent import DDPGAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    GAMMA = 0.99  # Discounts
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
    MODEL_NAME = 'model_3'
    model_dir = os.path.join(os.path.abspath(os.path.join(__file__, "../../..")), 'models')
    base_dir = os.path.join(model_dir, 'continuous_control', '%s' % (MODEL_NAME))
    if not os.path.exists(base_dir):
        print('creating .... ', base_dir)
        os.makedirs(base_dir)
    #
    STATS_JSON_PATH = os.path.join(base_dir, 'stats.json')
    CHECKPOINT_DIR = base_dir
    
    # Lambda Functions:
    EXPLORATION_POLICY_FN = lambda: OUNoise(size=Config.ACTION_SIZE, seed=2)
    ACTOR_NETWORK_FN = lambda: Actor(Config.ACTION_SIZE, Config.STATE_SIZE, (512, 256), seed=2).to(
            device)  # lambda: Actor(Config.STATE_SIZE, Config.ACTION_SIZE, seed=2, fc1_units=512, fc2_units=256).to(
    # device)
    ACTOR_OPTIMIZER_FN = lambda params: torch.optim.Adam(params, lr=Config.ACTOR_LEARNING_RATE)
    
    CRITIC_NETWORK_FN = lambda: Critic(Config.ACTION_SIZE, Config.STATE_SIZE, (512, 256), seed=2).to(
            device)  # lambda: Critic(Config.STATE_SIZE, Config.ACTION_SIZE, seed=2, fc1_units=512, fc2_units=256).to(
    # device)
    CRITIC_OPTIMIZER_FN = lambda params: torch.optim.Adam(params, lr=Config.CRITIC_LEARNING_RATE)
    
    MEMORY_FN = lambda: MemoryER(Config.BUFFER_SIZE, Config.BATCH_SIZE, seed=2, action_dtype='float')



class ContinuousControl:
    def __init__(self, env_type):
        if env_type == 'single':
            self.base_env = UnityEnvironment(file_name='Reacher_single.app')
        elif env_type == 'multi':
            self.base_env = UnityEnvironment(file_name='Reacher_multi.app')
        else:
            raise ValueError('Environment type not understood ....')
    
        self.brain_name = self.base_env.brain_names[0]
        self.brain = self.base_env.brains[self.brain_name]
        
    def reset(self):
        self.env_info = self.base_env.reset(train_mode=True)[self.brain_name]
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


def ddpg(env, agent, n_episodes=5000, max_t=2000):
    all_scores = []
    scores_window = deque(maxlen=100)
    NUM_AGENTS = 20
    running_time_step = 1
    for i_episode in range(1, n_episodes + 1):
        
        agent.reset()
        states = env.reset()
        scores = np.zeros(NUM_AGENTS)
        # print('statesstatesstatesstates :', states)
        for pp in range(max_t):
            # print (pp)
            actions = agent.act(states)
            # print('actionsactions: ', actions)
            # env_info = env.step(actions)[brain_name]
            # rewards = env_info.rewards
            # next_states = env_info.vector_observations
            # dones = env_info.local_done
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
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
    
    return all_scores



ddpg(
        env=ContinuousControl(env_type='multi'),
        agent = DDPGAgent(Config, None),
        n_episodes=5000, max_t=2000
)
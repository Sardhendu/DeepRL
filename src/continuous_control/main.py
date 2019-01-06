import numpy as np
from collections import deque
from unityagents import UnityEnvironment
from src.continuous_control.agent import DDPGAgent


class ContinuousControl:
    def __init__(self, env_type='single', mode='train'):
        """
        
        :param env_type:    "single" and "multi"
        :param mode:        "train" or "test"
        """
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
            self.action_size = self.brain.vector_action_space_size
        elif env_type == 'double':
            self.state_size = self.state.shape
            self.action_size = self.brain.vector_action_space_size
        else:
            raise ValueError('Environment type not understood ....')

        print('Number of dimensions in state space: ', self.state_size)
        print('NUmebr of action space: ', self.action_size)
        print('Number of agents: ', self.num_agents)
        
    def reset(self):
        self.env_info = self.base_env.reset(train_mode=self.train)[self.brain_name]
        self.num_agents = len(self.env_info.agents)
        self.get_state()
        return self.state

    def get_state(self):
        if self.env_type == 'single':
            self.state = self.env_info.vector_observations[0]
        elif self.env_type == 'double':
            pass
        else:
            raise ValueError('Environment type not understood ....')
        
    def step(self, action):
        self.env_info = self.base_env.step(action)[self.brain_name]  # send the action to the environment
        self.get_state()
        reward = self.env_info.rewards[0]
        done = self.env_info.local_done[0]
        return self.state, reward, done, None
        
    def close(self):
        self.base_env.close()
        
        
class DDPG:
    def __init__(self, args, env, env_type='vector'):
        self.env = env
        self.args = args
        self.score_window_size = 100
        
        self.agent = DDPGAgent(args, env_type, seed=0)
        
    def train(self, verbose=1):
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=self.score_window_size)  # last score_window_size scores

        # eps = self.args.EPSILON  # initialize epsilon
        running_time_step = 0
        for i_episode in range(1, self.args.NUM_EPISODES + 1):
            state = self.env.reset()
            self.agent.reset()
            score = 0
            for t in range(self.args.NUM_TIMESTEPS):
                action = self.agent.act(state)#np.random.randn(self.env.num_agents, self.env.action_size)#self.agent.act(
                # state, eps)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.step(state, action, reward, next_state, done, i_episode, running_time_step)
                state = next_state
                score += reward
                running_time_step += 1

                if done:
                    break

                if running_time_step == 300:
                    break
            scores_window.append(score)
            scores.append(score)
            avg_score = np.mean(scores_window)
            self.agent.save_stats()

            if avg_score >= 11 and i_episode > 600:
                if verbose:
                    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                                 np.mean(
                                                                                                         scores_window)))
                self.agent.save_stats()
                break

            if verbose:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
                if i_episode % 100 == 0:
                    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        
                    # torch.save(self.agent.local_network.state_dict(), self.saved_network)
        self.env.close()


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
    BATCH_SIZE = 64
    
    # Exploration parameter
    NOISE = True
    EPSILON_GREEDY = False
    EPSILON = 1
    EPSILON_DECAY = 0.995  # Epsilon decay for epsilon greedy policy
    EPSILON_MIN = 0.01  # Minimum epsilon to reach
    
    if (NOISE and EPSILON_GREEDY) or (not NOISE and not EPSILON_GREEDY):
        raise ValueError('Only one exploration policy either NOISE or EPSILON_GREEDY si to be chosen ..')

    # LEARNING PARAMETERS
    LEARNING_RATE = 0.0001
    GAMMA = 0.995           # Discounts
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
    
# Config()
    
env = ContinuousControl(mode='train')
dqn = DDPG(Config, env).train()
        
        
    
    
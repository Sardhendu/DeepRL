
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
        
    def train(self):
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=self.score_window_size)  # last score_window_size scores

        # eps = self.args.EPSILON  # initialize epsilon
        running_time_step = 0
        for i_episode in range(1, self.args.NUM_EPISODES + 1):
            state = self.env.reset()
            score = 0
            for t in range(self.args.NUM_TIMESTEPS):
                action = self.agent.act(state)#np.random.randn(self.env.num_agents, self.env.action_size)#self.agent.act(
                # state, eps)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.step(state, action, reward, next_state, done, i_episode, running_time_step)
                # print(action)
                # print('state_next: ', state_next)
                running_time_step += 1
                
                if running_time_step == 300:
                    break
                
                
    
class Config:
    NUM_EPISODES = 20
    NUM_TIMESTEPS = 2000
    
    EPSILON = 1
    BUFFER_SIZE = 100000
    BATCH_SIZE = 64

    STATE_SIZE = 33
    ACTION_SIZE = 4
    
    LEARNING_RATE = 0.0001
    UPDATE_AFTER_STEP = 4
    GAMMA = 0.995
    
    HARD_UPDATE = False
    HARD_UPDATE_FREQUENCY = 1000
    
env = ContinuousControl(mode='train')
dqn = DDPG(Config, env).train()
        
        
    
    
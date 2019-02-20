import os
import numpy as np
import torch

from src import utils
from src.logger import Logger
from src.navigation import model
from src.exploration import EpsilonGreedy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TrainVectorConfig:
    # Environment Parameters
    SEED = 0
    STATE_SIZE = 37
    ACTION_SIZE = 4
    NUM_AGENTS = 1
    BUFFER_SIZE = 10000  # int(1e05)
    BATCH_SIZE = 64
    NUM_EPISODES = 2000
    NUM_TIMESTEPS = 1000
    
    # Agent Params
    TAU = 0.001
    WEIGHT_DECAY = 0.0
    IS_HARD_UPDATE = False
    IS_SOFT_UPDATE = True
    SOFT_UPDATE_FREQUENCY = 4
    HARD_UPDATE_FREQUENCY = 2000
    
    # Model parameters
    AGENT_LEARNING_RATE = 0.0005
    DATA_TO_BUFFER_BEFORE_LEARNING = 64
    GAMMA = 0.99
    LEARNING_FREQUENCY = 4
    
    # Exploration parameter (Since this is a discrete task we have to use a discrete policy exploration 1.e epsilon
    # greedy)
    EPSILON_GREEDY = lambda: EpsilonGreedy(
            epsilon_init=1, epsilon_min=0.01, decay_value=0.99, decay_after_step=300, seed=0
    )  # Note decay_after_steps should approximately be equal to the num_timesteps in one episode
    
    # NETWORK PARAMETERS
    Q_LEARNING_TYPE = 'dqn'  # available values = {'dqn', 'dbl_dqn'}
    LOCAL_NETWORK = lambda: model.QNetwork(
            TrainVectorConfig.STATE_SIZE, TrainVectorConfig.ACTION_SIZE, TrainVectorConfig.SEED, network_name='net2'
    ).to(device)
    TARGET_NETWORK = lambda: model.QNetwork(
            TrainVectorConfig.STATE_SIZE, TrainVectorConfig.ACTION_SIZE, TrainVectorConfig.SEED, network_name='net2'
    ).to(device)
    OPTIMIZER_FN = lambda params: torch.optim.Adam(params, lr=TrainVectorConfig.AGENT_LEARNING_RATE)

    # LOG PATHS
    MODEL_NAME = 'model_2'
    pth = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    model_dir = pth + '/models'
    base_dir = os.path.join(model_dir, 'navigation', '%s' % (MODEL_NAME))
    
    if not os.path.exists(base_dir):
        print('creating .... ', base_dir)
        os.makedirs(base_dir)
    
    CHECKPOINT_DIR = base_dir
    SUMMARY_LOGGER_PATH = os.path.join(base_dir, 'summary')
    
    if not os.path.exists(SUMMARY_LOGGER_PATH):
        print('creating .... ', SUMMARY_LOGGER_PATH)
        os.makedirs(SUMMARY_LOGGER_PATH)
    
    SUMMARY_LOGGER = Logger(SUMMARY_LOGGER_PATH)
    
    # Exception checking
    if (IS_SOFT_UPDATE and IS_HARD_UPDATE) or (not IS_SOFT_UPDATE and not IS_HARD_UPDATE):
        raise ValueError('Only one of Hard Update and Soft Update is to be chosen ..')
    
    if SOFT_UPDATE_FREQUENCY < LEARNING_FREQUENCY or HARD_UPDATE_FREQUENCY < LEARNING_FREQUENCY:
        raise ValueError('Soft update frequency can not be smaller than the learning frequency')
    
    # elif env_type == 'visual':
    #     self.local_network = model.VisualQNEtwork(args.STATE_SIZE, args.ACTION_SIZE, seed,
    #                                               network_name=args.NET_NAME).to(device)
    #     self.target_network = model.VisualQNEtwork(args.STATE_SIZE, args.ACTION_SIZE, seed,
    #                                                network_name=args.NET_NAME).to(device)
    # else:
    #     raise ValueError('Env type not understood')

class TestVectorConfig:
    # Environment Parameters
    SEED = 0
    STATE_SIZE = 33
    ACTION_SIZE = 4
    NUM_AGENTS = 20

    # Exploration parameter
    EPSILON_GREEDY = lambda: EpsilonGreedy(
            epsilon_init=0, epsilon_min=0, decay_value=0, decay_after_step=1, seed=0
    )

    Q_LEARNING_TYPE = 'dqn'  # available values = {'dqn', 'dbl_dqn'}
    LOCAL_NETWORK = lambda: model.QNetwork(
            TrainVectorConfig.STATE_SIZE, TrainVectorConfig.ACTION_SIZE, TrainVectorConfig.SEED, network_name='net2'
    ).to(device)

    # LOG PATHS
    MODEL_NAME = 'model_1'
    CHECKPOINT_NUMBER = '436'
    pth = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    model_dir = pth + '/models'
    base_dir = os.path.join(model_dir, 'navigation', '%s' % (MODEL_NAME))

    if not os.path.exists(base_dir):
        print('creating .... ', base_dir)
        os.makedirs(base_dir)

    CHECKPOINT_DIR = base_dir
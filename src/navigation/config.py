import os
import numpy as np
import torch

from src import utils
from src.logger import Logger
from src.navigation import model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TrainVectorConfig:
    # Environment Parameters
    SEED = 0
    STATE_SIZE = 37
    ACTION_SIZE = 4
    NUM_AGENTS = 1
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64
    
    # Agent Params
    TAU = 1e-2
    WEIGHT_DECAY = 0.0
    IS_HARD_UPDATE = False
    IS_SOFT_UPDATE = True
    SOFT_UPDATE_FREQUENCY = 2
    HARD_UPDATE_FREQUENCY = 2000
    
    # Model parameters
    AGENT_LEARNING_RATE = 1e-3
    DATA_TO_BUFFER_BEFORE_LEARNING = 256
    GAMMA = 0.99
    LEARNING_FREQUENCY = 2
    
    # Exploration parameter (Since this is a discrete task we have to use a discrete policy exploration 1.e epsilon
    # greedy)
    # TODO: Write a function for epsilon greedy policy and call it here

    # NETWORK PARAMETERS
    Q_LEARNING_TYPE = 'dqn'  # available values = {'dqn', 'dbl_dqn'}
    LOCAL_NETWORK = lambda: model.QNetwork(
            TrainVectorConfig.STATE_SIZE, TrainVectorConfig.ACTION_SIZE, TrainVectorConfig.SEED, network_name='net2'
    ).to(device)
    TARGET_NETWORK= lambda: model.QNetwork(
            TrainVectorConfig.STATE_SIZE, TrainVectorConfig.ACTION_SIZE, TrainVectorConfig.SEED,  network_name='net2'
    ).to(device)
    OPTIMIZER_FN = lambda params: torch.optim.Adam(params, lr=TrainVectorConfig.AGENT_LEARNING_RATE)
    
    # LOG PATHS
    MODEL_NAME = 'model_1'
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
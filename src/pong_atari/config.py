
import os
from src.logger import Logger

class TrainConfig:
    LEARNING_RATE = 0.0001
    DISCOUNT = 0.99
    BETA = 0.01
    BETA_DECAY = 0.995
    
    # To perform Proximal policy Optimization
    CLIP_SURROGATE = True
    EPSILON_CLIP = 0.1
    EPSILON_CLIP_DECAY = 0.999
    TRAJECTORY_INNER_LOOP_CNT = 4
    
    # Setup
    NET_NAME = 'net1'
    NUM_EPISODES = 700
    HORIZON = 320                # Number of state-action samples in a trajectory
    NUM_PARALLEL_ENV = 8         # Number of environments for parallel trajectory sampling
    SAVE_AFTER_EPISODES = 100    # Number of episodes after which stats and checkpoints are collected
    
    # LOG PATHS
    MODEL_NAME = 'model_1'
    pth = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    model_dir = pth + '/models'
    base_dir = os.path.join(model_dir, 'pong_atari', '%s' % MODEL_NAME)

    if not os.path.exists(base_dir):
        print('creating .... ', base_dir)
        os.makedirs(base_dir)

    CHECKPOINT_DIR = base_dir
    SUMMARY_LOGGER_PATH = os.path.join(base_dir, 'summary')

    if not os.path.exists(SUMMARY_LOGGER_PATH):
        print('creating .... ', SUMMARY_LOGGER_PATH)
        os.makedirs(SUMMARY_LOGGER_PATH)

    SUMMARY_LOGGER = Logger(SUMMARY_LOGGER_PATH)

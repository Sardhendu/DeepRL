
import torch
import os
from src import utils
from src.exploration import OUNoise
from src.continuous_control.model import Actor, Critic
from src.buffer import MemoryER
from src.logger import Logger


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TrainConfig:
    import os
    # ENVIRONMEMT PARAMETER
    STATE_SIZE = 33
    ACTION_SIZE = 4
    NUM_AGENTS = 20
    NUM_EPISODES = 2000
    NUM_TIMESTEPS = 1000
    
    # MODEL PARAMETERS
    SEED = 0
    BUFFER_SIZE = int(1e05)
    BATCH_SIZE = 256
    DATA_TO_BUFFER_BEFORE_LEARNING = 256
    
    # LEARNING PARAMETERS
    ACTOR_LEARNING_RATE = 0.0001
    CRITIC_LEARNING_RATE = 0.0005
    GAMMA = 0.99  # Discounts
    LEARNING_FREQUENCY = 4
    
    # WEIGHTS UPDATE PARAMENTER
    IS_SOFT_UPDATE = True
    TAU = 0.001  # Soft update parameter for target_network
    SOFT_UPDATE_FREQUENCY = 4
    DECAY_TAU = False
    TAU_DECAY_RATE = 0.003
    TAU_MIN = 0.05
    
    IS_HARD_UPDATE = False
    HARD_UPDATE_FREQUENCY = 1000
    
    # Exploration parameter
    NOISE_FN = lambda: OUNoise(size=4, seed=0)  # (ACTION_SIZE, SEED_VAL)
    NOISE_AMPLITUDE_DECAY_FN = lambda: utils.Decay(
            decay_type='multiplicative',
            alpha=1, decay_rate=0.995, min_value=0.25,
            start_decay_after_step=1000,
            decay_after_every_step=300,
            decay_to_zero_after_step=30000   # When to stop using Noise
    )
    
    
    # MODELS
    ACTOR_NETWORK_FN = lambda: Actor(
            TrainConfig.STATE_SIZE, TrainConfig.ACTION_SIZE, seed=2, fc1_units=256, fc2_units=256).to(device)
    CRITIC_NETWORK_FN = lambda: Critic(
            TrainConfig.STATE_SIZE, TrainConfig.ACTION_SIZE, seed=2, fc1_units=256, fc2_units=256).to(device)
    
    # Optimization
    ACTOR_OPTIMIZER_FN = lambda params: torch.optim.Adam(params, lr=TrainConfig.ACTOR_LEARNING_RATE)
    CRITIC_OPTIMIZER_FN = lambda params: torch.optim.Adam(params, lr=TrainConfig.CRITIC_LEARNING_RATE)
    
    # Memory
    MEMORY_FN = lambda: MemoryER(TrainConfig.BUFFER_SIZE, TrainConfig.BATCH_SIZE, seed=2, action_dtype='float')
    
    # LOG PATHS
    MODEL_NAME = 'model_6'
    pth = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    model_dir = pth + '/models'
    base_dir = os.path.join(model_dir, 'continuous_control', '%s' % MODEL_NAME)

    if not os.path.exists(base_dir):
        print('creating .... ', base_dir)
        os.makedirs(base_dir)

    CHECKPOINT_DIR = base_dir
    SUMMARY_LOGGER_PATH = os.path.join(base_dir, 'summary')

    if not os.path.exists(SUMMARY_LOGGER_PATH):
        print('creating .... ', SUMMARY_LOGGER_PATH)
        os.makedirs(SUMMARY_LOGGER_PATH)

    SUMMARY_LOGGER = Logger(SUMMARY_LOGGER_PATH)


    # Constraints (Condition Check)
    if (IS_SOFT_UPDATE and IS_HARD_UPDATE) or (not IS_SOFT_UPDATE and not IS_HARD_UPDATE):
        raise ValueError('Only one of Hard Update and Soft Update is to be chosen ..')

    if SOFT_UPDATE_FREQUENCY < LEARNING_FREQUENCY:
        raise ValueError('Soft update frequency can not be smaller than the learning frequency')



class TestConfig:
    # Environment Parameters
    SEED = 0
    STATE_SIZE = 33
    ACTION_SIZE = 4
    NUM_AGENTS = 20

    # Exploration parameter
    NOISE_FN = None
    NOISE_AMPLITUDE_DECAY_FN = None
    
    ACTOR_NETWORK_FN = lambda: Actor(
            TrainConfig.STATE_SIZE, TrainConfig.ACTION_SIZE, seed=2, fc1_units=256, fc2_units=256).to(device)

    # LOG PATHS
    MODEL_NAME = 'model_6'
    CHECKPOINT_NUMBER = '1224'
    pth = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    model_dir = pth + '/models'
    base_dir = os.path.join(model_dir, 'continuous_control', '%s' % (MODEL_NAME))

    if not os.path.exists(base_dir):
        print('creating .... ', base_dir)
        os.makedirs(base_dir)

    CHECKPOINT_DIR = base_dir
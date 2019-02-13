
import torch

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
    
    # Lambda Functions:
    EXPLORATION_POLICY_FN = lambda: OUNoise(size=Config.ACTION_SIZE, seed=2)
    ACTOR_NETWORK_FN = lambda: Actor(Config.STATE_SIZE, Config.ACTION_SIZE, seed=2, fc1_units=512, fc2_units=256).to(
            device)
    ACTOR_OPTIMIZER_FN = lambda params: torch.optim.Adam(params, lr=Config.ACTOR_LEARNING_RATE)
    
    CRITIC_NETWORK_FN = lambda: Critic(Config.STATE_SIZE, Config.ACTION_SIZE, seed=2, fc1_units=512, fc2_units=256).to(
            device)
    CRITIC_OPTIMIZER_FN = lambda params: torch.optim.Adam(params, lr=Config.CRITIC_LEARNING_RATE)
    
    MEMORY_FN = lambda: MemoryER(Config.BUFFER_SIZE, Config.BATCH_SIZE, seed=2, action_dtype='float')
    
    # USE PATH
    MODEL_NAME = 'model_1'
    model_dir =  pth + '/models'
    base_dir = os.path.join(model_dir, 'continuous_control', '%s' % (MODEL_NAME))
    if not os.path.exists(base_dir):
        print('creating .... ', base_dir)
        os.makedirs(base_dir)
    #
    STATS_JSON_PATH = os.path.join(base_dir, 'stats.json')
    CHECKPOINT_DIR = base_dir

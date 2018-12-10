



class Config:
    print ('Reading Config File ............')
    # ENVIRONMEMT PARAMETER
    STATE_SIZE = 37
    ACTION_SIZE = 4
    NUM_EPISODES = 2000
    NUM_TIMESTEPS = 1000
    
    # MODEL PARAMETERS
    # SEED = 0
    BUFFER_SIZE = int(1e05)
    BATCH_SIZE = 64
    UPDATE_AFTER_STEP = 4
    
    # LEARNING PARAMETERS
    TAU = 0.001  # Soft update parameter for target_network
    GAMMA = 0.99  # Discount value
    LEARNING_RATE = 1e-04  # Learning rate for the network
    EPSILON = 1  # Epsilon value for action selection
    EPSILON_DECAY = 0.995  # Epsilon decay for epsilon greedy policy
    EPSILON_MIN = 0.01 # Minimum epsilon to reach

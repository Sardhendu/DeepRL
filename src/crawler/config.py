import torch
from src.crawler import model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Config:
    STATE_SIZE = 129
    ACTION_SIZE = 20
    NUM_AGENTS = 12
    NUM_EPISODES = 2000
    NUM_TIMESTEPS = 1000
    
    GAMMA = 1


class TrainConfig(Config):
    ACTOR_NETWORK_FN = lambda: model.Actor(
            Config.STATE_SIZE, Config.ACTION_SIZE, seed=2, fc1_units=256, fc2_units=256
    ).to(device)
    CRITIC_NETWORK_FN = lambda: model.Critic(
            Config.STATE_SIZE, Config.ACTION_SIZE, seed=2, fc1_units=256, fc2_units=256
    ).to(device)
    
    def __init__(self):
        Config.__init__(self)

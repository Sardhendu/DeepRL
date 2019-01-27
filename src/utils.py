import numpy as np
from collections import deque

def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter

    Idea
    ======
        Instead of performing a hard update on Fixed-Q-targets after say 10000 timesteps, perform soft-update
        (move slightly) every time-step
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

def hard_update(local_model, target_model):
    """Hard update model parameters.
    θ_target = θ_local

    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to

    Idea
    ======
        After t time-step copy the weights of local network to target network
    """
    # print('[Hard Update] Performing hard update .... ')
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(local_param.data)


def exp_decay(alpha, decay_rate, iteration_num, min_value=0):
   alpha_d = alpha * np.exp(-decay_rate*iteration_num)
   return max(min_value, alpha_d)


class Scores:
    """
        Scores are to be pushed after every Episode
    """
    def __init__(self, episode_score_window, agent_id):
        self.agent_id = agent_id
        self.episode_score_window = episode_score_window
        
        self.scores_window = deque(maxlen=episode_score_window)
        self.all_scores = []
    
    def push(self, scores, episode_num, logger=None):
        avg_score = np.mean(scores)
        self.scores_window.append(avg_score)
        self.all_scores.append(avg_score)
        
        if logger is not None:
            logger.add_scalars(
                    'agent%i/mean_%i_episode_rewards' % (self.agent_id, self.episode_score_window),
                    {'avg_score': avg_score},
                    episode_num)
            
    def fetch_window(self):
        return self.scores_window
    
    def fetch_all(self):
        return self.all_scores
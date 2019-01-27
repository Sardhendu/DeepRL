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


class Decay:
    def __init__(self, alpha, decay_rate, min_value=0, start_decay_after_step=0):
        """
        :param alpha:           int/float Initial value
        :param decay_rate:      float       By how much to decay every step
        :param min_value:       float/int   No decay beyond this point
        """
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.min_value = min_value
        self.start_decay_after_step = start_decay_after_step
        
        self.iteration_num = 0
    
    def exponential_decay(self):
        """
        NOTE: It is best to use this decay after every int(total_episodes/30) episodes.
        Exponential decay are very aggressive they decay pretty fast.
        Exponential decay best works for learning rate because we dont decay learning rate after every batch is
        but we decay it after every step and every step. And we may have only run our model for 30-50 steps
        :return:
        """
        self.alpha = self.alpha * np.exp(-self.decay_rate * self.iteration_num)
        self.iteration_num += 1
        return max(self.min_value, self.alpha)
    
    def multiplicative_decay(self):
        if self.iteration_num > self.start_decay_after_step:
            self.alpha = self.alpha * self.decay_rate
        self.iteration_num += 1
        return max(0, self.alpha)


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


def debug():
    obj_ = Decay(0.0001, 0.5, 0)
    
    decay_value = []
    for i in range(0, 10000):
        decay_value.append(obj_.exponential_decay())
    
    print(decay_value)
    
    import matplotlib.pyplot as plt
    plt.plot(decay_value)
    plt.show()
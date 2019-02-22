
import os
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim

from src.pong_atari.model import Model
from src.pong_atari.utils import collect_trajectories
from src import commons as cmn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, args, env):
        self.learning_rate = args.LEARNING_RATE
        self.discount = args.DISCOUNT
        self.beta = args.BETA
        self.beta_decay = args.BETA_DECAY
        
        # Whether to perform Proximal policy Optimization
        self.clip_surrogate = args.CLIP_SURROGATE
        # The Epsilon clip value the ratio of new_action_prob/old_action_prob is clamped at 1+epsilon_clip
        self.epsilon_clip = args.EPSILON_CLIP
        self.epsilon_clip_decay = args.EPSILON_CLIP_DECAY
        self.trajectory_inner_loop = args.TRAJECTORY_INNER_LOOP_CNT

        self.save_after_episodes = args.SAVE_AFTER_EPISODES
        self.save_model_path = args.CHECKPOINT_PATH
        self.save_stats_path = args.STATS_JSON_PATH
        
        if os.path.exists(self.save_stats_path):
            self.stats_dict = cmn.read_json(self.save_stats_path)
        else:
            self.stats_dict = defaultdict(list)
            
        self.env = env
        self.policy_nn = Model(args.NET_NAME).to(device)
        self.optimizer = optim.Adam(self.policy_nn.parameters(), lr=self.learning_rate)

        # Actions
        self.RIGHTFIRE = 4
        self.LEFTFIRE = 5

    def get_actions_probs(self, policy, states):
        """
        
        :param states: [trajectory_horizon, num_parallel_instances, num_consecutive_frames, img_height,
                            img_width], Grayscale
        :return:    sigmoid_activateions: action_probabilities [H, n] each column represents the action taken by each parallel nodes
        
        Operations:
                states_collated = [H*n, 2, 80, 80] : [mini_batch_size, num_consecutive_frames, img_height, img_width]
        """
        # States to Probabilities
        states = torch.stack(states)  # Convert the states to torch framework
        # Since we choose "n" parallel instances rendering 100 trajectory each we have to collate them to n*100
        # minibatch_size
        policy_input = states.view(-1, *states.shape[-3:])
        sigmoid_activations = policy.forward(policy_input)
    
        # Convert the activation back to probabilities for each parallel instances
        sigmoid_activations = sigmoid_activations.view(states.shape[:-3])
        return sigmoid_activations
    
        
    def surrogate(self, policy, old_action_probs, states, actions, rewards, clip_surrogate):
    
        """
        :param nn_policy:           Model (Neural Network)
        :param old_action_probs:    action probabilities assigned while sampling trajectories
        :param states:              [tau, n, nfr, img_h, img_w ] states in trajectories
        :param actions:             [H, n] Trajectory actions
        :param rewards:             [H, n] Rewards for state-action for each time step in trajectory
        :param discount:            Discount value
        :param beta:                decay param to control exploration
        :param clip_surrogate:      bool (If you want to clip the losses within 1-epsilon and 1+epsilon to avoid bad approximation)
        :return:
        
            rewards: [H,  n] (horizon, num_parallel_instances)
            action: represents: For simplicity we use two actions: RIGHTFIRE:4, LEFTFIRE:5
            state: [100, 4, 2, 80, 80] : [batch_size, num_parallel_instances, num_consecutive_frames, img_height,
            img_width], Grayscale
    
            Notation:
            1. tau: number of trajectories
            2. H: Horizon (Num of timesteps in a trajectory)
            3. n: Parallel nodes (number of parallel runs)
            4. nfr: Number of consecutive frames
            5. H*n: 1 mini-batch
    
            Operations:
            1. Discounted Future reward ([H, n]): Required for proper Credit assignment to REINFORCE method.
            2. Discounted Future reward ([H, n]): Normalized rewards to take into account the reward distribution of the
            trajectory
            3. Input States to policy Network ([H*n, 2, 80, 80]):
            4. sigmoid Activation ([H, N]): Action probability from policy network to each input states
                If action_prob <= 0.5, action = RIGHTFIRE
                If action_prob > 0.5, action = LEFTFIRE
            5. Clipped Surrogate:
                Follows the proximal policy optimization
        """
        
        rewards = np.asarray(rewards)  # [h, n]
        
        # Compute discounted rewards:
        discounts = pow(self.discount, np.arange(len(rewards)))
        discounted_rewards = rewards * discounts[:, np.newaxis]  # [H, n] * [H, 1] = [H, n]
        # Flip vertically, do consequtive cumulative sum, reflip-vertically
        discounted_future_rewards = discounted_rewards[::-1].cumsum(axis=0)[::-1]  # [H, n]
        
        # Normalize Rewards with mean and standard deviation, equivalent to Batch normalization
        mean = np.mean(discounted_future_rewards, axis=1)
        std = np.std(discounted_future_rewards, axis=1) + 1.0e-10
        discounted_future_rewards_norm = (discounted_future_rewards - mean[:, np.newaxis]) / std[:, np.newaxis]
        
        # Convert to Pytorch Tensors:
        actions = torch.tensor(actions, dtype=torch.int8, device=device)
        rewards = torch.tensor(discounted_future_rewards_norm, dtype=torch.float, device=device)
        old_action_probs = torch.tensor(old_action_probs, dtype=torch.float, device=device)

        # Run Policy Network and fetch output probabilities
        new_action_probs = self.get_actions_probs(policy, states)  # [H, N]
        # new_action_probs = torch.tensor(new_action_probs, dtype=torch.float, device=device)
        new_action_probs = torch.where(actions == self.RIGHTFIRE, new_action_probs, 1.0 - new_action_probs)
        
        # Loss is  *sum(reward_future_norm * d_theta(log(at|st)) or = *sum(reward_future_norm * p(at|st) / p(at|st)
        # loss = rewards*torch.log(new_action_probs)
        if clip_surrogate:
            reinforce_ratio = new_action_probs / old_action_probs
            reinforce_clip_ratio = torch.clamp(reinforce_ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip)
            surrogate_loss = torch.min(reinforce_ratio * rewards, reinforce_clip_ratio * rewards)
        else:
            reinforce_ratio = new_action_probs / old_action_probs
            reinforce_clip_ratio = torch.tensor(
                    np.array([0, 0], dtype=np.float32), dtype=torch.float32, device=device
            )
            surrogate_loss = reinforce_ratio*rewards
        # print(torch.mean(old_action_probs), torch.mean(new_action_probs), torch.mean(loss))
        
        # For regularization Regularization Cross-entropy loss: -(ylog(h) + (1-y)log(1-h)):
        # It steers the probability closer to 0.5 and helps avoiding straight probability of 0 and 1
        cross_entropy_reg = -1 * (
            new_action_probs * torch.log(old_action_probs + 1.e-10) +
            (1-new_action_probs) * torch.log((1-old_action_probs) + 1.e-10)
        )
        
        # return torch.mean(loss+beta*cross_entropy_reg)
        
        return (torch.mean(surrogate_loss + self.beta*cross_entropy_reg),
                torch.mean(reinforce_ratio),
                torch.mean(reinforce_clip_ratio),
                torch.mean(surrogate_loss)
                )
    
    
    def learn(self, horizon, num_parallel_env, episode_num):
        # Collect Trajectories
        old_probs, states, actions, rewards = collect_trajectories(self.env, self.policy_nn, horizon, num_parallel_env)

        # We take negative of log loss because pytorch by default performs gradient descent and we want to
        # perform gradient ascend
        for _ in range(0, self.trajectory_inner_loop):
            loss, rforce_ratio, rforce_clip_ratio, surr_loss = self.surrogate(
                    self.policy_nn, old_probs, states, actions, rewards, self.clip_surrogate
            )
            loss = -1*loss
            self.optimizer.zero_grad()      # Set gradients to zero to avoid overlaps
            loss.backward()                 # Perform Gradient Descent
            self.optimizer.step()
            
            # Store into stats dictionary
            self.stats_dict['loss'].append(-1*float(loss))
            self.stats_dict['beta_decay'].append(float(self.beta))
            self.stats_dict['reinforce_ratio'].append(float(rforce_ratio))
            self.stats_dict['reinforce_clip_ratio'].append(float(rforce_ratio))
            self.stats_dict['surrogate_ratio'].append(float(surr_loss))
            
            
            del loss
    
        self.beta *= self.beta_decay
        self.epsilon_clip *= self.epsilon_clip_decay

        # get the average reward of the parallel environments
        total_rewards = np.sum(rewards, axis=0)
        average_reward = np.mean(total_rewards)
        self.stats_dict['rewards'].append(average_reward)
        
        if ((episode_num+1)%self.save_after_episodes) == 0:
            torch.save(self.policy_nn, os.path.join(self.save_model_path, '%s.policy'%str(episode_num+1)))
            cmn.dump_json(self.save_stats_path, self.stats_dict)
        
        return rewards
    
    
    # def play(self, inp_batch):
    #     action_prob = self.policy_nn.forward(inp_batch)
    #
    #     # If action probability is greater than 0.5 then chose
    #     action = self.RIGHTFIRE if np.random.random() < action_prob else self.LEFTFIRE
    #     return action
    



# Episode 100	Average Score: 3.06
# Episode 200	Average Score: 16.87
# Episode 300	Average Score: 19.21
# Episode 400	Average Score: 21.63
# Episode 500	Average Score: 27.63
# Episode 600	Average Score: 25.74
# Episode 700	Average Score: 26.98
# Episode 800	Average Score: 24.58
# Episode 900	Average Score: 25.36
# Episode 1000	Average Score: 21.93
# Episode 1100	Average Score: 24.70
# Episode 1200	Average Score: 26.42
# Episode 1224	Average Score: 30.10
# Environment solved in 1124 episodes!	Average Score: 30.10


import os
import copy
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

import src.commons as cmn

# TODO: Reset the noise to after every episode
# TODO: Add Batchnormalization
# TODO: Try scaling the input features
# TODO: Density plot showing estimated Q values versus observed returns sampled from test (Refer DDPG Paper)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class RLAgent:
    def __init__(self, args, env_type):
        self.env_type = env_type

        self.ACTION_SIZE = args.ACTION_SIZE
        self.BATCH_SIZE = args.BATCH_SIZE
        self.SOFT_UPDATE = args.SOFT_UPDATE
        self.SOFT_UPDATE_FREQUENCY = args.SOFT_UPDATE_FREQUENCY
        self.GAMMA = args.GAMMA
        self.HARD_UPDATE = args.HARD_UPDATE
        self.HARD_UPDATE_FREQUENCY = args.HARD_UPDATE_FREQUENCY
        self.BUFFER_SIZE = args.BUFFER_SIZE
        self.BATCH_SIZE = args.BATCH_SIZE
        self.TAU = args.TAU
        self.DECAY_TAU = args.DECAY_TAU
        self.TAU_DECAY_RATE = args.TAU_DECAY_RATE
        self.TAU_MIN = args.TAU_MIN

        self.NOISE = args.NOISE
        self.EPSILON_GREEDY = args.EPSILON_GREEDY
        self.EPSILON = args.EPSILON_GREEDY
        self.EPSILON_DECAY = args.EPSILON_DECAY
        self.EPSILON_MIN = args.EPSILON_MIN

        self.LEARNING_FREQUENCY = args.LEARNING_FREQUENCY
    
        self.STATS_JSON_PATH = args.STATS_JSON_PATH
        self.CHECKPOINT_DIR = args.CHECKPOINT_DIR
        
        
        # Create the Local Network and Target Network for the Actor
        self.actor_local = args.ACTOR_NETWORK_FN()
        self.actor_target = args.ACTOR_NETWORK_FN()
        self.actor_local_optimizer = args.ACTOR_OPTIMIZER_FN(self.actor_local.parameters())

        # Create the Local Network and Target Network for the Critic
        self.critic_local = args.CRITIC_NETWORK_FN()
        self.critic_target = args.CRITIC_NETWORK_FN()
        self.critic_local_optimizer = args.CRITIC_OPTIMIZER_FN(self.critic_local.parameters())
        
        #### Methods
        self.exploration_policy = args.EXPLORATION_POLICY_FN()
        self.memory = args.MEMORY_FN()
        

        self.stats_dict = defaultdict(list)

    def act(self, state):
        """
        
        :param state:
        :return:        A numpy array with action value distribution
        
        ### Actor Action: Actons are chosen by actor
        Select action "at" according to the current policy and the exploration noise
        at = μ(st | θμ) + Nt
            
        """
        
        # Set the state to evaluation get actions and reset network to training phase
        # print('4223423 ', state)
        pass

    def soft_update(self, local_model, target_model, tau):
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

    def hard_update(self, local_model, target_model):
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
        # print('[Hard Update] Performuing hard update .... ')
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    
    
class DDPGAgent(RLAgent):
    def __init__(self, args, env_type):
        super().__init__(args, env_type)

        
    def act(self, state, add_noise=True, action_value_range=(-1, 1)):
        """
        :param state:               Input n dimension state feature space
        :param add_noise:           bool True or False
        :param action_prob_range:   The limit range of action values
        :return:                    The action value distribution with added noise (if required)
        
        at = μ(st | θμ) + Nt
        
        Here
            --> θμ : Weights of Actor Network
            --> Nt : Are the random noise that jitters the action distribution introducing randomness to for exploration
        """
        state = torch.from_numpy(state).float().to(device)
        state.requires_grad = False
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local.forward(state).cpu().data.numpy()
        self.actor_local.train()

        # print(actions)
        # Add Random noise to the action space distribution to foster exploration
        actions += self.exploration_policy.sample()
        
        # Clip the actions to the the min and max limit of action probs
        actions = np.clip(actions, action_value_range[0], action_value_range[1])
        return actions


    def step(self, states, actions, rewards, next_states, dones, episode_num, running_time_step):
    
        # Store experience to the replay buffer
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            # print('adding: ', state.shape, action.shape, reward, next_state.shape, done)
            self.memory.add(state, action, reward, next_state, done)
    
        # When the memory is atleast full as the batch size and if the step num is a factor of UPDATE_AFTER_STEP
        # then we learn the parameters of the network
        # Update the weights of local network and soft-update the weighs of the target_network
        # self.t_step = (self.t_step + 1) % self.UPDATE_AFTER_STEP  # Run from {1->UPDATE_AFTER_STEP}
        # print('[Step] Current Step is: ', self.tstep)
        learning_flag = 0
        if (running_time_step % self.LEARNING_FREQUENCY) == 0:
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.GAMMA)

                learning_flag = 1
    
        if learning_flag == 1:
            if self.HARD_UPDATE:
                if (running_time_step % self.HARD_UPDATE_FREQUENCY) == 0:
                    self.hard_update(self.actor_local, self.actor_target)
                    self.hard_update(self.critic_local, self.critic_target)
        
            elif self.SOFT_UPDATE:
                if (running_time_step % self.SOFT_UPDATE_FREQUENCY) == 0:
                    # print('Performing soft-update at: ', running_time_step)
                
                    if self.DECAY_TAU:
                        tau = cmn.exp_decay(self.TAU, self.TAU_DECAY_RATE, episode_num, self.TAU_MIN)
                    else:
                        tau = self.TAU
                
                    self.soft_update(self.critic_local, self.critic_target, tau)
                    self.soft_update(self.actor_local, self.actor_target, tau)
            else:
                raise ValueError('Only One of HARD_UPDATE and SOFT_UPDATE is to be activated')

                
    def learn(self, experiences, gamma):
        """
        
        :param experiences:
        :param gamma:
        :return:
        
        """
        # print('Learning .... ', episode_num, running_time_step)
        states, actions, rewards, next_states, dones = experiences
        
        #-------------------- Optimize Critic -----------------------#
        # The Critic is similar to TD-learning functionality, So we have to optimize it using the value function.
        # Run the critic network for the state to output a value given <state, action>
        expected_returns = self.critic_local.forward(states, actions) # We can also use critic_target to find the value as we did for DDQN
        # print(expected_returns)
        # We use the target network following the concept of Fixed-Q-network
        next_actions = self.actor_target(next_states)
        target_values = self.critic_target.forward(next_states, next_actions)
        target_returns = rewards + (gamma*target_values * (1-dones))
        critic_loss = F.mse_loss(expected_returns, target_returns)
        
        # optimize the critic loss
        self.critic_local_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_local_optimizer.step()

        # -------------------- Optimize Actor -----------------------#
        # The actor is similar to the Policy-gradient functionality. So we optimize it using sampled policy gradient.
        # Run the actor network for the current state to predict actions. We can not use "actions" from
        # experiences stored in the buffer because these actions might have been generated by the actor network
        # with old weights.
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # optimize the Actor loss
        self.actor_local_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_local_optimizer.step()

        # Store into stats dictionary
        self.critic_loss.append(abs(float(critic_loss)))
        self.actor_loss.append(abs(float(actor_loss)))
        self.rewards.append(rewards.cpu().data.numpy())
        

    def reset(self):
        self.exploration_policy.reset()
        
    
        self.rewards = []
        self.critic_loss = []
        self.actor_loss = []
        
    def save_stats(self):
        self.stats_dict['rewards'].append([np.mean(self.rewards), np.std(self.rewards)])
        self.stats_dict['actor_loss'].append([np.mean(self.actor_loss), np.std(self.actor_loss)])
        self.stats_dict['critic_loss'].append([np.mean(self.critic_loss), np.std(self.critic_loss)])
        # print('')
        # print(np.mean(self.rewards), np.var(self.rewards))
        # print(np.mean(self.actor_loss), np.var(self.actor_loss))
        # print(np.mean(self.critic_loss), np.var(self.critic_loss))
        
        cmn.dump_json(self.STATS_JSON_PATH, self.stats_dict)
        
    def save_checkpoints(self, episode_num):
        e_num = episode_num
        torch.save(
            self.actor_local.state_dict(), os.path.join(self.CHECKPOINT_DIR, 'actor_local_%s.pth' % str(e_num))
        )
        torch.save(
            self.actor_target.state_dict(), os.path.join(self.CHECKPOINT_DIR, 'actor_target_%s.pth' % str(e_num))
        )
        torch.save(
            self.critic_local.state_dict(), os.path.join(self.CHECKPOINT_DIR, 'critic_local_%s.pth' % str(e_num))
        )
        torch.save(
            self.critic_target.state_dict(), os.path.join(self.CHECKPOINT_DIR, 'critic_target_%s.pth' % str(e_num))
        )
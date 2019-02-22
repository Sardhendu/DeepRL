
import os
import numpy as np
import torch
import torch.nn.functional as F
import src.utils as utils

# TODO: Reset the noise to after every episode
# TODO: Add Batch-Normalization
# TODO: Try scaling the input features
# TODO: Density plot showing estimated Q values versus observed returns sampled from test (Refer DDPG Paper)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class DDPG:
    def __init__(self, args, env_type, mode='train', agent_id=0):
        self.env_type = env_type
        self.mode = mode
        self.agent_id = agent_id
        
        if mode == 'train':
            # Environment parameter
            self.ACTION_SIZE = args.ACTION_SIZE
            self.STATE_SIZE = args.STATE_SIZE
            
            # Learning Hyperparameters
            self.BUFFER_SIZE = args.BUFFER_SIZE
            self.BATCH_SIZE = args.BATCH_SIZE
            self.DATA_TO_BUFFER_BEFORE_LEARNING = args.DATA_TO_BUFFER_BEFORE_LEARNING
            self.LEARNING_FREQUENCY = args.LEARNING_FREQUENCY
            
            self.IS_SOFT_UPDATE = args.IS_SOFT_UPDATE
            self.SOFT_UPDATE_FREQUENCY = args.SOFT_UPDATE_FREQUENCY
            self.IS_HARD_UPDATE = args.IS_HARD_UPDATE
            self.HARD_UPDATE_FREQUENCY = args.HARD_UPDATE_FREQUENCY

            self.GAMMA = args.GAMMA
            self.TAU = args.TAU
            self.DECAY_TAU = args.DECAY_TAU
            self.TAU_DECAY_RATE = args.TAU_DECAY_RATE
            self.TAU_MIN = args.TAU_MIN

            # EXPLORATION - NOISE
            self.NOISE = args.NOISE_FN()
            self.NOISE_AMPLITUDE_DEACAY = args.NOISE_AMPLITUDE_DECAY_FN()
        
            # Logger
            self.CHECKPOINT_DIR = args.CHECKPOINT_DIR
            self.SUMMARY_LOGGER = args.SUMMARY_LOGGER
            
            # Create the Local Network and Target Network for the Actor
            self.actor_local = args.ACTOR_NETWORK_FN()
            self.actor_target = args.ACTOR_NETWORK_FN()
            self.actor_local_optimizer = args.ACTOR_OPTIMIZER_FN(self.actor_local.parameters())
    
            # Create the Local Network and Target Network for the Critic
            self.critic_local = args.CRITIC_NETWORK_FN()
            self.critic_target = args.CRITIC_NETWORK_FN()
            self.critic_local_optimizer = args.CRITIC_OPTIMIZER_FN(self.critic_local.parameters())
            
            #### MEMORY
            self.memory = args.MEMORY_FN()
            
        else:
            print('[Agent] Loading Actor/Critic weights')
            
            self.actor_local = args.ACTOR_NETWORK_FN()
            # self.critic_local = args.CRITIC_NETWORK_FN()
            self.CHECKPOINT_DIR = args.CHECKPOINT_DIR
            self.CHECKPOINT_NUMBER = args.CHECKPOINT_NUMBER

    def load_weights(self):
        """
            Add weights to the local network
        :return:
        """
        if self.mode == 'train':
            for ag_type in ['actor', 'critic']:
                checkpoint_path = os.path.join(
                        self.CHECKPOINT_DIR,
                        '%s_local_%s_%s.pth' % (str(ag_type), str(self.agent_id), str(self.CHECKPOINT_NUMBER))
                )
                print('Loading weights for %s_local for agent_%s' % (str(ag_type), str(self.agent_id)))
            
                if ag_type == 'actor':
                    self.actor_local.load_state_dict(torch.load(checkpoint_path))
                else:
                    self.critic_local.load_state_dict(torch.load(checkpoint_path))
    
        elif self.mode == 'test':
            checkpoint_path = os.path.join(
                    self.CHECKPOINT_DIR,
                    'actor_local_%s_%s.pth' % (str(self.agent_id), str(self.CHECKPOINT_NUMBER))
            )
            print('Loading weights for actor_local for agent_%s from \n %s' % (str(self.agent_id), str(checkpoint_path)))
            self.actor_local.load_state_dict(torch.load(checkpoint_path))
    
        else:
            raise ValueError('mode =  train or test permitted ....')
        
    def log(self, tag, value_dict, step):
        self.SUMMARY_LOGGER.add_scalars(tag, value_dict, step)
    
    
class DDPGAgent(DDPG):
    def __init__(self, args, env_type, mode, agent_id):
        super().__init__(args, env_type, mode,  agent_id)

    def act(self, state, action_value_range, running_timestep):
        """
        :param state:               Input n dimension state feature space
        :param add_noise:           bool True or False
        :param action_prob_range:   The limit range of action values
        :return:                    The action value distribution with added noise (if required)
        
        at = μ(st | θμ) + Nt
        
        Here
            --> θμ : Weights of Actor Network
            --> Nt : Are the random noise that jitters the action distribution introducing randomness for exploration
        """
        state = torch.from_numpy(state).float().to(device)
        state.requires_grad = False
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local.forward(state).cpu().data.numpy()
        self.actor_local.train()

        # print(actions)
        # Add Random noise to the action space distribution to foster exploration
        if self.mode == 'train':
            self.noise_amplitude = self.NOISE_AMPLITUDE_DEACAY.sample()
            self.noise_val = self.NOISE.sample() * self.noise_amplitude
            # print('self.noise_val: ', self.noise_val)
            # print('actions: ', actions)
            actions += self.noise_val
            
        if self.mode == 'train':
            tag = 'agent%i/noise_amplitude' % self.agent_id
            value_dict = {
                'noise_amplitude': float(self.noise_amplitude)
            }
            self.log(tag, value_dict, running_timestep)

            # tag = 'agent%i/noise_value' % self.agent_id
            # value_dict = {
            #     'noise_value_for_action_0': float(self.noise_val[0])
            # }
            # self.log(tag, value_dict, running_timestep)
        
        # Clip the actions to the the min and max limit of action probs
        actions = np.clip(actions, action_value_range[0], action_value_range[1])
        # print('Actions Value: ', actions)
        return actions

    def step(self, states, actions, rewards, next_states, dones, running_timestep):
    
        # Store experience to the replay buffer
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            # print('adding: ', state.shape, action.shape, reward, next_state.shape, done)
            self.memory.add(state, action, reward, next_state, done)
    
        # When the memory is at-least full as the batch size and if the step num is a factor of UPDATE_AFTER_STEP
        # then we learn the parameters of the network
        # Update the weights of local network and soft-update the weighs of the target_network
        # self.t_step = (self.t_step + 1) % self.UPDATE_AFTER_STEP  # Run from {1->UPDATE_AFTER_STEP}
        # print('[Step] Current Step is: ', self.tstep)
        if (running_timestep % self.LEARNING_FREQUENCY) == 0:
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.GAMMA, running_timestep)

        if running_timestep > self.DATA_TO_BUFFER_BEFORE_LEARNING:
            if self.IS_HARD_UPDATE:
                if (running_timestep % self.HARD_UPDATE_FREQUENCY) == 0:
                    utils.hard_update(self.actor_local, self.actor_target)
    
            elif self.IS_SOFT_UPDATE:
                if (running_timestep % self.SOFT_UPDATE_FREQUENCY) == 0:
                    utils.soft_update(self.critic_local, self.critic_target, self.TAU)
                    utils.soft_update(self.actor_local, self.actor_target, self.TAU)
            else:
                raise ValueError('Only One of HARD_UPDATE and SOFT_UPDATE is to be activated')

                
    def learn(self, experiences, gamma, running_timestep):
        """
        
        :param experiences:
        :param gamma:
        :return:
        
        """
        # print('Learning .... ', episode_num, running_timestep)
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

        # ------------------------- LOGGING --------------------------#
        tag = 'agent%i/actor_loss' % self.agent_id
        value_dict = {
            'actor_loss': abs(float(actor_loss))
        }
        self.log(tag, value_dict, running_timestep)

        tag = 'agent%i/critic_loss' % self.agent_id
        value_dict = {
            'critic_loss': abs(float(critic_loss)),
        }
        self.log(tag, value_dict, running_timestep)

    def reset(self):
        self.NOISE.reset()
        
    def save_checkpoints(self, episode_num):
        e_num = episode_num
        torch.save(
            self.actor_local.state_dict(), os.path.join(
                        self.CHECKPOINT_DIR, 'actor_local_%s_%s.pth' % (str(self.agent_id), str(e_num)))
        )
        torch.save(
            self.actor_target.state_dict(), os.path.join(
                        self.CHECKPOINT_DIR, 'actor_target_%s_%s.pth' % (str(self.agent_id), str(e_num)))
        )
        torch.save(
            self.critic_local.state_dict(), os.path.join(
                        self.CHECKPOINT_DIR, 'critic_local_%s_%s.pth' % (str(self.agent_id), str(e_num)))
        )
        torch.save(
            self.critic_target.state_dict(), os.path.join(
                        self.CHECKPOINT_DIR, 'critic_target_%s_%s.pth' % (str(self.agent_id), str(e_num)))
        )
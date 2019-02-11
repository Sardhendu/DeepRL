
##########  Experimentation Here

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src import utils
from src.buffer import MemoryER
from src.collab_compete import model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPG():
    """Meta agent that contains the two DDPG agents and shared replay buffer."""

    def __init__(self, args, mode):
        """

        :param args:    Configuration arguments
        :param mode:    Training or Testing mode
        """
        self.SEED = args.SEED
        self.ACTION_SIZE = args.ACTION_SIZE
        self.BUFFER_SIZE = args.BUFFER_SIZE
        self.BATCH_SIZE = args.BATCH_SIZE
        self.LEARNING_FREQUENCY = args.LEARNING_FREQUENCY
        self.GAMMA = args.GAMMA
        self.NUM_AGENT = args.NUM_AGENTS
        self.DATA_TO_BUFFER_BEFORE_LEARNING = args.DATA_TO_BUFFER_BEFORE_LEARNING

        self.SUMMARY_LOGGER = args.SUMMARY_LOGGER
        self.AGENTS = [DDPG(args, i, mode) for i in range(self.NUM_AGENT)]

        # create shared replay buffer
        self.MEMORY = MemoryER(self.BUFFER_SIZE, self.BATCH_SIZE, self.SEED, action_dtype='float')

    def step(self, all_states, all_actions, all_rewards, all_next_states, all_dones, running_timestep):
        all_states = np.expand_dims(np.stack(all_states, axis=0), axis=0)
        all_actions = np.expand_dims(np.stack(all_actions, axis=0), axis=0)
        all_rewards = np.expand_dims(np.stack(np.array(all_rewards, dtype=np.float32).reshape(-1, 1), axis=0), axis=0)
        all_next_states = np.expand_dims(np.stack(all_next_states, axis=0), axis=0)
        all_dones = np.expand_dims(np.stack(np.array(all_dones, dtype=np.bool).reshape(-1, 1), axis=0), axis=0)


        self.MEMORY.add(all_states, all_actions, all_rewards, all_next_states, all_dones)

        # Learn every update_every time steps.
        if running_timestep % self.LEARNING_FREQUENCY == 0:
            if len(self.MEMORY) > self.BATCH_SIZE:
                # We sample some experiences for each Agent
                experiences = [self.MEMORY.sample() for _ in range(self.NUM_AGENT)]
                self.learn(experiences, self.GAMMA, running_timestep)

        if running_timestep > self.DATA_TO_BUFFER_BEFORE_LEARNING:
            for ag_num, agent in enumerate(self.AGENTS):
                agent.update(running_timestep)

    def act(self, all_states, action_value_range, running_timestep):
        # pass each agent's state from the environment and calculate its action
        all_actions = []
        for agent, state in zip(self.AGENTS, all_states):
            action = agent.act(state, running_timestep=running_timestep)
            all_actions.append(action)
        return np.concatenate(all_actions)

    def learn(self, experiences, gamma, running_timestep):
        """

        :param experiences:             [num_agent] 1.e list of 2 nd_array.
                                        [[num_agents, states, agents, rewards, next_states, dones],
                                         [num_agents, states, agent, reward, next_states, dones]]
        :param gamma:                   discount
        :param running_timestep:
        :return:

        The input experiences contains samples (two sample) from each agent and each sample has experiences
        pertaining to each agents. Actor network have to be treated independently for each agent where the experience should
        pertain to only the agent.
        """
        # each agent uses its own actor to calculate next_actions
        all_next_actions = []
        all_action_pred = []
        for ag_id, agent in enumerate(self.AGENTS):
            states, _, _, next_states, _ = experiences[ag_id]
            agent_id = torch.tensor([ag_id]).to(device)
            state = states.index_select(1, agent_id).squeeze(1)
            next_state = next_states.index_select(1, agent_id).squeeze(1)
            action_pred = agent.actor_local(state)
            next_actions = agent.actor_target(next_state)

            # You might be wondering why do we get actions for the state, if we already have action for those
            # state using the replay buffer. Well the network weights might have changed after we stashed
            # the experience into the replay buffer. So here we get the fresh actions for the same states.
            all_action_pred.append(action_pred)
            all_next_actions.append(next_actions)

        # print(len(all_action_pred), len(all_next_actions))

        # each agent learns from its experience sample
        for i, agent in enumerate(self.AGENTS):
            agent.learn(i, experiences[i], gamma, all_next_actions, all_action_pred, running_timestep)

    def log(self, tag, value_dict, step):
        self.SUMMARY_LOGGER.add_scalars(tag, value_dict, step)

    def checkpoint(self, episode_num):
        for ag_num, agent in enumerate(self.AGENTS):
            agent.checkpoint(ag_num, episode_num)


class DDPG():
    """DDPG agent with own actor and critic."""

    def __init__(self, args, agent_id, mode):
        """
        :param args:            Config parameters
        :param agent_id:        The agent id to run
        :param mode:            train or test
        """
        self.agent_id = agent_id
        self.mode = mode
        self.SEED = args.SEED
        random.seed(self.SEED)

        self.NUM_AGENTS = args.NUM_AGENTS
        self.NOISE = args.NOISE_FN()
        self.STATE_SIZE = args.STATE_SIZE
        self.ACTION_SIZE = args.ACTION_SIZE
        self.TAU = args.TAU
        self.ACTOR_LEARNING_RATE = args.ACTOR_LEARNING_RATE
        self.CRITIC_LEARNING_RATE = args.CRITIC_LEARNING_RATE
        self.WEIGHT_DECAY = args.WEIGHT_DECAY

        self.IS_HARD_UPDATE = args.IS_HARD_UPDATE
        self.IS_SOFT_UPDATE = args.IS_SOFT_UPDATE
        self.SOFT_UPDATE_FREQUENCY = args.SOFT_UPDATE_FREQUENCY
        self.HARD_UPDATE_FREQUENCY = args.HARD_UPDATE_FREQUENCY
        self.NOISE_AMPLITUDE_DEACAY = args.NOISE_AMPLITUDE_DECAY_FN()

        self.CHECKPOINT_DIR = args.CHECKPOINT_DIR
        self.SUMMARY_LOGGER = args.SUMMARY_LOGGER

        # Actor Network
        self.actor_local = model.Actor(self.STATE_SIZE, self.ACTION_SIZE, [256, 256, 2], self.SEED).to(device)
        self.actor_target = model.Actor(self.STATE_SIZE, self.ACTION_SIZE, [256, 256, 2], self.SEED).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.ACTOR_LEARNING_RATE)

        # Critic Network
        self.critic_local = model.Critic(self.STATE_SIZE, self.ACTION_SIZE, [256, 256, 1], self.SEED).to(device)
        self.critic_target = model.Critic(self.STATE_SIZE, self.ACTION_SIZE, [256, 256, 1], self.SEED).to(device)
        self.critic_optimizer = optim.Adam(
                self.critic_local.parameters(), lr=self.CRITIC_LEARNING_RATE, weight_decay=self.WEIGHT_DECAY
        )

        # Set weights for local and target actor, respectively, critic the same
        utils.hard_update(self.actor_local, self.actor_target)
        utils.hard_update(self.critic_local, self.critic_target)

        # Noise process
        self.noise = args.NOISE_FN()

    def act(self, state, running_timestep):
        """Returns actions for given state as per current policy."""
        self.noise_amplitude = self.NOISE_AMPLITUDE_DEACAY.sample()
        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        # print(action)
        # tag = 'agent%i/action_val_before_noise' % self.agent_id
        # value_dict = {
        #     'action_value_0': float(action[0][0]),
        #     'action_value_1': float(action[0][1])
        # }
        # self.log(tag, value_dict, running_timestep)
        # print('Before Noise:       ', action)
        #
        if self.mode == 'train':
            self.noise_val = self.noise.sample() * self.noise_amplitude
            # print('Added  Noise:        ', self.noise_val)
            # print('After double noise =:', (2*self.noise_val + action))
            action += self.noise_val

        #     tag = 'agent%i/action_val_after_single_noise' % self.agent_id
        #     value_dict = {
        #         'action_value_0': float(action[0][0]),
        #         'action_value_1': float(action[0][1])
        #     }
        #     self.log(tag, value_dict, running_timestep)
        #     print('After single Noise: ', action)
        #
        #     # TODO: The action addition with noise was added twice mistakly but this gave better convergence, so we didnt remove it.
        #     action += self.noise_val
        #
        # print('After Double noise: ', action)
        # print('')


        if self.mode=='train' and self.log:
            tag = 'agent%i/noise_amplitude' % self.agent_id
            value_dict = {
                'noise_amplitude': float(self.noise_amplitude)
            }
            self.log(tag, value_dict, running_timestep)

            tag = 'agent%i/noise_value' % self.agent_id
            value_dict = {
                'noise_valu_for_action_0': float(self.noise_val[0]),
                'noise_valu_for_action_1': float(self.noise_val[1])
            }
            self.log(tag, value_dict, running_timestep)

            tag = 'agent%i/action_val_after_double_noise' % self.agent_id
            value_dict = {
                'action_value_0': float(action[0][0]),
                'action_value_1': float(action[0][1])
            }
            self.log(tag, value_dict, running_timestep)

        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, agent_id, experiences, gamma, all_next_actions, all_actions_pred, running_timestep):
        """
        :param agent_id:            The agent id to run
        :param experiences:         [all_states, all_actions, all_rewards, all_next_states, all_dones]
                                    :all_states : [batch_size, num_agents, ()]
        :param gamma:               Discount
        :param all_next_actions:    next_actions computed by actor_target for both the agents using next_states
        :param all_actions_pred:    action_pred computed by actor_local for both the agents using states
        :param running_timestep:   The running time step passed form the environment code
        :return:
        """

        all_states, all_actions, all_rewards, all_next_states, all_dones = experiences

        all_states = all_states.view(all_states.shape[0], -1)
        all_actions = all_actions.view(all_actions.shape[0], -1)
        all_rewards = all_rewards.view(all_rewards.shape[0], -1)
        all_next_states = all_next_states.view(all_next_states.shape[0], -1)
        all_dones = all_dones.view(all_dones.shape[0], -1)
        all_next_actions = torch.cat((all_next_actions[0], all_next_actions[1]), 1)
        agent_id = torch.tensor([agent_id]).to(device)

        # ---------------------------- update critic ---------------------------- #
        self.critic_optimizer.zero_grad()
        with torch.no_grad():
            q_targets_next = self.critic_target(all_next_states, all_next_actions)

        q_expected = self.critic_local(all_states, all_actions)
        # print('all_rewards: ', all_rewards)
        q_targets = all_rewards.index_select(1, agent_id) + \
                    (gamma * q_targets_next * (1 - all_dones.index_select(1,agent_id)))
        critic_loss = F.mse_loss(q_expected, q_targets.detach())
        # minimize loss
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # 1. We re-fetch the actions from the actor network. The reason being that the current experience
        # might be old and the actor weights might have been updated many times since the experience was collected
        # TODO: NOTE: all_action_pred ([action_agent_0, action_agent_1]) is a result of running the "actor_local"
        # network
        # TODO: for both the agents, however at this point only one of the agent is active for backpropagation and the
        # TODO: buffer for the other agent no longer exists. Therefore we detach the other agent's contribution from
        # the graph
        # TODO: current graph and only use its actions as another set of features to guide training.
        self.actor_optimizer.zero_grad()
        actions_pred = [actions if i == self.agent_id else actions.detach()
                        for i, actions in enumerate(all_actions_pred)]
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        actor_loss = -self.critic_local(all_states, actions_pred).mean()
        # minimize loss
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------------- LOGGING --------------------------#
        if self.log:
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

    def update(self, running_timestep):

        if self.IS_HARD_UPDATE:
            if (running_timestep % self.HARD_UPDATE_FREQUENCY) == 0:
                utils.hard_update(self.actor_local, self.actor_target)

        elif self.IS_SOFT_UPDATE:
            if (running_timestep % self.SOFT_UPDATE_FREQUENCY) == 0:
                utils.soft_update(self.critic_local, self.critic_target, self.TAU)
                utils.soft_update(self.actor_local, self.actor_target, self.TAU)
        else:
            raise ValueError('Only One of HARD_UPDATE and SOFT_UPDATE is to be activated')

    def log(self, tag, value_dict, step):
        self.SUMMARY_LOGGER.add_scalars(tag, value_dict, step)

    def checkpoint(self, ag_num, episode_num):
        torch.save(
                self.actor_local.state_dict(), os.path.join(
                        self.CHECKPOINT_DIR, 'actor_local_%s_%s.pth' % (str(ag_num), str(episode_num)))
        )
        torch.save(
                self.critic_local.state_dict(),
                os.path.join(self.CHECKPOINT_DIR, 'critic_local_%s_%s.pth' % (str(ag_num), str(episode_num)))
        )





















#
# import os
# import random
# import numpy as np
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
# from collections import namedtuple, deque
#
# from src import utils
# from src.collab_compete import model
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#
# class MADDPG():
#     """Meta agent that contains the two DDPG agents and shared replay buffer."""
#
#     def __init__(self, args, mode):
#         """
#
#         :param args:    Configuration arguments
#         :param mode:    Training or Testing mode
#         """
#         self.SEED = args.SEED
#         self.ACTION_SIZE = args.ACTION_SIZE
#         self.BUFFER_SIZE = args.BUFFER_SIZE
#         self.BATCH_SIZE = args.BATCH_SIZE
#         self.LEARNING_FREQUENCY = args.LEARNING_FREQUENCY
#         self.GAMMA = args.GAMMA
#         self.NUM_AGENT = args.NUM_AGENTS
#         self.DATA_TO_BUFFER_BEFORE_LEARNING = args.DATA_TO_BUFFER_BEFORE_LEARNING
#
#         self.SUMMARY_LOGGER = args.SUMMARY_LOGGER
#         self.AGENTS = [DDPG(args, i, mode) for i in range(self.NUM_AGENT)]
#
#         # create shared replay buffer
#         self.MEMORY = ReplayBuffer(self.BUFFER_SIZE, self.BATCH_SIZE, seed=self.SEED) #, action_dtype='float'
#
#     def step(self, all_states, all_actions, all_rewards, all_next_states, all_dones, running_timestep):
#         all_states = all_states.reshape(1, -1)  # reshape 2x24 into 1x48 dim vector
#         all_next_states = all_next_states.reshape(1, -1)  # reshape 2x24 into 1x48 dim vector
#         self.MEMORY.add(all_states, all_actions, all_rewards, all_next_states, all_dones)
#
#         # Learn every update_every time steps.
#         if running_timestep % self.LEARNING_FREQUENCY == 0:
#             if len(self.MEMORY) > self.BATCH_SIZE:
#                 experiences = [self.MEMORY.sample() for _ in range(self.NUM_AGENT)]
#                 self.learn(experiences, self.GAMMA, running_timestep)
#
#         if running_timestep > self.DATA_TO_BUFFER_BEFORE_LEARNING:
#             for ag_num, agent in enumerate(self.AGENTS):
#                 agent.update(running_timestep)
#
#     def act(self, all_states, action_value_range, running_timestep):
#         # pass each agent's state from the environment and calculate its action
#         all_actions = []
#         for agent, state in zip(self.AGENTS, all_states):
#             action = agent.act(state, running_timestep)
#             all_actions.append(action)
#         return np.array(all_actions).reshape(1, -1)  # reshape 2x2 into 1x4 dim vector
#
#     def learn(self, experiences, gamma, running_timestep):
#         # each agent uses its own actor to calculate next_actions
#         all_next_actions = []
#         all_action_pred = []
#         for i, agent in enumerate(self.AGENTS):
#             states, _, _, next_states, _ = experiences[i]
#             agent_id = torch.tensor([i]).to(device)
#             state = states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
#             next_state = next_states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
#
#             action_pred = agent.actor_local(state)
#             next_actions = agent.actor_target(next_state)
#
#             # You might be wondering why do we get actions for the state, if we already have action for those
#             # state using the replay buffer. Well the network weights might have changed after we stashed
#             # the experience into the replay buffer. So here we get the fresh actions for the same states.
#             all_action_pred.append(action_pred)
#             all_next_actions.append(next_actions)
#
#         # each agent learns from its experience sample
#         for i, agent in enumerate(self.AGENTS):
#             agent.learn(i, experiences[i], gamma, all_next_actions, all_action_pred, running_timestep)
#
#     def log(self, tag, value_dict, step):
#         self.SUMMARY_LOGGER.add_scalars(tag, value_dict, step)
#
#     def checkpoint(self, episode_num):
#         for ag_num, agent in enumerate(self.AGENTS):
#             agent.checkpoint(ag_num, episode_num)
#
#
# class DDPG():
#     """DDPG agent with own actor and critic."""
#
#     def __init__(self, args, agent_id, mode):
#         """
#         :param args:            Config parameters
#         :param agent_id:        The agent id to run
#         :param mode:            train or test
#         """
#         self.agent_id = agent_id
#         self.mode = mode
#         self.SEED = args.SEED
#         random.seed(self.SEED)
#
#         self.NOISE = args.NOISE_FN()
#         self.STATE_SIZE = args.STATE_SIZE
#         self.ACTION_SIZE = args.ACTION_SIZE
#         self.TAU = args.TAU
#         self.ACTOR_LEARNING_RATE = args.ACTOR_LEARNING_RATE
#         self.CRITIC_LEARNING_RATE = args.CRITIC_LEARNING_RATE
#         self.WEIGHT_DECAY = args.WEIGHT_DECAY
#
#         self.IS_HARD_UPDATE = args.IS_HARD_UPDATE
#         self.IS_SOFT_UPDATE = args.IS_SOFT_UPDATE
#         self.SOFT_UPDATE_FREQUENCY = args.SOFT_UPDATE_FREQUENCY
#         self.HARD_UPDATE_FREQUENCY = args.HARD_UPDATE_FREQUENCY
#         self.NOISE_AMPLITUDE_DEACAY = args.NOISE_AMPLITUDE_DECAY_FN()
#
#         self.CHECKPOINT_DIR = args.CHECKPOINT_DIR
#         self.SUMMARY_LOGGER = args.SUMMARY_LOGGER
#
#         # Actor Network
#         self.actor_local = model.Actor(self.STATE_SIZE, self.ACTION_SIZE, [256, 256, 2], self.SEED).to(device)
#         self.actor_target = model.Actor(self.STATE_SIZE, self.ACTION_SIZE, [256, 256, 2], self.SEED).to(device)
#         self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.ACTOR_LEARNING_RATE)
#
#         # Critic Network
#         self.critic_local = model.Critic(self.STATE_SIZE, self.ACTION_SIZE, [256, 256, 1], self.SEED).to(device)
#         self.critic_target = model.Critic(self.STATE_SIZE, self.ACTION_SIZE, [256, 256, 1], self.SEED).to(device)
#         self.critic_optimizer = optim.Adam(
#                 self.critic_local.parameters(), lr=self.CRITIC_LEARNING_RATE, weight_decay=self.WEIGHT_DECAY
#         )
#
#         # Set weights for local and target actor, respectively, critic the same
#         utils.hard_update(self.actor_local, self.actor_target)
#         utils.hard_update(self.critic_local, self.critic_target)
#
#         # Noise process
#         self.noise = args.NOISE_FN()
#
#     def act(self, state, running_timestep):
#         """Returns actions for given state as per current policy."""
#         self.noise_amplitude = self.NOISE_AMPLITUDE_DEACAY.sample()
#         state = torch.from_numpy(state).float().to(device)
#
#         self.actor_local.eval()
#         with torch.no_grad():
#             action = self.actor_local(state).cpu().data.numpy()
#         self.actor_local.train()
#
#         if self.mode == 'train':
#             self.noise_val = self.noise.sample() * self.noise_amplitude
#             # print(self.noise_val)
#             action += self.noise_val
#             action += self.noise_val
#
#         if self.log:
#             tag = 'agent%i/noise_amplitude' % self.agent_id
#             value_dict = {
#                 'noise_amplitude': float(self.noise_amplitude)
#             }
#             self.log(tag, value_dict, running_timestep)
#
#         return np.clip(action, -1, 1)
#
#     def reset(self):
#         self.noise.reset()
#
#     def learn(self, agent_id, experiences, gamma, all_next_actions, all_actions_pred, running_timestep):
#         """
#         :param agent_id:            The agent id to run
#         :param experiences:         [all_states, all_actions, all_rewards, all_next_states, all_dones]
#         :param gamma:               Discount
#         :param all_next_actions:    next_actions computed by actor_target for both the agents using next_states
#         :param all_actions_pred:    action_pred computed by actor_local for both the agents using states
#         :param running_timestep:   The running time step passed form the environment code
#         :return:
#         """
#
#         all_states, all_actions, all_rewards, all_next_states, all_dones = experiences
#         # print('all_states.shape: ', all_states.shape)
#
#         # ---------------------------- update critic ---------------------------- #
#         self.critic_optimizer.zero_grad()
#         agent_id = torch.tensor([agent_id]).to(device)
#         all_next_actions = torch.cat(all_next_actions, dim=1).to(device)
#         with torch.no_grad():
#             q_targets_next = self.critic_target(all_next_states, all_next_actions)
#
#         q_expected = self.critic_local(all_states, all_actions)
#         q_targets = all_rewards.index_select(1, agent_id) + (gamma * q_targets_next * (1 - all_dones.index_select(1,
#                                                                                                                   agent_id)))
#         critic_loss = F.mse_loss(q_expected, q_targets.detach())
#         # minimize loss
#         critic_loss.backward()
#         self.critic_optimizer.step()
#
#         # ---------------------------- update actor ---------------------------- #
#         self.actor_optimizer.zero_grad()
#         actions_pred = [actions if i == self.agent_id else actions.detach()
#                         for i, actions in enumerate(all_actions_pred)]
#         actions_pred = torch.cat(actions_pred, dim=1).to(device)
#         actor_loss = -self.critic_local(all_states, actions_pred).mean()
#         # minimize loss
#         actor_loss.backward()
#         self.actor_optimizer.step()
#
#         # ------------------------- LOGGING --------------------------#
#         if self.log:
#             tag = 'agent%i/actor_loss' % self.agent_id
#             value_dict = {
#                 'actor_loss': abs(float(actor_loss))
#             }
#             self.log(tag, value_dict, running_timestep)
#
#             tag = 'agent%i/critic_loss' % self.agent_id
#             value_dict = {
#                 'critic loss': abs(float(critic_loss)),
#             }
#             self.log(tag, value_dict, running_timestep)
#
#     def update(self, running_timestep):
#
#         if self.IS_HARD_UPDATE:
#             if (running_timestep % self.HARD_UPDATE_FREQUENCY) == 0:
#                 utils.hard_update(self.actor_local, self.actor_target)
#
#         elif self.IS_SOFT_UPDATE:
#             if (running_timestep % self.SOFT_UPDATE_FREQUENCY) == 0:
#                 utils.soft_update(self.critic_local, self.critic_target, self.TAU)
#                 utils.soft_update(self.actor_local, self.actor_target, self.TAU)
#         else:
#             raise ValueError('Only One of HARD_UPDATE and SOFT_UPDATE is to be activated')
#
#     def log(self, tag, value_dict, step):
#         self.SUMMARY_LOGGER.add_scalars(tag, value_dict, step)
#
#     def checkpoint(self, ag_num, episode_num):
#         torch.save(
#                 self.actor_local.state_dict(), os.path.join(
#                         self.CHECKPOINT_DIR, 'actor_local_%s_%s.pth' % (str(ag_num), str(episode_num)))
#         )
#         torch.save(
#                 self.critic_local.state_dict(),
#                 os.path.join(self.CHECKPOINT_DIR, 'critic_local_%s_%s.pth' % (str(ag_num), str(episode_num)))
#         )
# #
# #
# #
# #
# class ReplayBuffer:
#     """Fixed-size buffer to store experience tuples."""
#
#     def __init__(self, buffer_size, batch_size, seed):
#         """Initialize a ReplayBuffer object.
#         Params
#         ======
#             action_size (int): dimension of each action
#             buffer_size (int): maximum size of buffer
#             batch_size (int): size of each training batch
#             seed (int): Random seed
#         """
#         random.seed(seed)
#         self.MEMORY = deque(maxlen=buffer_size)
#         self.batch_size = batch_size
#         self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
#
#     def add(self, state, action, reward, next_state, done):
#         """Add a new experience to memory."""
#         e = self.experience(state, action, reward, next_state, done)
#         self.MEMORY.append(e)
#
#     def sample(self):
#         """Randomly sample a batch of experiences from memory."""
#         experiences = random.sample(self.MEMORY, k=self.batch_size)
#         states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
#         actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
#         rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
#         next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
#                 device)
#         dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
#                 device)
#         return (states, actions, rewards, next_states, dones)
#
#     def __len__(self):
#         """Return the current size of internal memory."""
#         return len(self.MEMORY)
#





# [Actor] Initializing the Actor network .....
# [Actor] Initializing the Actor network .....
# [INIT] Initializing Experience Replay Buffer .... .... ....
# Before Noise:        [[ 0.02830337 -0.0434454 ]]
# Added  Noise:         [0.17640523 0.04001572]
# After double noise =: [[0.38111384 0.03658604]]
# After single Noise:  [[ 0.2047086  -0.00342968]]
# After Double noise:  [[0.38111383 0.03658604]]
#
# Before Noise:        [[ 0.02715185 -0.04293306]]
# Added  Noise:         [0.0978738  0.22408932]
# After double noise =: [[0.22289944 0.40524558]]
# After single Noise:  [[0.12502564 0.18115626]]
# After Double noise:  [[0.22289944 0.40524557]]
#
# Before Noise:        [[ 0.03114995 -0.04573444]]
# Added  Noise:         [ 0.33670025 -0.06371443]
# After double noise =: [[ 0.70455045 -0.17316329]]
# After single Noise:  [[ 0.3678502  -0.10944887]]
# After Double noise:  [[ 0.70455045 -0.1731633 ]]
#
# Before Noise:        [[ 0.02886215 -0.04644768]]
# Added  Noise:         [0.17820157 0.1753402 ]
# After double noise =: [[0.38526529 0.30423273]]
# After single Noise:  [[0.20706372 0.12889253]]
# After Double noise:  [[0.3852653  0.30423272]]
#
# Before Noise:        [[ 0.03370981 -0.05163024]]
# Added  Noise:         [ 0.27587333 -0.01309741]
# After double noise =: [[ 0.58545646 -0.07782507]]
# After single Noise:  [[ 0.30958313 -0.06472766]]
# After Double noise:  [[ 0.58545643 -0.07782507]]
#
# Before Noise:        [[ 0.03231522 -0.0487376 ]]
# Added  Noise:         [0.16587569 0.29446652]
# After double noise =: [[0.3640666  0.54019544]]
# After single Noise:  [[0.19819091 0.24572892]]
# After Double noise:  [[0.3640666  0.54019547]]
#
# Before Noise:        [[ 0.03013395 -0.05171916]]
# Added  Noise:         [0.3105961 0.0010347]
# After double noise =: [[ 0.65132615 -0.04964975]]
# After single Noise:  [[ 0.34073004 -0.05068445]]
# After Double noise:  [[ 0.6513261  -0.04964975]]
#
# Before Noise:        [[ 0.03117386 -0.05057419]]
# Added  Noise:         [0.18538066 0.28366398]
# After double noise =: [[0.40193518 0.51675377]]
# After single Noise:  [[0.21655452 0.23308979]]
# After Double noise:  [[0.4019352 0.5167538]]
#
# Before Noise:        [[ 0.03212576 -0.05412824]]
# Added  Noise:         [ 0.41341459 -0.01963633]
# After double noise =: [[ 0.85895494 -0.0934009 ]]
# After single Noise:  [[ 0.44554034 -0.07376457]]
# After Double noise:  [[ 0.8589549 -0.0934009]]
#
# Before Noise:        [[ 0.03481433 -0.05244926]]
# Added  Noise:         [0.18888033 0.15570481]
# After double noise =: [[0.412575   0.25896035]]
# After single Noise:  [[0.22369467 0.10325555]]
# After Double noise:  [[0.412575   0.25896037]]
#
# Before Noise:        [[ 0.03493021 -0.05546212]]
# Added  Noise:         [0.09610342 0.04867098]
# After double noise =: [[0.22713706 0.04187984]]
# After single Noise:  [[ 0.13103363 -0.00679114]]
# After Double noise:  [[0.22713704 0.04187984]]
#
# Before Noise:        [[ 0.03434752 -0.05240119]]
# Added  Noise:         [0.2469919  0.05813258]
# After double noise =: [[0.52833132 0.06386398]]
# After single Noise:  [[0.2813394 0.0057314]]
# After Double noise:  [[0.5283313  0.06386398]]
#
# Before Noise:        [[ 0.02483886 -0.05001353]]
# Added  Noise:         [ 0.30866337 -0.10406624]
# After double noise =: [[ 0.6421656 -0.258146 ]]
# After single Noise:  [[ 0.33350223 -0.15407977]]
# After Double noise:  [[ 0.6421656 -0.258146 ]]
#
# Before Noise:        [[ 0.03488333 -0.0536081 ]]
# Added  Noise:         [0.21451897 0.03069431]
# After double noise =: [[0.46392127 0.00778052]]
# After single Noise:  [[ 0.2494023  -0.02291379]]
# After Double noise:  [[0.46392128 0.00778052]]
#
# Before Noise:        [[ 0.03468807 -0.05290968]]
# Added  Noise:         [0.41564179 0.05847958]
# After double noise =: [[0.86597164 0.06404948]]
# After single Noise:  [[0.45032987 0.0055699 ]]
# After Double noise:  [[0.8659717  0.06404947]]
#
# Before Noise:        [[ 0.03365932 -0.05278026]]
# Added  Noise:         [0.19783587 0.06390642]
# After double noise =: [[0.42933106 0.07503257]]
# After single Noise:  [[0.23149519 0.01112615]]
# After Double noise:  [[0.42933106 0.07503257]]
#
# Before Noise:        [[ 0.03482921 -0.05320338]]
# Added  Noise:         [ 0.26451694 -0.14837201]
# After double noise =: [[ 0.5638631  -0.34994739]]
# After single Noise:  [[ 0.29934615 -0.20157538]]
# After Double noise:  [[ 0.5638631 -0.3499474]]
#
# Before Noise:        [[ 0.03453835 -0.052162  ]]
# Added  Noise:         [0.13336927 0.06995535]
# After double noise =: [[0.30127689 0.0877487 ]]
# After single Noise:  [[0.16790763 0.01779335]]
# After Double noise:  [[0.3012769 0.0877487]]
#
# Before Noise:        [[ 0.02837053 -0.05334574]]
# Added  Noise:         [ 0.34786847 -0.00587822]
# After double noise =: [[ 0.72410747 -0.06510218]]
# After single Noise:  [[ 0.376239   -0.05922396]]
# After Double noise:  [[ 0.72410744 -0.06510218]]
#
# Before Noise:        [[ 0.03328988 -0.04961875]]
# Added  Noise:         [0.0746312  0.02923177]
# After double noise =: [[0.18255228 0.00884479]]
# After single Noise:  [[ 0.10792108 -0.02038698]]
# After Double noise:  [[0.18255228 0.00884479]]
#
# Before Noise:        [[ 0.03235    -0.05454718]]
# Added  Noise:         [ 0.1908329  -0.14699828]
# After double noise =: [[ 0.41401581 -0.34854375]]
# After single Noise:  [[ 0.2231829  -0.20154546]]
# After Double noise:  [[ 0.4140158  -0.34854373]]
#
# Before Noise:        [[ 0.03358329 -0.0469403 ]]
# Added  Noise:         [-0.1071905   0.21992455]
# After double noise =: [[-0.18079771  0.39290879]]
# After single Noise:  [[-0.07360721  0.17298424]]
# After Double noise:  [[-0.18079771  0.39290878]]
#
# Before Noise:        [[ 0.02685446 -0.05066392]]
# Added  Noise:         [ 0.11124275 -0.16875597]
# After double noise =: [[ 0.24933995 -0.38817586]]
# After single Noise:  [[ 0.13809721 -0.21941988]]
# After Double noise:  [[ 0.24933997 -0.38817585]]
#
# Before Noise:        [[ 0.03396369 -0.04408125]]
# Added  Noise:         [-0.21639146  0.2646849 ]
# After double noise =: [[-0.39881923  0.48528855]]
# After single Noise:  [[-0.18242776  0.22060364]]
# After Double noise:  [[-0.39881924  0.48528853]]
#
# Before Noise:        [[ 0.02907357 -0.04829215]]
# Added  Noise:         [-0.06683345 -0.1647166 ]
# After double noise =: [[-0.10459332 -0.37772536]]
# After single Noise:  [[-0.03775987 -0.21300876]]
# After Double noise:  [[-0.10459332 -0.37772536]]
#
# Before Noise:        [[ 0.03512936 -0.04041377]]
# Added  Noise:         [-0.2734794   0.26367241]
# After double noise =: [[-0.51182944  0.48693106]]
# After single Noise:  [[-0.23835003  0.22325864]]
# After Double noise:  [[-0.51182944  0.48693106]]
#
# Before Noise:        [[ 0.02815485 -0.04379999]]
# Added  Noise:         [-0.10788894 -0.25807233]
# After double noise =: [[-0.18762304 -0.55994464]]
# After single Noise:  [[-0.07973409 -0.3018723 ]]
# After Double noise:  [[-0.18762304 -0.5599446 ]]
#
# Before Noise:        [[ 0.03300256 -0.03853785]]
# Added  Noise:         [-0.23527571  0.26695474]
# After double noise =: [[-0.43754886  0.49537163]]
# After single Noise:  [[-0.20227315  0.22841689]]
# After Double noise:  [[-0.43754885  0.49537164]]
#
# Before Noise:        [[ 0.03047605 -0.04383776]]
# Added  Noise:         [-0.08505388 -0.18911429]
# After double noise =: [[-0.13963171 -0.42206634]]
# After single Noise:  [[-0.05457783 -0.23295204]]
# After Double noise:  [[-0.1396317  -0.42206633]]
#
# Before Noise:        [[ 0.03164908 -0.03964389]]
# Added  Noise:         [-0.26341656  0.19063741]
# After double noise =: [[-0.49518404  0.34163093]]
# After single Noise:  [[-0.23176748  0.15099351]]
# After Double noise:  [[-0.49518403  0.34163094]]
#
# Episode 1	Average Score: 0.000Before Noise:        [[ 0.02843429 -0.0434558 ]]
# Added  Noise:         [-0.13954184 -0.19670246]
# After double noise =: [[-0.25064939 -0.43686073]]
# After single Noise:  [[-0.11110755 -0.24015826]]
# After Double noise:  [[-0.2506494 -0.4368607]]
#
# Before Noise:        [[ 0.02712857 -0.04286777]]
# Added  Noise:         [-0.30521871 -0.01058646]
# After double noise =: [[-0.58330885 -0.06404069]]
# After single Noise:  [[-0.27809015 -0.05345423]]
# After Double noise:  [[-0.5833089  -0.06404069]]
#
# Before Noise:        [[ 0.02819972 -0.04087237]]
# Added  Noise:         [-0.10086795 -0.20737519]
# After double noise =: [[-0.17353619 -0.45562274]]
# After single Noise:  [[-0.07266823 -0.24824755]]
# After Double noise:  [[-0.17353618 -0.45562273]]
#
# Before Noise:        [[ 0.02546468 -0.0391941 ]]
# Added  Noise:         [-0.42245574  0.03727973]
# After double noise =: [[-0.8194468   0.03536537]]
# After single Noise:  [[-0.39699107 -0.00191436]]
# After Double noise:  [[-0.8194468   0.03536537]]
#
# Before Noise:        [[ 0.03010918 -0.04353552]]
# Added  Noise:         [-0.1764676  -0.17107437]
# After double noise =: [[-0.32282601 -0.38568426]]
# After single Noise:  [[-0.14635842 -0.21460989]]
# After Double noise:  [[-0.322826   -0.38568425]]
#
# Before Noise:        [[ 0.03424921 -0.04026116]]
# Added  Noise:         [-0.28617832  0.04458607]
# After double noise =: [[-0.53810743  0.04891097]]
# After single Noise:  [[-0.2519291   0.00432491]]
# After Double noise:  [[-0.5381074   0.04891097]]
#
# Before Noise:        [[ 0.02666076 -0.04422332]]
# Added  Noise:         [-0.03605739 -0.2688958 ]
# After double noise =: [[-0.04545402 -0.58201491]]
# After single Noise:  [[-0.00939663 -0.3131191 ]]
# After Double noise:  [[-0.04545401 -0.5820149 ]]
#
# Before Noise:        [[ 0.02900155 -0.04614713]]
# Added  Noise:         [-0.20301741 -0.03058285]
# After double noise =: [[-0.37703327 -0.10731284]]
# After single Noise:  [[-0.17401586 -0.07672999]]
# After Double noise:  [[-0.37703326 -0.10731284]]
#
# Before Noise:        [[ 0.03070853 -0.04354624]]
# Added  Noise:         [-0.11772849 -0.28644639]
# After double noise =: [[-0.20474846 -0.61643902]]
# After single Noise:  [[-0.08701997 -0.32999262]]
# After Double noise:  [[-0.20474847 -0.61643904]]
#
# Before Noise:        [[ 0.03145654 -0.04169635]]
# Added  Noise:         [-0.20372005 -0.02037889]
# After double noise =: [[-0.37598356 -0.08245413]]
# After single Noise:  [[-0.1722635  -0.06207524]]
# After Double noise:  [[-0.37598357 -0.08245413]]
#
# Before Noise:        [[ 0.02666878 -0.04664032]]
# Added  Noise:         [-0.2165842  -0.15339679]
# After double noise =: [[-0.40649963 -0.35343389]]
# After single Noise:  [[-0.18991542 -0.2000371 ]]
# After Double noise:  [[-0.40649962 -0.3534339 ]]
#
# Before Noise:        [[ 0.0320846  -0.04161727]]
# Added  Noise:         [-0.1265958  -0.17094643]
# After double noise =: [[-0.221107   -0.38351012]]
# After single Noise:  [[-0.0945112 -0.2125637]]
# After Double noise:  [[-0.22110699 -0.3835101 ]]
#
# Before Noise:        [[ 0.03156123 -0.04263719]]
# Added  Noise:         [-0.03527135  0.05920165]
# After double noise =: [[-0.03898148  0.07576611]]
# After single Noise:  [[-0.00371013  0.01656446]]
# After Double noise:  [[-0.03898148  0.07576611]]
#
# Before Noise:        [[ 0.03231094 -0.0414444 ]]
# Added  Noise:         [ 0.01027153 -0.16329695]
# After double noise =: [[ 0.052854   -0.36803829]]
# After single Noise:  [[ 0.04258247 -0.20474134]]
# After Double noise:  [[ 0.052854  -0.3680383]]
#
# Before Noise:        [[ 0.02890573 -0.04433765]]
# Added  Noise:         [-0.13705591  0.15576658]
# After double noise =: [[-0.24520609  0.2671955 ]]
# After single Noise:  [[-0.10815018  0.11142893]]
# After Double noise:  [[-0.24520609  0.2671955 ]]
#
# Before Noise:        [[ 0.03264843 -0.04170824]]
# Added  Noise:         [-0.0315869 -0.0165579]
# After double noise =: [[-0.03052536 -0.07482404]]
# After single Noise:  [[ 0.00106153 -0.05826614]]
# After Double noise:  [[-0.03052536 -0.07482404]]
#
# Before Noise:        [[ 0.03119507 -0.04184152]]
# Added  Noise:         [-0.09567003  0.23006549]
# After double noise =: [[-0.16014498  0.41828946]]
# After single Noise:  [[-0.06447496  0.18822397]]
# After Double noise:  [[-0.16014498  0.41828945]]
#
# Before Noise:        [[ 0.03239577 -0.04087543]]
# Added  Noise:         [0.00878778 0.0565831 ]
# After double noise =: [[0.04997132 0.07229078]]
# After single Noise:  [[0.04118354 0.01570767]]
# After Double noise:  [[0.04997132 0.07229078]]
#
# Before Noise:        [[ 0.03113917 -0.04163601]]
# Added  Noise:         [-0.08026952  0.37414272]
# After double noise =: [[-0.12939987  0.70664942]]
# After single Noise:  [[-0.04913035  0.33250672]]
# After Double noise:  [[-0.12939987  0.7066494 ]]
#
# Before Noise:        [[ 0.03384942 -0.0410213 ]]
# Added  Noise:         [0.02016082 0.08829458]
# After double noise =: [[0.07417106 0.13556785]]
# After single Noise:  [[0.05401024 0.04727327]]
# After Double noise:  [[0.07417107 0.13556784]]
#
# Before Noise:        [[ 0.03374403 -0.04293025]]
# Added  Noise:         [0.12008598 0.1832454 ]
# After double noise =: [[0.27391598 0.32356056]]
# After single Noise:  [[0.15383    0.14031516]]
# After Double noise:  [[0.27391598 0.32356057]]
#
# Before Noise:        [[ 0.03310213 -0.04136442]]
# Added  Noise:         [-0.1099118   0.17199006]
# After double noise =: [[-0.18672147  0.3026157 ]]
# After single Noise:  [[-0.07680967  0.13062565]]
# After Double noise:  [[-0.18672147  0.3026157 ]]
#
# Before Noise:        [[ 0.03393783 -0.04528787]]
# Added  Noise:         [-0.01523926  0.35012071]
# After double noise =: [[0.00345931 0.65495355]]
# After single Noise:  [[0.01869857 0.30483285]]
# After Double noise:  [[0.00345931 0.65495354]]
#
# Before Noise:        [[ 0.0333469  -0.04038858]]
# Added  Noise:         [-0.13478693  0.07144607]
# After double noise =: [[-0.23622696  0.10250356]]
# After single Noise:  [[-0.10144003  0.03105749]]
# After Double noise:  [[-0.23622696  0.10250356]]
#
# Before Noise:        [[ 0.03225517 -0.04625723]]
# Added  Noise:         [0.17934083 0.44565408]
# After double noise =: [[0.39093683 0.84505094]]
# After single Noise:  [[0.211596   0.39939687]]
# After Double noise:  [[0.39093682 0.84505093]]
#
# Before Noise:        [[ 0.03441748 -0.04020369]]
# Added  Noise:         [0.07218701 0.15133362]
# After double noise =: [[0.1787915  0.26246356]]
# After single Noise:  [[0.10660449 0.11112994]]
# After Double noise:  [[0.1787915  0.26246357]]
#
# Before Noise:        [[ 0.03552418 -0.04835942]]
# Added  Noise:         [0.06631714 0.56981247]
# After double noise =: [[0.16815846 1.09126551]]
# After single Noise:  [[0.10184132 0.521453  ]]
# After Double noise:  [[0.16815846 1.0912654 ]]
#
# Before Noise:        [[ 0.03472001 -0.04107192]]
# Added  Noise:         [0.03455862 0.20887922]
# After double noise =: [[0.10383725 0.37668652]]
# After single Noise:  [[0.06927863 0.1678073 ]]
# After Double noise:  [[0.10383724 0.3766865 ]]
#
# Episode 2	Average Score: 0.000Before Noise:        [[ 0.02819583 -0.04332345]]
# Added  Noise:         [0.15109476 0.46883959]
# After double noise =: [[0.33038536 0.89435573]]
# After single Noise:  [[0.1792906  0.42551613]]
# After Double noise:  [[0.33038536 0.8943557 ]]
#
# Before Noise:        [[ 0.02702832 -0.04279919]]
# Added  Noise:         [0.09078276 0.269768  ]
# After double noise =: [[0.20859384 0.49673682]]
# After single Noise:  [[0.11781108 0.22696881]]
# After Double noise:  [[0.20859385 0.49673682]]
#
# Before Noise:        [[ 0.03181523 -0.04546495]]
# Added  Noise:         [0.1660731  0.28857357]
# After double noise =: [[0.36396143 0.53168219]]
# After single Noise:  [[0.19788833 0.24310862]]
# After Double noise:  [[0.36396143 0.5316822 ]]
#
# Before Noise:        [[ 0.02834738 -0.04595442]]
# Added  Noise:         [0.10698917 0.36194139]
# After double noise =: [[0.24232571 0.67792836]]
# After single Noise:  [[0.13533655 0.31598696]]
# After Double noise:  [[0.24232571 0.6779283 ]]
#
# Before Noise:        [[ 0.03169684 -0.04979423]]
# Added  Noise:         [0.07170535 0.23032408]
# After double noise =: [[0.17510754 0.41085393]]
# After single Noise:  [[0.10340219 0.18052985]]
# After Double noise:  [[0.17510754 0.41085392]]
#
# Before Noise:        [[ 0.03116725 -0.04754654]]
# Added  Noise:         [0.04742544 0.49257656]
# After double noise =: [[0.12601812 0.93760658]]
# After single Noise:  [[0.07859268 0.44503003]]
# After Double noise:  [[0.12601812 0.9376066 ]]
#
# Before Noise:        [[ 0.03151493 -0.04810339]]
# Added  Noise:         [0.12817902 0.23652165]
# After double noise =: [[0.28787298 0.42493991]]
# After single Noise:  [[0.15969396 0.18841825]]
# After Double noise:  [[0.28787297 0.4249399 ]]
#
# Before Noise:        [[ 0.03235072 -0.04696623]]
# Added  Noise:         [-0.03667999  0.47261499]
# After double noise =: [[-0.04100925  0.89826375]]
# After single Noise:  [[-0.00432926  0.42564875]]
# After Double noise:  [[-0.04100925  0.89826375]]
#
# Before Noise:        [[ 0.03388338 -0.04898033]]
# Added  Noise:         [0.0415189  0.20422646]
# After double noise =: [[0.11692119 0.35947259]]
# After single Noise:  [[0.07540229 0.15524612]]
# After Double noise:  [[0.11692119 0.35947257]]
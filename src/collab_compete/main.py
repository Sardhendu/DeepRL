
#
# # TODO: Train separated DDPG agent to learn their own policy (Competitive environment)
# # TODO: Try the Centralized environment (Collaborative)
#
#
# from unityagents import UnityEnvironment
# import numpy as np
#
# import torch
#
#
# # env = UnityEnvironment('./Tennis.app')
# # brain_name = env.brain_names[0]
# # brain = env.brains[brain_name]
# # print(brain)
# #
# #
# #
# # env_info = env.reset(train_mode=True)[brain_name]
# #
# # # number of agents
# # num_agents = len(env_info.agents)
# # print('Number of agents:', num_agents)
# #
# # # size of each action
# # action_size = brain.vector_action_space_size
# # print('Size of each action:', action_size)
# #
# # # examine the state space
# # states = env_info.vector_observations
# # state_size = states.shape[1]
# # print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
# # print('The state for the first agent looks like:', states[0])
# #
# #
#
#
# class CollabCompete:
#
#     def __init__(self, mode='train'):
#         if mode != 'train':
#             print('[Mode] Setting to Test Mode')
#             self.train = False
#         else:
#             print('[Mode] Setting to Train Mode')
#             self.train = True
#
#
#         self.base_env = UnityEnvironment(file_name='Tennis.app')
#
#
#         self.brain_name = self.base_env.brain_names[0]
#         self.brain = self.base_env.brains[self.brain_name]
#         self.action_size = self.brain.vector_action_space_size
#
#
#     def reset(self):
#         self.env_info = self.base_env.reset(train_mode=self.train)[self.brain_name]
#         return self.get_state()
#
#
#     def get_state(self):
#         return self.env_info.vector_observations
#
#
#     def step(self, action):
#         # print(self.brain_name)
#         # print(action)
#         self.env_info = self.base_env.step(action)[self.brain_name]  # send the action to the environment
#         next_states = self.get_state()
#         rewards = self.env_info.rewards
#         dones = self.env_info.local_done
#         return next_states, rewards, dones, None
#
#
#     def close(self):
#         self.base_env.close()
#
#
#
#
#
#
#
#
# def seeding(seed=1):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#
# def main():
#     seeding()
#
#
#     env = CollabCompete()
#
#     num_agents = 2
#     action_size = 2
#     for i in range(1, 6):                                      # play game for 5 episodes
#         states = env.reset()
#         scores = np.zeros(num_agents)
#         print('States shape: ', states.shape)
#         while True:
#             actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
#             actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
#             print(actions)
#             next_states, rewards, dones, _ = env.step(actions)
#
#             print('Next states shape: ', next_states.shape)
#             scores += rewards                         # update the score (for each agent)
#             states = next_states                               # roll over states to next time step
#             if np.any(dones):                                  # exit loop if episode finished
#                 break
#         print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))
#
#
#
# main()

















# TODO: Train separated DDPG agent to learn their own policy (Competitive environment)
# TODO: Try the Centralized environment (Collaborative)



import numpy as np
import torch
from unityagents import UnityEnvironment


from src.collab_compete.agent import Agent
from src.collab_compete.config import Config


# env = UnityEnvironment('./Tennis.app')
# brain_name = env.brain_names[0]
# brain = env.brains[brain_name]
# print(brain)
#
#
#
# env_info = env.reset(train_mode=True)[brain_name]
#
# # number of agents
# num_agents = len(env_info.agents)
# print('Number of agents:', num_agents)
#
# # size of each action
# action_size = brain.vector_action_space_size
# print('Size of each action:', action_size)
#
# # examine the state space
# states = env_info.vector_observations
# state_size = states.shape[1]
# print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
# print('The state for the first agent looks like:', states[0])
#
#


class CollabCompete:

    def __init__(self, mode='train'):
        if mode != 'train':
            print('[Mode] Setting to Test Mode')
            self.train = False
        else:
            print('[Mode] Setting to Train Mode')
            self.train = True
        

        self.base_env = UnityEnvironment(file_name='Tennis.app')
        
        
        self.brain_name = self.base_env.brain_names[0]
        self.brain = self.base_env.brains[self.brain_name]
        self.action_size = self.brain.vector_action_space_size
    
    
    def reset(self):
        self.env_info = self.base_env.reset(train_mode=self.train)[self.brain_name]
        return self.get_state()
    
    
    def get_state(self):
        return self.env_info.vector_observations
    
    
    def step(self, action):
        # print(self.brain_name)
        # print(action)
        self.env_info = self.base_env.step(action)[self.brain_name]  # send the action to the environment
        next_states = self.get_state()
        rewards = self.env_info.rewards
        dones = self.env_info.local_done
        return next_states, rewards, dones, None
    
    
    def close(self):
        self.base_env.close()
    








def main():
    args = Config
    env = CollabCompete()

    num_agents = 2
    action_size = 2
    
    agent = Agent(args=args, mode='train')
    scores = [args.SCORES_FN(num) for num in range(0, args.NUM_AGENTS)]
    running_time_step = 1
    
    for episode_num in range(1, 1000):                                      # play game for 5 episodes
        
        print('Running Episode: ', episode_num)
        states = env.reset()
        scores_per_episode = np.zeros(num_agents)
        # print('States shape: ', states.shape)
        
        # Number of time steps is actually very less before the episode is done(One of the agent losses)
        for t in range(0, args.NUM_TIMESTEPS):
            actions = agent.act(states) # select an action (for each agent)
            # print('Actions: ', actions)
        
            next_states, rewards, dones, _ = env.step(actions)
            agent.step(states, actions, rewards, next_states, dones, episode_num, running_time_step)
            # print('Next states shape: ', next_states.shape)
            # print('RewardsRewards: ', rewards)
            scores_per_episode += rewards                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step

            running_time_step += 1
            
            if np.any(dones):                                  # exit loop if episode finished
                print('Number of steps before done: ', t)
                break

        _ = [
        scores[ag_num].push(
                scores=scores_per_episode[ag_num], episode_num=episode_num, logger=args.SUMMARY_LOGGER)
            for ag_num in range(0,args.NUM_AGENTS)
        ]
        
        # print ('scoresscoresscores: ', scores)
        #
        # args.SUMMARY_LOGGER.add_scalars('agent%i/losses' % agent_num,
        #                                 {'critic loss': critic_loss,
        #                                  'actor_loss': actor_loss},
        #                                 running_time_step)
            
        # print('Score (max over agents) from episode {}: {}'.format(episode_num, np.max(scores)))



main()


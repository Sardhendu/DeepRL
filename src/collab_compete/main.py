
# TODO: Train separated DDPG agent to learn their own policy (Competitive environment)
# TODO: Try the Centralized environment (Collaborative)



import numpy as np
import torch
from unityagents import UnityEnvironment


from src.collab_compete.agent import Agent, CentralizedAgent
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
#
# def main():
#     args = Config
#     env = CollabCompete()
#
#     num_agents = 2
#     action_size = 2
#
#     # agent = Agent(args=args, mode='train')
#     agent = CentralizedAgent(args=args, mode='train')
#
#     scores = [args.SCORES_FN(num) for num in range(0, args.NUM_AGENTS)]
#     scores_max = args.SCORES_FN('01_max')
#     running_time_step = 1
#
#     for episode_num in range(1, 10000):                                      # play game for 5 episodes
#
#         print('Running Episode: ', episode_num)
#         timesteps_before_done = 1
#         states = env.reset()
#         scores_per_episode = np.zeros(num_agents)
#         # print('States shape: ', states.shape)
#
#         # Number of time steps is actually very less before the episode is done(One of the agent losses)
#         for t in range(0, args.NUM_TIMESTEPS):
#             actions = agent.act(states,  action_value_range=(-1, 1)) # select an action (for each agent)
#             # print('Actions: ', actions)
#
#             next_states, rewards, dones, _ = env.step(actions)
#             agent.step(states, actions, rewards, next_states, dones, episode_num, running_time_step)
#             # print('Next states shape: ', next_states.shape)
#
#             scores_per_episode += rewards                      # update the score (for each agent)
#             states = next_states                               # roll over states to next time step
#
#             running_time_step += 1
#
#             if np.any(dones):                                  # exit loop if episode finished
#                 timesteps_before_done = t
#                 break
#
#         agent.reset()
#         _ = [
#         scores[ag_num].push(
#                 scores=scores_per_episode[ag_num], episode_num=episode_num, logger=args.SUMMARY_LOGGER)
#             for ag_num in range(0,args.NUM_AGENTS)
#         ]
#
#         max_score = max(scores_per_episode)
#         scores_max.push(
#                 scores=max_score, episode_num=episode_num, logger=args.SUMMARY_LOGGER
#         )
#         max_avg_score = scores_max.get_avg_score()
#         # print('max_avg_score: ', max_avg_score)
#
#         agent.SUMMARY_LOGGER.add_scalars(
#                 'common/tstep_per_episode',
#                 {'tsteo_per_episode': timesteps_before_done},
#                 episode_num
#         )
#
#         print('\rEpisode {}\tAverage Score: {:.3f}'.format(episode_num, max_avg_score), end="")
#
#         if max_avg_score>0.5:
#             agent.checkpoint(episode_num)
#             print('Voila! GOT THE DESIRED SCORE ..........')
#             break
#
#         print ('scoresscoresscores: ', scores)
#
#         args.SUMMARY_LOGGER.add_scalars('agent%i/losses' % agent_num,
#                                         {'critic loss': critic_loss,
#                                          'actor_loss': actor_loss},
#                                         running_time_step)
#
#         print('Score (max over agents) from episode {}: {}'.format(episode_num, np.max(scores)))







import numpy as np
import numpy as np

from src.collab_compete.agent import CentralizedAgent
from src.collab_compete.agent1 import MDDPG
from collections import deque
from unityagents import UnityEnvironment
from src.collab_compete.config import Config
env = UnityEnvironment(file_name="Tennis.app", no_graphics=True)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# agent = MADDPG(seed=2, noise_start=0.5, update_every=2, gamma=1, t_stop_noise=30000)
agent = MDDPG(args=Config, mode='train')
n_episodes = 10000
max_t = 1000
scores = []
scores_deque = deque(maxlen=100)
scores_avg = []

iteration_num = 0
for i_episode in range(1, n_episodes + 1):
    rewards = []
    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    state = env_info.vector_observations  # get the current state (for each agent)

    # loop over steps
    for t in range(max_t):
        # print('Time step: ', t)
        # select an action
        action = agent.act(state, action_value_range=(-1,1), running_time_step=iteration_num)
        # take action in environment and set parameters to new values
        
        # print('242432 ', action)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations
        
        # print('next_sate: ', next_state)
        
        rewards_vec = env_info.rewards
        done = env_info.local_done
        # update and train agent with returned information
        agent.step(state, action, rewards_vec, next_state, done, i_episode, iteration_num, log = True)
        # agent.step(state, action, rewards_vec, next_state, done, iteration_num)
        state = next_state
        rewards.append(rewards_vec)
        iteration_num += 1
        if any(done):
            break

    # calculate episode reward as maximum of individually collected rewards of agents
    episode_reward = np.max(np.sum(np.array(rewards), axis=0))

    scores.append(episode_reward)  # save most recent score to overall score array
    scores_deque.append(episode_reward)  # save most recent score to running window of 100 last scores
    current_avg_score = np.mean(scores_deque)
    scores_avg.append(current_avg_score)  # save average of last 100 scores to average score array

    print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, current_avg_score), end="")

    # log average score every 200 episodes
    if i_episode % 200 == 0:
        print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, current_avg_score))
        agent.checkpoint(episode_num=i_episode)

    # break and report success if environment is solved
    if np.mean(scores_deque) >= .5:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(i_episode, np.mean(scores_deque)))
        agent.checkpoint(episode_num=i_episode)
        break
        
    tag = 'common/avg_score'
    value_dict =  {'avg_score_100_episode': current_avg_score}
    step = i_episode

    agent.log(tag, value_dict, step)
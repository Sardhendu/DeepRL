
import numpy as np
from src.collab_compete.agent import MADDPG
from collections import deque
from unityagents import UnityEnvironment
from src.collab_compete.config import TrainConfig, TestConfig


class CollabCompeteEnv:

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
        self.env_info = self.base_env.step(action)[self.brain_name]  # send the action to the environment
        next_states = self.get_state()
        rewards = self.env_info.rewards
        dones = self.env_info.local_done
        return next_states, rewards, dones, None

    def close(self):
        self.base_env.close()


class CollabCompete:
    
    def __init__(self, args, mode):
        self.args = args
        self.env = CollabCompeteEnv(mode)

    def train(self):
        
        agent_centralized = MADDPG(args=self.args, mode='train')
        n_episodes = 10000
        max_t = 1000
        scores = []
        scores_deque = deque(maxlen=100)
        scores_avg = []

        running_timestep = 0
        for i_episode in range(1, n_episodes + 1):
            rewards = []
            state = self.env.reset()  # get the current state (for each agent)

            for t in range(max_t):
                # print('Time step: ', t)
                # select an action
                action = agent_centralized.act(state, action_value_range=(-1, 1), running_timestep=running_timestep)
                # take action in environment and set parameters to new values
                
                next_state, rewards_vec, done, _ = self.env.step(action)
                agent_centralized.step(state, action, rewards_vec, next_state, done, running_timestep)

                state = next_state
                rewards.append(rewards_vec)
                running_timestep += 1

                if any(done):
                    break

            # # calculate episode reward as maximum of individually collected rewards of agents
            agent_scores_per_episode = np.sum(np.array(rewards), axis=0)
            max_scores_per_episode = np.max(agent_scores_per_episode)
            # print(max_scores_per_episode)
            #
            scores.append(max_scores_per_episode)  # save most recent score to overall score array
            scores_deque.append(max_scores_per_episode)  # save most recent score to running window of 100 last scores
            current_avg_score = np.mean(scores_deque)
            scores_avg.append(current_avg_score)  # save average of last 100 scores to average score array

            ########################
            for ag_num in range(2):
                tag = 'agent%i/scores_per_episode' % ag_num
                value_dict = {
                    'scores_per_episode': float(agent_scores_per_episode[ag_num])
                }
                self.args.SUMMARY_LOGGER.add_scalars(tag, value_dict, i_episode)

            tag = 'common/avg_score'
            value_dict = {'avg_score_100_episode': current_avg_score}
            step = i_episode

            agent_centralized.log(tag, value_dict, step)
            ########################

            print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, current_avg_score), end="")

            # log average score every 200 episodes
            if i_episode % 200 == 0:
                print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, current_avg_score))
                agent_centralized.checkpoint(episode_num=i_episode)
            #
            # break and report success if environment is solved
            if np.mean(scores_deque) >= .7:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(i_episode,
                                                                                             np.mean(scores_deque)))
                agent_centralized.checkpoint(episode_num=i_episode)
                break

    def test(self, trials=10, steps=200):
        self.agent = MADDPG(args=self.args, mode='test')
        self.agent.load_weights()
        for i in range(trials):
            total_reward = np.zeros(self.args.NUM_AGENTS)
            print('Starting Testing ...')
            state = self.env.reset()
            for j in range(steps):
                action = self.agent.act(state, action_value_range=(-1, 1), running_timestep=j)
                # print(action)
                # self.env.render()
                next_states, rewards, dones, _ = self.env.step(action)
                total_reward += rewards
                
                if rewards != 0:
                    print("Current Reward:", np.max(rewards), "Total Reward:", np.max(total_reward))
                state = next_states
                if any(dones):
                    print('Done.')
                    break
        self.env.close()



train=False
if train:
    CollabCompete(args=TrainConfig, mode='train').train()
else:
    CollabCompete(args=TestConfig, mode='test').test()



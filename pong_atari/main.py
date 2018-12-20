from DeepRL.pong_atari.parallel_env import parallelEnv
from DeepRL.pong_atari.agent import Agent
import numpy as np

# widget bar to display progress

import progressbar as pb


class Reinforce:
    def __init__(self, args):
        self.args = args
        
        self.num_episodes = args.NUM_EPISODES
        self.horizon = args.HORIZON
        self.num_parallel_env = args.NUM_PARALLEL_ENV

        env = parallelEnv('PongDeterministic-v4', self.num_parallel_env, seed=12345)
        self.agent = Agent(args, env)
        
    def train(self):
        widget = ['training loop: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
        timer = pb.ProgressBar(widgets=widget, maxval= self.num_episodes).start()
        
        avg_rewards = []
        
        for e in range(self.num_episodes):
            #     print('Running Episode .. ', e)
            # collect trajectories
            rewards = self.agent.learn(self.horizon, self.num_parallel_env, e)
            total_rewards = np.sum(rewards, axis=0)
            
            # get the average reward of the parallel environments
            avg_rewards.append(np.mean(total_rewards))
            
            # display some progress every 20 iterations
            if (e + 1) % 20 == 0:
                print("Episode: {0:d}, score: {1:f}".format(e + 1, np.mean(total_rewards)))
                print(total_rewards)
            
            # update progress widget bar
            timer.update(e + 1)
        
        timer.finish()
        return avg_rewards
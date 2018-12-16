from DeepRL.pong_atari.parallel_env import parallelEnv
from DeepRL.pong_atari.agent import Agent
import numpy as np

# WARNING: running through all 800 episodes will take 30-45 minutes

# training loop max iterations
episode = 500
# episode = 800

# widget bar to display progress

import progressbar as pb

widget = ['training loop: ', pb.Percentage(), ' ',
          pb.Bar(), ' ', pb.ETA()]
timer = pb.ProgressBar(widgets=widget, maxval=episode).start()

# initialize environment
envs = parallelEnv('PongDeterministic-v4', n=8, seed=1234)

discount_rate = .99
beta = .01
tmax = 320

# keep track of progress
mean_rewards = []



agent = Agent(envs)

for e in range(episode):
    print('Running Episode .. ', e)
    # collect trajectories
    rewards = agent.learn(tmax, beta)
    total_rewards = np.sum(rewards, axis=0)
    
    # the regulation term also reduces
    # this reduces exploration in later runs
    beta *= .995
    
    # get the average reward of the parallel environments
    mean_rewards.append(np.mean(total_rewards))
    
    # display some progress every 20 iterations
    if (e + 1) % 20 == 0:
        print("Episode: {0:d}, score: {1:f}".format(e + 1, np.mean(total_rewards)))
        print(total_rewards)
    
    # update progress widget bar
    timer.update(e + 1)

timer.finish()
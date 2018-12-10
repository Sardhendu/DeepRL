
from collections import deque

import numpy as np
from unityagents import UnityEnvironment

from DeepRL.navigation import Agent
from DeepRL.navigation import Config as conf

# TODO: GEt and set the environment
# TODO: IMPLEMENT EPSILON GREEDY POLICY CODE WITH DECAY


env = UnityEnvironment(
    file_name="/Users/sam/All-Program/App/deep-reinforcement-learning/p1_navigation/Banana.app"
)

brain_name = env.brain_names[0]
brain = env.brains[brain_name]




# Loop for each episode
agent = Agent(conf, seed=0)
epsilon = conf.EPSILON

scores = []                        # list containing scores from each episode
scores_window = deque(maxlen=100)  # last 100 scores
for episode in range(0, conf.NUM_EPISODES):
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state
    score = 0  # initialize the score
    # for t in range(0, 1000):#conf.NUM_TIMESTEPS):
    while True:
    #     print(t)
        # Get the action based on current state
        action = agent.act(state, conf.EPSILON)
        env_info = env.step(action)[brain_name]
        state_next = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        # print(state_next, reward, done)
    
        agent.step(state, action, reward, state_next, done)
        
        if done:
            break

    epsilon = max(conf.EPSILON_MIN, conf.EPSILON_DECAY * epsilon)
    scores_window.append(score)  # save most recent score
    scores.append(score)  # save most recent score

    print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="")
    if episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

Collaboration and Competition
-----------

Train multiple agents to collaborate and compete in playing ping-pong.  

![Trained Agent][image1]

### Description:
This project is aimed to make a reinforcement learning multi-agent system that can learn to play ping pong. This repo contains two implementation, 
   * **Competitive Setting:** In this setting we train both the agent separately without sharing information between the two. A simple DDPG (Deep deterministic policy gradient) agent is employed for both of the agent.
   * **Collaborate Setting:** In this setting we train both the agent with independent Actors but with centralized Critic. A centralized critic is trained with states and actions of both the agents. Centralized critic uses knowledge to assist training but are not used is test time. A MADDPG (Multi-agent deep deterministic policy gradient) method is employed to learn in a collaborative environment.
   
### Implemented Functionality:
   * Buffer (Memory): Experience Replay and Priority Experience Replay.
   * Agent: MADDPG for Centralized critic (collaborat) and DDPG for ind
   * Neural Network Architecture: Multilayer Neural network Actor-Critic architecture.
   * Fixed-Q-targets: With soft update and Hard Update 
   * Policy Exploration: Ornstein-Uhlenbeck Noise.
   
#### Getting Started: [Jump to the Report](https://github.com/Sardhendu/DeepRL/blob/master/src/collab_compete/REPORT.md)
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
   
2) **continuous-control-multi.ipynb:** Performance of the Agent for various Hyperparameter-configuration for the multi-agent environment  
3) **config.py:** Hyperparameter configuration. 
4) **main.py:** Environment instantiation for Continuous Control environment. 
5) **agent.py:** MADDPG Agent
6) **model.py:** Actor-Critic neural net architecture
7) **buffer.py: (../src/buffer.py)** Implementation of Experience Replay and Priority Experience ReplayBuffer
8) **exploration.py: (../src/exploration.py)** Uses the Ornstein-Uhlenbeck Noise for policy exploration.
9) **report.md:** Displays the best implemented methods, agent score and future improvements.  

 
### Reference:
* Udacity Deep Reinforcement Learning : [Repository](https://github.com/udacity/deep-reinforcement-learning)
* Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG) [Paper](https://arxiv.org/pdf/1706.02275.pdf)
* Continuous Control with Deep Reinforcement Learning: (DDPG) [Paper](https://arxiv.org/pdf/1509.02971.pdf)
* Deep Reinforcement Learning with Double Q-Learning: [Paper](https://arxiv.org/pdf/1509.06461.pdf) 
* Continuous Control with deep renforcement learning: [Paper](https://arxiv.org/abs/1509.02971)




### TODO's
1. Build my own Unity Environment: [LINK](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Getting-Started-with-Balance-Ball.md)
3. Look at how efficient in Entropy Regularizer

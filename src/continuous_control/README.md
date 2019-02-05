[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"

Continuous Control
-----------

Train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

### Description:
This project is aimed to make a reinforcement learning DQN agent, a double-jointed arm move to the target locations. 

   * **Single Agent environment:** This environment consists of only one agent learning to continuously hit the target location.
   * **Multi Agent environment:** This environment consists of 20 agents simultaneously learning to hit the target location.
   
### Implemented Functionality:
   * Buffer (Memory): Experience Replay and Priority Experience Replay [TODO].
   * Agent: Deep deterministic policy gradient (An Actor-Critic architecture) 
   * Neural Network Architecture: Multilayer Neural network
   (Visual Environment)
   * Fixed-Q-targets: With soft update and Hard Update.
   * Policy Exploration: Ornstein-Uhlenbeck Noise.   

#### Getting Started: [Jump to the Report](https://github.com/Sardhendu/DeepRL/blob/master/src/continuous_control/REPORT.md)
1) **Single Agent Environment:** Download the environment from one of the links below. Select the environment that matches your 
 operating system:
     - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
   
   
2) **Multi Agent Environment:** Download the environment from one of the links below. Select the environment that matches your 
 operating system:
     - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
       
    
3) **continuous-control-multi.ipynb:** Performance of the Agent for various Hyperparameter-configuration for the multi-agent environment  
4) **main.py:** Environment instantiation for Continuous Control environment. 
5) **agent.py:** DDPG Agent
6) **model.py:** Actor-Critic neural net architecture
7) **buffer.py: (../src/buffer.py)** Implementation of Experience Replay and Priority Experience Replay Buffer
7) **exploration.py: (../src/exploration.py)** Uses the Ornstein-Uhlenbeck Noise for policy exploration.
8) **report.md:** Displays the best implemented methods, agent score and future improvements.  

    
### Reference:

* Udacity Deep Reinforcement Learning : [Repository](https://github.com/udacity/deep-reinforcement-learning)
* Human-level control through deep reinforcement learning [Paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
* Prioritized Action Replay: [Paper](https://arxiv.org/pdf/1511.05952.pdf)
* Deep Reinforcement Learning with Double Q-Learning: [Paper](https://arxiv.org/pdf/1509.06461.pdf) 
* Continuous Control with deep renforcement learning: [Paper](https://arxiv.org/abs/1509.02971)


### TODO's
* Use the implemented priority replay buffer.
* Try a large deep network with Actor-Critic heads where Actor and Critics are benefited by parameter sharing.

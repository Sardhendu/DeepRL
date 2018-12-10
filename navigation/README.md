[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

Navigation
-----------

Train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

### Description:
This project is aimed to make a reinforcement learning DQN agent learn to collect yellow banana. This repo contains two implementation, 
   * **Vector environment:** Learning to collect yellow banana using vector states with 37 dimensions that includes agent's velocity, along with ray-based perception of objects around agent's forward direction.
   * **Visual environment:** [Work in progress] Learning to collect yellow banana using image states, where each 
   image is of 84x84x3 
   shape. 
   
### Implemented Functionality:
   * Buffer (Memory): Experience Replay and Priority Experience Replay [TODO]
   * Agent: DQN (Deep Q network) and DDQN (Double Deep Q Network)
   * Neural Network Architecture: Multilayer Neural network (Vector Environemnt) and Convolution Neural Networks 
   (Visual Environment)
   * Fixed-Q-targets: With soft update and Hard Update   

#### Getting Started
1) Download the environment from one of the links below. Select the environment that matches your 
 operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
   (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.
2) **navigation.ipynb:** Performance of the Agent for various Hyperparameter-configuration   
3) **main.py:** Environment instantiation for Vector and Visual environment
4) **agent.py:** Functionality DQN ad DDQN agent
5) **model.py:** The neural net architecture for vector and visual environment
6) **buffer.py:** Implementation of Experience Replay and Priority Action Replay Buffer


#### Results
   
* Target average score to achieve: 13 

   1) **Vector Environment (Basic Model):**
     ![alt text](https://github.com/Sardhendu/DeepRL/blob/master/collect_banana/images/model1_score_plot.png)
     
   2) **Visual Environment:** [TODO]
   
    
### Reference:

* Udacity Deep Reinforcement Learning : [Repository](https://github.com/udacity/deep-reinforcement-learning)
* Human-level control through deep reinforcement learning [Paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
* Prioritized Action Replay: [Paper](https://arxiv.org/pdf/1511.05952.pdf)
* Deep Reinforcement Learning with Double Q-Learning: [Paper](https://arxiv.org/pdf/1509.06461.pdf) 


[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

Navigation
-----------

Train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

### Description:
This project is aimed to make a reinforcement learning DQN agent learn to collect yellow banana. This repo contains two implementation, 
   * **Vector environment:** Learning to collect yellow banana using vector states with 37 dimensions that includes agent's velocity, along with ray-based perception of objects around agent's forward direction.
   * **Visual environment:** Learning to collect yellow banana using image states, where each image is of 84x84x3 shape. 
   
### Implemented Functionality:
   * Buffer (Memory): Experience Replay and Priority Experience Replay
   * Agent: DQN (Deep Q network) and DDQN (Double Deep Q Network)
   * Neural Network Architecture: Multilayer Neural net and Convolution Neural Networks
   * Fixed-Q-targets: With soft update and Hard Update   

#### Get Started
   1) **navigation.ipynb:** Performance of the Agent for different Hyperparameter-configuration   
   2) **main.py:** Environment instantiation for Vector and Visual environment
   3) **agent.py:** Functionality DQN ad DDQN agent
   4) **model.py:** The neural net architecture for vector and visual environment
   5) **buffer.py:** Implementation of Experience Replay and Priority Action Replay Buffer 

#### Results

   1) **Vector Environment (Basic Model):**
   
    
### Reference:

* Udacity Deep Reinforcement Learning : [Repository](https://github.com/udacity/deep-reinforcement-learning)
* Human-level control through deep reinforcement learning [Paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)
* Prioritized Action Replay: [Paper](https://arxiv.org/pdf/1511.05952.pdf)
* Deep Reinforcement Learning with Double Q-Learning: [Paper](https://arxiv.org/pdf/1509.06461.pdf) 


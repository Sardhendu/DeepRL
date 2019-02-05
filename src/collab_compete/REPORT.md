## Multi Agent Deep Deterministic Policy Gradient method (MADDPG):

#### Background Theory: 
-------------

Multi-agent Deep deterministic policy gradient approach: 
   * Traditional methods such as Q-learning are not very efficient with multi-agent because of many reasons. Central to it however is the change in policy every step makes the environment non-stationary from the perspective of an individual agent when information does not flow between the agents.
   
   * In has also been shown that learning independent policy using DDPG perform poor in practise. 
   
   * MADDPG are simple extension to Actor-Critic method where the Critic network is augmented with additional information with policies from all agents. These critics are called as centralized critics.
   
   * Centralized critics are only to boost up learning during training, during testing actor network are used to play the game.
   
   * Using a DDPG Actor-Critic architecture it empowers the agent to learn a continuous policy. 
   
   
 
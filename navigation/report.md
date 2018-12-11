

Navigation (Report)
-----------


The navigation project to train an agent to collect "yellow banana" and avoid "blue banana" was trained for several 
hyperparameter configuration. Below are the description on some of the implemented functionality and there outcomes.


Implementation 
------- 
1. **Fixed-Q-targets:** The agent has two different implementation for Fixed-Q-targets. 
    1) *Soft-update:* In soft update the agent updates the weights of the target-network at each learning step (every 
    4th timestep) with a value of TAU.
        * θ_target = τ*θ_local + (1 - τ)*θ_target
    
    2) *Hard-update:* In hard update the agent updates the weights of the target-network after every t-timstep. 
        * θ_target = θ_local
    
2. **Double-Q-Network:** The double Q-network has a very subtle change over the DQN learning mechanism. To be precise
 in Double-Q-network while learning the action is chosen by the local network but values corresponding to the action 
 are fetched from the target network. 




Vector Environment: 
----

[*View results for other hyperparameter configuration*](https://github.com/Sardhendu/DeepRL/blob/master/navigation/navigation-vector.ipynb)
#### Network Graph:

![alt text](https://github.com/Sardhendu/DeepRL/blob/master/navigation/images/vector_nn.png)

#### Results:

    
   With several model configuration, we found that DQN with Soft-update reaches the target score faster when 
   compared to other mehtods. The banana collection state space is not sufficient (complex enough) to imply if one 
   method is better than another.
   
   * **DQN Soft-update with best Hyper-parameter-configuration:**
        ```python
             {
                NUM_EPISODES = 2000
                NUM_TIMESTEPS = 1000
                
                BUFFER_SIZE = 100000
                BATCH_SIZE = 64
                UPDATE_AFTER_STEP = 4
                
                SOFT_UPDATE = True
                TAU = 0.001                 # Soft update parameter for target_network
                
                GAMMA = 0.99                # Discount value
                EPSILON = 1                 # Epsilon value for action selection
                EPSILON_DECAY = 0.995       # Epsilon decay for epsilon greedy policy
                EPSILON_MIN = 0.01          # Minimum epsilon to reach
                
                LEARNING_RATE = 0.0005  # Learning rate for the network
                
                Q_LEARNING_TYPE = 'dqn' # dqn also available, ddqn is double dqn
           }

             Episode 100	Average Score: 1.01
             Episode 200	Average Score: 3.73
             Episode 300	Average Score: 7.62
             Episode 400	Average Score: 10.37
             Episode 488	Average Score: 12.94
             Environment solved in 489 episodes!	Average Score: 13.01
         ```    
            
   ![alt text](https://github.com/Sardhendu/DeepRL/blob/master/navigation/images/model1_score_plot.png)
      
   * **DQN Hard-update with best Hyper-parameter-configuration:**    
   
       ```python
             {
                NUM_EPISODES = 2000
                NUM_TIMESTEPS = 1000
                
                BUFFER_SIZE = 100000
                BATCH_SIZE = 64
                
                SOFT_UPDATE = False
                TAU = 0.001                 # Soft update parameter for target_network
           
                HARD_UPDATE = True
                HARD_UPDATE_FREQUENCY = 500
                
                GAMMA = 0.99                # Discount value
                EPSILON = 1                 # Epsilon value for action selection
                EPSILON_DECAY = 0.995       # Epsilon decay for epsilon greedy policy
                EPSILON_MIN = 0.01          # Minimum epsilon to reach
                
                LEARNING_RATE = 0.0005  # Learning rate for the network
                
                Q_LEARNING_TYPE = 'dqn' # dqn also available, ddqn is double dqn
             }
            
             Episode 100	Average Score: 1.05
             Episode 200	Average Score: 3.94
             Episode 300	Average Score: 7.66
             Episode 400	Average Score: 9.06
             Episode 500	Average Score: 9.70
             Episode 600	Average Score: 12.11
             Episode 629	Average Score: 12.97
             Environment solved in 630 episodes!	Average Score: 13.00
         ```    
           
   ![alt text](https://github.com/Sardhendu/DeepRL/blob/master/navigation/images/model8_score_plot.png)   
         
    
Visual Environment:
----

WORK IN PROGRESS




Ideas for future improvement:
-----
* Implement prioritized experience replay buffer. [In Progress] 
* Implement Dueling network architecture.
* Experiment with different Network architectures.
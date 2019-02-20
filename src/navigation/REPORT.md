

Navigation (Report)
-----------


The navigation project to train an agent to collect "yellow banana" and avoid "blue banana" was trained for several 
hyperparameter configuration. Below are the description on some of the implemented functionality and there outcomes.


Implementations 
------- 
1. **Fixed-Q-targets:** The agent has two different implementation for Fixed-Q-targets. 

   In normal scenario we use the local network weights (w) to find the expected Q-value. 
            
      * expected Q(s, a) = [wX + b]<sub>local_network</sub>
      
   According to Bellman equation the target Q-values are computed by current reward and a discounted return.
   
      * Q(s, a) = reward(s, a) + gamma * max_a [Q(s', a)<sub>local_network</sub>] <sub>local_network</sub>
      
    Using Q(s, a)<sub>local_network</sub> to find target value is not optimal because we would be using same parameters w<sub>local_network</sub> to for target Q(s, a) and expected Q(s, a) which would have high correlation between the TD target and the weights w<sub>local_network</sub> we are learning. So instead, we use the target network parameters w<sub>target_network</sub> to compute the target Q(s, a) :  
       
      * target Q(s, a) = reward(s, a) + gamma * max_a [Q(s', a)<sub>target_network</sub>] <sub>target_network</sub>

   1) *Soft-update:* In soft update the agent updates the weights of the target-network at each learning step (every 
    4th timestep) with a value of TAU.
        * θ_target = τ*θ_local + (1 - τ)*θ_target
        
   2) *Hard-update:* In hard update the agent updates the weights of the target-network after every t-timstep. 
        * θ_target = θ_local
        
      
2. **Double-Q-Network:** The double Q-network has a very subtle change over the DQN learning mechanism. In vanilla DQN with Fixed-Q-target we use one network (target_network) to select and evaluate an action. This can potentially lead to overoptimistic value estimates (since its the same network). Inorder to mitigate this Double-Q-network was introduced, that uses one network (local network) to choose the action and uses another network (target network) to evaluate actions. 

   The change here is :
   
   * target Q(s, a) = reward(s, a) + gamma * Q(s', argmax<sub>a</sub>Q(s', a)<sub>local_network</sub>)
   <sub>target_network</sub>
 
3. **Priority-Experience-Replay Buffer:** The idea for a Priority experience replay buffer is to sample experiences 
with higher TD error more often. Some experiences are more important than others. Since the experience replay buffer 
size is limited, these rare experiences might just get deleted without contributing to the agent's learning. 
Therefore, the priority experience replay helps in weighting (providing probabilities of sampling) to all experiences
 based on their sampling frequency and td-error (The more the td-error, the more is to learn from that experience). 

Vector Environment: 
----

[*View results for other hyperparameter configuration*](https://github.com/Sardhendu/DeepRL/blob/master/navigation/navigation-vector.ipynb)

#### Network Graph:

![alt text](https://github.com/Sardhendu/DeepRL/blob/master/src/navigation/images/vector_nn.png)

#### Results:

    
   With several model configuration, we found that DQN with Soft-update reaches the target score faster when 
   compared to other mehtods. The banana collection state space is not sufficient (complex enough) to imply if one 
   method is better than another.
   
   * **DQN Soft-update with best Hyper-parameter-configuration:**
        ```python
             {
                SEED = 0
                STATE_SIZE = 37
                ACTION_SIZE = 4
                NUM_AGENTS = 1
                BUFFER_SIZE = 10000  # int(1e05)
                BATCH_SIZE = 64
                NUM_EPISODES = 2000
                NUM_TIMESTEPS = 1000

                # Agent Params
                TAU = 0.001
                WEIGHT_DECAY = 0.0
                IS_HARD_UPDATE = False
                IS_SOFT_UPDATE = True
                SOFT_UPDATE_FREQUENCY = 4
                HARD_UPDATE_FREQUENCY = 2000

                # Model parameters
                AGENT_LEARNING_RATE = 0.0005
                DATA_TO_BUFFER_BEFORE_LEARNING = 64
                GAMMA = 0.99
                LEARNING_FREQUENCY = 4

                # Exploration parameter (Since this is a discrete task we have to use a discrete policy exploration 1.e epsilon
                # greedy)
                EPSILON_GREEDY = lambda: EpsilonGreedy(
                        epsilon_init=1, epsilon_min=0.01, decay_value=0.99, decay_after_step=300, seed=0
                )  # Note decay_after_steps should approximately be equal to the num_timesteps in one episode

                # NETWORK PARAMETERS
                Q_LEARNING_TYPE = 'dqn'  # available values = {'dqn', 'dbl_dqn'}
                LOCAL_NETWORK = lambda: model.QNetwork(
                        TrainVectorConfig.STATE_SIZE, TrainVectorConfig.ACTION_SIZE, TrainVectorConfig.SEED, network_name='net2'
                ).to(device)
                TARGET_NETWORK = lambda: model.QNetwork(
                        TrainVectorConfig.STATE_SIZE, TrainVectorConfig.ACTION_SIZE, TrainVectorConfig.SEED, network_name='net2'
                ).to(device)
                OPTIMIZER_FN = lambda params: torch.optim.Adam(params, lr=TrainVectorConfig.AGENT_LEARNING_RATE)
           }

            Episode 100	Average Score: 2.05
            Episode 200	Average Score: 8.03
            Episode 300	Average Score: 10.56
            Episode 400	Average Score: 12.09
            Episode 436	Average Score: 13.00
            Environment solved in 436 episodes!	Average Score: 13.00

         ```    
            
   ![alt text](https://github.com/Sardhendu/DeepRL/blob/master/src/navigation/images/model1_score_plot.png)
      
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
           
   ![alt text](https://github.com/Sardhendu/DeepRL/blob/master/src/navigation/images/model8_score_plot.png)   
         
    
Visual Environment:
----

WORK IN PROGRESS




Ideas for future improvement:
-----
* Implement Dueling network architecture.
* Experiment with different Network architectures.
## Deep Deterministic Policy Gradient method (DDPG):

#### Background Theory:
-------------
**On-Policy and Off-Policy**:


**Policy Gradient and Monte-Carlo**:
Monte-Carlo methods have high variance because they take the expectation of the state-action value across many episodes and these values can vary too much. These methods have low bias because there are lots of random event that takes place within each episode. Policy gradient relates to Monte-carlo approximation where random events are drawn forming a trajectory following a policy.

**DQN (Deep-Q-Network) and Temporal Difference**:
Temporal Difference methods are estimate of estimates. In TD methods we approximate the value-function using cumulative return (which is also an estimate) for every time-step. This leaves very little room for the value function to diverge, and hence has low variance. These methods, however can have high bias because if the first few estimates are not correct or way too off from the correct path then it gets difficult for TD-methods to get on track, hence resulting in high bias.

**Actor-Critic methods**:
Actor-Critic method make the best out of both. In uses a baseline network similar to policy gradient method and maintain low bias while evaluates the action-value using a DQN like network to check high variance posed by the baseline networks. In other words Actor-Critic learning algorithms are used to represent the policy function independently of the value function, where the actor is responsible for the policy and the Critic is responsible for the value function.

**DDPG (Deep Deterministic Policy Gradient)**:
In DQN, the network outputs an action-value function Q(s, a) for a discrete action, and we use argmax of the action-value function following an epsilon greedy policy to select an action. What if the action to be selected is not discrete but continuous. Imagine a pendulum moving 10 degree to the right. Taking argmax wont work, rather we have to predict the continuous value for the action to take. This problem is actually solved using DDPG. DDPG is an off-policy learning like DQN and has actor and critic network architecture. In other works DDPG is a neat way to generalize DQN for continuous action spaces while having a Actor-Critic architecture. The input of the actor network is the current state, and the output is a single real value representing an action chosen from a continuous action space. The critic’s output is simply the estimated Q-value of the current state and of the action given by the actor. Since DDPG is an off-policy learning algorithm, we can use large Replay Buffer to benefit from learning across a set of uncorrelated events.

**Actor (θμ)**: Approximate policy deterministically (note its not stochastic - no probability distribution) μ(st | θμ) = argmax(a)Q(s,a)

**Critic (θq)**: Evaluates the optimal action value function by using the actor best believed action: Q(s, μ(st;θμ) | θq)


#### Network Graph:

###### ACTOR:
![alt text](https://github.com/Sardhendu/DeepRL/blob/master/src/continuous_control/images/actor.png)
###### CRITIC:
![alt text](https://github.com/Sardhendu/DeepRL/blob/master/src/continuous_control/images/critic.png)

<div id="image-table">
    <table>
	    <tr>
    	    <td style="padding:5px">
        	    <img src="https://github.com/Sardhendu/DeepRL/blob/master/src/continuous_control/images/actor.png" width="300" height="400"><figcaption><center>Actor Network</center></figcaption>
      	    </td>
             <td style="padding:5px">
            	<img src="https://github.com/Sardhendu/DeepRL/blob/master/src/continuous_control/images/critic.png" width="400" height="400"><figcaption></center>Critic Network </center></figcaption>
             </td>
        </tr>
    </table>
</div>


#### Results:

    
   With several model configuration, we found that DQN with Soft-update reaches the target score faster when 
   compared to other mehtods. The banana collection state space is not sufficient (complex enough) to imply if one 
   method is better than another.
   
   * **DQN Soft-update with best Hyper-parameter-configuration:**
        ```python
             {
                class Config:
                import os
                # ENVIRONMEMT PARAMETER
                STATE_SIZE = 33
                ACTION_SIZE = 4
                NUM_EPISODES = 2000
                NUM_TIMESTEPS = 1000
                
                # MODEL PARAMETERS
                SEED = 0
                BUFFER_SIZE = int(1e05)
                BATCH_SIZE = 512
                
                # Exploration parameter
                NOISE = True
                EPSILON_GREEDY = False
                EPSILON = 1
                EPSILON_DECAY = 0.995  # Epsilon decay for epsilon greedy policy
                EPSILON_MIN = 0.01  # Minimum epsilon to reach
                
                if (NOISE and EPSILON_GREEDY) or (not NOISE and not EPSILON_GREEDY):
                    raise ValueError('Only one exploration policy either NOISE or EPSILON_GREEDY si to be chosen ..')
                
                # LEARNING PARAMETERS
                ACTOR_LEARNING_RATE = 0.0001
                CRITIC_LEARNING_RATE = 0.0005
                GAMMA = 0.99  # Discounts
                LEARNING_FREQUENCY = 4
                
                # WEIGHTS UPDATE PARAMENTER
                SOFT_UPDATE = True
                TAU = 0.001  # Soft update parameter for target_network
                SOFT_UPDATE_FREQUENCY = 4
                DECAY_TAU = False
                TAU_DECAY_RATE = 0.003
                TAU_MIN = 0.05
                
                HARD_UPDATE = False
                HARD_UPDATE_FREQUENCY = 1000
                
                if (SOFT_UPDATE and HARD_UPDATE) or (not SOFT_UPDATE and not HARD_UPDATE):
                    raise ValueError('Only one of Hard Update and Soft Update is to be chosen ..')
                
                if SOFT_UPDATE_FREQUENCY < LEARNING_FREQUENCY:
                    raise ValueError('Soft update frequency can not be smaller than the learning frequency')
                
                # Lambda Functions:
                EXPLORATION_POLICY_FN = lambda: OUNoise(size=Config.ACTION_SIZE, seed=2)
                ACTOR_NETWORK_FN = lambda: Actor(Config.ACTION_SIZE, Config.STATE_SIZE, (512, 256), seed=2).to(
                        device)  # lambda: Actor(Config.STATE_SIZE, Config.ACTION_SIZE, seed=2, fc1_units=512, fc2_units=256).to(
                # device)
                ACTOR_OPTIMIZER_FN = lambda params: torch.optim.Adam(params, lr=Config.ACTOR_LEARNING_RATE)
                
                CRITIC_NETWORK_FN = lambda: Critic(Config.ACTION_SIZE, Config.STATE_SIZE, (512, 256), seed=2).to(
                        device)  # lambda: Critic(Config.STATE_SIZE, Config.ACTION_SIZE, seed=2, fc1_units=512, fc2_units=256).to(
                # device)
                CRITIC_OPTIMIZER_FN = lambda params: torch.optim.Adam(params, lr=Config.CRITIC_LEARNING_RATE)
                
                MEMORY_FN = lambda: MemoryER(Config.BUFFER_SIZE, Config.BATCH_SIZE, seed=2, action_dtype='float')
                
                # USE PATH
                MODEL_NAME = 'model_1'
                model_dir =  pth + '/models'
                base_dir = os.path.join(model_dir, 'continuous_control', '%s' % (MODEL_NAME))
                if not os.path.exists(base_dir):
                    print('creating .... ', base_dir)
                    os.makedirs(base_dir)
                #
                STATS_JSON_PATH = os.path.join(base_dir, 'stats.json')
                CHECKPOINT_DIR = base_dir
            
                       }
            
                         Episode 100	Average Score: 1.01
                         Episode 200	Average Score: 3.73
                         Episode 300	Average Score: 7.62
                         Episode 400	Average Score: 10.37
                         Episode 488	Average Score: 12.94
                         Environment solved in 489 episodes!	Average Score: 13.01
         ```    
            
   ![alt text](https://github.com/Sardhendu/DeepRL/blob/master/src/continuous_control/images/score.png)
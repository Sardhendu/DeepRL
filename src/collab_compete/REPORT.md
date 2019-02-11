## Multi Agent Deep Deterministic Policy Gradient method (MADDPG):

#### Background Theory: 
-------------

Multi-agent Deep deterministic policy gradient approach: 
   * Traditional methods such as Q-learning are not very efficient with multi-agent because of many reasons. Central to it however is the change in policy every step makes the environment non-stationary from the perspective of an individual agent when information does not flow between the agents.
   
   * In has also been shown that learning independent policy using DDPG perform poor in practise. 
   
   * MADDPG are simple extension to Actor-Critic method where the Critic network is augmented with additional information with policies from all agents. These critics are called as centralized critics.
   
   * Centralized critics are only to boost up learning during training, during testing actor network are used to play the game.
   
   * Using a DDPG Actor-Critic architecture it empowers the agent to learn a continuous policy. 
   
   
#### Basic aglorithm functionality.
   * First we sample experiences of batch_size for all the agents (Here we have 2),
        An Experience (for 2 agents):
        
        --> Experience = (experience1, experience2)
        
        --> Sample for agent_1
            experience1: (state, action, reward, next_state, done)
                * state  = (batch_size, num_agents, state_size)
                * action = (batch_size, num_agents, action_size)
                * reward = (batch_size, num_agents, 1)
        
        --> Sample for agent_2  
            experience2: (state, action, reward, next_state, done)
                * state  = (batch_size, num_agents, state_size)
                * action = (batch_size, num_agents, action_size)
                * reward = (batch_size, num_agents, 1)
                
   * For the actor model
                
                
        Why do we take experiences for both agent separately, we could do the same by using only 1 sampled experience? Because if we use one 
                
   

Vector Environment: 
----

#### Network Graph:

###### ACTOR-CRITIC:

<div id="image-table">
    <table>
	    <tr>
    	    <td style="padding:5px">
        	    <img src="https://github.com/Sardhendu/DeepRL/blob/master/src/collab_compete/images/actor_.png" width="400" height="400"><figcaption><center>Actor Network</center></figcaption>
      	    </td>
             <td style="padding:5px">
            	<img src="https://github.com/Sardhendu/DeepRL/blob/master/src/collab_compete/images/critic_.png" width="400" height="400"><figcaption></center>Critic Network </center></figcaption>
             </td>
        </tr>
    </table>
</div>



### Mysterious Findings:
1. This is a multiagent task, where the agent may not learn anything till 2000 episode. Be patient!

2. The agent could show a combined score of > 0.5 for more that 100 consecutive episode but there can be scenarios that the score decreases after that, which means the agent learning is unstable.

3. **Setting 1**: I had written a redundant code, I repeated "action += self.noise_val" two times, something like.
     * action += self.noise_val
     * action += self.noise_val
        
   and I saw my learning converge faster in 1400 (The above result) was the result of this redundancy. This means when the action in negative this makes it more negative and when the action is positive, we make it more positive.
   
   So I added the action 3 times and then then the score decreases way too much. Well this makes sense because 3 times is way too much.
   
   Also, this particular setting was difficult to obtain by other small changes to hyperparameters
   
4. **Setting 2**: Keeping the above mistake in mind it was evident that the agent was benifited with low noise as noise starting at 1 or 2 and decayed slow and fast didn't help. Uisng low noise at 0.5 and further reducing it actually helped.

5. **Setting 3**: The agent learning changes drastically with number of floating points, **Expected but strange** Based on "setting 2", adding noise 2 times to action was equivallent of increasing the noise_amplitude by *2 which is **0.5*2 = 1.0** OR having **action += 2*(self.noise)**. Recording my experiments the results for both the cases was same but differed with setting 2 based on floating point. Below was the difference. These were the print statements.


        Action value after Double noise (amplitude start at 1)=  :  [[0.38111384 0.03658604]]
        Action value after Double noise (setting 2)              :  [[0.38111383 0.03658604]]
        
        Action value after Double noise (amplitude start at 1)=  :  [[0.22289944 0.40524558]]
        Action value after Double noise (setting 2)              :  [[0.22289944 0.40524557]]
        
        Action value after Double noise (amplitude start at 1)=  :  [[ 0.70455045 -0.17316329]]
        Action value after Double noise (setting 2)              :  [[ 0.70455045 -0.1731633 ]]
        
        Action value after Double noise (amplitude start at 1)=  :  [[0.38526529 0.30423273]]
        Action value after Double noise (setting 2)              :  [[0.3852653  0.30423272]]
        
        Action value after Double noise (amplitude start at 1)=  :  [[ 0.58545646 -0.07782507]]
        Action value after Double noise (setting 2)              :  [[ 0.58545643 -0.07782507]]
        
        Action value after Double noise (amplitude start at 1)=  :  [[0.3640666  0.54019544]]
        Action value after Double noise (setting 2)              :  [[0.3640666  0.54019547]]
        
        Action value after Double noise (amplitude start at 1)=  :  [[ 0.65132615 -0.04964975]]
        Action value after Double noise (setting 2)              :  [[ 0.6513261  -0.04964975]]
        
        Action value after Double noise (amplitude start at 1)=  :  [[0.40193518 0.51675377]]
        Action value after Double noise (setting 2)              :  [[0.4019352 0.5167538]]
        
        Action value after Double noise (amplitude start at 1)=  :  [[ 0.85895494 -0.0934009 ]]
        Action value after Double noise (setting 2)              :  [[ 0.8589549 -0.0934009]]
        
        Action value after Double noise (amplitude start at 1)=  :  [[0.412575   0.25896035]]
        Action value after Double noise (setting 2)              :  [[0.412575   0.25896037]]
        
        Action value after Double noise (amplitude start at 1)=  :  [[0.22713706 0.04187984]]
        Action value after Double noise (setting 2)              :  [[0.22713704 0.04187984]]

## Results:

   **Setting 1**: With several model configuration, we found that MADDPG with frequent Soft-update but lesser TAU reaches the target score. **The environment is solved in 1450 episode**
   
   * **Hyper-parameter-configuration:**
        ```python
             {
                SEED = 0
                STATE_SIZE = 24
                ACTION_SIZE = 2
                NUM_AGENTS = 2
                BUFFER_SIZE = 10000
                BATCH_SIZE = 256
                
                # Agent Params
                TAU = 1e-3
                WEIGHT_DECAY = 0.0
                IS_HARD_UPDATE = False
                IS_SOFT_UPDATE = True
                SOFT_UPDATE_FREQUENCY = 2
                HARD_UPDATE_FREQUENCY = 2000
                
                
                # Model parameters
                ACTOR_LEARNING_RATE = 1e-4
                CRITIC_LEARNING_RATE = 1e-3
                DATA_TO_BUFFER_BEFORE_LEARNING = 256
                GAMMA = 0.99
                LEARNING_FREQUENCY = 2
            
                # Exploration parameter
                NOISE_FN = lambda: OUNoise(size=2, seed=0)  # (ACTION_SIZE, SEED_VAL)
                NOISE_AMPLITUDE_DECAY_FN = lambda: utils.Decay(
                        decay_type='multiplicative',
                        alpha=0.5, decay_rate=1, min_value=0.25,
                        start_decay_after_step=256,
                        decay_after_every_step=150,   
                        decay_to_zero_after_step=30000
                )
             }
        
            Episode 200	Average Score: 0.024
            Episode 400	Average Score: 0.010
            Episode 600	Average Score: 0.067
            Episode 800	Average Score: 0.097
            Episode 1000	Average Score: 0.312
            Episode 1200	Average Score: 0.215
            Episode 1400	Average Score: 0.294
            Episode 1490	Average Score: 0.610
     ```   
    
   Episodic Scores:   
            
   ![alt text](https://github.com/Sardhendu/DeepRL/blob/master/src/collab_compete/images/scores.png)
   
   
   
   
   **Setting 2**: **The environment is solved in ~1000 episodes** and reached a score of ~0.7 at 1100 episodes **GREAT**. This further proves **MANAGING THE WAY YOUR AGENT EXPLORES IS OF PARAMOUNT IMPORTANCE**
   
   * **Hyper-parameter-configuration:**
        ```python
             {
                SEED = 0
                STATE_SIZE = 24
                ACTION_SIZE = 2
                NUM_AGENTS = 2
                BUFFER_SIZE = 10000
                BATCH_SIZE = 256
                
                # Agent Params
                TAU = 1e-3
                WEIGHT_DECAY = 0.0
                IS_HARD_UPDATE = False
                IS_SOFT_UPDATE = True
                SOFT_UPDATE_FREQUENCY = 2
                HARD_UPDATE_FREQUENCY = 2000
                
                
                # Model parameters
                ACTOR_LEARNING_RATE = 1e-4
                CRITIC_LEARNING_RATE = 1e-3
                DATA_TO_BUFFER_BEFORE_LEARNING = 256
                GAMMA = 0.99
                LEARNING_FREQUENCY = 2
            
                # Exploration parameter
                NOISE_FN = lambda: OUNoise(size=2, seed=0)  # (ACTION_SIZE, SEED_VAL)
                NOISE_AMPLITUDE_DECAY_FN = lambda: utils.Decay(
                        decay_type='multiplicative',
                        alpha=0.5, decay_rate=0.995, min_value=0.25,
                        start_decay_after_step=15000,
                        decay_after_every_step=100,
                        decay_to_zero_after_step=30000
                )
             }
        
             # The other change was -  Instead of having action += self.noise_val once, we had defined it twice as:
                 action += self.noise_val
                 action += self.noise_val
             
        
            [Actor] Initializing the Actor network ..... 
            [Actor] Initializing the Actor network ..... 
            [Actor] Initializing the Actor network ..... 
            [INIT] Initializing Experience Replay Buffer .... .... ....
            Episode 200	Average Score: 0.024
            Episode 400	Average Score: 0.010
            Episode 600	Average Score: 0.067
            Episode 800	Average Score: 0.095
            Episode 1000	Average Score: 0.407
            Episode 1029	Average Score: 0.702
            Environment solved in 1029 episodes!	Average Score: 0.702
     ```   
    
   Episodic Scores:   
            
   <div id="image-table">
    <table>
	    <tr>
    	    <td style="padding:5px">
        	    <img src="https://github.com/Sardhendu/DeepRL/blob/master/src/collab_compete/images/agent0_s2_ls.png" width="400" height="400"><figcaption><center>Agent-0</center></figcaption>
      	    </td>
             <td style="padding:5px">
            	<img src="https://github.com/Sardhendu/DeepRL/blob/master/src/collab_compete/images/agent1_s2_ls.png" width="400" height="400"><figcaption></center>Agent-1 </center></figcaption>
             </td>
        </tr>
    </table>
    </div>
   
   ![alt text](https://github.com/Sardhendu/DeepRL/blob/master/src/collab_compete/images/agent01_s2_score.png)
   
    
    
    
   **Setting 3**: **The environment is solved in ~2600 episodes** and reached a score of ~0.7 at 2700 episodes **GREAT**. But as stated change in floating point value increased the num_of_episodes by 1600 to reach the score of 0.7
   
   * **Hyper-parameter-configuration:**
        ```python
             {
                SEED = 0
                STATE_SIZE = 24
                ACTION_SIZE = 2
                NUM_AGENTS = 2
                BUFFER_SIZE = 10000
                BATCH_SIZE = 256
                
                # Agent Params
                TAU = 1e-3
                WEIGHT_DECAY = 0.0
                IS_HARD_UPDATE = False
                IS_SOFT_UPDATE = True
                SOFT_UPDATE_FREQUENCY = 2
                HARD_UPDATE_FREQUENCY = 2000
                
                
                # Model parameters
                ACTOR_LEARNING_RATE = 1e-4
                CRITIC_LEARNING_RATE = 1e-3
                DATA_TO_BUFFER_BEFORE_LEARNING = 256
                GAMMA = 0.99
                LEARNING_FREQUENCY = 2
            
                # Exploration parameter
                NOISE_FN = lambda: OUNoise(size=2, seed=0)  # (ACTION_SIZE, SEED_VAL)
                NOISE_AMPLITUDE_DECAY_FN = lambda: utils.Decay(
                        decay_type='multiplicative',
                        alpha=1, decay_rate=0.995, min_value=0.25,
                        start_decay_after_step=15000,
                        decay_after_every_step=100,
                        decay_to_zero_after_step=30000
                )
             }
        
            [Actor] Initializing the Actor network ..... 
            [Actor] Initializing the Actor network ..... 
            [Actor] Initializing the Actor network ..... 
            [INIT] Initializing Experience Replay Buffer .... .... ....
            Episode 200	Average Score: 0.011
            Episode 400	Average Score: 0.001
            Episode 600	Average Score: 0.019
            Episode 800	Average Score: 0.037
            Episode 1000	Average Score: 0.051
            Episode 1200	Average Score: 0.109
            Episode 1400	Average Score: 0.116
            Episode 1600	Average Score: 0.150
            Episode 1800	Average Score: 0.197
            Episode 2000	Average Score: 0.289
            Episode 2200	Average Score: 0.454
            Episode 2400	Average Score: 0.214
            Episode 2600	Average Score: 0.445
            Episode 2691	Average Score: 0.701
            Environment solved in 2691 episodes!	Average Score: 0.701
     ```   
    
   Episodic Scores:   
            
   <div id="image-table">
    <table>
	    <tr>
    	    <td style="padding:5px">
        	    <img src="https://github.com/Sardhendu/DeepRL/blob/master/src/collab_compete/images/agent0_s3_ls.png" width="400" height="400"><figcaption><center>Agent-0</center></figcaption>
      	    </td>
             <td style="padding:5px">
            	<img src="https://github.com/Sardhendu/DeepRL/blob/master/src/collab_compete/images/agent1_s3_ls.png" width="400" height="400"><figcaption></center>Agent-1 </center></figcaption>
             </td>
        </tr>
    </table>
    </div>
   
   ![alt text](https://github.com/Sardhendu/DeepRL/blob/master/src/collab_compete/images/agent01_s3_score.png)
         
  
### Future Work
1. Tune Hyperparameter to make learning more stable.
2. Implement Priority Sampling
3. Implement Competitive setting for the agent and compare results with the cooperative setting 
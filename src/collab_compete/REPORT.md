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
        action += self.noise_val
        action += self.noise_val
        
   and I saw my learning converge faster in 1400 (The above result) was the result of this redundancy. This means when the action in negative this makes it more negative and when the action is positive, we make it more positive.
   
   So I added the action 3 times and then then the score decreases way too much. Well this makes sense because 3 times is way too much.
   
   Also, this particular setting was difficult to obtain by other small changes to hyperparameters
4. **Setting 2**: Keeping the above mistake in mind it was evident that the agent was benifited with low noise as noise starting at 1 or 2 and decayed gradualy slow and fast didnt help. Uisng low noise at 0.5 and further reducing it actually helped. 



#### Results:

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
        	    <img src="https://github.com/Sardhendu/DeepRL/blob/master/src/collab_compete/images/agent0_best.png" width="400" height="400"><figcaption><center>Agent-0</center></figcaption>
      	    </td>
             <td style="padding:5px">
            	<img src="https://github.com/Sardhendu/DeepRL/blob/master/src/collab_compete/images/agent1_best.png" width="400" height="400"><figcaption></center>Agent-1 </center></figcaption>
             </td>
        </tr>
    </table>
    </div>
   
   ![alt text](https://github.com/Sardhendu/DeepRL/blob/master/src/collab_compete/images/avg_score_best.png)
   
    
         
   

### Findings:
1. Increasing the soft-update iteration =32 and soft update TAU from 0.001 to 0.01 helped with some spikes in the few episodes. Would hard update be beneficial?
2. Having HARD-update at 1000 instead of soft-update actually helped. agent_1 which wasnt learning anything performed better than agent_0. THere were multiple spikes where the average score over 100 consecutive window was more that 0.04 which was great given the fact that the score were always closer to -0.004. This was really a good trial after a long time.
3. After converting the code to MADDPG with Centralized critic learning hard update didnt perform as before (This is weird). However soft-update with Tau-0.01 and soft-update after 32 iteration, while learning was set at every 16 iteration helped the model learn great amount amid the training where the score reached an average of 0.08 - 0.1 from (episode 400-1200). But then after many iteration the score dropped.
   -> Was this because we have noise decay till 0.001, should we decay noise all the way to 0.000001.
   -> Should we decrease the TAU = 0.001 and let model shift slowely towards the target.
4. Noise amplitude decay from 1 to 0.25 with soft-update after every 15 steps didn't help much. Agent average reqard fluctuated from 0 to 0.05. Some iteration it learned and some it didn't. Also this setting had softupdate after every 15 iterations with TAU 0.001. 
5. Changing the above setting by having constant noise decay (0.5) over 30,000 iteration controlled fluctuation but the output was still not very appeasing. light fluctuation between 0 to 0.07
   
  
### Future Work
1. Tune Hyperparameter to make learning more stable.
2. Implement Priority Sampling
3. Implement Competitive setting for the agent and compare results with the cooperative setting 
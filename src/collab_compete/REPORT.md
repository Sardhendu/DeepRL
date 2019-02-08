## Multi Agent Deep Deterministic Policy Gradient method (MADDPG):

#### Background Theory: 
-------------

Multi-agent Deep deterministic policy gradient approach: 
   * Traditional methods such as Q-learning are not very efficient with multi-agent because of many reasons. Central to it however is the change in policy every step makes the environment non-stationary from the perspective of an individual agent when information does not flow between the agents.
   
   * In has also been shown that learning independent policy using DDPG perform poor in practise. 
   
   * MADDPG are simple extension to Actor-Critic method where the Critic network is augmented with additional information with policies from all agents. These critics are called as centralized critics.
   
   * Centralized critics are only to boost up learning during training, during testing actor network are used to play the game.
   
   * Using a DDPG Actor-Critic architecture it empowers the agent to learn a continuous policy. 
   
   

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


#### Results:

   With several model configuration, we found that MADDPG with frequent Soft-update but lesser TAU reaches the target score. The learning seemed pretty unstable, with score rising up slowely till a certain point and then falling to near 0. **The environment is solved in 1200 episode**
   
   * **Hyper-parameter-configuration:**
        ```python
             {
                STATE_SIZE = 24
                ACTION_SIZE = 2
                NUM_AGENTS = 2
                NUM_EPISODES = 1000
                NUM_TIMESTEPS = 1000
                #
                # # MODEL PARAMETERS
                # SEED = 0
                BUFFER_SIZE = int(1e04)
                BATCH_SIZE = 256
                DATA_TO_BUFFER_BEFORE_LEARNING = 256
                
                
                # LEARNING PARAMETERS
                ACTOR_LEARNING_RATE = 0.0001
                CRITIC_LEARNING_RATE = 0.001
                GAMMA = 0.998  # Discounts
                LEARNING_FREQUENCY = 2
                
                # WEIGHTS UPDATE PARAMETERS
                SOFT_UPDATE = True
                TAU = 0.0001  # Soft update parameter for target_network
                SOFT_UPDATE_FREQUENCY = 2  # 2
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
    
         
   
### Caution:
1. This is a multiagent task, where the agent may not learn anything till 2000 episode. Be patient!
2. The agent could show a combined score of > 0.5 for more that 100 consecutive episode but there can be scenarios 
that the score decreases after that, which means the agent learning is unstable.

### Findings:
1. Increasing the softupdate iteration =32 and soft update TAU from 0.001 to 0.01 helped with some spikes in the few episodes. Would hard update be beneficial?
2. Having HARD-update at 1000 instead of soft-update actually helped. agent_1 which wasnt learning anything performed better than agent_0. THere were multiple spikes where the average score over 100 consecutive window was more that 0.04 which was great given the fact that the score were always closer to -0.004. This was really a good trial after a long time.
3. After converting the code to MADDPG with Centralized critic learning hard update didnt perform as before (This is weird). However soft-update with Tau-0.01 and soft-update after 32 iteration, while learning was set at every 16 iteration helped the model learn great amount amid the training where the score reached an average of 0.08 - 0.1 from (episode 400-1200). But then after many iteration the score dropped.
   -> Was this becasue we have noise decay till 0.001, should we decay noise all the way to 0.000001.
   -> Should we decrease the TAU = 0.001 and let model shift slowely towards the target.
4. Noise amplitude decay from 1 to 0.25 with soft-update after every 15 steps didn't help much. Agent average reqard fluctuated from 0 to 0.05. Some iteration it learned and some it didn't. Also this setting had softupdate after every 15 iterations with TAU 0.001. 
5. Changint the above setting by having constant noise decay (0.5) over 30,000 iteration controlled fluctuation but the output was still not very appeasing. light fluctuation between 0 to 0.07
   
  
### Future Work
1. Tune Hyperparameter to make learning more stable.
2. Implement Priority Sampling
3. Implement Competitive setting for the agent and compare results with the cooperative setting 
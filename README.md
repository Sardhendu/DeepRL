

# Implemented Algorithms:

####* Deep-Q-Network (DQN) and Double-DQN [HERE](https://github.com/Sardhendu/DeepRL/blob/master/src/navigation/agent.py)   

####* REINFORCE - Policy gradient [HERE](https://github.com/Sardhendu/DeepRL/blob/master/src/pong_atari/agent.py)

####* Deep Deterministic Policy Gradient (DDPG) [HERE](https://github.com/Sardhendu/DeepRL/blob/master/src/continuous_control/agent.py)

####* Multi Agent Deep Deterministic Policy Gradient (MADDPG) [HERE](https://github.com/Sardhendu/DeepRL/blob/master/src/collab_compete/agent.py) 
    

[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

Project 1: Navigation [*Link*](https://github.com/Sardhendu/DeepRL/tree/master/src/navigation)
-----------

Train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]
 
    
   * **Task:** Episodic
   * **Reward:** +1 for collecting a yellow banana, -1 is provided for collecting a blue banana.  
   * **State space:** 
      * **Vector Environment:** 37 dimensions that includes agent's velocity, along with ray-based perception of objects around agent's forward direction.  
      * **Visual Environment:** (84, 84, 3) where 84x84 is the image size.
   * **Action space:** 
       - **`0`** - move forward.
       - **`1`** - move backward.
       - **`2`** - turn left.
       - **`3`** - turn right.
       
       
       
[//]: # (Image References)

[image2]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"


 Project 2: Continuous Control [*Link*](https://github.com/Sardhendu/DeepRL/tree/master/src/continuous_control)
-----------

Train anagent (double-jointed arm) is to maintain its position at the target location for as many time steps as possible 

![Trained Agent][image2]
 
    
   * **Task:** Continuous
   * **Reward:** +0.1 for agent's hand in the goal location  
   * **State space:** 
      * **Single Agent Environment:** (1, 33)
         * 33 dimensions consisting of position, rotation, velocity, and angular velocities of the arm.
         * 1 agent   
      * **Multi Agent Environment:** (20, 33)
         * 33 dimensions consisting of position, rotation, velocity, and angular velocities of the arm.
         * 20 agent 
   * **Action space:** 
       Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.
 
 
[//]: # (Image References)      
       
[image3]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"

![Trained Agent][image3] 
            
 
 
 
[//]: # (Image References)

[image4]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"


 Project 3: Collaboration and Competition [*Link*](https://github.com/Sardhendu/DeepRL/tree/master/src/collab_compete)
-----------

Train two agents to play ping pong. And, the goal of each agent is to keep the ball in play. 

![Trained Agent][image4]
 
    
   * **Task:** Continuous
   * **Reward:** +0.1 for hitting the ball over net 
                  -0.01 if the ball hits the ground or goes out of bounds 
   * **State space:** 
      * **Multi Agent Environment:** (2, 24)
         * 24 dimensions consisting of position, rotation, velocity and etc.   
   * **Action space:** 
       Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.
   * **Target Score**
        The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.
 
 
 
 ### TODO:
 1. Modular code for each environment.
 2. Dueling Network Architectures with DQN
 3. Lambda return (n-step bootstrap)
 4. Actor-Critic
 5. Crawler for Continuous Control
 6. Add Tensorflow graphs instead of manual dictionary graphs.
 7. Continuous control Test phase.

 

## To install with Docker

Create a docker Image:

   ```bash
      docker build --tag deep_rl .
   ```

Run the Image, expose Jupyter Notebook at port 8888 and mount the working directory:
   ```bash
      docker run -it -p 8888:8888 -v /path/to/your/local/workspace:/workspace/DeepRL --name deep_rl deep_rl
   ```
   
    
Start Jupyter Notebook:
   ```bash
      jupyter notebook --no-browser --allow-root --ip 0.0.0.0
   ```
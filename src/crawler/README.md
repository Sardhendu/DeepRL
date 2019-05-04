


1. Agents: Number of Agents: 12
    -> A creature with 4 arms and 4 forearms. 

2. States: Number of states (Observation): 129 
    -> Position for each limb
    -> Rotation of each limb
    -> Velocity of each limb
    -> Angular Velocity of each limb
    -> Acceleration of each limb
    -> Angular acceleration of each limb

3. Actions: (Continuous (-1, 1)), Count: 20
    -> Corresponds to target rotation for each joint.
    
4. Reward:
    -> +0.03 times body velocity in the goal direction.
    -> +0.01 times body direction alignment with goal direction.
      
5. Solving the Environment:
    -> The output action space is continuous, therefore we need to choose algorithm that can output continuous values.
       For Example:
        1. Policy Gradient Methods when learning on-policy
        2. DDPG (Deep Deterministic Policy Gradient) when learning off-policy
        3. Actor-Critic to make the best of each on-policy and off-policy
        
6. Implementation: 
    Here we implement the Actor-Critic (A2C) method.        

7. Actor-Critic:
    Policy based methods such as REINFORCE have high variance but low bias. In a nutshell, these method collect 
    many-many trajectories and based on the outcome (won or lost), these update their network weights. If the game was 
    lost then the network weights are updated such that the probability of produced action decreases and if the game 
    was won then the probability of the produced actions are increased. In this process however, some actions that 
    might actually be beneficial could be downvoted because of the game was lost.  
    
    The problem with Policy based method that follows similar paradigm to "Monte-Carlo" based approaches is that 
    rewards are calculated at the end of the episode. However, if rewards were calculated after every experience <s, 
    a> then we could have easily penalized bad actions even if the agent won the game at the end. 
     
     To the rescue comes actor critic methods where a policy based method is used as a baseline/actor model, that 
     decides on the action to take. And, a TD based method is used as a critic to calculate the discounted future
     rewards after every experience. In other words the critic is used to evaluate whether the action chosen was good
      or not. 
      
     How to train the model:
     Policy Gradient :
        Policy update: ∆θ = &alpha * &nabla; log π(st, at, θ) * R(t) 
        
     Actor - Critic :
        Policy update: ∆θ = œ * &nabla; log π(st, at, θ) * Q(st,at)
        where,  Q(st, at) =  rt + gamma * Q(s(t+1), a(t+1)) 

    Step 1: First we get the distribution of actions π(q(a|s;θπ)) form the actor network (Policy gradient method) 
     using 
    current state. Using the action distribution we select the best action and get the reward and next state i.e. our 
    experience tuple <s, a, r, s'>.
    
    Step 2: We evaluate the policy chosen by the actor network using the critic network (Temporal Difference method).
     We use experience tuple <s, a, r, s'> get critic estimate of <s'> and compute: r + gamma*V(s',θv). Now we train
      the critic
     
    Step 3: We know calculate the advantage using Critic: A(s, a) = r + gamma*V(s',θv) - V(s,θv).
    
    Step 4: Finally we Train the Actor using hte advantage.  
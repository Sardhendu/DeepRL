[//]: # (Image References)

[image1]: https://github.com/Sardhendu/DeepRL/blob/master/pong_atari/images/pong-v0.gif "Trained Agent"

Play Pong
-----------
Train an agent to play Pong using pixels.  

![Trained Agent][image1]

### Description:
This project is aimed to teach a reinforcement learning "REINFORCE" agent to play Pong (atari) using pixels. 
Reinforce are policy gradient methods in which the agent directly optimizes for the policy. This repo contains the vanilla "REINFORCE" algorithm and other enhancements such as Importance Sampling and Proximal Policy Optimization.

### Implemented Functionality:
   * Vanila REINFORCE: 
   * Importance Sampling: Used to overcome the poor credit assignment problems with vanilla REINFORCE
   * Proximal Policy Optimization: Used to overcome poor approximation between Old and New policies.

### Notes:
Theres are several improvements to vanila REINFORCE
1. **Noisy Gradient:** 
   * *Mini Batch of Trajectories:* The policy (p) is optimized by maximizing the reward. On realty there can be (inf) 
   trajectories. Sampling and learning form one trajectory doesn't make sense, since one trajectory is not 
   representative of enough  information. This case of learning from one trajectory resonates to learning with one 
   example with supervised learning, which cause gradients to be very noisy. Hense the **solution** here is to just 
   to accumulate **"m"** trajectories and learn the policy gradient by averaging m trajectories. Where "m" is the 
   mini-batch size. Learning from multiple trajectories help in alleviating the degree of noise in gradients.
    
   Some benefit of having "m" trajectories are:
   * We can collect and operate on multiple trajectories in parallel.
   * Having multiple trajectories and averaging them makes policy gradient methods more closer to MonteCarlo methods.
    Which makes it a low bias and high (but not too high) variance.
   * Also, having mini batch of trajectories have in **"reward normalization"**. Reward distribution changes or 
   shifts as training continues. Hence it is important to evaluate rewards based in its distribution, which can be done by normalization, which is not possible when we don't have mini-batch of trajectories.
   
2. **Credit Assignment:** A trajectory consist of many *<state, action, reward>* pairs. Since the rewards for are 
provided at the end of an episode, having a negative reward doesn't mean that all the *<state, action>* pair were bad
. Only a few bad choices could prompt the trajectory in having a negative reward. Hence it is important to assign 
credit based on individual choices. In short Credit assignment helps identify good vs bad action in a trajectory.

As an intuition if you see *<choice1, choice2, choice3>* in many trajectories that led to a negative reward, you can be 
more sure that the sequence *<choice1, choice2, choice3>* is a bad sequence and should be downvoted.

The underlying idea in Credit assignment is     

### Reference:

* Udacity Deep Reinforcement Learning : [Repository](https://github.com/udacity/deep-reinforcement-learning)
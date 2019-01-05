### Deep Deterministic Policy Gradient method (DDPG):

DDPG follows a structure to Actor-Critic method while it approximates DQN for a continuous space. The continuous control environment is a continuous process where the srm..

DDPG can take advantage of many enhancements proposed to vanilla DQN such as replay buffer, Fixed-Q-target by employing the soft-update technique. The **Actor** in DDPG optimizes to find the best action given the state, while the **Critic** evaluates the output of the probability of the action space. 

**Policy Gradient and Monte-Carlo**:
Monte-Carlo methods have high variance because they take the expectation of the state-action value across many episodes and these values can vary too much. These methods have low bias because there are lots of random event that takes place within each episode. Policy gradient relates to Monte-carlo approximation where random events are drawn forming a trajectory following a policy.

**DQN and Temporal Difference**:
Temporal Difference methods are estimate of estimates. In TD methods we approximate the value-function using cumulative return (which is also an estimate) for every time-step. This leaves much less room for the value function to diverge, and hence has low variance. These methods, however can have high bias because if the first few estimates are not correct or way too off from the correct path then it gets difficult for TD-methods to get on track, hence resulting in high bias.

**Actor-Critic methods**:
Actor-Critic method make the best out of both. In uses a baseline network similar to policy gradient method and maintain low bias while evaluates the action-value using a DQN like network to check high variance posed by the baseline networks. In other words Actor-Critic learning algorithms are used to represent the polucy function independently of the value function, where the actor is responsible for the policy and the Critic is responsible for the value function.
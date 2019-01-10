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
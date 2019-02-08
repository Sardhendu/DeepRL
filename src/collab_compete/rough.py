import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import copy
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class ActorModel(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, input_size, output_size, seed, fc1_units=256, fc2_units=256):
        """Initialize parameters and build actor model.
        Params
        ======
            input_size (int):  number of dimensions for input layer
            output_size (int): number of dimensions for output layer
            seed (int): random seed
            fc1_units (int): number of nodes in first hidden layer
            fc2_units (int): number of nodes in second hidden layer
        """
        super(ActorModel, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, output_size)
        self.bn = nn.BatchNorm1d(fc1_units)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights with near zero values."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor network that maps states to actions."""
        # #print('Model Actor: ', state.shape)
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        x = F.relu(self.fc1(state))
        x = self.bn(x)
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x


class CriticModel(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, input_size, seed, fc1_units=256, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            input_size (int): number of dimensions for input layer
            seed (int): random seed
            fc1_units (int): number of nodes in the first hidden layer
            fc2_units (int): number of nodes in the second hidden layer
        """
        super(CriticModel, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights with near zero values."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        """Build a critic network that maps (states, actions) pairs to Q-values."""
        # #print('Model Critic', states.shape, actions.shape)
        xs = torch.cat((states, actions), dim=1)
        x = F.relu(self.fc1(xs))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor_Critic_Models():
    """
    Create object containing all models required per DDPG agent:
    local and target actor and local and target critic
    """
    def __init__(self, n_agents, state_size=24, action_size=2, seed=0):
        """
        Params
        ======
            n_agents (int): number of agents
            state_size (int): number of state dimensions for a single agent
            action_size (int): number of action dimensions for a single agent
            seed (int): random seed
        """
        self.actor_local = ActorModel(state_size, action_size, seed).to(device)
        self.actor_target = ActorModel(state_size, action_size, seed).to(device)
        critic_input_size = (state_size+action_size)*n_agents
        self.critic_local = CriticModel(critic_input_size, seed).to(device)
        self.critic_target = CriticModel(critic_input_size, seed).to(device)





class MADDPG():
    """Meta agent that contains the two DDPG agents and shared replay buffer."""
    
    def __init__(self, action_size=2, seed=0,
                 n_agents=2,
                 buffer_size=10000,
                 batch_size=256,
                 gamma=0.99,
                 update_every=2,
                 noise_start=1.0,
                 noise_decay=1.0,
                 t_stop_noise=30000):
        """
        Params
        ======
            action_size (int): dimension of each action
            seed (int): Random seed
            n_agents (int): number of distinct agents
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            noise_start (float): initial noise weighting factor
            noise_decay (float): noise decay rate
            update_every (int): how often to update the network
            t_stop_noise (int): max number of timesteps with noise applied in training
        """
        
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.gamma = gamma
        self.n_agents = n_agents
        self.noise_weight = noise_start
        self.noise_decay = noise_decay
        self.t_step = 0
        self.noise_on = True
        self.t_stop_noise = t_stop_noise
        
        # create two agents, each with their own actor and critic
        models = [Actor_Critic_Models(n_agents=n_agents) for _ in range(n_agents)]
        self.agents = [DDPG(i, models[i]) for i in range(n_agents)]
        
        # create shared replay buffer
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)
    
    def step(self, all_states, all_actions, all_rewards, all_next_states, all_dones, iteration_num):
        all_states = all_states.reshape(1, -1)  # reshape 2x24 into 1x48 dim vector
        all_next_states = all_next_states.reshape(1, -1)  # reshape 2x24 into 1x48 dim vector
        self.memory.add(all_states, all_actions, all_rewards, all_next_states, all_dones)
        
        # if t_stop_noise time steps are achieved turn off noise
        if self.t_step > self.t_stop_noise:
            self.noise_on = False
        
        self.t_step = self.t_step + 1
        
        # Learn every update_every time steps.
        if self.t_step % self.update_every == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                # #print('Learning at step: ', iteration_num)
                # sample from the replay buffer for each agent
                experiences = [self.memory.sample() for _ in range(self.n_agents)]
                self.learn(experiences, self.gamma)
    
    def act(self, all_states, add_noise=True):
        # pass each agent's state from the environment and calculate its action
        all_actions = []
        for agent, state in zip(self.agents, all_states):
            action = agent.act(state, noise_weight=self.noise_weight, add_noise=self.noise_on)
            self.noise_weight *= self.noise_decay
            # #print('self.noise_weight: ', self.noise_weight)
            
            all_actions.append(action)
        return np.array(all_actions).reshape(1, -1)  # reshape 2x2 into 1x4 dim vector
    
    def learn(self, experiences, gamma):
        # each agent uses its own actor to calculate next_actions
        all_next_actions = []
        all_actions = []
        for i, agent in enumerate(self.agents):
            states, _, _, next_states, _ = experiences[i]
            agent_id = torch.tensor([i]).to(device)
            # extract agent i's state and get action via actor network
            state = states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            action = agent.actor_local(state)
            all_actions.append(action)
            # extract agent i's next state and get action via target actor network
            next_state = next_states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            next_action = agent.actor_target(next_state)
            all_next_actions.append(next_action)
        
        # each agent learns from its experience sample
        for i, agent in enumerate(self.agents):
            agent.learn(i, experiences[i], gamma, all_next_actions, all_actions)
    
    def save_agents(self):
        # save models for local actor and critic of each agent
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), f"checkpoint_actor_agent_{i}.pth")
            torch.save(agent.critic_local.state_dict(), f"checkpoint_critic_agent_{i}.pth")


class DDPG():
    """DDPG agent with own actor and critic."""
    
    def __init__(self, agent_id, model, action_size=2, seed=0,
                 tau=1e-3,
                 lr_actor=1e-4,
                 lr_critic=1e-3,
                 weight_decay=0.0):
        """
        Params
        ======
            model: model object
            action_size (int): dimension of each action
            seed (int): Random seed
            tau (float): for soft update of target parameters
            lr_actor (float): learning rate for actor
            lr_critic (float): learning rate for critic
            weight_decay (float): L2 weight decay
        """
        random.seed(seed)
        self.id = agent_id
        self.action_size = action_size
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        
        # Actor Network
        self.actor_local = model.actor_local
        self.actor_target = model.actor_target
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        
        # Critic Network
        self.critic_local = model.critic_local
        self.critic_target = model.critic_target
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)
        
        # Set weights for local and target actor, respectively, critic the same
        self.hard_copy_weights(self.actor_target, self.actor_local)
        self.hard_copy_weights(self.critic_target, self.critic_local)
        
        # Noise process
        self.noise = OUNoise(action_size, seed)
    
    def hard_copy_weights(self, target, source):
        """ copy weights from source to target network (part of initialization)"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def act(self, state, noise_weight=1.0, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        #print(state)
        # calculate action values
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        #print(action)
        if add_noise:
            # #print(noise_weight)
            self.noise_val = self.noise.sample() * noise_weight
            action += self.noise_val
        return np.clip(action, -1, 1)
    
    def reset(self):
        #print ('23742837489237472834792397489')
        self.noise.reset()
    
    def learn(self, agent_id, experiences, gamma, all_next_actions, all_actions):
        """Update policy and value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            all_next_actions (list): each agent's next_action (as calculated by its actor)
            all_actions (list): each agent's action (as calculated by its actor)
        """
        
        states, actions, rewards, next_states, dones = experiences
        # #print('learning: agent_id', agent_id, states.shape, action.shape, next_states.shape)
        
        # ---------------------------- update critic ---------------------------- #
        # get predicted next-state actions and Q values from target models
        self.critic_optimizer.zero_grad()
        agent_id = torch.tensor([agent_id]).to(device)
        all_next_actions = torch.cat(all_next_actions, dim=1).to(device)
        with torch.no_grad():
            q_targets_next = self.critic_target(next_states, all_next_actions)
        # compute Q targets for current states (y_i)
        q_expected = self.critic_local(states, actions)
        # #print( 'done:', (1 - dones.index_select(1, agent_id)).shape, q_targets_next.shape)
        # q_targets = reward of this timestep + discount * Q(st+1,at+1) from target network
        # print(dones.index_select(1, agent_id).cpu().data.numpy() == 1)
        # idx = np.arange(len(dones)).reshape(-1,1)[dones.index_select(1, agent_id).cpu().data.numpy() == 1]
        # print('idxidx : ', idx)
        # rwd = rewards.index_select(1, agent_id).cpu().data.numpy()[idx]
        # print(rwd)
        
        q_targets = rewards.index_select(1, agent_id) + (gamma * q_targets_next * (1 - dones.index_select(1, agent_id)))
        # #print('q_targetsq_targets: ', q_targets.shape)
        # #print('q_targets.detach()q_targets.detach(): ', q_targets.detach().shape)
        # compute critic loss
        critic_loss = F.mse_loss(q_expected, q_targets.detach())
        # minimize loss
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #
        # compute actor loss
        self.actor_optimizer.zero_grad()
        # detach actions from other agents
        actions_pred = [actions if i == self.id else actions.detach() for i, actions in enumerate(all_actions)]
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        # print('states, actions pred: ', states.shape, actions_pred.shape)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # minimize loss
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""
    
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        random.seed(seed)
        np.random.seed(seed)
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        #print ('237189237891738937972987jcjshdgchjdgchjdghcjgdcghjdgjhhgjhgj')
        self.state = copy.copy(self.mu)
    
    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): Random seed
        """
        random.seed(seed)
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)







from unityagents import UnityEnvironment

env = UnityEnvironment(file_name="Tennis.app", no_graphics=True)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


agent = MADDPG(seed=2, noise_start=0.5, update_every=2, gamma=1, t_stop_noise=30000)
n_episodes = 6000
max_t = 1000
scores = []
scores_deque = deque(maxlen=100)
scores_avg = []

iteration_num = 0
for i_episode in range(1, n_episodes + 1):
    rewards = []
    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    state = env_info.vector_observations  # get the current state (for each agent)
    
    # loop over steps
    for t in range(max_t):
        # #print('Time step: ', t)
        # select an action
        action = agent.act(state)
        # take action in environment and set parameters to new values
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations
        rewards_vec = env_info.rewards
        done = env_info.local_done
        # update and train agent with returned information
        agent.step(state, action, rewards_vec, next_state, done, iteration_num)
        state = next_state
        rewards.append(rewards_vec)
        iteration_num += 1
        if any(done):
            break
    
    # calculate episode reward as maximum of individually collected rewards of agents
    episode_reward = np.max(np.sum(np.array(rewards), axis=0))
    
    scores.append(episode_reward)  # save most recent score to overall score array
    scores_deque.append(episode_reward)  # save most recent score to running window of 100 last scores
    current_avg_score = np.mean(scores_deque)
    scores_avg.append(current_avg_score)  # save average of last 100 scores to average score array
    
    print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, current_avg_score), end="")
    
    # log average score every 200 episodes
    if i_episode % 200 == 0:
        print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, current_avg_score))
        agent.save_agents()
    
    # break and report success if environment is solved
    if np.mean(scores_deque) >= .5:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(i_episode, np.mean(scores_deque)))
        agent.save_agents()
        break

import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    
    def __init__(self, args):
        self.NUM_AGENTS = args.NUM_AGENTS
        self.ACTION_SIZE = args.ACTION_SIZE
        self.GAMMA = args.GAMMA
        print(self.NUM_AGENTS, self.ACTION_SIZE)
        
        self.actor = args.ACTOR_NETWORK_FN()
        self.critic = args.CRITIC_NETWORK_FN()
    
    def act(self, state):
        """ ACtor: Baseline Model is a policy gradient model that outputs actions probabilities
        
        :param state: ndarray [num_agents, num_states]
        :return:
        """
        state = torch.from_numpy(state).float().to(device)

        self.actor.eval()
        with torch.no_grad():
            actions = self.actor.forward(state).cpu().data.numpy()
        self.actor.train()
        
        # print(actions.shape)
        actions = np.clip(actions, -1, 1)
        
        return actions
        # actor_action = np.random.randn(self.NUM_AGENTS, self.ACTION_SIZE)
        # return actor_action
    
    def step(self, state, action, reward, next_states, dones, running_timestep):
        experience = [state, action, reward, next_states, dones]
        self.learn(experience, gamma=self.GAMMA, running_timestep=running_timestep)
    
    def learn(self, experiences, gamma, running_timestep):
        """
        
        :param experiences:  List if state, action, reward, next_state
                            state: torch(tensor) [num_agents, num_states]
                            action: torch(tensor)
                            reward torch(tensor) [num_agents]
                            next_state: torch(tensor) [num_agents, num_states]
                            dones: torch(tensor)
                            
        :param gamma:
        :param running_timestep:
        :return:
        """
        states, actions, rewards, next_states, dones = experiences
        
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).float().to(device)
        expected_value = self.critic(states, actions).cpu().data.numpy()
        print(expected_value)
        pass

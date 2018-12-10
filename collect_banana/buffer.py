import random
import numpy as np
import torch
from collections import deque, namedtuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MemoryER:
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        print('[INIT] Initializing Replay Buffer .... .... ....')
        self.memory = deque(maxlen=buffer_size)  # FIFO
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # print('1212121212 ', state.shape, action, reward, next_state.shape, done)
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(
                device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(
                device)
        next_states = torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def debug_MemoryER():
    obj_buf = MemoryER( 10, 5, seed=413)
    
    obj_buf.add(1,2,3,4,0)
    obj_buf.add(2,6,2,4,0)
    obj_buf.add(0,2,9,5,0)
    obj_buf.add(9,9,9,9,1)
    obj_buf.add(1,1,1,1,1)
    obj_buf.add(1,2,3,4,0)
    obj_buf.add(2,6,2,4,0)
    obj_buf.add(0,2,9,5,0)
    obj_buf.add(9,9,9,9,1)
    obj_buf.add(1,1,1,1,1)
    
    print (obj_buf.memory)
    
    obj_buf.add(9,9,9,9,1)
    obj_buf.add(1,1,1,1,1)
    
    print('')
    print (obj_buf.memory)
    print('')
    sample_data = obj_buf.sample()
    print(sample_data)
# =[0.9, 2.0, 3.5, 0.6, 0])


def debug_MemoryPER():
    buffer = MemoryPER(buffer_size=7, batch_size=3, seed=84)
    buffer.add(0.5, 2, 3, 0.6, 0)
    print('')
    print(len(buffer))
    buffer.add(0.9, 2.0, 3.5, 0.6, 0)
    print('')
    buffer.add(0.9, 2.0, 3.5, 0.6, 0)
    print(len(buffer))
    print('')
    buffer.add(0.9, 2.0, 1.5, 0.6, 0)
    print('')
    buffer.add(0.9, 2.0, 4.5, 0.6, 0)
    print(len(buffer))
    print('')
    buffer.add(0.9, 2.0, 7.5, 0.6, 0)
    print('')
    buffer.add(0.9, 1.0, 1.5, 0.6, 0)
    print('')
    buffer.add(0.9, 2.0, 7.5, 0.6, 0)
    print('')
    buffer.update(tree_idx=np.array([7, 6]), abs_errors=np.array([0.8, 0.2]))
    print(len(buffer))

    # a = buffer.sample(3)
    # print(a)
    
# debug_MemoryPER()
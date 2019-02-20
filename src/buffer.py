import random
import numpy as np
import torch
from collections import deque, namedtuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MemoryER:
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self, buffer_size, batch_size, seed, action_dtype='long'):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        print('[INIT] Initializing Experience Replay Buffer .... .... ....')
        
        random.seed(seed)
        self.memory = deque(maxlen=buffer_size)  # FIFO
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.action_dtype = action_dtype
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        if self.action_dtype == 'long':
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        elif self.action_dtype == 'float':
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        else:
            raise ValueError('Only float and double type accepted for actions')
        
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class SumTree(object):
    """
    :: This part of code is taken from simoninithomas (Deep Reinforcement Learning course)
    https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with
    %20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20%28%2B%20double%20DQNs%20and
    %20Prioritized%20Experience%20Replay%29.ipynb

    And the This SumTree code is modified version of Morvan Zhou:
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5
    .2_Prioritized_Replay_DQN/RL_brain.py
    """
    data_pointer = 0
    
    """
    Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    """
    
    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes (final nodes) that contains experiences
        
        # Generate the tree with all nodes values = 0, Uniform Priority
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)
        
        """
        tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
        """
        
        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)
        # print('Tree: ', len(self.tree), self.tree)
        # print('Data: ', self.data)
    
    """
    Here we add our priority score in the sumtree leaf and add the experience in data
    """
    
    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1
        
        """ tree:
            0
           / \
          0   0
         / \ / \
tree_index  0 0  0  We fill the leaves from left to right
        """
        
        # Update data frame
        self.data[self.data_pointer] = data
        # print('Add to Data: ', self.data)
        
        # Update the leaf
        self.update(tree_index, priority)
        
        # Add 1 to data_pointer
        self.data_pointer += 1
        
        if self.data_pointer >= self.capacity:  # If we're above the capacity, you go back to first index (we overwrite)
            self.data_pointer = 0
    
    """
    Update the leaf priority score and propagate the change through tree
    """
    
    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        # print('Tree Before Change: ', self.tree)
        # print('change: ', change)
        
        # then propagate the change through tree
        while tree_index != 0:  # this method is faster than the recursive loop in the reference code
            
            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES

                0
               / \
              1   2
             / \ / \
            3  4 5  [6]

            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
            
            # print('Tree After Change update: ', self.tree)
    
    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    """
    
    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0
        
        while True:  # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # print('Left Right Tree Index: ', left_child_index, right_child_index)
            
            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            
            else:  # downward search, always search for a higher priority node
                # Go down using the tree path to the correct leaf based on the va;lue (v)
                if v <= self.tree[left_child_index]:
                    # print('YESYES: ', v, self.tree[left_child_index])
                    parent_index = left_child_index
                
                else:
                    v -= self.tree[left_child_index]
                    # print('NONONO: ', v,  self.tree[left_child_index])
                    parent_index = right_child_index
        
        data_index = leaf_index - self.capacity + 1
        # print('data index: ', data_index)
        
        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node


class MemoryPER(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    :: This part of code is taken from simoninithomas (Deep Reinforcement Learning course)
    https://github.com/simoninithomas/Deep_reinforcement_learning_Course

    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and
    # sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1
    
    PER_b_increment_per_sampling = 0.001
    
    absolute_error_upper = 1.  # clipped abs error
    
    def __init__(self, buffer_size, batch_size, seed, action_dtype='long'):
        # Making the tree
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """
        print('[INIT] Initializing Priority Experience Replay Buffer .... .... ....')
        
        self.tree = SumTree(buffer_size)
        self.batch_size = batch_size
        self.len = 1
        self.action_dtype = action_dtype
    
    def add(self, state, action, reward, next_state, done):
        """
            STORE WHEN A NEW EXPERIENCE COMES IN:
            The idea is, when a new experience comes in we store the it having a priority of 1.
            After the batch is full, and we sample an experience and calculate the TD error.
            Only then we update the priority of the experience.
        :param experience:
        :return:
        """
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        
        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        
        experience = [state, action, reward, next_state, done]
        self.tree.add(max_priority, experience)  # set the max p for new p
        
        self.len += 1
    
    def sample(self):
        """
        - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
        - Then a value is uniformly sampled from each range
        - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
        - Then, we calculate IS weights for each minibatch element
        """
        
        n = self.batch_size
        
        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)
        
        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        # print('Tree Total Priority: ', self.tree.total_priority)
        priority_segment = self.tree.total_priority / n  # priority segment
        # print('priority_segment: ', priority_segment)
        
        # This term is involved in the coefficient (weights) of the update function to provide importance sampling.
        # Here we increase the PER_b each time we sample a new minibatch. At initial phases we want more random sample,
        # and then at later stages we would increase the probability of sampling by priority to 1
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1
        # print('PER_b: ', self.PER_b)
        
        # Calculating the max_weight
        # print('Min Tree Capacity: ', self.tree.tree[-self.tree.capacity:])
        
        # This formula will work only when we full the buffer capicity before sampling else pmin would be 0
        # until the buffer is filled and will behave weird while learning with b_ISweights = 0
        # p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        # print('p_min; ', p_min)
        # max_weight = (p_min * n) ** (-self.PER_b)
        # print('max_weight: ', max_weight)
        max_wi = 0
        priority_arr = []
        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            # print(a, b)
            value = np.random.uniform(a, b)
            # print('value: ', value)
            
            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)
            # print(index, priority, data)
            priority_arr.append(priority)
            # P(j)
            sampling_probabilities = priority / self.tree.total_priority
            
            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            wi = np.power(n * sampling_probabilities, -self.PER_b)
            max_wi = max(max_wi, wi)
            b_ISWeights[i, 0] = wi  # / max_weight
            
            b_idx[i] = index
            
            # print(type(data[0]), type(data[3]))
            if i == 0:
                # Create a sample array that will contain the minibatch
                state_b = np.zeros((self.batch_size, len(data[0])), dtype=np.float)  # [batch_size, state_size]
                action_b = np.zeros((self.batch_size, 1), dtype=type(data[1]))
                reward_b = np.zeros((self.batch_size, 1), dtype=type(data[2]))
                state_next_b = np.zeros((self.batch_size, len(data[3])), dtype=np.float)
                done_b = np.zeros((self.batch_size, 1), dtype=type(data[4]))
            
            state_b[i, :] = data[0]
            action_b[i, :] = data[1]
            reward_b[i, :] = data[2]
            state_next_b[i, :] = data[3]
            done_b[i, :] = data[4]
        
        # print('b_ISWeights: ', b_ISWeights)
        b_ISWeights /= max_wi
        # print('b_ISWeightdcdcdcs: ', b_ISWeights)
        
        states = torch.from_numpy(state_b).float().to(device)
        if self.action_dtype == 'long':
            actions = torch.from_numpy(action_b).long().to(device)
        elif self.action_dtype == 'float':
            actions = torch.from_numpy(action_b).float().to(device)
        rewards = torch.from_numpy(reward_b).float().to(device)
        next_states = torch.from_numpy(state_next_b).float().to(device)
        dones = torch.from_numpy(done_b.astype(np.uint8)).float().to(device)
        
        weights = torch.from_numpy(b_ISWeights).float().to(device)
        
        # print(type(b_ISWeights), b_ISWeights.dtype, weights)
        # print(b_idx)
        # print(priority_arr)
        return b_idx, [states, actions, rewards, next_states, dones], weights
        
        # return b_idx, memory_b, b_ISWeights
    
    """
    Update the priorities on the tree
    """
    
    def update(self, tree_idx, abs_errors):
        """
            UPDATE THE LEAF PRIORITY (USING TD ERROR) WHEN THE EXPERIENCE WAS USED TO MAKE LEARN,
        :param tree_idx:
        :param abs_errors:
        :return:
        """
        # print('Updating .............')
        # print(tree_idx)
        # print('')
        # print(abs_errors)
        # print('.....................')
        # We don't want priority to be 0, if the abs_error is zero then we add a small value PER_e=0.01
        abs_errors += self.PER_e  # convert to abs and avoid 0
        #
        # TD error can be more than 1 and we want to limit the priority to 1
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        
        # If the priority of few examples are very high say 0.9 then using sampling we would end up training this subse
        # many many times, which may introduce overfitting. Inoder, to avoid this we raise the priority to a certain
        # power. Low values of PER_a would help random sample more that priority based sample
        # For example p1=0.8, p2=0.1. pow(0.8, 0.6)=0.87468, pow(0.1, 0.6)=0.25. p2 increases more than p1
        ps = np.power(clipped_errors, self.PER_a)
        
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return self.len


def debug_MemoryER():
    obj_buf = MemoryER(10, 5, seed=413)
    
    obj_buf.add(1, 2, 3, 4, 0)
    obj_buf.add(2, 6, 2, 4, 0)
    obj_buf.add(0, 2, 9, 5, 0)
    obj_buf.add(9, 9, 9, 9, 1)
    obj_buf.add(1, 1, 1, 1, 1)
    obj_buf.add(1, 2, 3, 4, 0)
    obj_buf.add(2, 6, 2, 4, 0)
    obj_buf.add(0, 2, 9, 5, 0)
    obj_buf.add(9, 9, 9, 9, 1)
    obj_buf.add(1, 1, 1, 1, 1)
    
    print(obj_buf.memory)
    
    obj_buf.add(9, 9, 9, 9, 1)
    obj_buf.add(1, 1, 1, 1, 1)
    
    print('')
    print(obj_buf.memory)
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

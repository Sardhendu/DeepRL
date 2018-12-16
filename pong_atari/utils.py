
import numpy as np
import torch



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RIGHTFIRE = 4
LEFTFIRE = 5

def preprocess_single(image, bkg_color = np.array([144, 72, 17])):
    img = np.mean(image[34:-16:2,::2]-bkg_color, axis=-1)/255.
    return img

# convert outputs of parallelEnv to inputs to pytorch neural net
# this is useful for batch processing especially on the GPU
def preprocess_batch(images, bkg_color = np.array([144, 72, 17])):
    list_of_images = np.asarray(images)
    if len(list_of_images.shape) < 5:
        list_of_images = np.expand_dims(list_of_images, 1)
        
    # subtract bkg and crop the scpre pannels from top, bottom, left and right
    list_of_images_prepro = np.mean(list_of_images[:,:,34:-16:2,::2]-bkg_color,
                                    axis=-1)/255.
    batch_input = np.swapaxes(list_of_images_prepro,0,1)
    return torch.from_numpy(batch_input).float().to(device)




def collect_trajectories(envs, policy, tmax=200, nrand=5, n=4):
    
    # number of parallel instances n=len( envs.ps)

    # initialize returning lists and start the game!
    state_list=[]
    reward_list=[]
    action_probs_list=[]
    action_list=[]
    
    envs.reset()
    # start all parallel agents
    envs.step([1]*n)
    # Perform nrand random steps
    for j in range(nrand):
        frame1, reward1, _, _ = envs.step(np.random.choice([RIGHTFIRE, LEFTFIRE],n))
        frame2, reward2, _, _ = envs.step([0]*n)
    
    for t in range(tmax):
        # prepare the input
        # preprocess_batch properly converts two frames into
        # shape (n, 2, 80, 80), the proper input for the policy
        # this is required when building CNN with pytorch
        batch_input = preprocess_batch([frame1,frame2] )
        # probs will only be used as the pi_old
        # no gradient propagation is needed
        # so we move it to the cpu
        action_probs = policy(batch_input).squeeze().cpu().detach().numpy()

        # Assign Random actions to each state
        action = np.where(np.random.rand(n) < action_probs, RIGHTFIRE, LEFTFIRE)
        action_probs = np.where(action == RIGHTFIRE, action_probs, 1.0 - action_probs)

        # advance the game (0=no action)
        # we take one action and skip game forward
        frame1, reward1, is_done, _ = envs.step(action)
        frame2, reward2, is_done, _ = envs.step([0]*n)
        
        reward = reward1 + reward2
        # print(reward1, reward2)
        
        # store the result
        state_list.append(batch_input)
        reward_list.append(reward)
        action_probs_list.append(action_probs)
        action_list.append(action)
        
        # stop if any of the trajectories is done
        # we want all the lists to be rectangular
        if is_done.any():
            break
    
    # return pi_theta, states, actions, rewards, probability
    return action_probs_list, state_list, action_list, reward_list



def debug():
    import gym
    from DeepRL.pong_atari.parallel_environments import parallelEnv
    from DeepRL.pong_atari.model import Model

    # PongDeterministic does not contain random frameskip
    # so is faster to train than the vanilla Pong-v4 environment
    env = gym.make('PongDeterministic-v4')

    print("List of available actions: ", env.unwrapped.get_action_meanings())
    envs = parallelEnv('PongDeterministic-v4', n=4, seed=12345)

    policy = Model()
    prob, state, action, reward = collect_trajectories(envs, policy, tmax=100)
    print (prob)
    
# debug()
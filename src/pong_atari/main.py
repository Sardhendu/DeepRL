# widget bar to display progress

import matplotlib
matplotlib.use('PS')                # This helps with teh matplotlib error on macosx. But note this has to be in the
# top of every other matplotlib import

import gym
import numpy as np
import progressbar as pb
import torch
from IPython.display import display
from JSAnimation.IPython_display import display_animation

from matplotlib import animation
import matplotlib.pyplot as plt
from src.pong_atari.config import TrainConfig
from src.pong_atari.agent import ReinforceAgent
from src.pong_atari.parallel_env import parallelEnv
from src.pong_atari.utils import preprocess_batch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RIGHTFIRE = 4
LEFTFIRE = 5


def animate_frames(frames):
    plt.axis('off')
    
    # color option for plotting
    # use Greys for greyscale
    cmap = None if len(frames[0].shape) == 3 else 'Greys'
    patch = plt.imshow(frames[0], cmap=cmap)
    
    fanim = animation.FuncAnimation(plt.gcf(), lambda x: patch.set_data(frames[x]), frames=len(frames), interval=30)
    
    display(display_animation(fanim, default_mode='once'))
    
    
class PongAtariEnv:
    def __init__(self, args, env_type='parallel', mode='train'):
        self.args = args

        if mode == 'train':
            print('[Train] In training mode ....')
            self.num_episodes = args.NUM_EPISODES
            self.horizon = args.HORIZON
            self.num_parallel_env = args.NUM_PARALLEL_ENV
            self.env = parallelEnv('PongDeterministic-v4', self.num_parallel_env, seed=12345)
            self.agent = ReinforceAgent(args, self.env, env_type, mode, agent_id=0)
        else:
            print('[Train] In testing mode ....')
            self.env = gym.make('PongDeterministic-v4')
            
        
    def train(self):
        widget = ['training loop: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
        timer = pb.ProgressBar(widgets=widget, maxval= self.num_episodes).start()
        
        avg_rewards = []
        
        for e in range(self.num_episodes):

            # collect trajectories
            rewards = self.agent.learn(self.horizon, self.num_parallel_env, e)
            total_rewards = np.sum(rewards, axis=0)
            
            # get the average reward of the parallel environments
            avg_rewards.append(np.mean(total_rewards))
            
            # display some progress every 20 iterations
            if (e + 1) % 20 == 0:
                print("Episode: {0:d}, score: {1:f}".format(e + 1, np.mean(total_rewards)))
                print(total_rewards)
            
            # update progress widget bar
            timer.update(e + 1)
        
        timer.finish()
        
    def test(self, num_timesteps=2000, nrand=5, preprocess_output=None):
        policy_nn = torch.load(self.args.CHECKPOINT_PATH)
        self.env.reset()
        
        self.env.step(1)
        
        # Warm-up: Perform some random states in the beginning
        for _ in range(0, nrand):
            frame1, reward1, done, _ = self.env.step(np.random.choice([RIGHTFIRE, LEFTFIRE]))
            frame2, reward2, done, _ = self.env.step(0)

        anim_frames = []
        
        for t in range(0, num_timesteps):
            frame_batch = preprocess_batch([frame1, frame2])
            action_prob = policy_nn.forward(frame_batch)

            # If action probability is greater than 0.5 then chose
            action = RIGHTFIRE if np.random.random() < action_prob else LEFTFIRE
            
            frame1, reward1, done, _ = self.env.step(action)
            feame2, reward2, done, _ = self.env.step(0)

            if preprocess_output is None:
                anim_frames.append(frame1)
            else:
                anim_frames.append(preprocess_batch(frame1))

            if done:
                break

        self.env.close()

        animate_frames(anim_frames)
        return

            
        
        
mode = "test"
if __name__ == "__main__":
    if mode == 'train':
        obj_pong = PongAtariEnv(TrainConfig, env_type='parallel', mode='train')
        obj_pong.train()
    else:
        obj_pong = PongAtariEnv(TrainConfig, env_type='parallel', mode='test')
        obj_pong.test(num_timesteps=2000, nrand=5)

    
    
    

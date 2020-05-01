'''
This script define the agent we would use to train
'''

import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers.joypad_space import JoypadSpace
from gym import spaces
from gym.spaces.box import Box
from gym import Wrapper
from collections import deque    # ordered collections

import torch
import torch.multiprocessing as mp

# hyper parameters
from params import *
from model import A3C
from utils import Memory,preprocess,video
from optimizer import Adam_global

class Reward(Wrapper):
    def __init__(self,env):
        """
        Args:
            env -- the gym environment passed in
        """
        super(Reward).__init__(env)
        self.observation_space = Box(low=0,high=255,shape=(1,84,84))    # define observation space
        self.curr_score = 0

    def step(self,action):
        """
        update step reward here
        """
        state,reward,done,info = self.env.step(action)    # obtain the 
        state = preprocess(state)
        reward+=(info['score']-self.curr_score)/50
        self.curr_score = info['score']
        if done:
            if info['flag_get']:
                reward+=50
            else:
                reward-=50
        return state, reward,done,info

    def reset(self):
        """
        reset 
        """
        state = self.env.reset()
        self.curr_score = 0
        return preprocess(state)

class SkipEnv(Wrapper):
    def __init__(self,env,skip=4):
        """
        Return only every 'skip' frame
        Default skip frames: 4
        """
        super(SkipEnv,self).__init__(env)
        self.observation_space = Box(low=0,high=255,shape=(4,84,84))
        self.skip = skip
        self.skip_frame = deque(maxlen=skip)    # buffer to store skip observation

    def step(self,action):
        skip_reward = 0.0
        done = None
        for _ in range(self.skip):
            state,reward,done,info = self.env.step(action)
            self.skip_frame.append(state)
            skip_reward+=reward
            if done:
                break
        states = np.stack(self.skip_frame,axis=0)
        return states.astype(np.float32),reward,done,info

    def reset(self):
        """
        clear past frame buffer and init to first state
        """
        self.skip_frame.clear()
        state = self.env.reset()
        self.skip_frame.append(state)
        return state

def gym_env(world,stage,version,actions):
    '''
    Define the Super Mario Individual Stages to use.
    @ https://github.com/Kautenja/gym-super-mario-bros
    Inputs:
    world: a number in {1,2,3,4,5,6,7,8} indicating the world
    stage: a number in {1,2,3,4} indicating the stage
    version: a number in {0,1,2,3} specifying the ROM
    actions: static action sets for binary to discrete action space wrappers
    Outputs:
    env: Individual environment to use
    num_state: number of SuperMario Space
    num_action : number of action
    '''
    env = gym_super_mario_bros.make('SuperMarioBros--{}--{}--v{}'.format(world,stage,version))
    if actions == 'RIGHT_ONLY':
        act = RIGHT_ONLY
    elif actions == 'SIMPLE_MOVEMENT':
        act = SIMPLE_MOVEMENT
    elif actions == 'COMPLEX_MOVEMENT':
        act = COMPLEX_MOVEMENT
    env = JoypadSpace(env,act)
    env = Reward(env)
    env = SkipEnv(env)    # skip frame, default = 4
    num_state = env.observation_space.shape[0]
    num_action = env.action_space.n

    return env, num_state, num_action
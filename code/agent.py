'''
This script define the agent we would use to train
API:
    #To be Define
'''

import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers.joypad_space import JoypadSpace
from collections import deque    # ordered collections

import torch

# hyper parameters
from params import *
from model import A3C
from utils import Memory
from optimizer import Adam_global
from env import preprocess, video
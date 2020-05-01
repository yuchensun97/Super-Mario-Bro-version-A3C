"""
This script define the helper function for the agent to use
"""
import numpy as np
from collections import deque 
import torch.nn as nn

import cv2

from params import *

def init_weight(layers):
    for layer in layers:
        if type(layer) == nn.Conv2d or type(layer)==nn.Linear:
            nn.init.xavier_uniform_(layer.weight) # initialize weights
            nn.init.constant_(layer.bias,0)    # bias set to 0
        elif type(layer) == nn.LSTMCell:
            nn.init.constant_(layer.bias_ih,0)
            nn.init.constant_(layer.bias_hh,0) 

def preprocess(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)   # convert to grey scale to improve trainning speed
        frame = cv2.resize(frame,(84,84))[None,:,:]/255.
    else:
        return np.zeros((1,84,84))
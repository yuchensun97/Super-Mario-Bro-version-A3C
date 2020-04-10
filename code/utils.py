"""
This script define the helper function for the agent to use
"""
import numpy as np
from collections import deque 
import torch.nn as nn

import cv2

from params import *

class Memory:
    def __init__(self,max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self,experience):
        self.buffer.append(experience)

    def sample(self,batch_size):
        buffer_size = len(self.buffer)
        idx = np.random.choice(np.arange(buffer_size),size=batch_size,replace=False)
        return [self.buffer[i] for i in idx]

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
        frame = cv2.resize(frame,(128,128))[None,:,:]/255.
    else:
        return np.zeros((1,128,128))

def video(video_frame,state,new_episode_bool):
    """
    Args:
        video_frame -- a deque object used to store frames
        state -- agent's observation
        new_episode_bool -- a boolean used to determine a new episode
    """
    frame = preprocess(state)

    if new_episode_bool:
        # clear stacked frames
        stacked_frame = deque([np.zeros((128,128),dtype=np.int) for i in range(stack_size)],maxlen=4)
        stacked_frame.append(frame)
        stacked_frame.append(frame)
        stacked_frame.append(frame)
        stacked_frame.append(frame)

        stacked_state = np.stack(stacked_frame,axis=2)
    else:
        # append frame to deque
        stacked_frame.append(frame)
        stacked_state = np.stack(stacked_frame,axis=2)

    return stacked_state,stacked_frame
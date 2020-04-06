"""
This script define the helper function for the agent to use
"""
import numpy as np
from collections import deque 
import torch.nn as nn
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
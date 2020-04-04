'''
This script define the A3C model
'''

import torch.nn as nn
import numpy as np
import random

class A3C(nn.Module):
    def __init__(self, num_state, num_action):
        """
        Build your nn architecture here
        """
        super(A3C,self).__init__()
        pass

    def forward(self,x):
        """
        Forward
        """
        pass

    def choose_action(self,state):
        """
        choose action based on the probability or argmax
        """
        pass

    def loss(self, state,action,value):
        """
        compute loss
        """
        pass
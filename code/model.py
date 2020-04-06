'''
This script define the A3C architecture
'''

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
from utils import init_weight

class A3C(nn.Module):
    def __init__(self, num_state, num_action):
        """
        Build your nn architecture here
        """
        super(A3C,self).__init__()
        self.num_state = num_state
        self.num_action = num_action
        self.conv1 = nn.Conv2d(num_state,32,3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32,32,3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(32,32,3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(32,32,3,stride=1,padding=1)
        self.mp = nn.MaxPool2d((2,2))
        self.bn = nn.BatchNorm2d(32)
        self.lstm = nn.LSTMCell(32*8*8,512)
        self.actor = nn.Linear(512,num_action)
        self.critic = nn.Linear(512,1)
        init_weight([self.conv1,self.conv2,self.conv3,self.conv4,self.lstm,self.actor,self.critic])

    def forward(self,x,hx,cx):
        """
        Forward
        """
        x = F.relu(self.conv1(x))
        x = self.mp(x)
        x = self.bn(x)
        x = F.relu(self.conv2(x))
        x = self.mp(x)
        x = self.bn(x)
        x = F.relu(self.conv3(x))
        x = self.mp(x)
        x = self.bn(x)
        x = F.relu(self.conv4(x))
        x = self.mp(x)
        x = self.bn(x)
        hx,cx = self.lstm(x.view(x.size(0),-1),(hx,cx))
        return self.actor(hx),self.critic(hx),hx,cx
'''
This script define the training and testing function
for global training
'''

import os
import time

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical
from collections import deque

from src.agent import Reward,SkipEnv, gym_env
from src.model import A3C
from src.optimizer import Adam_global
from src.params import *
#from src.Args import Args
from src.utils import *
from single_thread import train

os.environ['OMP_NUM_THREADS'] = '1'

def globalTrain():
    torch.manual_seed(123)

    env,num_state,num_action = gym_env(world,stage,version,actions)    # define environment
    #env.seed(123+idx)

    shared_model = A3C(num_state,num_action)
    shared_model.share_memory()

    #optimizer = Adam_global(shared_model.parameters(), lr=Args.lr, betas = Args.betas ,eps = Args.eps, weight_decay = Args.weight_decay)
    optimizer = Adam_global(shared_model.parameters(), lr=lr, betas = betas ,eps = eps, weight_decay = weight_decay)


    processes = []
    counter = mp.Value('i', 0)
    lock = mp.Lock()

    for index in range(num_processes):
        process = mp.Process(target=train, args=(index, shared_model, optimizer,counter,lock))
        process.start()
        processes.append(process)
    # process = mp.Process(target=test, args=(num_processes,  shared_model, optimizer, global_counter))
    # process.start()
    # processes.append(process)
    for process in processes:
        process.join()

if __name__ == "__main__":
    globalTrain()
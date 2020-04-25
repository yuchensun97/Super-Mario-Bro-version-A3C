import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical

from src.agent import Reward,SkipEnv, gym_env
from src.model import A3C
from src.optimizer import Adam_global
from src.params import *
from src.utils import *

def train(idx, shared_model,optimizer,global_counter):
    '''
    A3C for EACH actor-learner thread

    Inputs:
    idx: a scalar, indicting the idx th thread
    shared_model: The global model
    optimizer: The optimizer used for local gradient descent
    global_counter: a scalar, global shared counter

    Returns:
    None
    '''
    # initialization
    torch.manual_seed(123+idx)

    env,num_state,num_action = gym_env(world,stage,version,actions)    # define environment
    env.seed(123+idx)

    model = A3C(num_state,num_action)

    model.train()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    step_counter = 0

    while True:
        # sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            hx = torch.zeros((1,256),dtype=torch.float)
            cx = torch.zeros((1,256),dtype=torch.float)
        else:
            hx = hx.detach()
            cx = cx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        # repeat until terminal or max steps reach
        for step in range(num_local_steps):
            step_counter+=1

            # perform action according to policy
            logits,value,hx,cx = model(state,hx,cx)
            prob = F.softmax(logits,dim=-1)    # probability of choosing each actions
            log_prob = F.log_softmax(logits,dim=1)
            entropy = -(log_prob * prob).sum(1,keepdim=True)
            entropy.append(entropy)

            m = Categorical(prob)
            action = m.sample().item()    # choosing actions based on multinomial distribution

            # recieve reward and new state
            state,reward,done,_ = env.step(action)
            global_counter += 1

            if done or step_counter >= num_global_step:
                step_counter = 0
                state = env.reset()
            
            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break
        
        # obtain critic values
        if not done:
            _,R,_,_ = model(state,hx,cx)
        else:
            R = torch.zeros((1,1),dtype = torch.float)

        # TODO: gradient acsent
        # TODO: perform asynchronous update
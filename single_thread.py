'''
This script define the training and testing function
for each actor-learner thread
'''

import pickle
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.multiprocessing as mp
import math
import timeit
from torch.distributions import Categorical
from collections import deque
from os import path

from src.agent import Reward,SkipEnv, gym_env
from src.model import A3C
from src.optimizer import Adam_global
from src.params import *
from src.utils import *

def save_model(model):

    return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'trained_model.pth'))


def train(idx, shared_model,optimizer,counter,lock):
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
    start = timeit.default_timer()

    env,num_state,num_action = gym_env(world,stage,version,actions)    # define environment
    env.seed(123+idx)

    model = A3C(num_state,num_action)

    model.train()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    step_counter = 0
    curr_episode = 0
    terminated = 0
    success = 0
    fail = 0
    acts = []
    record_reward = []
    record_reward_average = []
    record_acts = []

    while True:
        curr_episode += 1
        # sync with the shared model
        model.load_state_dict(shared_model.state_dict())

        # save data
        if curr_episode % 50 == 0:
            interval_timer = timeit.default_timer()
            print('Current episode:{}, terminated:{},\
                    success:{}, fail:{},elasped time:{}'.format(curr_episode,terminated,success,fail,interval_timer-start))

            if curr_episode > 50:
                with open('record_acts.txt','wb') as fp:
                    pickle.dump(record_acts,fp)
                
                with open('record_reward_average.txt','wb') as fp:
                    pickle.dump(record_reward_average,fp)

            
        if done:
            hx = torch.zeros((1,512),dtype=torch.float)
            cx = torch.zeros((1,512),dtype=torch.float)
            terminated += 1
        else:
            hx = hx.detach()
            cx = cx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        # reset gradient
        action_loss = 0
        critic_loss = 0

        # repeat until terminal or max steps reach
        for step in range(num_local_steps):
            step_counter+=1

            # perform action according to policy
            logits,value,hx,cx = model(state,hx,cx)
            prob = F.softmax(logits,dim=1)    # probability of choosing each actions
            log_prob = F.log_softmax(logits,dim=1)
            entropy = -(log_prob * prob).sum(1,keepdim=True)
            entropies.append(entropy)

            m = Categorical(prob)
            action = m.sample().item()    # choosing actions based on multinomial distribution
            acts.append(action)

            # recieve reward and new state
            state,reward,done,info = env.step(action)

            with lock:
                counter.value += 1

            if done or step_counter >= num_global_step:
                step_counter = 0
                state = env.reset()
                if info['flag_get']:
                    success = success + 1
                else:
                    fail = fail + 1

            
            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob[0,action])
            rewards.append(reward)
            record_reward.append(reward)

            if done:
                break
        
        # obtain critic values
        if not done:
            _,R,_,_ = model(state,hx,cx)
            R = R.detach()
        else:
            R = torch.zeros((1,1),dtype = torch.float)
            record_acts.append(acts)
            avg_reward = sum(record_reward)
            record_reward_average.append(avg_reward)

            record_reward = []
            acts = []

        # gradient acsent
        values.append(R)
        esitimator = torch.zeros((1,1),dtype=torch.float)
        for i in reversed(range(len(rewards))):
            R = rewards[i] + discount * R
            advantage_fc = rewards[i] + discount * values[i+1] - values[i]

            # approximate the actor gradient using Generalized Advantage Estimator
            esitimator = discount * tau * esitimator + advantage_fc
            # accumulate gradients wrt the actor
            action_loss = action_loss + log_probs[i] * esitimator.detach() + beta * entropies[i]
            # accumulate gradients wrt the critic
            critic_loss = critic_loss + (R-values[i])**2/2

        # perform asynchronous update
        optimizer.zero_grad()
        total_loss = critic_loss_coef * critic_loss - action_loss
        nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
        total_loss.backward()

        # ensure current model and shared model has shared gradients
        for curr_param, shared_param in zip(model.parameters(),shared_model.parameters()):
            if shared_param is not None:
                break
            shared_param._grad = curr_param.grad

        optimizer.step()

        if info['flag_get']:
            save_model(shared_model)

        if curr_episode == int(num_global_step/num_local_steps):
            end = timeit.default_timer()
            print('Training process {} terminated, run {} episodes, \n \
                    with {} success and {} failure,elasped time {}'.format(idx,terminated,success,fail,end-start))

            return 
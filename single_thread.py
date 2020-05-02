'''
This script define the training and testing function
for each actor-learner thread
'''


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
    curr_episode = 0

    while True:
        curr_episode += 1
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

        # reset gradient
        action_loss = 0
        critic_loss = 0

        # repeat until terminal or max steps reach
        for step in range(num_local_steps):
            step_counter+=1

            # perform action according to policy
            logits,value,hx,cx = model(state,hx,cx)
            prob = F.softmax(logits,dim=-1)    # probability of choosing each actions
            log_prob = F.log_softmax(logits,dim=1)
            entropy = -(log_prob * prob).sum(1,keepdim=True)
            entropies.append(entropy)

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
            log_probs.append(log_prob[0,action])
            rewards.append(reward)

            if done:
                break
        
        # obtain critic values
        if not done:
            _,R,_,_ = model(state,hx,cx)
            R = R.detach()
        else:
            R = torch.zeros((1,1),dtype = torch.float)

        # gradient acsent
        values.append(R)
        esitimator = torch.zeros(1,1)
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
        total_loss = critic_loss - action_loss
        total_loss.backward()

        # ensure current model and shared model has shared gradients
        for curr_param, shared_param in zip(model.parameters(),shared_model.parameters()):
            if shared_param is not None:
                break
            shared_param._grad = curr_param.grad

        optimizer.step()

        if curr_episode == int(num_global_step/num_local_steps):
            print('Training process {} terminated'.format(idx))
            return 


def test(idx,shared_model,global_counter):
    torch.manual_seed(123+idx)
    env,num_state,num_action = gym_env(world,stage,version,actions)
    model = A3C(num_state,num_action)
    model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    step_counter = 0
    total_reward = 0
    acts = deque(maxlen = max_actions)

    while True:
        step_counter += 1

        if done:
            model.load_state_dict(shared_model.state_dict())

        with torch.no_grad():
            if done:
                hx = torch.zeros((1,256),dtype=torch.float)
                cx = torch.zeros((1,256),dtype=torch.float)
            else:
                hx = hx.detach()
                cx = cx.detach()
        
        action,value,hx,cx = model(state,hx,cx)
        prob = F.softmax(action,dim=1)
        action = torch.max(prob).item()
        state,reward,done,_ = env.step(action)
        env.render()
        acts.append(action)
        total_reward += reward

        if step_counter > num_global_step or acts.count(actions[0]) == acts.maxlen:
            done = True
        
        if done:
            print('number of step {}, episode reward{}, episode length{}'.format(
                    global_counter, total_reward, step_counter
            ))
            step_counter = 0
            total_reward = 0
            acts.clear()
            state = env.reset()
        state = torch.from_numpy(state)
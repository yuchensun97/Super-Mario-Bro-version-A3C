import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical
from collections import deque
from os import path

from src.agent import Reward,SkipEnv, gym_env
from src.model import A3C
from src.optimizer import Adam_global
from src.params import *
from src.utils import *

def test_local(idx):
    torch.manual_seed(123+idx)
    env,num_state,num_action = gym_env(world,stage,version,actions)
    model = A3C(num_state,num_action)
    model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)),'trained_model.pth'),map_location='cpu'))
    model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    step_counter = 0
    total_reward = 0
    acts = deque(maxlen = max_actions)

    for _ in range(200):
        step_counter += 1

        with torch.no_grad():
            if done:
                hx = torch.zeros((1,256),dtype=torch.float)
                cx = torch.zeros((1,256),dtype=torch.float)
            else:
                hx = hx.detach()
                cx = cx.detach()
        
            action,value,hx,cx = model(state,hx,cx)
            prob = F.softmax(action,dim=1)
            m = Categorical(prob)
            action = m.sample().item()
            state,reward,done,info = env.step(int(action))
            state = torch.from_numpy(state)
            env.render()
            acts.append(action)
            total_reward += reward

            if done:
                # state = env.reset()
                # state = torch.from_numpy(state)
                break

    print(acts)
    print(len(acts))

def dummy_test(idx):
    torch.manual_seed(123+idx)
    env,num_state,num_action = gym_env(world,stage,version,actions)
    acts_8 =[0, 1, 3, 4, 6, 6, 1, 2, 2, 4, 4, 5, 5, 2, 4, 1, 6, 1, 6, 6, 0, 2, 4, 4, 5, 0, 4, 6, 2, 3, 3, 1, 6, 5, 4, 2, 1, 0, 3, 1, 2, 4, 3, 6, 3, 2, 2, 4, 1, 5, 1, 3, 0, 4, 6, 6, 1, 4, 0, 3, 0, 4, 6, 2, 0, 2, 4, 1, 5, 5, 2, 4, 1, 4, 0, 4, 6, 0, 5, 5, 5, 4, 5, 0, 3, 6, 2, 5, 3, 3, 2, 3, 3, 3, 3, 6, 5, 0, 4, 2, 5, 3, 3, 3, 2, 4, 0, 0, 2, 4, 2, 3, 1, 5, 5, 3, 4, 0, 3, 3, 4, 6, 4, 2, 4, 5, 4, 0, 6, 2, 4, 2, 3, 6, 1, 0, 4, 2, 3, 0, 2, 3, 2, 6, 5, 2, 0, 6, 4, 2, 3, 0, 5, 3, 0, 6, 2, 5, 4, 2, 2, 4, 3, 6, 1, 5, 1, 3, 1, 5, 3, 2, 4, 5, 6, 2, 3, 5, 0, 6, 5, 2, 2, 4, 5, 2, 1, 3, 0, 5, 6, 3, 1, 0, 2, 1, 5, 1, 2, 0, 0, 3, 4, 2, 4, 0, 6, 1, 5, 1, 6, 3, 5, 4, 1, 6, 5, 5, 1, 2, 1, 4, 0, 3, 2, 1, 2, 4, 5, 1, 0, 1, 4, 6, 5, 0, 5, 1, 4, 0, 4, 5, 4, 5, 2, 5, 6, 2, 3, 3, 4, 2, 0, 6, 6, 6, 4, 5, 5, 6, 1, 4, 3, 5, 3, 5, 0, 4, 3, 0, 5, 3, 3, 4, 3, 6, 5, 3, 0, 0, 2, 4, 1, 0, 2, 6, 5, 0, 3, 3, 3, 4, 4, 0, 0, 4]
    done = True

    for act in acts_8:
        if done:
            state = env.reset()
        state, reward, done, info = env.step(act)
        env.render()


if __name__ == "__main__":
    torch.manual_seed(123)
    dummy_test(0)
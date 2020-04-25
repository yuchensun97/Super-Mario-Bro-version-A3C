"""
Using Adam as optimizer to to improve training speed
"""
import torch

class Adam_global(torch.optim.Adam):
    def __init__(self,params,lr,betas,eps,weight_decay):
        super(Adam_global,self).__init__(params,lr,betas,eps,weight_decay)
        # state initialization
        for group in self.param_groups:
            for key in group['params']:
                state = self.state[key]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(key.data)
                state['exp_avg_sq'] = torch.zeros_like(key.data)
                # share memory
                state['exp_ave'].share_memory_()
                state['exp_avg_sq'].share_memory_()
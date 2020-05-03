"""
This script define the parameters for A3C algorithm
"""
discount = 0.99

# define optimizer parameters
lr = 0.001    # learning rate
betas = (0.9,0.999)
eps = 1e-8
weight_decay = 0

# gradient ascent
beta = 0.01    # control the strength of the entropy regularization term
tau = 1.0    # parameters for the 

# define environment
world = 1
stage = 1
version = 0
actions = 'SIMPLE_MOVEMENT'

# training process
num_local_steps = 50
num_global_step = 5e6
num_processes = 6
max_actions = 200    # maximum number of actions in an episode
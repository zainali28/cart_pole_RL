import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# setting up environment
env = gym.make("CartPole-v1")

# setting up matplotlib
# is_python = 'inline' in matplotlib.get_backend()
# if is_python:
#     from IPython import display

# automatically update the graph without explicit commands
plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available()
else "mps" if torch.backends.mps.is_available() else
"cpu")

# single transition in environment
# next_state and reward based on current state and action
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Replay Memory to store recent transitions
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
# implementing DQN algo by using a feed-forward nn
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # called with either a single element 
    # or a batch of elements in case of optimization
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# number of transitions sampled from replay buffer
# discount factor gamma
# exploration value eps_start that decay by eps_decay to eps_end
# follows a epsilon greedy policy
# tau is update rate of target network
# lr is learning rate of optimizer 
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005

# getting no. of actions from gym action space
n_actions = env.action_space.n
# getting number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
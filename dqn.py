import torch.nn as nn
import torch
from magent2.environments import battle_v4

class DQN(nn.Module):

    def __init__(self, observation_shape, action_shape):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(observation_shape, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_shape)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
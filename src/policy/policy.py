import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from policy import Policy


class FCPolicy(Policy):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> Categorical:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return Categorical(logits=logits)

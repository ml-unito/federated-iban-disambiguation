import torch
import torch.nn as nn
from torch.functional import F


class LR(nn.Module):
    def __init__(self, input_dim=8, output_dim=2):
        super(LR, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_dim)
        return F.softmax(self.fc(x), dim=1)
import torch
import torch.nn as nn

class WakeupModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(31, 64),
            nn.Dropout(0.5),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(128, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.seq(x)
        x = self.linear1(x)
        return x
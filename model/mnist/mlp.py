import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
torch.manual_seed(0)
np.random.seed(0)


class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(28*28, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        out = self.classifier(x)
        return F.softmax(out,dim = 1)

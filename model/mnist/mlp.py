import torch
from torch import nn
import torch.nn.functional as F


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
        return F.softmax(out, dim=1)

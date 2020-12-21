import torch
import torchvision

import torch.nn as nn
import torch.optim as optim

class BasicNet(nn.Module):
    """Simple CNN with a few convolutional 
    and pooling layers, used as an MVP in development. 
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size = 3, padding = 1)
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 8, kernel_size = 3, padding= 1)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8*56*56, 32)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.pool2(out)
        out = out.view(-1, 8*56*56)
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out
from torch import nn as nn
from torch.nn import functional as F

import torch

class Net(nn.Module):
    def __init__(self,inputsize):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, inputsize, kernel_size=5)
        self.conv2 = nn.Conv2d(inputsize, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        #print(x)
        #print("outputs")
        #for i in range(10):
        #   print(torch.sigmoid(x)[i])

        #print("queried")
        return torch.sigmoid(x)

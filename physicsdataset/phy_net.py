from torch import nn as nn
from torch.nn import functional as F
from physicsdataset.phy_variables import variables
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.norm0 = nn.BatchNorm1d(30)

        self.fc1 = nn.Linear(30,128)
        self.norm1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128,256)
        self.norm2 = nn.BatchNorm1d(256)


        self.fc3 = nn.Linear(256,1)

        #self.b1d = nn.BatchNorm1d()



      
#Do I want to track the mean and variance??
    def forward(self, x): #add batch normilization and dropout layers, more layers,more neurons, output plots

        x = self.norm0(x)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.norm1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.norm2(x)

        x = self.fc3(x)
        x = torch.sigmoid(x)

        return x

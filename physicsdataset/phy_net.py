from torch import nn as nn
from torch.nn import functional as F
from physicsdataset.phy_variables import variables
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(30,128)
        self.fc2 = nn.Linear(128,1)
        #self.fc3 = nn.Linear(256,1)



      

    def forward(self, x):
        print("into net {}".format(x))

        x = self.fc1(x)
        print("middle {}".format(x))
        x = F.relu(x)
        x = self.fc2(x)

       # x = F.relu(x)
        #x = self.fc3(x)
        print("into sigmoid {}".format(x))
        x = torch.sigmoid(x)
        print("from net {}".format(x))
        return x

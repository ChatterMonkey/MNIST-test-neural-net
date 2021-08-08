from torch import nn as nn
from torch.nn import functional as F
from physicsdataset.phy_variables import variables
import torch

class Net(nn.Module):
    def __init__(self):
        print('initilizing')
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=1)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(600, 50)
        self.fc2 = nn.Linear(50, 1)


    def forward(self, x):
        print("imput")
        print(x.size())

        x = self.conv1(x)
        print("after conv 1")
        print(x.size())
        x = F.relu(x)
        print("after relu")
        print(x.size())
        x = self.conv2(x)
        print("after conv 2")
        print(x.size())
        x= self.conv2_drop(x)
        print("after drop")
        print(x.size())

        x = F.relu(x)
        print("after relu")
        print(x.size())


        x = x.view(variables.train_batch_size,-1)
        print("after view as")
        print(x.size())

        x = self.fc1(x)

        print("after fully connected layer")
        print(x.size())


        x = F.relu(x)

        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        print("after second fully connected layer")
        print(x.size())
        #print(x)
        #print("outputs")
        #for i in range(10):
        #   print(torch.sigmoid(x)[i])

        #print("queried")
        return torch.sigmoid(x)

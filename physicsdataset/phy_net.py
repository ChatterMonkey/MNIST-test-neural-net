from torch import nn as nn
from torch.nn import functional as F
from physicsdataset.phy_variables import variables
import torch
import time

class Net_256_512_512_256(nn.Module):
    def __init__(self):
        super(Net_256_512_512_256, self).__init__()



        self.norm0 = nn.BatchNorm1d(30)


        self.initialdrop = nn.Dropout(0.8)
        self.secondarydrop = nn.Dropout(0.5)

        self.fc1 = nn.Linear(30,256)
        self.norm1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256,512)
        self.norm2 = nn.BatchNorm1d(512)


        self.fc3 = nn.Linear(512,512)
        self.norm3 = nn.BatchNorm1d(512)


        self.fc4 = nn.Linear(512,256)
        self.norm4 = nn.BatchNorm1d(256)



        self.fc5 = nn.Linear(256,1)

        #self.b1d = nn.BatchNorm1d()



      
#Do I want to track the mean and variance??
    def forward(self, x): #add batch normilization and dropout layers, more layers,more neurons, output plots

        # x = self.norm0(x)
        print("X is {}".format(x))
        #x = self.initialdrop(x)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.norm1(x)
        x = self.secondarydrop(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.norm2(x)
        x = self.secondarydrop(x)



        x = self.fc3(x)
        x = F.relu(x)
        x = self.norm3(x)
        x = self.secondarydrop(x)


        x = self.fc4(x)
        x = F.relu(x)
        x = self.norm4(x)
        x = self.secondarydrop(x)


        x = self.fc5(x)
        x = torch.sigmoid(x)

        return x

class Net_256_512(nn.Module):
    def __init__(self):
        super(Net_256_512, self).__init__()



        self.norm0 = nn.BatchNorm1d(30)


        self.initialdrop = nn.Dropout(0.8)
        self.secondarydrop = nn.Dropout(0.5)

        self.fc1 = nn.Linear(30,256)
        print("weights ")
        print(self.fc1.weight)
        self.norm1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256,512)
        self.norm2 = nn.BatchNorm1d(512)


        self.fc3 = nn.Linear(512,1)



        #self.b1d = nn.BatchNorm1d()




#Do I want to track the mean and variance??
    def forward(self, x): #add batch normilization and dropout layers, more layers,more neurons, output plots
        #print("trying to forward")
        #print("weights ")
        #print(self.fc1.weight)
        # x = self.norm0(x)

        #x = self.initialdrop(x)
        #print("x is {}".format(x))
       # print(x.shape)
       # print(self.fc1)

        x = self.fc1(x)
      #  print("x is {}".format(x))
        x = F.relu(x)
       # print("x is {}".format(x))
        x = self.norm1(x)
      #  print("x is {}".format(x))
        x = self.secondarydrop(x)
      #  print("x is {}".format(x))
        x = self.fc2(x)
      #  print("x is {}".format(x))
        x = F.relu(x)
      #  print("x is {}".format(x))
        x = self.norm2(x)
     #   print("x is {}".format(x))
        x = self.secondarydrop(x)
       # print("x is {}".format(x))



        x = self.fc3(x)
      #  print("x is {}".format(x))


        x = torch.sigmoid(x)
    #    print("x is {}".format(x))
        #time.sleep(100)
        return x


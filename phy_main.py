from physicsdataset.phy_loaders import opendata
from net import Net
import torch

train_data, train_target = opendata(10)
print(train_target)
print(len(train_data[0]))
network = Net(30)


for j in range(len(train_data)):
    for i in range(len(train_data[j])):



        train_data[j][i] = float(train_data[j][i])

train_data = torch.FloatTensor(train_data)
print(train_data[0])
output = network(train_data[0])
print(output)

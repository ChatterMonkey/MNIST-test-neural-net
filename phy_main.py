from physicsdataset.phy_loaders import opendata
from physics_net import Net
import torch

train_data, train_target = opendata(10)
print(train_target)
print(len(train_data[0]))
network = Net()


for j in range(len(train_data)):
    for i in range(len(train_data[j])):



        train_data[j][i] = float(train_data[j][i])

train_data = torch.FloatTensor(train_data)
print(train_data[0])
train_data = train_data[0].reshape(5,6)
print(train_data)
train_data = train_data.unsqueeze(0).unsqueeze(0)
print(train_data.size())

output = network(train_data)
print(output)

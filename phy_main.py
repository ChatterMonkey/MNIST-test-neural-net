from physicsdataset.phy_loaders import opendata
from physics_net import Net
import torch
from phy_variables import variables




variables.set_train_batch_size(10)


train_data, train_target = opendata(variables.train_batch_size + 1)

torch.manual_seed(0)

print(train_target)


print(len(train_data[0]))

network = Net()


for j in range(len(train_data)):
    for i in range(len(train_data[j])):
        train_data[j][i] = float(train_data[j][i])


train_data = torch.FloatTensor(train_data)

print("first training data")

print(train_data)

print(train_data.size())







train_data = train_data.reshape(variables.train_batch_size,1,5,6)

print(train_data.size())

output = network(train_data)
print(output)

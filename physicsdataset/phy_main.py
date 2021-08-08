from physicsdataset.phy_loaders import open_test_data, open_training_data
from physicsdataset.physics_net import Net
import torch
import torch.optim as optm
from physicsdataset.phy_variables import variables
import torch.nn.functional as f

train_batch_size = 100000
torch.manual_seed(0)
network = Net()
optimizer = optm.SGD(network.parameters(),0.1,0.5)



variables.set_train_batch_size(train_batch_size)
train_data, train_target = open_training_data(variables.train_batch_size )
train_target_tensor = torch.zeros([variables.train_batch_size,1])
for i in range(variables.train_batch_size):
    train_target_tensor[i][0] = train_target[i]

for j in range(len(train_data)):
    for i in range(len(train_data[j])):
        train_data[j][i] = float(train_data[j][i])
train_data = torch.FloatTensor(train_data)
train_data = train_data.reshape(variables.train_batch_size,1,5,6)

for epoch in range(10):
    optimizer.zero_grad()
    output = network(train_data)
    #print(output.size())
    #print(train_target_tensor.size())
    loss = f.binary_cross_entropy(output,train_target_tensor)
    print("LOSS = {}".format(loss))
    loss.backward()
    optimizer.step()



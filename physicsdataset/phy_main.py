from physicsdataset.phy_loaders import open_test_data, open_training_data
from physicsdataset.physics_net import Net
import torch
import torch.optim as optm
from physicsdataset.phy_variables import variables
import torch.nn.functional as f

#there are 250000 events total

train_batch_size = 20000
test_batch_size = 50000
#test_batch_size = 3
num_train_batches = 10
num_test_batches = 1
variables.set_train_batch_size(train_batch_size)
variables.set_test_batch_size(test_batch_size)
torch.manual_seed(0)
network = Net()
optimizer = optm.SGD(network.parameters(),0.1,0.9)

print("processing training data...")
train_data, train_target = open_training_data(num_train_batches)
print("done processing training data")

for epoch in range(3):
    for batch in range(num_train_batches):
        optimizer.zero_grad()

        train_batch_data = train_data[batch]
        train_batch_target = train_target[batch]

        target_t = torch.zeros([variables.train_batch_size, 1])
        data_t = torch.zeros([variables.train_batch_size,30])

        for event in range(variables.train_batch_size):
            target_t[event][0] = train_batch_target[event]

        for event in range(variables.train_batch_size):
            for variable in range(30):
                data_t[event][variable] = float(train_batch_data[event][variable])



        output = network(data_t)

        loss = f.binary_cross_entropy(output,target_t)
        print("LOSS = {}".format(loss))
        loss.backward()
        optimizer.step()




print("processing testing data...")
test_data, test_target = open_test_data(num_test_batches)
print("done processing testing data")

test_data = test_data[0]
test_target = test_target[0]

target_t = torch.zeros([variables.test_batch_size, 1])
data_t = torch.zeros([variables.test_batch_size,30])

for event in range(variables.test_batch_size):
    target_t[event][0] = test_target[event]

for event in range(variables.test_batch_size):
    for variable in range(30):
        data_t[event][variable] = float(test_data[event][variable])



#print(target_t)
#print(data_t)
output = network(data_t,test_batch_size)

#print(output)
#print(target_t.size())

num_correct = 0
for guess in range(variables.test_batch_size):
    if output[guess][0] > 0.5:
        if target_t[guess][0]  == 1:
            num_correct += 1
    else:
        if target_t[guess][0] ==0:
            num_correct += 1

print("{} correct".format(num_correct))
print("{}% accuracy".format(num_correct/variables.test_batch_size*100))










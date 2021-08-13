from physicsdataset.phy_loaders import open_test_data, open_training_data
from physicsdataset.physics_net import Net
import torch
import torch.optim as optm
from physicsdataset.phy_variables import variables
import torch.nn.functional as f

#there are 250000 events total
num_variables = 30
train_batch_size = 1
test_batch_size = 2
#test_batch_size = 3
num_train_batches = 2
num_test_batches = 1
variables.set_train_batch_size(train_batch_size)
variables.set_test_batch_size(test_batch_size)
torch.manual_seed(1)
network = Net()
optimizer = optm.Adam(network.parameters(),0.001)

print("processing training data...")
train_data, train_target = open_training_data(num_train_batches)
#train_data = [[[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]],[[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]],[[0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]]]
#train_data = [[[0.1]],[[0.2]],[0.3]]
#train_target = [[0],[1],[1]]


print("done processing training data")
print(train_target)
print(train_data)
print(len(train_data))
print(len(train_data[0]))



for epoch in range(2000):
    for batch in range(num_train_batches):
        network.train()
        optimizer.zero_grad()

        train_batch_data = train_data[batch]


        train_batch_target = train_target[batch]

        target_t = torch.zeros([variables.train_batch_size, 1])
        data_t = torch.zeros([variables.train_batch_size,num_variables])

        for event in range(variables.train_batch_size):
            target_t[event][0] = train_batch_target[event]

        for event in range(variables.train_batch_size):
            for variable in range(num_variables):
                data_t[event][variable] = float(train_batch_data[event][variable])
        print("")
        print("")
        print(train_batch_data)
        print(train_batch_target)
        print(data_t)
        print(target_t)
        #print(data_t)
        output = network(data_t)
        #print(list(network.parameters())[0])
        print("output is {}".format(output))
        print("target is {}".format(target_t))



        loss = f.binary_cross_entropy(output,target_t)
        print("LOSS = {}".format(loss))


        loss.backward()

        optimizer.step()

        print("output is now {}".format(network(data_t)))


network.eval()

print("processing testing data...")
variables.set_train_batch_size(test_batch_size)
test_data, test_target = open_training_data(num_test_batches)


#test_data = [[[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2],[0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]]]
#test_target = [[0,1,1]]
#test_data = [[[0.1],[0.2],[0.3]]]

print("done processing testing data")
print(test_data)
print(test_target)
test_data = test_data[0]
test_target = test_target[0]

target_t = torch.zeros([variables.test_batch_size, 1])
data_t = torch.zeros([variables.test_batch_size,num_variables])


for event in range(variables.test_batch_size):
    print(event)

    target_t[event][0] = test_target[event]

for event in range(variables.test_batch_size):
    for variable in range(num_variables):
        data_t[event][variable] = float(test_data[event][variable])


print(target_t)
print(data_t)

output = network(data_t)
print(output)

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










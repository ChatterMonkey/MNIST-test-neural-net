from physicsdataset.phy_loaders import open_test_data, open_training_data
from physicsdataset.phy_net import Net
from physicsdataset.phy_train import train
from physicsdataset.phy_test import test
import torch
import torch.optim as optm
from tqdm import tqdm
from physicsdataset.phy_variables import variables

loss_function_tuple = (("MeanSquaredError","mse"),("SignificanceLoss","sl"),("BineryCrossEntropy","bce"))

loss_function_id = 2
num_epochs = 2

num_train_batches = 5
num_test_batches = 1
train_batch_size = 20
test_batch_size = train_batch_size * num_train_batches

variables.set_epochs(num_epochs)
variables.set_train_batch_size(train_batch_size)
variables.set_test_batch_size(test_batch_size)

torch.manual_seed(1)
network = Net()
optimizer = optm.Adam(network.parameters(),0.001)

print("processing training data...")
train_data, train_target = open_training_data(num_train_batches)
print("done processing training data")

for epoch in tqdm(range(variables.num_epochs), colour = "green"):
    for batch in range(num_train_batches):
        train_batch_data = train_data[batch]
        train_batch_target = train_target[batch]

        target_t = torch.zeros([variables.train_batch_size, 1])
        data_t = torch.zeros([variables.train_batch_size,variables.num_variables])

        for event in range(variables.train_batch_size):
            target_t[event][0] = train_batch_target[event]

        for event in range(variables.train_batch_size):
            for variable in range(variables.num_variables):
                data_t[event][variable] = abs(float(train_batch_data[event][variable])/300)

        train(network,optimizer,data_t,target_t,loss_function_id)


print("processing testing data...")
test_data, test_target = open_test_data(num_test_batches)
print("done processing testing data")


for batch in tqdm(range(num_test_batches), colour = "cyan"):

    test_data = test_data[batch]
    test_target = test_target[batch]

    target_t = torch.zeros([variables.test_batch_size, 1])
    data_t = torch.zeros([variables.test_batch_size,variables.num_variables])

    for event in range(variables.test_batch_size):
        target_t[event][0] = test_target[event]

    for event in range(variables.test_batch_size):
        for variable in range(variables.num_variables):
            data_t[event][variable] = abs(float(test_data[event][variable])/variables.normalization_constant)

    num_correct, loss = test(network,data_t,target_t,loss_function_id)





print("{} correct".format(num_correct))
print("{}% accuracy".format(num_correct/variables.test_batch_size*100))










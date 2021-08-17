from physicsdataset.phy_loaders import open_test_data, open_training_data
from physicsdataset.phy_net import Net
from physicsdataset.phy_train import train
from physicsdataset.phy_test import test
from physicsdataset.data_manager import add_data,visulize
from physicsdataset.phys_roc_maker import calculate_roc_curve_points
import torch
import torch.optim as optm
from tqdm import tqdm
from physicsdataset.phy_variables import variables


# 200000
loss_function_id = 0
num_epochs = 5
learning_rate = 0.5
num_training_batches = 100
num_testing_batches = 100
train_batch_size = 64
test_batch_size = 64

variables.set_params(train_batch_size,test_batch_size,num_training_batches,num_testing_batches,loss_function_id,learning_rate,num_epochs)

torch.manual_seed(1)
network = Net()
optimizer = optm.Adam(network.parameters(),0.001)


train_data, train_target = open_training_data(num_training_batches)

correct_list = []
training_loss = []
testing_loss = []
test_data, test_target = open_test_data(num_testing_batches)
for epoch in tqdm(range(variables.num_epochs), colour = "green",desc= "Training"):
    total_num_correct = 0
    total_loss = 0
    for batch in range(num_training_batches):
        train_batch_data = train_data[batch]
        train_batch_target = train_target[batch]

        target_t = torch.zeros([variables.train_batch_size, 1])
        data_t = torch.zeros([variables.train_batch_size,variables.num_variables])

        for event in range(variables.train_batch_size):
            target_t[event][0] = train_batch_target[event]

        for event in range(variables.train_batch_size):
            for variable in range(variables.num_variables):
                data_t[event][variable] = abs(float(train_batch_data[event][variable])/300)

        loss = train(network,optimizer,data_t,target_t,loss_function_id)
        total_loss += loss

    training_loss.append(total_loss/(num_training_batches * train_batch_size))




    total_test_loss  = 0
    for batch in range(num_testing_batches):


        test_data_batch = test_data[batch]
        test_target_batch = test_target[batch]

        target_t = torch.zeros([variables.test_batch_size, 1])
        data_t = torch.zeros([variables.test_batch_size,variables.num_variables])

        for event in range(variables.test_batch_size):
            target_t[event][0] = test_target_batch[event]

        for event in range(variables.test_batch_size):
            for variable in range(variables.num_variables):
                data_t[event][variable] = abs(float(test_data_batch[event][variable])/variables.normalization_constant)

        num_correct, loss = test(network,data_t,target_t,loss_function_id)
        total_test_loss += loss.item()
        total_num_correct += num_correct
    testing_loss.append(total_loss/(num_testing_batches*test_batch_size))
    correct_list.append(total_num_correct)


torch.save(network.state_dict(),"../phy_nets/net1.pth")


network_path= "../phy_nets/net1.pth"
cutoffs= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
tp,fp = calculate_roc_curve_points(cutoffs,network,loss_function_id,variables.num_testing_batches)

for entry in range(len(correct_list)):
    correct_list[entry] = correct_list[entry]/(variables.test_batch_size * num_testing_batches)*100
add_data(network_path,training_loss,testing_loss, correct_list,tp,fp)

visulize(plot_last=True)
print("{} correct, {}% accuracy".format(correct_list[-1], correct_list[-1] ))










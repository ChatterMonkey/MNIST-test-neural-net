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
test_data, test_target = open_test_data(num_testing_batches)

accuracy_each_epoch = []
training_loss_each_epoch = []
testing_loss_each_epoch = []


for epoch in tqdm(range(variables.num_epochs), colour = "green",desc= "Training"):

    num_correct_this_epoch = 0
    training_loss_this_epoch = 0
    testing_loss_this_epoch = 0

    for batch in range(num_training_batches):
        loss = train(network,optimizer,train_data,test_data,loss_function_id)
        training_loss_this_epoch += loss

    training_loss_each_epoch.append(training_loss_this_epoch)

    for batch in range(num_testing_batches):

        num_correct, loss = test(network,test_data,test_target,loss_function_id)
        testing_loss_this_epoch += loss.item()
        num_correct_this_epoch += num_correct

    testing_loss_each_epoch.append(testing_loss_this_epoch)
    accuracy_each_epoch.append(num_correct_this_epoch/(num_testing_batches*test_batch_size)*100)


torch.save(network.state_dict(),"../phy_nets/net1.pth")


network_path= "../phy_nets/net1.pth"
cutoffs= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
tp,fp = calculate_roc_curve_points(cutoffs,network,loss_function_id,variables.num_testing_batches)

add_data(network_path,training_loss_each_epoch,testing_loss_each_epoch, accuracy_each_epoch,tp,fp)

visulize(plot_last=True)

print("{} correct, {}% accuracy".format(accuracy_each_epoch[-1], accuracy_each_epoch[-1] ))










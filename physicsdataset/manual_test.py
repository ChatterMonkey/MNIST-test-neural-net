from physicsdataset.phy_loaders import open_test_data, open_training_data
from physicsdataset.phy_net import Net
from physicsdataset.phy_train import train
from physicsdataset.phy_test import test
from physicsdataset.data_manager import add_data,visulize
from physicsdataset.phys_roc_maker import calculate_roc_curve_points
from os.path import exists
import torch
import torch.optim as optm
from tqdm import tqdm
from physicsdataset.phy_variables import variables

# 250000
loss_function_id = 1
num_epochs = 1600
learning_rate = 0.001
num_training_batches = 3125
#num_training_batches = 1
num_testing_batches = 781
train_batch_size = 64
test_batch_size = 64


variables.set_params(train_batch_size,test_batch_size,num_training_batches,num_testing_batches,loss_function_id,learning_rate,num_epochs)




test_data_path = "../loaded_data/test_data_nb_" + str(num_testing_batches) + "_bs_" + str(variables.test_batch_size) + ".pt"
test_target_path = "../loaded_data/test_target_nb_" + str(num_testing_batches) + "_bs_" + str(variables.test_batch_size) + ".pt"

test_data = torch.load(test_data_path)
test_target = torch.load(test_target_path)

network = Net()
network.load_state_dict(torch.load("../phy_nets/test_16quadruple.pth"))

correct = 0
true_p = 0
false_p = 0

for batch in range(num_testing_batches):

    batch_test_data = test_data[batch]

    batch_test_target = test_target[batch]



    num_correct,loss, tp,fp = test(network,batch_test_data,batch_test_target,2,True,0.5)
    correct += num_correct
    true_p += tp
    false_p += fp
print(correct)
print(num_testing_batches*test_batch_size)
print(correct/(num_testing_batches*test_batch_size))
print(true_p)
print(false_p)

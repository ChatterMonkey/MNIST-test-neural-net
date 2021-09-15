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


variables.set_params(4000,4000,50,12,0,0,1600)



test_data_path = "../non_normalized_loaded_data/test_data_nb_" + str(12) + "_bs_" + str(4000) + ".pt"
test_target_path = "../non_normalized_loaded_data/test_target_nb_" + str(12) + "_bs_" + str(4000) + ".pt"

test_data = torch.load(test_data_path)
test_target = torch.load(test_target_path)

network = Net()
network.load_state_dict(torch.load("../new_phy_nets/bce_0.001_256_512_256.pth"))

correct = 0
true_p = 0
false_p = 0

for batch in range(12):

    batch_test_data = test_data[batch]

    batch_test_target = test_target[batch]
    network.eval()

    num_correct,loss, tp,fp = test(network,batch_test_data,batch_test_target,2,True,0.5)
    correct += num_correct
    print(correct)
    true_p += tp
    false_p += fp
print(correct)
print(12*4000)
print(correct/(12*4000))
print(true_p)
print(false_p)

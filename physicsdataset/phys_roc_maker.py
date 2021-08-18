import torch
from physicsdataset.phy_net import Net
from tqdm import tqdm
from physicsdataset.phy_loaders import open_test_data
from physicsdataset.phy_variables import variables
from physicsdataset.phy_test import test

def load_network(path):
    network_state = torch.load(path)
    network = Net()
    network.load_state_dict(network_state)
    return network


def calculate_roc_curve_points(cutoffs, network,loss_function_id, data, target):

    false_positives = []
    true_positives = []
    data_size = data.size()
    num_batches = data_size[0]
    batch_size = data_size[1]


    for cutoff in tqdm(enumerate(cutoffs), colour ="blue", desc="Generating ROC curve"):

        for batch in range(num_batches):


            data_batch = data[batch]
            target_batch = target[batch]

            num_correct,loss, tp,fp = test(network,data_batch,target_batch,loss_function_id,True,cutoff[1])

            true_positives.append(tp/batch_size)
            false_positives.append(fp/batch_size)
    return true_positives,false_positives





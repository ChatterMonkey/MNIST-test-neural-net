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

    data_sizes = list(data.size())
    num_batches = data_sizes[0]
    batch_size = data_sizes[1]


    for cutoff in tqdm(enumerate(cutoffs), colour ="blue", desc="Generating ROC curve"):
        tp_total = 0
        fp_total = 0

        for batch in range(num_batches):


            data_batch = data[batch]
            target_batch = target[batch]

            num_correct,loss, tp,fp = test(network,data_batch,target_batch,loss_function_id,True,cutoff[1])
            tp_total += tp
            fp_total += fp

        true_positives.append(tp_total/(batch_size * num_batches))
        false_positives.append(fp_total/(batch_size* num_batches))
    return true_positives,false_positives





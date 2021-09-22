import torch
from physicsdataset.phy_net import Net_256_512_512_256
from tqdm import tqdm
from physicsdataset.phy_loaders import open_test_data
from physicsdataset.phy_variables import variables
from physicsdataset.phy_test import test

def load_network(path):
    network_state = torch.load(path)
    network = Net_256_512_512_256()
    network.load_state_dict(network_state)
    return network


def calculate_roc_curve_points(cutoffs, network,loss_function_id, data, target):

    false_positives = []
    true_positives = []





    data_sizes = list(data.size())
    num_batches = data_sizes[0]
    #print(num_batches)
    batch_size = data_sizes[1]
    #print(batch_size)


    frac_signal  = torch.sum(target).item()/(num_batches*batch_size)
    frac_background = 1 - torch.sum(target).item()/(num_batches*batch_size)
    print(frac_signal)
    print(frac_background)



    for cutoff in enumerate(cutoffs):
        #print("Cut off is {}".format(cutoff))
        #print(cutoff[1])

        tp_total = 0
        fp_total = 0

        for batch in range(num_batches):


            data_batch = data[batch]
            target_batch = target[batch]

            num_correct,loss, tp,fp = test(network,data_batch,target_batch,loss_function_id,True,cutoff[1])
            tp_total += tp
            fp_total += fp
        #print(tp_total)
        #print(fp_total)

        true_positives.append(tp_total/(batch_size * num_batches*frac_signal))
        false_positives.append(fp_total/(batch_size* num_batches*frac_background))
    return true_positives,false_positives





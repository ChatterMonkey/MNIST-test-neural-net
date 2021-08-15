

from mnist.mnist_variables import variables
from mnist.functions import subset_data
import torch
from mnist.test import test
from mnist.net import Net
from mnist.loaders import test_loader

def load_network(path):
    network_state = torch.load(path)
    network = Net()
    network.load_state_dict(network_state)
    return network


def calculate_roc_curve_points(cutoffs, network,loss_function_id,using_full_dataset):

    false_positives = []
    true_positives = []


    for i in range(len(cutoffs)):
        for batch, (data, target) in enumerate(test_loader):
            if not using_full_dataset:
                data,target = subset_data(data,target,7)
            test_losses,total_number_correct,true_positive_count,false_positive_count = test(network, data,target,loss_function_id,cutoffs[i],False)
            #print("fpc = {}, tpc = {}".format(false_positive_count,true_positive_count))

            false_positives.append(false_positive_count)
            true_positives.append(true_positive_count)

            if batch > 0:
                #print("set batch size to full testing set")
                break

    #print(false_positives)
    #print(true_positives)

    false_positive_rates = []
    true_positive_rates = []
    background_count = variables.mnist_test_size*9/10
    if not using_full_dataset:
        background_count = background_count/10
    for i in range(len(false_positives)):
        false_positive_rates.append(false_positives[i]/background_count)

    for i in range(len(true_positives)):
        true_positive_rates.append(true_positives[i]/1000)

    return true_positive_rates,false_positive_rates



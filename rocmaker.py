

from variables import variables
from functions import subset_data
import torch
from test import test
from net import Net
import matplotlib.pyplot as plt
from loaders import test_loader

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



def plot_roc(true_positive_rates,false_positive_rate,filepath):

    plt.plot(false_positive_rate,true_positive_rates)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    for i in range(0,len(cutoffs)):
        plt.annotate(xy=[false_positive_rate[i],true_positive_rates[i]], s=cutoffs[i])
    plt.savefig(filepath)
    return



cutoffs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

network = load_network("/Users/mayabasu/PycharmProjects/MNIST-test-neural-net2/neuralnets/sl_80_full_001.pth")

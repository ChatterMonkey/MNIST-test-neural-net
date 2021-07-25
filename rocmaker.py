


from functions import subset_data
import torch
from test import test
from net import Net
from variables import *
import matplotlib.pyplot as plt
from loaders import test_loader

def load_network(path):
    network_state = torch.load(path)
    network = Net()
    network.load_state_dict(network_state)
    return network





def make_roc_curve(cutoffs, network,filepath,loss_function_id,subset):

    false_positives = []
    true_positives = []


    for i in range(len(cutoffs)):

        for batch, (data, target) in enumerate(test_loader):
            if subset:
                data,target = subset_data(data,target,7)
            test_losses,total_number_correct,true_positive_count,false_positive_count = test(network, data,target,loss_function_id,cutoffs[i],False)
            print("fpc = {}, tpc = {}".format(false_positive_count,true_positive_count))

            false_positives.append(false_positive_count)
            true_positives.append(true_positive_count)

            if batch > 0:
                print("set batch size to full testing set")
                break

    print(false_positives)
    print(true_positives)

    false_positives_scaled = []
    true_positives_scaled = []
    for i in range(len(false_positives)):
        false_positives_scaled.append(false_positives[i]/9000)

    for i in range(len(true_positives)):
        true_positives_scaled.append(true_positives[i]/1000)

    plt.plot(false_positives_scaled,true_positives_scaled)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    for i in range(0,len(cutoffs)):
        plt.annotate(xy=[false_positives_scaled[i],true_positives_scaled[i]], s=cutoffs[i])
    plt.savefig(filepath)
    return



cutoffs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

network = load_network("/Users/mayabasu/PycharmProjects/MNIST-test-neural-net2/neuralnets/sl_80_full_001.pth")

make_roc_curve(cutoffs, network, "roc_curves/sl_80_full_001_roc.png", 1,False) #check the path, number, and True or false


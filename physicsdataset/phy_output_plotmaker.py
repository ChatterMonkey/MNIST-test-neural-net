import matplotlib.pyplot as plt
from physicsdataset.phy_net import Net_256_512_512_256
from physicsdataset.phy_test import test
from os.path import exists
import torch

def sort_outputs(network):

    test_data_path = "../non_normalized_loaded_data/test_data_nb_" + str(12) + "_bs_" + str(4000) + ".pt"
    test_target_path = "../non_normalized_loaded_data/test_target_nb_" + str(12) + "_bs_" + str(4000) + ".pt"


    test_data = torch.load(test_data_path)
    test_target = torch.load(test_target_path)


    background_outputs = [] #values of the output neuron when fed background
    signal_outputs = [] #values of the output neuron when fed signal

    for batch in range(12):
        batch_test_data = test_data[batch]
        batch_test_target = test_target[batch]


        network.eval()
        output = network(batch_test_data)

        print(output)




        for i in range(len(batch_test_target)):
            if batch_test_target[i] == 1:

                signal_outputs.append(output[i][0].item())
            else:

                background_outputs.append(output[i][0].item())
    return(signal_outputs,background_outputs)


def plot_output(path):
    if exists("../phy_output_plots/" + path + ".png"):
        print("PATH ALREADY EXISTS")
        return
    else:
        open("../phy_output_plots/" + path + ".png","w")

    net_params = torch.load("../new_phy_nets/" + path)
    net = Net_256_512_512_256()
    net.load_state_dict(net_params)

    signal_outputs, background_outputs = sort_outputs(net)
    plt.yscale("log")
    counts, bins, patches =plt.hist(background_outputs,color="orange",alpha = 0.5,bins = 100)
    countss, binss, patchess = plt.hist(signal_outputs, color="green",alpha = 0.5,bins=100)


    rects = patches
    labels = [int(counts[i]) for i in range(len(rects))]

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height+0.01, label,
                ha='center', va='bottom',fontsize=5,color="red",alpha= 1)
    rects = patchess
    labels = [int(countss[i]) for i in range(len(rects))]

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height+0.01, label,
                ha='center', va='bottom',fontsize=5,color="blue",alpha= 1)



    plt.legend([ "Background","Signal"], loc ="upper right")

    plt.ylabel("log of the number of times the network outputted each value")
    plt.xlabel("output neuron value")
    plt.title(path)
    plt.savefig("../phy_output_plots/" + path + ".png")



plot_output("ae_0.001_256_512_0.5.pth")









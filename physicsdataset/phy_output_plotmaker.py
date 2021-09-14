import matplotlib.pyplot as plt
from physicsdataset.phy_net import Net
import torch

def sort_outputs(network):
    background_outputs = [] #values of the output neuron when fed background
    signal_outputs = [] #values of the output neuron when fed signal



    for batch, (data,target) in enumerate(test_loader):

        if not using_full_dataset:
            data, target = subset_data(data,target,variables.background)
        target = prepare_target(target)

        network.eval()

        output = network(data)




        for i in range(len(target)):
            if target[i] == 1:

                signal_outputs.append(output[i][0].item())
            else:

                background_outputs.append(output[i][0].item())
    return(signal_outputs,background_outputs)


def plot_output(path):

    net_params = torch.load(path)
    net = Net()
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
    plt.title("MSE outputs, 150 0.1 Full")
    plt.savefig("phy_output_plots/mse_150_full_0_1plot.png")











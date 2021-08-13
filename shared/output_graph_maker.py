from mnist.functions import subset_data, prepare_target
from mnist.loaders import test_loader
from mnist.mnist_variables import variables
import matplotlib.pyplot as plt
from mnist.net import Net
import torch

def sort_outputs(network,using_full_dataset):
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



net_params = torch.load('neuralnets/mse_150_full_0_1.pth')
net = Net()
net.load_state_dict(net_params)

signal_outputs, background_outputs = sort_outputs(net,True)
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
plt.savefig("output_plots/mse_150_full_0_1plot.png")




#plt.yscale("log")

#counts, bins, patches =
#print(histogram)
#print(counts)
#print(bins)


#new_histogram = list(histogram)
#print(new_histogram)
#for i in range(len(new_histogram[0])):
#    new_histogram[0][i]= new_histogram[0][i]*10000

#print(new_histogram[0])
#print(len(new_histogram[0]))
#print(len(new_histogram[1]))
#print(new_histogram[2])


#plt.hist(new_histogram[0])
#print(plt.hist(background_outputs,color="orange",alpha = 0.5,bins = 100))
#plt.hist(background_outputs,color="orange",alpha = 0.5,bins = 100)




#print()






